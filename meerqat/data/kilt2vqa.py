# coding: utf-8
"""Usage:
kilt2vqa.py ner <subset>
kilt2vqa.py ned <subset>
kilt2vqa.py generate mentions <subset> [--threshold=<threshold>]
kilt2vqa.py generate vq <subset> [<categories_to_exclude>...]
kilt2vqa.py count_entities <subset> [--threshold=<threshold>]
kilt2vqa.py labelstudio <subset>
"""
import json
import numpy as np
import random
import re
import spacy
from spacy.gold import align
from spacy.symbols import DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
from spacy.symbols import dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root

from docopt import docopt
from tqdm import tqdm
from tabulate import tabulate
import warnings

from datasets import load_dataset, load_from_disk
from meerqat.data.loading import map_kilt_triviaqa, DATA_ROOT_PATH
from meerqat.data.wiki import HUMAN, RESERVED_IMAGES, special_path_to_file_name, SPECIAL_PATH_URI_PREFIX
from meerqat.data.utils import md5
from meerqat.visualization.utils import viz_spacy, simple_stats

# spacy constants for NER
INVALID_ENTITIES = {DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL}
# TODO check root and obj: in practice, it never happened on TriviaQA dev set
VALID_DEP = {dobj, nsubj, pobj, obj, nsubjpass, poss, obl, root}

# spacy constants for pronoun-mention generation
HE_SHE_DEP = {spacy.symbols.NAMES[dep] for dep in [nsubj, nsubjpass]}
HIM_HER_DEP = {spacy.symbols.NAMES[dep] for dep in [dobj, obj, obl]}
HIS_HERS_DEP = {spacy.symbols.NAMES[poss]}

# Wikidata constants for pronoun-mention generation
#            'male'      'trans. male'
HE_GENDER = {'Q6581097', 'Q2449503'}
#             'female'    'trans. female'
SHE_GENDER = {'Q6581072', 'Q1052281'}
#            'intersex'  'non-binary'
NA_GENDER = {'Q1097630', 'Q48270'}
#             'male'    'female'
ANIMAL_SEX = {'Q44148', 'Q43445'}

# set random seed to get consistent random examples
np.random.seed(0)


def wer(a, b):
    """Compute Word Error Rate (word-level Levenshtein distance) using spacy"""
    length = max(len(a), len(b))
    return align(a, b)[0] / length


def item2placeholder(item, model=None):
    """Make input question suitable for VQA
    by replacing an explicit entity mention and its syntactic children by a placeholder.
    e.g. 'Who wrote the opera Carmen?' -> 'Who wrote {mention}'
    Note that, not only the entity mention ('Carmen') but its syntactic children ('the opera')
    are replaced by the placeholder.

    The final goal is to find an image that represents the entity
    and fill the placeholder with an appropriate (ambiguous) mention (e.g. 'this opera', 'it')

    Parameters
    ----------
    item: dict
        original question should be in 'input' key
    model: spacy.lang.en.English
        Full spacy pipeline, we use both NER and dependency parsing

    Returns
    -------
    item: dict
        same as input with extra keys:
        - "placeholder": List[dict]
          One dict like {"input": str, "entity": dict, "dependency": str}
        - "spacy_input": dict
          Original input, POS and NER-tagged with spacy in dict format
          (using Doc.to_json())

    Usage
    -----
    hugging_face_dataset.map(item2placeholder, fn_kwargs={"model": spacy_model})
    """
    item['placeholder'] = []
    question = model(item['input'])
    item['spacy_input'] = question.to_json()
    # filter questions without entities
    if not question.ents:
        return item
    potential_questions = {}
    for e in question.ents:
        # filter invalid entities
        if e.label in INVALID_ENTITIES:
            continue
        for token in e:
            # filter invalid dependencies
            if token.dep not in VALID_DEP:
                continue
            # get leftmost and rightmost syntactic children
            # min/max hack is in case the "valid dependency token" is not the head in the entity span
            # e.g. "Who wrote the poem ‘The Lady of the Lake’?", "Lake" is pobj but a leaf
            start = min(token.left_edge.i, e.start)
            end = max(token.right_edge.i, e.end-1)
            potential_questions[(start, end)] = (e, token)
    # keep only the biggest span for overlapping mentions
    for (start, end), (e, token) in potential_questions.items():
        included = False
        for other_start, other_end in potential_questions:
            # included from the left
            if start >= other_start and end < other_end:
                included = True
            # included from the right
            elif start > other_start and end <= other_end:
                included = True
        if not included:
            # replace entity and its syntactic children by a placeholder
            placeholder = question[:start].text_with_ws + "{mention}" + token.right_edge.whitespace_ + question[end + 1:].text
            item['placeholder'].append({'input': placeholder,
                                        'entity': e.as_doc().to_json(),
                                        'dependency': token.dep_})
    return item


def stats(kilt_subset):
    stat_dict = {
        "placeholders": 0,
        "originals": len(kilt_subset),
        "distinct source": 0
    }
    for item in kilt_subset:
        len_placeholder = len(item["placeholder"])
        stat_dict["placeholders"] += len_placeholder
        stat_dict["distinct source"] += min(1, len_placeholder)

    return tabulate([stat_dict], headers="keys")


def stringify(kilt_subset, field="placeholder", include_answer=True, include_provenance=True, include_dep=False):
    results = []
    invalid = []
    for item in kilt_subset:
        if item[field]:
            result = [f"Q: {item['input']}"]
            for vq in item[field]:
                result.append(f"-> {vq['input']} {vq['dependency'] if include_dep else ''}")
            if include_answer:
                result.append(f"A: {item['output']['answer'][0]}")
            if include_provenance:
                result.append(f"\t{item['output']['provenance'][0]['title'][0]}")
            results.append("\n".join(result))
        else:
            invalid.append(viz_spacy(item['spacy_input']))
    return "\n\n\n".join(results), "\n".join(invalid)


def ner(subset):
    """
    1st step: Named Entity Recognition (NER):
    Goes through the kilt subset and apply 'item2placeholder' function (see its docstring)
    Save the resulting dataset to f"{DATA_ROOT_PATH}/meerqat_{subset}"
    """

    # load model and data
    model = spacy.load("en_core_web_lg")
    kilt_tasks = map_kilt_triviaqa()
    kilt_subset = kilt_tasks[subset]

    # go through the dataset and make input question suitable for VQA
    fn_kwargs = {"model": model}
    kilt_subset = kilt_subset.map(item2placeholder, fn_kwargs=fn_kwargs)
    print(stats(kilt_subset))

    # save data
    output_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    kilt_subset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")

    # show N random examples
    N = 100
    indices = np.arange(kilt_subset.shape[0])
    np.random.shuffle(indices)
    randoms = [kilt_subset[i.item()] for i in indices[:N]]
    results, invalid = stringify(randoms)
    print(f"\nGenerated questions out of {N} random examples:\n")
    print(results)
    print(f"\nPruned questions out of {N} random examples:\n")
    print(invalid)


def disambiguate(item, wikipedia, wikipedia_ids, pedia_index):
    """Go through candidate pages from TriviaQA and compute WER between entity mention and Wikipedia title/aliases
    One should filter entities with a minimal WER of 0.5 (see 'wer' key)
    """
    for vq in item["placeholder"]:
        ent = vq["entity"]['text'].lower().strip().split()
        wers = {}
        # process each wikipedia article only once (answer might come from different paragraphs but it's irrelevant for this)
        provenances = {provenance['wikipedia_id'][0]: re.sub("\(.+\)", "", provenance['title'][0].lower()).strip() for
                       provenance in item['output']['provenance']}
        for wid, title in provenances.items():
            aliases = {title}
            # get aliases from wikipedia
            pedia_index.setdefault(wid, np.where(wikipedia_ids == wid)[0].item())
            wiki_item = wikipedia[pedia_index[wid]]
            aliases.update({alias.lower().strip() for alias in wiki_item['wikidata_info']['aliases']['alias']})
            # compute WER and keep minimal for all aliases
            word_er = min([wer(ent, alias.split()) for alias in aliases])
            wers[wid] = word_er
        # keep minimal WER for all candidate articles
        best_provenance = min(wers, key=wers.get)
        best_wer = wers[best_provenance]
        wiki_item = wikipedia[pedia_index[best_provenance]]
        vq["entity"]['wikidata_info'] = wiki_item['wikidata_info']
        vq["entity"]['wikipedia_id'] = wiki_item['wikipedia_id']
        vq["entity"]["wer"] = best_wer
    return item


def ned(subset):
    """
    2nd step: Named Entity Disambiguation (NED) using TriviaQA provided list
    Assumes that you already ran NER and loads dataset from f"{DATA_ROOT_PATH}/meerqat_{subset}"
    and wikipedia from DATA_ROOT_PATH
    """
    # load data
    dataset = load_from_disk(DATA_ROOT_PATH / f"meerqat_{subset}")
    wikipedia = load_dataset('kilt_wikipedia', cache_dir=DATA_ROOT_PATH)['full']
    wikipedia_ids = np.array(wikipedia["wikipedia_id"])
    pedia_index = {}
    fn_kwargs = {"wikipedia": wikipedia, "wikipedia_ids": wikipedia_ids, "pedia_index": pedia_index}

    # go through dataset
    dataset = dataset.map(disambiguate, fn_kwargs=fn_kwargs)

    # save data
    output_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset.save_to_disk(output_path)
    print(f"Successfully saved output to '{output_path}'")


def count_entities(subset, wer_threshold=0.5):
    path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(path)
    entities = {}

    for item in tqdm(dataset):
        for vq in item['placeholder']:
            entity = vq['entity']
            if entity['wer'] > wer_threshold:
                continue
            wikidata_id = entity['wikidata_info']['wikidata_id']
            entities.setdefault(wikidata_id, {})
            entities[wikidata_id]["wikipedia_id"] = entity["wikipedia_id"]
            entities[wikidata_id].setdefault("n_questions", 0)
            entities[wikidata_id]["n_questions"] += 1

    output_path = path / "entities.json"
    with open(output_path, 'w') as file:
        json.dump(entities, file)
    print(f"\nSuccessfully saved output to {output_path}")

    print(simple_stats([entity["n_questions"] for entity in entities.values()]))


def generate_mention(item, entities, wer_threshold=0.5, feminine_labels={}):
    for vq in item["placeholder"]:
        entity = vq['entity']
        ambiguous_mentions = {
            "pronouns": [],
            "man_woman": [],
            "occupation": [],
            "instanceof": []
        }

        # filter ambiguous entities
        if entity['wer'] > wer_threshold:
            vq['ambiguous_mentions'] = ambiguous_mentions
            continue

        dependency = vq['dependency']
        qid = entity['wikidata_info']['wikidata_id']
        entity_data = entities[qid]

        gender = entity_data.get('gender', {}).get('value')
        gender = gender.split("/")[-1] if gender else gender
        human = HUMAN in entity_data.get('instanceof', {})
        taxon_rankLabel = entity_data.get('taxon_rankLabel', {}).get('value')
        # man_woman and pronouns
        if gender not in ANIMAL_SEX:
            # man_woman
            if gender in HE_GENDER:
                ambiguous_mentions["man_woman"].append("this man")
            elif gender in SHE_GENDER:
                ambiguous_mentions["man_woman"].append("this woman")
            elif gender in NA_GENDER or not gender:
                pass
            else:
                warnings.warn(f"No case were set for this gender: '{gender}'")

            # pronouns
            if dependency in HE_SHE_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("he")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("she")
            elif dependency in HIM_HER_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("him")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("her")
            elif dependency in HIS_HERS_DEP:
                if gender in HE_GENDER:
                    ambiguous_mentions["pronouns"].append("his")
                elif gender in SHE_GENDER:
                    ambiguous_mentions["pronouns"].append("hers")
            else:
                warnings.warn(f"No case were set for this dependency: '{dependency}'")

        # occupation
        if entity_data.get('occupation') and human:
            for occupation in entity_data['occupation'].values():
                feminine_label = feminine_labels.get(occupation['value'])
                if feminine_label and gender in SHE_GENDER:
                    occupation_label = feminine_label
                # default label is default value since most names in English don't have genders
                else:
                    occupation_label = occupation['label']['value']
                ambiguous_mentions['occupation'].append(f"this {occupation_label}")
        # taxon rank (e.g. "species") or class (aka instanceof)
        elif not human:
            # taxon rank
            if taxon_rankLabel:
                ambiguous_mentions['instanceof'].append(f"this {taxon_rankLabel}")
            # class (instanceof)
            else:
                for instanceof in entity_data.get('instanceof', {}).values():
                    feminine_label = feminine_labels.get(instanceof['value'])
                    if feminine_label and gender in SHE_GENDER:
                        instanceof_label = feminine_label
                    # default label is default value since most names in English don't have genders
                    else:
                        instanceof_label = instanceof['label']['value']
                    ambiguous_mentions['instanceof'].append(f"this {instanceof_label}")
        vq['ambiguous_mentions'] = ambiguous_mentions

    return item


def generate_mentions(subset, wer_threshold=0.5):
    """3rd step: generate ambiguous mentions given entities attributes (run `wiki.py data` first)"""
    # load data
    dataset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(dataset_path)
    with open(dataset_path / "entities.json", 'r') as file:
        entities = json.load(file)
    feminine_labels_path = dataset_path / "feminine_labels.json"
    if feminine_labels_path.exists():
        with open(feminine_labels_path, "r") as file:
            feminine_labels = json.load(file)
    else:
        feminine_labels = {}
    fn_kwargs = {
        "entities": entities,
        "wer_threshold": wer_threshold,
        "feminine_labels": feminine_labels
    }

    # go through dataset
    dataset = dataset.map(generate_mention, fn_kwargs=fn_kwargs)

    # save data
    dataset.save_to_disk(dataset_path)
    print(f"Successfully saved output to '{dataset_path}'")


def generate_vq(item, entities):
    """
    Generate a image (url), question, answer triple by choosing:
        - uniformly a mention type and a mention from this mention type
        - the image with the best score (with its title sorted last in "titles").
                Tries to use a unique image per entity.

    Parameters
    ----------
    item: Dataset item
    entities: dict (see wiki.py)

    Returns
    -------
    item: Dataset item
        with a new 'vq' key (List[dict])
    """
    item['vq'] = []
    kilt_id = item['id']
    for placeholder in item['placeholder']:
        mention_types = [mention_type for mention_type in placeholder.get('ambiguous_mentions', {}).values() if mention_type]
        if not mention_types:
            continue
        qid = placeholder['entity']['wikidata_info']['wikidata_id']
        # entity might have been filtered before-hand -> get qid instead of "[qid]"
        entity = entities.get(qid, {})
        titles = entity.get("titles")
        if not titles:
            continue
        # try to use unique images per entity -> pop titles
        if len(titles) > 1:
            # note we assume that the images are sorted in ascending order w.r.t. their score
            title = titles.pop()
        else:
            title = titles[0]
        url = SPECIAL_PATH_URI_PREFIX+title.replace(" ", "_")

        # choose mention type (e.g. pronoun or occupation) uniformly from all types (that are not empty)
        mention_type = random.choice(mention_types)
        # choose mention uniformly from all mentions in this type (e.g. Barack Obama is a politician and a statesperson)
        mention = random.choice(mention_type)

        inp = placeholder['input'].format(mention=mention)
        meerqat_id = md5("".join((kilt_id, qid, inp, url)))
        vq = {'input': inp,
              'url': url,
              'wikidata_id': qid,
              'meerqat_id': meerqat_id}
        item['vq'].append(vq)

    return item


def generate_vqs(subset, exclude_categories=set()):
    """
    Parameters
    ----------
    subset: str
        Name of the subset to load (e.g. validation_triviaqa)
    exclude_categories: set, optional
        Exclude image where these keywords are included in one of its categories
        e.g. {'cosplay'} might save you some trouble with GDPR
        Defaults to empty set (i.e. keep all)
    """
    # load data
    dataset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    dataset = load_from_disk(dataset_path)
    with open(dataset_path / "entities.json", 'r') as file:
        entities = json.load(file)

    # sort images and remove forbidden ones (move to wiki.py if it's too slow?)
    for entity in entities.values():
        images = entity.get("images")
        if not images:
            continue

        # remove reserved images (e.g. illustrative_image) from the candidates
        for reserved_image_key in RESERVED_IMAGES:
            for reserved_image in map(special_path_to_file_name, entity.get(reserved_image_key, {})):
                images.pop(reserved_image, None)

        # Exclude image where these keywords are included in one of its categories
        if exclude_categories:
            todel = []
            for title, image in images.items():
                del_image = False
                for image_category in image.get("categories", []):
                    image_category = image_category.lower()
                    for category_to_exclude in exclude_categories:
                        if category_to_exclude in image_category:
                            del_image = True
                            break
                    if del_image:
                        todel.append(title)
                        break
            for title in todel:
                images.pop(title)

        # sort images w.r.t. their score in ASCENDING order (allows simpler use of pop)
        entity["titles"] = sorted(images, key=lambda title: len(images[title]['heuristics']))

    # go through dataset
    dataset = dataset.map(generate_vq, fn_kwargs=dict(entities=entities))

    # save data
    dataset.save_to_disk(dataset_path)
    print(f"Successfully saved output to '{dataset_path}'")


def labelstudio(subset):
    """convert dataset to the Label Studio JSON format"""
    raise NotImplementedError()


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    wer_threshold = float(args['--threshold']) if args['--threshold'] else 0.5

    if args['ner']:
        ner(subset)
    elif args['ned']:
        ned(subset)
    elif args['count_entities']:
        count_entities(subset, wer_threshold=wer_threshold)
    elif args['generate']:
        if args['mentions']:
            generate_mentions(subset, wer_threshold=wer_threshold)
        elif args['vq']:
            categories_to_exclude = set(args['<categories_to_exclude>'])
            generate_vqs(subset, categories_to_exclude)
    elif args['labelstudio']:
        labelstudio(subset)

