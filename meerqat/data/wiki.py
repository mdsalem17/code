# coding: utf-8
"""Usage:
wiki.py data entities <subset>
wiki.py data depicted <subset>
wiki.py commons sparql depicts <subset>
wiki.py commons sparql depicted <subset>
"""
import time
import json
import warnings

from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError
from tqdm import tqdm
from docopt import docopt

from meerqat.data.loading import DATA_ROOT_PATH

# Template for wikidata to query 'instance of' (P31), 'commons category' (P373),
# 'image' (P18), 'occupation' (P106) and 'gender' (P21) given a list of entities
# should be used like 'WIKIDATA_QUERY % "wd:Q76 wd:Q78579194 wd:Q42 wd:Q243"'
# i.e. entity ids are space-separated and prefixed by 'wd:'
WIKIDATA_QUERY = """
SELECT ?entity ?entityLabel ?instanceof ?instanceofLabel ?commons ?image ?occupation ?occupationLabel ?gender ?genderLabel
{
  VALUES ?entity { %s }
  OPTIONAL{ ?entity wdt:P373 ?commons . }
  ?entity wdt:P31 ?instanceof .
  OPTIONAL { ?entity wdt:P18 ?image . }
  OPTIONAL { ?entity wdt:P21 ?gender . }
  OPTIONAL { ?entity wdt:P106 ?occupation . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# template for beta-commons SPARQL API to query images that depict (P180) entities
# same usage as WIKIDATA_QUERY
COMMONS_SPARQL_QUERY = """
SELECT ?depicted_entity ?commons_entity ?special_path ?url ?encoding WHERE {
  VALUES ?depicted_entity { %s }
  ?commons_entity wdt:P180 ?depicted_entity .
  ?commons_entity schema:contentUrl ?url .
  ?commons_entity schema:encodingFormat ?encoding .
  # restrict media to be images handleable by PIL.Image
  VALUES ?encoding { "image/png" "image/jpg" "image/jpeg" "image/tiff" "image/gif" }
  bind(iri(concat("http://commons.wikimedia.org/wiki/Special:FilePath/", wikibase:decodeUri(substr(str(?url),53)))) AS ?special_path)
}
"""
# query entities depicted in images given image identifier (see above for more details)
COMMONS_DEPICTED_ENTITIES_QUERY = """
SELECT ?commons_entity ?depicted_entity WHERE {
  VALUES ?commons_entity { %s }
  ?commons_entity wdt:P180 ?depicted_entity .
}
"""
COMMONS_SPARQL_ENDPOINT = "https://wcqs-beta.wmflabs.org/sparql"


def query_sparql_entities(query, endpoint, wikidata_ids, prefix='wd:',
                          n=100, return_format=JSON, description=None):
    """
    Queries query%entities by batch of n (defaults 100)
    where entities is n QIDs in wikidata_ids space-separated and prefixed by prefix
    (should be 'wd:' for Wikidata entities and 'sdc:' for Commons entities)

    Returns query results
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(return_format)
    results, qids = [], []
    # query only n qid at a time
    for i, qid in enumerate(tqdm(wikidata_ids, desc=description)):
        qids.append(prefix+qid)
        if (i + 1) % n == 0 or i == (len(wikidata_ids) - 1):
            sparql.setQuery(query % " ".join(qids))
            try:
                response = sparql.query()
            except HTTPError:
                # HACK: sleep 60s to avoid 'HTTP Error 429: Too Many Requests'
                time.sleep(60)
                # try one more time
                try: 
                    response = sparql.query()
                except HTTPError as e:
                    warnings.warn(f"Query failed twice after waiting 60s in-between, skipping the following qids:\n{qids}")
                    qids = []
                    continue
            results += response.convert()['results']['bindings']
            qids = []
    print(f"Query succeeded! Got {len(results)} results")

    return results


def update_from_data(entities):
    """Updates entities with info queried in from Wikidata"""

    # query Wikidata
    results = query_sparql_entities(WIKIDATA_QUERY, WIKIDATA_ENDPOINT, entities.keys(),
                                    description="Querying Wikidata")

    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['entity']['value'].split('/')[-1]
        # handle keys/attributes that are unique
        for unique_key in ({'entityLabel', 'gender', 'genderLabel', 'image', 'commons'} & result.keys()):
            # simply add or update the key/attribute
            entities[qid][unique_key] = result[unique_key]
        # handle keys/attributes that may be multiple
        for multiple_key in ({'instanceof', 'occupation'} & result.keys()):
            # create a new dict for this key/attribute so we don't duplicate data
            entities[qid].setdefault(multiple_key, {})
            # store corresponding label in the 'label' field
            result[multiple_key]['label'] = result[multiple_key + 'Label']
            # value (e.g. QID) of the attribute serves as key
            multiple_value = result[multiple_key]['value']
            entities[qid][multiple_key][multiple_value] = result[multiple_key]

    return entities


def update_from_commons_sparql(entities):
    # query Wikimedia Commons
    results = query_sparql_entities(COMMONS_SPARQL_QUERY, COMMONS_SPARQL_ENDPOINT,
                                    entities.keys(),
                                    description="Querying Wikimedia Commons")

    # update entities with results
    for result in tqdm(results, desc="Updating entities"):
        qid = result['depicted_entity']['value'].split('/')[-1]
        commons_qid = result['commons_entity']['value']
        # create a new key 'depictions' to store depictions in a dict
        entities[qid].setdefault("depictions", {})
        # use commons_qid (e.g. https://commons.wikimedia.org/entity/M88412327) as key in this dict
        entities[qid]["depictions"].setdefault(commons_qid, {})
        entities[qid]["depictions"][commons_qid]['url'] = result['url']
        entities[qid]["depictions"][commons_qid]['special_path'] = result['special_path']

    return entities


def query_depicted_entities(depictions):
    # query Wikimedia Commons
    results = query_sparql_entities(COMMONS_DEPICTED_ENTITIES_QUERY,
                                    COMMONS_SPARQL_ENDPOINT,
                                    depictions.keys(), prefix="sdc:",
                                    description="Querying Wikimedia Commons")
    # update depictions with results
    for result in tqdm(results, desc="Updating depictions"):
        qid = result['commons_entity']['value'].split('/')[-1]
        depictions[qid].append(result["depicted_entity"]['value'])
    return depictions


def depiction_instanceof_heuristic(depictions, entities):
    for qid, entity in tqdm(entities.items(), desc="Applying 'instanceof' heuristic"):
        if 'instanceof' not in entity:
            continue
        instanceof = entity['instanceof'].keys()
        entity_depictions = entity.get("depictions", {})    
        for mid, depiction in entity_depictions.items():
            mid = mid.split('/')[-1]
            depiction["prominent_instanceof_heuristic"] = True
            # iterate over all other entities depicted in depiction
            for other_qid in depictions[mid]:
                other_qid = other_qid.split('/')[-1]
                # skip self
                if other_qid == qid:
                    continue
                other_entity = entities[other_qid]
                other_instanceof = other_entity.get('instanceof', {}).keys()
                # heuristic: the depiction is prominent if the entity is the only one of the same instance
                # e.g. pic of Barack Obama and Joe Biden -> not prominent
                #      pic of Barack Obama and the Eiffel Tower -> prominent
                if instanceof & other_instanceof:
                    depiction["prominent_instanceof_heuristic"] = False
                    break
    return entities


def keep_prominent_depictions(entities):
    for entity in entities.values():
        depictions = entity.get("depictions")
        if not depictions:
            continue
        # filter out non-prominent depictions
        entity["depictions"] = {mid: depiction for mid, depiction in depictions.items()
                                if depiction.get('prominent_instanceof_heuristic', False)}
    return entities


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']

    # load entities
    subset_path = DATA_ROOT_PATH / f"meerqat_{subset}"
    path = subset_path / "entities.json"
    with open(path) as file:
        entities = json.load(file)
    depictions_path = subset_path / "depictions.json"

    # update from Wikidata or Wikimedia Commons
    if args['data']:
        if args['entities']:
            output = update_from_data(entities)

        elif args['depicted']:
            # load depictions
            with open(depictions_path) as file:
                depictions = json.load(file)
            depicted_entities = {qid.split('/')[-1]: {"n_questions": 0} 
                                 for depiction in depictions.values() 
                                 for qid in depiction}
            # query data about all depicted entities
            depicted_entities = update_from_data(depicted_entities)
            # update with the original entities data
            depicted_entities.update(entities)
            # apply "instance of" heuristic to tell if a depiction is prominent or not
            # note the result is saved in 'entities' as it is entity-dependent
            # (the same picture can be prominent for entity A but not for B and C)
            output = depiction_instanceof_heuristic(depictions, depicted_entities)

    elif args['commons']:
        if args['sparql']:
            # find images that depict the entities
            if args['depicts']:
                output = update_from_commons_sparql(entities)

            # find entities depicted in the images
            elif args['depicted']:
                # get depictions
                depictions = {depiction.split('/')[-1]: []
                              for entity in entities.values()
                              for depiction in entity.get("depictions", {})}
                output = query_depicted_entities(depictions)
                path = depictions_path

    # save output
    with open(path, 'w') as file:
        json.dump(output, file)

    print(f"Successfully saved output to {path}")