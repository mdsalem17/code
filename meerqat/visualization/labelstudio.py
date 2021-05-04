# coding: utf-8
"""Usage: 
labelstudio.py html [--simple <path>...]
labelstudio.py stats [<path>...]

Options:
--simple            Only show valid VQA triple instead of original question and discarded questions.
"""

import json
from docopt import docopt
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from tabulate import tabulate

from meerqat.data.labelstudio import retrieve_vqa
from meerqat.visualization.kilt2vqa import HTML_FORMAT, TD_FORMAT, SIMPLE_TD_FORMAT

DISCARD_FORMAT = "&#9888; DISCARD: {reason}"


def write_html(completions, show_original=True):
    tds = []
    for completion_path in tqdm(completions):
        completion_path = Path(completion_path)
        with open(completion_path, 'r') as file:
            completion = json.load(file)
        vqa = retrieve_vqa(completion)
        discard = vqa.pop("discard", None)
        if discard is None:
            vq = vqa["vq"]
        elif not show_original:
            continue
        else:
            vq = DISCARD_FORMAT.format(reason=discard)
        if show_original:
            td = TD_FORMAT.format(
                original_question=vqa['question'],
                url=vqa['image'],
                generated_question=vq,
                answer=vqa['answer'],
                qid=vqa['wikidata_id']
            )
        else:
            td = SIMPLE_TD_FORMAT.format(
                url=vqa['image'],
                generated_question=vq,
                answer=vqa['answer']
            )
        tds.append(td)
    html = HTML_FORMAT.format(tds="\n".join(tds))
    with open(completion_path.parent.parent / 'vqa.html', 'w') as file:
        file.write(html)


def stats(completions):
    counter = Counter()
    for completion_path in tqdm(completions):
        completion_path = Path(completion_path)
        with open(completion_path, 'r') as file:
            completion = json.load(file)
        vqa = retrieve_vqa(completion)
        discard = vqa.pop("discard", None)
        if discard is None:
            counter['ok'] += 1
        else:
            counter[discard] += 1
    print(tabulate([counter], headers='keys'))


def main():
    args = docopt(__doc__)
    completions = args['<path>']
    if args['html']:
        show_original = not args['--simple']
        write_html(completions, show_original)
    elif args['stats']:
        stats(completions)


if __name__ == '__main__':
    main()
