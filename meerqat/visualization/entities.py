# coding: utf-8
"""Usage:
entities.py <subset>
entities.py images [--n=<n> --max=<max>] <subset>

Options:
--n=<n>           Number of entities [default: 10]
--max=<max>       Max. number of images per entity [default: 18]
"""

from docopt import docopt
import json
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from meerqat.data.loading import DATA_ROOT_PATH
from meerqat.data.wiki import get_license, license_score, special_path_to_file_name, RESERVED_IMAGES
from meerqat.visualization.utils import simple_stats

HTML_FORMAT = """
<html>
    <head>
        <link rel="stylesheet" href="../styles.css">
    </head>
    <a href="{previous}">previous</a>
    {entity}
</html>
"""
ENTITY_FORMAT = """
<h2>{qid} ({label}) illustrative image: <img src="{url}" width="400"></h2>
<table>
    {trs}
</table>
"""
TR_format = "<tr>{tds}</tr>"
TD_FORMAT = '<td><p>{description}</p><p>{license}</p><p>{score}</p><p>{heuristics}</p><img src="{url}" width="200"></td>'


def count_entities(entities, distinct=False):
    """Note this counts labels and not QIDs

    Parameters
    ----------
    entities: List[dict]
    distinct: bool
        Whether to count distinct entities or # of questions per entity
        e.g. if we have 2 questions about Barack Obama, 'distinct' counts one human
        and '!distinct' counts 2
        This has no effect on "depiction_dist"
        which counts the # of depictions per (distinct) entity

    Returns
    -------
    counters: dict[Counter]
    """
    counters = {
        "commons": Counter(),
        "image": Counter(),
        "instanceof": Counter(),
        "gender": Counter(),
        "occupation": Counter(),
        "depictions": Counter(),
        "depiction_dist": []
    }
    for entity in entities.values():
        n = 1 if distinct else entity["n_questions"]

        # is commons category, image or depictions available ?
        for key in ["commons", "image", "depictions"]:
            counters[key][bool(entity.get(key))] += n

        # how many depictions per entity ?
        counters["depiction_dist"].append(len(entity.get("depictions", [])))

        # does it have a gender ? if yes, which one ?
        genderLabel = entity.get('genderLabel')
        if genderLabel:
            counters["gender"][genderLabel["value"]] += n

        # else count all available values
        for key in ["instanceof", "occupation"]:
            if key not in entity:
                continue
            for item in entity[key].values():
                counters[key][item["label"]["value"]] += n
    return counters


def visualize_entities(counters, path=Path.cwd(), subset="meerqat"):
    # pie-plot counters with lot of values
    for key in ["instanceof", "occupation"]:
        counter = counters[key]

        # keep only the last decile for a better readability
        values = np.array(list(counter.values()))
        labels = np.array(list(counter.keys()))
        deciles = np.quantile(values, np.arange(0., 1.1, 0.1))
        where = values > deciles[-2]
        filtered_values = np.concatenate((values[where], values[~where].sum(keepdims=True)))
        filtered_labels = np.concatenate((labels[where], ["other"]))

        # plot and save figure
        plt.figure(figsize=(16, 16))
        title = f"Distribution of {key} in {subset}"
        plt.title(title)
        plt.pie(filtered_values, labels=filtered_labels)
        output = path / title.replace(" ", "_")
        plt.savefig(output)
        plt.close()
        print(f"Successfully saved {output}")

    # barplot distributions and print some stats
    for key in ["depiction_dist"]:
        counter = counters[key]

        # print some stats
        print(simple_stats(counter, tablefmt="latex"))

        # barplot
        plt.figure(figsize=(16, 16))
        title = f"Distribution of {key} in {subset}"
        plt.title(title)
        plt.hist(counter, bins=50, density=False)
        output = path / title.replace(" ", "_")
        plt.savefig(output)
        plt.close()
        print(f"Successfully saved {output}")

    # print statistics for counters with fue values
    for key in ["gender", "commons", "image", "depictions"]:
        counter = counters[key]
        print(key)
        print(tabulate([counter], headers="keys", tablefmt="latex"), "\n\n")


def visualize_images(entities, output, n=10, max_images=18):
    previous = Path()
    for _ in tqdm(range(n)):
        qid, entity = entities.popitem()
        label = entity.get("entityLabel", {}).get("value")
        images = entity.get('images')
        illustrative_image = entity.get('image', {})
        if not (label and images and illustrative_image):
            continue
        # remove reserved images (e.g. illustrative_image) from the candidates
        for reserved_image_key in RESERVED_IMAGES:
            for reserved_image in map(special_path_to_file_name, entity.get(reserved_image_key, {})):
                images.pop(reserved_image, None)

        # TODO remove images with "cosplay" in category

        trs, tds = [], []
        # sort images 1. by heuristic score, 2. by permissivity of the license
        sorted_images = sorted(images,
                               key=lambda name: (len(images[name]['heuristics']), license_score(images[name])),
                               reverse=True)
        for i, title in enumerate(sorted_images[:max_images]):
            image = images[title]
            url = image.get("url")
            if not url:
                continue
            td = TD_FORMAT.format(description=image.get("description", {}).get("value", ""),
                                  url=url,
                                  license=get_license(image),
                                  score=len(image['heuristics']),
                                  heuristics=", ".join(image['heuristics']))
            tds.append(td)
            if (i + 1) % 6 == 0 or i == max_images - 1:
                trs.append(TR_format.format(tds="\n".join(tds)))
                tds = []
        html_entity = ENTITY_FORMAT.format(qid=qid, label=label, url=next(iter(illustrative_image)), trs="\n".join(trs))
        html = HTML_FORMAT.format(entity=html_entity, previous=previous.name)
        previous = output / f"{qid}.html"
        with open(previous, 'w') as file:
            file.write(html)


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    subset = args['<subset>']
    path = DATA_ROOT_PATH / f"meerqat_{subset}" / "entities.json"
    with open(path) as file:
        entities = json.load(file)
    output = DATA_ROOT_PATH / "visualization" / str(hash(str(args)))
    output.mkdir(exist_ok=True)

    if args['images']:
        n = int(args['--n'])
        max_images = int(args['--max'])
        visualize_images(entities, output, n=n, max_images=max_images)
    else:
        counters = count_entities(entities)
        visualize_entities(counters, output, subset)
