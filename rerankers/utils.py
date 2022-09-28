import json
import numpy as np
from datasets import load_from_disk, set_caching_enabled
set_caching_enabled(False)


import torch
import os.path as osp
from tqdm import tqdm
import pickle, json, random
from functools import partial

from ranx import Qrels, Run, evaluate, compare
from ranx import fuse, optimize_fusion
###############################################################
def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)

###############################################################

def parse_value(value, search_key):
    if search_key == "passage":
        return str(int(value))
    elif search_key == "document":
        return str(value)
    else: 
        raise('invalid search_key')


def get_face_predictions(image_preds, faces):
    """
        Get Image Ranker (e.g. RRT) predictions about faces
    """
    assert(len(faces) == len(image_preds))
    n_queries = len(faces)

    face_predictions = []
    for i in range(n_queries):
        
        if not faces[i]:
            continue
        
        query_dict = {}
        query_dict["indices"] = image_preds[i]['indices']
        query_dict["scores"]         = np.array(image_preds[i]["scores"])
        #query_dict["ranks"]          = np.array(image_preds[i]["ranks"])
        #query_dict["ranked_indices"] = np.array(image_preds[i]["ranked_indices"])        
        query_dict["ranks"]          = (-query_dict["scores"]).argsort()
        query_dict["ranked_indices"] = list(np.array(query_dict["indices"])[query_dict["ranks"]])
        query_dict["gold_indices"]   = image_preds[i]["gold_indices"]
        
        if "images" in image_preds[i].keys():
            query_dict["gold_images"]  = image_preds[i]["gold_images"]
            query_dict["images"]       = image_preds[i]["images"]
            query_dict["image_scores"] = image_preds[i]["image_scores"]
            query_dict["gold_images"]  = image_preds[i]["gold_images"]
        
        face_predictions.append(query_dict)
        
    return face_predictions


def get_ir_ranking(ranker_preds):
    """
        Get IR original ranking From any Ranker
    """
    n_queries = len(ranker_preds)

    formatted_predictions = []
    for i in range(n_queries):
        query_dict = {}
        query_dict["indices"]        = ranker_preds[i]["indices"]
        scores = np.linspace(0, 1, len(ranker_preds[i]["indices"]))
        scores = scores / scores.sum()
        query_dict["scores"]         = np.flip(scores)
        query_dict["ranks"]          = np.arange(0, len(ranker_preds[i]["indices"]), 1)
        query_dict["ranked_indices"] = ranker_preds[i]["indices"]
        
        query_dict["gold_indices"]   = ranker_preds[i]["gold_indices"]
        
        #['indices', 'scores', 'ranks', 'ranked_indices', 'gold_indices', 'images', 'image_scores', 'gold_images']
        if "images" in ranker_preds[i].keys():
            query_dict["gold_images"]  = ranker_preds[i]["gold_images"]
            query_dict["images"]       = ranker_preds[i]["images"]
            query_dict["image_scores"] = ranker_preds[i]["image_scores"]
            query_dict["gold_images"]  = ranker_preds[i]["gold_images"]
        
        formatted_predictions.append(query_dict)
        
    return formatted_predictions



def format_text_ranker(text_preds, normalize_before=True, use_softmax=True):
    """
        Format text ranker
        Ouput scores are standardized (mean=0 and std=1)
        + scores are defined as log probabilities
    """
    n_queries = len(text_preds)
    logsoftmax = torch.nn.LogSoftmax(dim=0)

    formatted_predictions = []
    for i in range(n_queries):
        query_dict = {}
        
        query_dict["indices"] = text_preds[i]["indices"]
        scores = text_preds[i]["scores"]
        
        if normalize_before:
            scores   = _normalize(scores)
        if use_softmax:
            scores   = logsoftmax(torch.tensor(scores)).numpy()
        
        query_dict["scores"]         = np.array(scores)
        query_dict["ranks"]          = np.array(text_preds[i]["ranks"])
        query_dict["ranked_indices"] = np.array(text_preds[i]["ranked_indices"])
        
        query_dict["gold_indices"]   = text_preds[i]["gold_indices"]
        
        formatted_predictions.append(query_dict)
        
    return formatted_predictions




def grid_search_for_fusion(ranker1, ranker2, alphas, search_key, func):
    results = []
    metrics = []
    for i in range(len(alphas)):
        alpha = float(alphas[i])
        combined_predictions = func(ranker1, ranker2, alpha, 1 - alpha)
        metric = compute_metrics(combined_predictions, search_key=search_key)
        metrics.append(metric)
        results.append(metric['map'])
    
    index_array = np.argsort(-np.array(results))
    _alpha      = float(np.round(alphas[index_array[0]], 2))
    
    print("Winning combination is: (", _alpha, ", ", 1.0 - _alpha, ")")
    
    print(metrics[index_array[0]])
    
    return _alpha, index_array, results, metrics



def _normalize(array):
    _array = np.array(array)
    
    if _array.std() == 0.0:
        return np.ones_like(_array)
        
    
    _array = (_array - _array.mean())/(_array.std())
    return _array



def get_key_by_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def document_level_predictions(predictions, passage2image):
    
    # for every question, get the list of the top 100 search results
    for i in tqdm(range(len(predictions))):
        
        index_to_img = {}
        indices = predictions[i]['indices']
        scores  = predictions[i]['scores'] if len(predictions[i]['scores']) > 0 else [0] * len(predictions[i]['indices'])
        index_to_score = dict(zip(indices, scores))
        
        for index in indices:
            img = passage2image[str(index)]
            index_to_img[index] = img
        
        images = list(set(index_to_img.values()))
        
        predictions[i]['images'] = images
        predictions[i]['image_scores'] = [index_to_score[get_key_by_value(index_to_img, image)] for image in images]
        
        predictions[i]["gold_images"]   = list(set([passage2image[str(index)] for index in predictions[i]["gold_indices"]]))
    
    return predictions
     


def optimize_fusion_rankers(rankers, search_key="document", 
                            norm="zmuv", method="wsum", metric="map"):
    parse = partial(parse_value, search_key=search_key)
    
    n_queries = len(rankers[0])
    for i in range(len(rankers)):
        assert(n_queries == len(rankers[i]))
        
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
    qrels_dict = {}
    runs = [{} for i in range(len(rankers))]
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        
        ok_inds = rankers[0][i][gold_key]
        if len(ok_inds) == 0:
            qrels_dict[q_str] = {"DUMMY_RUN": 0}
        else:
            qrels_dict[q_str] = dict([('d_' + parse(key), 1) for key in ok_inds])
        
        for r in range(len(rankers)):
            item = rankers[r][i]
            rankings = item[ranking_key]
            index_to_score = dict(zip(item[original_key], item[score_key]))
            runs[r][q_str] = dict([('d_' + parse(rankings[index]), index_to_score[rankings[index]]) for index in range(len(rankings))])
    
    qrels = Qrels(qrels_dict)
    runs = [Run(runs[r]) for r in range(len(rankers))]
    
    best_params = optimize_fusion(qrels=qrels,
        runs=runs,
        norm=norm,
        method=method,
        metric=metric,  # The metric to maximize during optimization
    )
    
    return best_params
    


def fuse_rankers(rankers, params=None, search_key="document", norm="zmuv", method="wsum"):
    parse = partial(parse_value, search_key=search_key)

    n_queries = len(rankers[0])
    for i in range(len(rankers)):
        assert(n_queries == len(rankers[i]))
        
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
    qrels_dict = {}
    runs = [{} for i in range(len(rankers))]
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        
        for r in range(len(rankers)):
            item = rankers[r][i]
            rankings = item[ranking_key]
            index_to_score = dict(zip(item[original_key], item[score_key]))
            runs[r][q_str] = dict([('d_' + parse(rankings[index]), index_to_score[rankings[index]]) for index in range(len(rankings))])
    
    #qrels = Qrels(qrels_dict)
    runs = [Run(runs[r]) for r in range(len(rankers))]
    if params is None:
        combined_test_run = fuse(runs=runs, norm=norm, method=method)
    else:
        combined_test_run = fuse(runs=runs, norm=norm, method=method, params=params)
    
    return combined_test_run


def create_runs(rankers, names, search_key="passage"):
    parse = partial(parse_value, search_key=search_key)
    
    n_queries = len(rankers[0])
    for i in range(len(rankers)):
        assert(n_queries == len(rankers[i]))
    
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
    
    qrels_dict = {}
    runs = [{} for i in range(len(rankers))]
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        
        for r in range(len(rankers)):
            item = rankers[r][i]
            rankings = item[ranking_key]
            index_to_score = dict(zip(item[original_key], item[score_key]))
            runs[r][q_str] = dict([('d_' + parse(rankings[index]), index_to_score[rankings[index]]) for index in range(len(rankings))])
    
    runs = [Run(runs[r], name=names[r]) for r in range(len(rankers))]
    
    return runs


def create_qrels(rankers, search_key="passage"):
    parse = partial(parse_value, search_key=search_key)

    n_queries = len(rankers[0])
    for i in range(len(rankers)):
        assert(n_queries == len(rankers[i]))
        
    qrels_dict = {}
    runs = [{} for i in range(len(rankers))]
    
    if search_key == "passage":
        gold_key     = "gold_indices"
    else:
        gold_key     = "gold_images"
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        
        ok_inds = rankers[0][i][gold_key]
        if len(ok_inds) == 0:
            qrels_dict[q_str] = {"DUMMY_RUN": 0}
        else:
            qrels_dict[q_str] = dict([('d_' + parse(key), 1) for key in ok_inds])
     
    qrels = Qrels(qrels_dict)
    
    return qrels



def ranx_evaluate(qrels, run):
    # metrics to be used during evaluation at [1, 5, 10]
    metric_names = ["map", "mrr", "precision", "hit_rate", "recall"]
    unwanted     = ["map@1", "mrr@1", "map@5", "mrr@5", "map@10", "mrr@10", "hit_rate@1", "mrr@"]
    m_list = [metric for metric in metric_names]
    kappas = [1, 5, 10]
    
    for i in range(len(kappas)):
        m_list.extend([metric+'@'+str(kappas[i]) for metric in metric_names])
    
    m_list = ['mrr', 'hit_rate', 'precision',
                    'precision@1', 'precision@5', 'precision@20',
                    'hit_rate@5', 'hit_rate@20']
    
    metrics = evaluate(qrels, run, m_list)
    
    return metrics



def ranx_compare(qrels, runs,
                 max_p=0.01, rounding_digits=4,
                 show_percentages=True):
    
    # metrics to be used during evaluation at [1, 5, 10]
    metric_names = ["map", "mrr", "precision", "hit_rate", "recall"]
    unwanted     = ["map@1", "mrr@1", "map@5", "mrr@5", "map@10", "mrr@10", "hit_rate@1", "mrr@"]
    m_list = [metric for metric in metric_names]
    kappas = [1, 5, 10]
    
    for i in range(len(kappas)):
        m_list.extend([metric+'@'+str(kappas[i]) for metric in metric_names])
    
    m_list = ['mrr', 'hit_rate', 'precision',
                    'precision@1', 'precision@5', 'precision@20',
                    'hit_rate@5', 'hit_rate@20']
    
    report = compare(qrels=qrels, 
                     runs=runs,
                     metrics=m_list, 
                     max_p=max_p,
                     rounding_digits=rounding_digits,
                     show_percentages=show_percentages)
    
    return report


# ## Metrics

def reformat(out, decimals=2):
    for key, value in out.items():
            out[key] = np.around(value*100, decimals=decimals)
    return out

def compute_metrics(ranker_predictions, search_key="passage", decimals=2):

    # metrics to be used during evaluation at [1, 5, 10]
    metric_names = ["map", "mrr", "precision", "hit_rate", "recall"]
    unwanted     = ["map@1", "mrr@1", "map@5", "mrr@5", "map@10", "mrr@10", "hit_rate@1", "mrr@"]
    m_list = [metric for metric in metric_names]
    kappas = [1, 5, 10]
    
    for i in range(len(kappas)):
        m_list.extend([metric+'@'+str(kappas[i]) for metric in metric_names])
    
    m_list = ['mrr', 'precision@1', 'precision@5', 
              'precision@20', 'hit_rate@5', 'hit_rate@20']
    n_queries = len(ranker_predictions)
    qrels_dict = {}
    run_dict = {}
    
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
    
    parse = partial(parse_value, search_key=search_key)
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        item = ranker_predictions[i]
        ok_inds = item[gold_key]
        if len(ok_inds) == 0:
            qrels_dict[q_str] = {"DUMMY_RUN": 0}
        else:
            qrels_dict[q_str] = dict([('d_' + parse(key), 1) for key in ok_inds])
        
        rankings = item[ranking_key]
        index_to_score = dict(zip(item[original_key], item[score_key]))
        run_dict[q_str] = dict([('d_' + parse(rankings[index]), index_to_score[rankings[index]]) for index in range(len(rankings))])
    
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    metrics = evaluate(qrels, run, m_list)
    metrics = reformat(metrics)
    
    return metrics


# ## Comparaison

def compare_rankers(rankers, search_key="passage",
                    max_p=0.01, decimals=2,
                    show_percentages=True,):

    # metrics to be used during evaluation at [1, 5, 10]
    metric_names = ["map", "mrr", "precision", "hit_rate", "recall"]
    unwanted     = ["map@1", "mrr@1", "map@5", "mrr@5", "map@10", "mrr@10", "hit_rate@1", "mrr@"]
    m_list = [metric for metric in metric_names]
    kappas = [1, 5, 10]
    
    for i in range(len(kappas)):
        m_list.extend([metric+'@'+str(kappas[i]) for metric in metric_names])
    
    m_list = ['mrr', 'hit_rate', 'precision',
                    'precision@1', 'precision@5', 'precision@20',
                    'hit_rate@5', 'hit_rate@20']
    n_queries = len(rankers[0])
    for i in range(len(rankers)):
        assert(n_queries == len(rankers[i]))
    
    qrels_dict = {}
    runs = [{} for i in range(len(rankers))]
    
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
    
    parse = partial(parse_value, search_key=search_key)
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        
        ok_inds = rankers[0][i][gold_key]
        if len(ok_inds) == 0:
            qrels_dict[q_str] = {"DUMMY_RUN": 0}
        else:
            qrels_dict[q_str] = dict([('d_' + parse(key), 1) for key in ok_inds])
        
        for r in range(len(rankers)):
            item = rankers[r][i]
            rankings = item[ranking_key]
            index_to_score = dict(zip(item[original_key], item[score_key]))
            runs[r][q_str] = dict([('d_' + parse(rankings[index]), index_to_score[rankings[index]]) for index in range(len(rankings))])
    
    qrels = Qrels(qrels_dict)
    runs = [Run(runs[r]) for r in range(len(rankers))]

    report = compare(qrels=qrels, 
                     runs=runs,
                     metrics=m_list, 
                     max_p=max_p,
                     rounding_digits=2+decimals,
                     show_percentages=show_percentages)
    
    return report


# ## Fusion

def combine_image_rankers(arcface_preds, image_preds, normalize_before=True, use_softmax=True):
    """
        Combine Image Rankers (e.g. ArcFace and RRT)
        Ouput scores are standardized (mean=0 and std=1)
        + scores are defined as log probabilities
    """
    assert(len(arcface_preds) == len(image_preds))
    n_queries = len(arcface_preds)
    logsoftmax = torch.nn.LogSoftmax(dim=0)

    combined_predictions = []
    for i in range(n_queries):
        query_dict = {}
        indices = image_preds[i]['indices']
        query_dict["indices"] = indices
        
        if arcface_preds[i]["face"]:
            #we transform face scores of human entities into log probabilities space using logsoftmax
            arcface_indices = arcface_preds[i]["used_indices"]
            arcface_scores  = arcface_preds[i]["scores"]
            #we transform scores of non-human entities into log probabilities space using logsoftmax
            image_scores  = image_preds[i]["scores"]
            
            if normalize_before:
                arcface_scores = _normalize(arcface_preds[i]["scores"])
                image_scores   = _normalize(image_preds[i]["scores"])
            
            if use_softmax:
                arcface_scores = logsoftmax(torch.tensor(arcface_scores)).numpy()
                image_scores   = logsoftmax(torch.tensor(image_scores)).numpy()
            
            arcface_indices_to_scores = dict(zip(indices, arcface_scores))
            image_indices_to_scores   = dict(zip(indices, image_scores))
            
            scores = []
            for index in indices:
                if index in arcface_indices:
                    scores.append(arcface_indices_to_scores[index])
                else:
                    scores.append(image_indices_to_scores[index])
            
            query_dict["scores"]         = np.array(scores)
            query_dict["ranks"]          = (-query_dict["scores"]).argsort()
            query_dict["ranked_indices"] = list(np.array(query_dict["indices"])[query_dict["ranks"]])
            
        else:
            if normalize_before:
                image_scores   = _normalize(image_preds[i]["scores"])
            if use_softmax:
                image_scores   = logsoftmax(torch.tensor(image_scores)).numpy()
            
            query_dict["scores"]         = np.array(image_scores)
            query_dict["ranks"]          = np.array(image_preds[i]["ranks"])
            query_dict["ranked_indices"] = np.array(image_preds[i]["ranked_indices"])
        
        query_dict["gold_indices"]   = image_preds[i]["gold_indices"]
        
        combined_predictions.append(query_dict)
        
    return combined_predictions



def remove_prefix_and_map(array, prefix='d_', mapping=str):
    new_array = []
    for value in array:
        new_array.append(mapping(value[len(prefix):]))
    return new_array


def run_to_predictions(run, search_key="passage"):
    run_dict = run.to_dict()
    n_queries = len(run_dict)
    predictions = []
    
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
        mapping      = int
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        mapping      = str
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        query_dict = {}
        original_values = remove_prefix_and_map(list(run_dict[q_str].keys()), prefix='d_', mapping=mapping)
        query_dict[original_key] = original_values
        query_dict[ranking_key] = original_values
        query_dict[score_key] = list(run_dict[q_str].values())
        predictions.append(query_dict)
        
    return predictions


def predictions_add_gold(predictions, reference_preds, search_key="passage"):
    n_queries = len(predictions)
    
    if search_key == "passage":
        gold_key     = "gold_indices"
        ranking_key  = "ranked_indices"
        original_key = "indices"
        score_key    = "scores" 
    else:
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
    
    for i in range(n_queries):
        predictions[i][gold_key] = reference_preds[i][gold_key]
        
    return predictions


def document_level_predictions(predictions, passage2image):
    
    # for every question, get the list of the top 100 search results
    for i in tqdm(range(len(predictions))):
        
        index_to_img = {}
        indices = predictions[i]['indices']
        scores  = predictions[i]['scores'] if len(predictions[i]['scores']) > 0 else [0] * len(predictions[i]['indices'])
        index_to_score = dict(zip(indices, scores))
        
        for index in indices:
            img = passage2image[str(index)]
            index_to_img[index] = img
        
        images = list(set(index_to_img.values()))
        
        predictions[i]['images'] = images
        predictions[i]['image_scores'] = [index_to_score[get_key_by_value(index_to_img, image)] for image in images]
        
        predictions[i]["gold_images"]   = list(set([passage2image[str(index)] for index in predictions[i]["gold_indices"]]))
    
    return predictions
     


def passage_level_predictions(predictions, reference_preds, passage2image):
    n_queries = len(predictions)
    
    gold_key     = "gold_indices"
    ranking_key  = "ranked_indices"
    original_key = "indices"
    score_key    = "scores"
    
    for i in range(n_queries):
        img_scores = dict(zip(predictions[i]["images"], predictions[i]["image_scores"]))
        indices = reference_preds[i][original_key]
        predictions[i][original_key] = indices
        predictions[i][ranking_key] = indices
        predictions[i][gold_key] = reference_preds[i][gold_key]
        
        scores = []
        for idx in indices:
            image = passage2image[str(idx)]
            scores.append(img_scores[image])
            
        predictions[i][score_key] = scores
        
    return predictions


def combine_image_rankers_for_face_predictions(arcface_preds, rrt_preds, alpha, beta, normalize_before=True, use_softmax=True):
    """
        Combine Image Rankers (e.g. ArcFace and RRT)
        Ouput scores are standardized (mean=0 and std=1)
        + scores are defined as log probabilities
    """
    assert(len(arcface_preds) == len(rrt_preds))
    n_queries = len(arcface_preds)
    logsoftmax = torch.nn.LogSoftmax(dim=0)

    combined_predictions = []
    for i in range(n_queries):
        query_dict = {}
        images = rrt_preds[i]['images']
        query_dict["images"] = images
        
        gold_key     = "gold_images"
        ranking_key  = "images"
        original_key = "images"
        score_key    = "image_scores"
        
        #we transform face scores of human entities into log probabilities space using logsoftmax
        #arcface_images = arcface_preds[i]["images"]
        arcface_scores  = arcface_preds[i]["image_scores"]
        #we transform scores of non-human entities into log probabilities space using logsoftmax
        rrt_scores  = rrt_preds[i]["image_scores"]

        if normalize_before:
            arcface_scores = _normalize(arcface_scores)
            rrt_scores     = _normalize(rrt_scores)

        if use_softmax:
            arcface_scores = logsoftmax(torch.tensor(arcface_scores)).numpy()
            rrt_scores     = logsoftmax(torch.tensor(rrt_scores)).numpy()

        arcface_img_to_score = dict(zip(images, arcface_scores))
        rrt_img_to_score     = dict(zip(images, rrt_scores))

        scores = []
        for img in images:
            scores.append(alpha * arcface_img_to_score[img] + beta * rrt_img_to_score[img])

        query_dict["image_scores"]  = np.array(scores)
        query_dict["image_ranks"]   = (-query_dict["image_scores"]).argsort()
        query_dict["ranked_images"] = list(np.array(query_dict["images"])[query_dict["image_ranks"]])
        query_dict["gold_images"]   = rrt_preds[i]["gold_images"]
        
        combined_predictions.append(query_dict)
        
    return combined_predictions



def combine_document_level_image_rankers(face_only_predictions, no_face_predictions, faces):
    """
        Combine Image Rankers (e.g. ArcFace and RRT)
        Ouput scores are standardized (mean=0 and std=1)
        + scores are defined as log probabilities
    """
    assert(len(faces) == len(face_only_predictions) + len(no_face_predictions))
    n_queries = len(faces)
    face_counter = 0
    no_face_counter = 0

    combined_predictions = []
    for i in range(n_queries):
        if faces[i]:
            query_dict =  face_only_predictions[face_counter]
            face_counter = face_counter + 1
        else:
            query_dict =  no_face_predictions[no_face_counter]
            no_face_counter = no_face_counter + 1
        
        combined_predictions.append(query_dict)
        
    return combined_predictions

