import sys
import numpy as np

import torch
import os.path as osp
from tqdm import tqdm
import pickle, json, random
from functools import partial

from ranx import Qrels, Run, evaluate, compare
from ranx import fuse, optimize_fusion

from utils import *

def main(argv):
    passage2image = json_load('../data/viquae_wikipedia/passage2image.json')
    
    aranker_predictions = pickle_load('rankers/dev/arcface_ranker_predictions.pkl')
    test_aranker_predictions = pickle_load('rankers/test/arcface_ranker_predictions.pkl')
    
    iranker_predictions = pickle_load('rankers/dev/image_ranker_predictions.pkl')
    test_iranker_predictions = pickle_load('rankers/test/image_ranker_predictions.pkl')
    
    faces = [aranker_predictions[i]['face'] for i in range(len(aranker_predictions))]
    test_faces = [test_aranker_predictions[i]['face'] for i in range(len(test_aranker_predictions))]
    
    iranker_face_only_predictions = get_face_predictions(iranker_predictions, faces)
    iranker_face_only_predictions = document_level_predictions(
        iranker_face_only_predictions, passage2image)

    test_iranker_face_only_predictions = get_face_predictions(test_iranker_predictions, test_faces)
    test_iranker_face_only_predictions = document_level_predictions(
        test_iranker_face_only_predictions, passage2image)
    
    aranker_face_only_predictions = get_face_predictions(aranker_predictions, faces)
    aranker_face_only_predictions = document_level_predictions(
        aranker_face_only_predictions, passage2image)
    
    test_aranker_face_only_predictions = get_face_predictions(test_aranker_predictions, test_faces)
    test_aranker_face_only_predictions = document_level_predictions(
        test_aranker_face_only_predictions, passage2image)
    
    optim_norms = [#"zmuv", "max", "min-max",
                   "sum", "rank", "borda"]

    optim_methods = [#"rbc","wmnz", "wsum", "bayesfuse", "slidefuse", "probfuse", "rrf", "w_bordafuse", #"w_condorcet",
               "posfuse", "mapfuse"]#, "segfuse", "logn_isr", "mixed"]

    rankers = [aranker_face_only_predictions, iranker_face_only_predictions]
    search_key = "document"
    metric_to_optimize = 'mrr'
    optim_results = []
    for norm in optim_norms:
        for method in optim_methods:
            best_params = optimize_fusion_rankers(rankers, search_key=search_key, norm=norm, method=method, metric="map")
            combined_test_run = fuse_rankers(rankers, search_key=search_key, norm=norm, method=method, params=best_params)
            dev_qrels = create_qrels(rankers, search_key=search_key)
            output = ranx_evaluate(dev_qrels, combined_test_run)

            fusion_dict = {}
            fusion_dict["norm"]          = norm
            fusion_dict["method"]        = method
            fusion_dict["best_params"]   = best_params
            fusion_dict["metrics"]       = output
            optim_results.append(fusion_dict)

    
    metric_scores = np.array([output['metrics'][metric_to_optimize] for output in optim_results])
    
    no_faces = ~np.array(faces)
    test_no_faces = ~np.array(test_faces)
    
    aranker_no_face_predictions = get_face_predictions(aranker_predictions, no_faces)
    aranker_no_face_predictions = document_level_predictions(
        aranker_no_face_predictions, passage2image)

    test_aranker_no_face_predictions = get_face_predictions(test_aranker_predictions, test_no_faces)
    test_aranker_no_face_predictions = document_level_predictions(
        test_aranker_no_face_predictions, passage2image)
    
    iranker_no_face_predictions = get_face_predictions(iranker_predictions, no_faces)
    iranker_no_face_predictions = document_level_predictions(
        iranker_no_face_predictions, passage2image)

    test_iranker_no_face_predictions = get_face_predictions(test_iranker_predictions, test_no_faces)
    test_iranker_no_face_predictions = document_level_predictions(
        test_iranker_no_face_predictions, passage2image)
    
    w_index = metric_scores.argmax()
    w_norm = optim_results[w_index]["norm"]
    w_method = optim_results[w_index]["method"]
    w_best_params = optim_results[w_index]["best_params"]
    rankers = [aranker_face_only_predictions, iranker_face_only_predictions]
    w_combined_run = fuse_rankers(rankers, search_key=search_key, norm=w_norm, method=w_method, params=w_best_params)
    w_combined_face_only_predictions = run_to_predictions(w_combined_run, search_key=search_key)
    w_combined_face_only_predictions = predictions_add_gold(w_combined_face_only_predictions, 
                                                            iranker_face_only_predictions, 
                                                            search_key=search_key)
    w_combined_face_only_predictions = passage_level_predictions(w_combined_face_only_predictions,
                                                                 iranker_face_only_predictions, passage2image)

    
    test_rankers = [test_aranker_face_only_predictions, test_iranker_face_only_predictions]
    test_w_combined_run = fuse_rankers(test_rankers, search_key=search_key, 
                                       norm=w_norm, method=w_method, params=w_best_params)
    test_w_combined_face_only_predictions = run_to_predictions(test_w_combined_run, 
                                                              search_key=search_key)
    test_w_combined_face_only_predictions = predictions_add_gold(
        test_w_combined_face_only_predictions, 
        test_iranker_face_only_predictions, 
        search_key=search_key)
    test_w_combined_face_only_predictions = passage_level_predictions(test_w_combined_face_only_predictions,
                                                                 test_iranker_face_only_predictions, passage2image)

    w_combined_predictions = combine_document_level_image_rankers(
        w_combined_face_only_predictions,
        iranker_no_face_predictions, 
        faces)

    test_w_combined_predictions = combine_document_level_image_rankers(
        test_w_combined_face_only_predictions,
        test_iranker_no_face_predictions, 
        test_faces)
    
    print('DEV:',  compute_metrics(w_combined_predictions, search_key='document'))
    print('TEST:', compute_metrics(test_w_combined_predictions, search_key='document'))

    pickle_save('output/dev/unified_image_ranker.pkl', w_combined_predictions)
    pickle_save('output/test/unified_image_ranker.pkl', test_w_combined_predictions)
    
    

if __name__ == "__main__":
    main(sys.argv[1:])

