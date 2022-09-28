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


    tranker_predictions = pickle_load('rankers/dev/text_ranker_predictions.pkl')
    tranker_predictions = format_text_ranker(tranker_predictions)
    ir_predictions = get_ir_ranking(tranker_predictions)


    test_tranker_predictions = pickle_load('rankers/test/text_ranker_predictions.pkl')
    test_tranker_predictions = format_text_ranker(test_tranker_predictions)
    test_ir_predictions = get_ir_ranking(tranker_predictions)
    
    
    uranker_predictions = pickle_load('output/dev/unified_image_ranker.pkl')
    test_uranker_predictions = pickle_load('output/test/unified_image_ranker.pkl')

    optim_norms = [#"zmuv", "max", "borda", 
                   "sum", "rank", "min-max"]

    optim_methods = [#"rbc","wmnz", "bayesfuse", "slidefuse", "probfuse", "rrf", "w_bordafuse", #"w_condorcet",
               "wsum", "posfuse", "mapfuse"]#, "segfuse", "logn_isr", "mixed"]

    rankers = [tranker_predictions, uranker_predictions]
    search_key = "passage"
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
    metric_scores.max(), metric_scores[(-metric_scores).argsort()[:5]], metric_scores.argmax(), optim_results[metric_scores.argmax()]
    
    w_index = metric_scores.argmax()
    w_norm = optim_results[w_index]["norm"]
    w_method = optim_results[w_index]["method"]
    w_best_params = optim_results[w_index]["best_params"]
    rankers = [tranker_predictions, uranker_predictions]
    w_combined_run = fuse_rankers(rankers, search_key=search_key, norm=w_norm, method=w_method, params=w_best_params)
    w_combined_predictions = run_to_predictions(w_combined_run, search_key=search_key)
    w_combined_predictions = predictions_add_gold(w_combined_predictions, 
                                                  tranker_predictions, 
                                                  search_key=search_key)

    print(compute_metrics(w_combined_predictions, search_key=search_key))
    
    
    test_rankers = [test_tranker_predictions, test_uranker_predictions]
    test_w_combined_run = fuse_rankers(test_rankers, search_key=search_key, norm=w_norm, method=w_method, params=w_best_params)
    test_w_combined_predictions = run_to_predictions(test_w_combined_run, search_key=search_key)
    test_w_combined_predictions = predictions_add_gold(test_w_combined_predictions, 
                                                  test_tranker_predictions, 
                                                  search_key=search_key)
    
    print('DEV:', compute_metrics(w_combined_predictions, search_key='passage'))
    print('TEST:', compute_metrics(test_w_combined_predictions, search_key='passage'))
    
    pickle_save('output/dev/unified_text_and_image_ranker.pkl', w_combined_predictions)
    pickle_save('output/test/unified_text_and_image_ranker', w_combined_predictions)

    

if __name__ == "__main__":
    main(sys.argv[1:])


