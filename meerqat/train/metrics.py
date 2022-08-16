"""Metrics to be used in trainer."""

def retrieval(eval_prediction, ignore_index=-100):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_prediction: EvalPrediction (dict-like)
        predictions: np.ndarray
            shape (dataset_size, N*M)
            This corresponds to the log-probability of the relevant passages per batch (N*M == batch size)
        label_ids: np.ndarray
            shape (dataset_size, )
            Label at the batch-level (each value should be included in [0, N-1] inclusive)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """
    print(f"eval_prediction.predictions.shape: {eval_prediction.predictions.shape}")
    print(f"               .label_ids.shape: {eval_prediction.label_ids.shape}")
    metrics = {}

    log_probs = eval_prediction.predictions
    dataset_size, N_times_M = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    mrr, ignored_predictions = 0, 0
    for ranking, label in zip(rankings, eval_prediction.label_ids):
        if label == ignore_index:
            ignored_predictions += 1
            continue
        # +1 to count from 1 instead of 0
        rank = (ranking == label).nonzero()[0].item() + 1
        mrr += 1/rank
    mrr /= (dataset_size-ignored_predictions)
    # print(f"dataset_size: {dataset_size}, ignored_predictions: {ignored_predictions}")
    metrics["MRR@N*M"] = mrr

    # argmax to get index of prediction (equivalent to `log_probs.argmax(axis=1)`)
    predictions = rankings[:, 0]
    # print(f"predictions[:100] {predictions.shape}:\n{predictions[:100]}")
    # print(f"eval_prediction.label_ids[:100] {eval_prediction.label_ids.shape}:\n{eval_prediction.label_ids[:100]}")
    # hits@1
    where = eval_prediction.label_ids != ignore_index
    metrics["hits@1"] = (predictions[where] == eval_prediction.label_ids[where]).mean()

    return metrics


def ranker(eval_prediction, ignore_index=-100):
    """
    Computes metric for ranker evaluation (at the batch-level)
    
    Parameters
    ----------
    eval_prediction: EvalPrediction (dict-like)
        predictions: np.ndarray
            shape (dataset_size, N*M)
            This corresponds to the log-probability of the relevant passages per batch (N*M == batch size)
        label_ids: np.ndarray
            shape (dataset_size, )
            Label at the batch-level (each value should be included in [0, N-1] inclusive)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """
    
    print(f"eval_prediction.predictions.shape: {eval_prediction.predictions.shape}")
    print(f"               .label_ids.shape: {len(eval_prediction.label_ids)}")
    
    log_probs = eval_prediction.predictions
    dataset_size, N_times_M = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    
    # metrics to be used during evaluation at [1, 5, 10]
    metric_names = ["mrr", "precision", "hit_rate"]
    unwanted     = ["mrr@1", "mrr@5", "mrr@20", "hit_rate", "hit_rate@1", "precision"]
    m_list = [metric for metric in metric_names]
    kappas = [1, 5, 20]
    
    for i in range(len(kappas)):
        m_list.extend([metric+'@'+str(kappas[i]) for metric in metric_names])
    
    m_list = [element for element in m_list if element not in unwanted]
    
    # use label_ids to create qrels and runs for ranx evaluation
    _, indices, relevants = eval_prediction.label_ids
    
    n_queries, n_passages = indices.shape
    qrels_dict = {}
    run_dict = {}
    
    for i in range(n_queries):
        q_str = "q_"+str(int(i))
        if relevants[0].max() == -1:
            qrels_dict[q_str] = {"DUMMY_RUN": 0}
        else:
            ok_inds = relevants[i][relevants[i] != -1]
            qrels_dict[q_str] = dict([('d_' + str(int(key)), 1) for key in ok_inds])
        
        run_dict[q_str] = dict([('d_' + str(int(indices[i][index])), 1) for index in rankings[i]])
        
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    metrics = evaluate(qrels, run, m_list)
    
    return metrics
