# Fusing Rerankers
The predictions of every reranker are stored in the following format: list of dict (query_dict) where the query_dict contains
```
(dict_keys(['indices', 'scores', 'ranks', 'ranked_indices', 'gold_indices']),
 dict_keys(['indices', 'scores', 'ranks', 'ranked_indices', 'gold_indices', 'images', 'image_scores', 'gold_images']))
 
 
query_dict = {}
query_dict["indices"] = the indices of the passages in the kb
query_dict["scores"]  = the passage scores given by the reranker
query_dict["ranks"]  = the passage ranks something like (-query_dict["scores"]).argsort()
query_dict["gold_indices"] = the indices of relevant passages

In case of image rerankers these additional keys are available, but not necessary. In utils.py, 
there are functions that insert them when needed.
query_dict["images"]       = the names of the images associated with the passages
query_dict["image_scores"] = the image scores of the passages
query_dict["gold_images"]  = the image names of relevant passages
```

To fuse image rerankers, run:
```sh
python fuse_image_rerankers.py
```

Otherwise, to fuse text reranker with RRT+ArcFace reranker, run:
```sh
python fuse_image_rerankers.py
```