from typing import Dict, Tuple
import torch
import numpy as np
import faiss


# return features by key
def extract_features(dataloader, model) -> Dict[str, np.ndarray]:
    model.eval()

    feats = {}

    print("Extracting features...")
    for i, (x_batch, keys) in enumerate(dataloader):
        x_batch = x_batch.cuda()

        y_batch = model(x_batch)

        for j, key in enumerate(keys):
            feats[key] = y_batch[j].cpu().detach().numpy()

        if i % 50 == 0:
            print(f"[{i}/{len(dataloader)}]", flush=True)

    return feats


# find the top k most similar images in the database for each query image
# returns a dictionary of keys for each query image
def find_top_k(
    database_features: Dict[str, np.ndarray],
    query_features: Dict[str, np.ndarray],
    k: int = 25,
) -> Dict[str, np.ndarray]:
    # get a list of keys and a list of features in the same order
    database_keys = list(database_features.keys())
    database_features = np.array(list(database_features.values())).astype("float32")

    query_keys = list(query_features.keys())
    query_features = np.array(list(query_features.values())).astype("float32")

    index = faiss.IndexFlatL2(database_features.shape[1])
    index.add(database_features)

    _, indexes = index.search(query_features, k)

    # convert indexes to keys
    indexes = np.vectorize(lambda x: database_keys[x])(indexes)

    # convert to dictionary
    indexes = dict(zip(query_keys, indexes))

    return indexes
