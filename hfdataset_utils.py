# cacheの逐次削除 cite : https://discuss.huggingface.co/t/keeping-only-current-dataset-state-in-cache/6740

import os
def delete_dataset(dataset):
    cached_files = [cache_file["filename"] for cache_file in dataset.cache_files]
    del dataset
    for cached_file in cached_files:
        os.remove(cached_file)

# This line creates a new dataset using map, and deletes the old dataset
dataset, _ = dataset.map(...), delete_dataset(dataset)