# saves the oscar dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

#%%
import os
import sentencepiece as  spm
from tqdm import tqdm
import numpy as np
from datasets import load_dataset , DatasetDict# huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 6

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
# dataset = load_dataset('oscar', 'unshuffled_deduplicated_ja', split='train')
# dataset = load_dataset('oscar', 'unshuffled_deduplicated_ja', split="train", num_proc=num_proc)
# dataset = load_dataset('oscar', 'unshuffled_deduplicated_ja', split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)], num_proc=num_proc)
# dataset = load_dataset('oscar', 'unshuffled_deduplicated_ja', split=["train[:1%]","train[1%:2%]"], num_proc=num_proc)

# split_dataset = dataset[0].train_test_split(test_size=0.05, seed=2357, shuffle=True)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)


split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
#DatasetDict({
#    train: Dataset({
#         features: ['id', 'text'],
#         num_rows: 39476690
#     })
#     val: Dataset({
#         features: ['id', 'text'],
#         num_rows: 19749
#     })
# })

enc = spm.SentencePieceProcessor(model_file='./charOscar4e6.model')
# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
def process(example):
    ids = enc.encode(example['text'])[1:]# # encode_ordinary ignores any special tokens
    ids.append(enc.eos_id()) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
