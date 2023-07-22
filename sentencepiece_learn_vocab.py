#%%
# saves the oscar dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 6

# num = 39676690
dataset = load_dataset('oscar', 'unshuffled_deduplicated_ja',  streaming=True)["train"]
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
iter_num =  int(4e6)
count = 0

with open("text.txt", "a", encoding="UTF-8") as f:
    for example in dataset:
        f.write(example["text"])
        count +=1
        if count %100000==0:
            print(f"count:{count}/{iter_num}")
        if count ==iter_num:
            break

import sentencepiece as spm
spm.SentencePieceTrainer.Train(
    input="text.txt",
    model_prefix='charOscar',
    vocab_size=50000,
    pad_id=0,                
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='<|endoftext|>',
    user_defined_symbols=['[MASK]'],
    model_type='char'
)
