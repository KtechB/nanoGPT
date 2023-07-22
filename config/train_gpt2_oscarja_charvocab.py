# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_oscarja_charvocab.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

dataset = 'oscar_deduplicated_ja_slice'
# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 10#1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
