#!/bin/bash

#SBATCH .....args

module load pytorch

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 seq_acc.py \
    --project_name "Dummy_Transformer_Project" \
    --run_name "dummy_run" \
    --model_name "dummy_transformer" \
    --root_dir "transformer_checkpoints" \
    --data_dir "transformer_data" \
    --device "cuda" \
    --epochs 30 \
    --training_batch_size 32 \
    --test_batch_size 32 \
    --valid_batch_size 32 \
    --num_workers 32 \
    --embedding_size 512 \
    --ff_dims 8192 \
    --d_ff 4096 \
    --nhead 8 \
    --num_layers 3 \
    --warmup_ratio 0 \
    --clip_grad_norm 1 \
    --weight_decay 1e-3 \
    --dropout 0.1 \
    --src_max_len 300 \
    --tgt_max_len 300 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 10 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
