#!/bin/bash

module load pytorch/2.1.0-cu12

nvidia-smi

python seq_acc.py \
    --project_name "Seq_acc" \
    --run_name "test" \
    --model_name "transformer" \
    --root_dir "$SCRATCH/QCD" \
    --data_dir "$SCRATCH/QCD/data/QCD_small_normal" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 128 \
    --test_batch_size 128 \
    --valid_batch_size 128 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 4096 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --warmup_ratio 0.03 \
    --dropout 0.1 \
    --src_max_len 896 \
    --tgt_max_len 896 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True \
    --test_shuffle False \
    --pin_memory True \
    --world_size 2 \
    --save_freq 10 \
    --seed 42 \
    --log_freq 20 \

