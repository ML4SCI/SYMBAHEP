#!/bin/bash


module load pytorch/2.1.0-cu12

torchrun main.py \
    --project_name "Transformer_QCD" \
    --run_name "run_1" \
    --model_name "transformer" \
    --root_dir "$SCRATCH" \
    --data_dir "$SCRATCH/QCD_small.csv" \
    --device "cuda" \
    --epochs 100 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 4096 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --warmup_ratio 0.1 \
    --dropout 0.1 \
    --src_max_len 896 \
    --tgt_max_len 896 \
    --curr_epoch 0 \
    --optimizer_lr 1e-4 \
    --use_half_precision False \
    --train_shuffle True \
    --test_shuffle False \
    --pin_memory True \
    --world_size 1 \
    --save_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --debug True

