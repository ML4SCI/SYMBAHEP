#!/bin/bash

#SBATCH .... args

module load pytorch/X

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 main.py \
    --project_name "Transformer_EW_2-to-2" \
    --run_name "run_3layers_$SLURM_JOB_ID" \
    --model_name "transformer" \
    --root_dir "$SCRATCH/EW/2-to-2" \
    --data_dir "$SCRATCH/EW/data/EW_normal_2-to-2" \
    --device "cuda" \
    --epochs 30 \
    --training_batch_size 32 \
    --test_batch_size 32 \
    --valid_batch_size 32 \
    --num_workers 32 \
    --embedding_size 1024 \
    --hidden_dim 8192 \
    --nhead 16 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --warmup_ratio 0 \
    --dropout 0.1 \
    --weight_decay 0.001 \
    --src_max_len 512 \
    --tgt_max_len 730 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 9 \
    --resume_best True \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
