#!/bin/bash

#SBATCH ... args


module load pytorch/X

nvidia-smi

srun torchrun --standalone --nproc_per_node 2 main.py \
    --project_name "RSineKANformer_QED_2-to-2" \
    --run_name "run_normal_3layers_$SLURM_JOB_ID" \
    --model_name "rsinekanformer_32x2_new" \
    --root_dir "$SCRATCH/QED/RSineKAN/2-to-2" \
    --data_dir "$SCRATCH/QED/data/QED_normal_2-to-2" \
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
