#!/bin/bash

#SBATCH .....args

module load pytorch/X

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 seq_acc_qed.py \
    --project_name "QED_kan-32x2_1024" \
    --run_name "run_prefix_$SLURM_JOB_ID" \
    --model_name "rsinekanformer_32x2_1024" \
    --root_dir "$SCRATCH/QED/2-to-3" \
    --data_dir "$SCRATCH/QED/data/QED_normal_2-to-3" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 1024 \
    --ff_dims 8192 \
    --d_ff 4096 \
    --nhead 16 \
    --num_layers 3 \
    --warmup_ratio 0 \
    --clip_grad_norm 1 \
    --weight_decay 1e-3 \
    --dropout 0.1 \
    --src_max_len 602 \
    --tgt_max_len 1202 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 9 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
