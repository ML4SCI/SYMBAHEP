#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 09:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --output="/pscratch/sd/r/ritesh11/ew_logs/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/ew_logs/slurm-%j.out"
#SBATCH --mail-user=ritesh.slurm@gmail.com
#SBATCH --mail-type=ALL


module load pytorch/2.1.0-cu12

nvidia-smi

srun torchrun --standalone --nproc_per_node 2 main.py \
    --project_name "RSineKANformer_EW" \
    --run_name "run_normal_3layers_$SLURM_JOB_ID" \
    --model_name "rsinekanformer" \
    --root_dir "$SCRATCH/EW/2-to-2/RSineKAN2" \
    --data_dir "$SCRATCH/EW/data/EW_normal_2-to-2" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 512 \
    --ff_dims 8192,512 \
    --d_ff 4096 \
    --nhead 8 \
    --num_layers 3 \
    --warmup_ratio 0 \
    --weight_decay 1e-3 \
    --dropout 0.1 \
    --src_max_len 512 \
    --tgt_max_len 512 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --update_lr 1e-5 \
    --resume_best True \
    --is_constant_lr \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 10 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
