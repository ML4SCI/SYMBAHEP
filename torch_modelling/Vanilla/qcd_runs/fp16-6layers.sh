#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --output="/pscratch/sd/r/ritesh11/qcd_logs/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/qcd_logs/slurm-%j.out"
#SBATCH --mail-user=ritesh.slurm@gmail.com
#SBATCH --mail-type=ALL


module load pytorch/2.1.0-cu12

nvidia-smi

srun torchrun --standalone --nproc_per_node 2 main.py \
    --project_name "Transformer_QCD" \
    --run_name "run_aug_$SLURM_JOB_ID" \
    --model_name "transformer" \
    --root_dir "$SCRATCH/QCD/fp16" \
    --data_dir "$SCRATCH/QCD/data/QCD_aug.csv" \
    --device "cuda" \
    --epochs 100 \
    --training_batch_size 128 \
    --test_batch_size 128 \
    --valid_batch_size 128 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 4096 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --warmup_ratio 0.03 \
    --dropout 0.1 \
    --src_max_len 896 \
    --tgt_max_len 896 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --use_half_precision \
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 10 \
    --test_freq 5 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
