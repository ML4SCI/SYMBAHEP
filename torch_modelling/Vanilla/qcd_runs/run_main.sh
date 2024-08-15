#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --output="/pscratch/sd/r/ritesh11/logs/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/logs/slurm-%j.out"
#SBATCH --mail-user=ritesh.work.personal@gmail.com
#SBATCH --mail-type=ALL

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export MASTER_ADDR=localhost
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

module load pytorch/2.1.0-cu12

nvidia-smi

srun python main.py \
    --project_name "Transformer_QCD" \
    --run_name "run_$SLURM_JOB_ID" \
    --model_name "transformer" \
    --root_dir "$SCRATCH" \
    --data_dir "$SCRATCH/QCD_small.csv" \
    --device "cuda" \
    --epochs 100 \
    --training_batch_size 48 \
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
    --world_size 2 \
    --save_freq 10 \
    --seed 42 \
    --log_freq 20 \
    --debug False

