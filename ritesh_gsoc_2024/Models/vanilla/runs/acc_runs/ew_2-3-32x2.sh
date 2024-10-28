#!/bin/bash

#SBATCH .. args


module load pytorch/X

nvidia-smi

srun torchrun --standalone --nproc_per_node 4 seq_acc.py \
    --project_name "Transformer_EW_2-to-3_32x2" \
    --run_name "final_small_3layers_$SLURM_JOB_ID" \
    --model_name "transformer_SBS_HLR_small" \
    --root_dir "$SCRATCH/EW/2-to-3/trained_models" \
    --data_dir "$SCRATCH/EW/data/EW_small_2-to-3" \
    --device "cuda" \
    --epochs 50 \
    --training_batch_size 64 \
    --test_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 512 \
    --hidden_dim 8192 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --warmup_ratio 0 \
    --dropout 0.1 \
    --weight_decay 1e-3 \
    --src_max_len 302 \
    --tgt_max_len 302 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5 \
    --train_shuffle True\
    --pin_memory True \
    --world_size 2 \
    --save_freq 9 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
