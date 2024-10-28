#!/bin/bash

#SBATCH ... args


module load pytorch/2.1.0-cu12

nvidia-smi

wandb agent project/Name/ID --count 100