#!/bin/bash

#SBATCH .. args


module load pytorch

nvidia-smi

wandb agent project/Name/ID --count 100