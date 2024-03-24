#!/bin/bash

#SBATCH --job-name=cool
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=128G

#SBATCH --output cool_%j.txt
#SBATCH --error cool_%j.err


# gpu -> gpu (a40) or himem
# pre-training:
torchrun main_pretrain.py --warmup_epochs 40 --epochs 800 \
                          --blr 1.5e-4 --weight_decay 0.05 

