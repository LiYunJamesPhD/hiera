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
torchrun main_finetune.py --finetune checkpoint/mae_hiera_base_224.pth \
                          --epochs 150 --blr 5e-4 --layer_decay 0.65 \
                          --weight_decay 0.05 --drop_path 0.1 \
                          --dist_eval --output_dir out_dir_finetune --log_dir out_dir_finetune


