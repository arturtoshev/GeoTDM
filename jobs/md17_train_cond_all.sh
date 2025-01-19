#!/bin/bash
#SBATCH --job-name=md17_train_cond_all
#SBATCH --output=outputs/md17_train_cond_all_%j.log
#SBATCH --error=outputs/md17_train_cond_all_%j.err
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

source venv/bin/activate

# Check if the port is free, if not, find a free one
PORT=16888
while lsof -i:$PORT >/dev/null 2>&1; do
    PORT=$((PORT+1))
done

torchrun \
    --nproc_per_node=2 \
    --master_port=$PORT \
    experiments/md17_train.py \
    --train_yaml_file=configs/md17_train_cond_all.yaml