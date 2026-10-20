#!/bin/bash
#SBATCH --job-name=alfworld_eval
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/radev/scratch/cohan/hs2352/SDAR/eval_alfworld/logs/%x_%j.log

# Usage: sbatch eval_alfworld/eval.sbatch <checkpoint-or-hf-dir> [extra eval_alfworld.py args...]
# gpu_devel caps a user at 4 CPUs / 32G / 2 GPUs; 16 parallel games keep host memory well under 32G.
bash /gpfs/radev/scratch/cohan/hs2352/SDAR/eval_alfworld/run_eval.sh "$@"
