#!/bin/bash
#SBATCH --job-name=alfworld_opsd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/radev/scratch/cohan/hs2352/SDAR/slurm_full/logs/%x_%j.log

# flash-attn in the env is built for sm_80 only (A100; also runs on A40/L40S), so H100/H200 are excluded.
# Resubmitting this file resumes from the latest checkpoint under checkpoints/verl_agent_alfworld/.
# For wandb logging: export WANDB_API_KEY=... before sbatch (the env is inherited).
bash /gpfs/radev/scratch/cohan/hs2352/SDAR/slurm_full/run_alfworld_full.sh opsd "$@"
