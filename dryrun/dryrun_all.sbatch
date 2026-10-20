#!/bin/bash
#SBATCH --job-name=sdar_dryrun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/radev/scratch/cohan/hs2352/SDAR/dryrun/logs/slurm_%j.log

cd /gpfs/radev/scratch/cohan/hs2352/SDAR
for m in ${METHODS:-grpo opsd grpo_opsd}; do
    ray stop --force >/dev/null 2>&1
    bash dryrun/run_alfworld_dryrun.sh $m > dryrun/logs/$m.log 2>&1
    echo "DRYRUN $m EXIT $?"
done
