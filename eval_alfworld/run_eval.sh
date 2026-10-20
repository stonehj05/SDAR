#!/bin/bash
# Evaluate a checkpoint on ALFWorld valid_seen + valid_unseen (every game once).
# Usage: bash eval_alfworld/run_eval.sh <model> [extra eval_alfworld.py args...]
#   <model> = a verl FSDP checkpoint (.../global_step_N or .../global_step_N/actor; merged to .../global_step_N/hf
#             on first use) or any HuggingFace model directory (e.g. the base Qwen3-1.7B snapshot).
# Results go to eval_alfworld/results/<experiment>_<step>/ unless --out_dir is passed.
set -euo pipefail
MODEL_IN=${1:?"usage: $0 <checkpoint-or-hf-dir> [extra args]"}
shift

source /gpfs/radev/scratch/cohan/hs2352/miniforge3/etc/profile.d/conda.sh
conda activate /gpfs/radev/scratch/cohan/hs2352/envs/sdar
cd /gpfs/radev/scratch/cohan/hs2352/SDAR

export HF_HOME=/gpfs/radev/scratch/cohan/hs2352
export ALFWORLD_DATA=/gpfs/radev/scratch/cohan/hs2352/data/alfworld
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID:-local}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Slurm on Misha exports ROCR_VISIBLE_DEVICES too, which confuses GPU device selection.
unset ROCR_VISIBLE_DEVICES

MODEL_IN=$(realpath "$MODEL_IN")
[ -d "$MODEL_IN/actor" ] && MODEL_IN=$MODEL_IN/actor
if ls "$MODEL_IN"/model_world_size_*_rank_0.pt >/dev/null 2>&1; then
    STEP_DIR=$(dirname "$MODEL_IN")
    HF_DIR=$STEP_DIR/hf
    if [ ! -f "$HF_DIR/config.json" ]; then
        echo "Merging FSDP checkpoint $MODEL_IN -> $HF_DIR"
        python scripts/model_merger.py merge --backend fsdp --local_dir "$MODEL_IN" --target_dir "$HF_DIR"
    fi
    NAME=$(basename "$(dirname "$STEP_DIR")")_$(basename "$STEP_DIR")
else
    HF_DIR=$MODEL_IN
    NAME=$(basename "$MODEL_IN")
fi

python eval_alfworld/eval_alfworld.py --model "$HF_DIR" --out_dir eval_alfworld/results/$NAME "$@"
