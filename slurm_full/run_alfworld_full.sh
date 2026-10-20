#!/bin/bash
# Full ALFWorld training for the SDAR baselines, launched by the *.sbatch files in this directory.
# Usage: bash slurm_full/run_alfworld_full.sh {grpo|opsd|grpo_opsd} [extra hydra overrides...]
# Hyperparameters follow examples/{grpo_trainer/run_alfworld_qwen3_1b.sh, opsd_trainer/run_alfworld_qwen3.sh,
# grpo_opsd_trainer/run_alfworld_qwen3.sh}. Differences: 4 GPUs (Misha nodes have 4) instead of 8,
# periodic checkpoints so a job hitting the 2-day limit can be resubmitted and auto-resume, and
# env.alfworld.require_think=False (Qwen3 with enable_thinking=False never emits <think>, so every action
# would otherwise count as invalid).
set -x
METHOD=${1:?"usage: $0 grpo|opsd|grpo_opsd"}
shift

source /gpfs/radev/scratch/cohan/hs2352/miniforge3/etc/profile.d/conda.sh
conda activate /gpfs/radev/scratch/cohan/hs2352/envs/sdar
cd /gpfs/radev/scratch/cohan/hs2352/SDAR

export HF_HOME=/gpfs/radev/scratch/cohan/hs2352
export ALFWORLD_DATA=/gpfs/radev/scratch/cohan/hs2352/data/alfworld
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID:-local}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
# Unbuffered stdout so per-step metrics show up in the Slurm log as they happen.
export PYTHONUNBUFFERED=1
# Slurm on Misha exports ROCR_VISIBLE_DEVICES too; verl refuses to start if it is set alongside CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
# Smaller micro batches than upstream (32): upstream sizes OOM on 44G L40S/A40 in update_actor.
# (expandable_segments would help too, but vLLM sleep mode asserts it is off.)

MODEL=${MODEL:-$HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
N_GPUS=${N_GPUS:-4}
NUM_CPUS=${SLURM_CPUS_PER_TASK:-32}
train_data_size=16
val_data_size=128
group_size=8
DATA_DIR=/gpfs/radev/scratch/cohan/hs2352/data/verl-agent/text

if [ -n "$WANDB_API_KEY" ]; then LOGGER="['console','wandb']"; else LOGGER="['console']"; fi

case $METHOD in
  grpo)
    ENTRY=verl.trainer.main_ppo
    EXP=grpo_qwen3_1.7b
    ALGO_ARGS=()
    ;;
  opsd)  # SDAR without GRPO loss
    ENTRY=verl.trainer.main_sdar
    EXP=opsd_qwen3_1.7b_coef0.01_beta5.0_skillallfalse
    ALGO_ARGS=(+actor_rollout_ref.actor.pg_loss_coef=0
      +algorithm.sdar.sdar_coef=0.01 +algorithm.sdar.gate_beta=5.0
      +algorithm.sdar.skills_dir=skills/alfworld +algorithm.sdar.skill_all=false)
    ;;
  grpo_opsd)  # SDAR without gate (gate_beta=0)
    ENTRY=verl.trainer.main_sdar
    EXP=grpo_opsd_qwen3_1.7b_coef0.01_beta0.0_skillallfalse
    ALGO_ARGS=(+algorithm.sdar.sdar_coef=0.01 +algorithm.sdar.gate_beta=0.0
      +algorithm.sdar.skills_dir=skills/alfworld +algorithm.sdar.skill_all=false)
    ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir /gpfs/radev/scratch/cohan/hs2352/data/verl-agent

python3 -m $ENTRY \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    "${ALGO_ARGS[@]}" \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    +env.alfworld.require_think=False \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    +ray_init.num_cpus=$NUM_CPUS \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$EXP \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "$@"
