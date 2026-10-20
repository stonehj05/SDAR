#!/bin/bash
# Small-scale dry run of the ALFWorld baselines on a 2xA40 node.
# Usage: bash dryrun/run_alfworld_dryrun.sh {grpo|opsd|grpo_opsd} [extra hydra overrides...]
# Algorithm flags mirror examples/{grpo_trainer/run_alfworld.sh, opsd_trainer/run_alfworld_3b.sh,
# grpo_opsd_trainer/run_alfworld_3b.sh} with the Qwen3 variants' model + enable_thinking=False;
# only sizes/resources are shrunk.
set -x
METHOD=${1:?"usage: $0 grpo|opsd|grpo_opsd"}
shift

source /gpfs/radev/scratch/cohan/hs2352/miniforge3/etc/profile.d/conda.sh
conda activate /gpfs/radev/scratch/cohan/hs2352/envs/sdar
cd /gpfs/radev/scratch/cohan/hs2352/SDAR

export HF_HOME=/gpfs/radev/scratch/cohan/hs2352
export ALFWORLD_DATA=/gpfs/radev/scratch/cohan/hs2352/data/alfworld
export RAY_TMPDIR=/tmp/ray_$USER
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export VLLM_USE_V1=1
# Slurm on Misha exports ROCR_VISIBLE_DEVICES too; verl refuses to start if it is set alongside CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
# Don't back up vLLM weights to host RAM when sleeping (saves ~4G/GPU under the 32G cap).
export VERL_VLLM_SLEEP_LEVEL=2

MODEL=${MODEL:-$HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}
N_GPUS=${N_GPUS:-2}
TRAIN_STEPS=${TRAIN_STEPS:-2}
# Each env is its own Ray process; keep the count low to fit the 32G gpu_devel memory cap.
train_data_size=${TRAIN_DATA_SIZE:-2}
val_data_size=${VAL_DATA_SIZE:-2}
group_size=${GROUP_SIZE:-2}
DATA_DIR=/gpfs/radev/scratch/cohan/hs2352/data/verl-agent/text

case $METHOD in
  grpo)
    ENTRY=verl.trainer.main_ppo
    ALGO_ARGS=()
    ;;
  opsd)  # SDAR without GRPO loss
    ENTRY=verl.trainer.main_sdar
    ALGO_ARGS=(+actor_rollout_ref.actor.pg_loss_coef=0
      +algorithm.sdar.sdar_coef=0.01 +algorithm.sdar.gate_beta=5.0
      +algorithm.sdar.skills_dir=skills/alfworld +algorithm.sdar.skill_all=false)
    ;;
  grpo_opsd)  # SDAR without gate (gate_beta=0)
    ENTRY=verl.trainer.main_sdar
    ALGO_ARGS=(+algorithm.sdar.sdar_coef=0.01 +algorithm.sdar.gate_beta=0.0
      +algorithm.sdar.skills_dir=skills/alfworld +algorithm.sdar.skill_all=false)
    ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

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
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_data_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO:-1} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO:-1} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO:-1} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    "${ALGO_ARGS[@]}" \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=10 \
    +env.alfworld.require_think=False \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=dryrun_$METHOD \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TRAIN_STEPS \
    trainer.val_before_train=False "$@"
