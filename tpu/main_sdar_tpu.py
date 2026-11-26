"""SDAR / RA-OPSD ALFWorld training on Cloud TPU.

Runs SDAR's own training loop (verl.trainer.main_sdar's SDARTaskRunner.run -> SkillSDRayTrainer /
RAOPSDRayTrainer.fit, agent_system's ALFWorld envs, TrajectoryCollector, GRPO, reliability weights) unchanged; only
the verl worker group is replaced by sdar_tpu.worker_group.TPUActorRolloutRefWorkerGroup (vLLM-TPU engines + a
JAX/Tunix actor). The hyperparameters are those of slurm_full/run_alfworld_full.sh, including its 4-GPU dp layout
(trainer.n_gpus_per_node=4), which fixes padding, batch balancing and the mini/micro-batch split.

Usage (on the Ray head of a cluster started with TCOD's scripts/tpu/start_cluster.sh):
  python tpu/main_sdar_tpu.py {grpo_opsd|opsd|ra_entropy|ra_consistency|ra_disagreement} [key=value overrides...]
"""

import argparse
import os
import sys
import types

SDAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPU_DIR = os.path.join(SDAR, "tpu")
PYDEPS = os.path.join(TPU_DIR, "pydeps")
sys.path[:0] = [SDAR, TPU_DIR, PYDEPS]
# verl.trainer.main_ppo decorates main() with hydra.main; only its dataset helpers are used here.
sys.modules.setdefault("hydra", types.SimpleNamespace(main=lambda **kw: (lambda f: f)))

from omegaconf import OmegaConf, open_dict  # noqa: E402

HOME = os.path.expanduser("~")


def run_overrides(method: str, model: str, data_dir: str, exp: str, ckpt_dir: str, ra_alpha: float, ra_k: int):
    """slurm_full/run_alfworld_full.sh, verbatim except paths and the logger."""
    train_data_size, val_data_size, group_size = 16, 128, 8
    common = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/test.parquet",
        f"data.train_batch_size={train_data_size}",
        f"data.val_batch_size={val_data_size}",
        "data.max_prompt_length=2048",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        "+data.apply_chat_template_kwargs.enable_thinking=False",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.actor.optim.lr=1.0e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.4",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.use_invalid_action_penalty=True",
        "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1",
        "algorithm.use_kl_in_reward=False",
    ]
    if method == "grpo_opsd":
        algo = ["+algorithm.sdar.sdar_coef=0.01", "+algorithm.sdar.gate_beta=0.0",
                "+algorithm.sdar.skills_dir=skills/alfworld", "+algorithm.sdar.skill_all=false"]
    elif method == "opsd":
        algo = ["+actor_rollout_ref.actor.pg_loss_coef=0", "+algorithm.sdar.sdar_coef=0.01",
                "+algorithm.sdar.gate_beta=5.0", "+algorithm.sdar.skills_dir=skills/alfworld",
                "+algorithm.sdar.skill_all=false"]
    elif method.startswith("ra_"):
        # RA_BASE=grpo_opsd (the script's default): gate_beta=0
        algo = ["+algorithm.sdar.gate_beta=0.0", "+algorithm.sdar.sdar_coef=0.01",
                "+algorithm.sdar.skills_dir=skills/alfworld", "+algorithm.sdar.skill_all=false",
                f"+algorithm.ra_opsd.signal={method[3:]}", f"+algorithm.ra_opsd.alpha={ra_alpha}",
                f"+algorithm.ra_opsd.consistency_samples={ra_k}"]
    else:
        raise ValueError(f"unknown method {method}")
    tail = [
        "env.env_name=alfworld/AlfredTWEnv",
        "env.seed=0",
        "env.max_steps=50",
        "+env.alfworld.require_think=False",
        f"env.rollout.n={group_size}",
        "env.resources_per_worker.num_cpus=0.1",
        "trainer.critic_warmup=0",
        "trainer.logger=[console,tensorboard]",
        "trainer.project_name=verl_agent_alfworld",
        f"trainer.experiment_name={exp}",
        "trainer.n_gpus_per_node=4",
        "trainer.ray_wait_register_center_timeout=600",
        "trainer.nnodes=1",
        "trainer.save_freq=5",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.resume_mode=auto",
        "trainer.test_freq=5",
        "trainer.total_epochs=150",
        "trainer.val_before_train=True",
        f"trainer.default_local_dir={ckpt_dir}",
    ]
    return common + algo + tail


def build_config(overrides):
    base = OmegaConf.load(os.path.join(SDAR, "verl/trainer/config/ppo_trainer.yaml"))
    OmegaConf.set_struct(base, False)
    cli = OmegaConf.from_dotlist([o[1:] if o.startswith("+") else o for o in overrides])
    return OmegaConf.merge(base, cli)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("overrides", nargs="*", help="extra key=value config overrides (applied last)")
    ap.add_argument("--model", default="/dev/shm/models/Qwen3-1.7B")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--run-root", default="/dev/shm/tcod_sync/sdar",
                    help="checkpoints + weight-sync dirs (must be visible at the same path on every worker)")
    ap.add_argument("--ra-alpha", type=float, default=1.0)
    ap.add_argument("--ra-k", type=int, default=4)
    ap.add_argument("--student-engines", type=int, default=12)
    ap.add_argument("--engine-mem", type=float, default=0.7)
    ap.add_argument("--trainer-chips", type=int, default=4)
    ap.add_argument("--hf-keep-every", type=int, default=25, help="keep an HF export every N steps (0: never)")
    ap.add_argument("--compute-dtype", default="bfloat16")
    args = ap.parse_args()

    method = args.method
    if args.exp:
        exp = args.exp
    elif method.startswith("ra_"):
        exp = f"ra_{method[3:]}_grpo_opsd_qwen3_1.7b_alpha{args.ra_alpha}_tpu"
    else:
        exp = {"grpo_opsd": "grpo_opsd_qwen3_1.7b_coef0.01_beta0.0_skillallfalse_tpu",
               "opsd": "opsd_qwen3_1.7b_coef0.01_beta5.0_skillallfalse_tpu"}[method]
    run_dir = os.path.join(args.run_root, exp)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    sync_dir = os.path.join(run_dir, "sync")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sync_dir, exist_ok=True)
    data_dir = os.path.join(TPU_DIR, "data", "text")
    config = build_config(run_overrides(method, args.model, data_dir, exp, ckpt_dir, args.ra_alpha, args.ra_k)
                          + args.overrides)
    os.environ.setdefault("TENSORBOARD_DIR", os.path.join(run_dir, "tensorboard"))
    os.chdir(SDAR)  # skills_dir=skills/alfworld is relative to the repo, as on GPU

    import ray

    alf_data = os.environ.get("ALFWORLD_DATA", os.path.join(HOME, "alf-data"))
    env_vars = {"ALFWORLD_DATA": alf_data, "TMPDIR": "/dev/shm/tmp", "TOKENIZERS_PARALLELISM": "true",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    os.environ.update({k: v for k, v in env_vars.items() if k not in ("OMP_NUM_THREADS",)})
    ray.init(address="auto", namespace=f"sdar_{exp}", log_to_driver=True, runtime_env={
        "py_modules": [os.path.join(SDAR, "agent_system"), os.path.join(TPU_DIR, "sdar_tpu"),
                       os.path.join(PYDEPS, "gymnasium"), os.path.join(PYDEPS, "farama_notifications")],
        "env_vars": env_vars,
    })

    # ---------------------------------------------------------------- SDARTaskRunner.run (verl/trainer/main_sdar.py)
    from pprint import pprint

    OmegaConf.resolve(config)
    sdar_cfg = config.algorithm.get("sdar", {})
    with open_dict(config):
        config.actor_rollout_ref.actor.use_sdar_loss = True
        config.actor_rollout_ref.actor.sdar_loss_coef = sdar_cfg.get("sdar_coef", 0.1)
        config.actor_rollout_ref.actor.sdar_gate_beta = sdar_cfg.get("gate_beta", 5.0)
        use_ra_opsd = config.algorithm.get("ra_opsd", {}).get("signal", "none") != "none"
        config.actor_rollout_ref.actor.use_opsd_weights = use_ra_opsd
    pprint(OmegaConf.to_container(config, resolve=True))

    local_path = config.actor_rollout_ref.model.path
    from agent_system.environments import make_envs

    envs, val_envs = make_envs(config)

    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)
    processor = hf_processor(local_path, trust_remote_code=False, use_fast=True)

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {Role.ActorRollout: None, Role.Critic: None}
    mapping = {Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = None
        mapping[Role.RefPolicy] = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping=mapping)

    from agent_system.reward_manager import EpisodeRewardManager

    reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
    val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
    assert config.actor_rollout_ref.rollout.n == 1

    from agent_system.multi_turn_rollout import TrajectoryCollector

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn

    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    from verl.trainer.ppo.rlsd_utils import SkillProvider

    skill_provider = SkillProvider(skills_dir=sdar_cfg.get("skills_dir", "skills/alfworld"),
                                   skill_all=sdar_cfg.get("skill_all", False))
    print(f"[SDAR] Available skills: {list(skill_provider.skill_contents.keys())}")
    print(f"[SDAR] sdar_coef: {config.actor_rollout_ref.actor.sdar_loss_coef}  "
          f"gate_beta: {config.actor_rollout_ref.actor.sdar_gate_beta}")
    if use_ra_opsd:
        from verl.trainer.ppo.ra_opsd_ray_trainer import RAOPSDRayTrainer as trainer_cls

        print(f"[SDAR] RA-OPSD reliability weighting: {OmegaConf.to_container(config.algorithm.ra_opsd)}")
    else:
        from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer as trainer_cls

    # ---------------------------------------------------------------- TPU workers
    wg = make_worker_group(config, tokenizer, args, sync_dir)

    class TPUTrainer(trainer_cls):
        def init_workers(self):
            self.actor_rollout_wg = wg
            self.ref_policy_wg = wg
            self.async_rollout_mode = False

    trainer = TPUTrainer(
        config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager, ray_worker_group_cls=None, reward_fn=reward_fn,
        val_reward_fn=val_reward_fn, train_dataset=train_dataset, val_dataset=val_dataset, collate_fn=collate_fn,
        train_sampler=train_sampler, device_name="cpu", traj_collector=traj_collector, envs=envs,
        val_envs=val_envs, skill_provider=skill_provider)
    trainer.init_workers()
    try:
        trainer.fit()
        print("[SDAR-TPU] training finished")
    finally:
        # shut vLLM down from inside each engine so its EngineCore child releases the chip
        ray.get([e.shutdown.remote() for e in wg.engines])


def make_worker_group(config, tokenizer, args, sync_dir):
    import json

    import ray

    from sdar_tpu.engine import SDARvLLMEngine
    from sdar_tpu.tunix_worker import SDARTunixWorker
    from sdar_tpu.worker_group import TPUActorRolloutRefWorkerGroup
    from trinity.common.config import InferenceModelConfig
    from trinity.common.models.tpu_env import whole_host_runtime_env
    from trinity.trainer.tunix.worker import default_seq_buckets

    model = config.actor_rollout_ref.model.path
    rollout = config.actor_rollout_ref.rollout
    actor = config.actor_rollout_ref.actor
    max_prompt, max_resp = config.data.max_prompt_length, config.data.max_response_length
    with open(os.path.join(model, "generation_config.json")) as f:
        gen_cfg = json.load(f)
    eos = gen_cfg.get("eos_token_id", tokenizer.eos_token_id)
    pad = gen_cfg.get("pad_token_id", tokenizer.pad_token_id)

    cluster = ray.cluster_resources()
    labels = [lab for lab in ("explorer_tpu", "teacher_tpu") if lab in cluster]
    slots = [lab for lab in labels for _ in range(int(cluster[lab]))]
    if len(slots) < args.student_engines:
        raise RuntimeError(f"{args.student_engines} engines requested, {len(slots)} non-trainer chips labelled")
    engines = []
    for i in range(args.student_engines):
        mc = InferenceModelConfig(
            model_path=model, engine_type="vllm_tpu", tensor_parallel_size=1,
            gpu_memory_utilization=args.engine_mem, max_model_len=max_prompt + max_resp,
            max_prompt_tokens=max_prompt, max_response_tokens=max_resp, temperature=rollout.temperature,
            top_p=rollout.top_p, top_k=rollout.top_k, enable_prefix_caching=True, enable_chunked_prefill=True,
            min_response_tokens=0, logprobs=0, repetition_penalty=1.0, dtype="bfloat16", seed=i,
            enable_thinking=False)
        mc.ray_namespace = ray.get_runtime_context().namespace
        engines.append(ray.remote(SDARvLLMEngine).options(
            name=f"sdar_engine_{i}", num_cpus=0, resources={"TPU": 1, slots[i]: 1}).remote(config=mc))

    n = args.trainer_chips
    resources = {"TPU": n}
    if "trainer_tpu" in cluster:
        resources["trainer_tpu"] = n
    block = 512
    worker_cfg = {
        "model_path": model, "flash_block_size": block, "remat": "DECODER", "compute_dtype": args.compute_dtype,
        # prompt (student or skill-augmented teacher, both <= max_prompt_length) + max_response_length
        "seq_buckets": sorted(set(default_seq_buckets(max_prompt, max_resp, block) + [1536])),
        "max_response_tokens": max_resp, "temperature": rollout.temperature, "score_rows_per_chip": 4,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "optimizer": {"lr": actor.optim.lr, "betas": [0.9, 0.999], "weight_decay": actor.optim.weight_decay,
                      "clip_grad": actor.grad_clip},
        "loss": {"clip_ratio_low": actor.clip_ratio_low if actor.clip_ratio_low is not None else actor.clip_ratio,
                 "clip_ratio_high": actor.clip_ratio_high if actor.clip_ratio_high is not None else actor.clip_ratio,
                 "clip_ratio_c": actor.get("clip_ratio_c", 3.0), "pg_loss_coef": actor.get("pg_loss_coef", 1.0),
                 "entropy_coeff": actor.entropy_coeff, "use_kl_loss": actor.use_kl_loss,
                 "kl_loss_coef": actor.kl_loss_coef, "use_sdar_loss": actor.get("use_sdar_loss", False),
                 "sdar_loss_coef": actor.get("sdar_loss_coef", 0.1), "sdar_gate_beta": actor.get("sdar_gate_beta", 5.0),
                 "use_opsd_weights": actor.get("use_opsd_weights", False)},
    }
    assert actor.kl_loss_type == "low_var_kl" and actor.loss_agg_mode == "token-mean"
    assert actor.policy_loss.get("loss_mode", "vanilla") == "vanilla" and actor.ppo_epochs == 1
    assert actor.optim.warmup_style == "constant" and actor.optim.get("lr_warmup_steps", -1) <= 0 \
        and not actor.optim.lr_warmup_steps_ratio
    print(f"[SDAR-TPU] trainer config: {worker_cfg}")
    trainer_actor = ray.remote(SDARTunixWorker).options(
        name="sdar_tunix_worker", num_cpus=1, resources=resources,
        runtime_env=whole_host_runtime_env(n)).remote(worker_cfg)
    ray.get(trainer_actor.ready.remote())
    ray.get([e.prepare.remote() for e in engines])
    print(f"[SDAR-TPU] trainer ({n} chips) and {len(engines)} rollout engines ready")
    return TPUActorRolloutRefWorkerGroup(config, tokenizer, engines, trainer_actor, sync_dir,
                                         hf_keep_every=args.hf_keep_every, eos_token_id=eos, pad_token_id=pad)


if __name__ == "__main__":
    main()
