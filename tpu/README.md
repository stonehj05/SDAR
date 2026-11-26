# SDAR / RA-OPSD on Cloud TPU

This folder runs SDAR's ALFWorld training (GRPO+OPSD and the three reliability-aware OPSD variants) and the
`eval_alfworld` protocol on a Cloud TPU v4 slice instead of GPUs.

SDAR's own training loop is used unchanged: `SkillSDRayTrainer` / `RAOPSDRayTrainer.fit`, the `agent_system`
ALFWorld environments, `TrajectoryCollector`, GRPO and the reliability weights. Only verl's GPU worker group is
replaced:

| Piece | GPU (`slurm_full/`) | TPU (`tpu/`) |
|---|---|---|
| Rollout | verl vLLM rollout, TP 2 | 12 vLLM-TPU engines, 1 chip each (`sdar_tpu/engine.py`) |
| Actor + ref | verl FSDP actor | JAX/Tunix actor on 4 chips, fp32 master weights, bf16 compute, frozen ref copy (`sdar_tpu/tunix_worker.py`) |
| Worker group | `RayWorkerGroup` | `TPUActorRolloutRefWorkerGroup` (`sdar_tpu/worker_group.py`) |
| Weight sync | NCCL | the actor writes an HF checkpoint to a shared dir, engines reload it in place |

The virtual world size stays 4 (the GPU job's `n_gpus_per_node=4`), so padding, batch balancing and the
mini/micro-batch split match the GPU run. Hyperparameters are those of `slurm_full/run_alfworld_full.sh`
(see `run_overrides` in `main_sdar_tpu.py`).

The TPU backends (vLLM-TPU engine wrapper, Tunix log-prob / HF export / resume code, cluster scripts) come from
the `tpu-port` branch of [TCOD](https://github.com/stonehj05/TCOD), which provides the Python environment.

## What it was run on

- One **TPU v4-32** slice: 4 VMs ("workers"), 4 chips each, 30.75 GiB HBM per chip, 400 GB RAM and a 97 GB boot
  disk per VM.
  - Worker 0: the driver and the 4-chip actor.
  - Workers 1-3: the 12 rollout engines.
- Qwen3-1.7B, 150 steps per method.
  - A step takes about 8-10 min, so a run is about 22 h.
  - The full seen + unseen eval of one checkpoint takes about 5 min, engine start-up included.
- Results (`eval_alfworld/results/*_tpu*`, success rate seen / unseen):

  | Model | Seen | Unseen |
  |---|---|---|
  | Qwen3-1.7B base (TPU) | 10.0 | 10.4 |
  | `ra_entropy` step 150 | 35.7 | 34.3 |
  | `ra_consistency` step 150 | 42.1 | 42.5 |
  | `ra_disagreement` step 150 | 39.3 | 41.8 |

Other TPU generations and chip counts have not been tested. See "Other slice sizes" below.

## 1. Set up the slice (once)

Every command runs on **worker 0** unless a step says otherwise. `<your-tpu>` and `<zone>` are your slice's name and
zone.

### 1.1 SSH between workers

`gcloud` pushes your key to every worker the first time; after that the scripts use plain ssh with
`~/.ssh/google_compute_engine`.

```bash
gcloud compute tpus tpu-vm ssh <your-tpu> --zone <zone> --worker=all --command true
```

### 1.2 TCOD environment on every worker (~20-30 min)

This creates `~/venv-vllm-tpu` with the exact tested package set (vllm-tpu 0.29, jax 0.11, google-tunix 0.1.7,
ray 2.58, Python 3.12.14). It also applies the vLLM-TPU v4 attention-kernel patch, downloads the ALFWorld games
to `~/alf-data`, and installs Trinity (the TCOD repo) into the venv.

```bash
gcloud compute tpus tpu-vm ssh <your-tpu> --zone <zone> --worker=all --command '
  [ -d ~/TCOD ] || git clone -b tpu-port https://github.com/stonehj05/TCOD ~/TCOD
  MODELS=Qwen/Qwen3-1.7B bash ~/TCOD/scripts/tpu/setup_node.sh'
```

`setup_node.sh` installs the packages with `--no-deps` on purpose: a normal `pip install -r` fails to resolve
(`tpu-inference` pins `qwix==0.1.2`, `google-tunix` needs `0.1.8`). See `~/TCOD/scripts/tpu/README.md` for
details and troubleshooting.

### 1.3 Qwen3-1.7B on tmpfs, on every worker

The engines and the actor load the model from `/dev/shm/models/Qwen3-1.7B` (the `--model` default), so it must
exist at that path on all four workers. `/dev/shm` is wiped on reboot; redo this step after one.

```bash
gcloud compute tpus tpu-vm ssh <your-tpu> --zone <zone> --worker=all --command '
  ~/venv-vllm-tpu/bin/hf download Qwen/Qwen3-1.7B --local-dir /dev/shm/models/Qwen3-1.7B'
```

### 1.4 SDAR on worker 0

The SDAR repo is only needed on worker 0. Ray ships `agent_system/`, `tpu/sdar_tpu/` and the extra packages to the
other workers.

```bash
git clone https://github.com/stonehj05/SDAR ~/SDAR
~/.local/bin/uv pip install --python ~/venv-vllm-tpu/bin/python --no-deps --target ~/SDAR/tpu/pydeps \
  gymnasium==1.1.1 codetiming==1.4.0 torchdata==0.11.0 farama-notifications==0.0.6
```

`tpu/pydeps` holds pure-Python packages that SDAR needs but the TCOD environment does not have. They are kept
outside the venv so the TCOD environment stays exactly as pinned.

### 1.5 Check the install (CPU only, ~1 min)

This checks that the JAX loss equals the loss SDAR's torch `dp_actor` builds (PPO clip, entropy, low-var KL, SDAR
OPSD loss with and without RA-OPSD weights), in value and gradient:

```bash
cd ~/SDAR && JAX_PLATFORMS=cpu PYTHONPATH=$HOME/SDAR:$HOME/SDAR/tpu:$HOME/SDAR/tpu/pydeps ~/venv-vllm-tpu/bin/python tpu/tests/test_loss_parity.py
```

It should print four `grads equal` lines and `OK`.

## 2. Start the cluster (after every slice restart)

```bash
bash ~/TCOD/scripts/tpu/start_cluster.sh
```

This script does three things:

- Starts Ray, with worker 0 as head and the `trainer_tpu` label, and workers 1-3 labelled `explorer_tpu`. The
  engines may also use `teacher_tpu` workers, so `WORKER_ROLES="trainer teacher teacher explorer"` works too.
- NFS-exports `/dev/shm/tcod_sync` from worker 0 and mounts it at the same path on the other workers. Runs live
  under `/dev/shm/tcod_sync/sdar/`, because every worker must see the weight-sync dir.
- Creates `/dev/shm/tmp` and `/dev/shm/ray_tmp`.

Check that Ray sees all 16 chips:

```bash
~/venv-vllm-tpu/bin/ray status
```

## 3. Train

```bash
cd ~/SDAR && PYTHONUNBUFFERED=1 nohup ~/venv-vllm-tpu/bin/python tpu/main_sdar_tpu.py ra_consistency > /dev/shm/tcod_sync/sdar/ra_consistency.console.log 2>&1 &
```

**Methods:** `ra_entropy`, `ra_consistency`, `ra_disagreement` (RA-OPSD on top of GRPO+OPSD with `gate_beta=0`,
the `slurm_full/alfworld_ra_*.sbatch` jobs), `grpo_opsd` and `opsd`.

**Options:**

- `--ra-alpha` (default 1.0) and `--ra-k` (default 4).
- `--model` (default `/dev/shm/models/Qwen3-1.7B`).
- `--exp <name>` to rename the run.
- `--hf-keep-every N` (default 25): keep an HF export every N steps.
- Any SDAR config key as `key=value` after the method, e.g. `trainer.total_training_steps=150`.

**Output:** `/dev/shm/tcod_sync/sdar/<exp>/`, e.g. `ra_consistency_grpo_opsd_qwen3_1.7b_alpha1.0_tpu/`.

- `ckpt/global_step_N/actor/huggingface`: HF export every `--hf-keep-every` steps.
- `ckpt/global_step_N/actor/tunix_state`: resume state, 20 GB, kept for the latest checkpoint only.
- `tensorboard/`.
- `sync/`: per-step weights for the engines, deleted as it goes.

**Progress** (step, validation success, seconds per step):

```bash
grep -aE "^step:" /dev/shm/tcod_sync/sdar/ra_consistency.console.log | grep -oE "^step:[0-9]+|val/success_rate:[0-9.]+|timing_s/step:[0-9.]+" | tr '\n' ' ' | sed 's/ step:/\nstep:/g' | tail -5
```

The run prints `[SDAR-TPU] training finished` at the end.

**Stop:** kill the driver by PID; `pgrep -f main_sdar_tpu.py` shows it. Don't use `pkill -f` with a pattern that
also appears in your own shell command.

**Resume:** run the same command again. It loads the latest `ckpt/global_step_N` (`trainer.resume_mode=auto`),
including the optimizer state.

**Smoke test** (4 steps, about 30 min including compilation), for a new machine:

```bash
cd ~/SDAR && ~/venv-vllm-tpu/bin/python -c "import pandas as pd; pd.read_parquet('tpu/data/text/test.parquet').head(8).to_parquet('/dev/shm/sdar_smoke_test8.parquet')"
cd ~/SDAR && ~/venv-vllm-tpu/bin/python tpu/main_sdar_tpu.py ra_consistency --exp smoke_ra_consistency data.train_batch_size=4 data.val_batch_size=8 data.val_files=/dev/shm/sdar_smoke_test8.parquet env.max_steps=4 trainer.total_training_steps=4 trainer.test_freq=2 trainer.save_freq=2 --hf-keep-every 2
```

It should end with `[SDAR-TPU] training finished`. A second run of the same command resumes from step 4 and
exits at once, which checks loading. Delete `/dev/shm/tcod_sync/sdar/smoke_ra_consistency` afterwards.

## 4. Evaluate

`eval_alfworld_tpu.py` is `eval_alfworld/eval_alfworld.py` on vLLM-TPU, with the same protocol: every
valid_seen / valid_unseen game once, the same prompts, history 2, temperature 0.4, 512 tokens and 50 steps. It
spreads the games over 12 single-chip engines.

It needs the 12 non-trainer chips, so run it while no training is running. Several models can be evaluated in one
call; the engines reload weights between them.

```bash
cd ~/SDAR && PYTHONUNBUFFERED=1 ~/venv-vllm-tpu/bin/python tpu/eval_alfworld_tpu.py \
  --models /dev/shm/tcod_sync/sdar/ra_consistency_grpo_opsd_qwen3_1.7b_alpha1.0_tpu/ckpt/global_step_150/actor/huggingface \
  --names ra_consistency_grpo_opsd_qwen3_1.7b_alpha1.0_tpu_global_step_150
```

Results go to `eval_alfworld/results/<name>/{seen,unseen}/{summary.json,results.jsonl}` and `config.json`.

## Other slice sizes

The driver asks for `--trainer-chips` (default 4) chips with the `trainer_tpu` label and `--student-engines`
(default 12) single-chip engines on `explorer_tpu` / `teacher_tpu` chips. The eval takes the same idea with
`--engines`. On a different slice, change those numbers; the training batch layout (world size 4) does not
depend on them.

The actor uses every chip of one host (`whole_host_runtime_env`). A trainer spanning several hosts, or a host
with a chip count other than 4, has not been tried. On a single VM, `start_cluster.sh` skips NFS; give the actor
and the engines chips that add up to the host's count.

## Known issues

- **Worker 0's `/dev/shm` is RAM.** HF exports (3.8 GB each), the 20 GB resume state and the sync dir all live
  there, and Ray kills workers once the node passes 95% memory.
  - Keep `df -h /dev/shm` on worker 0 under about 60% before a long run.
  - Move or delete old `global_step_*/actor/huggingface` dirs you do not need.
  - A crashed save can leave `tunix_state.tmp` or stale `sync/` dirs. The next start and the next save clean
    them up.
- **Boot disks fill from system logs.** On these VMs Google's `tpu_agents` health container can flood
  `/var/log/syslog` and `/var/log/kern.log` with OOM reports, several GB per day.
  - When a worker's root disk is full, vLLM engines can hang at start-up.
  - Check every worker before a run:

    ```bash
    for ip in $(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints | tr ',' '\n' | cut -d: -f3); do ssh -i ~/.ssh/google_compute_engine $ip 'hostname; df -h / | tail -1'; done
    ```

  - Fix it on the affected worker:

    ```bash
    sudo truncate -s 0 /var/log/syslog /var/log/kern.log
    ```

  - `sudo docker restart <tpu_agents container id>` stops the flood.
- **Harmless log noise.** `ModuleNotFoundError: vllm._C` warnings come from the GPU vLLM import path. An
  `OSError: [Errno 28]` from vLLM's `usage_lib` thread only means its usage-stats file could not be written.
- **First steps are slow.** XLA compiles the actor's step and the engines' kernels on the first one or two
  steps and on the first validation.

## Files

| Path | What it is |
|---|---|
| `main_sdar_tpu.py` | Driver: builds the GPU job's config, runs SDAR's trainer with the TPU worker group |
| `sdar_tpu/worker_group.py` | verl worker-group API (`generate_sequences`, `compute_log_prob`, `compute_ref_log_prob`, `update_actor`, `save_checkpoint`, `load_checkpoint`) on TPU |
| `sdar_tpu/tunix_worker.py` | Ray actor: JAX/Tunix Qwen3, AdamW, SDAR loss (`sdar_policy_loss`), HF export, resume state |
| `sdar_tpu/engine.py` | vLLM-TPU engine: token-id batch generation, in-place weight reload |
| `eval_alfworld_tpu.py` | `eval_alfworld` protocol on TPU engines |
| `tests/test_loss_parity.py` | CPU check: JAX loss == SDAR torch loss |
| `data/text/{train,test}.parquet` | The placeholder prompt files `examples.data_preprocess.prepare --mode text` writes (ALFWorld itself comes from the envs) |
| `pydeps/` | Extra pure-Python packages (not committed; step 1.4) |
