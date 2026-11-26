"""eval_alfworld/eval_alfworld.py on TPU: every valid_seen / valid_unseen game played once, same prompts, history,
action projection, sampling (temperature 0.4, 512 tokens) and 50-step limit, with vLLM-TPU engines (one chip each)
instead of a single GPU vLLM. Chunks of --num_parallel games run concurrently, one chunk per engine.

Several models are evaluated in one go: engines start from the first and reload weights for each next one.

  cd ~/SDAR && ~/venv-vllm-tpu/bin/python tpu/eval_alfworld_tpu.py --models /dev/shm/models/Qwen3-1.7B \
      .../ckpt/global_step_150/actor/huggingface --names qwen3_1.7b_base_tpu ra_entropy_..._global_step_150
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

HOME = os.path.expanduser("~")
TPU_DIR = os.path.dirname(os.path.abspath(__file__))
SDAR = os.path.dirname(TPU_DIR)
PYDEPS = os.path.join(TPU_DIR, "pydeps")
sys.path[:0] = [SDAR, TPU_DIR, PYDEPS, os.path.join(SDAR, "eval_alfworld")]

import numpy as np  # noqa: E402
import ray  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from eval_alfworld import FixedGameEnvs, build_prompt_ids, collect_games, task_type_of  # noqa: E402


class GameWorker:
    """eval_alfworld.GameWorker, defined here so Ray ships it by value: hosts other than this one see
    ~/SDAR/eval_alfworld as a namespace package and cannot import the original."""

    def __init__(self, base_env):
        self.base_env = base_env
        self.env = None

    def load(self, gamefile):
        if self.env is not None:
            self.env.close()
        self.base_env.game_files = [gamefile]
        self.base_env.num_games = 1
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(0)

    def reset(self):
        obs, infos = self.env.reset()
        infos["observation_text"] = obs
        return obs, infos

    def step(self, action):
        obs, scores, dones, infos = self.env.step([action])
        infos["observation_text"] = obs
        return obs, scores, dones, infos


def play_chunk(chunk, engine, workers, tokenizer, sampling, args):
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection

    envs = FixedGameEnvs(workers)
    cfg = OmegaConf.create({"env": {"history_length": args.history_length}})
    manager = AlfWorldEnvironmentManager(envs, partial(alfworld_projection, require_think=args.require_think), cfg)
    envs.load(chunk)
    obs, infos = manager.reset({})
    n = len(chunk)
    done = np.zeros(n, dtype=bool)
    won = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    invalid = np.zeros(n, dtype=int)
    n_truncated = 0
    for _ in range(args.max_steps):
        active = np.where(~done)[0]
        prompts = []
        for i in active:
            ids, truncated = build_prompt_ids(tokenizer, obs["text"][i], args)
            n_truncated += truncated
            prompts.append(ids)
        outputs = ray.get(engine.generate_ids.remote(prompts, sampling))
        responses = [""] * n
        for j, i in enumerate(active):
            responses[i] = tokenizer.decode(outputs[j], skip_special_tokens=True)
        obs, rewards, dones, infos = manager.step(responses)
        for i in active:
            steps[i] += 1
            invalid[i] += not bool(infos[i]["is_action_valid"])
            if dones[i]:
                done[i] = True
                won[i] = float(infos[i]["won"])
        if done.all():
            break
    for i in np.where(~done)[0]:
        won[i] = float(infos[i]["won"])
    records = [{"gamefile": os.path.relpath(g, os.environ.get("ALFWORLD_DATA", "")), "task_type": task_type_of(g),
                "won": won[i], "steps": int(steps[i]), "invalid_actions": int(invalid[i])}
               for i, g in enumerate(chunk)]
    return records, n_truncated


def run_split(split, engines, worker_sets, tokenizer, sampling, out_dir, args):
    _, games = collect_games(split)
    if args.limit > 0:
        games = games[: args.limit]
    chunks = [games[s: s + args.num_parallel] for s in range(0, len(games), args.num_parallel)]
    print(f"[{split}] {len(games)} games in {len(chunks)} chunks over {len(engines)} engines", flush=True)
    t0 = time.time()
    records, n_truncated = [], 0

    def lane(k):
        out = []
        for c in range(k, len(chunks), len(engines)):
            out.append(play_chunk(chunks[c], engines[k], worker_sets[k], tokenizer, sampling, args))
            print(f"[{split}] chunk {c + 1}/{len(chunks)} done, {time.time() - t0:.0f}s", flush=True)
        return out

    with ThreadPoolExecutor(len(engines)) as pool:
        for lane_out in pool.map(lane, range(len(engines))):
            for recs, nt in lane_out:
                records += recs
                n_truncated += nt
    records.sort(key=lambda r: r["gamefile"])

    by_type = defaultdict(list)
    for r in records:
        by_type[r["task_type"]].append(r["won"])
    summary = {
        "split": split,
        "num_games": len(records),
        "success_rate": float(np.mean([r["won"] for r in records])),
        "per_task_success_rate": {t: float(np.mean(v)) for t, v in sorted(by_type.items())},
        "per_task_num_games": {t: len(v) for t, v in sorted(by_type.items())},
        "avg_steps": float(np.mean([r["steps"] for r in records])),
        "invalid_action_ratio": float(sum(r["invalid_actions"] for r in records)
                                      / max(1, sum(r["steps"] for r in records))),
        "truncated_prompts": int(n_truncated),
        "seconds": time.time() - t0,
    }
    out = os.path.join(out_dir, split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="HF dirs; engines start from the first")
    p.add_argument("--names", nargs="+", required=True, help="result dir name per model")
    p.add_argument("--results_root", default=os.path.join(SDAR, "eval_alfworld", "results"))
    p.add_argument("--splits", nargs="+", default=["seen", "unseen"])
    p.add_argument("--engines", type=int, default=12)
    p.add_argument("--num_parallel", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--history_length", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--require_think", action="store_true")
    p.add_argument("--engine_mem", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    assert len(args.models) == len(args.names)

    alf_data = os.environ.get("ALFWORLD_DATA", os.path.join(HOME, "alf-data"))
    env_vars = {"ALFWORLD_DATA": alf_data, "TMPDIR": "/dev/shm/tmp", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    os.environ.update({k: v for k, v in env_vars.items() if k != "OMP_NUM_THREADS"})
    ray.init(address="auto", namespace="sdar_eval", log_to_driver=True, runtime_env={
        "py_modules": [os.path.join(SDAR, "agent_system"), os.path.join(TPU_DIR, "sdar_tpu"),
                       os.path.join(PYDEPS, "gymnasium"), os.path.join(PYDEPS, "farama_notifications")],
        "env_vars": env_vars,
    })

    from transformers import AutoTokenizer

    from sdar_tpu.engine import SDARvLLMEngine
    from trinity.common.config import InferenceModelConfig

    tokenizer = AutoTokenizer.from_pretrained(args.models[0])
    cluster = ray.cluster_resources()
    slots = [lab for lab in ("explorer_tpu", "teacher_tpu") if lab in cluster for _ in range(int(cluster[lab]))]
    engines = []
    for i in range(args.engines):
        mc = InferenceModelConfig(
            model_path=args.models[0], engine_type="vllm_tpu", tensor_parallel_size=1,
            gpu_memory_utilization=args.engine_mem, max_model_len=args.max_prompt_length + args.max_response_length,
            max_prompt_tokens=args.max_prompt_length, max_response_tokens=args.max_response_length,
            temperature=args.temperature, top_p=args.top_p, top_k=-1, enable_prefix_caching=True,
            enable_chunked_prefill=True, min_response_tokens=0, logprobs=0, repetition_penalty=1.0,
            dtype="bfloat16", seed=i, enable_thinking=args.enable_thinking)
        mc.ray_namespace = "sdar_eval"
        engines.append(ray.remote(SDARvLLMEngine).options(
            name=f"sdar_eval_engine_{i}", num_cpus=0, resources={"TPU": 1, slots[i]: 1}).remote(config=mc))
    try:
        ray.get([e.prepare.remote() for e in engines])
        print(f"{len(engines)} engines ready", flush=True)
        sampling = {"max_tokens": args.max_response_length, "temperature": args.temperature,
                    "top_p": args.top_p if args.temperature > 0 else 1.0, "top_k": -1}
        all_summaries = {}
        for m, (model, name) in enumerate(zip(args.models, args.names)):
            if m > 0:
                ray.get([e.load_weights.remote(model) for e in engines])
            out_dir = os.path.join(args.results_root, name)
            os.makedirs(out_dir, exist_ok=True)
            summaries = {}
            for split in args.splits:
                base_env, _ = collect_games(split)
                worker_cls = ray.remote(num_cpus=0.1)(GameWorker)
                worker_sets = [[worker_cls.remote(base_env) for _ in range(args.num_parallel)] for _ in engines]
                summaries[split] = run_split(split, engines, worker_sets, tokenizer, sampling, out_dir, args)
                for ws in worker_sets:
                    for w in ws:
                        ray.kill(w)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump({**vars(args), "model": model, "out_dir": out_dir, "backend": "vllm_tpu"}, f, indent=2)
            all_summaries[name] = (model, summaries)
            print(f"\n===== ALFWorld eval: {model}", flush=True)
            for split, s in summaries.items():
                print(f"{split:>7}: success {s['success_rate']:.4f} over {s['num_games']} games | "
                      f"avg steps {s['avg_steps']:.1f} | invalid actions {s['invalid_action_ratio']:.3f}", flush=True)
                for t, v in s["per_task_success_rate"].items():
                    print(f"         {t:<32} {v:.3f}  (n={s['per_task_num_games'][t]})", flush=True)
    finally:
        ray.get([e.shutdown.remote() for e in engines])


if __name__ == "__main__":
    main()
