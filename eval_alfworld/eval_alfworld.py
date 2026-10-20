"""Evaluate a (merged, HuggingFace-format) policy on the full ALFWorld valid_seen / valid_unseen splits.

Unlike the in-training validation (each env worker draws a random game, so games repeat and some are never
played), every game in the split is played exactly once. Prompts, history, action parsing and episode limits
reuse the training code path (AlfWorldEnvironmentManager + alfworld_projection), so numbers are comparable.

Usage:
    python eval_alfworld/eval_alfworld.py --model /path/to/hf_model --splits seen unseen --out_dir results/x
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from functools import partial

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import ray
from omegaconf import OmegaConf

SPLITS = {"seen": "eval_in_distribution", "unseen": "eval_out_of_distribution"}
TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]
ALF_CONFIG = os.path.join(REPO, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")


def task_type_of(gamefile):
    # pick_two_obj_and_place contains "pick_" but not "pick_and_place", so plain substring order is safe.
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return "unknown"


class GameWorker:
    """Ray actor holding one TextWorld env that can be pointed at a specific game file."""

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


class FixedGameEnvs:
    """Same interface as env_package.alfworld.envs.AlfworldEnvs, but plays a given list of games."""

    def __init__(self, workers):
        self.workers = workers
        self.active = workers
        self.prev_admissible_commands = []

    def load(self, gamefiles):
        self.active = self.workers[: len(gamefiles)]
        ray.get([w.load.remote(g) for w, g in zip(self.active, gamefiles)])
        self.prev_admissible_commands = [None] * len(gamefiles)

    def reset(self):
        text_obs, infos_out = [], []
        for i, (obs, info) in enumerate(ray.get([w.reset.remote() for w in self.active])):
            info = {k: v[0] for k, v in info.items()}
            text_obs.append(obs[0])
            self.prev_admissible_commands[i] = info["admissible_commands"]
            infos_out.append(info)
        return text_obs, None, infos_out

    def step(self, actions):
        assert len(actions) == len(self.active)
        text_obs, rewards, dones, infos_out = [], [], [], []
        results = ray.get([w.step.remote(a) for w, a in zip(self.active, actions)])
        for i, (obs, scores, done, info) in enumerate(results):
            info = {k: v[0] for k, v in info.items()}
            text_obs.append(obs[0])
            dones.append(done[0])
            infos_out.append(info)
            self.prev_admissible_commands[i] = info["admissible_commands"]
            rewards.append(10.0 * float(info["won"]))
        return text_obs, None, rewards, dones, infos_out

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands


def collect_games(split):
    import yaml
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

    with open(ALF_CONFIG) as f:
        config = yaml.safe_load(f)
    base_env = get_environment(config["env"]["type"])(config, train_eval=SPLITS[split])
    return base_env, sorted(base_env.game_files)


def build_prompt_ids(tokenizer, text, args):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=args.enable_thinking,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    truncated = len(ids) > args.max_prompt_length
    if truncated:
        ids = ids[-args.max_prompt_length :]
    return ids, truncated


def run_split(split, llm, tokenizer, sampling_params, args):
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    from vllm.inputs import TokensPrompt

    base_env, games = collect_games(split)
    if args.limit > 0:
        games = games[: args.limit]
    print(f"[{split}] {len(games)} games, {args.num_parallel} in parallel", flush=True)

    worker_cls = ray.remote(num_cpus=0.1)(GameWorker)
    workers = [worker_cls.remote(base_env) for _ in range(min(args.num_parallel, len(games)))]
    envs = FixedGameEnvs(workers)
    cfg = OmegaConf.create({"env": {"history_length": args.history_length}})
    manager = AlfWorldEnvironmentManager(envs, partial(alfworld_projection, require_think=args.require_think), cfg)

    records = []
    n_truncated = 0
    t0 = time.time()
    for start in range(0, len(games), args.num_parallel):
        chunk = games[start : start + args.num_parallel]
        envs.load(chunk)
        obs, infos = manager.reset({})
        n = len(chunk)
        done = np.zeros(n, dtype=bool)
        won = np.zeros(n)
        steps = np.zeros(n, dtype=int)
        invalid = np.zeros(n, dtype=int)
        trajs = [[] for _ in range(n)]

        for _ in range(args.max_steps):
            active = np.where(~done)[0]
            prompts = []
            for i in active:
                ids, truncated = build_prompt_ids(tokenizer, obs["text"][i], args)
                n_truncated += truncated
                prompts.append(TokensPrompt(prompt_token_ids=ids))
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            responses = [""] * n
            for j, i in enumerate(active):
                responses[i] = outputs[j].outputs[0].text
            obs, rewards, dones, infos = manager.step(responses)

            for i in active:
                steps[i] += 1
                valid = bool(infos[i]["is_action_valid"])
                invalid[i] += not valid
                if args.save_traj:
                    trajs[i].append({"response": responses[i], "valid": valid, "obs": obs["anchor"][i]})
                if dones[i]:
                    done[i] = True
                    won[i] = float(infos[i]["won"])
            if done.all():
                break
        for i in np.where(~done)[0]:
            won[i] = float(infos[i]["won"])

        for i, g in enumerate(chunk):
            rec = {
                "gamefile": os.path.relpath(g, os.environ.get("ALFWORLD_DATA", "")),
                "task_type": task_type_of(g),
                "won": won[i],
                "steps": int(steps[i]),
                "invalid_actions": int(invalid[i]),
            }
            if args.save_traj:
                rec["trajectory"] = trajs[i]
            records.append(rec)
        done_games = start + n
        acc = np.mean([r["won"] for r in records])
        print(f"[{split}] {done_games}/{len(games)} games, running success {acc:.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)

    for w in workers:
        ray.kill(w)

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
        "invalid_action_ratio": float(sum(r["invalid_actions"] for r in records) / max(1, sum(r["steps"] for r in records))),
        "truncated_prompts": int(n_truncated),
        "seconds": time.time() - t0,
    }
    out = os.path.join(args.out_dir, split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HuggingFace-format model directory")
    p.add_argument("--splits", nargs="+", default=["seen", "unseen"], choices=list(SPLITS))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_parallel", type=int, default=16, help="games played concurrently (one env process each)")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--history_length", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.4, help="training validation uses 0.4; 0 = greedy")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode (training used it off)")
    p.add_argument("--require_think", action="store_true", help="count actions without <think> as invalid")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only play the first N games per split (debug)")
    p.add_argument("--save_traj", action="store_true", help="store responses/observations per step")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    ray.init(num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), include_dashboard=False,
             object_store_memory=1_000_000_000)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_response_length,
    )

    summaries = {}
    for split in args.splits:
        summaries[split] = run_split(split, llm, tokenizer, sampling_params, args)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n===== ALFWorld eval:", args.model)
    for split, s in summaries.items():
        print(f"{split:>7}: success {s['success_rate']:.4f} over {s['num_games']} games | "
              f"avg steps {s['avg_steps']:.1f} | invalid actions {s['invalid_action_ratio']:.3f}")
        for t, v in s["per_task_success_rate"].items():
            print(f"         {t:<32} {v:.3f}  (n={s['per_task_num_games'][t]})")


if __name__ == "__main__":
    main()
