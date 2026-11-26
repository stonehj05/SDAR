"""Worker group with the interface SDAR's RayPPOTrainer subclasses call on `actor_rollout_wg` / `ref_policy_wg`
(verl/workers/fsdp_workers.py ActorRolloutRefWorker + vllm_rollout_spmd.vLLMRollout), backed by
  - N vLLM-TPU engines (one chip each) for generate_sequences,
  - one SDARTunixWorker actor (the trainer's chips) for compute_log_prob / compute_ref_log_prob / update_actor.

`world_size` is the dp size of the GPU run being reproduced (trainer.n_gpus_per_node * nnodes). It is not the
number of TPU chips: it only fixes how verl pads (pad_dataproto_to_divisor), balances (_balance_batch) and splits
the batch into per-rank mini/micro batches, which update_actor reproduces exactly (each verl micro-batch is one
TPU micro-batch with its own token-mean; gradients are summed / (gradient_accumulation * world_size), as FSDP's
all-reduce mean of per-rank accumulated gradients does).

After every update_actor the new weights are exported as a bf16 HF checkpoint and loaded into every engine (verl's
hybrid engine resyncs vLLM from FSDP before the next generation).
"""

import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # vllm_rollout_spmd._pre_process_inputs: strip the left padding
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


class TPUActorRolloutRefWorkerGroup:
    def __init__(self, config, tokenizer, engines: List, trainer_actor, sync_dir: str, hf_keep_every: int,
                 eos_token_id, pad_token_id: int):
        self.config = config
        self.tokenizer = tokenizer
        self.engines = engines
        self.trainer = trainer_actor
        self.world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
        self.sync_dir = sync_dir
        self.hf_keep_every = hf_keep_every
        # fsdp_workers.generate_sequences overwrites meta_info eos/pad with the model's generation_config
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        rollout = config.actor_rollout_ref.rollout
        self.response_length = rollout.response_length
        self.temperature = rollout.temperature
        # vLLMRollout.__init__: SamplingParams(n=1, logprobs=0, max_tokens=response_length, + rollout config keys)
        self.train_sampling = {"max_tokens": rollout.response_length, "temperature": rollout.temperature,
                               "top_p": rollout.top_p, "top_k": rollout.top_k, "ignore_eos": rollout.ignore_eos}
        self.val_sampling = dict(self.train_sampling, temperature=rollout.val_kwargs.temperature,
                                 top_p=rollout.val_kwargs.top_p, top_k=rollout.val_kwargs.top_k)
        self.greedy_sampling = dict(self.train_sampling, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0)
        self._synced_dir: Optional[str] = None
        self.timing: Dict[str, float] = {}
        # weight exports left behind by a crashed earlier attempt (4 GB each on tmpfs); a resume re-exports
        for d in os.listdir(sync_dir):
            shutil.rmtree(os.path.join(sync_dir, d), ignore_errors=True)

    # ------------------------------------------------------------------ rollout

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        t0 = time.time()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
        raw = [list(np.asarray(x).tolist()) if not isinstance(x, list) else x
               for x in non_tensor_batch.pop("raw_prompt_ids")]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sampling = self.greedy_sampling
        elif is_validate:
            sampling = self.val_sampling
        else:
            sampling = self.train_sampling

        # contiguous shards, one per engine
        n_eng = len(self.engines)
        bounds = np.linspace(0, batch_size, n_eng + 1).astype(int)
        futures = [self.engines[e].generate_ids.remote(raw[bounds[e]:bounds[e + 1]], sampling)
                   for e in range(n_eng) if bounds[e + 1] > bounds[e]]
        response = [r for part in ray.get(futures) for r in part]
        response = [r[:self.response_length] for r in response]

        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.response_length)
        # rollout log-probs are not used for training (verl recomputes old log-probs with the actor)
        rollout_log_probs = torch.full(response.shape, -1.0, dtype=torch.float32)
        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1).unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=self.eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch = TensorDict({"prompts": idx, "responses": response, "input_ids": seq,
                            "rollout_log_probs": rollout_log_probs, "attention_mask": attention_mask,
                            "position_ids": position_ids}, batch_size=batch_size)
        self.timing["generate"] = self.timing.get("generate", 0.0) + time.time() - t0
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    # ------------------------------------------------------------------ log-probs

    def _rows(self, data: DataProto, keys=()) -> List[Dict]:
        responses = data.batch["responses"]
        L = responses.size(1)
        input_ids = data.batch["input_ids"]
        attn = data.batch["attention_mask"]
        rows = []
        for i in range(input_ids.size(0)):
            p_ids, p_mask = input_ids[i, :-L], attn[i, :-L]
            r_mask = attn[i, -L:]
            n = int(r_mask.sum())
            row = {"prompt": p_ids[p_mask.bool()].numpy().astype(np.int32),
                   "response": responses[i, :n].numpy().astype(np.int32)}
            for k in keys:
                row[k] = data.batch[k][i, :n].float().numpy()
            rows.append(row)
        return rows

    def _scores_to_tensor(self, values: List[np.ndarray], L: int) -> torch.Tensor:
        out = torch.zeros((len(values), L), dtype=torch.float32)
        for i, v in enumerate(values):
            out[i, :len(v)] = torch.from_numpy(np.asarray(v, dtype=np.float32))
        return out

    def compute_log_prob(self, data: DataProto) -> DataProto:
        t0 = time.time()
        L = data.batch["responses"].size(1)
        res = ray.get(self.trainer.score.remote(self._rows(data), False))
        out = DataProto.from_dict(
            tensors={"old_log_probs": self._scores_to_tensor(res["log_probs"], L),
                     "entropys": self._scores_to_tensor(res["entropys"], L)},
            meta_info={"temperature": self.temperature})
        self.timing["compute_log_prob"] = self.timing.get("compute_log_prob", 0.0) + time.time() - t0
        return out

    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        t0 = time.time()
        L = data.batch["responses"].size(1)
        res = ray.get(self.trainer.score.remote(self._rows(data), True))
        out = DataProto.from_dict(tensors={"ref_log_prob": self._scores_to_tensor(res["log_probs"], L)})
        self.timing["compute_ref_log_prob"] = self.timing.get("compute_ref_log_prob", 0.0) + time.time() - t0
        return out

    # ------------------------------------------------------------------ update

    def update_actor(self, data: DataProto) -> DataProto:
        t0 = time.time()
        actor_cfg = self.config.actor_rollout_ref.actor
        assert not data.meta_info.get("multi_turn", False), "multi_turn loss_mask is not supported"
        use_kl = actor_cfg.use_kl_loss
        use_sdar = actor_cfg.get("use_sdar_loss", False)
        use_w = actor_cfg.get("use_opsd_weights", False)
        keys = ["old_log_probs", "advantages"] + (["ref_log_prob"] if use_kl else []) + \
            (["teacher_log_probs"] if use_sdar else [])
        rows = self._rows(data, keys)
        L = data.batch["responses"].size(1)
        weights = data.batch["opsd_weights"].float().numpy() if use_w else np.ones(len(rows), np.float32)
        for i, r in enumerate(rows):
            n = len(r["response"])
            r["mask"] = data.batch["attention_mask"][i, -L:][:n].float().numpy()
            r["old"] = r.pop("old_log_probs")
            r["adv"] = r.pop("advantages")
            r["ref"] = r.pop("ref_log_prob") if use_kl else np.zeros(n, np.float32)
            r["teacher"] = r.pop("teacher_log_probs") if use_sdar else np.zeros(n, np.float32)
            r["weight"] = float(weights[i])

        # verl: DP_COMPUTE_PROTO splits the batch into world_size contiguous chunks; each rank splits its chunk
        # into mini-batches of ppo_mini_batch_size (already divided by world size) and those into micro-batches.
        bs = len(rows)
        ws = self.world_size
        assert bs % ws == 0, f"batch {bs} not divisible by dp world size {ws}"
        per_rank = bs // ws
        mini = actor_cfg.ppo_mini_batch_size // ws  # fsdp_workers: *= rollout.n (1), //= world size
        micro = actor_cfg.ppo_micro_batch_size_per_gpu
        assert mini % micro == 0
        grad_accum = (mini // micro) * ws
        mini_batches = []
        for m0 in range(0, per_rank, mini):
            micro_list = []
            for r in range(ws):
                lo, hi = r * per_rank + m0, r * per_rank + min(m0 + mini, per_rank)
                micro_list += [list(range(a, min(a + micro, hi))) for a in range(lo, hi, micro)]
            mini_batches.append(micro_list)
        metrics = ray.get(self.trainer.update.remote(rows, mini_batches, grad_accum))
        metrics["actor/lr"] = actor_cfg.optim.lr
        if use_kl:
            metrics["actor/kl_coef"] = actor_cfg.kl_loss_coef
        if use_sdar:
            metrics["sdar/coef"] = actor_cfg.get("sdar_loss_coef", 0.1)
        t1 = time.time()
        self.sync_weights(step_tag=f"step_{int(time.time())}")
        metrics["timing_tpu/weight_sync_s"] = time.time() - t1
        metrics["timing_tpu/update_actor_s"] = t1 - t0
        for k, v in self.timing.items():
            metrics[f"timing_tpu/{k}_s"] = v
        self.timing = {}
        return DataProto(meta_info={"metrics": metrics})

    # ------------------------------------------------------------------ weights / checkpoints

    def sync_weights(self, step_tag: str) -> None:
        new_dir = os.path.join(self.sync_dir, step_tag)
        ray.get(self.trainer.save_hf.remote(new_dir))
        ray.get([e.load_weights.remote(new_dir) for e in self.engines])
        if self._synced_dir and self._synced_dir != new_dir:
            shutil.rmtree(self._synced_dir, ignore_errors=True)
        self._synced_dir = new_dir

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """<global_step_N>/actor: exact trainer state (fp32 weights, AdamW, step) for resuming, kept for the latest
        checkpoint only, plus an HF export every `hf_keep_every` steps (kept)."""
        os.makedirs(local_path, exist_ok=True)
        state_dir = os.path.join(local_path, "tunix_state")
        ray.get(self.trainer.export_state.remote(state_dir))
        if self.hf_keep_every > 0 and global_step % self.hf_keep_every == 0:
            ray.get(self.trainer.save_hf.remote(os.path.join(local_path, "huggingface")))
        root = os.path.dirname(os.path.dirname(local_path))
        for d in os.listdir(root):
            if not d.startswith("global_step_") or os.path.join(root, d, "actor") == local_path:
                continue
            # older states, and partial ones (tunix_state.tmp) from a save that crashed
            for name in ("tunix_state", "tunix_state.tmp"):
                old = os.path.join(root, d, "actor", name)
                if os.path.isdir(old):
                    shutil.rmtree(old, ignore_errors=True)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        step = ray.get(self.trainer.load_state.remote(os.path.join(local_path, "tunix_state")))
        print(f"[SDAR-TPU] restored trainer state at step {step} from {local_path}")
        self.sync_weights(step_tag=f"resume_{step}")
