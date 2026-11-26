"""JAX/Tunix counterpart of verl's FSDP ActorRolloutRefWorker (actor + ref parts) for SDAR.

Runs as one Ray actor that owns the trainer's TPU chips. It holds
  - fp32 master weights + AdamW state of the actor, sharded over the chips (`fsdp` mesh axis),
  - a frozen copy of the initial weights as the reference policy (verl's ref worker),
and computes in bf16 with fp32 master weights, like verl's FSDP mixed precision (param_dtype bf16,
reduce/buffer fp32).

What it reproduces (verl/workers/actor/dp_actor.py, SDAR's version):
  compute_log_prob     -> `score(rows, ref=False, entropy=True)`: per-token log-probs + entropy at the current weights
  compute_ref_log_prob -> `score(rows, ref=True)`
  update_policy        -> `update(...)`: verl's mini-batch / micro-batch loop for one dp layout (see worker_group.py),
                          loss = pg_loss * pg_loss_coef - entropy_coeff * entropy + kl_loss_coef * low_var_kl(ref)
                                 + sdar_loss_coef * SDAR loss (gate sigmoid(beta * (teacher - student)), optional
                                 per-sample RA-OPSD weights), every term token-mean over the micro-batch,
                          micro loss / gradient_accumulation, grad clip, AdamW step (skipped on a non-finite norm).
"""

import math
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def _masked_mean(x, mask):
    # verl_F.masked_mean
    return (x * mask).sum() / (mask.sum() + 1e-8)


def sdar_policy_loss(log_prob, entropy, mb, lc):
    """dp_actor.update_policy's per-micro-batch loss (before / gradient_accumulation), all terms token-mean."""
    mask = mb["mask"]
    old, adv = mb["old"], mb["adv"]
    # core_algos.compute_policy_loss (vanilla, dual-clip)
    neg_kl = jnp.clip(log_prob - old, -20.0, 20.0)
    ratio = jnp.exp(neg_kl)
    ppo_kl = _masked_mean(-neg_kl, mask)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - lc["clip_ratio_low"], 1.0 + lc["clip_ratio_high"])
    clip1 = jnp.maximum(pg1, pg2)
    pg_clipfrac = _masked_mean((pg2 > pg1).astype(jnp.float32), mask)
    pg3 = -adv * lc["clip_ratio_c"]
    clip2 = jnp.minimum(pg3, clip1)
    pg_clipfrac_lower = _masked_mean(((clip1 > pg3) & (adv < 0)).astype(jnp.float32), mask)
    pg_losses = jnp.where(adv < 0, clip2, clip1)
    pg_loss = _masked_mean(pg_losses, mask)
    policy_loss = pg_loss * lc["pg_loss_coef"]
    if lc["entropy_coeff"] != 0:
        policy_loss = policy_loss - _masked_mean(entropy, mask) * lc["entropy_coeff"]
    aux = {"pg_loss": pg_loss, "pg_clipfrac": pg_clipfrac, "ppo_kl": ppo_kl, "pg_clipfrac_lower": pg_clipfrac_lower}
    if lc["use_kl_loss"]:
        # core_algos.kl_penalty(low_var_kl)
        kl = mb["ref"] - log_prob
        kl_loss = _masked_mean(jnp.clip(jnp.exp(kl) - kl - 1.0, -10.0, 10.0), mask)
        policy_loss = policy_loss + kl_loss * lc["kl_loss_coef"]
        aux["kl_loss"] = kl_loss
    if lc["use_sdar_loss"]:
        # sdar_utils.compute_sdar_loss (sample_weights = RA-OPSD per-turn weights)
        teacher = mb["teacher"]
        delta = teacher - jax.lax.stop_gradient(log_prob)
        gate = jax.lax.stop_gradient(jax.nn.sigmoid(lc["sdar_gate_beta"] * delta))
        gated_kl = gate * (teacher - log_prob)
        if lc["use_opsd_weights"]:
            gated_kl = mb["weights"][:, None] * gated_kl
        sdar_loss = _masked_mean(gated_kl, mask)
        policy_loss = policy_loss + sdar_loss * lc["sdar_loss_coef"]
        denom = jnp.clip(mask.sum(), min=1.0)
        aux.update({
            "sdar/gate_mean": (gate * mask).sum() / denom,
            "sdar/gate_active_ratio": ((gate > 0.5) * mask).sum() / denom,
            "sdar/teacher_gap_mean": (delta * mask).sum() / denom,
            "sdar/loss": sdar_loss,
        })
    return policy_loss, aux


class SDARTunixWorker:
    def __init__(self, cfg: Dict[str, Any]):
        from tunix.models.qwen3 import model as qwen3_lib
        from tunix.utils.mesh import create_mesh

        from trinity.trainer.tunix import hf_io
        from trinity.trainer.tunix.logps import completion_logps

        self.cfg = cfg
        n = jax.device_count()
        self.n_chips = n
        self.mesh = create_mesh((n, 1), ("fsdp", "tp"))
        self.model_dir = hf_io.resolve_model_dir(cfg["model_path"])
        compute_dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[cfg.get("compute_dtype", "bfloat16")]
        import dataclasses

        from tunix.models import automodel

        name = hf_io.tunix_model_name(self.model_dir)
        model_cfg = dataclasses.replace(
            automodel.call_model_config(name),
            use_flash_attention=True,
            flash_attention_block_size=cfg["flash_block_size"],
            remat_config=getattr(qwen3_lib.RematConfig, cfg["remat"]),
            dtype=compute_dtype,  # activations / matmul dtype; params stay fp32 (master weights)
            param_dtype=jnp.float32,
        )
        with self.mesh:
            model = automodel.create_model_from_safe_tensors(name, self.model_dir, model_cfg, self.mesh,
                                                             dtype=jnp.float32)
        self.graphdef, self.params, self.rest = nnx.split(model, nnx.Param, ...)
        del model
        self.param_shardings = jax.tree.map(lambda x: x.sharding, self.params)
        with self.mesh:
            # verl's reference policy: the initial weights, frozen.
            self.ref_params = jax.jit(lambda p: jax.tree.map(jnp.copy, p), out_shardings=self.param_shardings)(
                self.params)
        opt = cfg["optimizer"]
        # torch.optim.AdamW(lr, betas, eps=1e-8, weight_decay) + clip_grad_norm_(grad_clip), constant lr
        self.tx = optax.chain(
            optax.clip_by_global_norm(opt["clip_grad"]),
            optax.adamw(opt["lr"], b1=opt["betas"][0], b2=opt["betas"][1], eps=1e-8,
                        weight_decay=opt["weight_decay"]),
        )
        with self.mesh:
            self.opt_state = self.tx.init(self.params)
        self.pad_id = cfg["pad_token_id"]
        self.eos_id = cfg["eos_token_id"]
        self.step = 0
        graphdef, rest = self.graphdef, self.rest
        pad_id, eos_id, temperature = self.pad_id, self.eos_id, cfg["temperature"]
        lc = cfg["loss"]

        def score(params, prompt, comp):
            model = nnx.merge(graphdef, params, rest)
            lp, ent = completion_logps(model, prompt, comp, pad_id, eos_id, temperature, return_entropy=True)
            return jax.lax.stop_gradient(lp), jax.lax.stop_gradient(ent)

        def loss_fn(params, mb):
            model = nnx.merge(graphdef, params, rest)
            log_prob, entropy = completion_logps(model, mb["prompt"], mb["comp"], pad_id, eos_id, temperature,
                                                 return_entropy=True)
            policy_loss, aux = sdar_policy_loss(log_prob, entropy, mb, lc)
            return policy_loss / mb["grad_accum"], aux

        def grad_step(params, mb, acc):
            (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            return aux, jax.tree.map(jnp.add, acc, grads)

        def apply_grads(params, opt_state, grads):
            updates, opt_state = self.tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        ps = self.param_shardings
        self._score = jax.jit(score)
        self._grad_step = jax.jit(grad_step, donate_argnums=(2,), out_shardings=(None, ps))
        self._grad_norm = jax.jit(optax.global_norm)
        self._apply_grads = jax.jit(apply_grads, donate_argnums=(0, 1))
        self._zeros = jax.jit(lambda p: jax.tree.map(jnp.zeros_like, p), out_shardings=ps)

    # ------------------------------------------------------------------ batching

    def _bucket(self, total_len: int) -> int:
        for b in self.cfg["seq_buckets"]:
            if total_len <= b:
                return b
        raise ValueError(f"sequence of {total_len} tokens exceeds the largest bucket {self.cfg['seq_buckets'][-1]}")

    def _pack(self, rows: List[Dict[str, np.ndarray]], idx: List[int]):
        """Rows idx -> left-padded prompt [B, P] + right-padded completion [B, C], B a multiple of the chip count
        (filler rows are fully masked copies)."""
        c = self.cfg["max_response_tokens"]
        g = list(idx)
        n_real = len(g)
        while len(g) % self.n_chips:
            g.append(g[-1])
        p = self._bucket(max(len(rows[i]["prompt"]) for i in g) + c) - c
        prompt = np.full((len(g), p), self.pad_id, np.int32)
        comp = np.full((len(g), c), self.pad_id, np.int32)
        real = np.zeros((len(g),), bool)
        real[:n_real] = True
        for r, i in enumerate(g):
            prompt[r, p - len(rows[i]["prompt"]):] = rows[i]["prompt"]
            comp[r, :len(rows[i]["response"])] = rows[i]["response"]
        return prompt, comp, real

    # ------------------------------------------------------------------ compute_log_prob / compute_ref_log_prob

    def score(self, rows: List[Dict[str, np.ndarray]], ref: bool = False) -> Dict[str, List[np.ndarray]]:
        """Per-token log-probs and entropies of each row's response tokens (lengths = len(row['response']))."""
        params = self.ref_params if ref else self.params
        per = self.cfg["score_rows_per_chip"] * self.n_chips
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
        logps = [None] * len(rows)
        ents = [None] * len(rows)
        with self.mesh:
            for a in range(0, len(order), per):
                g = order[a:a + per]
                prompt, comp, _ = self._pack(rows, g)
                lp, ent = self._score(params, jnp.asarray(prompt), jnp.asarray(comp))
                lp, ent = np.asarray(lp), np.asarray(ent)
                for r, i in enumerate(g):
                    n = len(rows[i]["response"])
                    logps[i], ents[i] = lp[r, :n], ent[r, :n]
        return {"log_probs": logps, "entropys": ents}

    # ------------------------------------------------------------------ update_policy

    def update(self, rows: List[Dict[str, Any]], mini_batches: List[List[List[int]]], grad_accum: int) -> Dict:
        """One verl update_policy call (ppo_epochs = 1).

        rows: dicts with prompt, response, mask, old, adv, ref, teacher (per response token), weight (scalar).
        mini_batches: for each optimizer step, the list of micro-batches (row indices); each micro-batch is one
            dp rank's micro-batch in verl (ppo_micro_batch_size_per_gpu rows) and is reduced with its own token-mean.
        grad_accum: verl's gradient_accumulation * dp world size (every micro loss is divided by it).
        """
        c = self.cfg["max_response_tokens"]
        metrics: Dict[str, list] = {}
        t0 = time.time()
        for micro_list in mini_batches:
            with self.mesh:
                acc = self._zeros(self.params)
            for g in micro_list:
                prompt, comp, real = self._pack(rows, g)
                b = len(real)
                arrs = {k: np.zeros((b, c), np.float32) for k in ("mask", "old", "adv", "ref", "teacher")}
                weights = np.ones((b,), np.float32)
                for r, i in enumerate(g):
                    row = rows[i]
                    n = len(row["response"])
                    for k in ("mask", "old", "adv", "ref", "teacher"):
                        arrs[k][r, :n] = row[k]
                    weights[r] = row["weight"]
                arrs["mask"][~real] = 0.0  # filler copies contribute nothing
                mb = {"prompt": jnp.asarray(prompt), "comp": jnp.asarray(comp),
                      **{k: jnp.asarray(v) for k, v in arrs.items()},
                      "weights": jnp.asarray(weights), "grad_accum": jnp.float32(grad_accum)}
                with self.mesh:
                    aux, acc = self._grad_step(self.params, mb, acc)
                for k, v in aux.items():
                    metrics.setdefault(k if "/" in k else f"actor/{k}", []).append(float(v))
            with self.mesh:
                grad_norm = float(self._grad_norm(acc))
                if math.isfinite(grad_norm):
                    self.params, self.opt_state = self._apply_grads(self.params, self.opt_state, acc)
                else:  # dp_actor._optimizer_step: skip the update on a non-finite gradient
                    print(f"WARN: rank 0 grad_norm is not finite: {grad_norm}")
                del acc
            metrics.setdefault("actor/grad_norm", []).append(grad_norm)
        self.step += 1
        metrics["time/tunix_update_s"] = [time.time() - t0]
        return metrics

    # ------------------------------------------------------------------ checkpoints

    def save_hf(self, output_dir: str) -> str:
        from trinity.trainer.tunix import hf_io

        with self.mesh:
            model = nnx.merge(self.graphdef, self.params, self.rest)
            return hf_io.save_hf_checkpoint(model, self.model_dir, output_dir)

    def export_state(self, out_dir: str) -> Dict:
        from trinity.trainer.tunix import resume

        return resume.export_state(self, out_dir, self.step, prune_others=False)

    def load_state(self, in_dir: str) -> int:
        from trinity.trainer.tunix import resume

        return resume.load_state(self, in_dir)

    def ready(self) -> bool:
        return True
