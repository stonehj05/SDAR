"""CPU check: sdar_tpu.tunix_worker.sdar_policy_loss (JAX) == the loss dp_actor.update_policy builds from SDAR's
torch functions (compute_policy_loss, agg_loss(entropy), kl_penalty(low_var_kl), compute_sdar_loss with RA-OPSD
sample weights), in value and in gradient w.r.t. the log-probs and entropies.

  JAX_PLATFORMS=cpu PYTHONPATH=~/SDAR:~/SDAR/tpu:~/SDAR/tpu/pydeps python tpu/tests/test_loss_parity.py
"""

import numpy as np
import torch

import jax
import jax.numpy as jnp

from sdar_tpu.tunix_worker import sdar_policy_loss
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.trainer.ppo.sdar_utils import compute_sdar_loss


def torch_loss(lp, ent, d, lc):
    pg_loss, *_ = compute_policy_loss(old_log_prob=d["old"], log_prob=lp, advantages=d["adv"], response_mask=d["mask"],
                                      cliprange=0.2, cliprange_low=lc["clip_ratio_low"],
                                      cliprange_high=lc["clip_ratio_high"], clip_ratio_c=lc["clip_ratio_c"],
                                      loss_agg_mode="token-mean")
    loss = pg_loss * lc["pg_loss_coef"] - agg_loss(ent, d["mask"], "token-mean") * lc["entropy_coeff"]
    kld = kl_penalty(logprob=lp, ref_logprob=d["ref"], kl_penalty="low_var_kl")
    loss = loss + agg_loss(kld, d["mask"], "token-mean") * lc["kl_loss_coef"]
    sdar, _ = compute_sdar_loss(student_log_probs=lp, teacher_log_probs=d["teacher"], response_mask=d["mask"],
                                gate_beta=lc["sdar_gate_beta"], loss_agg_mode="token-mean",
                                sample_weights=d["weights"] if lc["use_opsd_weights"] else None)
    return loss + sdar * lc["sdar_loss_coef"]


def check(beta, use_w, pg_coef, seed):
    rng = np.random.default_rng(seed)
    B, T = 8, 37
    mask = (rng.random((B, T)) < 0.8).astype(np.float32)
    mask[:, 0] = 1
    old = -rng.random((B, T)).astype(np.float32) * 3
    lp = old + rng.normal(0, 0.3, (B, T)).astype(np.float32)
    ent = rng.random((B, T)).astype(np.float32) * 2
    adv = np.repeat(rng.normal(0, 1, (B, 1)), T, 1).astype(np.float32) * mask
    ref = old + rng.normal(0, 0.2, (B, T)).astype(np.float32)
    teacher = old + rng.normal(0, 0.5, (B, T)).astype(np.float32)
    weights = (2 / (1 + np.exp(-rng.normal(0, 1, B)))).astype(np.float32)
    lc = {"clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 3.0, "pg_loss_coef": pg_coef,
          "entropy_coeff": 0.001, "use_kl_loss": True, "kl_loss_coef": 0.01, "use_sdar_loss": True,
          "sdar_loss_coef": 0.01, "sdar_gate_beta": beta, "use_opsd_weights": use_w}
    d_np = {"mask": mask, "old": old, "adv": adv, "ref": ref, "teacher": teacher, "weights": weights}

    d_t = {k: torch.tensor(v, dtype=torch.float64) for k, v in d_np.items()}
    lp_t = torch.tensor(lp, dtype=torch.float64, requires_grad=True)
    ent_t = torch.tensor(ent, dtype=torch.float64, requires_grad=True)
    lt = torch_loss(lp_t, ent_t, d_t, lc)
    lt.backward()

    jax.config.update("jax_enable_x64", True)
    d_j = {k: jnp.asarray(v, jnp.float64) for k, v in d_np.items()}
    f = lambda a, b: sdar_policy_loss(a, b, d_j, lc)[0]
    lj, (glp, gent) = jax.value_and_grad(f, argnums=(0, 1))(jnp.asarray(lp, jnp.float64),
                                                           jnp.asarray(ent, jnp.float64))
    np.testing.assert_allclose(float(lj), lt.item(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(np.asarray(glp), lp_t.grad.numpy(), rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(np.asarray(gent), ent_t.grad.numpy(), rtol=1e-5, atol=1e-10)
    print(f"beta={beta} weights={use_w} pg_coef={pg_coef}: loss {float(lj):.6f} == {lt.item():.6f}, grads equal")


if __name__ == "__main__":
    for i, (beta, w, pg) in enumerate([(0.0, True, 1.0), (0.0, False, 1.0), (5.0, True, 1.0), (5.0, False, 0.0)]):
        check(beta, w, pg, i)
    print("OK")
