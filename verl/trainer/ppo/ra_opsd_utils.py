"""
Reliability-Aware OPSD (RA-OPSD) utilities.

L_RA-OPSD = sum_t w_t * D(pi_S(. | h_t) || pi_T(. | h_t, I_t))

Each training sample is one agent turn, so w_t is a per-sample weight. It is derived from a reliability
score s_t (higher = more reliable teacher), computed by one of three signals:

  entropy       s_t = -mean_token H(pi_T(. | h_t, I_t, y_<k))   lower teacher entropy -> more reliable
  consistency   s_t = (# majority action) / K over K actions sampled from the PI-conditioned teacher
  disagreement  s_t = -mean_token KL_k3(pi_S || pi_T)          larger teacher-student gap -> less reliable

Scores are standardized over the batch and mapped to weights with w = 2 * sigmoid(alpha * z), then
(optionally) rescaled to mean 1 so the distillation loss keeps the same overall scale as unweighted OPSD.
"""

from collections import Counter
from typing import List

import torch

from agent_system.environments.env_package.alfworld.projection import alfworld_projection

SIGNALS = ("entropy", "consistency", "disagreement")


def _turn_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(values.dtype)
    return (values * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def entropy_score(teacher_entropys: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Negative mean teacher entropy over the turn's response tokens, (bs,)."""
    return -_turn_mean(teacher_entropys.float(), response_mask)


def disagreement_score(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Negative per-turn KL(pi_S || pi_T), estimated on the student's own tokens with the non-negative
    k3 estimator exp(r) - 1 - r, r = log pi_T - log pi_S. Returns (bs,)."""
    r = (teacher_log_probs.float() - student_log_probs.float()).clamp(-20, 20)
    kl = torch.exp(r) - 1 - r
    return -_turn_mean(kl, response_mask)


def consistency_score(responses: List[str], num_samples: int) -> torch.Tensor:
    """Self-consistency of K teacher samples per turn.

    responses: decoded teacher generations, grouped per turn (len = bs * K, K consecutive per turn).
    Each turn's score is the fraction of the K samples that produced the most common valid action;
    responses without a parsable <action> never agree with anything. Returns (bs,).
    """
    actions, valids = alfworld_projection(list(responses), None, require_think=False)
    scores = []
    for i in range(0, len(actions), num_samples):
        group = [a for a, v in zip(actions[i : i + num_samples], valids[i : i + num_samples]) if v]
        top = Counter(group).most_common(1)[0][1] if group else 0
        scores.append(top / num_samples)
    return torch.tensor(scores, dtype=torch.float32)


def reliability_to_weights(scores: torch.Tensor, alpha: float = 1.0, normalize: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Map reliability scores (higher = more reliable) to per-turn weights in (0, 2)."""
    scores = scores.float()
    z = (scores - scores.mean()) / (scores.std(unbiased=False) + eps)
    weights = 2 * torch.sigmoid(alpha * z)
    if normalize:
        weights = weights / weights.mean().clamp(min=eps)
    return weights


def weight_metrics(scores: torch.Tensor, weights: torch.Tensor, signal: str) -> dict:
    return {
        f"ra_opsd/{signal}_score_mean": scores.mean().item(),
        f"ra_opsd/{signal}_score_std": scores.std(unbiased=False).item(),
        "ra_opsd/weight_mean": weights.mean().item(),
        "ra_opsd/weight_std": weights.std(unbiased=False).item(),
        "ra_opsd/weight_min": weights.min().item(),
        "ra_opsd/weight_max": weights.max().item(),
    }
