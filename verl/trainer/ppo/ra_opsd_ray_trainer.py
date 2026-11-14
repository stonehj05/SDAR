"""
Reliability-Aware OPSD (RA-OPSD) trainer.

Same loop as SkillSDRayTrainer (SDAR); the teacher forward pass additionally produces a per-turn reliability
weight `opsd_weights` (bs,) that dp_actor multiplies into the distillation loss. See ra_opsd_utils.py for the
three reliability signals.
"""

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import ra_opsd_utils
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class RAOPSDRayTrainer(SkillSDRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("ra_opsd", {})
        self.ra_signal = cfg.get("signal", "entropy")
        assert self.ra_signal in ra_opsd_utils.SIGNALS, f"unknown ra_opsd.signal {self.ra_signal}, expected one of {ra_opsd_utils.SIGNALS}"
        self.ra_alpha = cfg.get("alpha", 1.0)
        self.ra_normalize = cfg.get("normalize", True)
        self.ra_consistency_samples = cfg.get("consistency_samples", 4)

    def _get_training_teacher_log_probs(self, batch: DataProto):
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]
        response_mask = batch.batch["response_mask"]

        if self.ra_signal == "entropy":
            scores = ra_opsd_utils.entropy_score(teacher_output.batch["entropys"], response_mask)
        elif self.ra_signal == "disagreement":
            scores = ra_opsd_utils.disagreement_score(batch.batch["old_log_probs"], teacher_log_probs, response_mask)
        else:
            scores = self._teacher_consistency(teacher_batch)

        weights = ra_opsd_utils.reliability_to_weights(scores, alpha=self.ra_alpha, normalize=self.ra_normalize)
        batch.batch["opsd_weights"] = weights
        return teacher_log_probs, ra_opsd_utils.weight_metrics(scores, weights, self.ra_signal)

    def _teacher_consistency(self, teacher_batch: DataProto) -> torch.Tensor:
        """Sample K actions per turn from the PI-conditioned teacher and score their agreement."""
        prompt_length = self.config.data.max_prompt_length
        prompts = DataProto.from_dict(
            tensors={
                key: teacher_batch.batch[key][:, :prompt_length]
                for key in ("input_ids", "attention_mask", "position_ids")
            },
        )
        prompts = prompts.repeat(repeat_times=self.ra_consistency_samples, interleave=True)
        prompts_padded, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_wg.world_size)
        output = unpad_dataproto(self.actor_rollout_wg.generate_sequences(prompts_padded), pad_size=pad_size)
        texts = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
        return ra_opsd_utils.consistency_score(texts, self.ra_consistency_samples)
