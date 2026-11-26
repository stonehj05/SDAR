"""vLLM-TPU rollout engine for SDAR: TCOD's vLLMTPURolloutModel plus token-id batch generation with explicit
sampling params (verl's vLLMRollout builds SamplingParams itself) and in-place weight reload from an HF dir."""

import asyncio
from typing import Dict, List

from trinity.common.models.vllm_tpu_model import vLLMTPURolloutModel


class SDARvLLMEngine(vLLMTPURolloutModel):
    async def generate_ids(self, prompts: List[List[int]], sampling: Dict) -> List[List[int]]:
        """One completion per prompt; returns the generated token ids (eos included, as vLLM returns them)."""
        import vllm
        from vllm.sampling_params import RequestOutputKind

        params = vllm.SamplingParams(
            n=1,
            max_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling.get("min_p", 0.0),
            ignore_eos=sampling.get("ignore_eos", False),
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        async def one(ids):
            self.request_id += 1
            async for out in self.async_llm.generate(
                    request_id=f"sdar-{self.request_id}", prompt={"prompt_token_ids": list(ids)},
                    sampling_params=params):
                if out.finished:
                    return list(out.outputs[0].token_ids)
            raise RuntimeError("vLLM request did not finish")

        return list(await asyncio.gather(*[one(p) for p in prompts]))

    async def load_weights(self, checkpoint_dir: str) -> int:
        await self.async_llm.reset_prefix_cache()
        n = await self._collective_rpc("update_weight_from_checkpoint", args=(checkpoint_dir,))
        await self.async_llm.reset_prefix_cache()
        return n
