"""TPU backend for SDAR (verl) training: vLLM-TPU rollout engines + a JAX/Tunix actor, driven by SDAR's own
RayPPOTrainer subclasses through a worker group with verl's ActorRolloutRefWorker interface (worker_group.py)."""
