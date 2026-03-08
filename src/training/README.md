# Training Scripts

Each script trains Llama-3.1-8B-Instruct on MATH using a different method,
saving initial and final weights for sparsity comparison.

## Files

- `train_sdft.py` — SDFT training (extends official DistilTrainer). Supports both trace and answer-only demonstrations via config.
- `train_sft.py` — Standard SFT using TRL's SFTTrainer.
- `train_grpo.py` — GRPO with outcome-based reward (answer correctness).
- `train_offline_distil.py` — Offline distillation: generate teacher outputs first, then train off-policy on them.

## Weight Snapshots

All training scripts save `theta_init.pt` (state dict before training) alongside
the final checkpoint for post-hoc sparsity analysis.
