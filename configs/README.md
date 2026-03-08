# Training Configurations

YAML configs for each training method. All share the same base model and dataset
but differ in training objective and data presentation.

## Files

- `base.yaml` — Shared settings (model, tokenizer, hardware, logging)
- `sdft_trace.yaml` — SDFT with full reasoning trace as demonstration
- `sdft_answer.yaml` — SDFT with answer-only demonstration
- `sft_trace.yaml` — Standard SFT on full reasoning traces
- `sft_answer.yaml` — SFT on answer-only data
- `offline_distil.yaml` — Offline distillation (teacher-generated, off-policy)
- `grpo.yaml` — GRPO with outcome-based math reward
