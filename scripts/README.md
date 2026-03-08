# Launch Scripts

Shell scripts to run experiments. Each script loads the appropriate config
and launches training with accelerate/torchrun.

## Files

- `run_sdft_trace.sh` — SDFT with full reasoning trace demonstrations
- `run_sdft_answer.sh` — SDFT with answer-only demonstrations
- `run_sft_trace.sh` — SFT on full reasoning traces
- `run_sft_answer.sh` — SFT on answer-only data
- `run_offline_distil.sh` — Offline distillation baseline
- `run_grpo.sh` — GRPO with math outcome reward
- `run_all.sh` — Sequential launcher for all experiments
- `run_analysis.sh` — Run all post-hoc analyses on completed checkpoints
