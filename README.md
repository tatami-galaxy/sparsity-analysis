# Sparsity Analysis of Self-Distillation Fine-Tuning

Analyzing parameter update sparsity in SDFT vs RL vs SFT, bridging two key findings:
- **SDFT** (Shenfeld et al., 2026): Self-distillation enables continual learning via on-policy KL minimization against an ICL-conditioned teacher
- **RL Subnetworks** (Mukherjee et al., 2025): RL naturally updates only 5-30% of parameters, driven by on-distribution training data

## Research Questions

1. How sparse are SDFT parameter updates compared to RL (GRPO) and SFT?
2. Does demonstration richness (full trace vs answer-only) affect SDFT sparsity?
3. Does SDFT sparsity vary with task difficulty (MATH Levels 1-5)?
4. Do SDFT and RL modify overlapping subnetworks on the same task?
5. Does on-policy generation (vs offline distillation) causally drive sparsity?

## Experimental Matrix

| Method | Training signal | Control variable |
|---|---|---|
| SFT (trace) | Full solutions | Off-policy baseline |
| SFT (answer) | Answer only | Weak off-policy baseline |
| SDFT (trace) | ICL teacher w/ solution | **Primary experiment** |
| SDFT (answer) | ICL teacher w/ answer only | Demonstration ablation |
| Offline distillation | Teacher outputs, off-policy | On-policy ablation |
| GRPO | Outcome reward (correct/incorrect) | RL reference point |

## Setup

- **Model**: Llama-3.1-8B-Instruct
- **Dataset**: MATH (Hendrycks), stratified by difficulty level (1-5)
- **Sparsity metric**: `|θ_final - θ_init| ≤ 10⁻⁵` (bfloat16 precision, per Mukherjee et al.)

## Project Structure

```
sparsity-analysis/
├── configs/             # Training configs for each method
├── data/                # Dataset loading and preprocessing
├── src/
│   ├── training/        # Training scripts (SDFT, SFT, GRPO, offline distil)
│   ├── analysis/        # Post-hoc sparsity, rank, and overlap analysis
│   └── utils/           # Shared utilities (weight snapshots, metrics, logging)
├── scripts/             # Shell scripts to launch experiments
├── results/
│   ├── checkpoints/     # Saved model weights
│   ├── figures/         # Generated plots
│   └── logs/            # Training logs
└── README.md
```

## Usage

```bash
# 1. Prepare data
python -m data.prepare_math

# 2. Run training (see scripts/ for launch commands)
bash scripts/run_sdft_trace.sh
bash scripts/run_sft_trace.sh
bash scripts/run_grpo.sh

# 3. Analyze sparsity
python -m src.analysis.sparsity --checkpoint results/checkpoints/<run>
python -m src.analysis.rank --checkpoint results/checkpoints/<run>
python -m src.analysis.overlap --runs results/checkpoints/sdft,results/checkpoints/grpo
```

## References

- Shenfeld et al. "Self-Distillation Enables Continual Learning" (arXiv:2601.19897)
- Mukherjee et al. "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models" (arXiv:2505.11711)
- Hendrycks et al. "Measuring Mathematical Problem Solving With the MATH Dataset"
