# Analysis Scripts

Post-hoc analysis of trained checkpoints. All scripts take a checkpoint path
(containing both `theta_init.pt` and final weights) as input.

## Files

- `sparsity.py` — Compute parameter-level sparsity (`|θ_final - θ_init| ≤ 10⁻⁵`), overall and per-layer. Outputs CSV + summary stats.
- `rank.py` — SVD of update matrices `Δθ = θ_final - θ_init` per parameter matrix. Reports effective rank.
- `overlap.py` — Subnetwork overlap between two runs. Computes one-sided overlap metric from Mukherjee et al.
- `visualize.py` — Generate figures: sparsity bar charts, layerwise heatmaps, difficulty-stratified plots.
