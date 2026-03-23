"""
Compute parameter-level sparsity of fine-tuning updates.

Implements the sparsity metric from "Reinforcement Learning Finetunes Small
Subnetworks in Large Language Models" (arXiv:2505.11711):

    sparsity(θ⁰, θ¹) = 1 − ‖θ¹ − θ⁰‖₀ / n

where ‖·‖₀ counts elements with |Δ| > threshold (default 1e-5 for bfloat16).

Usage:
    # Single checkpoint
    python -m src.analysis.sparsity \
        --theta_init results/sft/numinamath/Qwen3-4B-Base/theta_init \
        --checkpoint results/sft/numinamath/Qwen3-4B-Base/checkpoint-100

    # All checkpoints in a run directory
    python -m src.analysis.sparsity \
        --run_dir results/sft/numinamath/Qwen3-4B-Base

    # Custom threshold
    python -m src.analysis.sparsity \
        --run_dir results/rl/Qwen3-4B-Instruct-2507 \
        --threshold 1e-8
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a state dict from a directory of safetensors files."""
    path = Path(path)
    safetensor_files = sorted(path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")

    state_dict = {}
    for f in safetensor_files:
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                state_dict[key] = st.get_tensor(key)
    return state_dict


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

# Regex to extract layer index and matrix type from parameter names.
# Covers LLaMA / Qwen / Mistral / OLMo naming conventions.
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

MATRIX_TYPE_PATTERNS = {
    "q_proj": re.compile(r"q_proj\.weight$"),
    "k_proj": re.compile(r"k_proj\.weight$"),
    "v_proj": re.compile(r"v_proj\.weight$"),
    "o_proj": re.compile(r"o_proj\.weight$"),
    "gate_proj": re.compile(r"gate_proj\.weight$"),
    "up_proj": re.compile(r"up_proj\.weight$"),
    "down_proj": re.compile(r"down_proj\.weight$"),
    "input_layernorm": re.compile(r"input_layernorm\.weight$"),
    "post_attention_layernorm": re.compile(r"post_attention_layernorm\.weight$"),
    "embed_tokens": re.compile(r"embed_tokens\.weight$"),
    "lm_head": re.compile(r"lm_head\.weight$"),
    "norm": re.compile(r"model\.norm\.weight$"),
}


def classify_param(name: str) -> tuple[int | None, str]:
    """Return (layer_index_or_None, matrix_type) for a parameter name."""
    layer_match = LAYER_RE.search(name)
    layer_idx = int(layer_match.group(1)) if layer_match else None

    for mtype, pattern in MATRIX_TYPE_PATTERNS.items():
        if pattern.search(name):
            return layer_idx, mtype

    return layer_idx, "other"


# ---------------------------------------------------------------------------
# Sparsity computation
# ---------------------------------------------------------------------------

def compute_sparsity(
    theta_init: dict[str, torch.Tensor],
    theta_final: dict[str, torch.Tensor],
    threshold: float = 1e-5,
    compute_rank: bool = True,
    rank_threshold: float = 1e-5,
    exact_zero: bool = False,
) -> dict:
    """Compute sparsity metrics between two state dicts.

    Returns a dict with:
        - global_sparsity: overall sparsity across all parameters
        - total_params, changed_params, unchanged_params
        - per_param: list of per-parameter results
        - per_layer: dict mapping layer_idx -> sparsity
        - per_matrix_type: dict mapping matrix_type -> sparsity
        - per_layer_matrix: dict mapping (layer_idx, matrix_type) -> sparsity
    """
    # Only compare parameters present in both
    common_keys = sorted(set(theta_init.keys()) & set(theta_final.keys()))
    if not common_keys:
        raise ValueError("No common parameter keys between theta_init and checkpoint")

    # should be zero
    init_only = set(theta_init.keys()) - set(theta_final.keys())
    final_only = set(theta_final.keys()) - set(theta_init.keys())
    if init_only:
        print(f"  Warning: {len(init_only)} keys only in theta_init (skipped)")
    if final_only:
        print(f"  Warning: {len(final_only)} keys only in checkpoint (skipped)")

    # Accumulators
    total_params = 0
    total_changed = 0

    # Per-group accumulators: {group_key: [total, changed]}
    layer_stats = defaultdict(lambda: [0, 0])
    matrix_type_stats = defaultdict(lambda: [0, 0])
    layer_matrix_stats = defaultdict(lambda: [0, 0])

    per_param = []

    for key in common_keys:
        w_init = theta_init[key].float()
        w_final = theta_final[key].float()

        if w_init.shape != w_final.shape:
            print(f"  Warning: shape mismatch for {key}: {w_init.shape} vs {w_final.shape}, skipping")
            continue

        delta = w_final - w_init
        n = delta.numel()
        if exact_zero:
            changed = (delta != 0).sum().item()
        else:
            changed = (delta.abs() > threshold).sum().item()
        sparsity = 1.0 - changed / n if n > 0 else 1.0

        total_params += n
        total_changed += changed

        layer_idx, mtype = classify_param(key)

        # Accumulate group stats
        if layer_idx is not None:
            layer_stats[layer_idx][0] += n
            layer_stats[layer_idx][1] += changed
            layer_matrix_stats[(layer_idx, mtype)][0] += n
            layer_matrix_stats[(layer_idx, mtype)][1] += changed

        matrix_type_stats[mtype][0] += n
        matrix_type_stats[mtype][1] += changed

        # Per-param record
        entry = {
            "name": key,
            "layer": layer_idx,
            "matrix_type": mtype,
            "num_params": n,
            "changed": changed,
            "sparsity": sparsity,
            "delta_l2": delta.norm().item(),
            "delta_linf": delta.abs().max().item(),
        }

        # TODO : check
        # Rank analysis for 2D weight matrices
        if compute_rank and delta.dim() == 2 and min(delta.shape) > 1:
            sv = torch.linalg.svdvals(delta)
            max_rank = min(delta.shape)
            # Effective rank: number of singular values > threshold
            effective_rank = (sv > rank_threshold).sum().item()
            entry["rank"] = effective_rank
            entry["max_rank"] = max_rank
            entry["rank_ratio"] = effective_rank / max_rank

        per_param.append(entry)

    # Build group summaries
    def _group_summary(stats: dict) -> dict:
        return {
            k: {
                "total_params": v[0],
                "changed": v[1],
                "sparsity": 1.0 - v[1] / v[0] if v[0] > 0 else 1.0,
            }
            for k, v in sorted(stats.items())
        }

    global_sparsity = 1.0 - total_changed / total_params if total_params > 0 else 1.0

    return {
        "global_sparsity": global_sparsity,
        "total_params": total_params,
        "changed_params": total_changed,
        "unchanged_params": total_params - total_changed,
        "threshold": None if exact_zero else threshold,
        "exact_zero": exact_zero,
        "per_param": per_param,
        "per_layer": _group_summary(layer_stats),
        "per_matrix_type": _group_summary(matrix_type_stats),
        "per_layer_matrix": {
            f"layer_{k[0]}_{k[1]}": v
            for k, v in _group_summary(layer_matrix_stats).items()
        },
    }


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def compute_overlap(
    theta_init: dict[str, torch.Tensor],
    theta_a: dict[str, torch.Tensor],
    theta_b: dict[str, torch.Tensor],
    threshold: float = 1e-5,
    exact_zero: bool = False,
) -> dict:
    """Compute subnetwork overlap between two fine-tuned checkpoints.

    Implements the one-sided overlap metrics from Section 3.2 of the paper:
        o_1 = |I_1 ∩ I_2| / |I_1|   (fraction of A's updates also updated in B)
        o_2 = |I_1 ∩ I_2| / |I_2|   (fraction of B's updates also updated in A)

    where I_1, I_2 are the sets of parameters with |delta| > threshold.

    Also computes the random baseline: o_{1,random} = |I_2| / n_total.
    """
    common_keys = sorted(
        set(theta_init.keys()) & set(theta_a.keys()) & set(theta_b.keys())
    )
    if not common_keys:
        raise ValueError("No common parameter keys across all three state dicts")

    total_params = 0
    changed_a = 0
    changed_b = 0
    changed_both = 0

    for key in common_keys:
        w_init = theta_init[key].float()
        w_a = theta_a[key].float()
        w_b = theta_b[key].float()

        if w_init.shape != w_a.shape or w_init.shape != w_b.shape:
            continue

        if exact_zero:
            mask_a = (w_a - w_init) != 0
            mask_b = (w_b - w_init) != 0
        else:
            mask_a = (w_a - w_init).abs() > threshold
            mask_b = (w_b - w_init).abs() > threshold

        n = mask_a.numel()
        total_params += n
        changed_a += mask_a.sum().item()
        changed_b += mask_b.sum().item()
        changed_both += (mask_a & mask_b).sum().item()

    sparsity_a = 1.0 - changed_a / total_params if total_params > 0 else 1.0
    sparsity_b = 1.0 - changed_b / total_params if total_params > 0 else 1.0
    o1 = changed_both / changed_a if changed_a > 0 else 0.0
    o2 = changed_both / changed_b if changed_b > 0 else 0.0
    o1_random = changed_b / total_params if total_params > 0 else 0.0
    o2_random = changed_a / total_params if total_params > 0 else 0.0

    return {
        "o1": o1,
        "o2": o2,
        "o1_random_baseline": o1_random,
        "o2_random_baseline": o2_random,
        "sparsity_a": sparsity_a,
        "sparsity_b": sparsity_b,
        "changed_a": changed_a,
        "changed_b": changed_b,
        "changed_both": changed_both,
        "total_params": total_params,
        "threshold": None if exact_zero else threshold,
        "exact_zero": exact_zero,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary(results: dict, checkpoint_name: str = ""):
    """Print a human-readable summary of sparsity results."""
    header = f"Sparsity Analysis"
    if checkpoint_name:
        header += f" — {checkpoint_name}"
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")

    print(f"\n  Global sparsity:  {results['global_sparsity']*100:.2f}%")
    print(f"  Total params:     {results['total_params']:,}")
    print(f"  Changed params:   {results['changed_params']:,}")
    print(f"  Unchanged params: {results['unchanged_params']:,}")
    if results.get("exact_zero"):
        print(f"  Sparsity mode:    exact zero (delta != 0)")
    else:
        print(f"  Threshold:        {results['threshold']:.0e}")

    # Per matrix type
    print(f"\n  {'Matrix Type':<30} {'Sparsity':>10} {'Changed':>12} {'Total':>12}")
    print(f"  {'-'*66}")
    for mtype, stats in sorted(results["per_matrix_type"].items()):
        print(
            f"  {mtype:<30} {stats['sparsity']*100:>9.2f}% "
            f"{stats['changed']:>11,} {stats['total_params']:>11,}"
        )

    # Per layer (condensed)
    print(f"\n  {'Layer':<10} {'Sparsity':>10} {'Changed':>12} {'Total':>12}")
    print(f"  {'-'*46}")
    for layer_idx, stats in sorted(results["per_layer"].items()):
        print(
            f"  {layer_idx:<10} {stats['sparsity']*100:>9.2f}% "
            f"{stats['changed']:>11,} {stats['total_params']:>11,}"
        )

    # Rank summary (if available)
    ranked = [p for p in results["per_param"] if "rank" in p]
    if ranked:
        avg_rank_ratio = sum(p["rank_ratio"] for p in ranked) / len(ranked)
        print(f"\n  Rank analysis ({len(ranked)} weight matrices):")
        print(f"    Average rank ratio: {avg_rank_ratio*100:.1f}% of max rank")
        min_rr = min(p["rank_ratio"] for p in ranked)
        max_rr = max(p["rank_ratio"] for p in ranked)
        print(f"    Range: {min_rr*100:.1f}% — {max_rr*100:.1f}%")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(run_dir: str) -> list[tuple[str, int]]:
    """Find all checkpoint-N directories in a run directory.

    Returns sorted list of (path, step) tuples.
    """
    run_path = Path(run_dir)
    checkpoints = []
    for d in run_path.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((str(d), int(m.group(1))))
    # Also check for 'final' directory
    final = run_path / "final"
    if final.is_dir():
        checkpoints.append((str(final), -1))  # -1 signals "final"

    return sorted(checkpoints, key=lambda x: x[1] if x[1] >= 0 else float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_single(
    theta_init_path: str,
    checkpoint_path: str,
    threshold: float = 1e-5,
    compute_rank: bool = True,
    rank_threshold: float = 1e-5,
    exact_zero: bool = False,
    output_dir: str | None = None,
) -> dict:
    """Analyze sparsity for a single checkpoint."""
    # assume checkpoint path is something like : ../../model-name/checkpoint-200
    checkpoint_name = checkpoint_path.split('/')
    checkpoint_name = checkpoint_name[-2] + '_' + checkpoint_name[-1]

    print(f"Loading theta_init from {theta_init_path} ...")
    theta_init = load_state_dict(theta_init_path)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    theta_final = load_state_dict(checkpoint_path)

    print(f"Computing sparsity (threshold={threshold:.0e}, rank={compute_rank}) ...")
    results = compute_sparsity(theta_init, theta_final, threshold=threshold, compute_rank=compute_rank, rank_threshold=rank_threshold, exact_zero=exact_zero)

    print_summary(results, checkpoint_name)

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sparsity_{checkpoint_name}.json")

    # Strip per_param for the saved summary (can be very large)
    summary = {k: v for k, v in results.items() if k != "per_param"}
    summary["theta_init"] = theta_init_path
    summary["checkpoint"] = checkpoint_path

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")

    return results


def analyze_run(
    run_dir: str,
    threshold: float = 1e-5,
    compute_rank: bool = True,
    rank_threshold: float = 1e-5,
    exact_zero: bool = False,
    output_dir: str | None = None,
) -> list[dict]:
    """Analyze sparsity for all checkpoints in a training run."""
    run_path = Path(run_dir)
    theta_init_path = str(run_path / "theta_init")

    if not Path(theta_init_path).exists():
        raise FileNotFoundError(f"No theta_init directory found in {run_dir}")

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {run_dir}")

    print(f"Found {len(checkpoints)} checkpoints in {run_dir}")

    # Load theta_init once
    print(f"Loading theta_init from {theta_init_path} ...")
    theta_init = load_state_dict(theta_init_path)

    all_results = []
    trajectory = []  # (step, global_sparsity) for trajectory summary
    prev_state_dict = None
    prev_step = None

    for ckpt_path, step in checkpoints:
        ckpt_name = Path(ckpt_path).name
        print(f"\n--- {ckpt_name} (step {step}) ---")

        print(f"Loading checkpoint from {ckpt_path} ...")
        theta_final = load_state_dict(ckpt_path)

        # Cumulative sparsity: sparsity(θ_init, θ_k)
        results = compute_sparsity(theta_init, theta_final, threshold=threshold, compute_rank=compute_rank, rank_threshold=rank_threshold, exact_zero=exact_zero)
        print_summary(results, ckpt_name)

        # Consecutive sparsity: sparsity(θ_{k-1}, θ_k)
        consecutive_sparsity = None
        if prev_state_dict is not None:
            print(f"  Computing consecutive sparsity (step {prev_step} -> {step}) ...")
            consec_results = compute_sparsity(
                prev_state_dict, theta_final, threshold=threshold, compute_rank=False,
            )
            consecutive_sparsity = consec_results["global_sparsity"]
            print(f"  Consecutive sparsity: {consecutive_sparsity*100:.2f}%")

        # Save per-checkpoint JSON
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"sparsity_{ckpt_name}.json")
        else:
            out_path = os.path.join(ckpt_path, "sparsity.json")

        summary = {k: v for k, v in results.items() if k != "per_param"}
        summary["theta_init"] = theta_init_path
        summary["checkpoint"] = ckpt_path
        summary["step"] = step
        if consecutive_sparsity is not None:
            summary["consecutive_sparsity"] = consecutive_sparsity
            summary["consecutive_from_step"] = prev_step

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {out_path}")

        all_results.append(results)
        traj_entry = {"step": step, "global_sparsity": results["global_sparsity"]}
        if consecutive_sparsity is not None:
            traj_entry["consecutive_sparsity"] = consecutive_sparsity
        trajectory.append(traj_entry)

        # Keep current checkpoint for next consecutive comparison
        del prev_state_dict
        prev_state_dict = theta_final
        prev_step = step

    del prev_state_dict

    # Save trajectory summary
    traj_out = os.path.join(output_dir or run_dir, "sparsity_trajectory.json")
    with open(traj_out, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\nSparsity trajectory saved to {traj_out}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute update sparsity: sparsity(θ⁰, θ¹) = 1 − ‖θ¹ − θ⁰‖₀/n"
    )

    # Mode: single checkpoint or full run
    parser.add_argument("--theta_init", type=str, default=None,
                        help="Path to theta_init directory (safetensors)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint directory")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to run directory (analyzes all checkpoints)")

    # Overlap mode
    parser.add_argument("--checkpoint_a", type=str, default=None,
                        help="First checkpoint for overlap comparison")
    parser.add_argument("--checkpoint_b", type=str, default=None,
                        help="Second checkpoint for overlap comparison")

    # Options
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Threshold for considering a parameter changed (default: 1e-5)")
    parser.add_argument("--no_rank", action="store_true",
                        help="Skip rank analysis (faster)")
    parser.add_argument("--rank_threshold", type=float, default=1e-5,
                        help="Threshold for counting effective rank via singular values (default: 1e-5)")
    parser.add_argument("--exact_zero", action="store_true",
                        help="Use exact zero check (delta != 0) instead of threshold-based sparsity")
    parser.add_argument("--output_dir", type=str, default="results/sparsity",
                        help="Output directory for JSON results")

    args = parser.parse_args()

    if args.theta_init and args.checkpoint_a and args.checkpoint_b:
        # Overlap mode
        print(f"Loading theta_init from {args.theta_init} ...")
        theta_init = load_state_dict(args.theta_init)
        print(f"Loading checkpoint A from {args.checkpoint_a} ...")
        theta_a = load_state_dict(args.checkpoint_a)
        print(f"Loading checkpoint B from {args.checkpoint_b} ...")
        theta_b = load_state_dict(args.checkpoint_b)

        overlap = compute_overlap(theta_init, theta_a, theta_b, threshold=args.threshold, exact_zero=args.exact_zero)

        print(f"\n{'='*70}")
        print(f"Subnetwork Overlap Analysis")
        print(f"{'='*70}")
        print(f"  Checkpoint A:       {args.checkpoint_a}")
        print(f"  Checkpoint B:       {args.checkpoint_b}")
        print(f"  Sparsity A:         {overlap['sparsity_a']*100:.2f}%")
        print(f"  Sparsity B:         {overlap['sparsity_b']*100:.2f}%")
        print(f"  o1 (A covered by B): {overlap['o1']*100:.2f}%  (random baseline: {overlap['o1_random_baseline']*100:.2f}%)")
        print(f"  o2 (B covered by A): {overlap['o2']*100:.2f}%  (random baseline: {overlap['o2_random_baseline']*100:.2f}%)")
        print(f"  Changed in both:    {overlap['changed_both']:,}")
        if overlap.get("exact_zero"):
            print(f"  Sparsity mode:      exact zero (delta != 0)")
        else:
            print(f"  Threshold:          {overlap['threshold']:.0e}")
        print(f"{'='*70}\n")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, "overlap.json")
            overlap["checkpoint_a"] = args.checkpoint_a
            overlap["checkpoint_b"] = args.checkpoint_b
            overlap["theta_init"] = args.theta_init
            with open(out_path, "w") as f:
                json.dump(overlap, f, indent=2)
            print(f"Saved overlap results to {out_path}")
    elif args.run_dir:
        analyze_run(
            run_dir=args.run_dir,
            threshold=args.threshold,
            compute_rank=not args.no_rank,
            rank_threshold=args.rank_threshold,
            exact_zero=args.exact_zero,
            output_dir=args.output_dir,
        )
    elif args.theta_init and args.checkpoint:
        analyze_single(
            theta_init_path=args.theta_init,
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
            compute_rank=not args.no_rank,
            rank_threshold=args.rank_threshold,
            exact_zero=args.exact_zero,
            output_dir=args.output_dir,
        )
    else:
        parser.error(
            "Provide one of:\n"
            "  --run_dir (analyze all checkpoints)\n"
            "  --theta_init + --checkpoint (single checkpoint)\n"
            "  --theta_init + --checkpoint_a + --checkpoint_b (overlap)"
        )


if __name__ == "__main__":
    main()
