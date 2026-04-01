"""
Compute token-level KL divergence between base and fine-tuned models
using fixed dataset completions (teacher-forced).

Implements the forward KL metric from "RL's Razor: Why Online Reinforcement
Learning Forgets Less" (arXiv:2509.04259):

    D_KL(π_base ‖ π_ft) = Σ_t Σ_v π_base(v|x,y<t) log [π_base(v|x,y<t) / π_ft(v|x,y<t)]

Evaluated per-token over the full vocabulary, averaged across a dataset of
prompts with reference completions. Datasets are loaded and formatted using
the same logic as training (src.train.train_sft), ensuring prompts and
completions match exactly what the model was trained on.

Both models are loaded simultaneously on separate GPUs (base on cuda:0,
checkpoint on cuda:1) and KL is computed on-the-fly per sample — no disk
caching needed. Requires CUDA_VISIBLE_DEVICES to expose 2 GPUs.

Usage:
    # Single checkpoint
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_fixed_data \
        --base_model Qwen/Qwen3-4B-Base \
        --checkpoint results/sft/numinamath/Qwen3-4B-Base/checkpoint-100 \
        --dataset numinamath \
        --max_samples 500

    # All checkpoints in a run directory
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_fixed_data \
        --run_dir results/sft/numinamath/Qwen3-4B-Base \
        --dataset numinamath \
        --max_samples 500

    # With custom chat template (e.g. base model using instruct template)
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_fixed_data \
        --base_model Qwen/Qwen3-4B \
        --checkpoint results/sft/deepmath/Qwen3-4B/checkpoint-200 \
        --dataset deepmath \
        --chat_template_model Qwen/Qwen3-4B-Instruct \
        --max_samples 500
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train.train_sft import (
    SYSTEM_PROMPT,
    load_numinamath,
    load_deepmath,
    load_competition_math,
    format_sft,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_prompt_completion_pairs(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_samples: int | None = None,
    max_length: int = 2048,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Load (prompt, completion) pairs using the same loaders as training.

    Formats examples as chat messages via format_sft(), then uses
    apply_chat_template to produce the exact prompt and completion strings
    the model was trained on.

    Returns list of dicts with keys "prompt_ids" (tokenized prompt as tensor)
    and "completion_ids" (tokenized completion as tensor).
    """
    # Load full dataset, filter by length, then truncate to max_samples
    # so we always get max_samples usable examples (if available)
    if dataset == "numinamath":
        ds = load_numinamath(max_samples=None, seed=seed)
    elif dataset == "deepmath":
        ds = load_deepmath(max_samples=None, seed=seed)
    elif dataset == "competition_math":
        ds = load_competition_math(max_samples=None, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            "Supported: numinamath, deepmath, competition_math"
        )

    # Format into chat messages (same as SFT training)
    ds = ds.map(format_sft, remove_columns=ds.column_names, num_proc=4)

    # Filter by sequence length first
    pre_filter = len(ds)
    ds = ds.filter(
        lambda x: len(tokenizer.apply_chat_template(x["messages"], tokenize=True)) <= max_length,
        num_proc=4,
    )
    if len(ds) < pre_filter:
        print(f"  Filtered by max_length={max_length}: {pre_filter} -> {len(ds)}")

    # Then truncate to max_samples
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"  Loaded {len(ds)} formatted examples from {dataset}")

    # Convert to tokenized prompt/completion pairs
    pairs = []
    for example in ds:
        messages = example["messages"]

        # Prompt = system + user turns (with generation prompt appended)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        # Full sequence = all messages
        full_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

        # Completion ids = everything after the prompt
        completion_ids = full_ids[len(prompt_ids):]

        if len(completion_ids) == 0:
            continue

        pairs.append({
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        })

    return pairs


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_pairs(
    pairs: list[dict],
    max_length: int = 2048,
) -> list[dict[str, torch.Tensor]]:
    """Convert pre-tokenized prompt/completion pairs into tensors.

    Returns list of dicts with "input_ids" (tensor) and "prompt_len" (int).
    """
    tokenized = []
    for pair in pairs:
        prompt_ids = pair["prompt_ids"]
        completion_ids = pair["completion_ids"]

        full_ids = prompt_ids + completion_ids
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        prompt_len = min(len(prompt_ids), max_length)

        tokenized.append({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "prompt_len": prompt_len,
        })
    return tokenized


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load a model onto a specific GPU device."""
    print(f"  Loading model from {model_path} -> {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Online KL computation (no disk caching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_online(
    base_model: AutoModelForCausalLM,
    ft_model: AutoModelForCausalLM,
    tokenized_data: list[dict],
) -> dict:
    """Compute per-token forward KL divergence on-the-fly.

    Both models are already loaded on separate GPUs. For each sample:
      1. Forward pass through both models
      2. Compute log-softmax on completion tokens
      3. Compute per-token KL and discard full vocab distributions

    KL(π_base ‖ π_ft) = Σ_v π_base(v) [log π_base(v) - log π_ft(v)]
    Only computed over completion tokens (after prompt_len).
    """
    base_device = next(base_model.parameters()).device
    ft_device = next(ft_model.parameters()).device

    per_sample = []
    all_token_kls = []

    for i, sample in enumerate(tokenized_data):
        prompt_len = sample["prompt_len"]
        input_ids = sample["input_ids"].unsqueeze(0)

        # Forward pass on both models (each on its own GPU)
        base_out = base_model(input_ids=input_ids.to(base_device))
        ft_out = ft_model(input_ids=input_ids.to(ft_device))

        # logits[:, :-1, :] predicts tokens at positions [1, 2, ..., seq_len-1]
        seq_len = base_out.logits.shape[1] - 1

        # logprobs[t] predicts token t+1, so logprobs[prompt_len-1] predicts
        # the first completion token
        start = max(prompt_len - 1, 0)
        if start >= seq_len:
            per_sample.append({
                "index": i,
                "num_completion_tokens": 0,
                "mean_kl": 0.0,
                "max_kl": 0.0,
                "sum_kl": 0.0,
            })
            continue

        # Only compute log-softmax on the completion slice to save memory
        base_lp = F.log_softmax(
            base_out.logits[0, start:seq_len, :].float(), dim=-1,
        )
        ft_lp = F.log_softmax(
            ft_out.logits[0, start:seq_len, :].float().to(base_device), dim=-1,
        )

        base_probs = base_lp.exp()
        token_kl = (base_probs * (base_lp - ft_lp)).sum(dim=-1)
        token_kl = token_kl.clamp(min=0.0).cpu()

        num_tokens = token_kl.shape[0]
        per_sample.append({
            "index": i,
            "num_completion_tokens": num_tokens,
            "mean_kl": token_kl.mean().item(),
            "max_kl": token_kl.max().item(),
            "sum_kl": token_kl.sum().item(),
        })
        all_token_kls.append(token_kl)

        if (i + 1) % 50 == 0 or i == len(tokenized_data) - 1:
            print(f"    Processed {i + 1}/{len(tokenized_data)} samples")

    if all_token_kls:
        flat_kls = torch.cat(all_token_kls)
        aggregate = {
            "mean_kl": flat_kls.mean().item(),
            "median_kl": flat_kls.median().item(),
            "max_kl": flat_kls.max().item(),
            "std_kl": flat_kls.std().item(),
            "total_completion_tokens": flat_kls.shape[0],
            "num_samples": len(per_sample),
        }
        sample_means = [s["mean_kl"] for s in per_sample if s["num_completion_tokens"] > 0]
        if sample_means:
            aggregate["mean_sample_kl"] = sum(sample_means) / len(sample_means)
    else:
        aggregate = {
            "mean_kl": 0.0,
            "median_kl": 0.0,
            "max_kl": 0.0,
            "std_kl": 0.0,
            "total_completion_tokens": 0,
            "num_samples": len(per_sample),
        }

    return {
        "aggregate": aggregate,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary(results: dict, checkpoint_name: str = ""):
    header = "KL Divergence Analysis (fixed data)"
    if checkpoint_name:
        header += f" — {checkpoint_name}"
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")

    agg = results["aggregate"]
    print(f"\n  Mean KL (per-token):     {agg['mean_kl']:.6f}")
    print(f"  Median KL (per-token):   {agg['median_kl']:.6f}")
    print(f"  Max KL (per-token):      {agg['max_kl']:.6f}")
    print(f"  Std KL (per-token):      {agg['std_kl']:.6f}")
    print(f"  Mean KL (per-sample):    {agg.get('mean_sample_kl', 0.0):.6f}")
    print(f"  Total completion tokens: {agg['total_completion_tokens']:,}")
    print(f"  Num samples:             {agg['num_samples']}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(run_dir: str) -> list[tuple[str, int]]:
    run_path = Path(run_dir)
    checkpoints = []
    for d in run_path.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((str(d), int(m.group(1))))
    final = run_path / "final"
    if final.is_dir():
        checkpoints.append((str(final), -1))
    return sorted(checkpoints, key=lambda x: x[1] if x[1] >= 0 else float("inf"))


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------

def analyze_single(
    base_model: str,
    checkpoint_path: str,
    dataset: str,
    max_samples: int | None = None,
    max_length: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> dict:
    """Compute KL divergence between a base model and a single checkpoint."""
    checkpoint_name = checkpoint_path.replace("/", "_")

    print(f"Loading tokenizer from {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(chat_template_model)
        tokenizer.chat_template = template_tok.chat_template
        print(f"  Using chat template from: {chat_template_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset} ...")
    pairs = load_prompt_completion_pairs(
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
        seed=seed,
    )

    tokenized = tokenize_pairs(pairs, max_length=max_length)
    print(f"  Tokenized {len(tokenized)} samples")

    print(f"\nLoading both models ...")
    base_mod = load_model(base_model, device=torch.device("cuda:0"), dtype=dtype)
    ft_mod = load_model(checkpoint_path, device=torch.device("cuda:1"), dtype=dtype)

    print(f"\nComputing KL divergence ...")
    results = compute_kl_online(base_mod, ft_mod, tokenized)

    del base_mod, ft_mod
    torch.cuda.empty_cache()

    print_summary(results, checkpoint_name)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"kl_{checkpoint_name}.json")
        save_data = {
            "base_model": base_model,
            "checkpoint": checkpoint_path,
            "dataset": dataset,
            "max_samples": max_samples,
            "max_length": max_length,
            "num_samples": len(pairs),
            **results["aggregate"],
            "per_sample": results["per_sample"],
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved results to {out_path}")

    return results


def analyze_run(
    run_dir: str,
    dataset: str,
    max_samples: int | None = None,
    max_length: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> list[dict]:
    """Compute KL divergence for all checkpoints in a training run.

    Base model stays loaded on cuda:0 across all checkpoints.
    Each checkpoint is loaded onto cuda:1, KL is computed, then freed.
    """
    run_path = Path(run_dir)

    theta_init_path = str(run_path / "theta_init")
    config_path = run_path / "config.json"

    if Path(theta_init_path).exists():
        base_model = theta_init_path
    elif config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        base_model = cfg.get("_name_or_path", cfg.get("model_name_or_path"))
        if not base_model:
            raise ValueError(f"Cannot determine base model from {config_path}")
    else:
        raise FileNotFoundError(
            f"No theta_init or config.json in {run_dir}. "
            "Provide --base_model explicitly."
        )

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {run_dir}")
    print(f"Found {len(checkpoints)} checkpoints in {run_dir}")

    print(f"Loading tokenizer from {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(chat_template_model)
        tokenizer.chat_template = template_tok.chat_template
        print(f"  Using chat template from: {chat_template_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset} ...")
    pairs = load_prompt_completion_pairs(
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
        seed=seed,
    )

    tokenized = tokenize_pairs(pairs, max_length=max_length)

    all_results = []
    trajectory = []

    print(f"\nLoading base model (stays resident on cuda:0) ...")
    base_mod = load_model(base_model, device=torch.device("cuda:0"), dtype=dtype)

    for ckpt_path, step in checkpoints:
        ckpt_name = Path(ckpt_path).name
        print(f"\n--- {ckpt_name} (step {step}) ---")

        ft_mod = load_model(ckpt_path, device=torch.device("cuda:1"), dtype=dtype)

        print(f"  Computing KL divergence ...")
        results = compute_kl_online(base_mod, ft_mod, tokenized)

        del ft_mod
        torch.cuda.empty_cache()

        print_summary(results, ckpt_name)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"kl_{ckpt_name}.json")
        else:
            out_path = os.path.join(ckpt_path, "kl_divergence.json")

        save_data = {
            "base_model": base_model,
            "checkpoint": ckpt_path,
            "step": step,
            "dataset": dataset,
            "max_samples": max_samples,
            "max_length": max_length,
            "num_samples": len(pairs),
            **results["aggregate"],
            "per_sample": results["per_sample"],
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  Saved to {out_path}")

        all_results.append(results)
        trajectory.append({
            "step": step,
            "mean_kl": results["aggregate"]["mean_kl"],
            "median_kl": results["aggregate"]["median_kl"],
            "max_kl": results["aggregate"]["max_kl"],
            "mean_sample_kl": results["aggregate"].get("mean_sample_kl", 0.0),
        })

    del base_mod
    torch.cuda.empty_cache()

    traj_out = os.path.join(output_dir or run_dir, "kl_trajectory.json")
    with open(traj_out, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\nKL trajectory saved to {traj_out}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute forward KL divergence D_KL(π_base ‖ π_ft) using fixed dataset completions"
    )

    # Model paths
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path or HF name of the base model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single fine-tuned checkpoint")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to run directory (analyzes all checkpoints)")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["numinamath", "deepmath", "competition_math"],
                        help="Training dataset (uses same loader as train_sft.py)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")

    # Options
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--output_dir", type=str, default="results/kl_divergence_fixed_data",
                        help="Output directory for JSON results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset sampling")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from (e.g. instruct variant for a base model)")

    args = parser.parse_args()

    dtype = torch.bfloat16

    if args.run_dir:
        analyze_run(
            run_dir=args.run_dir,
            dataset=args.dataset,
            max_samples=args.max_samples,
            max_length=args.max_length,
            dtype=dtype,
            output_dir=args.output_dir,
            seed=args.seed,
            chat_template_model=args.chat_template_model,
        )
    elif args.base_model and args.checkpoint:
        analyze_single(
            base_model=args.base_model,
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            max_samples=args.max_samples,
            max_length=args.max_length,
            dtype=dtype,
            output_dir=args.output_dir,
            seed=args.seed,
            chat_template_model=args.chat_template_model,
        )
    else:
        parser.error(
            "Provide one of:\n"
            "  --run_dir (analyze all checkpoints, base model auto-detected from theta_init)\n"
            "  --base_model + --checkpoint (single checkpoint)"
        )


if __name__ == "__main__":
    main()
