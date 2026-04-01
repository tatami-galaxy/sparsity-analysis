"""
Compute token-level KL divergence between base and fine-tuned models
using model-generated completions.

For each model (base and fine-tuned), generates completions from dataset
prompts, then teacher-forces both models through each set of completions
to measure forward KL divergence on the model's own output distribution.

Datasets are loaded using the same logic as training (src.train.train_sft),
and prompts are formatted with chat templates to match training exactly.

This gives two KL measurements:
    - KL_base: D_KL(π_base ‖ π_ft) on base model completions
    - KL_ft:   D_KL(π_base ‖ π_ft) on fine-tuned model completions

Uses sequential processing to keep peak GPU memory at ~1 model:
    Phase 1: Load base model, generate completions, cache logprobs on
             base completions, free model.
    Phase 2: Load fine-tuned model, generate completions, cache logprobs
             on ft completions, free model.
    Phase 3: Load base model again, cache logprobs on ft completions, free.
    Phase 4: Load fine-tuned model again, cache logprobs on base completions, free.
    Phase 5: Compute KL from cached logprobs.

Usage:
    # Single checkpoint
    python -m src.analysis.kl_divergence_generation \
        --base_model Qwen/Qwen3-4B-Base \
        --checkpoint results/sft/numinamath/Qwen3-4B-Base/checkpoint-100 \
        --dataset numinamath \
        --max_samples 200

    # All checkpoints in a run directory
    python -m src.analysis.kl_divergence_generation \
        --run_dir results/sft/numinamath/Qwen3-4B-Base \
        --dataset numinamath \
        --max_samples 200

    # With custom chat template and generation parameters
    python -m src.analysis.kl_divergence_generation \
        --base_model Qwen/Qwen3-4B \
        --checkpoint results/rl/deepmath/Qwen3-4B/checkpoint-200 \
        --dataset deepmath \
        --chat_template_model Qwen/Qwen3-4B-Instruct \
        --max_samples 200 \
        --max_new_tokens 512 \
        --temperature 0.7
"""

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train.train_sft import (
    SYSTEM_PROMPT,
    load_numinamath,
    load_deepmath,
    load_competition_math,
)


# ---------------------------------------------------------------------------
# Dataset loading (prompts only)
# ---------------------------------------------------------------------------

def load_prompts(
    dataset: str,
    tokenizer: AutoTokenizer,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Load prompts using the same loaders as training, formatted with chat template.

    Returns list of chat-templated prompt strings (system + user + generation prompt).
    """
    if dataset == "numinamath":
        ds = load_numinamath(max_samples=max_samples, seed=seed)
    elif dataset == "deepmath":
        ds = load_deepmath(max_samples=max_samples, seed=seed)
    elif dataset == "competition_math":
        ds = load_competition_math(max_samples=max_samples, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            "Supported: numinamath, deepmath, competition_math"
        )

    print(f"  Loaded {len(ds)} examples from {dataset}")

    prompts = []
    for example in ds:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt_str)

    return prompts


# ---------------------------------------------------------------------------
# Generation + logprob caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_cache_logprobs(
    model_path: str,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    cache_dir: str,
    generation_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[list[str], list[str]]:
    """Load model, generate completions from prompts, cache logprobs, free model.

    For each prompt:
        1. Generate a completion.
        2. Teacher-force the full (prompt + generation) sequence and save
           log-probs over the vocabulary at each position.

    Returns (generation_paths, logprob_cache_paths).
    """
    print(f"  Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    gen_paths = []
    cache_paths = []

    for i, prompt in enumerate(prompts):
        prompt_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_enc["input_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]

        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        full_ids = output_ids[0]

        # Save generation metadata
        gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
        gen_data = {
            "index": i,
            "prompt": prompt,
            "generation": gen_text,
            "prompt_len": prompt_len,
            "total_len": full_ids.shape[0],
            "gen_len": full_ids.shape[0] - prompt_len,
        }
        gen_path = os.path.join(generation_dir, f"gen_{i:06d}.json")
        with open(gen_path, "w") as f:
            json.dump(gen_data, f, indent=2)
        gen_paths.append(gen_path)

        # Teacher-force through full sequence to get logprobs
        outputs = model(input_ids=full_ids.unsqueeze(0))
        logprobs = F.log_softmax(outputs.logits[0, :-1, :].float(), dim=-1)

        cache_path = os.path.join(cache_dir, f"logprobs_{i:06d}.pt")
        torch.save(logprobs.cpu(), cache_path)
        cache_paths.append(cache_path)

        # Save prompt_len and input_ids for later KL computation
        meta_path = os.path.join(cache_dir, f"meta_{i:06d}.pt")
        torch.save({"input_ids": full_ids.cpu(), "prompt_len": prompt_len}, meta_path)

        if (i + 1) % 50 == 0 or i == len(prompts) - 1:
            print(f"    Generated + cached {i + 1}/{len(prompts)} samples")

    del model
    torch.cuda.empty_cache()

    return gen_paths, cache_paths


@torch.no_grad()
def cache_logprobs_for_sequences(
    model_path: str,
    sequence_cache_dir: str,
    output_cache_dir: str,
    num_samples: int,
    dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    """Load a model and compute logprobs for pre-existing sequences.

    Reads input_ids from meta files in sequence_cache_dir, computes logprobs
    with the given model, saves to output_cache_dir.
    """
    print(f"  Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    cache_paths = []
    for i in range(num_samples):
        meta_path = os.path.join(sequence_cache_dir, f"meta_{i:06d}.pt")
        meta = torch.load(meta_path, weights_only=True)
        input_ids = meta["input_ids"].unsqueeze(0).to(model.device)

        outputs = model(input_ids=input_ids)
        logprobs = F.log_softmax(outputs.logits[0, :-1, :].float(), dim=-1)

        cache_path = os.path.join(output_cache_dir, f"logprobs_{i:06d}.pt")
        torch.save(logprobs.cpu(), cache_path)
        cache_paths.append(cache_path)

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"    Cached {i + 1}/{num_samples} samples")

    del model
    torch.cuda.empty_cache()

    return cache_paths


# ---------------------------------------------------------------------------
# KL computation
# ---------------------------------------------------------------------------

def compute_kl_from_caches(
    base_cache_paths: list[str],
    ft_cache_paths: list[str],
    sequence_cache_dir: str,
) -> dict:
    """Compute per-token forward KL divergence from cached log-probs.

    KL(π_base ‖ π_ft) = Σ_v π_base(v) [log π_base(v) - log π_ft(v)]

    Only computed over completion tokens (after prompt_len).
    """
    per_sample = []
    all_token_kls = []

    for i, (base_path, ft_path) in enumerate(zip(base_cache_paths, ft_cache_paths)):
        base_logprobs = torch.load(base_path, weights_only=True)
        ft_logprobs = torch.load(ft_path, weights_only=True)

        meta_path = os.path.join(sequence_cache_dir, f"meta_{i:06d}.pt")
        meta = torch.load(meta_path, weights_only=True)
        prompt_len = meta["prompt_len"]
        seq_len = base_logprobs.shape[0]

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

        base_lp = base_logprobs[start:]
        ft_lp = ft_logprobs[start:]

        base_probs = base_lp.exp()
        token_kl = (base_probs * (base_lp - ft_lp)).sum(dim=-1)
        token_kl = token_kl.clamp(min=0.0)

        num_tokens = token_kl.shape[0]
        per_sample.append({
            "index": i,
            "num_completion_tokens": num_tokens,
            "mean_kl": token_kl.mean().item(),
            "max_kl": token_kl.max().item(),
            "sum_kl": token_kl.sum().item(),
        })
        all_token_kls.append(token_kl)

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

def print_summary(results: dict, label: str = ""):
    header = "KL Divergence Analysis (generation-based)"
    if label:
        header += f" — {label}"
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")

    for source in ["base_completions", "ft_completions"]:
        if source not in results:
            continue
        agg = results[source]["aggregate"]
        print(f"\n  [{source}]")
        print(f"    Mean KL (per-token):     {agg['mean_kl']:.6f}")
        print(f"    Median KL (per-token):   {agg['median_kl']:.6f}")
        print(f"    Max KL (per-token):      {agg['max_kl']:.6f}")
        print(f"    Std KL (per-token):      {agg['std_kl']:.6f}")
        print(f"    Mean KL (per-sample):    {agg.get('mean_sample_kl', 0.0):.6f}")
        print(f"    Total completion tokens: {agg['total_completion_tokens']:,}")
        print(f"    Num samples:             {agg['num_samples']}")

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
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> dict:
    """Compute KL divergence using model-generated completions.

    Phase 1: Base model generates completions + caches own logprobs.
    Phase 2: Fine-tuned model generates completions + caches own logprobs.
    Phase 3: Base model caches logprobs on ft completions.
    Phase 4: Fine-tuned model caches logprobs on base completions.
    Phase 5: Compute KL from caches.
    """
    checkpoint_name = Path(checkpoint_path).name

    print(f"Loading tokenizer from {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(chat_template_model)
        tokenizer.chat_template = template_tok.chat_template
        print(f"  Using chat template from: {chat_template_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset} ...")
    prompts = load_prompts(
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=max_samples,
        seed=seed,
    )
    print(f"  {len(prompts)} chat-formatted prompts")

    num_samples = len(prompts)

    with tempfile.TemporaryDirectory(prefix="kl_gen_cache_") as cache_root:
        base_gen_dir = os.path.join(cache_root, "base_gen")
        base_lp_on_base_dir = os.path.join(cache_root, "base_lp_on_base")
        ft_gen_dir = os.path.join(cache_root, "ft_gen")
        ft_lp_on_ft_dir = os.path.join(cache_root, "ft_lp_on_ft")
        base_lp_on_ft_dir = os.path.join(cache_root, "base_lp_on_ft")
        ft_lp_on_base_dir = os.path.join(cache_root, "ft_lp_on_base")
        for d in [base_gen_dir, base_lp_on_base_dir, ft_gen_dir, ft_lp_on_ft_dir,
                  base_lp_on_ft_dir, ft_lp_on_base_dir]:
            os.makedirs(d)

        # Phase 1: Base model — generate + cache logprobs on own completions
        print(f"\nPhase 1: Base model — generate + cache logprobs ...")
        base_gen_paths, base_lp_on_base_paths = generate_and_cache_logprobs(
            base_model, prompts, tokenizer, base_lp_on_base_dir, base_gen_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, dtype=dtype,
        )

        # Phase 2: Fine-tuned model — generate + cache logprobs on own completions
        print(f"\nPhase 2: Fine-tuned model — generate + cache logprobs ...")
        ft_gen_paths, ft_lp_on_ft_paths = generate_and_cache_logprobs(
            checkpoint_path, prompts, tokenizer, ft_lp_on_ft_dir, ft_gen_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, dtype=dtype,
        )

        # Phase 3: Base model — cache logprobs on ft completions
        print(f"\nPhase 3: Base model — logprobs on fine-tuned completions ...")
        base_lp_on_ft_paths = cache_logprobs_for_sequences(
            base_model, ft_lp_on_ft_dir, base_lp_on_ft_dir, num_samples, dtype=dtype,
        )

        # Phase 4: Fine-tuned model — cache logprobs on base completions
        print(f"\nPhase 4: Fine-tuned model — logprobs on base completions ...")
        ft_lp_on_base_paths = cache_logprobs_for_sequences(
            checkpoint_path, base_lp_on_base_dir, ft_lp_on_base_dir, num_samples, dtype=dtype,
        )

        # Phase 5: Compute KL
        print(f"\nPhase 5: Computing KL divergence ...")

        # KL on base completions: D_KL(π_base ‖ π_ft) on sequences from π_base
        kl_base = compute_kl_from_caches(
            base_lp_on_base_paths, ft_lp_on_base_paths, base_lp_on_base_dir,
        )
        # KL on ft completions: D_KL(π_base ‖ π_ft) on sequences from π_ft
        kl_ft = compute_kl_from_caches(
            base_lp_on_ft_paths, ft_lp_on_ft_paths, ft_lp_on_ft_dir,
        )

    results = {
        "base_completions": kl_base,
        "ft_completions": kl_ft,
    }

    print_summary(results, checkpoint_name)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"kl_gen_{checkpoint_name}.json")
        save_data = {
            "base_model": base_model,
            "checkpoint": checkpoint_path,
            "dataset": dataset,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_samples": num_samples,
            "base_completions": kl_base["aggregate"],
            "ft_completions": kl_ft["aggregate"],
            "base_completions_per_sample": kl_base["per_sample"],
            "ft_completions_per_sample": kl_ft["per_sample"],
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved results to {out_path}")

    return results


def analyze_run(
    run_dir: str,
    dataset: str,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> list[dict]:
    """Compute generation-based KL for all checkpoints in a training run.

    Base model generations + logprobs are cached once (phases 1 & 3 reuse).
    For each checkpoint, only the fine-tuned model's phases run fresh.
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
    prompts = load_prompts(
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=max_samples,
        seed=seed,
    )
    print(f"  {len(prompts)} chat-formatted prompts")

    num_samples = len(prompts)
    all_results = []
    trajectory = []

    with tempfile.TemporaryDirectory(prefix="kl_gen_cache_") as cache_root:
        # Phase 1 (one-time): Base model — generate + cache logprobs
        base_gen_dir = os.path.join(cache_root, "base_gen")
        base_lp_on_base_dir = os.path.join(cache_root, "base_lp_on_base")
        os.makedirs(base_gen_dir)
        os.makedirs(base_lp_on_base_dir)

        print(f"\nBase model — generate + cache logprobs (one-time) ...")
        base_gen_paths, base_lp_on_base_paths = generate_and_cache_logprobs(
            base_model, prompts, tokenizer, base_lp_on_base_dir, base_gen_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, dtype=dtype,
        )

        for ckpt_path, step in checkpoints:
            ckpt_name = Path(ckpt_path).name
            print(f"\n{'='*50}")
            print(f"--- {ckpt_name} (step {step}) ---")
            print(f"{'='*50}")

            ft_gen_dir = os.path.join(cache_root, f"ft_gen_{ckpt_name}")
            ft_lp_on_ft_dir = os.path.join(cache_root, f"ft_lp_on_ft_{ckpt_name}")
            base_lp_on_ft_dir = os.path.join(cache_root, f"base_lp_on_ft_{ckpt_name}")
            ft_lp_on_base_dir = os.path.join(cache_root, f"ft_lp_on_base_{ckpt_name}")
            for d in [ft_gen_dir, ft_lp_on_ft_dir, base_lp_on_ft_dir, ft_lp_on_base_dir]:
                os.makedirs(d)

            # Fine-tuned model — generate + cache logprobs on own completions
            print(f"\n  Fine-tuned model — generate + cache logprobs ...")
            ft_gen_paths, ft_lp_on_ft_paths = generate_and_cache_logprobs(
                ckpt_path, prompts, tokenizer, ft_lp_on_ft_dir, ft_gen_dir,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, dtype=dtype,
            )

            # Base model — cache logprobs on ft completions
            print(f"\n  Base model — logprobs on fine-tuned completions ...")
            base_lp_on_ft_paths = cache_logprobs_for_sequences(
                base_model, ft_lp_on_ft_dir, base_lp_on_ft_dir, num_samples, dtype=dtype,
            )

            # Fine-tuned model — cache logprobs on base completions
            print(f"\n  Fine-tuned model — logprobs on base completions ...")
            ft_lp_on_base_paths = cache_logprobs_for_sequences(
                ckpt_path, base_lp_on_base_dir, ft_lp_on_base_dir, num_samples, dtype=dtype,
            )

            # Compute KL
            kl_base = compute_kl_from_caches(
                base_lp_on_base_paths, ft_lp_on_base_paths, base_lp_on_base_dir,
            )
            kl_ft = compute_kl_from_caches(
                base_lp_on_ft_paths, ft_lp_on_ft_paths, ft_lp_on_ft_dir,
            )

            results = {
                "base_completions": kl_base,
                "ft_completions": kl_ft,
            }
            print_summary(results, ckpt_name)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"kl_gen_{ckpt_name}.json")
            else:
                out_path = os.path.join(ckpt_path, "kl_divergence_gen.json")

            save_data = {
                "base_model": base_model,
                "checkpoint": ckpt_path,
                "step": step,
                "dataset": dataset,
                "max_samples": max_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_samples": num_samples,
                "base_completions": kl_base["aggregate"],
                "ft_completions": kl_ft["aggregate"],
                "base_completions_per_sample": kl_base["per_sample"],
                "ft_completions_per_sample": kl_ft["per_sample"],
            }
            with open(out_path, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"  Saved to {out_path}")

            all_results.append(results)
            trajectory.append({
                "step": step,
                "base_completions_mean_kl": kl_base["aggregate"]["mean_kl"],
                "ft_completions_mean_kl": kl_ft["aggregate"]["mean_kl"],
                "base_completions_mean_sample_kl": kl_base["aggregate"].get("mean_sample_kl", 0.0),
                "ft_completions_mean_sample_kl": kl_ft["aggregate"].get("mean_sample_kl", 0.0),
            })

            # Clean up ft caches for this checkpoint
            for d in [ft_gen_dir, ft_lp_on_ft_dir, base_lp_on_ft_dir, ft_lp_on_base_dir]:
                for f_path in Path(d).iterdir():
                    f_path.unlink()

    traj_out = os.path.join(output_dir or run_dir, "kl_gen_trajectory.json")
    with open(traj_out, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\nKL trajectory saved to {traj_out}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute forward KL divergence D_KL(π_base ‖ π_ft) using "
            "model-generated completions from both base and fine-tuned models"
        )
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

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate per sample (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6, 0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p nucleus sampling (default: 0.95)")

    # Options
    parser.add_argument("--output_dir", type=str, default="results/kl_divergence",
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
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
