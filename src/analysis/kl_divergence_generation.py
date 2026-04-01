"""
Compute token-level KL divergence between base and fine-tuned models
using model-generated completions.

For each model (base and fine-tuned), generates completions from dataset
prompts using vLLM (fast batched generation), then loads both models on
separate GPUs via HF and computes KL on-the-fly — no logprob caching.

Datasets are loaded using the same logic as training (src.train.train_sft),
and prompts are formatted with chat templates to match training exactly.

This gives two KL measurements:
    - KL_base: D_KL(π_base ‖ π_ft) on base model completions
    - KL_ft:   D_KL(π_base ‖ π_ft) on fine-tuned model completions

Phases (single checkpoint):
    Phase 1: vLLM base model — batch generate completions, save sequences.
    Phase 2: vLLM fine-tuned model — batch generate completions, save sequences.
    Phase 3: Load both HF models (base on cuda:0, ft on cuda:1),
             compute KL on-the-fly for both completion sets.

Requires CUDA_VISIBLE_DEVICES to expose 2 GPUs.

Usage:
    # Single checkpoint
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_generation \
        --base_model Qwen/Qwen3-4B-Base \
        --checkpoint results/sft/numinamath/Qwen3-4B-Base/checkpoint-100 \
        --dataset numinamath \
        --max_samples 200

    # All checkpoints in a run directory
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_generation \
        --run_dir results/sft/numinamath/Qwen3-4B-Base \
        --dataset numinamath \
        --max_samples 200

    # With custom chat template and generation parameters
    CUDA_VISIBLE_DEVICES=4,5 python -m src.analysis.kl_divergence_generation \
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
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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
    # Load full dataset, then truncate to max_samples
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

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

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
# vLLM generation
# ---------------------------------------------------------------------------

def generate_completions(
    model_path: str,
    prompts: list[str],
    tokenizer: AutoTokenizer,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
) -> None:
    """Batch generate completions with vLLM, save sequences as meta files.

    For each prompt, saves:
        - gen_{i}.json: generation metadata (prompt, generation text, lengths)
        - meta_{i}.pt: tokenized full sequence (input_ids, prompt_len) for
                        downstream teacher-forcing
    """
    print(f"  Loading vLLM engine for {model_path} ...")
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    print(f"  Generating {len(prompts)} completions ...")
    outputs = llm.generate(prompts, sampling_params)

    # Save each generation as meta file + json
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        gen_text = output.outputs[0].text

        # Tokenize full sequence (prompt + generation) for teacher-forcing
        full_text = prompt + gen_text
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Save meta for teacher-forcing
        meta_path = os.path.join(output_dir, f"meta_{i:06d}.pt")
        torch.save({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "prompt_len": prompt_len,
        }, meta_path)

        # Save generation metadata
        gen_path = os.path.join(output_dir, f"gen_{i:06d}.json")
        with open(gen_path, "w") as f:
            json.dump({
                "index": i,
                "generation": gen_text,
                "prompt_len": prompt_len,
                "total_len": len(full_ids),
                "gen_len": len(full_ids) - prompt_len,
            }, f, indent=2)

    print(f"  Saved {len(outputs)} generations to {output_dir}")

    # Free vLLM engine
    destroy_model_parallel()
    del llm
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load a model onto a specific GPU device."""
    print(f"  Loading HF model from {model_path} -> {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Online KL computation from saved sequences (no logprob caching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_from_sequences(
    base_model: AutoModelForCausalLM,
    ft_model: AutoModelForCausalLM,
    sequence_dir: str,
    num_samples: int,
) -> dict:
    """Compute per-token forward KL divergence on-the-fly from saved sequences.

    Reads input_ids from meta files in sequence_dir, forward passes both
    models, computes KL per-token and discards full vocab distributions.

    KL(π_base ‖ π_ft) = Σ_v π_base(v) [log π_base(v) - log π_ft(v)]
    Only computed over completion tokens (after prompt_len).
    """
    base_device = next(base_model.parameters()).device
    ft_device = next(ft_model.parameters()).device

    per_sample = []
    all_token_kls = []

    for i in range(num_samples):
        meta_path = os.path.join(sequence_dir, f"meta_{i:06d}.pt")
        meta = torch.load(meta_path, weights_only=True)
        prompt_len = meta["prompt_len"]
        input_ids = meta["input_ids"].unsqueeze(0)

        # Forward pass on both models (each on its own GPU)
        base_out = base_model(input_ids=input_ids.to(base_device))
        ft_out = ft_model(input_ids=input_ids.to(ft_device))

        # logits[:, :-1, :] predicts tokens at positions [1, 2, ..., seq_len-1]
        seq_len = base_out.logits.shape[1] - 1

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

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"    Processed {i + 1}/{num_samples} samples")

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
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> dict:
    """Compute KL divergence using model-generated completions.

    Phase 1: vLLM base model — batch generate completions, save sequences.
    Phase 2: vLLM fine-tuned model — batch generate completions, save sequences.
    Phase 3: Load both HF models, compute KL on-the-fly for both completion sets.
    """
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
    prompts = load_prompts(
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=max_samples,
        seed=seed,
    )
    print(f"  {len(prompts)} chat-formatted prompts")

    num_samples = len(prompts)

    with tempfile.TemporaryDirectory(prefix="kl_gen_seq_") as cache_root:
        base_seq_dir = os.path.join(cache_root, "base_seq")
        ft_seq_dir = os.path.join(cache_root, "ft_seq")
        os.makedirs(base_seq_dir)
        os.makedirs(ft_seq_dir)

        # Phase 1: vLLM base model — generate completions
        print(f"\nPhase 1: vLLM base model — generating completions ...")
        generate_completions(
            base_model, prompts, tokenizer, base_seq_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        # Phase 2: vLLM fine-tuned model — generate completions
        print(f"\nPhase 2: vLLM fine-tuned model — generating completions ...")
        generate_completions(
            checkpoint_path, prompts, tokenizer, ft_seq_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        # Phase 3: Load both HF models, compute KL on-the-fly
        print(f"\nPhase 3: Loading both HF models ...")
        base_mod = load_model(base_model, device=torch.device("cuda:0"), dtype=dtype)
        ft_mod = load_model(checkpoint_path, device=torch.device("cuda:1"), dtype=dtype)

        print(f"\n  Computing KL on base completions ...")
        kl_base = compute_kl_from_sequences(base_mod, ft_mod, base_seq_dir, num_samples)

        print(f"\n  Computing KL on ft completions ...")
        kl_ft = compute_kl_from_sequences(base_mod, ft_mod, ft_seq_dir, num_samples)

        del base_mod, ft_mod
        torch.cuda.empty_cache()

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
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
    dtype: torch.dtype = torch.bfloat16,
    output_dir: str | None = None,
    seed: int = 42,
    chat_template_model: str | None = None,
) -> list[dict]:
    """Compute generation-based KL for all checkpoints in a training run.

    Base model completions (vLLM) are generated once. Per checkpoint:
    vLLM generates ft completions, then both HF models are loaded on
    separate GPUs and KL is computed on-the-fly for both completion sets.
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

    with tempfile.TemporaryDirectory(prefix="kl_gen_seq_") as cache_root:
        # One-time: base model generation
        base_seq_dir = os.path.join(cache_root, "base_seq")
        os.makedirs(base_seq_dir)

        print(f"\nvLLM base model — generating completions (one-time) ...")
        generate_completions(
            base_model, prompts, tokenizer, base_seq_dir,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        for ckpt_path, step in checkpoints:
            ckpt_name = Path(ckpt_path).name
            print(f"\n{'='*50}")
            print(f"--- {ckpt_name} (step {step}) ---")
            print(f"{'='*50}")

            ft_seq_dir = os.path.join(cache_root, f"ft_seq_{ckpt_name}")
            os.makedirs(ft_seq_dir)

            # vLLM: generate ft completions
            print(f"\n  vLLM fine-tuned model — generating completions ...")
            generate_completions(
                ckpt_path, prompts, tokenizer, ft_seq_dir,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )

            # Load both HF models, compute KL on-the-fly
            print(f"\n  Loading both HF models ...")
            base_mod = load_model(base_model, device=torch.device("cuda:0"), dtype=dtype)
            ft_mod = load_model(ckpt_path, device=torch.device("cuda:1"), dtype=dtype)

            print(f"\n  Computing KL on base completions ...")
            kl_base = compute_kl_from_sequences(base_mod, ft_mod, base_seq_dir, num_samples)

            print(f"\n  Computing KL on ft completions ...")
            kl_ft = compute_kl_from_sequences(base_mod, ft_mod, ft_seq_dir, num_samples)

            del base_mod, ft_mod
            torch.cuda.empty_cache()

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

            # Clean up ft sequences for this checkpoint
            for f_path in Path(ft_seq_dir).iterdir():
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
            "model-generated completions (vLLM) with HF teacher-forced logprobs"
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

    # Generation (vLLM)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate per sample (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6, 0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p nucleus sampling (default: 0.95)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size (default: 1)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max context length for vLLM KV cache (default: 4096, 0 for model default)")

    # Options
    parser.add_argument("--output_dir", type=str, default="results/kl_divergence",
                        help="Output directory for JSON results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset sampling")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from (e.g. instruct variant for a base model)")

    args = parser.parse_args()

    dtype = torch.bfloat16
    max_model_len = args.max_model_len if args.max_model_len > 0 else None

    if args.run_dir:
        analyze_run(
            run_dir=args.run_dir,
            dataset=args.dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=max_model_len,
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
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=max_model_len,
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
