"""
Evaluate language models on MATH and other math benchmarks.

Usage:
    python -m src.eval.run_eval \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset math500 \
        --output_dir results/eval

    # Multiple models
    python -m src.eval.run_eval \
        --model meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --dataset math500 \
        --output_dir results/eval

    # Filter by difficulty level
    python -m src.eval.run_eval \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset math500 \
        --levels 1 2 3 \
        --output_dir results/eval
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)


# ---------------------------------------------------------------------------
# Dataset loaders – each returns list[dict] with keys:
#   problem, answer, level (int), subject, unique_id (optional)
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, callable] = {}


def register_dataset(name):
    def wrapper(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return wrapper


@register_dataset("math500")
def load_math500(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        level = int(row["level"])
        if levels and level not in levels:
            continue
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": row["solution"],
            "level": level,
            "subject": row["subject"],
            "unique_id": row.get("unique_id", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# For model outputs: prioritize \boxed{}, then fall back to other latex / expressions
PRED_EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]

# For gold answers: typically a clean latex string
GOLD_EXTRACTION_CONFIG = [
    LatexExtractionConfig(),
    ExprExtractionConfig(),
]

def parse_answer(text: str, is_gold: bool = False) -> list | None:
    """Parse a math answer string into a verified representation.

    Returns the parsed result or None if parsing fails.
    """
    config = GOLD_EXTRACTION_CONFIG if is_gold else PRED_EXTRACTION_CONFIG
    try:
        return parse(text, extraction_config=config)
    except Exception:
        return None


def is_equiv(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers are mathematically equivalent."""
    gold_parsed = parse_answer(gold, is_gold=True)
    pred_parsed = parse_answer(pred, is_gold=False)
    if gold_parsed is None or pred_parsed is None:
        return False
    try:
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False
    

def evaluate_model(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> dict:
    """Run evaluation and return results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Build prompts
    prompts = []
    # TODO : test prompts
    for p in problems:
        messages = format_prompt(p["problem"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # Generate
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.1f}s ({len(problems)/elapsed:.1f} problems/s)")

    # Score
    results = []
    for prob, output in zip(problems, outputs):
        response = output.outputs[0].text
        pred_parsed = parse_answer(response, is_gold=False)
        correct = is_equiv(response, prob["answer"]) if pred_parsed else False
        results.append({
            **prob,
            "response": response,
            "pred_answer": str(pred_parsed) if pred_parsed else None,
            "correct": correct,
        })

    return {
        "model": model_name,
        "results": results,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(eval_output: dict):
    """Print accuracy breakdown by level and subject."""
    model = eval_output["model"]
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    print(f"\n{'='*60}")
    print(f"Results: {model}")
    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*60}")

    # By level
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    print(f"\n{'Level':<10} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 35)
    for level in sorted(by_level):
        d = by_level[level]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"Level {level:<4} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # By subject
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    print(f"\n{'Subject':<25} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 50)
    for subj in sorted(by_subject):
        d = by_subject[subj]
        acc = d["correct"] / d["total"] * 100 if d["total"] else 0
        print(f"{subj:<25} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")

    # Extraction failures
    no_answer = sum(1 for r in results if r["pred_answer"] is None)
    if no_answer:
        print(f"\nExtraction failures (no \\boxed{{}}): {no_answer}/{total}")


def save_results(eval_output: dict, output_dir: str):
    """Save full results and summary to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_slug = eval_output["model"].replace("/", "_")

    # Full per-problem results
    results_path = os.path.join(output_dir, f"{model_slug}_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_output["results"], f, indent=2)

    # Summary
    results = eval_output["results"]
    total = len(results)
    correct = sum(r["correct"] for r in results)

    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_level[r["level"]]["total"] += 1
        by_level[r["level"]]["correct"] += int(r["correct"])

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_subject[r["subject"]]["total"] += 1
        by_subject[r["subject"]]["correct"] += int(r["correct"])

    summary = {
        "model": eval_output["model"],
        "dataset_size": total,
        "overall_accuracy": correct / total if total else 0,
        "elapsed_s": eval_output["elapsed_s"],
        "by_level": {
            str(k): {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_level.items())
        },
        "by_subject": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(by_subject.items())
        },
        "extraction_failures": sum(
            1 for r in results if r["pred_answer"] is None
        ),
    }
    summary_path = os.path.join(output_dir, f"{model_slug}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on math benchmarks")
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="HuggingFace model name(s) or local checkpoint path(s)",
    )
    parser.add_argument(
        "--dataset", default="math500", choices=list(DATASET_REGISTRY.keys()),
        help="Benchmark dataset to evaluate on",
    )
    parser.add_argument(
        "--levels", nargs="*", type=int, default=None,
        help="Filter to specific MATH difficulty levels (1-5)",
    )
    parser.add_argument("--output_dir", default="results/eval")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    # Load dataset
    loader = DATASET_REGISTRY[args.dataset]
    problems = loader(levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    if args.levels:
        print(f"  Filtered to levels: {args.levels}")

    # Evaluate each model
    for model_name in args.model:
        eval_output = evaluate_model(
            model_name=model_name,
            problems=problems,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        print_report(eval_output)
        save_results(eval_output, args.output_dir)


if __name__ == "__main__":
    main()
