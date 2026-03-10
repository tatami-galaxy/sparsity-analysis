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

from datasets import load_dataset
from vllm import LLM, SamplingParams

import re

from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)
from src.utils.utils import get_root_dir


# ---------------------------------------------------------------------------
# Answer extraction and equivalence checking
# ---------------------------------------------------------------------------

PRED_EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]
GOLD_EXTRACTION_CONFIG = [
    LatexExtractionConfig(),
    ExprExtractionConfig(),
]


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _normalize(s: str) -> str:
    """Normalize a math answer string for string comparison."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove \text{} wrapper
    m = re.fullmatch(r"\\text\{(.+)\}", s)
    if m:
        s = m.group(1).strip()
    # Remove display commands
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number (int, float, or simple fraction)."""
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        pass
    # a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    # -\frac{a}{b}
    m = re.fullmatch(r"-\\frac\{(\d+)\}\{(\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return -int(m.group(1)) / int(m.group(2))
    return None


def is_equiv(pred: str, gold: str) -> bool:
    """Check equivalence using layered strategies:
    1. Normalized string match (fast, handles most cases)
    2. Numeric comparison (fractions, decimals)
    3. math_verify symbolic comparison (fallback for complex expressions)
    """
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    # 1. Exact string match after normalization
    if pred_n == gold_n:
        return True
    # 2. Numeric comparison
    pred_v = _try_parse_number(pred_n)
    gold_v = _try_parse_number(gold_n)
    if pred_v is not None and gold_v is not None:
        return abs(pred_v - gold_v) < 1e-6
    # 3. Symbolic comparison via math_verify
    try:
        gold_parsed = parse(gold, extraction_config=GOLD_EXTRACTION_CONFIG)
        pred_parsed = parse(pred, extraction_config=PRED_EXTRACTION_CONFIG)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


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

def evaluate_model(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
) -> dict:
    """Run evaluation and return results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Problems:   {len(problems)}")
    print(f"{'='*60}")

    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Build prompts
    prompts = []
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
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False
        results.append({
            **prob,
            "response": response,
            "pred_answer": pred_answer,
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
    parser.add_argument("--output_dir", default="eval")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max context length for vLLM KV cache. Use 0 for model default.")

    args = parser.parse_args()

    # root dir
    root = get_root_dir()

    # Load dataset
    loader = DATASET_REGISTRY[args.dataset]
    problems = loader(levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}")
    if args.levels:
        print(f"  Filtered to levels: {args.levels}")

    # Evaluate each model
    for model_name in args.model:
        output_dir = root+'/results/'+args.dataset+'/'+model_name.split('/')[-1]
        eval_output = evaluate_model(
            model_name=model_name,
            problems=problems,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len or None,
        )
        print_report(eval_output)
        save_results(eval_output, output_dir)


if __name__ == "__main__":
    main()
