"""
ICL validation for SDFT (Section 3.2 of arXiv:2601.19897).

Before running SDFT, verify the base model has sufficient in-context learning
ability for the self-teacher mechanism to work.  Two tests:

  1. **Optimality**: The teacher (base model conditioned on a demonstration)
     should achieve substantially higher accuracy than the base model alone.
  2. **Minimal deviation**: The teacher distribution should remain close to the
     base distribution (low KL divergence), ensuring SDFT operates within a
     reasonable trust region.

Usage:
    python -m src.analysis.sdft_icl_test \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/icl_test

    # KL measurement on a small subset (slower, uses HF forward passes)
    python -m src.analysis.sdft_icl_test \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --measure_kl --kl_samples 50
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.eval.run_eval import extract_boxed_answer, is_equiv
from src.train.train_sdft import TEACHER_TEMPLATE_1, TEACHER_TEMPLATE_2
from src.train.train_sft import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Copying detection
# ---------------------------------------------------------------------------

def _longest_common_subsequence_len(a: list, b: list) -> int:
    """Length of the longest common subsequence (token-level)."""
    if not a or not b:
        return 0
    # Space-optimized DP — only need two rows.
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0] * (len(b) + 1)
        for j, bj in enumerate(b):
            curr[j + 1] = prev[j] + 1 if ai == bj else max(prev[j + 1], curr[j])
        prev = curr
    return prev[-1]


def copying_ratio(response: str, demonstration: str) -> float:
    """Fraction of response tokens that appear (in order) in the demonstration.

    Returns the token-level LCS length / response length.  A ratio near 1.0
    means the response is essentially a copy of the demonstration.
    """
    # Simple whitespace tokenization — model-agnostic and fast enough.
    resp_toks = response.split()
    demo_toks = demonstration.split()
    if not resp_toks:
        return 0.0
    lcs = _longest_common_subsequence_len(resp_toks, demo_toks)
    return lcs / len(resp_toks)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_math500_with_solutions() -> list[dict]:
    """Load MATH-500 with gold solutions (needed as teacher demonstrations)."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": row["solution"],
            "level": int(row["level"]),
            "subject": row["subject"],
        })
    return out


def load_numinamath_sample(n: int = 200, seed: int = 42) -> list[dict]:
    """Load a random sample from NuminaMath-1.5 for ICL testing."""
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    ds = ds.filter(
        lambda x: (
            x["problem_is_valid"] == "Yes"
            and x["solution_is_valid"] == "Yes"
            and x["solution"] is not None
            and x["answer"] is not None
            and len(x["solution"].strip()) > 0
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        solution = row["solution"].strip()
        if "\\boxed" not in solution:
            solution += f"\n\nThe answer is $\\boxed{{{row['answer']}}}$."
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": solution,
            "level": 0,
            "subject": row.get("source", ""),
        })
    return out


def load_deepmath_sample(n: int = 200, seed: int = 42) -> list[dict]:
    """Load a random sample from DeepMath-103K for ICL testing."""
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.filter(
        lambda x: (
            x["r1_solution_1"] is not None
            and len(x["r1_solution_1"].strip()) > 0
            and x["final_answer"] is not None
            and len(x["final_answer"].strip()) > 0
        ),
        num_proc=4,
    )
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        solution = row["r1_solution_1"].strip()
        if "\\boxed" not in solution:
            solution += f"\n\nThe answer is $\\boxed{{{row['final_answer']}}}$."
        out.append({
            "problem": row["question"],
            "answer": row["final_answer"],
            "solution": solution,
            "level": 0,
            "subject": row.get("topic", ""),
        })
    return out


DATASET_LOADERS = {
    "math500": load_math500_with_solutions,
    "numinamath": load_numinamath_sample,
    "deepmath": load_deepmath_sample,
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_student_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def build_teacher_prompt(problem: str, solution: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEACHER_TEMPLATE_2.format(
        #{"role": "user", "content": TEACHER_TEMPLATE_1.format(
            question=problem, demonstration=solution,
        )},
    ]


# ---------------------------------------------------------------------------
# Test 1: Accuracy comparison (vLLM)
# ---------------------------------------------------------------------------

def accuracy_test(
    model_name: str,
    problems: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 4096,
) -> dict:
    """Compare base vs teacher accuracy on the given problems."""

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        **({"max_model_len": max_model_len} if max_model_len else {}),
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    # -- Build prompts for both conditions --
    student_prompts, teacher_prompts = [], []
    for p in problems:
        s_msgs = build_student_prompt(p["problem"])
        t_msgs = build_teacher_prompt(p["problem"], p["solution"])
        student_prompts.append(tokenizer.apply_chat_template(
            s_msgs, tokenize=False, add_generation_prompt=True,
        ))
        teacher_prompts.append(tokenizer.apply_chat_template(
            t_msgs, tokenize=False, add_generation_prompt=True,
        ))

    # -- Generate (base) --
    print(f"\n  Generating base responses ({len(problems)} problems)...")
    t0 = time.time()
    base_outputs = llm.generate(student_prompts, sampling)
    base_time = time.time() - t0
    print(f"  Base generation: {base_time:.1f}s")

    # -- Generate (teacher) --
    print(f"  Generating teacher responses ({len(problems)} problems)...")
    t0 = time.time()
    teacher_outputs = llm.generate(teacher_prompts, sampling)
    teacher_time = time.time() - t0
    print(f"  Teacher generation: {teacher_time:.1f}s")

    # -- Score --
    per_problem = []
    for i, prob in enumerate(problems):
        base_resp = base_outputs[i].outputs[0].text
        teacher_resp = teacher_outputs[i].outputs[0].text

        base_pred = extract_boxed_answer(base_resp)
        teacher_pred = extract_boxed_answer(teacher_resp)

        base_correct = is_equiv(base_pred, prob["answer"]) if base_pred else False
        teacher_correct = is_equiv(teacher_pred, prob["answer"]) if teacher_pred else False

        copy_ratio = copying_ratio(teacher_resp, prob["solution"])

        per_problem.append({
            "problem": prob["problem"],
            "answer": prob["answer"],
            "level": prob.get("level", 0),
            "subject": prob.get("subject", ""),
            "base_pred": base_pred,
            "base_correct": base_correct,
            "teacher_pred": teacher_pred,
            "teacher_correct": teacher_correct,
            "teacher_copy_ratio": copy_ratio,
        })

    n = len(per_problem)
    copy_threshold = 0.5
    novel = [r for r in per_problem if r["teacher_copy_ratio"] < copy_threshold]
    copied = [r for r in per_problem if r["teacher_copy_ratio"] >= copy_threshold]

    base_acc = sum(r["base_correct"] for r in per_problem) / n
    teacher_acc = sum(r["teacher_correct"] for r in per_problem) / n
    teacher_novel_acc = (
        sum(r["teacher_correct"] for r in novel) / len(novel) if novel else 0
    )

    return {
        "base_accuracy": base_acc,
        "teacher_accuracy": teacher_acc,
        "teacher_novel_accuracy": teacher_novel_acc,
        "accuracy_gain": teacher_acc - base_acc,
        "novel_accuracy_gain": teacher_novel_acc - base_acc,
        "n_problems": n,
        "n_novel": len(novel),
        "n_copied": len(copied),
        "mean_copy_ratio": sum(r["teacher_copy_ratio"] for r in per_problem) / n,
        "base_extraction_failures": sum(1 for r in per_problem if r["base_pred"] is None),
        "teacher_extraction_failures": sum(1 for r in per_problem if r["teacher_pred"] is None),
        "copy_threshold": copy_threshold,
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Test 2: KL divergence measurement (HuggingFace forward passes)
# ---------------------------------------------------------------------------

def kl_test(
    model_name: str,
    problems: list[dict],
    max_new_tokens: int = 512,
) -> dict:
    """Measure mean token-level KL(teacher || base) on student-generated completions.

    Uses HuggingFace for full-vocabulary logprobs on a small sample.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    kl_values = []

    for i, prob in enumerate(problems):
        # -- Tokenize prompts --
        s_text = tokenizer.apply_chat_template(
            build_student_prompt(prob["problem"]),
            tokenize=False, add_generation_prompt=True,
        )
        t_text = tokenizer.apply_chat_template(
            build_teacher_prompt(prob["problem"], prob["solution"]),
            tokenize=False, add_generation_prompt=True,
        )

        # TODO : change hardcoded max_length?
        s_enc = tokenizer(s_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        t_enc = tokenizer(t_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

        # -- Generate completion from base (student prompt) --
        with torch.no_grad():
            gen = model.generate(
                **s_enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = gen[0, s_enc["input_ids"].shape[1]:]
        if completion_ids.numel() == 0:
            continue

        # -- Forward pass: base (student prompt + completion) --
        base_ids = torch.cat([s_enc["input_ids"][0], completion_ids]).unsqueeze(0)
        base_mask = torch.ones_like(base_ids)

        # -- Forward pass: teacher (teacher prompt + completion) --
        teacher_ids = torch.cat([t_enc["input_ids"][0], completion_ids]).unsqueeze(0)
        teacher_mask = torch.ones_like(teacher_ids)

        Lp_s = s_enc["input_ids"].shape[1]
        Lp_t = t_enc["input_ids"].shape[1]
        Lc = completion_ids.shape[0]

        with torch.no_grad():
            base_logits = model(input_ids=base_ids, attention_mask=base_mask).logits
            teacher_logits = model(input_ids=teacher_ids, attention_mask=teacher_mask).logits

        # Logprobs over completion tokens only:
        #   logits[:, t, :] predicts token at position t+1.
        #   First completion token is at position Lp, so logit at Lp-1.
        base_log_p = torch.log_softmax(
            base_logits[0, Lp_s - 1 : Lp_s - 1 + Lc, :].float(), dim=-1
        )
        teacher_log_p = torch.log_softmax(
            teacher_logits[0, Lp_t - 1 : Lp_t - 1 + Lc, :].float(), dim=-1
        )

        # D_KL(teacher || base) = Σ_v π_t(v) · (log π_t(v) - log π_base(v))
        per_token_kl = (teacher_log_p.exp() * (teacher_log_p - base_log_p)).sum(-1)
        mean_kl = per_token_kl.mean().item()
        kl_values.append(mean_kl)

        if (i + 1) % 10 == 0:
            print(f"  KL progress: {i+1}/{len(problems)}, running mean = {sum(kl_values)/len(kl_values):.4f} nats")

    return {
        "mean_kl_teacher_base": sum(kl_values) / len(kl_values) if kl_values else 0,
        "n_samples": len(kl_values),
        "per_sample_kl": kl_values,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_accuracy_report(results: dict, dataset_name: str):
    n = results["n_problems"]
    n_novel = results["n_novel"]
    n_copied = results["n_copied"]

    print(f"\n{'='*60}")
    print(f"ICL Accuracy Test — {dataset_name} ({n} problems)")
    print(f"{'='*60}")
    print(f"  Base (no demo):           {results['base_accuracy']*100:.1f}%")
    print(f"  Teacher (w/ demo, all):   {results['teacher_accuracy']*100:.1f}%")
    print(f"  Teacher (novel only):     {results['teacher_novel_accuracy']*100:.1f}%  "
          f"({n_novel} novel, {n_copied} copied, "
          f"threshold={results['copy_threshold']:.0%})")
    print(f"  Mean copy ratio:          {results['mean_copy_ratio']:.2f}")
    print(f"  Novel accuracy gain:      {results['novel_accuracy_gain']*100:+.1f}pp")
    print(f"  Extraction fails:         base={results['base_extraction_failures']}, "
          f"teacher={results['teacher_extraction_failures']}")

    # Breakdown by level if available
    by_level: dict[int, dict] = {}
    for r in results["per_problem"]:
        lv = r.get("level", 0)
        if lv == 0:
            continue
        if lv not in by_level:
            by_level[lv] = {"base_c": 0, "teacher_c": 0, "novel_c": 0,
                            "novel_n": 0, "total": 0}
        by_level[lv]["total"] += 1
        by_level[lv]["base_c"] += int(r["base_correct"])
        by_level[lv]["teacher_c"] += int(r["teacher_correct"])
        if r["teacher_copy_ratio"] < results["copy_threshold"]:
            by_level[lv]["novel_n"] += 1
            by_level[lv]["novel_c"] += int(r["teacher_correct"])

    if by_level:
        print(f"\n  {'Level':<8} {'Base':>8} {'Teacher':>9} {'Novel':>9} {'Gain':>7}")
        print(f"  {'-'*45}")
        for lv in sorted(by_level):
            d = by_level[lv]
            ba = d["base_c"] / d["total"] * 100
            ta = d["teacher_c"] / d["total"] * 100
            na = d["novel_c"] / d["novel_n"] * 100 if d["novel_n"] else 0
            print(f"  {lv:<8} {ba:>7.1f}% {ta:>8.1f}% {na:>8.1f}% {na-ba:>+6.1f}%")

    gain = results["novel_accuracy_gain"]
    if gain > 0.05:
        print(f"\n  >> ICL effective: novel teacher gains {gain*100:.1f}pp.  SDFT should help.")
    elif gain > 0:
        print(f"\n  >> Marginal ICL gain ({gain*100:.1f}pp on novel responses).  "
              f"SDFT benefit may be limited.")
    else:
        print(f"\n  >> No ICL gain on novel responses.  "
              f"SDFT is unlikely to help for this model/task.")


def print_kl_report(results: dict):
    print(f"\n{'='*60}")
    print(f"ICL Minimal Deviation Test ({results['n_samples']} samples)")
    print(f"{'='*60}")
    print(f"  Mean D_KL(teacher || base): {results['mean_kl_teacher_base']:.4f} nats")
    if results["mean_kl_teacher_base"] < 1.0:
        print(f"  >> Teacher stays close to base — trust-region assumption holds.")
    else:
        print(f"  >> Teacher deviates substantially from base — check ICL prompt.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ICL validation for SDFT (Section 3.2, arXiv:2601.19897)",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="math500",
                        choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--output_dir", type=str, default="results/icl_test")

    # Accuracy test
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)

    # KL test
    parser.add_argument("--measure_kl", action="store_true",
                        help="Also measure KL divergence (slower, uses HF forward passes)")
    parser.add_argument("--kl_samples", type=int, default=50,
                        help="Number of problems for KL measurement")
    parser.add_argument("--kl_max_new_tokens", type=int, default=512,
                        help="Max completion length for KL generation")

    # Sampling
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of problems to sample (for numinamath/deepmath)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # -- Load dataset --
    loader = DATASET_LOADERS[args.dataset]
    import inspect
    sig = inspect.signature(loader)
    kwargs = {}
    if "n" in sig.parameters:
        kwargs["n"] = args.n_samples
    if "seed" in sig.parameters:
        kwargs["seed"] = args.seed
    problems = loader(**kwargs)
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # -- Test 1: Accuracy --
    print(f"\n--- Test 1: Accuracy comparison ---")
    acc_results = accuracy_test(
        model_name=args.model,
        problems=problems,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len or None,
    )
    print_accuracy_report(acc_results, args.dataset)

    # -- Test 2: KL divergence (optional) --
    kl_results = None
    if args.measure_kl:
        print(f"\n--- Test 2: KL divergence measurement ---")
        kl_subset = problems[:args.kl_samples]
        kl_results = kl_test(
            model_name=args.model,
            problems=kl_subset,
            max_new_tokens=args.kl_max_new_tokens,
        )
        print_kl_report(kl_results)

    # -- Save results --
    output_dir = os.path.join(args.output_dir, args.dataset, args.model.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "accuracy": {k: v for k, v in acc_results.items() if k != "per_problem"},
    }
    if kl_results:
        summary["kl"] = {k: v for k, v in kl_results.items() if k != "per_sample_kl"}

    summary_path = os.path.join(output_dir, "icl_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    details_path = os.path.join(output_dir, "icl_test_details.json")
    detail_data = {"accuracy_per_problem": acc_results["per_problem"]}
    if kl_results:
        detail_data["kl_per_sample"] = kl_results["per_sample_kl"]
    with open(details_path, "w") as f:
        json.dump(detail_data, f, indent=2)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
