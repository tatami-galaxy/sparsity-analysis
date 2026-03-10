"""
GRPO training on NuminaMath-1.5 using TRL's GRPOTrainer.

Outcome-based reward: +1 if the model's boxed answer matches the gold answer, 0 otherwise.

Saves initial model weights (theta_init) alongside checkpoints
for post-hoc sparsity analysis (|theta_final - theta_init| <= 10^-5).

Usage:
    python -m src.train.train_rl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/rl

    # With vLLM generation (colocate mode)
    python -m src.train.train_rl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/rl \
        --use_vllm

    # Filter by source
    python -m src.train.train_rl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/rl \
        --sources olympiads

    # Multi GPU
    accelerate launch -m src.train.train_rl ...
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


# ---------------------------------------------------------------------------
# Answer verification (reused from eval)
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
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    m = re.fullmatch(r"\\text\{(.+)\}", s)
    if m:
        s = m.group(1).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _try_parse_number(s: str) -> float | None:
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    m = re.fullmatch(r"-\\frac\{(\d+)\}\{(\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return -int(m.group(1)) / int(m.group(2))
    return None


def is_equiv(pred: str, gold: str) -> bool:
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    if pred_n == gold_n:
        return True
    pred_v = _try_parse_number(pred_n)
    gold_v = _try_parse_number(gold_n)
    if pred_v is not None and gold_v is not None:
        return abs(pred_v - gold_v) < 1e-6
    try:
        gold_parsed = parse(gold, extraction_config=GOLD_EXTRACTION_CONFIG)
        pred_parsed = parse(pred, extraction_config=PRED_EXTRACTION_CONFIG)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def accuracy_reward(completions, answer, **kwargs):
    """Outcome-based reward: 1.0 if predicted answer matches gold, 0.0 otherwise.

    The `answer` kwarg comes from the dataset column of the same name.
    """
    rewards = []
    for completion, gold in zip(completions, answer):
        # Handle conversational format (list of message dicts)
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion
        pred = extract_boxed_answer(text)
        if pred is not None and is_equiv(pred, gold):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def load_numinamath(
    max_samples: int | None = None,
    sources: list[str] | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load NuminaMath-1.5 with optional filtering."""
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")

    # Filter to valid problems with answers
    ds = ds.filter(
        lambda x: (
            x["problem_is_valid"] == "Yes"
            and x["answer"] is not None
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )

    if sources:
        ds = ds.filter(lambda x: x["source"] in sources, num_proc=4)

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def format_grpo(example):
    """Format a NuminaMath example for GRPO.

    Returns a dict with:
    - prompt: chat messages (system + user) for the model to complete
    - answer: gold answer string (passed to reward function as kwarg)
    """
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
    return {"prompt": prompt, "answer": example["answer"]}


# ---------------------------------------------------------------------------
# Initial weight snapshot
# ---------------------------------------------------------------------------

def save_theta_init(model: AutoModelForCausalLM, output_dir: str):
    """Save initial model weights for sparsity analysis. Only runs on main process."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return

    theta_init_dir = os.path.join(output_dir, "theta_init")
    os.makedirs(theta_init_dir, exist_ok=True)

    model.save_pretrained(theta_init_dir, safe_serialization=True)

    # Remove non-weight files to save space
    for fname in os.listdir(theta_init_dir):
        if not fname.endswith((".safetensors", ".bin", ".pt")):
            os.remove(os.path.join(theta_init_dir, fname))

    print(f"Saved initial weights to {theta_init_dir}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Save initial weights before any training
    args.output_dir = args.output_dir + '/' + args.model.split('/')[-1]
    save_theta_init(model, args.output_dir)

    # Load and format dataset
    sources = args.sources if args.sources else None
    ds = load_numinamath(
        max_samples=args.max_samples,
        sources=sources,
        seed=args.seed,
    )
    print(f"Loaded {len(ds)} training examples")
    if sources:
        print(f"  Filtered to sources: {sources}")

    ds = ds.map(
        format_grpo,
        remove_columns=[c for c in ds.column_names if c not in ["answer"]],
        num_proc=4,
    )

    # Training config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        beta=args.beta,
        # vLLM
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        seed=args.seed,
        report_to="tensorboard",
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Print training plan
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"Training plan (GRPO):")
    print(f"  Dataset size:        {len(ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Num generations:     {args.num_generations}")
    print(f"  Max completion len:  {args.max_completion_length}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Beta (KL coeff):     {args.beta}")
    print(f"  Log every:           {args.logging_steps} steps")
    print(f"  Save every:          {args.save_steps} steps")
    print(f"  Warmup steps:        {args.warmup_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  vLLM:                {args.use_vllm}")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))

    # Save training config for reproducibility
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO on NuminaMath-1.5")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/rl")

    # Data
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Filter to specific NuminaMath sources (e.g. olympiads cn_k12)",
    )

    # GRPO
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL penalty coefficient (0 = no KL)")

    # vLLM
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for faster generation (colocate mode)")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.3,
                        help="GPU memory fraction for vLLM (colocate mode)")

    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
