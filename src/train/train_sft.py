"""
SFT training on NuminaMath-1.5 using TRL's SFTTrainer.

Saves initial model weights (theta_init) alongside checkpoints
for post-hoc sparsity analysis (|theta_final - theta_init| <= 10^-5).

Usage:
    python -m src.train.train_sft \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/sft/llama-8b \
        --num_train_epochs 2 \
        --learning_rate 5e-6

    CUDA_VISIBLE_DEVICES=4,5 uv run accelerate launch --config_file configs/ds_zero2.yaml -m src.train.train_sft --model meta-llama/Llama-3.1-8B-Instruct

    # Filter by source (e.g. only olympiad problems)
    python -m src.train.train_sft \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/sft/llama-8b-olympiads \
        --sources olympiads

    # Answer-only SFT (no reasoning trace)
    python -m src.train.train_sft \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output_dir results/sft/llama-8b-answer-only \
        --answer_only
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


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

    # Filter to valid problems with solutions and answers
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

    if sources:
        ds = ds.filter(lambda x: x["source"] in sources, num_proc=4)

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds


def format_sft(example, answer_only: bool = False):
    """Format a NuminaMath example into chat messages for SFT.

    Args:
        answer_only: If True, the assistant response is just the boxed answer
                     (no reasoning trace). Used for the answer-only SFT baseline.
    """
    if answer_only:
        response = f"\\boxed{{{example['answer']}}}"
    else:
        solution = example["solution"].strip()
        # Append boxed answer if not already present
        if "\\boxed" not in solution:
            solution += f"\n\nThe answer is $\\boxed{{{example['answer']}}}$."
        response = solution

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
        {"role": "assistant", "content": response},
    ]
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Initial weight snapshot
# ---------------------------------------------------------------------------

def save_theta_init(model: AutoModelForCausalLM, output_dir: str):
    """Save initial model weights for sparsity analysis. Only runs on main process.

    Saves a state_dict mapping parameter names to their initial values.
    This is used to compute |theta_final - theta_init| after training.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return

    theta_init_dir = os.path.join(output_dir, "theta_init")
    os.makedirs(theta_init_dir, exist_ok=True)

    # Save as sharded safetensors (same format as HF checkpoints)
    model.save_pretrained(theta_init_dir, safe_serialization=True)

    # Remove non-weight files to save space (we only need the tensors)
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
        #attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # Clear sampling params from generation_config to avoid validation errors
    # (some models ship with temperature/top_p set, which conflicts with do_sample=False)
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    # Save initial weights before any
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
        format_sft,
        fn_kwargs={"answer_only": args.answer_only},
        remove_columns=ds.column_names,
        num_proc=4,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        #num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,  # keep all checkpoints for sparsity analysis
        seed=args.seed,
        report_to="tensorboard",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Print training plan
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"Training plan:")
    print(f"  Dataset size:        {len(ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Log every:           {args.logging_steps} steps")
    print(f"  Save every:          {args.save_steps} steps")
    print(f"  Warmup steps:        {args.warmup_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
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
    parser = argparse.ArgumentParser(description="SFT on NuminaMath-1.5")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/sft")

    # Data
    parser.add_argument("--max_samples", type=int, default=None,)
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Filter to specific NuminaMath sources (e.g. olympiads cn_k12)",
    )
    parser.add_argument(
        "--answer_only", action="store_true",
        help="Train on answer-only responses (no reasoning trace)",
    )

    # Training hyperparameters
    #parser.add_argument("--num_train_epochs", type=int, default=1)
    # max_steps overrides num_train_epochs
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
