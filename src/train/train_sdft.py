"""
SDFT: Self-Distillation Fine-Tuning on NuminaMath-1.5.

On-policy algorithm from "Self-Distillation Fine-Tuning" (arXiv:2601.19897).

The model simultaneously serves as student and teacher via in-context learning:
  - Student: receives (system + question), generates response on-policy
  - Teacher: receives (system + question + expert_solution), provides target distribution

Loss: analytic KL divergence over completion tokens (per-sequence normalized):
    forward (default, matches reference code):
                      D_KL(πt ‖ πθ)  = Σ_v πt(v) · (log πt(v) − log πθ(v))
    reverse (matches paper equations):
                      D_KL(πθ ‖ πt)  = Σ_v πθ(v) · (log πθ(v) − log πt(v))

Teacher weights updated via EMA after each optimizer step: φ ← α·θ + (1−α)·φ

~2.5× the compute of SFT (on-policy generation + two forward passes per step).

Usage:
    python -m src.train.train_sdft \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir results/sdft

    CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \\
        --config_file configs/ds_zero2.yaml \\
        -m src.train.train_sdft \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import copy
import json
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.eval.run_eval import extract_boxed_answer, is_equiv
from src.train.train_sft import (
    SYSTEM_PROMPT,
    load_math500,
    load_numinamath,
    save_theta_init,
)


# ---------------------------------------------------------------------------
# Teacher prompt template (from SDFT paper, Appendix A)
# ---------------------------------------------------------------------------

TEACHER_TEMPLATE = (
    "{question}\n\n"
    "This is an example for a response to the question:\n"
    "{demonstration}\n\n"
    "Now answer with a response of your own, including the thinking process:"
)


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_sdft(example):
    """Create student and teacher message lists for one NuminaMath example.

    Student sees only the question.
    Teacher sees the question + expert solution via the ICL template.
    """
    question = example["problem"]
    solution = example["solution"].strip()
    if "\\boxed" not in solution:
        solution += f"\n\nThe answer is $\\boxed{{{example['answer']}}}$."

    student_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    teacher_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEACHER_TEMPLATE.format(
            question=question, demonstration=solution
        )},
    ]
    return {"student_messages": student_messages, "teacher_messages": teacher_messages}


class SDFTDataCollator:
    """Apply chat template, tokenize, and pad student and teacher prompts."""

    def __init__(self, tokenizer, max_prompt_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __call__(self, features):
        def _encode(key):
            texts = [
                self.tokenizer.apply_chat_template(
                    f[key], tokenize=False, add_generation_prompt=True
                )
                for f in features
            ]
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
            )

        s = _encode("student_messages")
        t = _encode("teacher_messages")
        return {
            "student_input_ids": s["input_ids"],
            "student_attention_mask": s["attention_mask"],
            "teacher_input_ids": t["input_ids"],
            "teacher_attention_mask": t["attention_mask"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_padded_batch(
    prompt_ids: list[torch.Tensor],
    completion_ids: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad [prompt ‖ completion] sequences and build a completion mask.

    comp_mask[i, t] = True  iff  logits[i, t, :]  predicts a completion token
                             for item i  (positions [L_p−1, L_p−1+L_c)).

    Returns:
        input_ids  : [B, max_len]
        attn_mask  : [B, max_len]
        comp_mask  : [B, max_len − 1]  bool
    """
    B = len(prompt_ids)
    lens = [p.shape[0] + c.shape[0] for p, c in zip(prompt_ids, completion_ids)]
    max_len = max(lens)

    ids  = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    comp = torch.zeros(B, max_len - 1, dtype=torch.bool)

    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        Lp, Lc = p.shape[0], c.shape[0]
        ids[i, : Lp + Lc] = torch.cat([p, c])
        mask[i, : Lp + Lc] = 1
        # logits[:, t, :] predicts token at input position t+1.
        # First completion token is at input position Lp → logit position Lp−1.
        comp[i, Lp - 1 : Lp - 1 + Lc] = True

    return ids, mask, comp


def _apply_skip_mask(
    per_token_kl: torch.Tensor,
    comp_mask: torch.Tensor,
    skip_n: int,
) -> torch.Tensor:
    """Zero out the first skip_n completion tokens per sequence.

    Reference impl uses skip_n=3 to suppress noisy KL on structural tokens
    (e.g. <think>, opening newlines) that have high variance.
    """
    weight = torch.ones_like(per_token_kl)
    idx = 0
    for row in comp_mask:
        n = int(row.sum().item())
        weight[idx : idx + min(skip_n, n)] = 0.0
        idx += n
    return per_token_kl * weight


# ---------------------------------------------------------------------------
# EMA callback — teacher = exponential moving average of student
# ---------------------------------------------------------------------------

class EMACallback(TrainerCallback):
    """After each optimizer step: φ ← α·θ + (1−α)·φ."""

    def __init__(self, teacher_model: AutoModelForCausalLM, ema_alpha: float = 0.01):
        self.teacher_model = teacher_model
        self.ema_alpha = ema_alpha

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Move teacher to the same device as the student (set by Trainer).
        device = next(model.parameters()).device
        self.teacher_model.to(device)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Unwrap DDP if needed.
        student = model.module if hasattr(model, "module") else model
        alpha = self.ema_alpha
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), self.teacher_model.parameters()):
                pt.data.mul_(1.0 - alpha).add_(ps.data, alpha=alpha)


# ---------------------------------------------------------------------------
# Step-based accuracy callback (replaces on_evaluate for SDFT)
# ---------------------------------------------------------------------------

class StepEvalAccuracyCallback(TrainerCallback):
    """Generation-based accuracy on MATH-500, fired every eval_steps via on_step_end."""

    def __init__(
        self,
        eval_problems: list[dict],
        tokenizer,
        max_new_tokens: int = 2048,
        eval_steps: int = 100,
    ):
        self.eval_problems = eval_problems
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_steps = eval_steps
        self._trainer = None  # set after trainer creation for direct logging

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model.eval()

        correct = 0
        extraction_failures = 0
        device = next(model.parameters()).device

        for prob in self.eval_problems:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = extract_boxed_answer(response)
            if pred is None:
                extraction_failures += 1
            elif is_equiv(pred, prob["answer"]):
                correct += 1

        model.train()

        if local_rank != 0:
            return

        total = len(self.eval_problems)
        acc = correct / total if total else 0
        print(
            f"\n[Step {state.global_step}] Eval accuracy: "
            f"{correct}/{total} = {acc*100:.1f}% "
            f"(extraction failures: {extraction_failures})"
        )
        if self._trainer is not None:
            self._trainer.log({
                "eval_accuracy": acc,
                "eval_extraction_failures": extraction_failures,
            })


# ---------------------------------------------------------------------------
# SDFT Trainer
# ---------------------------------------------------------------------------

class SDFTTrainer(Trainer):
    """On-policy self-distillation trainer.

    Each training step:
      1. Generate completion  y ~ πθ(·|CtxS(x))  [no grad]
      2. Teacher forward on  (CtxT(x, c) ‖ y)    [no grad]
      3. Student forward on  (CtxS(x) ‖ y)        [with grad]
      4. Loss = analytic reverse KL  D_KL(πθ ‖ π_teacher)  per completion token
    """

    def __init__(
        self,
        *args,
        teacher_model: AutoModelForCausalLM,
        sdft_tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
        generation_temperature: float = 1.0,
        skip_first_n_tokens: int = 3,
        kl_direction: str = "forward",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.sdft_tokenizer = sdft_tokenizer
        self.max_new_tokens = max_new_tokens
        self.generation_temperature = generation_temperature
        self.skip_first_n_tokens = skip_first_n_tokens
        self.kl_direction = kl_direction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        s_ids  = inputs["student_input_ids"]       # [B, Ls]
        s_mask = inputs["student_attention_mask"]   # [B, Ls]
        t_ids  = inputs["teacher_input_ids"]        # [B, Lt]
        t_mask = inputs["teacher_attention_mask"]   # [B, Lt]

        B      = s_ids.shape[0]
        device = s_ids.device
        pad_id = self.sdft_tokenizer.pad_token_id

        # ── 1. On-policy generation from student (no grad) ───────────────────
        model.eval()
        completions, student_prompts, teacher_prompts = [], [], []

        with torch.no_grad():
            for i in range(B):
                sp = s_ids[i][s_mask[i].bool()]   # strip right-padding
                gen = model.generate(
                    sp.unsqueeze(0),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.generation_temperature,
                    pad_token_id=pad_id,
                    eos_token_id=self.sdft_tokenizer.eos_token_id,
                )
                completions.append(gen[0, sp.shape[0]:])
                student_prompts.append(sp)
                teacher_prompts.append(t_ids[i][t_mask[i].bool()])

        model.train()

        # ── 2. Build padded full sequences (prompt ‖ completion) ─────────────
        sf_ids, sf_mask, comp_s = _build_padded_batch(student_prompts, completions, pad_id)
        tf_ids, tf_mask, comp_t = _build_padded_batch(teacher_prompts, completions, pad_id)

        sf_ids  = sf_ids.to(device);  sf_mask = sf_mask.to(device)
        tf_ids  = tf_ids.to(device);  tf_mask = tf_mask.to(device)
        comp_s  = comp_s.to(device);  comp_t  = comp_t.to(device)

        # ── 3. Teacher forward (no grad) ──────────────────────────────────────
        with torch.no_grad():
            t_logits = self.teacher_model(
                input_ids=tf_ids, attention_mask=tf_mask
            ).logits  # [B, L_t, V]
            # Extract completion positions and compute log-probs over full vocab.
            # float() for numerical stability (log_softmax in fp32).
            teacher_log_p = torch.log_softmax(
                t_logits[:, :-1, :][comp_t].float(), dim=-1
            )  # [N, V]
            del t_logits

        # ── 4. Student forward (with grad) ────────────────────────────────────
        s_logits = model(input_ids=sf_ids, attention_mask=sf_mask).logits  # [B, L_s, V]
        student_log_p = torch.log_softmax(
            s_logits[:, :-1, :][comp_s].float(), dim=-1
        )  # [N, V]
        del s_logits

        # ── 5. KL divergence over full vocabulary per completion token ────────
        if self.kl_direction == "reverse":
            # Reverse KL: D_KL(πθ ‖ πt) = Σ_v πθ(v) · (log πθ(v) − log πt(v))
            # Matches paper equation; equivalent to F.kl_div(teacher, student, log_target=True).sum(-1)
            per_token_kl = (
                student_log_p.exp() * (student_log_p - teacher_log_p)
            ).sum(-1)  # [N]
        else:
            # Forward KL: D_KL(πt ‖ πθ) = Σ_v πt(v) · (log πt(v) − log πθ(v))
            # Reference codebase default (alpha=0).
            per_token_kl = (
                teacher_log_p.exp() * (teacher_log_p - student_log_p)
            ).sum(-1)  # [N]

        if self.skip_first_n_tokens > 0:
            per_token_kl = _apply_skip_mask(per_token_kl, comp_s, self.skip_first_n_tokens)

        # Per-sequence average, then batch average (matches reference implementation).
        offset = 0
        seq_losses = []
        for row in comp_s:
            n = int(row.sum().item())
            if n > 0:
                seq_losses.append(per_token_kl[offset:offset + n].mean())
            offset += n
        loss = torch.stack(seq_losses).mean()

        return (loss, None) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args):

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    # Clear sampling params from generation_config to avoid do_sample=False conflicts.
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    # ── Teacher (EMA copy, frozen, eval mode) ─────────────────────────────────
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # ── Output dir + initial weights snapshot ────────────────────────────────
    args.output_dir = f"{args.output_dir}/{args.model.split('/')[-1]}"
    save_theta_init(model, args.output_dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = load_numinamath(max_samples=args.max_samples, seed=args.seed)
    print(f"Loaded {len(ds)} training examples from NuminaMath-1.5")

    train_ds = ds.map(
        format_sdft,
        remove_columns=ds.column_names,
        num_proc=4,
    )

    # Filter: drop examples where either prompt exceeds max_prompt_length.
    def _fits(example):
        for key in ("student_messages", "teacher_messages"):
            text = tokenizer.apply_chat_template(
                example[key], tokenize=False, add_generation_prompt=True
            )
            if len(tokenizer.encode(text)) > args.max_prompt_length:
                return False
        return True

    pre = len(train_ds)
    train_ds = train_ds.filter(_fits, num_proc=4)
    print(f"Filtered by max_prompt_length={args.max_prompt_length}: {pre} → {len(train_ds)}")

    # ── Eval set (MATH-500, accuracy only — no SDFT eval loss) ───────────────
    raw_eval_problems = load_math500()
    print(f"Eval set: {len(raw_eval_problems)} MATH-500 problems")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",   # accuracy eval driven by StepEvalAccuracyCallback
        seed=args.seed,
        report_to="tensorboard",
        remove_unused_columns=False,  # must keep student_/teacher_ columns
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ema_cb = EMACallback(teacher_model, ema_alpha=args.ema_alpha)
    callbacks = [ema_cb]

    if args.eval_accuracy:
        eval_cb = StepEvalAccuracyCallback(
            eval_problems=raw_eval_problems,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            eval_steps=args.eval_steps,
        )
        callbacks.append(eval_cb)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SDFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=SDFTDataCollator(tokenizer, max_prompt_length=args.max_prompt_length),
        callbacks=callbacks,
        # SDFT-specific
        teacher_model=teacher_model,
        sdft_tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        generation_temperature=args.generation_temperature,
        skip_first_n_tokens=args.skip_first_n_tokens,
        kl_direction=args.kl_direction,
    )
    if args.eval_accuracy:
        eval_cb._trainer = trainer

    # ── Summary ───────────────────────────────────────────────────────────────
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(train_ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"SDFT training plan:")
    print(f"  Train size:          {len(train_ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Max new tokens:      {args.max_new_tokens}")
    print(f"  Generation temp:     {args.generation_temperature}")
    print(f"  EMA alpha:           {args.ema_alpha}")
    print(f"  Skip first tokens:   {args.skip_first_n_tokens}")
    print(f"  KL direction:        {args.kl_direction}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    trainer.save_model(os.path.join(args.output_dir, "final"))

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDFT on NuminaMath-1.5")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/sdft")

    # Data
    parser.add_argument("--max_samples", type=int, default=None)

    # Generation (on-policy sampling)
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max completion length for on-policy generation")
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="Max tokenized length for student/teacher prompts")

    # SDFT-specific hyperparameters
    parser.add_argument("--ema_alpha", type=float, default=0.01,
                        help="EMA rate for teacher update: φ ← α·θ + (1−α)·φ")
    parser.add_argument("--skip_first_n_tokens", type=int, default=3,
                        help="Skip first n completion tokens in KL loss (structural tokens)")
    parser.add_argument("--kl_direction", type=str, default="forward",
                        choices=["reverse", "forward"],
                        help="KL direction: 'forward' matches reference codebase default, 'reverse' matches paper equations")

    # Training
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_accuracy", action="store_true",
                        help="Enable generation-based accuracy eval (slow)")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
