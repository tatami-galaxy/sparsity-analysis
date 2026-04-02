"""
SDPO: Self-Distilled Policy Optimization on NuminaMath-1.5.

On-policy algorithm from "Self-Distilled Policy Optimization" (arXiv:2601.20802).

Like SDFT, the model serves as both student and teacher via in-context learning,
but the teacher's demonstration comes from successful *peer rollouts* rather than
gold solutions from the dataset.

Each training step:
  1. Generate G completions per question from the student policy  [no grad]
  2. Score completions with rule-based reward (correct boxed answer?)
  3. For each rollout, find a successful peer (different rollout, same question)
  4. Teacher forward on  (question + peer_solution || completion)  [no grad]
  5. Student forward on  (question || completion)                  [with grad]
  6. Loss = JSD_alpha(student, teacher) over top-K logits + tail bucket

Teacher weights updated via EMA after each optimizer step: phi <- alpha*theta + (1-alpha)*phi

Usage:
    python -m src.train.train_sdpo \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output_dir results/sdpo

    CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \\
        --config_file configs/ds_zero2.yaml \\
        -m src.train.train_sdpo \\
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import copy
import json
import os

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.eval.run_math_eval import extract_boxed_answer, is_equiv
from src.train.math.train_sft import (
    SYSTEM_PROMPT,
    load_math500,
    load_numinamath,
    save_theta_init,
)
from src.train.math.train_sdft import (
    TEACHER_TEMPLATE,
    EMACallback,
    StepEvalAccuracyCallback,
    _build_padded_batch,
)


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_sdpo(example):
    """Format a NuminaMath example for SDPO.

    Only prepares student messages + metadata. Teacher prompts are built
    dynamically inside compute_loss based on which peer rollouts succeed.
    """
    question = example["problem"]
    answer = example["answer"]

    solution = example["solution"].strip()
    if "\\boxed" not in solution:
        solution += f"\n\nThe answer is $\\boxed{{{answer}}}$."

    student_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return {
        "student_messages": student_messages,
        "question": question,
        "answer": answer,
        "gold_solution": solution,
    }


class SDPODataCollator:
    """Tokenize student prompts and pass through metadata for SDPO."""

    def __init__(self, tokenizer, max_prompt_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __call__(self, features):
        texts = [
            self.tokenizer.apply_chat_template(
                f["student_messages"], tokenize=False, add_generation_prompt=True
            )
            for f in features
        ]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        return {
            "student_input_ids": encoded["input_ids"],
            "student_attention_mask": encoded["attention_mask"],
            "questions": [f["question"] for f in features],
            "answers": [f["answer"] for f in features],
            "gold_solutions": [f["gold_solution"] for f in features],
        }


# ---------------------------------------------------------------------------
# Top-K + tail bucket helpers
# ---------------------------------------------------------------------------

def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail bucket representing remaining probability mass.

    Args:
        log_probs: [N, K] log-probabilities of the top-K entries.

    Returns:
        [N, K+1] where the last entry is log(1 - sum(top_k_probs)).
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))  # log(1 - exp(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _compute_jsd(
    student_log_p: torch.Tensor,
    teacher_log_p: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence per token.

    JSD_alpha = alpha * KL(M || teacher) + (1-alpha) * KL(M || student)
    where M = alpha * teacher + (1-alpha) * student  (mixture in prob space).

    Returns [N] per-token JSD values.
    """
    alpha_t = torch.tensor(alpha, dtype=student_log_p.dtype, device=student_log_p.device)

    mixture_log_p = torch.logsumexp(
        torch.stack([
            student_log_p + torch.log(1 - alpha_t),
            teacher_log_p + torch.log(alpha_t),
        ]),
        dim=0,
    )

    kl_teacher = F.kl_div(
        mixture_log_p, teacher_log_p, reduction="none", log_target=True
    )
    kl_student = F.kl_div(
        mixture_log_p, student_log_p, reduction="none", log_target=True
    )

    jsd = alpha * kl_teacher + (1 - alpha) * kl_student
    return jsd.sum(-1)


# ---------------------------------------------------------------------------
# SDPO Trainer
# ---------------------------------------------------------------------------

class SDPOTrainer(Trainer):
    """On-policy self-distilled policy optimization trainer.

    Each training step:
      1. Generate G completions per question from student  [no grad]
      2. Score completions (correct answer?)
      3. Build teacher prompts using successful peer solutions
      4. Teacher forward on  (question + peer_solution || y)  [no grad]
      5. Student forward on  (question || y)                  [with grad]
      6. Loss = JSD(student, teacher) over top-K logits + tail bucket
    """

    def __init__(
        self,
        *args,
        teacher_model: AutoModelForCausalLM,
        sdpo_tokenizer: AutoTokenizer,
        num_generations: int = 8,
        max_new_tokens: int = 4096,
        jsd_alpha: float = 0.5,
        distillation_topk: int | None = 100,
        success_reward_threshold: float = 0.5,
        use_gold_fallback: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.sdpo_tokenizer = sdpo_tokenizer
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.jsd_alpha = jsd_alpha
        self.distillation_topk = distillation_topk
        self.success_reward_threshold = success_reward_threshold
        self.use_gold_fallback = use_gold_fallback

    def _score_completion(self, completion_ids: torch.Tensor, gold_answer: str) -> tuple[float, str]:
        """Decode a completion and check correctness against gold answer."""
        text = self.sdpo_tokenizer.decode(completion_ids, skip_special_tokens=True)
        pred = extract_boxed_answer(text)
        if pred is not None and is_equiv(pred, gold_answer):
            return 1.0, text
        return 0.0, text

    def _build_teacher_prompt_ids(self, question: str, demonstration: str) -> torch.Tensor:
        """Tokenize a teacher prompt (question + demonstration)."""
        teacher_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEACHER_TEMPLATE.format(
                question=question, demonstration=demonstration,
            )},
        ]
        text = self.sdpo_tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True,
        )
        return self.sdpo_tokenizer.encode(text, return_tensors="pt")[0]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        s_ids  = inputs["student_input_ids"]       # [B, Ls]
        s_mask = inputs["student_attention_mask"]   # [B, Ls]
        questions      = inputs["questions"]        # list[str], len B
        answers        = inputs["answers"]          # list[str], len B
        gold_solutions = inputs["gold_solutions"]   # list[str], len B

        B      = s_ids.shape[0]
        G      = self.num_generations
        device = s_ids.device
        pad_id = self.sdpo_tokenizer.pad_token_id

        # ── 1. On-policy generation: G completions per question (no grad) ─────
        model.eval()
        # Per-question lists of length G
        all_completions: list[list[torch.Tensor]] = []
        all_rewards:     list[list[float]]        = []
        all_texts:       list[list[str]]          = []
        student_prompt_ids: list[torch.Tensor]    = []

        with torch.no_grad():
            for i in range(B):
                sp = s_ids[i][s_mask[i].bool()]
                student_prompt_ids.append(sp)
                q_comp, q_rew, q_txt = [], [], []
                for _ in range(G):
                    gen = model.generate(
                        sp.unsqueeze(0),
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        pad_token_id=pad_id,
                        eos_token_id=self.sdpo_tokenizer.eos_token_id,
                    )
                    comp = gen[0, sp.shape[0]:]
                    reward, text = self._score_completion(comp, answers[i])
                    q_comp.append(comp)
                    q_rew.append(reward)
                    q_txt.append(text)
                all_completions.append(q_comp)
                all_rewards.append(q_rew)
                all_texts.append(q_txt)

        model.train()

        # Log reward stats
        flat_rewards = [r for q_rew in all_rewards for r in q_rew]
        n_success = sum(1 for r in flat_rewards if r >= self.success_reward_threshold)
        self._metrics = getattr(self, "_metrics", {})

        # ── 2. Build teacher prompts from successful peers ────────────────────
        flat_student_prompts:  list[torch.Tensor] = []
        flat_teacher_prompts:  list[torch.Tensor] = []
        flat_completions:      list[torch.Tensor] = []
        distillation_mask:     list[float]         = []

        for i in range(B):
            success_indices = [
                g for g in range(G)
                if all_rewards[i][g] >= self.success_reward_threshold
            ]

            for g in range(G):
                flat_completions.append(all_completions[i][g])
                flat_student_prompts.append(student_prompt_ids[i])

                # Peer = successful rollout of same question, *different* index
                peer_indices = [j for j in success_indices if j != g]

                if peer_indices:
                    demo = all_texts[i][peer_indices[0]]
                    t_ids = self._build_teacher_prompt_ids(questions[i], demo)
                    flat_teacher_prompts.append(t_ids.to(device))
                    distillation_mask.append(1.0)
                elif self.use_gold_fallback:
                    t_ids = self._build_teacher_prompt_ids(
                        questions[i], gold_solutions[i],
                    )
                    flat_teacher_prompts.append(t_ids.to(device))
                    distillation_mask.append(1.0)
                else:
                    # Masked out — use student prompt as placeholder
                    flat_teacher_prompts.append(student_prompt_ids[i])
                    distillation_mask.append(0.0)

        distillation_mask = torch.tensor(distillation_mask, device=device)

        # If no sample has a peer, return zero loss
        if distillation_mask.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero, None) if return_outputs else zero

        # ── 3. Padded sequences  [prompt || completion] ───────────────────────
        N = B * G
        sf_ids, sf_mask, comp_s = _build_padded_batch(
            flat_student_prompts, flat_completions, pad_id,
        )
        tf_ids, tf_mask, comp_t = _build_padded_batch(
            flat_teacher_prompts, flat_completions, pad_id,
        )

        sf_ids  = sf_ids.to(device);  sf_mask = sf_mask.to(device)
        tf_ids  = tf_ids.to(device);  tf_mask = tf_mask.to(device)
        comp_s  = comp_s.to(device);  comp_t  = comp_t.to(device)

        # ── 4. Teacher forward (no grad) ──────────────────────────────────────
        with torch.no_grad():
            t_logits = self.teacher_model(
                input_ids=tf_ids, attention_mask=tf_mask,
            ).logits
            teacher_log_p = torch.log_softmax(
                t_logits[:, :-1, :][comp_t].float(), dim=-1,
            )  # [total_tokens, V]
            del t_logits

        # ── 5. Student forward (with grad) ────────────────────────────────────
        s_logits = model(
            input_ids=sf_ids, attention_mask=sf_mask,
        ).logits
        student_log_p = torch.log_softmax(
            s_logits[:, :-1, :][comp_s].float(), dim=-1,
        )  # [total_tokens, V]
        del s_logits

        # ── 6. JSD over top-K + tail or full vocab ───────────────────────────
        if self.distillation_topk is not None:
            K = self.distillation_topk
            # Use student's top-K indices for both (matches reference impl)
            topk_vals, topk_idx = student_log_p.topk(K, dim=-1)   # [T, K]
            teacher_topk = teacher_log_p.gather(-1, topk_idx)      # [T, K]

            student_dist = _add_tail_bucket(topk_vals)    # [T, K+1]
            teacher_dist = _add_tail_bucket(teacher_topk)  # [T, K+1]
        else:
            student_dist = student_log_p
            teacher_dist = teacher_log_p

        per_token_jsd = _compute_jsd(
            student_dist, teacher_dist, alpha=self.jsd_alpha,
        )  # [total_tokens]

        del student_log_p, teacher_log_p

        # ── 7. Mask and aggregate: per-sequence mean, then batch mean ─────────
        offset = 0
        seq_losses = []
        for idx in range(N):
            n = int(comp_s[idx].sum().item())
            if n > 0 and distillation_mask[idx] > 0:
                seq_losses.append(per_token_jsd[offset:offset + n].mean())
            offset += n

        if not seq_losses:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return (zero, None) if return_outputs else zero

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
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
    )
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

    train_ds = ds.map(format_sdpo, remove_columns=ds.column_names, num_proc=4)

    # Filter by student prompt length
    def _fits(example):
        text = tokenizer.apply_chat_template(
            example["student_messages"], tokenize=False, add_generation_prompt=True,
        )
        return len(tokenizer.encode(text)) <= args.max_prompt_length

    pre = len(train_ds)
    train_ds = train_ds.filter(_fits, num_proc=4)
    print(f"Filtered by max_prompt_length={args.max_prompt_length}: {pre} -> {len(train_ds)}")

    # ── Eval set ─────────────────────────────────────────────────────────────
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
        eval_strategy="no",
        seed=args.seed,
        report_to="tensorboard",
        remove_unused_columns=False,
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
    trainer = SDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=SDPODataCollator(tokenizer, max_prompt_length=args.max_prompt_length),
        callbacks=callbacks,
        # SDPO-specific
        teacher_model=teacher_model,
        sdpo_tokenizer=tokenizer,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        jsd_alpha=args.jsd_alpha,
        distillation_topk=args.distillation_topk,
        success_reward_threshold=args.success_reward_threshold,
        use_gold_fallback=args.use_gold_fallback,
    )
    if args.eval_accuracy:
        eval_cb._trainer = trainer

    # ── Summary ───────────────────────────────────────────────────────────────
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = len(train_ds) // effective_batch
    print(f"\n{'='*60}")
    print(f"SDPO training plan:")
    print(f"  Train size:          {len(train_ds)}")
    print(f"  Devices:             {num_devices}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Max steps:           {args.max_steps}")
    print(f"  Num generations:     {args.num_generations}")
    print(f"  Max new tokens:      {args.max_new_tokens}")
    print(f"  JSD alpha:           {args.jsd_alpha}")
    print(f"  Distillation top-K:  {args.distillation_topk}")
    print(f"  EMA alpha:           {args.ema_alpha}")
    print(f"  Success threshold:   {args.success_reward_threshold}")
    print(f"  Gold fallback:       {args.use_gold_fallback}")
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
    parser = argparse.ArgumentParser(description="SDPO on NuminaMath-1.5")

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/sdpo")

    # Data
    parser.add_argument("--max_samples", type=int, default=None)

    # Generation (on-policy sampling)
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of rollouts per question (G)")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max completion length for on-policy generation")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="Max tokenized length for student/teacher prompts")

    # SDPO-specific hyperparameters
    parser.add_argument("--jsd_alpha", type=float, default=0.5,
                        help="JSD mixture weight (0.5 = symmetric JSD)")
    parser.add_argument("--distillation_topk", type=int, default=100,
                        help="Top-K logits for distillation (0 = full vocab)")
    parser.add_argument("--ema_alpha", type=float, default=0.05,
                        help="EMA rate for teacher update: phi <- alpha*theta + (1-alpha)*phi")
    parser.add_argument("--success_reward_threshold", type=float, default=0.5,
                        help="Min reward for a rollout to serve as peer demonstration")
    parser.add_argument("--use_gold_fallback", action="store_true",
                        help="Fall back to gold solution when no peer rollout succeeds")

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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Convert 0 -> None for full-vocab mode
    if args.distillation_topk == 0:
        args.distillation_topk = None

    train(args)


if __name__ == "__main__":
    main()
