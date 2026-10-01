#!/usr/bin/env python3
"""Train Talkie with offline teacher-trajectory context distillation.

The frozen base model is the teacher and sees context plus question. The LoRA
student sees only the question. Both branches are teacher-forced over a frozen
teacher continuation. Paper-faithful OPSD is in train_opsd_online.py.

Multi-GPU: launch with torchrun for DDP training:
    torchrun --nproc_per_node=4 train_fixed.py --train-file ... --output-dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "awilliamson/talkie-1930-13b-it-vllm"
TARGET_MODULES = (
    "attn_query", "attn_key", "attn_value", "attn_resid",
    "mlp_gate", "mlp_linear", "mlp_resid",
)
SYSTEM_PROMPT = "Answer the factual question concisely."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--divergence", choices=("forward_kl", "reverse_kl", "jsd"),
                   default="forward_kl")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-answer-tokens", type=int, default=128)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", nargs="+", default=list(TARGET_MODULES))
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--seed", type=int, default=1930)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume-from", type=Path)
    return p.parse_args()


def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def read_jsonl(path, limit=None):
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            continuation = next((row.get(k) for k in (
                "teacher_continuation", "teacher_answer", "answer"
            ) if str(row.get(k, "")).strip()), None)
            missing = [k for k in ("context", "question") if not str(row.get(k, "")).strip()]
            if missing or continuation is None:
                raise ValueError(
                    f"{path}:{line_no}: requires context, question, and a frozen "
                    f"teacher_continuation (aliases: teacher_answer, answer); missing {missing}"
                )
            row = dict(row)
            row["teacher_continuation"] = str(continuation).strip()
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No training rows in {path}")
    return rows


def render_prompt(tokenizer, context, question):
    user = f"Context:\n{context}\n\nQuestion:\n{question}" if context else question
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True,
    )
    # transformers 5 returns BatchEncoding here; older versions return IDs.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if torch.is_tensor(encoded):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def fit_teacher_prompt(tokenizer, context, question, budget):
    ids = render_prompt(tokenizer, context, question)
    if len(ids) <= budget:
        return ids
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    overflow = len(ids) - budget
    keep = max(0, len(context_ids) - overflow - 8)
    while keep >= 0:
        shortened = tokenizer.decode(context_ids[-keep:] if keep else [], skip_special_tokens=True)
        ids = render_prompt(tokenizer, shortened, question)
        if len(ids) <= budget:
            return ids
        keep -= max(8, len(ids) - budget)
    raise ValueError("Question and chat template alone exceed max sequence length")


class OPSDDataset(Dataset):
    def __init__(self, rows, tokenizer, max_seq_len, max_answer_tokens):
        self.items = []
        eos = tokenizer.eos_token_id
        for row in rows:
            answer = tokenizer(
                row["teacher_continuation"], add_special_tokens=False
            ).input_ids[:max_answer_tokens]
            if eos is not None and (not answer or answer[-1] != eos):
                answer.append(eos)
            if not answer:
                continue
            budget = max_seq_len - len(answer)
            student = render_prompt(tokenizer, "", row["question"])
            if len(student) > budget:
                raise ValueError(f"Student prompt exceeds token budget for qid={row.get('qid')}")
            teacher = fit_teacher_prompt(
                tokenizer, row["context"], row["question"], budget
            )
            self.items.append({
                "qid": row.get("qid"), "teacher": teacher + answer,
                "student": student + answer, "answer_len": len(answer),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def pad_branch(items, key, pad_id):
    lengths = [len(x[key]) for x in items]
    width = max(lengths)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    positions = []
    for i, (item, length) in enumerate(zip(items, lengths)):
        ids[i, :length] = torch.tensor(item[key])
        mask[i, :length] = 1
        answer_len = item["answer_len"]
        # Logit at position j predicts token j+1.  The first continuation
        # token is predicted by the final prompt position.
        positions.extend((i, j) for j in range(length - answer_len - 1, length - 1))
    rows, cols = zip(*positions)
    return ids, mask, torch.tensor(rows), torch.tensor(cols)


def make_collator(pad_id):
    def collate(items):
        t = pad_branch(items, "teacher", pad_id)
        s = pad_branch(items, "student", pad_id)
        return {"teacher": t, "student": s, "qids": [x["qid"] for x in items]}
    return collate


class OPSDModel(torch.nn.Module):
    """Wrapper for DDP-compatible OPSD training.

    DDP requires forward passes to go through the wrapper for gradient sync.
    This module encapsulates the selected-logits computation.
    """
    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, ids, mask, rows, cols):
        """Compute logits at selected positions (student branch)."""
        base = self.peft_model.get_base_model()
        hidden = base.model(
            input_ids=ids, attention_mask=mask, return_dict=True
        ).last_hidden_state
        chosen = hidden[rows, cols]
        return base.lm_head(chosen).float()

    def teacher_logits(self, ids, mask, rows, cols):
        """Compute teacher logits with adapter disabled (no grad)."""
        with self.peft_model.disable_adapter():
            return self.forward(ids, mask, rows, cols)


def divergence_loss(teacher_logits, student_logits, kind, temperature):
    teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_p, student_p = teacher_logp.exp(), student_logp.exp()
    if kind == "forward_kl":
        token_loss = (teacher_p * (teacher_logp - student_logp)).sum(-1)
    elif kind == "reverse_kl":
        token_loss = (student_p * (student_logp - teacher_logp)).sum(-1)
    else:
        log_m = torch.logaddexp(teacher_logp, student_logp) - math.log(2.0)
        token_loss = 0.5 * (
            (teacher_p * (teacher_logp - log_m)).sum(-1)
            + (student_p * (student_logp - log_m)).sum(-1)
        )
    return token_loss.mean() * temperature ** 2


def linear_warmup_decay(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def save_checkpoint(model, optimizer, scheduler, output_dir, step, epoch, micro_step, rank=0):
    if rank != 0:
        return
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP if needed
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.peft_model.save_pretrained(checkpoint)
    torch.save({
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "global_step": step, "epoch": epoch, "micro_step": micro_step,
    }, checkpoint / "trainer_state.pt")


def main():
    a = parse_args()
    if a.temperature <= 0:
        raise ValueError("--temperature must be positive")

    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(a.seed, rank)
    if is_main:
        a.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(a.train_file, a.limit)
    dataset = OPSDDataset(rows, tokenizer, a.max_seq_len, a.max_answer_tokens)

    # Distributed sampler for multi-GPU
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None

    def epoch_loader(epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
            generator = None
        else:
            generator = torch.Generator().manual_seed(a.seed + epoch)
        return DataLoader(
            dataset, batch_size=a.batch_size, shuffle=(sampler is None),
            sampler=sampler, generator=generator,
            num_workers=a.num_workers,
            collate_fn=make_collator(tokenizer.pad_token_id),
        )

    base = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device)
    config = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=a.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    if a.resume_from:
        peft_model = PeftModel.from_pretrained(base, a.resume_from, is_trainable=True)
    else:
        peft_model = get_peft_model(base, config)
    if a.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()

    # Wrap in OPSDModel for DDP compatibility
    opsd_model = OPSDModel(peft_model)
    if world_size > 1:
        opsd_model = DDP(opsd_model, device_ids=[local_rank], find_unused_parameters=False)
    opsd_model.train()

    if is_main:
        peft_model.print_trainable_parameters()
        print(f"Training with {world_size} GPU(s)")

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=a.learning_rate, weight_decay=a.weight_decay
    )

    # Adjust gradient accumulation for multi-GPU to maintain effective batch size
    effective_grad_accum = max(1, a.gradient_accumulation_steps // world_size)
    if is_main and world_size > 1:
        print(f"Gradient accumulation adjusted: {a.gradient_accumulation_steps} -> {effective_grad_accum} "
              f"(effective batch size: {a.batch_size * world_size * effective_grad_accum})")

    micro_batches_per_epoch = math.ceil(len(dataset) / (a.batch_size * world_size))
    updates_per_epoch = math.ceil(micro_batches_per_epoch / effective_grad_accum)
    total_steps = updates_per_epoch * a.epochs
    warmup_steps = round(total_steps * a.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )
    global_step, start_epoch, start_micro_step = 0, 0, 0
    if a.resume_from:
        state = torch.load(a.resume_from / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step, start_epoch = state["global_step"], state["epoch"]
        start_micro_step = state.get("micro_step", 0)

    if is_main:
        manifest = vars(a).copy()
        manifest.update({
            "model": str(a.model), "train_file": str(a.train_file),
            "output_dir": str(a.output_dir), "rows": len(dataset),
            "target_modules": list(a.target_modules),
            "teacher": "base_model_adapter_disabled",
            "method": "offline_teacher_trajectory",
            "trajectory_policy": "frozen_privileged_teacher",
            "on_policy": False,
            "loss_positions": "teacher_continuation_only",
            "world_size": world_size,
            "effective_batch_size": a.batch_size * world_size * effective_grad_accum,
        })
        with (a.output_dir / "run_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2, default=str); f.write("\n")

    log_path = a.output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(start_epoch, a.epochs):
        loader = epoch_loader(epoch)
        for micro_step, batch in enumerate(loader, 1):
            if epoch == start_epoch and micro_step <= start_micro_step:
                continue

            # Teacher forward (no grad, adapter disabled)
            t_ids, t_mask, t_rows, t_cols = (x.to(device) for x in batch["teacher"])
            with torch.no_grad():
                if world_size > 1:
                    teacher_logits = opsd_model.module.teacher_logits(t_ids, t_mask, t_rows, t_cols)
                else:
                    teacher_logits = opsd_model.teacher_logits(t_ids, t_mask, t_rows, t_cols)

            # Student forward (through DDP wrapper for gradient sync)
            s_ids, s_mask, s_rows, s_cols = (x.to(device) for x in batch["student"])
            student_logits = opsd_model(s_ids, s_mask, s_rows, s_cols)

            loss = divergence_loss(
                teacher_logits, student_logits, a.divergence, a.temperature
            )
            (loss / effective_grad_accum).backward()

            do_update = (
                micro_step % effective_grad_accum == 0
                or micro_step == len(loader)
            )
            if not do_update:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main:
                record = {
                    "step": global_step, "epoch": epoch,
                    "loss": float(loss.detach()),
                    "lr": scheduler.get_last_lr()[0],
                    "answer_tokens": teacher_logits.shape[0],
                    "elapsed_seconds": time.time() - started,
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if global_step == 1 or global_step % a.log_every == 0:
                    print(json.dumps(record), flush=True)
                if a.save_every and global_step % a.save_every == 0:
                    save_checkpoint(opsd_model, optimizer, scheduler, a.output_dir,
                                    global_step, epoch, micro_step, rank)

        start_micro_step = 0

    # Save final adapter (only on rank 0)
    if is_main:
        unwrapped = opsd_model.module if world_size > 1 else opsd_model
        unwrapped.peft_model.save_pretrained(a.output_dir / "final_adapter")
        tokenizer.save_pretrained(a.output_dir / "final_adapter")
        save_checkpoint(opsd_model, optimizer, scheduler, a.output_dir,
                        global_step, a.epochs, 0, rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()
