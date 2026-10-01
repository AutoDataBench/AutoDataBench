"""
train.py — 单轮 FC LoRA SFT（自动检测 GPU 数量，支持 1-4 卡并行）

超参 follow Hammer 官方 train.sh（spec 第 5 节），唯一偏离 epochs 1→3。
用法:
  单卡: python train.py --data artifacts/baselines/random_16k.jsonl --output runs/random_16k --seed 42
  多卡(自动): 同上，内部检测 GPU 数自动调 accelerate launch
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

_TASK_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--base_model", default=str(_TASK_ROOT / "model/Qwen2.5-1.5B-Instruct"))
parser.add_argument("--max_seq_len", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=2)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=64)
parser.add_argument("--no_grad_ckpt", action="store_true",
                    help="关闭 gradient checkpointing（显存够时更快）")
args = parser.parse_args()

# ── 自动检测 GPU 数量，多卡时通过 accelerate 启动 ──
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 1 and "ACCELERATE_LAUNCHED" not in os.environ:
    print(f"Detected {NUM_GPUS} GPUs, relaunching with accelerate...")
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(NUM_GPUS),
        "--multi_gpu",
        "--mixed_precision", "bf16",
    ] + sys.argv
    os.environ["ACCELERATE_LAUNCHED"] = "1"
    sys.exit(subprocess.call(cmd))
print(f"Running with {NUM_GPUS} GPU(s)")

# ── Tokenizer ──
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 数据预处理：ChatML 格式化 + assistant-only loss mask ──
SYSTEM_TEMPLATE = (
    "You are a function calling assistant. "
    "Decide which function(s) to call to satisfy the user request.\n"
    "Respond ONLY with a JSON array of calls: "
    '[{"name": ..., "arguments": {...}}].\n'
    "If no function is relevant, respond with an empty array: [].\n"
    "Available tools:\n{tools_json}"
)

def format_example(example):
    tools_json = json.dumps(example["tools"], ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.replace("{tools_json}", tools_json)},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": json.dumps(example["gold_calls"], ensure_ascii=False)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full = tokenizer(text, truncation=True, max_length=args.max_seq_len, padding=False)
    input_ids = full["input_ids"]

    # assistant-only loss mask。
    # 注意：Qwen2.5 chat template 无 {% generation %} 标记，
    # return_assistant_tokens_mask 会静默返回全 0 —— 必须检测并降级到
    # prompt 长度定位法（单轮场景 assistant 是最后一段，可靠）。
    labels = [-100] * len(input_ids)
    assistant_mask = None
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, return_assistant_tokens_mask=True,
            return_dict=True, truncation=True, max_length=args.max_seq_len
        )
        if sum(encoded.get("assistant_masks", [])) > 0:
            assistant_mask = encoded["assistant_masks"]
    except (TypeError, KeyError):
        pass
    if assistant_mask is not None:
        labels = [tid if m == 1 else -100 for tid, m in zip(input_ids, assistant_mask)]
    else:
        # fallback: 定位 assistant 段（单轮场景下 assistant 是最后一段）
        prompt_messages = messages[:2]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=args.max_seq_len)["input_ids"])
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

# ── 加载数据 ──
# 注意：不能对原始行做 Dataset.from_list——gold_calls 里参数值类型混杂
# （str/int/bool/list + irrelevance 的空列表），Arrow 无法推断统一 schema。
# 先 format 成纯 int 列表再建 Dataset。
raw = [json.loads(l) for l in open(args.data)]
dataset = Dataset.from_list([format_example(r) for r in raw])
print(f"Loaded {len(dataset)} examples")

# ── 模型 + LoRA ──
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
except (ImportError, ValueError) as e:
    print(f"flash_attention_2 unavailable ({e}); falling back to sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
# gradient_checkpointing + LoRA 必需：否则输入 embedding 不带梯度，backward 炸
model.enable_input_require_grads()
model.print_trainable_parameters()

# ── 训练配置（多卡时 per_device_batch_size 不变，总 batch = per_device × n_gpus × grad_accum）──
training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.00833,  # Hammer 原值（≈1/120）
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    seed=args.seed,
    data_seed=args.seed,
    gradient_checkpointing=not args.no_grad_ckpt,
    # 若开 ckpt：必须 use_reentrant=False（transformers 4.51 默认 True，
    # 对 LoRA 这类大部分参数冻结的模型传梯度不正确）。
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,             # 长度分组减少 padding 浪费（序列长尾 381 vs 2124）
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,  # 多卡 DDP 必需
)

collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt", label_pad_token_id=-100)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
trainer.train()

# ── 保存（仅主进程）──
if trainer.is_world_process_zero():
    model.save_pretrained(os.path.join(args.output, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output, "adapter"))
    print(f"Saved adapter to {args.output}/adapter")
