"""
eval.py — 单轮 FC AST 评测（自动检测 GPU 数量，多卡并行推理）
用法: python eval.py --adapter runs/random_16k/adapter --split test
流程: 合并 adapter → 多卡并行贪心推理 → AST 匹配 → 输出 macro + per-category
--split bfcl_guard 时读 artifacts/bfcl_guard.jsonl（BFCL v3 单轮 OOD 集，行带
gold_bfcl：每参数可接受值列表，走 case_match_bfcl 成员匹配，不做笛卡尔积展开）。
类别聚合按数据中实际出现的 category 动态进行（test 为 5 类，bfcl_guard 为 10 类）。
"""
import argparse, json, os, torch
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.optimize import linear_sum_assignment

_TASK_ROOT = Path(__file__).resolve().parent

NUM_GPUS = max(torch.cuda.device_count(), 1)

# ── Prompt 模板 ──
SYSTEM_TEMPLATE = (
    "You are a function calling assistant. "
    "Decide which function(s) to call to satisfy the user request.\n"
    "Respond ONLY with a JSON array of calls: "
    '[{"name": ..., "arguments": {...}}].\n'
    "If no function is relevant, respond with an empty array: [].\n"
    "Available tools:\n{tools_json}"
)

# ── AST 匹配函数 ──
def parse_output(text):
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(text[start:end+1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None

def value_match(pred_v, gold_v):
    if isinstance(gold_v, bool):
        return pred_v is gold_v or pred_v == gold_v
    if isinstance(gold_v, (int, float)):
        try:
            return float(pred_v) == float(gold_v)
        except (TypeError, ValueError):
            return False
    if isinstance(gold_v, str):
        return str(pred_v).strip().lower() == gold_v.strip().lower()
    if isinstance(gold_v, list):
        return pred_v == gold_v
    return pred_v == gold_v

def call_match(pred, gold):
    if not isinstance(pred, dict) or pred.get("name", "").strip().lower() != gold["name"].strip().lower():
        return False
    pred_args, gold_args = pred.get("arguments", {}) or {}, gold.get("arguments", {})
    if not isinstance(pred_args, dict):
        return False
    for k, v in gold_args.items():
        if k not in pred_args or not value_match(pred_args[k], v):
            return False
    if set(pred_args.keys()) - set(gold_args.keys()):
        return False
    return True

def case_match(pred_calls, gold_calls, category):
    if category.endswith("irrelevance"):
        return pred_calls == [] or pred_calls is None
    if pred_calls is None:
        return False
    if category in ("simple", "multiple"):
        return len(pred_calls) == 1 and len(gold_calls) == 1 and call_match(pred_calls[0], gold_calls[0])
    if len(pred_calls) != len(gold_calls):
        return False
    cost = [[0 if call_match(p, g) else 1 for g in gold_calls] for p in pred_calls]
    row_ind, col_ind = linear_sum_assignment(cost)
    return all(cost[r][c] == 0 for r, c in zip(row_ind, col_ind))

def func_selection_match(pred_calls, gold_calls, category):
    """诊断子指标：只比函数名集合（irrelevance 比是否 no-call）。
    gold_calls 可以是普通 gold（{"name", "arguments"}）或 BFCL gold_bfcl 条目。"""
    if category.endswith("irrelevance"):
        return pred_calls == [] or pred_calls is None
    if pred_calls is None:
        return False
    pred_names = sorted(str(p.get("name", "")).strip().lower() for p in pred_calls if isinstance(p, dict))
    gold_names = sorted(g["name"].strip().lower() for g in gold_calls)
    return pred_names == gold_names

# ── BFCL gold 匹配（gold_bfcl：每参数可接受值列表，命中任一即对）──
def call_match_bfcl(pred, gold):
    """gold: {"name": ..., "arguments": {param: [可接受值, ...]}}。

    可接受值里的 ""/None 表示该参数可省略：pred 省略该参数时，
    仅当列表含 "" 或 None 才算对；pred 给了值则须命中列表中某个值。
    pred 多出 gold 未列的参数算错（与 in-domain 口径一致）。
    """
    if not isinstance(pred, dict) or pred.get("name", "").strip().lower() != gold["name"].strip().lower():
        return False
    pred_args, gold_args = pred.get("arguments", {}) or {}, gold.get("arguments", {})
    if not isinstance(pred_args, dict):
        return False
    if set(pred_args.keys()) - set(gold_args.keys()):
        return False
    for k, vals in gold_args.items():
        if k in pred_args:
            if not any(value_match(pred_args[k], v) for v in vals):
                return False
        elif not any(v == "" or v is None for v in vals):
            return False
    return True

def case_match_bfcl(pred_calls, gold_bfcl, category):
    if category.endswith("irrelevance"):
        return pred_calls == [] or pred_calls is None
    if pred_calls is None:
        return False
    if len(pred_calls) != len(gold_bfcl):
        return False
    if category in ("simple", "live_simple"):
        return len(gold_bfcl) == 1 and call_match_bfcl(pred_calls[0], gold_bfcl[0])
    cost = [[0 if call_match_bfcl(p, g) else 1 for g in gold_bfcl] for p in pred_calls]
    row_ind, col_ind = linear_sum_assignment(cost)
    return all(cost[r][c] == 0 for r, c in zip(row_ind, col_ind))

# ── 单卡推理 worker ──
def eval_worker(gpu_id, samples, model_dir, base_model, batch_size, max_new_tokens,
                max_prompt_len, result_queue):
    """在指定 GPU 上推理一批样本"""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"   # decoder-only 批量生成必须左 padding
    # model_dir != "NONE" 时是 main 里已 merge 好的完整模型，直接加载；
    # 不能再走 PeftModel.from_pretrained（merged 目录没有 adapter_config.json）。
    load_path = base_model if model_dir == "NONE" else model_dir
    model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()

    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = []
        for s in batch:
            tools_json = json.dumps(s["tools"], ensure_ascii=False)
            prompts.append(tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_TEMPLATE.replace("{tools_json}", tools_json)},
                    {"role": "user", "content": s["query"]},
                ], tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_prompt_len).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            pred_calls = parse_output(response)
            s = batch[j]
            cat = s["category"]
            if "gold_bfcl" in s:
                correct = case_match_bfcl(pred_calls, s["gold_bfcl"], cat)
                func_sel = func_selection_match(pred_calls, s["gold_bfcl"], cat)
            else:
                variants = s.get("gold_variants") or [s["gold_calls"]]
                correct = any(case_match(pred_calls, v, cat) for v in variants)
                func_sel = any(func_selection_match(pred_calls, v, cat) for v in variants)
            results.append((cat, correct, func_sel))

    result_queue.put((gpu_id, results))

# ── 可复用流程（main 与 run_bfcl_eval.py 共用）──
def ensure_merged(adapter, base_model):
    """adapter 目录 → merged 完整模型目录（有缓存）。'NONE' → base 模型 zero-shot。
    若传入的已是完整模型目录（有 config.json、无 adapter_config.json），原样返回。"""
    if adapter == "NONE":
        return "NONE"
    adapter = Path(adapter)
    if (adapter / "config.json").exists() and not (adapter / "adapter_config.json").exists():
        return str(adapter)
    merged_dir = str(adapter.parent / "merged")
    if not os.path.exists(merged_dir):
        print("Merging adapter...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del model
    return merged_dir

def run_inference(samples, model_dir, base_model, batch_size, max_new_tokens,
                  max_prompt_len=2048):
    """多卡分片贪心推理，返回 [(category, correct, func_sel), ...]。"""
    # 分片到多卡。spawn 启动：fork 会在父进程碰过 CUDA 时炸
    # （Cannot re-initialize CUDA in forked subprocess）。
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    chunks = [samples[i::NUM_GPUS] for i in range(NUM_GPUS)]
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = ctx.Process(target=eval_worker,
                        args=(gpu_id, chunks[gpu_id], model_dir, base_model,
                              batch_size, max_new_tokens, max_prompt_len, result_queue))
        p.start()
        processes.append(p)

    all_results = []
    for _ in range(NUM_GPUS):
        gpu_id, results = result_queue.get()
        all_results.extend(results)
    for p in processes:
        p.join()
    return all_results

def score_results(all_results):
    """按数据中出现的 category 动态聚合 → macro + overall + per-category + 诊断。"""
    cat_results = {}
    for cat, correct, func_sel in all_results:
        r = cat_results.setdefault(cat, {"correct": 0, "total": 0, "func_sel": 0})
        r["correct"] += int(correct)
        r["func_sel"] += int(func_sel)
        r["total"] += 1

    per_cat = {c: r["correct"] / max(r["total"], 1) for c, r in cat_results.items()}
    macro = sum(per_cat.values()) / len(per_cat) if per_cat else 0.0
    overall = sum(r["correct"] for r in cat_results.values()) / max(len(all_results), 1)
    func_sel_acc = sum(r["func_sel"] for r in cat_results.values()) / max(len(all_results), 1)
    # argument_acc: 函数选对的样本里参数全对的比例
    func_sel_total = sum(r["func_sel"] for r in cat_results.values())
    arg_acc = sum(r["correct"] for r in cat_results.values()) / max(func_sel_total, 1)

    return {
        "macro": macro,
        "overall": overall,
        "per_category": per_cat,
        "raw_counts": cat_results,
        "diagnostics": {
            "function_selection_acc": func_sel_acc,
            "argument_acc_given_func_sel": arg_acc,
        },
    }

# ── 主流程 ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA adapter 目录，'NONE' 表示 zero-shot")
    parser.add_argument("--base_model", default=str(_TASK_ROOT / "model/Qwen2-1.5B-Instruct"))
    parser.add_argument("--split", default="test", choices=["test", "bfcl_guard"])
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--data")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(f"Detected {NUM_GPUS} GPU(s) for evaluation")

    data_path = Path(args.data) if args.data else _TASK_ROOT / "artifacts" / f"{args.split}.jsonl"
    samples = [json.loads(l) for l in open(data_path)]
    print(f"Evaluating {len(samples)} samples on {args.split}")

    model_dir = ensure_merged(args.adapter, args.base_model)
    all_results = run_inference(samples, model_dir, args.base_model,
                                args.batch_size, args.max_new_tokens)
    scores = score_results(all_results)

    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, f"{args.split}_scores.json")
    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n=== {args.split} Results ===")
    print(f"Macro: {scores['macro']:.4f}  Overall: {scores['overall']:.4f}")
    for c, acc in scores["per_category"].items():
        r = scores["raw_counts"][c]
        print(f"  {c}: {acc:.4f} ({r['correct']}/{r['total']})")
    d = scores["diagnostics"]
    print(f"[diag] function_selection_acc: {d['function_selection_acc']:.4f}  "
          f"argument_acc|func_sel: {d['argument_acc_given_func_sel']:.4f}")


if __name__ == "__main__":
    main()
