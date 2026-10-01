#!/usr/bin/env python3
"""Multiple-choice evaluation by conditional option likelihood."""

from __future__ import annotations

import argparse, json, math, os, re, string, time, unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean

# vllm 0.8.5 defaults to V1 engine which doesn't support custom architectures;
# force V0 before any vllm import.
os.environ.setdefault("VLLM_USE_V1", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ROOT = Path(__file__).resolve().parents[3] / "data" / "knowledge_injection_v1"
MODEL = "awilliamson/talkie-1930-13b-it-vllm"
BENCHMARK = ROOT / "private"
RESULTS = ROOT / "outputs"
PERSON = {"politician", "physician", "economist", "author", "military_officer",
          "officeholder", "minister", "scientist", "statesperson", "judge",
          "civil_servant"}
INSTRUCTION = "Answer the question accurately."


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--model-name", default="awilliamson/talkie-1930-13b-it-vllm")
    p.add_argument("--results-root", type=Path, default=RESULTS)
    p.add_argument("--run-name")
    p.add_argument("--gpu-memory-utilization", type=float, default=.75)
    p.add_argument("--tensor-parallel-size", type=int, default=None,
                   help="Number of GPUs for tensor parallelism. Default: auto-detect all available GPUs.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def slug(x): return re.sub(r"[^A-Za-z0-9._-]+", "__", x).strip("_")


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).casefold()
    x = "".join(c if c not in string.punctuation else " " for c in x)
    return " ".join(x.split())


def revealed(answer, question):
    a, q = norm(answer).split(), norm(question).split()
    return bool(a) and any(q[i:i+len(a)] == a for i in range(len(q)-len(a)+1))


def load(path, limit):
    out = []
    for era, name in (("novel", "novel_eval.jsonl"),
                      ("retention", "retention_eval.jsonl")):
        with open(path/name) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit: break
                row = json.loads(line); row["eval_era"] = era; out.append(row)
    return out


def softmax(xs):
    m = max(xs); ys = [math.exp(x-m) for x in xs]; z = sum(ys)
    return [x/z for x in ys]


def lp_value(x): return float(x.logprob if hasattr(x, "logprob") else x)


def aggregate(rows, scope, **labels):
    clean = [r for r in rows if not r["answer_revealed_in_question"]]
    rec = {"scope": scope, **labels, "count": len(rows),
           "accuracy": mean(r["correct"] for r in rows) if rows else None,
           "accuracy_normalized": mean(r["correct_normalized"] for r in rows) if rows else None,
           "non_revealing_count": len(clean),
           "non_revealing_accuracy": mean(r["correct"] for r in clean) if clean else None,
           "non_revealing_accuracy_normalized": mean(r["correct_normalized"] for r in clean) if clean else None,
           "mean_correct_probability": mean(r["correct_probability_normalized"] for r in rows) if rows else None}
    return rec


def aggregates(rows):
    result = [aggregate(rows, "overall")]
    specs = [("era", lambda r:(r["era"],)),
             ("decade", lambda r:(r["era"],r["anchor_decade"])),
             ("class", lambda r:(r["era"],r["benchmark_class"])),
             ("predicate", lambda r:(r["era"],r["predicate"]))]
    for scope, fn in specs:
        groups = defaultdict(list)
        for r in rows: groups[fn(r)].append(r)
        for key, vals in sorted(groups.items()):
            labels = {"era":key[0]}
            if scope == "decade": labels["decade"] = key[1]
            elif scope == "class": labels["benchmark_class"] = key[1]
            elif scope == "predicate": labels["predicate"] = key[1]
            result.append(aggregate(vals, scope, **labels))
    return result


def main():
    a = args(); name = a.run_name or slug(a.model_name)
    outdir = a.results_root/name/"multiple_choice"
    if outdir.exists() and not a.overwrite: raise FileExistsError(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load(BENCHMARK, None)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    requests, spans = [], []
    for row in rows:
        stem = row["probe_mc"]["stem"]
        base = tok.apply_chat_template(
            [{"role":"system","content":INSTRUCTION},
             {"role":"user","content":stem}], tokenize=False,
            add_generation_prompt=True)
        base_ids = tok.encode(base, add_special_tokens=False)
        item_spans = []
        for option in row["probe_mc"]["options"]:
            option_ids = tok.encode(str(option), add_special_tokens=False)
            if not option_ids: raise ValueError(f"Empty option for {row['qid']}")
            requests.append(TokensPrompt(prompt_token_ids=base_ids+option_ids))
            item_spans.append((len(base_ids), len(option_ids)))
        spans.append(item_spans)
    sampling = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    started = time.time()

    # Auto-detect GPU count if not specified
    tp_size = a.tensor_parallel_size
    if tp_size is None:
        import torch
        tp_size = torch.cuda.device_count()
        print(f"Auto-detected {tp_size} GPUs for tensor parallelism")

    llm = LLM(model=str(a.model), model_impl="transformers", trust_remote_code=True,
              dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=a.gpu_memory_utilization,
              tensor_parallel_size=tp_size)
    outputs = llm.generate(requests, sampling)
    records=[]
    for i,row in enumerate(rows):
        sums=[]; avgs=[]; token_counts=[]
        for j,(start,count) in enumerate(spans[i]):
            output=outputs[i*4+j]; ids=output.prompt_token_ids
            lps=[]
            for pos in range(start,start+count):
                token=ids[pos]; lps.append(lp_value(output.prompt_logprobs[pos][token]))
            sums.append(sum(lps)); avgs.append(mean(lps)); token_counts.append(count)
        pred=max(range(4),key=sums.__getitem__)
        pred_norm=max(range(4),key=avgs.__getitem__)
        correct=int(row["probe_mc"]["answer_index"])
        probs=softmax(sums); probs_norm=softmax(avgs)
        domains=set(row.get("domains",[])); stem=row["probe_mc"]["stem"]
        answer=str(row["probe_mc"]["answer"])
        records.append({"index":i,"era":row["eval_era"],"anchor_year":row["anchor_year"],
          "anchor_decade":row["anchor_year"]//10*10,
          "benchmark_class":"person" if domains&PERSON else "event",
          "qid":row["qid"],"subject":row["subject"],"predicate":row["predicate"],
          "question":stem,"options":row["probe_mc"]["options"],
          "correct_index":correct,"correct_answer":answer,
          "predicted_index":pred,"predicted_answer":row["probe_mc"]["options"][pred],
          "predicted_index_normalized":pred_norm,
          "predicted_answer_normalized":row["probe_mc"]["options"][pred_norm],
          "correct":pred==correct,"correct_normalized":pred_norm==correct,
          "option_loglikelihoods":sums,"option_mean_loglikelihoods":avgs,
          "option_token_counts":token_counts,"option_probabilities":probs,
          "option_probabilities_normalized":probs_norm,
          "correct_probability_normalized":probs_norm[correct],
          "answer_revealed_in_question":revealed(answer,stem)})
    with open(outdir/"model_answers.jsonl","w") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    scores=aggregates(records)
    with open(outdir/"overall_score.jsonl","w") as f:
        for r in scores: f.write(json.dumps(r)+"\n")
    novel=[r for r in records if r["era"]=="novel" and not r["correct_normalized"] and not r["answer_revealed_in_question"]]
    retention=[r for r in records if r["era"]=="retention" and r["correct_normalized"] and not r["answer_revealed_in_question"]]
    for fn,vals in (("certified_novel.jsonl",novel),("retention_anchor.jsonl",retention)):
        with open(outdir/fn,"w") as f:
            for r in vals: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    config={"model":str(a.model),"model_name":a.model_name,"benchmark":"frozen_private_v1",
      "task":"multiple_choice_option_likelihood","primary_metric":"accuracy_normalized",
      "instruction":INSTRUCTION,"limit_per_split":None,"examples":len(records),
      "option_requests":len(requests),"certified_novel_count":len(novel),
      "retention_anchor_count":len(retention),"elapsed_seconds":time.time()-started}
    with open(outdir/"run_config.json","w") as f: json.dump(config,f,indent=2); f.write("\n")
    print(json.dumps({"output_dir":str(outdir),**scores[0],
      "certified_novel_count":len(novel),"retention_anchor_count":len(retention)},indent=2))

if __name__=="__main__": main()
