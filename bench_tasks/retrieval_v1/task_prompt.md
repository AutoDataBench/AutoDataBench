# Task

**Improve the training dataset for the dense retrieval model so that
the encoder, retrained on your final dataset, achieves the highest
possible nDCG@10 on the held-out test split — within the resource
budgets declared in `quota_remaining()`.**

## Workspace isolation

You operate in a sandboxed workspace at `/workspace`. The raw data pool
has been copied to your workspace directory.

**Rules:**
- All file operations must happen within `/workspace`
- Do not read, write, or explore files outside `/workspace`
- Use `python()` to process data and write results to `/workspace`
- Submit datasets using paths under `/workspace`
- **Do NOT write any GPU-based training or inference code in `python()`.** The only way to interact with GPU models is through the `small_model_call`, `small_model_embed`, and `train_and_eval` tools. You may use `python()` for data processing only.

**Network access:** The workspace has no network connectivity. External API calls, web requests, and package downloads will fail.

**Filesystem access:** You cannot read or write files outside `/workspace`. Attempts to access paths outside the workspace will result in permission errors.

**Task documentation:** The following reference files have been copied to `/workspace` for your convenience:
- `background.md` — background on dense retrieval and the training pipeline
- `format_spec.md` — required data format specification
- `task_prompt.md` — this file (task description and rules)

You can read these files anytime via `python()` to refresh your understanding.

**Creating files:** You can create new files in `/workspace` for your workflow:
- Python scripts (`.py`) to organize complex data processing logic
- Markdown notes (`.md`) to track your observations and decisions
- Intermediate datasets (`.jsonl`, `.csv`) to save processing results
- Any other files needed for your data curation work

All files you create in `/workspace` persist across `python()` calls and will be
available throughout the experiment.

**Available packages for data processing:** `python()` runs in an environment with these pre-installed libraries:
- `numpy`, `pandas` — numerical and tabular data
- `scikit-learn` — nearest neighbors, dimensionality reduction, clustering
- `faiss` (faiss-cpu) — fast similarity search and nearest neighbor retrieval on large embedding sets. IndexHNSWFlat is recommended，since it is fast even on full dataset

You may also use standard-library modules (`json`, `collections`, `random`, etc.). GPU-based models must be accessed through the dedicated tools (`small_model_call`, `small_model_embed`, `train_and_eval`), not through `python()`.

## Small model tools

The `small_model_call` and `small_model_embed` tools can be used in two ways. `small_model_call` uses Qwen3-4B-Instruct-2507. `small_model_embed` uses Qwen3-Embedding-0.6B.

**Context window**: Both models have a **4096-token** context window.

**Important: Tokens ≠ Characters**
- English text: 1 token ≈ 4 characters (4096 tokens ≈ **16000 chars**)
- Mixed/Chinese text: 1 token ≈ 2-3 characters (4096 tokens ≈ **8000-12000 chars**)
- **Do NOT truncate at 4096 characters** — that would only be ~1000 English tokens

**Automatic truncation**: Inputs longer than 4096 tokens are **automatically truncated by the model** — you do NOT need to manually truncate. The model will process the first 4096 tokens and produce a valid output.

**If you must truncate manually** (e.g., to save tokens), use these safe thresholds:
- English text: **6000-7000 characters**
- Mixed/Chinese text: **4000-5000 characters**
- **Best approach**: Don't truncate manually — let the model handle it

The tool will emit a warning if any input appears to exceed 4096 tokens.

**1. As standalone tools** (via direct tool invocation):
```
small_model_call(prompt="Rate relevance 1-5", inputs=["document text 1", "document text 2"])
small_model_embed(texts=["text to embed 1", "text to embed 2"])
```

**2. From within `python()`** (for batch processing and complex workflows):

Both bridge functions return a **dict** — the actual data lives in a sub-key:

```python
from bench_helpers.bridge import small_model_call, small_model_embed
import numpy as np

# Generation — returns dict with key "responses" (list[str])
result = small_model_call("Rate relevance 1-5", ["document text 1", "document text 2"])
scores = result["responses"]          # ["4", "2"]
# result also has: "tokens_used" (int), "n_calls" (int), "quota_warnings" (list)

# Embedding — returns dict with key "embeddings" (list[list[float]])
result = small_model_embed(["query text", "document text"])
vecs = np.array(result["embeddings"]) # shape (2, 1024), dtype float32
dim = result["dim"]                   # 1024
np.save("/workspace/embs.npy", vecs)  # save the numpy array
# result also has: "tokens_used" (int), "quota_warnings" (list)
```

### Embedding dimension and memory

`small_model_embed` uses **Qwen3-Embedding-0.6B** which produces **1024-dimensional** vectors (float32). Memory estimate:
- 100K texts × 1024 dims × 4 bytes ≈ **400 MB** per array
- Full pool (818K texts) ≈ **3.2 GB**
- **Don't over-reduce dimensions** — going below 1024-dim significantly hurts quality

### ⚠️ Query embedding instruction prefix

Qwen3-Embedding requires a task instruction prefix for **queries** to produce high-quality embeddings. Without it, embedding quality drops significantly.

**For queries** (search queries, questions) — wrap each with an instruction:
```python
task = "Given a web search query, retrieve relevant passages that answer the query"
wrapped_queries = [f"Instruct: {task}\nQuery:{q}" for q in queries]
result = small_model_embed(wrapped_queries)
```

**For documents** (passages, paragraphs) — pass as-is, no wrapper needed:
```python
result = small_model_embed(documents)
```

### Chunking strategy for the full pool: For example, split 800k data into 8 chunks of ~100K texts each. Each chunk is one bridge round and produces a ~200–400 MB `.npy` file. Concatenate with `np.concatenate()` after all chunks are done.

Both usage patterns consume the same `small_model_tokens` quota. The bridge protocol handles the GPU interaction transparently.

### ⚠️ Bridge round limit (CRITICAL)

**Each call to `small_model_call()` or `small_model_embed()` inside `python()` consumes one "bridge round."** A single `python()` invocation has a hard limit of **500 bridge rounds**. If your code makes more than 500 calls, the script is terminated and you receive an error.

**⚠️ Common mistake (causes script termination):**
```python
# BAD: 133 bridge rounds (6613 samples / 50 per batch)
for i in range(0, len(samples), 50):
    batch = samples[i:i+50]
    results = small_model_call("Rate relevance", [s['doc'] for s in batch])
    ...
```

**✅ Correct approach — maximize batch size:**
```python
# GOOD: 1 bridge round (all samples in one call)
all_prompts = [s['positive_doc'] for s in samples]
results = small_model_call("Rate relevance", all_prompts)
```

**Planning guidelines:**
- **Calculate bridge rounds before writing code**: `len(samples) / batch_size = number of rounds`
- **Maximize batch size**: `small_model_call` has no batch size limit — pass all inputs at once when possible
- **Split across multiple `python()` calls**: If you must loop, keep each `python()` call under 500 rounds, or use separate `python()` invocations

### Using small_model effectively

You have **2.5B small_model_tokens** budget. Use it for data analysis and improvement.

**Key principles:**
- **Maximize batch size**: Process as many samples as possible per call to maximize efficiency
- **Be explicit about output format**: Provide clear format specifications in your prompts
- **Save results to files**: Write to `/workspace/*.jsonl` for persistence

`small_model_call` can perform any text-in/text-out operation.

`small_model_embed` computes embeddings. 

## Pool location

The raw data pool is at `{POOL_PATH}`. Use `data_read`
to sample from it and `data_stats` to get aggregate statistics. You cannot
modify the pool.

## Data format

Each row is one training example:

```json
{"query": "...", "positive_doc": "...", "hard_negative_docs": ["neg1", "neg2"], "subset": "msmarco_passage"}
```

- `query` — search query (non-empty string)
- `positive_doc` — relevant document (non-empty string)
- `hard_negative_docs` — list of hard negative documents (max 20 items per row;
  **all rows must have the same list length** — e.g., all `[]`, or all 2 items)
- `subset` — source dataset identifier (string). The pool contains data from
  7 subsets: `msmarco_passage` (515K), `hotpotqa` (169K), `nq` (59K),
  `fever` (37K), `scidocsrr` (21K), `fiqa` (14K), `arguana` (4K).
  Total: 818K rows. This field is informational — the trainer ignores it —
  but it is useful for subset-aware curation strategies.

**Important:** The raw data pool currently has all `hard_negative_docs` set to
empty lists `[]`. You can train directly on this data (in-batch negatives will
still provide learning signal)

See `format_spec.md` for full specification.

## Building your dataset

You may produce your training dataset by any combination of:

- **Filtering** the pool (drop noisy / mislabeled rows)
- **Re-labeling** rows (with `small_model_call` as a relevance judge — but treat
  its labels as noisy)
- **Mining hard negatives** (using `small_model_embed` to embed queries and
  documents, then finding near-miss documents for each query)
- **Synthesizing** new rows (with `small_model_call`)
- **De-duplication** (remove near-duplicate queries or documents)

Write your candidate datasets to JSONL files in `/workspace`. When you're ready,
call `submit_final(dataset_path=...)` with a `/workspace` path.

## Inspect your data before training

After generating or modifying a dataset, reading back a sample of rows
before calling `train_and_eval` is one of the highest-leverage things you
can do. A quick visual check often catches issues that would otherwise waste an entire training run.

## Plan before you act

**Before making any changes**, call `quota_remaining()` to understand your budget. Consider:
- How many training runs can you afford?
- How will you allocate your small_model_tokens budget?
- What's your strategy for exploring the data vs. exploiting improvements?
- You have **24 hours total** for the experiment


**Practical guidance:**
- **All rows must have the same number of hard negatives** (e.g., all 0, all 1, or all 2). Mixed lengths cause the training framework to silently discard all hard negatives.
- Maximum 5 hard negatives per row; extras are silently truncated


### Training cost

**Each `train_and_eval` call costs:**
- 1 eval_call (from your eval_calls budget)
- Rows in your dataset (from your train_samples budget)
- Compute time for training + MTEB evaluation on 5 benchmarks

Budget your training runs carefully. Each run should be justified by a clear hypothesis.

### When to submit

**Don't submit too early.** Consider:
- How many training runs remain?
- What's the improvement trend?
- Are there benchmarks still worth improving?

`submit_final` is **terminal** — it ends the episode. Pick the path deliberately. If you exhaust the `train_samples` axis without submitting, the system will auto-submit your best-so-far snapshot.
