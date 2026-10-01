# Retrieval Task — Background

## What this task is

You are improving the **training data** for a dense retrieval model. The
trainer is a small bi-encoder (`MiniLM-L6-H384-uncased`, ~22M parameters) fine-tuned on
(query, positive_doc, hard_negative_docs) examples. After training, the
encoder embeds queries and documents; relevance is the cosine similarity
between the two embeddings.

## What the data looks like

Each row in the pool is a JSON object:

```json
{
  "query": "what is photosynthesis?",
  "positive_doc": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
  "hard_negative_docs": ["The capital of France is Paris, which is known for the Eiffel Tower..."],
  "subset": "msmarco_passage"
}
```

- `query` — natural language question or search query (string, non-empty)
- `positive_doc` — a document relevant to the query (string, non-empty)
- `hard_negative_docs` — zero or more plausible-but-irrelevant documents
  (list of strings; length must be uniform across all rows in your dataset)
- `subset` — source dataset identifier (string). The pool is a mixture of
  7 subsets with different characteristics:

  | Subset | Rows | Domain |
  |---|---|---|
  | `msmarco_passage` | 515,018 | General web search |
  | `hotpotqa` | 169,032 | Multi-hop QA |
  | `nq` | 58,568 | Natural questions |
  | `fever` | 36,760 | Fact verification |
  | `scidocsrr` | 20,608 | Scientific documents |
  | `fiqa` | 14,131 | Financial QA |
  | `arguana` | 4,065 | Argumentative text |

A signle query may correspond to multiple positive_doc, which formulate 
multiple data lines with the same query but different positive_doc.

The raw data pool currently has all `hard_negative_docs` set to empty
lists `[]`. The pool may also contain other issues: mislabeled positive
docs, near-duplicate queries, or queries paired with the wrong document.
Mining good hard negatives is one of the most impactful things you can
do to improve performance.

## Training pipeline details

Understanding how the model is trained helps you make better data curation
decisions.

**Model architecture**: `MiniLM-L6-H384-uncased` (~22M params) — a BERT-based
bi-encoder with 6 layers and 384-dimensional hidden size. Queries and
documents are encoded independently into 384-dimensional vectors.
Relevance = cosine similarity.

**Loss function**: `MultipleNegativesRankingLoss`. For each training
example, the loss pushes the query embedding closer to its positive doc
and farther from all negatives. Negatives come from two sources:

1. **Explicit hard negatives** — the `hard_negative_docs` in each row.
   These are the most informative negatives when well-chosen.
2. **In-batch negatives** — the positive docs from other examples in
   the same training batch. These are automatic and free.

**Important**: The `hard_negative_docs` list must have the **same length for every row**
in your dataset. The training framework (sentence-transformers v3+) uses a
`zip(*texts)` operation internally that silently truncates at the shortest list,
discarding all hard negatives if even one row has fewer items than the others.
For example, if 99% of rows have 2 hard negatives but 1% have 0, all hard
negatives across the entire dataset will be silently ignored.

**Training configuration**:
- **1 epoch**, batch size 128, AdamW optimizer, learning rate 5e-5
  with linear warmup (10%)
- **Max sequence length 512 tokens** — query and document are encoded
  independently. Documents longer than 512 tokens are truncated.
  Very long documents waste capacity without adding signal.
- **Loss**: MultipleNegativesRankingLoss (cross-entropy over similarity
  scores, treating the positive as the correct "class" among all negatives).

**Implications for data curation**:
- Documents beyond 512 tokens are truncated, so ultra-long docs add no
  value. Consider filtering or truncating them.
- Hard negatives that are topically related but incorrect are much
  more valuable than random negatives.
- Maximum 5 hard negatives per row; extras are
  silently truncated.
- **All rows must have the same number of hard negatives.** If you cannot
  find hard negatives for some queries, either drop those rows or use a
  uniform count (e.g., all rows have 2 hard negatives, all rows have 0).

## Scoring

Your final dataset is used to retrain the encoder, which is then
evaluated on 5 MTEB retrieval benchmarks with **nDCG@10** (the higher
the better, range [0, 1]). You see val metrics from `train_and_eval`;
test metrics are revealed only after submission.

**Evaluation benchmarks** (all from MTEB/BEIR):

| Benchmark | Domain | What it tests |
|---|---|---|
| ArguAna | Argumentative | Counter-argument retrieval |
| FEVERHardNegatives | Fact-checking | Evidence retrieval with hard negatives |
| FiQA2018 | Finance | Financial QA retrieval |
| HotpotQAHardNegatives | Multi-hop | Supporting fact retrieval |
| SCIDOCS | Scientific | Scientific document citation retrieval |

The `train_and_eval` result includes:
- `val_scores`: overall nDCG@10 averaged across benchmarks (higher = better)
- `score_breakdown`: per-benchmark nDCG@10, nDCG@1, nDCG@5, MRR@10,
  MAP@10 — use this to identify which benchmarks need improvement
- `training_time_seconds` and dataset statistics
