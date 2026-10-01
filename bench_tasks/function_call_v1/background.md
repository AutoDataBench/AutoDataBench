# Function Calling Data Curation Task Background

## Task Overview

This task evaluates whether an agent can curate high-quality single-turn
function-calling (FC) training data from a noisy pool. Your only control is
the training dataset you submit; the base model, training recipe, and
evaluation pipeline are fixed. You iterate: submit a dataset → receive test
scores → refine your dataset.

## Base Model & Training Recipe (fixed, fully disclosed)

- **Base model**: Qwen2-1.5B-Instruct
- **Fine-tuning**: LoRA (r=32, alpha=64, dropout=0, targets = q/k/v/o/gate/up/down proj)
- **Optimization**: lr=5e-5, cosine schedule, 1 epoch, effective batch 32,
  max sequence length 2048, assistant-only loss (the model is trained to
  produce the gold call JSON given system+user context)
- **Determinism**: fixed training seed + greedy decoding. Identical
  submissions produce identical scores.

## Evaluation Protocol (fixed, fully disclosed)

**Prompt template** (training and inference use exactly this):

```
system: You are a function calling assistant. Decide which function(s) to call.
Respond ONLY with a JSON array: [{"name": ..., "arguments": {...}}].
If no function is relevant, respond with [].
Available tools: <tools JSON>

user: <query>

assistant: <gold_calls JSON>   (training target / model output)
```

**Inference**: greedy decoding (`do_sample=False`, `max_new_tokens=512`).

**AST matching rules**:
- Function name must match exactly (case-insensitive, stripped)
- All gold arguments must be present with matching values; EXTRA arguments
  fail (hallucination is penalized); numeric values compare by value (5 == 5.0)
- `parallel` / `parallel_multiple`: order-insensitive exact set match
  (Hungarian bipartite matching over calls)
- `irrelevance`: predicting `[]` is correct

**Primary metric**: `macro = mean(per-category accuracy)` over five categories.
Diagnostic sub-metrics (not ranked): function-selection accuracy and
argument accuracy given correct selection.

## The Five Categories (BFCL semantics)

| Category | Shape |
|---|---|
| `simple` | 1 tool available, 1 call |
| `multiple` | >1 tools available, 1 call (must pick the right one) |
| `parallel` | >1 calls, all to the same function |
| `parallel_multiple` | >1 calls across >1 distinct functions |
| `irrelevance` | no available tool fits the query → correct answer is `[]` |

## Test Set (your iteration signal AND the settlement set)

- 2,000 rows: **400 per category (balanced)**
- The test functions were held out at dataset construction: **no test
  function ever appears in the training pool** — test performance measures
  generalization to unseen schemas, not memorization
- Every `train_and_eval` call returns the test macro score plus the
  per-category breakdown. There is no separate validation set: the test is
  both your feedback signal and the final settlement metric
- Test rows are clean (no label corruption)

## Data Pool

The pool derives from xLAM-function-calling-60k (APIGen-generated, canonical
schema normalized). ~40,000 rows with the **natural** category distribution
(imbalanced — inspect it yourself with `data_stats`). The pool is NOT clean:
it contains various quality issues introduced at construction time. Finding,
characterizing, and dealing with them is the core of this task. Nothing about
the issue types or rates is disclosed — treat the pool like a real, messy
industrial dataset.

## Success Criteria

Maximize the test macro score under your budget. Reference points: the
zero-shot base model is weak on this test; training on a naive random subset
improves it substantially; careful curation improves it further. The gap
between naive and best-achievable is exactly what this benchmark measures.
