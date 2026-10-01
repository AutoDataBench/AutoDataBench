# Required Dataset Format

Your final dataset (and any candidate dataset you pass to
`train_and_eval` or `validate_format`) must be a **JSONL file** —
one JSON object per line.

## Required fields

Each row MUST contain exactly these three fields:

| Field                  | Type         | Constraint                                |
|------------------------|--------------|-------------------------------------------|
| `query`                | string       | non-empty (after `.strip()`)              |
| `positive_doc`         | string       | non-empty (after `.strip()`)              |
| `hard_negative_docs`   | list[string] | max 5 items; each non-empty; **uniform length across all rows** |

Each row is one complete training example:
- `query` — the search query (anchor)
- `positive_doc` — a document relevant to the query
- `hard_negative_docs` — zero or more plausible-but-irrelevant documents.
  The list length **must be the same for every row** in the dataset (e.g., all
  rows have `[]`, or all rows have 2 items). Maximum 5 items per row; extras
  are silently truncated by the trainer. Empty strings within the list are
  also silently ignored.

Extra fields are allowed (e.g., `source`, `synth: true`) but ignored
by the trainer.

## Validation

Call `validate_format(dataset_path)` for a fast pre-check. The
returned shape:

```json
{
  "valid": true,
  "n_rows": 1024,
  "errors":   [{"row": <int>, "msg": "<reason>"}],
  "warnings": [{"row": <int>, "msg": "<reason>"}]
}
```

- `errors` block submission: `train_and_eval` rejects datasets with
  any errors and refunds the quota.
- `warnings` (e.g., empty strings in `hard_negative_docs`) do not block
  submission but indicate the item will be ignored.

The first 10 errors and 10 warnings are reported; if your dataset
has more, fix the obvious cases first and re-validate.

## Examples

```json
{"query": "what is photosynthesis?", "positive_doc": "Photosynthesis is the process by which plants convert light energy into chemical energy...", "hard_negative_docs": ["The capital of France is Paris, which is known for..."]}
{"query": "how does gravity work?", "positive_doc": "Gravity is a fundamental force of nature that attracts objects with mass...", "hard_negative_docs": ["Photosynthesis requires sunlight, water, and carbon dioxide..."]}
{"query": "best programming language?", "positive_doc": "Python is widely regarded for its readability and versatility...", "hard_negative_docs": ["The Roman Empire fell in 476 AD due to barbarian invasions..."]}
```
