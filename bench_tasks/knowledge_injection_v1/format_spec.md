# OPSD Submission Format Specification

## Required Format

Submissions must be UTF-8 encoded JSONL (JSON Lines) files. Each line is a separate JSON object representing one training example.

## Required Fields

Every row must contain exactly these four required fields:

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `context_qid` | string | Wikidata QID identifying the source context | Must match pattern `Q[1-9][0-9]*` and exist in the public context pool |
| `context` | string | The full Wikipedia summary extract for this QID | Must exactly match the frozen `summary_extract` from the public pool; no rewriting, truncation, or concatenation |
| `question` | string | A factual question answerable from the context | Must end with `?`, be 4-80 words, and have a unique answer supported by the context |
| `teacher_continuation` | string | A concise, grounded answer to the question | Must be non-empty, ≤4000 characters, and directly supported by the context |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for this row (auto-generated if omitted) |
| `metadata` | object | Additional metadata (must not contain benchmark answers or hidden labels) |

No other fields are allowed. Unknown fields will cause validation failure.

## Validation Rules

1. **QID validity**: `context_qid` must exist in `data/public/context_pool.jsonl`
2. **Context integrity**: `context` must be byte-identical to the `summary_extract` for that QID
3. **Question format**: Must end with `?` and be 4-80 words
4. **Answer support**: `teacher_continuation` must be grounded in the paired `context`
5. **No duplicates**: 
   - No duplicate `id` values
   - No duplicate `(context_qid, question)` pairs (case-insensitive, whitespace-normalized)
6. **Row limit**: Maximum 10,000 rows per submission
7. **Non-empty**: Submission must contain at least one row

## Example Valid Row

```json
{
  "context_qid": "Q101003145",
  "context": "An earthquake with a moment magnitude of 7.0 occurred on 30 October 2020 about 14 km (8.7 mi) northeast of the Greek island of Samos. Although Samos was closest to the epicentre, it was the large Turkish city İzmir, 70 km (43 mi) northeast that was heavily affected—more than 700 residential and commercial structures were seriously damaged or destroyed. One hundred and seventeen people died in İzmir Province while an additional 1,034 were injured. In Greece, there were two fatalities and 19 injured.",
  "question": "How many people died in İzmir Province due to the 2020 Aegean Sea earthquake?",
  "teacher_continuation": "One hundred and seventeen people died in İzmir Province."
}
```

## Example with Optional Fields

```json
{
  "id": "sample-001",
  "context_qid": "Q101206",
  "context": "Ottmar Schreiner was a German lawyer and left-wing politician. He was known as one of the leading leftists in his party, SPD.",
  "question": "What political party was Ottmar Schreiner a member of?",
  "teacher_continuation": "Ottmar Schreiner was a member of the SPD (Social Democratic Party of Germany).",
  "metadata": {
    "domain": "politics_government",
    "retrieval_method": "entity_match",
    "confidence": 0.95
  }
}
```

## Validation Tool

Validate your submission locally before submitting:

```bash
python scripts/validate_submission.py --submission your_file.jsonl
```

The validator will report:
- Number of rows validated
- Number of unique contexts used
- Any validation errors (with row numbers)
- Whether the submission passes all checks

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `unknown context_qid` | QID not in public pool | Only use QIDs from `context_pool.jsonl` |
| `context is not the frozen paragraph` | Context was modified | Copy `summary_extract` exactly, no edits |
| `question must be a 4-80 word question` | Question too short/long or missing `?` | Ensure question ends with `?` and is 4-80 words |
| `duplicate context/question pair` | Same QID+question appears twice | Vary questions or use different contexts |
| `submission exceeds 10000 rows` | Too many rows | Reduce to ≤10,000 rows |

## Training Pipeline

After validation, the fixed harness executes:

1. **Validation**: `validate_submission.py` checks format and normalizes the submission
2. **Training**: `train_fixed.py` runs offline OPSD with LoRA (multi-GPU DDP)
3. **Merging**: `merge_fixed.py` merges the LoRA adapter into the base model
4. **Evaluation**: `evaluate_fixed.py` runs private novel/retention benchmarks (vLLM tensor parallel)
5. **Scoring**: Computes `macro_normalized = (novel + retention) / 2`

The agent does not control any of these steps; only the input JSONL submission.
