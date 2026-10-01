# FC Submission Format Specification

## Required Format

Submissions must be UTF-8 encoded JSONL (JSON Lines) files. Each line is a
separate JSON object representing one training example.

## Required Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `query` | string | User's natural-language request | Non-empty |
| `tools` | list[object] | Available function schemas | Non-empty; each tool has `name`, `description`, `parameters` |
| `gold_calls` | list[object] | Target function calls | `[]` iff `category == "irrelevance"`; otherwise non-empty, each call's `name` must exist in `tools` |
| `category` | string | One of the five categories | `simple` / `multiple` / `parallel` / `parallel_multiple` / `irrelevance` |

## Tool Schema Shape

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "city": {"type": "string", "description": "City name", "required": true},
    "unit": {"type": "string", "description": "Temperature unit", "required": false}
  }
}
```

- `parameters` is an object mapping parameter name → `{type, description, required}`
- `type` ∈ `string` / `integer` / `number` / `boolean` / `array` / `object` / `any`

## Call Shape

```json
{"name": "get_weather", "arguments": {"city": "San Francisco"}}
```

- `arguments` values should be JSON-typed consistently with the parameter
  declarations (training teaches the model to respect these types)

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the row (auto-generated if omitted) |
| `meta` / `metadata` | object | Free-form metadata (must not contain benchmark answers) |

No other fields are allowed. Unknown fields cause validation failure.

## Validation Rules

1. **JSON validity**: every line parses to a JSON object
2. **Field whitelist**: only the fields above; missing required fields fail
3. **Category validity**: one of the five categories
4. **Tools non-empty**: each tool has a `name`; `parameters` is an object
5. **Call/tool agreement**: every called function exists in the row's `tools`
6. **Irrelevance contract**: `irrelevance` ⇔ `gold_calls == []`
7. **No duplicate ids**
8. **Row limit**: maximum 16,000 rows per submission; minimum 1

## Category Consistency (advisory)

The validator warns (does not fail) when the category label disagrees with
the tools/calls shape:

| category | expected shape |
|---|---|
| `simple` | 1 tool, 1 call |
| `multiple` | >1 tools, 1 call |
| `parallel` | >1 calls, 1 distinct function |
| `parallel_multiple` | >1 calls, >1 distinct functions |
| `irrelevance` | 0 calls |

## Example Valid Rows

```json
{"id": "ex-1", "query": "What's the weather in San Francisco?", "tools": [{"name": "get_weather", "description": "Get current weather for a city", "parameters": {"city": {"type": "string", "description": "City name", "required": true}}}], "gold_calls": [{"name": "get_weather", "arguments": {"city": "San Francisco"}}], "category": "simple"}
```

```json
{"id": "ex-2", "query": "Book the 9am meeting room and email the agenda to the team.", "tools": [{"name": "book_room", "description": "Book a meeting room", "parameters": {"time": {"type": "string", "description": "Start time", "required": true}}}, {"name": "send_email", "description": "Send an email", "parameters": {"to": {"type": "string", "description": "Recipient", "required": true}, "body": {"type": "string", "description": "Email body", "required": true}}}], "gold_calls": [{"name": "book_room", "arguments": {"time": "9am"}}, {"name": "send_email", "arguments": {"to": "team", "body": "agenda"}}], "category": "parallel_multiple"}
```

```json
{"id": "ex-3", "query": "Tell me a joke about penguins.", "tools": [{"name": "get_weather", "description": "Get current weather for a city", "parameters": {"city": {"type": "string", "description": "City name", "required": true}}}], "gold_calls": [], "category": "irrelevance"}
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `missing=[...]` | required field absent | add the field |
| `unknown=[...]` | field outside the whitelist | remove it |
| `call to unknown function` | gold call names a function not in `tools` | fix the call or the tools list |
| `irrelevance rows must have gold_calls == []` | non-empty calls on irrelevance row | set `gold_calls` to `[]` or change category |
| `submission exceeds 16000 rows` | too many rows | reduce to ≤ 16,000 |
| `duplicate id` | repeated id values | make ids unique |

## Training Pipeline (what happens after you submit)

1. **Validation**: this format check runs first; rejected submissions cost nothing
2. **Training**: fixed LoRA recipe, 1 epoch, fixed seed (disclosed in background.md)
3. **Evaluation**: greedy decoding over the 2,000-row held-out test; AST matching
4. **Scoring**: `macro = mean(per-category accuracy)` returned with the breakdown

The agent controls none of these steps — only the input JSONL.
