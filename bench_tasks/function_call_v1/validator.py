"""Validation for function-calling training data."""

import json
from pathlib import Path

ALLOWED = {"id", "query", "tools", "gold_calls", "category", "meta", "metadata"}
REQUIRED = {"query", "tools", "gold_calls", "category"}
CATEGORIES = {"simple", "multiple", "parallel", "parallel_multiple", "irrelevance"}


class FunctionCallValidator:
    def validate(self, dataset_path: str):
        path = Path(dataset_path)
        if not path.is_file():
            return _result(0, [{"row": -1, "msg": "file not found"}], [])
        errors, warnings, ids = [], [], set()
        n_rows = 0
        with path.open() as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                n_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append({"row": line_number, "msg": f"invalid JSON: {error.msg}"})
                    continue
                if not isinstance(row, dict):
                    errors.append({"row": line_number, "msg": "row must be an object"})
                    continue
                missing, unknown = REQUIRED - row.keys(), row.keys() - ALLOWED
                if missing or unknown:
                    errors.append({"row": line_number, "msg": f"missing={sorted(missing)} unknown={sorted(unknown)}"})
                    continue
                category, tools, calls = row["category"], row["tools"], row["gold_calls"]
                if category not in CATEGORIES:
                    errors.append({"row": line_number, "msg": f"invalid category: {category!r}"})
                    continue
                if not isinstance(row["query"], str) or not row["query"].strip():
                    errors.append({"row": line_number, "msg": "query must be a non-empty string"})
                if not isinstance(tools, list) or not tools:
                    errors.append({"row": line_number, "msg": "tools must be a non-empty list"})
                    tool_names = set()
                else:
                    tool_names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
                    if len(tool_names) != len(tools) or None in tool_names or any(not isinstance(tool.get("parameters", {}), dict) for tool in tools):
                        errors.append({"row": line_number, "msg": "each tool needs a unique name and object parameters"})
                if not isinstance(calls, list):
                    errors.append({"row": line_number, "msg": "gold_calls must be a list"})
                elif category == "irrelevance" and calls:
                    errors.append({"row": line_number, "msg": "irrelevance rows require gold_calls == []"})
                elif category != "irrelevance" and not calls:
                    errors.append({"row": line_number, "msg": f"{category} requires at least one call"})
                else:
                    for call in calls:
                        if not isinstance(call, dict) or call.get("name") not in tool_names or not isinstance(call.get("arguments", {}), dict):
                            errors.append({"row": line_number, "msg": "each call must reference a listed tool and have object arguments"})
                            break
                row_id = str(row.get("id") or f"row-{line_number:06d}")
                if row_id in ids:
                    errors.append({"row": line_number, "msg": f"duplicate id: {row_id!r}"})
                ids.add(row_id)
                if n_rows > 16_000:
                    errors.append({"row": -1, "msg": "submission exceeds 16000 rows"})
                    break
                if len(errors) >= 10:
                    break
        if not n_rows:
            errors.append({"row": -1, "msg": "submission is empty"})
        return _result(n_rows, errors, warnings)


def _result(n_rows, errors, warnings):
    return {"valid": not errors, "n_rows": n_rows, "errors": errors[:10], "warnings": warnings[:10]}
