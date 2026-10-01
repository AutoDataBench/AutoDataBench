"""Validation for retrieval training datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RetrievalValidator:
    required = {"query", "positive_doc", "hard_negative_docs"}
    max_issues = 10

    def validate(self, dataset_path: str) -> dict[str, Any]:
        path = Path(dataset_path)
        if not path.is_file():
            return self._result(0, [{"row": -1, "msg": "file not found"}], [])

        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        negative_lengths: dict[int, int] = {}
        n_rows = 0

        with path.open() as file:
            for line_number, line in enumerate(file, start=1):
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

                missing = self.required - row.keys()
                if missing:
                    errors.append(
                        {"row": line_number, "msg": f"missing fields: {sorted(missing)}"}
                    )
                    continue
                for field in ("query", "positive_doc"):
                    if not isinstance(row[field], str) or not row[field].strip():
                        errors.append(
                            {"row": line_number, "msg": f"{field} must be a non-empty string"}
                        )

                negatives = row["hard_negative_docs"]
                if not isinstance(negatives, list):
                    errors.append(
                        {"row": line_number, "msg": "hard_negative_docs must be a list"}
                    )
                else:
                    negative_lengths.setdefault(len(negatives), line_number)
                    if len(negatives) > 20:
                        errors.append(
                            {"row": line_number, "msg": "at most 20 hard negatives are allowed"}
                        )
                    for value in negatives:
                        if not isinstance(value, str) or not value.strip():
                            errors.append(
                                {
                                    "row": line_number,
                                    "msg": "hard negatives must be non-empty strings",
                                }
                            )
                            break
                if len(errors) >= self.max_issues:
                    break

        if n_rows == 0:
            errors.append({"row": -1, "msg": "dataset is empty"})
        if len(negative_lengths) > 1:
            errors.append(
                {
                    "row": -1,
                    "msg": "all rows must contain the same number of hard negatives",
                }
            )
        return self._result(n_rows, errors, warnings)

    def _result(self, n_rows, errors, warnings):
        return {
            "valid": not errors,
            "n_rows": n_rows,
            "errors": errors[: self.max_issues],
            "warnings": warnings[: self.max_issues],
        }

