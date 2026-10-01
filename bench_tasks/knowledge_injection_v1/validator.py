"""Validation for OPSD training data."""

import json
import re
from pathlib import Path

ALLOWED = {"id", "context_qid", "context", "question", "teacher_continuation", "metadata"}
REQUIRED = {"context_qid", "context", "question", "teacher_continuation"}
QID = re.compile(r"Q[1-9][0-9]*\Z")


class KnowledgeInjectionValidator:
    def __init__(self, pool_path: str, max_rows: int = 10_000) -> None:
        self.pool_path = Path(pool_path)
        self.max_rows = max_rows
        self._contexts = None

    def _load_contexts(self):
        if self._contexts is None:
            with self.pool_path.open() as file:
                self._contexts = {row["qid"]: row["summary_extract"] for row in map(json.loads, file)}

    def validate(self, dataset_path: str):
        path = Path(dataset_path)
        if not path.is_file():
            return _result(0, [{"row": -1, "msg": "file not found"}])
        self._load_contexts()
        errors, ids, pairs = [], set(), set()
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
                qid, context = str(row["context_qid"]), str(row["context"])
                question, answer = str(row["question"]).strip(), str(row["teacher_continuation"]).strip()
                if not QID.fullmatch(qid) or qid not in self._contexts:
                    errors.append({"row": line_number, "msg": f"invalid or unknown context_qid: {qid!r}"})
                elif context != self._contexts[qid]:
                    errors.append({"row": line_number, "msg": f"context is not the frozen paragraph for {qid}"})
                if not question.endswith("?") or not 4 <= len(question.split()) <= 80:
                    errors.append({"row": line_number, "msg": "question must be 4-80 words and end with '?'"})
                if not answer or len(answer) > 4000:
                    errors.append({"row": line_number, "msg": "teacher_continuation must contain 1-4000 characters"})
                pair = (qid, " ".join(question.casefold().split()))
                row_id = str(row.get("id") or f"row-{line_number:06d}")
                if pair in pairs or row_id in ids:
                    errors.append({"row": line_number, "msg": "duplicate id or (context_qid, question) pair"})
                pairs.add(pair)
                ids.add(row_id)
                if n_rows > self.max_rows:
                    errors.append({"row": -1, "msg": f"submission exceeds {self.max_rows} rows"})
                if len(errors) >= 10:
                    break
        if n_rows == 0:
            errors.append({"row": -1, "msg": "submission is empty"})
        return _result(n_rows, errors)


def _result(n_rows, errors):
    return {"valid": not errors, "n_rows": n_rows, "errors": errors[:10], "warnings": []}
