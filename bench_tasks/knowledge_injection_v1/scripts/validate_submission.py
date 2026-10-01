#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import MAX_ROWS, PUBLIC, read_jsonl, write_jsonl

ALLOWED = {"id", "context_qid", "context", "question", "teacher_continuation", "metadata"}
REQUIRED = {"context_qid", "context", "question", "teacher_continuation"}
QID = re.compile(r"Q[1-9][0-9]*\Z")


def validate(submission: Path, output: Path | None = None) -> dict:
    pool = {r["qid"]: r["summary_extract"] for r in read_jsonl(PUBLIC / "context_pool.jsonl")}
    accepted, ids, pairs = [], set(), set()
    for line_no, row in enumerate(read_jsonl(submission), 1):
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: row must be an object")
        unknown = set(row) - ALLOWED
        missing = REQUIRED - set(row)
        if missing or unknown:
            raise ValueError(f"line {line_no}: missing={sorted(missing)} unknown={sorted(unknown)}")
        qid = str(row["context_qid"])
        if not QID.fullmatch(qid) or qid not in pool:
            raise ValueError(f"line {line_no}: unknown context_qid {qid!r}")
        if not str(row["context"]).strip():
            raise ValueError(f"line {line_no}: empty contexts cannot be used for training")
        if row["context"] != pool[qid]:
            raise ValueError(f"line {line_no}: context is not the frozen paragraph for {qid}")
        question = str(row["question"]).strip()
        answer = str(row["teacher_continuation"]).strip()
        if not (question.endswith("?") and 4 <= len(question.split()) <= 80):
            raise ValueError(f"line {line_no}: question must be a 4-80 word question")
        if not answer or len(answer) > 4000:
            raise ValueError(f"line {line_no}: invalid teacher_continuation")
        pair = (qid, " ".join(question.casefold().split()))
        if pair in pairs:
            raise ValueError(f"line {line_no}: duplicate context/question pair")
        pairs.add(pair)
        rid = str(row.get("id") or f"row-{line_no:06d}")
        if rid in ids:
            raise ValueError(f"line {line_no}: duplicate id {rid!r}")
        ids.add(rid)
        accepted.append({"id": rid, "qid": qid, "context": row["context"],
                         "question": question, "teacher_continuation": answer})
        if len(accepted) > MAX_ROWS:
            raise ValueError(f"submission exceeds {MAX_ROWS} rows")
    if not accepted:
        raise ValueError("submission is empty")
    if output:
        write_jsonl(output, accepted)
    return {"rows": len(accepted), "unique_contexts": len({r["qid"] for r in accepted}),
            "max_rows": MAX_ROWS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.submission, args.output), indent=2))


if __name__ == "__main__":
    main()
