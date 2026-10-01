"""Small JSONL helpers for agent-written Python."""

import json
import random
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def sample_jsonl(path: str, n: int, seed: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    with open(path) as file:
        for index, line in enumerate(line for line in file if line.strip()):
            row = json.loads(line)
            if index < n:
                sample.append(row)
            else:
                replacement = rng.randint(0, index)
                if replacement < n:
                    sample[replacement] = row
    return sample


def count_lines(path: str) -> int:
    with open(path) as file:
        return sum(bool(line.strip()) for line in file)
