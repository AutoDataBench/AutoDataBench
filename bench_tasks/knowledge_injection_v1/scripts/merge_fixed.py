#!/usr/bin/env python3
"""Merge an OPSD LoRA adapter into the vLLM-compatible Talkie base."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "awilliamson/talkie-1930-13b-it-vllm"
EXTRA_FILES = (
    "configuration_talkie.py", "modeling_talkie.py", "chat_template.jinja",
    "generation_config.json",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    marker = a.output / ".opsd_merged_scratch"
    if a.output.exists():
        if not a.overwrite:
            raise FileExistsError(a.output)
        if not marker.exists():
            raise ValueError(f"Refusing to replace unmarked directory: {a.output}")
        shutil.rmtree(a.output)
    a.output.mkdir(parents=True)
    marker.touch()

    base = AutoModelForCausalLM.from_pretrained(
        a.base, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, a.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(a.output, safe_serialization=True, max_shard_size="30GB")
    tokenizer = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    tokenizer.save_pretrained(a.output)
    local_base = Path(a.base)
    for name in EXTRA_FILES:
        if local_base.is_dir():
            source = local_base / name
        else:
            try:
                source = Path(hf_hub_download(repo_id=a.base, filename=name))
            except Exception:
                continue
        if source.exists():
            shutil.copy2(source, a.output / name)
    # save_pretrained may replace the directory contents but not the marker.
    marker.touch(exist_ok=True)
    print(f"Merged {a.adapter} -> {a.output}")


if __name__ == "__main__":
    main()
