#!/usr/bin/env python3
"""Fixed AutoDataBench knowledge-injection run.

Orchestrates: validate → train (multi-GPU DDP) → merge → evaluate (vLLM tensor parallel).
Automatically detects and uses all available GPUs.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from common import DEFAULT_MODEL
from validate_submission import validate

HERE = Path(__file__).resolve().parent


def detect_gpu_count() -> int:
    """Auto-detect available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def run_train(train_file: Path, output_dir: Path, model: str, num_gpus: int):
    """Launch training with torchrun for multi-GPU DDP."""
    train_script = HERE / "train_fixed.py"

    if num_gpus > 1:
        # Use torchrun for DDP
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            str(train_script),
            "--train-file", str(train_file),
            "--output-dir", str(output_dir),
            "--model", model,
        ]
        print(f"Launching DDP training with {num_gpus} GPUs...")
    else:
        # Single GPU: direct invocation
        cmd = [
            sys.executable, str(train_script),
            "--train-file", str(train_file),
            "--output-dir", str(output_dir),
            "--model", model,
        ]
        print("Launching single-GPU training...")

    subprocess.run(cmd, check=True)


def run_merge(adapter: Path, output: Path, base: str):
    """Merge LoRA adapter into base model."""
    subprocess.run([
        sys.executable, str(HERE / "merge_fixed.py"),
        "--adapter", str(adapter),
        "--output", str(output),
        "--base", base,
    ], check=True)


def run_evaluate(model: Path, model_name: str, run_name: str, results_root: Path,
                 gpu_memory_utilization: float, tensor_parallel_size: int):
    """Run evaluation with vLLM tensor parallelism."""
    subprocess.run([
        sys.executable, str(HERE / "evaluate_fixed.py"),
        "--model", str(model),
        "--model-name", model_name,
        "--run-name", run_name,
        "--results-root", str(results_root),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ], check=True)


def main():
    p = argparse.ArgumentParser(description="Fixed AutoDataBench knowledge-injection run")
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu-memory-utilization", type=float, default=.75)
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs to use. Default: auto-detect all available.")
    a = p.parse_args()

    # Auto-detect GPUs if not specified
    num_gpus = a.num_gpus if a.num_gpus is not None else detect_gpu_count()
    print(f"Using {num_gpus} GPU(s)")

    a.output.mkdir(parents=True, exist_ok=True)

    # Resolve model path to absolute if it's a relative local path
    if a.model and not a.model.startswith('/') and '/' in a.model:
        _project_root = HERE.parents[2]  # scripts -> task -> bench_tasks -> repository
        _candidate = _project_root / a.model
        if _candidate.exists():
            a.model = str(_candidate.resolve())

    # 1. Validate submission
    canonical = a.output / "validated_submission.jsonl"
    stats = validate(a.submission, canonical)
    print(f"Validation passed: {stats['rows']} rows")

    # 2. Train (multi-GPU DDP)
    training = a.output / "training"
    run_train(canonical, training, a.model, num_gpus)
    print("Training completed")

    # 3. Merge LoRA
    merged = a.output / "merged_model"
    run_merge(training / "final_adapter", merged, a.model)
    print("LoRA merge completed")

    # 4. Evaluate (vLLM tensor parallel)
    evaluation = a.output / "evaluation"
    run_evaluate(merged, "submission", "submission", evaluation,
                 a.gpu_memory_utilization, 1)
    print("Evaluation completed")

    # 5. Aggregate results
    score = evaluation / "submission" / "multiple_choice" / "overall_score.jsonl"
    era = {}
    for line in score.open():
        row = json.loads(line)
        if row.get("scope") == "era":
            era[row["era"]] = row
    novel = era["novel"]["accuracy_normalized"]
    retention = era["retention"]["accuracy_normalized"]
    result = {
        "primary_metric": "macro_normalized",
        "macro_normalized": (novel + retention) / 2,
        "novel_normalized": novel,
        "retention_normalized": retention,
        "submission": stats,
        "score_file": str(score),
        "num_gpus": num_gpus,
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
