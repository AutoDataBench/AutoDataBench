# AutoDataBench

[![arXiv](https://img.shields.io/badge/arXiv-2609.40097-b31b1b.svg)](https://arxiv.org/abs/2609.40097)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-FFD21E.svg)](https://huggingface.co/AutoDataBench)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

AutoDataBench is a controlled testbed for evaluating **data intelligence**:
an agent's ability to diagnose, organize, and construct training data. It holds
models, training recipes, evaluation, and compute budgets fixed so that gains
can be attributed to the agent's data interventions.

[[Paper](https://arxiv.org/abs/2609.40097)]
[[Data and models](https://huggingface.co/AutoDataBench)]

## Benchmarks

| Task | Objective | Base model | Primary metric |
| --- | --- | --- | --- |
| Retrieval | Improve dense-retrieval training data | MiniLM | nDCG@10 |
| Knowledge injection | Teach post-1930 facts while retaining prior knowledge | Talkie-13B | Macro normalized accuracy |
| Function calling | Curate noisy tool-use training examples | Qwen2-1.5B-Instruct | Macro AST accuracy |

Each task gives the agent a data pool and a fixed resource budget. The agent
iteratively builds candidate datasets, trains and validates a model, and
submits its best dataset for held-out evaluation.

## Installation

AutoDataBench requires Python 3.10 or newer.

```bash
git clone https://github.com/AutoDataBench/AutoDataBench.git
cd AutoDataBench
python -m pip install -e .
```

Install the dependencies for the task you want to run:

```bash
python -m pip install -e '.[retrieval]'
python -m pip install -e '.[knowledge-injection]'
python -m pip install -e '.[function-call]'
```

For development, install `.[dev]` and run `pytest`.

## Running a benchmark

AutoDataBench uses [Inspect AI](https://inspect.aisi.org.uk/) as its agent
runtime. Run a task with any model supported by Inspect:

```bash
inspect eval benchmark_task.py@retrieval_v1 --model <provider/model>
inspect eval benchmark_task.py@knowledge_injection_v1 --model <provider/model>
inspect eval benchmark_task.py@function_call_v1 --model <provider/model>
```

The agent works in an isolated `/workspace` without network access. Its tools
provide controlled access to data, small-model inference, validation, and
training. This preserves the resource accounting used in the paper.

## Data and models

Datasets and model weights are distributed through the
[AutoDataBench organization on Hugging Face](https://huggingface.co/AutoDataBench),
not this Git repository. Setup instructions and expected paths are documented
for [retrieval](bench_tasks/retrieval_v1/DATA.md),
[knowledge injection](bench_tasks/knowledge_injection_v1/DATA.md), and
[function calling](bench_tasks/function_call_v1/DATA.md).

## Outputs and OOD evaluation

Runs are saved under `runs/<task>_<timestamp>_<id>/`. Each run records the
submitted and best validation datasets, the best checkpoint, task
configuration, scores, quota usage, agent messages, and a JSONL trajectory.
Only the current best checkpoint is retained.

Run the paper's OOD evaluations on a saved checkpoint with:

```bash
python -m bench_tasks.retrieval_v1.ood runs/<retrieval-run>/artifacts/best_checkpoint
python -m bench_tasks.function_call_v1.ood runs/<function-run>/artifacts/best_checkpoint
```

Function-calling OOD evaluation expects the separately distributed BFCL guard
at `data/function_call_v1/bfcl_guard.jsonl`. Knowledge injection reports novel
knowledge and retention performance in its primary evaluation.

## Repository structure

```text
bench_core/       Framework-independent task and budget contracts
bench_inspect/    Inspect AI runtime adapter and agent tools
bench_helpers/    Helpers available to agent-written Python programs
bench_tasks/      Task configurations, backends, prompts, and evaluators
benchmark_task.py Inspect task entry points
```

## Citation

If you use AutoDataBench, please cite:

```bibtex
@misc{yuan2026autodatabench,
  title         = {AutoDataBench: A Data-centric Testbed for Accelerating Auto Research},
  author        = {Ruifeng Yuan and Yizhi Li and Yaxin Du and Fengyu Cai and Yiqi Liu and Hou Pong Chan and Chenghua Lin and Yun Chen and Jian Yang and Bryan Dai and Pinyan Lu and Chenghao Xiao},
  year          = {2026},
  eprint        = {2609.40097},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.40097}
}
```
