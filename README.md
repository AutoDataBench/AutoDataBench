# AutoDataBench

AutoDataBench evaluates agents that improve training datasets under explicit
resource budgets. An agent can inspect a data pool, create a candidate dataset,
train and validate a model, and submit its best dataset for held-out scoring.

The repository currently contains the framework-independent contracts and the
Inspect AI runtime adapter. Benchmark tasks will follow in later releases.

## Design

The framework has three layers:

1. `bench_core` defines tasks, resource budgets, backend interfaces, and
   best-result tracking.
2. A runtime adapter exposes those interfaces as tools to an agent and manages
   an evaluation run.
3. Each benchmark task supplies its configuration, data access, validation,
   training, and held-out scoring implementations.

The core package deliberately has no dependency on an agent framework or a
machine-learning stack. `bench_inspect` is the small adapter that connects the
core contracts to Inspect AI.

## Runtime lifecycle

For each sample, the adapter initializes quota and best-result state, runs the
agent with eight benchmark tools, automatically submits the best validation
result if needed, and scores the submitted dataset on the held-out split.

Agent-written Python runs without network access in a private `/workspace`.
The `bench_helpers` package lets a Python script call the same small-model,
validation, and training backends as the direct tools. Each call is handled by
the host and the script is replayed with the response cached; this preserves
the quota and best-result state used by the original experiments.

## Installation

AutoDataBench requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Data and models

Datasets, model weights, checkpoints, and experiment logs are not stored in
this repository. Download instructions and Hugging Face references will be
added with each benchmark task.

## Retrieval benchmark

The first included task is retrieval_v1. Its configuration and data setup are
in bench_tasks/retrieval_v1. Install its training stack with:

    python -m pip install -e '.[retrieval]'

Run it with:

    inspect eval benchmark_task.py@retrieval_v1 --model <provider/model>

## Knowledge-injection benchmark

`knowledge_injection_v1` constructs offline OPSD data for Talkie-13B and
scores post-1930 knowledge together with pre-1930 retention. Install its
training stack and run it with:

    python -m pip install -e '.[knowledge-injection]'
    inspect eval benchmark_task.py@knowledge_injection_v1 --model <provider/model>
