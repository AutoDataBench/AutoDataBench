# AutoDataBench

AutoDataBench evaluates agents that improve training datasets under explicit
resource budgets. An agent can inspect a data pool, create a candidate dataset,
train and validate a model, and submit its best dataset for held-out scoring.

This repository is being opened in small, reviewable stages. The current
version contains the framework-independent contracts. The agent runtime and
benchmark tasks will follow in later releases.

## Design

The framework has three layers:

1. `bench_core` defines tasks, resource budgets, backend interfaces, and
   best-result tracking.
2. A runtime adapter exposes those interfaces as tools to an agent and manages
   an evaluation run.
3. Each benchmark task supplies its configuration, data access, validation,
   training, and held-out scoring implementations.

The core package deliberately has no dependency on an agent framework or a
machine-learning stack.

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

