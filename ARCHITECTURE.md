# Architecture

## Scope

This repository provides a single training path and a single evaluation path
for airport wind-speed forecasting experiments.

Stage 0 implements one CPU-friendly baseline for:

- data source: `series`;
- model: `gru`;
- task: past 24 hours to next 24 hours.

Future support for `series_15min`, `EC`, additional models, and richer metrics
must extend the existing interfaces instead of adding copied entry points or
experiment-specific trainers.

## Repository structure

```text
config/
  gru/
scripts/
  train.py
  evaluate.py
src/windlab/
  config.py
  registry.py
  trainer.py
  evaluator.py
  losses.py
  metrics.py
  utils.py
  data/
  models/
tests/
```

## Entry points

- Training entry point: `scripts/train.py`
- Evaluation entry point: `scripts/evaluate.py`

These scripts only parse arguments, load configuration, and delegate to
reusable `windlab` functions.

No model-specific or experiment-specific `train_*.py` or `evaluate_*.py`
scripts are allowed.

## Data flow

```text
preprocessed split files
    -> loader validation
    -> airport and variable selection
    -> train-only normalization
    -> split-local window construction
    -> model-ready batch arrays
    -> train/evaluate pipeline
```

`docs/data.md` is the human-readable source of truth for supported data shapes,
ordering, units, and mask semantics.

## Configuration

Experiments are declared in YAML under `config/<model>/<experiment>.yaml`.

Each experiment is self-contained and must define:

- data source and data root;
- selected airports, variables, and targets;
- input and forecast lengths;
- normalization behavior;
- model settings;
- evaluation metrics and mask behavior;
- runtime output settings.

Deep inheritance is intentionally avoided. Shell wrappers may select a config
file and override operational settings such as output location, but not define a
second configuration system.

## Dependency boundaries

Allowed high-level dependency direction:

```text
scripts -> windlab.config
scripts -> windlab.trainer / windlab.evaluator

trainer / evaluator -> config, registry, data, metrics, losses, utils, models
models -> utils
data -> utils
config -> utils
registry -> no runtime business dependencies
metrics / losses -> utils
```

Forbidden patterns:

- `data` importing `trainer` or `evaluator`;
- `config` importing `trainer`, `evaluator`, or model implementations;
- duplicated Trainer classes;
- experiment-name branches inside generic modules;
- additional Python train/evaluate entry points.

The repository includes automated architecture checks for these constraints.

## Baseline training design

The local baseline is intentionally lightweight and CPU-friendly:

- input data comes from the `series` source;
- the model interface is GRU-shaped and produces
  `[batch, forecast_steps, airport, target]`;
- local fitting uses a deterministic NumPy implementation so the foundation can
  be validated without GPU, network access, or external research data.

This is a framework baseline for connectivity and contracts, not a claim of
state-of-the-art training.

## Required artifacts

Each successful training run saves:

- `config.yaml`
- `checkpoint.pt`
- `metrics.json`
- `normalization.npz`

Evaluation reuses the saved configuration, checkpoint, and normalization state
from the run directory.
