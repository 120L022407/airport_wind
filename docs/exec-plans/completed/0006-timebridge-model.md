# 0006 TimeBridge Model Integration

## Goal

Add a TimeBridge model to the existing unified forecasting framework for the
current 24h -> 24h airport wind-speed task.

## Non-goals

- No new train/evaluate entry points.
- No copied Trainer, Evaluator, Dataset, or config system from the paper repo.
- No batch/output contract changes.
- No real-data or GPU experiments in this milestone.
- No attempt to reproduce paper benchmark scores.

## Scope

- Implement the TimeBridge core model in `src/windlab/models/timebridge.py`.
- Register `timebridge` in the existing model registry.
- Extend explicit config validation for `model.name: timebridge`.
- Add baseline and minimal ablation configs under `config/timebridge/`.
- Document faithful parts and project adaptations in `docs/models.md`.
- Add focused synthetic tests and a tiny CPU smoke test through the unified
  entry points.

## Design Notes

- Keep the existing contract:
  `inputs [batch, input_steps, airport, feature] -> prediction [batch, forecast_steps, airport, target]`.
- Adapt the paper's `x_enc + x_mark_enc` input by splitting the last
  `shared_time_feature_count` features from the current airport-feature tensor
  as shared time channels, following `docs/data.md`.
- Preserve the paper's core blocks:
  patch embedding, integrated attention, optional patch sampling, and
  cointegrated attention.
- Keep the paper-style per-window normalization / de-normalization inside the
  model.
- Do not add the paper repo's trainer-side options such as custom training
  schedules or extra losses when they are not part of the released core model.

## Acceptance Commands

```bash
python3 -m pytest tests/test_config.py tests/test_model_shapes.py tests/test_smoke.py
python3 -m ruff check .
python3 -m ruff format --check .
PYTHONPATH=src python3 -m mypy src tests scripts
PYTHONPATH=src python3 scripts/check_architecture.py
python3 -m pytest
```

## Risks

- The released official model code stores a `revin` flag but performs its own
  per-window normalization directly in `forecast`; this integration follows the
  released code behavior instead of exposing a dead config flag.
- Official code expects separate temporal marker inputs; this project adapts the
  documented shared time features from the current `series` layout.
- Some official training arguments appear trainer-specific rather than model
  core; they are intentionally not migrated.

## Done Definition

- `model.name: timebridge` builds through the existing config system.
- Unified training and evaluation instantiate TimeBridge through the registry.
- Synthetic model tests, smoke test, lint, type check, architecture check, and
  full CPU test suite pass.
