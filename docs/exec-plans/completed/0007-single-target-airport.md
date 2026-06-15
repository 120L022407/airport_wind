# 0007 Single Target Airport

## Goal

Change the current forecasting target from all four airports to Shenzhen-only
(`ZGSZ`) while keeping the existing unified training and evaluation framework.

## Non-goals

- No new train/evaluate entry points.
- No new Trainer, Evaluator, Dataset, or second config system.
- No input-side airport reduction unless required by the documented data
  contract.
- No real-data training or full benchmark runs.

## Scope

- Add explicit `data.target_airports` support to the config schema.
- Keep `data.airports` as the input airport selection.
- Slice targets, masks, and target airport metadata independently from input
  airports.
- Build model output airport count from the target airport axis.
- Update evaluator reporting to label target-airport figures correctly.
- Update current experiment YAMLs, docs, and focused synthetic tests.

## Acceptance Commands

```bash
python3 -m pytest tests/test_config.py tests/test_windows.py tests/test_model_shapes.py
python3 -m pytest tests/test_smoke.py
python3 -m ruff check .
python3 -m ruff format --check .
PYTHONPATH=src python3 -m mypy src tests scripts
PYTHONPATH=src python3 scripts/check_architecture.py
```

## Risks

- Existing code assumed one airport list for both inputs and targets. This
  change introduces separate target-airport metadata and must preserve figure
  labeling and checkpoint rebuild behavior.

## Done Definition

- Input remains four-airport by default.
- Target/prediction shape becomes `[batch, 24, 1, 1]` for current configs.
- Unified training/evaluation still works without model branches.
