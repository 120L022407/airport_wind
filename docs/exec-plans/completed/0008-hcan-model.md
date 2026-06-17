# 0008 HCAN Model

## Goal

Integrate HCAN into the existing unified training framework as one new model
configuration path, without adding parallel train/evaluate pipelines.

## Scope

- Add one HCAN model implementation through the existing model registry.
- Extend the generic loss path so models can contribute auxiliary loss terms
  through the existing output `aux` channel.
- Fit HCAN hierarchy boundaries from the training split only.
- Add config schema validation, baseline YAMLs, docs, and focused tests.

## Acceptance Commands

```bash
python3 -m pytest tests/test_config.py tests/test_model_shapes.py tests/test_smoke.py
python3 -m pytest
python3 -m ruff check .
PYTHONPATH=src python3 -m mypy src tests scripts
PYTHONPATH=src python3 scripts/check_architecture.py
```

## Risks

- HCAN is model-agnostic in the paper, but the current project lacks a generic
  backbone feature interface; this adaptation will use a documented internal
  GRU-style backbone while keeping the HCAN core modules faithful.
