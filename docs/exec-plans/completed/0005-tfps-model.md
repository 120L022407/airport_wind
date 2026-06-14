# 0005 TFPS Model Integration

## Goal

Add a TFPS model inspired by "Learning Pattern-Specific Experts for Time Series
Forecasting Under Patch-level Distribution Shift" to the existing unified
forecasting framework for the current 24h -> 24h airport wind-speed task.

## Non-goals

- Do not add training, evaluation, dataset, or configuration entry points.
- Do not migrate the paper repository's Trainer, Evaluator, Dataset, or CLI.
- Do not change `ForecastBatch` or model output contracts.
- Do not implement real-data training, GPU training, ECMWF/ERA5 fusion, or new
  losses in this milestone.
- Do not claim reproduction of paper benchmark scores.

## Scope

- Implement TFPS as one registered PyTorch model under `src/windlab/models/`.
- Extend the existing explicit model config validation for `model.name: tfps`.
- Add baseline and ablation YAML configs under `config/tfps/`.
- Document faithful components and project adaptations in `docs/models.md`.
- Add synthetic CPU tests for TFPS shape, forward, backward, batch size one, and
  invalid configuration failures.

## Design

- Input remains `[batch, input_steps, airport, feature]`.
- Output remains `{"prediction": [batch, forecast_steps, airport, target], ...}`.
- Airports and input variables are flattened into variates, matching existing
  PatchTST/iTransformer adaptations.
- Time branch uses patch embedding, learnable position embeddings, and a
  Transformer encoder over patch tokens.
- Frequency branch uses patch embedding, learnable position embeddings, and
  FNet-style 2D FFT mixing over patch and hidden dimensions.
- Each branch uses learnable subspace bases to produce pattern affinities.
- Each branch routes patch-level tokens through top-k MLP experts using the
  branch's pattern affinities.
- Branch outputs are concatenated and projected to the configured forecast
  horizon, then projected to airport target variables.

## Implementation Steps

1. Add `TFPSModel` and internal TFPS modules.
2. Register `tfps` in the model package.
3. Add explicit `tfps` config validation.
4. Add TFPS baseline and ablation configs.
5. Update model documentation with paper mapping and task adaptation.
6. Add focused synthetic tests.
7. Run the smallest relevant lint, type, and unit checks available locally.

## Acceptance Commands

```bash
python -m pytest tests/test_config.py tests/test_model_shapes.py
python -m pytest tests/test_losses.py tests/test_smoke.py
python -m ruff check src tests scripts
python -m mypy src tests scripts
```

If local PyTorch is unavailable, record the failed command and environment
blocker instead of running real training or changing dependencies.

## Risks

- The paper includes an additional clustering regularization term, but the
  current Trainer supports only one registered supervised loss without auxiliary
  model-specific terms. This milestone keeps masked MSE unchanged and exposes
  clustering outputs in `aux` for future loss-extension planning.
- Official code uses task-specific RevIN/decomposition options; project-wide
  normalization already fits train-only statistics, so this integration does not
  add RevIN by default.
- Official EDESC implementation assumes dimensions divisible by the number of
  experts. The project config must reject incompatible dimensions early.

## Done Definition

- `model.name: tfps` builds through the existing config system and registry.
- Unified training and evaluation can instantiate TFPS without Trainer/Evaluator
  branches.
- Baseline and ablation configs keep the same data split, input/output shape,
  normalization, and metrics semantics as the GRU baseline.
- Related tests and static checks pass, or any environment blocker is reported.
