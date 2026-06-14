# 0002 PyTorch GRU Baseline

> Status: Active. Move to `docs/exec-plans/completed/` when finished.
> Testing note: implementation is being completed first; test additions and
> execution are deferred by user request on 2026-06-14.

## Goal

Replace the current NumPy placeholder GRU baseline with a real PyTorch GRU
training and evaluation path while preserving the existing single entry points,
configuration system, data contracts, split-local windows, train-only
normalization, metric masks, and experiment artifact layout.

## Non-goals

- Do not add a second training or evaluation entry point.
- Do not add experiment-specific Trainer classes.
- Do not change data layouts, split semantics, or mask definitions.
- Do not implement other models, ERA5 fusion, ECMWF, discrete prediction, or
  ramp losses.
- Do not run real research training or GPU experiments locally.

## Implementation Steps

1. Extend config schema and `config/gru/baseline_hourly.yaml` with GRU layer
   count, dropout, learning rate, batch size, epoch count, patience, weight
   decay, and device.
2. Replace `src/windlab/models/gru.py` with `torch.nn.GRU` plus a projection
   head that returns `prediction` shaped
   `[batch, forecast_steps, airport, target]`.
3. Add PyTorch masked MSE loss while preserving NumPy metric computation for
   saved/evaluation metrics.
4. Update the generic Trainer to use `DataLoader`, forward, masked loss,
   backward, optimizer, validation, early stopping, best and last checkpoints,
   and training logs.
5. Update Evaluator to load best checkpoint, rebuild the model through the
   registry, restore normalization, run independent evaluation, and save
   metrics.
6. Update tests for GRU shape, batch size 1, multi-step output, backward,
   masked loss, checkpoint reload, and CPU smoke training for at least 2 epochs.
7. Run focused tests, architecture check, type check, and the full CPU test
   suite. Move this plan to `completed/` when done.

## Acceptance

- `python scripts/train.py --config config/gru/baseline_hourly.yaml` starts a
  real PyTorch GRU training run when data is available.
- Synthetic CPU smoke training runs at least 2 epochs.
- Loss backpropagates successfully.
- Best checkpoint can be loaded by `scripts/evaluate.py`.
- Relevant tests pass without real research data, GPU, or network at runtime.
- Architecture checks still enforce one training entry point and one evaluation
  entry point.
