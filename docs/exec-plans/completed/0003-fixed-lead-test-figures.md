# 0003 Fixed Lead Test Figures

> Status: Active. Move to `docs/exec-plans/completed/` after implementation and
> validation.

## Goal

Add fixed-lead test prediction figures to the existing evaluation flow without
adding another evaluation entry point or changing metrics, training, data
splits, or model structure.

## Implementation

- Add a reporting module that builds continuous fixed-lead series from
  `predictions[:, lead_index]`, `targets[:, lead_index]`, and
  `target_time_index[:, lead_index]`.
- Generate 8 PNG files under `outputs/<run_id>/figures/` for leads 1, 6, 12,
  and 24, each with full and first-300 views.
- Integrate reporting into `Evaluator.evaluate()` after loading the best
  checkpoint and producing test predictions.
- Print the generated figure paths from the existing `scripts/evaluate.py`
  path.
- Add focused tests for fixed-lead slicing, path generation, short series,
  length mismatch errors, and array immutability.

## Non-goals

- No overlap-window averaging.
- No flattened `[window, horizon]` plotting.
- No changes to metrics, data splits, normalization, model, or training logic.
