# 0004 Add Forecasting Models

> Status: Active. Move to `docs/exec-plans/completed/` after implementation and
> validation.

## Goal

Add PatchTST, iTransformer, and DLinear to the existing unified training
framework through the current model registry and configuration system.

## Scope

- Preserve the existing train/evaluate entry points.
- Preserve the current batch/output contract:
  `[batch, input_steps, airport, feature] -> prediction [batch, forecast_steps, airport, target]`.
- Extend model config validation with explicit per-model fields.
- Register all new models through `MODELS`.
- Add focused model contract tests for shape, forward, backward, batch size 1,
  CPU operation, and invalid config failures.

## Non-goals

- No new Trainer, Evaluator, Dataset, train script, evaluate script, or shell
  experiment system.
- No metric, normalization, split, or data-contract changes.
- No claim of paper-result reproduction.

## References

- PatchTST: patching and channel-independent Transformer backbone.
- iTransformer: inverted variate-token Transformer.
- DLinear: decomposition plus linear trend/seasonal projection.
