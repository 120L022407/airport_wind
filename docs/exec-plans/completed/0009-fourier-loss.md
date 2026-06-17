## Goal

Add the Fourier Amplitude and Correlation Loss paper method to the unified
training framework as a reusable configurable loss term that any model can
enable through YAML.

## Scope

- extend the config schema with a loss section;
- refactor the current loss builder into a composable loss pipeline;
- implement Fourier amplitude / correlation loss terms with paper-aligned
  behavior and time-series adaptation;
- keep the existing single trainer and single evaluator entry points;
- update baseline/example configs, docs, and focused tests.

## Non-goals

- no new model-specific trainers or evaluation paths;
- no real-data or long GPU training;
- no changes to batch/output contracts unless strictly required.

## Steps

1. Add execution-safe loss config parsing with a default backward-compatible
   `mse` setup for existing non-HCAN configs.
2. Convert the current loss implementation into registry-backed composable loss
   terms, including the existing HCAN auxiliary term.
3. Implement the Fourier amplitude/correlation term with documented mask and
   1D forecast-horizon adaptation.
4. Add example YAMLs and focused unit/smoke coverage, then run lint, type
   check, and the relevant CPU tests.
