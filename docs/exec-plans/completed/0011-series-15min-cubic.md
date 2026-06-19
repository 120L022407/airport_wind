## Goal

Add `series_15min_cubic` as a supported series-like data source, plus matching
15min cube baseline and FACL experiment configs and background launch scripts
for every implemented model.

## Scope

- extend config and series data loading to accept `series_15min_cubic`;
- document the new source as sharing the `series_15min` contract with a
  different generation method;
- add `baseline_15min_cube(.yaml)` and `baseline_15min_cube_facl(.yaml)` for
  all implemented models with `batch_size: 128`;
- add background train+evaluate shell launchers for the new cube baselines;
- update focused config and series loader/window tests.

## Non-goals

- no new Python train/evaluate entry points;
- no trainer/evaluator branching by experiment name;
- no local real-data training.

## Steps

1. Extend source validation and series-like loader registration for
   `series_15min_cubic`.
2. Add the new cube configs and background launcher scripts by reusing the
   existing 15min baseline patterns.
3. Update focused docs/tests and run only the relevant config and data checks.
