## Goal

Make Fourier Amplitude and Correlation Loss mask behavior configurable through
the existing unified loss config so sparse-mask 15min experiments can opt out
of the current strict full-horizon filtering.

## Scope

- extend Fourier loss config validation with `mask_mode`;
- implement `strict_real_only` and `all_points` handling inside the existing
  reusable Fourier loss term;
- update 15min FACL baseline configs to use `mask_mode: all_points`;
- refresh focused docs and unit tests without changing trainer, evaluator, or
  batch/output contracts.

## Non-goals

- no new training or evaluation entry points;
- no metric semantic changes for `real_observation_only`;
- no dataset, model, or long-running training changes.

## Steps

1. Add backward-compatible config parsing and validation for Fourier
   `mask_mode`.
2. Refactor the Fourier term's internal mask selection so the mode decides
   whether to require fully observed sequences or use all forecast points.
3. Update the 15min FACL YAMLs, focused docs, and minimal config/loss tests,
   then run only the relevant validation scope.
