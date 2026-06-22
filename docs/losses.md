# Losses

The unified training framework uses configuration-driven loss composition.

Current loss config shape:

```yaml
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: fourier_amplitude_correlation
      weight: 0.2
      params:
        mode: paper_random
        alpha: 0.1
        mask_mode: strict_real_only
```

Another supported composite example:

```yaml
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
    - name: patch_wise_structural
      weight: 3.0
      params:
        patch_len_threshold: 24
        mask_mode: all_points
```

If `loss` is omitted, the framework defaults to:

```yaml
loss:
  name: composite
  terms:
    - name: mse
      weight: 1.0
```

## `mse`

- masked mean squared error over the forecast contract
  `[batch, horizon, airport, target]`;
- respects the existing observed-target mask semantics.

## `hcan_auxiliary`

- wraps the current HCAN direct + hierarchical auxiliary objective;
- requires the model output to provide `aux["hcan"]`;
- should not be combined with `mse` in the same composite config, because it
  already includes the direct forecast reconstruction term.

## `fourier_amplitude_correlation`

Faithful core from the paper "Fourier Amplitude and Correlation Loss":

- Fourier Amplitude Loss (FAL): MSE between Fourier amplitudes;
- Fourier Correlation Loss (FCL): `1 - correlation` in Fourier space;
- paper-style `paper_random` mode: training alternates between FAL and FCL with
  a threshold that decreases from `1` to `0`;
- the final selected Fourier loss is scaled by the square root of the signal
  length, matching the official implementation's scale factor adaptation.

Project adaptation for the current wind-speed task:

- the paper operates on 2D imagery, while this project predicts 1D future
  sequences, so FFT is applied only along the forecast horizon;
- the current mask contract is point-wise, so the Fourier term exposes an
  explicit `mask_mode` to control whether sparse observed masks should filter
  whole forecast sequences or be ignored for the Fourier term itself;
- evaluation/validation loss should be deterministic, so in `paper_random` mode
  the framework uses the scheduled expectation `p * FCL + (1 - p) * FAL` when
  gradients are disabled, while training uses stochastic switching as in the
  paper.

Supported params:

- `mode: paper_random | fal | fcl`
- `alpha`: required only for `paper_random`; the ratio of training steps where
  the probability threshold has already reached `0`
- `mask_mode: strict_real_only | all_points`

`mask_mode` semantics:

- `strict_real_only`: keep the previous default behavior and use only fully
  observed `[batch, airport, target]` sequences across the whole forecast
  horizon;
- `all_points`: ignore the observed-target mask inside the Fourier term and use
  every forecast sequence, which is the recommended setting for current 15min
  sparse-mask FACL baselines.

## `patch_wise_structural`

Faithful core from the paper "Patch-wise Structural Loss for Time Series
Forecasting":

- adaptively derives one patch length from the dominant Fourier period of the
  target batch;
- compares predicted and target patches through three structural components:
  linear correlation, variance-distribution KL, and patch-mean difference;
- applies the paper's gradient-norm-based dynamic weighting before summing the
  three structural components;
- is designed to be added on top of a point-wise reconstruction loss such as
  MSE, with the composite term weight acting as the paper's `ps_lambda`.

Project adaptation for the current wind-speed task:

- the paper is a reusable loss, not a new forecasting backbone, so it is
  integrated through the existing loss registry instead of adding a fake model
  type;
- the official code computes PS loss only during training and validates with
  plain MSE, so this term returns zero when gradients are disabled, preserving
  the unified trainer while matching the official validation behavior;
- the official code takes gradient norms with respect to backbone-specific
  `model.projector` parameters; the unified framework avoids model-specific loss
  hooks, so dynamic weighting is computed against the forecast prediction tensor
  itself as a model-agnostic surrogate;
- for short or high-frequency horizons, the official formula can produce
  `patch_len=0` or `stride=0`; this implementation clamps to
  `patch_len >= 2` and `stride >= 1` as a stability safeguard;
- the paper has no missing-target mask, so this term exposes explicit
  `mask_mode` handling for project datasets with sparse real-observation masks.

Supported params:

- `patch_len_threshold`: positive integer; matches the official upper bound on
  adaptive patch length
- `mask_mode: strict_real_only | all_points`

`mask_mode` semantics:

- `strict_real_only`: use only fully observed `[batch, airport, target]`
  sequences across the full forecast horizon;
- `all_points`: ignore the observed-target mask inside the PS term, which is
  the recommended setting for current 15min sparse-mask baselines.
