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
- the current mask contract is point-wise; the Fourier term therefore uses only
  fully observed `[batch, airport, target]` sequences across the whole forecast
  horizon and skips partially observed sequences;
- evaluation/validation loss should be deterministic, so in `paper_random` mode
  the framework uses the scheduled expectation `p * FCL + (1 - p) * FAL` when
  gradients are disabled, while training uses stochastic switching as in the
  paper.

Supported params:

- `mode: paper_random | fal | fcl`
- `alpha`: required only for `paper_random`; the ratio of training steps where
  the probability threshold has already reached `0`
