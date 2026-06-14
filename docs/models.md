# Models

All models consume the existing forecast input contract:

```text
inputs: [batch, input_steps, airport, feature]
prediction: [batch, forecast_steps, airport, target]
```

The current baseline task uses `[batch, 24, 4, 13] -> [batch, 24, 4, 1]`.

## GRU

Faithful core:

- uses `torch.nn.GRU` over the temporal dimension;
- uses the final hidden state for direct multi-step prediction.

Project adaptation:

- airport and feature axes are flattened before the GRU;
- a projection head maps the hidden state to `[forecast_steps, airport, target]`.

## PatchTST

Faithful core, following "A Time Series is Worth 64 Words":

- splits each variate sequence into temporal patches;
- embeds patches as Transformer tokens;
- applies a channel-independent shared Transformer encoder.

Project adaptation:

- airport and feature axes are treated as flattened variates;
- the channel-independent horizon forecasts are projected to configured airport
  target variables.

## iTransformer

Faithful core, following "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting":

- treats each variate history as one token;
- applies Transformer attention across variate tokens;
- applies feed-forward blocks to the variate-token representations.

Project adaptation:

- airport and feature axes are flattened into variates;
- per-variate horizon forecasts are projected to configured airport target
  variables.

## DLinear

Faithful core, following "Are Transformers Effective for Time Series
Forecasting?":

- decomposes each input variate into trend and seasonal components with a moving
  average;
- applies linear temporal projections to trend and seasonal components.

Project adaptation:

- flattened input variates are projected to the configured airport target
  variables after the trend/seasonal forecast.

## TFPS

Faithful core, following "Learning Pattern-Specific Experts for Time Series
Forecasting Under Patch-level Distribution Shift":

- splits each variate sequence into temporal patches and adds learnable
  positional embeddings;
- uses a time-domain PatchTST-style Transformer encoder;
- uses a frequency-domain FNet-style encoder that applies 2D FFT mixing over the
  patch and hidden dimensions and keeps the real component;
- builds per-branch learnable subspace affinities for pattern identification;
- routes patch-level representations through top-k MLP pattern experts;
- concatenates time-domain and frequency-domain outputs before the forecast
  head.

Project adaptation:

- airport and feature axes are flattened into variates, matching the existing
  PatchTST adaptation;
- the final forecast head maps per-variate horizon forecasts to configured
  airport target variables;
- project-wide train-only normalization is used instead of adding RevIN by
  default;
- the current unified Trainer only supports one supervised masked loss, so TFPS
  clustering affinities are exposed in `aux` but the paper's extra clustering
  regularization is not added in this milestone.

Available ablations:

- `config/tfps/baseline_hourly.yaml`: time + frequency + pattern identifier +
  pattern experts;
- `config/tfps/time_only_hourly.yaml`: disables the frequency branch;
- `config/tfps/no_pattern_experts_hourly.yaml`: keeps dual-domain encoders but
  disables pattern identifier and pattern experts.

## TimeBridge

Faithful core, following "TimeBridge: Non-Stationarity Matters for Long-term
Time Series Forecasting":

- segments each input variate into temporal patches;
- applies integrated attention to model within-variate patch dependencies after
  TimeBridge-style stabilization;
- optionally applies patch sampling to reduce patch count before decoding;
- applies cointegrated attention across variates to preserve long-term
  non-stationary relationships;
- decodes per-variate horizon forecasts from flattened patch representations.

Project adaptation:

- the current input contract does not provide a separate `x_mark_enc`, so the
  last `shared_time_feature_count` per-airport features are treated as shared
  time channels and appended once, matching the documented `series` layout;
- airport-specific non-time features are flattened into variates, while shared
  time channels are excluded from the forecast head;
- the model keeps the released code's per-window normalization /
  de-normalization inside `forward`, on top of the framework's train-only data
  normalization;
- trainer-side options from the paper repo such as `alpha`, custom training
  schedules, and the released but unused `revin` flag are not migrated because
  they are not part of the released core model path.

Available ablations:

- `config/timebridge/baseline_hourly.yaml`: integrated attention + patch
  sampling + cointegrated attention;
- `config/timebridge/no_cointegration_hourly.yaml`: removes cointegrated
  attention while keeping the short-term branch.
