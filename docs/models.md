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
