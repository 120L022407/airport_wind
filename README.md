# airport_wind_lab

`airport_wind_lab` is a small, configuration-driven research framework for
airport wind-speed forecasting.

Current foundation scope:

- one Python training entry point: `scripts/train.py`;
- one Python evaluation entry point: `scripts/evaluate.py`;
- one baseline experiment: `config/gru/baseline_hourly.yaml`;
- CPU-only local validation with synthetic or tiny fixture data.

The repository does not regenerate raw data or chronological splits. It consumes
already-preprocessed split files and keeps training, validation, and test order
unchanged.

## Local commands

```bash
make test
make typecheck
make check
python scripts/train.py --config config/gru/baseline_hourly.yaml
python scripts/evaluate.py --run-dir outputs/<run-name>
```

Real research training should be run manually on the server with prepared data.
