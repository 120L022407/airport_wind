## Data layer

The framework consumes already-preprocessed train, validation, and test files.

Supported data sources are:

- `series`;
- `series_15min`;
- `EC`.

The authoritative human-readable definitions of file layouts, shapes,
variable ordering, airport ordering, units, masks, time resolutions, and grid
coordinates are stored in `docs/data.md`.

The data pipeline is:

```text
preprocessed split files
    -> loaders
    -> variable and airport selection
    -> source alignment
    -> normalization fitted on the training split
    -> split-local window construction
    -> model batch
```

Responsibilities:

- `data/loaders.py`: load files and validate metadata and array structure;
- `data/series.py`: organize existing train, validation, and test splits;
- `data/normalization.py`: fit on train and transform all splits;
- `data/windows.py`: construct windows independently inside each split.

The framework must not:

- recreate raw-data preprocessing;
- recreate the existing chronological splits;
- fit normalization statistics on validation or test data;
- create windows across split boundaries;
- silently align data sources with different time resolutions.