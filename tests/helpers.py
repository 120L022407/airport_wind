from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SERIES_AIRPORTS = ["ZGSZ", "ZGGG", "VHHH", "VMMC"]
SERIES_VARIABLES = [
    "sknt",
    "wind_dir_sin",
    "wind_dir_cos",
    "tmpf",
    "alti",
    "dwpf",
    "relh",
    "gust",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_norm",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_synthetic_data_root(base_path: Path) -> Path:
    data_root = base_path / "synthetic_data"
    data_root.mkdir(parents=True, exist_ok=True)
    create_series_fixture(data_root / "series", source="series")
    create_series_fixture(data_root / "series_15min", source="series_15min")
    create_ec_fixture(data_root / "EC")
    return data_root


def create_series_fixture(
    dataset_root: Path,
    source: str,
    *,
    shapes: dict[str, int] | None = None,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    split_shapes = shapes or {"train": 72, "val": 56, "test": 56}
    for split_name, time_length in split_shapes.items():
        array = rng.normal(
            loc=5.0,
            scale=2.0,
            size=(time_length, len(SERIES_AIRPORTS), len(SERIES_VARIABLES)),
        )
        np.save(dataset_root / f"{split_name}.npy", array)
        if source == "series_15min":
            mask = np.array(
                [(index % 2) == 0 for index in range(time_length)],
                dtype=bool,
            )
            np.save(dataset_root / f"{split_name}_original_mask.npy", mask)

    _write_json(
        dataset_root / "metadata.json",
        {
            "source": source,
            "airports": SERIES_AIRPORTS,
            "variables": SERIES_VARIABLES,
            "time_resolution": "15min" if source == "series_15min" else "1h",
            "units": {"sknt": "m/s", "gust": "m/s"},
        },
    )


def create_ec_fixture(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    shapes = {"train": 48, "val": 36, "test": 36}
    for split_name, time_length in shapes.items():
        array = rng.normal(loc=1.0, scale=0.5, size=(time_length, 5, 9, 9))
        np.save(dataset_root / f"{split_name}.npy", array)
