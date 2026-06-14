"""Load preprocessed split datasets from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

SERIES_SPLITS = ("train", "val", "test")
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class DatasetLoadError(ValueError):
    """Raised when a preprocessed dataset root is invalid."""


@dataclass(frozen=True)
class LoadedDataset:
    source: str
    splits: dict[str, FloatArray]
    metadata: dict[str, Any]
    original_masks: dict[str, BoolArray] | None = None


def _load_required_array(path: Path) -> FloatArray:
    if not path.is_file():
        raise DatasetLoadError(f"Missing array file: {path}")
    return cast(FloatArray, np.load(path).astype(np.float64, copy=False))


def _load_required_bool_array(path: Path) -> BoolArray:
    if not path.is_file():
        raise DatasetLoadError(f"Missing array file: {path}")
    return cast(BoolArray, np.load(path).astype(bool, copy=False))


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetLoadError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatasetLoadError("metadata.json must contain a JSON object.")
    return payload


def _load_split_arrays(dataset_root: Path) -> dict[str, FloatArray]:
    splits: dict[str, FloatArray] = {}
    for split_name in SERIES_SPLITS:
        splits[split_name] = _load_required_array(dataset_root / f"{split_name}.npy")
    return splits


def _validate_series_metadata(
    dataset_root: Path,
    source: str,
    splits: dict[str, FloatArray],
) -> tuple[dict[str, Any], dict[str, BoolArray] | None]:
    metadata = _load_metadata(dataset_root / "metadata.json")
    airports = metadata.get("airports")
    variables = metadata.get("variables")
    time_resolution = metadata.get("time_resolution")
    metadata_source = metadata.get("source")

    if not isinstance(airports, list) or not all(
        isinstance(item, str) for item in airports
    ):
        raise DatasetLoadError("metadata.airports must be a list of strings.")
    if not isinstance(variables, list) or not all(
        isinstance(item, str) for item in variables
    ):
        raise DatasetLoadError("metadata.variables must be a list of strings.")
    if not isinstance(time_resolution, str) or not time_resolution:
        raise DatasetLoadError("metadata.time_resolution must be a non-empty string.")
    if metadata_source != source:
        raise DatasetLoadError(
            f"metadata.source must be {source!r}, got {metadata_source!r}."
        )

    expected_airports = len(airports)
    expected_variables = len(variables)
    for split_name, array in splits.items():
        if array.ndim != 3:
            raise DatasetLoadError(
                f"{split_name}.npy for {source} must have shape "
                "[time, airport, variable]."
            )
        if array.shape[1] != expected_airports or array.shape[2] != expected_variables:
            raise DatasetLoadError(
                f"{split_name}.npy shape {array.shape} does not match metadata "
                f"airport={expected_airports}, variable={expected_variables}."
            )

    original_masks: dict[str, BoolArray] | None = None
    if source == "series_15min":
        original_masks = {}
        for split_name, array in splits.items():
            mask = _load_required_bool_array(
                dataset_root / f"{split_name}_original_mask.npy"
            )
            if mask.ndim != 1 or mask.shape[0] != array.shape[0]:
                raise DatasetLoadError(
                    f"{split_name}_original_mask.npy must have shape [time]."
                )
            original_masks[split_name] = cast(BoolArray, mask.astype(bool, copy=False))
    return metadata, original_masks


def _validate_ec_dataset(
    dataset_root: Path,
    splits: dict[str, FloatArray],
) -> dict[str, Any]:
    channel_count: int | None = None
    lat_size: int | None = None
    lon_size: int | None = None
    for split_name, array in splits.items():
        if array.ndim != 4:
            raise DatasetLoadError(
                f"{split_name}.npy for EC must have shape [time, channel, lat, lon]."
            )
        _, split_channels, split_lat, split_lon = array.shape
        if channel_count is None:
            channel_count = split_channels
            lat_size = split_lat
            lon_size = split_lon
            continue
        if (
            split_channels != channel_count
            or split_lat != lat_size
            or split_lon != lon_size
        ):
            raise DatasetLoadError("EC split arrays must share channel/lat/lon shape.")
    return {
        "source": "EC",
        "channels": ["t2m", "d2m", "sp", "u10", "v10"],
        "lat_size": lat_size,
        "lon_size": lon_size,
    }


def load_dataset_root(dataset_root: str | Path, source: str) -> LoadedDataset:
    """Load one supported preprocessed dataset root."""

    root_path = Path(dataset_root)
    if not root_path.is_dir():
        raise DatasetLoadError(f"Dataset root does not exist: {root_path}")
    splits = _load_split_arrays(root_path)

    if source in {"series", "series_15min"}:
        metadata, original_masks = _validate_series_metadata(root_path, source, splits)
        return LoadedDataset(
            source=source,
            splits=splits,
            metadata=metadata,
            original_masks=original_masks,
        )
    if source == "EC":
        metadata = _validate_ec_dataset(root_path, splits)
        return LoadedDataset(
            source=source,
            splits=splits,
            metadata=metadata,
            original_masks=None,
        )
    raise DatasetLoadError(f"Unsupported data source: {source}")
