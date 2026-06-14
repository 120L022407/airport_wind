"""Selection and organization for split-local airport series data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from windlab.config import ExperimentConfig
from windlab.registry import DATA_BUILDERS

from .loaders import DatasetLoadError, load_dataset_root

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreparedSeriesSplit:
    name: str
    values: FloatArray
    targets: FloatArray
    observed_target_mask: BoolArray
    airport_ids: list[str]
    input_feature_names: list[str]
    target_feature_names: list[str]
    time_index: IntArray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedSeriesData:
    source: str
    train: PreparedSeriesSplit
    val: PreparedSeriesSplit
    test: PreparedSeriesSplit


def _resolve_indices(
    all_names: list[str],
    selected_names: list[str],
    label: str,
) -> list[int]:
    indices: list[int] = []
    missing: list[str] = []
    for name in selected_names:
        try:
            indices.append(all_names.index(name))
        except ValueError:
            missing.append(name)
    if missing:
        raise DatasetLoadError(f"Unknown {label}: {', '.join(missing)}")
    return indices


def _build_split(
    *,
    split_name: str,
    array: FloatArray,
    airport_ids: list[str],
    airport_indices: list[int],
    input_feature_names: list[str],
    input_indices: list[int],
    target_feature_names: list[str],
    target_indices: list[int],
    original_mask: BoolArray | None,
    time_resolution: str,
) -> PreparedSeriesSplit:
    selected_values = array[:, airport_indices, :][:, :, input_indices]
    targets = array[:, airport_indices, :][:, :, target_indices]
    if original_mask is None:
        observed_mask = np.ones(targets.shape, dtype=bool)
    else:
        observed_mask = np.broadcast_to(
            original_mask[:, None, None],
            targets.shape,
        ).copy()
    time_index = np.arange(array.shape[0], dtype=np.int64)
    metadata: dict[str, Any] = {
        "time_resolution": time_resolution,
        "time_length": int(array.shape[0]),
    }
    return PreparedSeriesSplit(
        name=split_name,
        values=selected_values.astype(np.float64, copy=False),
        targets=targets.astype(np.float64, copy=False),
        observed_target_mask=observed_mask,
        airport_ids=airport_ids,
        input_feature_names=input_feature_names,
        target_feature_names=target_feature_names,
        time_index=time_index,
        metadata=metadata,
    )


def build_series_data(config: ExperimentConfig) -> PreparedSeriesData:
    """Build selected series splits from a preprocessed dataset root."""

    source_root = Path(config.data.root) / config.data.source
    loaded = load_dataset_root(source_root, config.data.source)
    if config.data.source not in {"series", "series_15min"}:
        raise DatasetLoadError("build_series_data only supports series-like sources.")

    airports = loaded.metadata["airports"]
    variables = loaded.metadata["variables"]
    if not isinstance(airports, list) or not isinstance(variables, list):
        raise DatasetLoadError("Series metadata is missing airports or variables.")

    airport_indices = _resolve_indices(airports, config.data.airports, "airports")
    input_indices = _resolve_indices(
        variables,
        config.data.input_variables,
        "variables",
    )
    target_indices = _resolve_indices(
        variables, config.data.target_variables, "target variables"
    )

    train_mask = (
        None if loaded.original_masks is None else loaded.original_masks["train"]
    )
    val_mask = None if loaded.original_masks is None else loaded.original_masks["val"]
    test_mask = None if loaded.original_masks is None else loaded.original_masks["test"]

    time_resolution = loaded.metadata["time_resolution"]
    if time_resolution != config.data.time_resolution:
        raise DatasetLoadError(
            "Configured time resolution "
            f"{config.data.time_resolution!r} does not match dataset metadata "
            f"{time_resolution!r}."
        )

    return PreparedSeriesData(
        source=config.data.source,
        train=_build_split(
            split_name="train",
            array=loaded.splits["train"],
            airport_ids=config.data.airports,
            airport_indices=airport_indices,
            input_feature_names=config.data.input_variables,
            input_indices=input_indices,
            target_feature_names=config.data.target_variables,
            target_indices=target_indices,
            original_mask=train_mask,
            time_resolution=time_resolution,
        ),
        val=_build_split(
            split_name="val",
            array=loaded.splits["val"],
            airport_ids=config.data.airports,
            airport_indices=airport_indices,
            input_feature_names=config.data.input_variables,
            input_indices=input_indices,
            target_feature_names=config.data.target_variables,
            target_indices=target_indices,
            original_mask=val_mask,
            time_resolution=time_resolution,
        ),
        test=_build_split(
            split_name="test",
            array=loaded.splits["test"],
            airport_ids=config.data.airports,
            airport_indices=airport_indices,
            input_feature_names=config.data.input_variables,
            input_indices=input_indices,
            target_feature_names=config.data.target_variables,
            target_indices=target_indices,
            original_mask=test_mask,
            time_resolution=time_resolution,
        ),
    )


if "series" not in DATA_BUILDERS.keys():
    DATA_BUILDERS.register("series", build_series_data)
if "series_15min" not in DATA_BUILDERS.keys():
    DATA_BUILDERS.register("series_15min", build_series_data)
