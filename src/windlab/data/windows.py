"""Split-local window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from windlab.config import ExperimentConfig

from .series import PreparedSeriesData, PreparedSeriesSplit

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class WindowedSplit:
    name: str
    inputs: FloatArray
    targets: FloatArray
    observed_target_mask: BoolArray
    sample_ids: list[str]
    input_time_index: IntArray
    target_time_index: IntArray
    airport_ids: list[str]
    target_airport_ids: list[str]
    input_feature_names: list[str]
    target_feature_names: list[str]


@dataclass(frozen=True)
class WindowedData:
    train: WindowedSplit
    val: WindowedSplit
    test: WindowedSplit


def _window_count(
    time_length: int,
    input_steps: int,
    forecast_steps: int,
) -> int:
    return time_length - input_steps - forecast_steps + 1


def _build_windowed_split(
    split: PreparedSeriesSplit,
    input_steps: int,
    forecast_steps: int,
) -> WindowedSplit:
    sample_count = _window_count(split.values.shape[0], input_steps, forecast_steps)
    if sample_count <= 0:
        raise ValueError(
            f"Split {split.name} is too short for input_steps={input_steps} and "
            f"forecast_steps={forecast_steps}."
        )

    airport_count = split.values.shape[1]
    feature_count = split.values.shape[2]
    target_airport_count = split.targets.shape[1]
    target_feature_count = split.targets.shape[2]

    inputs = np.zeros(
        (sample_count, input_steps, airport_count, feature_count),
        dtype=np.float64,
    )
    targets = np.zeros(
        (sample_count, forecast_steps, target_airport_count, target_feature_count),
        dtype=np.float64,
    )
    masks = np.zeros(
        (sample_count, forecast_steps, target_airport_count, target_feature_count),
        dtype=bool,
    )
    input_time_index = np.zeros((sample_count, input_steps), dtype=np.int64)
    target_time_index = np.zeros((sample_count, forecast_steps), dtype=np.int64)
    sample_ids: list[str] = []

    for start in range(sample_count):
        input_end = start + input_steps
        target_end = input_end + forecast_steps
        inputs[start] = split.values[start:input_end]
        targets[start] = split.targets[input_end:target_end]
        masks[start] = split.observed_target_mask[input_end:target_end]
        input_time_index[start] = split.time_index[start:input_end]
        target_time_index[start] = split.time_index[input_end:target_end]
        sample_ids.append(f"{split.name}-{start:05d}")

    return WindowedSplit(
        name=split.name,
        inputs=inputs,
        targets=targets,
        observed_target_mask=masks,
        sample_ids=sample_ids,
        input_time_index=input_time_index,
        target_time_index=target_time_index,
        airport_ids=list(split.airport_ids),
        target_airport_ids=list(split.target_airport_ids),
        input_feature_names=list(split.input_feature_names),
        target_feature_names=list(split.target_feature_names),
    )


def build_windowed_data(
    prepared: PreparedSeriesData,
    config: ExperimentConfig,
) -> WindowedData:
    return WindowedData(
        train=_build_windowed_split(
            prepared.train,
            config.data.input_steps,
            config.data.forecast_steps,
        ),
        val=_build_windowed_split(
            prepared.val,
            config.data.input_steps,
            config.data.forecast_steps,
        ),
        test=_build_windowed_split(
            prepared.test,
            config.data.input_steps,
            config.data.forecast_steps,
        ),
    )
