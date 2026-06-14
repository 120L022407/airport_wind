"""Train-only normalization for split arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NormalizationState:
    mean: FloatArray
    std: FloatArray
    feature_names: list[str]
    axes: tuple[int, ...]


def fit_normalization(
    train_values: FloatArray,
    feature_names: list[str],
    axes: tuple[int, ...] = (0, 1),
) -> NormalizationState:
    mean = train_values.mean(axis=axes, keepdims=True)
    std = train_values.std(axis=axes, keepdims=True)
    safe_std = np.where(std == 0.0, 1.0, std)
    return NormalizationState(
        mean=mean.astype(np.float64),
        std=safe_std.astype(np.float64),
        feature_names=list(feature_names),
        axes=axes,
    )


def apply_normalization(values: FloatArray, state: NormalizationState) -> FloatArray:
    return cast(FloatArray, ((values - state.mean) / state.std).astype(np.float64))


def save_normalization_state(path: str | Path, state: NormalizationState) -> None:
    np.savez(
        Path(path),
        mean=state.mean,
        std=state.std,
        feature_names=np.array(state.feature_names, dtype=object),
        axes=np.array(state.axes, dtype=np.int64),
    )


def load_normalization_state(path: str | Path) -> NormalizationState:
    with np.load(Path(path), allow_pickle=True) as payload:
        mean = cast(FloatArray, np.array(payload["mean"], dtype=np.float64))
        std = cast(FloatArray, np.array(payload["std"], dtype=np.float64))
        feature_names = [str(item) for item in payload["feature_names"].tolist()]
        axes = tuple(int(item) for item in payload["axes"].tolist())
    return NormalizationState(
        mean=mean,
        std=std,
        feature_names=feature_names,
        axes=axes,
    )
