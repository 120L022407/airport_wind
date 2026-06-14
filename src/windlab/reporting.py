"""Evaluation reporting utilities."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "airport_wind_lab_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer[Any]]

DEFAULT_LEADS = (1, 6, 12, 24)
DEFAULT_FIRST_N = 300


class ReportingError(ValueError):
    """Raised when reporting inputs are inconsistent."""


@dataclass(frozen=True)
class FixedLeadSeries:
    lead: int
    times: NDArray[np.generic]
    predictions: FloatArray
    targets: FloatArray


def build_fixed_lead_series(
    *,
    predictions: FloatArray,
    targets: FloatArray,
    lead: int,
    target_timestamps: NDArray[np.generic] | None = None,
    target_index: int = 0,
) -> FixedLeadSeries:
    """Extract a continuous fixed-lead series without flattening horizons."""

    _validate_prediction_target_shapes(predictions, targets)
    lead_index = lead - 1
    if lead_index < 0 or lead_index >= predictions.shape[1]:
        raise ReportingError(
            f"lead={lead} is out of range for horizon={predictions.shape[1]}."
        )
    if target_index < 0 or target_index >= predictions.shape[3]:
        raise ReportingError(
            f"target_index={target_index} is out of range for "
            f"target_count={predictions.shape[3]}."
        )

    fixed_predictions = predictions[:, lead_index, :, target_index]
    fixed_targets = targets[:, lead_index, :, target_index]
    times = _select_times(
        target_timestamps=target_timestamps,
        lead_index=lead_index,
        expected_length=fixed_predictions.shape[0],
    )
    if len(times) != fixed_predictions.shape[0] or len(times) != fixed_targets.shape[0]:
        raise ReportingError(
            "Length mismatch for fixed-lead series: "
            f"times={len(times)}, predictions={fixed_predictions.shape[0]}, "
            f"targets={fixed_targets.shape[0]}."
        )

    return FixedLeadSeries(
        lead=lead,
        times=times,
        predictions=fixed_predictions.copy(),
        targets=fixed_targets.copy(),
    )


def save_test_prediction_figures(
    *,
    predictions: FloatArray,
    targets: FloatArray,
    output_dir: str | Path,
    target_timestamps: NDArray[np.generic] | None = None,
    airport_labels: Sequence[str] | None = None,
    target_name: str = "wind_speed",
    leads: Sequence[int] = DEFAULT_LEADS,
    first_n: int = DEFAULT_FIRST_N,
) -> list[Path]:
    """Save fixed-lead full and first-N test prediction figures."""

    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _validate_prediction_target_shapes(predictions, targets)
    airport_count = predictions.shape[2]
    labels = _airport_labels(airport_labels, airport_count)

    saved_paths: list[Path] = []
    for lead in leads:
        series = build_fixed_lead_series(
            predictions=predictions,
            targets=targets,
            lead=lead,
            target_timestamps=target_timestamps,
        )
        full_path = figure_dir / f"test_predictions_lead_{lead}_full.png"
        _plot_fixed_lead_series(
            series=series,
            airport_labels=labels,
            target_name=target_name,
            output_path=full_path,
            view_label="full",
            max_points=None,
        )
        saved_paths.append(full_path)

        first_path = figure_dir / f"test_predictions_lead_{lead}_first_300.png"
        _plot_fixed_lead_series(
            series=series,
            airport_labels=labels,
            target_name=target_name,
            output_path=first_path,
            view_label="first 300",
            max_points=first_n,
        )
        saved_paths.append(first_path)

    return saved_paths


def _validate_prediction_target_shapes(
    predictions: FloatArray,
    targets: FloatArray,
) -> None:
    if predictions.ndim != 4 or targets.ndim != 4:
        raise ReportingError(
            "predictions and targets must have shape "
            "[window, horizon, airport, target]."
        )
    if predictions.shape != targets.shape:
        raise ReportingError(
            f"predictions shape {predictions.shape} does not match "
            f"targets shape {targets.shape}."
        )
    if predictions.shape[0] == 0:
        raise ReportingError("Cannot plot empty prediction arrays.")


def _select_times(
    *,
    target_timestamps: NDArray[np.generic] | None,
    lead_index: int,
    expected_length: int,
) -> NDArray[np.generic]:
    if target_timestamps is None:
        return np.arange(expected_length)
    timestamps = np.asarray(target_timestamps)
    if timestamps.ndim == 1:
        selected = timestamps
    elif timestamps.ndim == 2:
        if lead_index >= timestamps.shape[1]:
            raise ReportingError(
                f"lead_index={lead_index} is out of range for timestamp "
                f"horizon={timestamps.shape[1]}."
            )
        selected = timestamps[:, lead_index]
    else:
        raise ReportingError(
            "target_timestamps must have shape [window] or [window, horizon]."
        )
    if len(selected) != expected_length:
        raise ReportingError(
            "Length mismatch for fixed-lead series: "
            f"times={len(selected)}, predictions={expected_length}, "
            f"targets={expected_length}."
        )
    return selected.copy()


def _airport_labels(
    airport_labels: Sequence[str] | None,
    airport_count: int,
) -> list[str]:
    if airport_labels is None:
        return [f"airport_{index}" for index in range(airport_count)]
    if len(airport_labels) != airport_count:
        raise ReportingError(
            f"airport_labels length {len(airport_labels)} does not match "
            f"airport_count={airport_count}."
        )
    return list(airport_labels)


def _plot_fixed_lead_series(
    *,
    series: FixedLeadSeries,
    airport_labels: Sequence[str],
    target_name: str,
    output_path: Path,
    view_label: str,
    max_points: int | None,
) -> None:
    point_count = (
        len(series.times) if max_points is None else min(max_points, len(series.times))
    )
    times = series.times[:point_count]
    predictions = series.predictions[:point_count]
    targets = series.targets[:point_count]
    airport_count = predictions.shape[1]

    fig_height = max(3.0, 2.4 * airport_count)
    fig, axes = plt.subplots(
        airport_count,
        1,
        figsize=(12, fig_height),
        sharex=True,
        squeeze=False,
    )
    for airport_index, axis in enumerate(axes[:, 0]):
        axis.plot(times, targets[:, airport_index], label="Observed", linewidth=1.2)
        axis.plot(
            times,
            predictions[:, airport_index],
            label="Prediction",
            linewidth=1.2,
        )
        axis.set_ylabel("Wind speed (m/s)")
        axis.set_title(str(airport_labels[airport_index]))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    lead_label = f"{series.lead}h"
    fig.suptitle(f"Test predictions lead {lead_label} - {view_label} - {target_name}")
    axes[-1, 0].set_xlabel(_x_label(times))
    if _is_datetime_like(times):
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _is_datetime_like(values: NDArray[np.generic]) -> bool:
    return np.issubdtype(values.dtype, np.datetime64)


def _x_label(values: NDArray[np.generic]) -> str:
    return "Time" if _is_datetime_like(values) else "Time index"
