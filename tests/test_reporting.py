from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.axes import Axes
from numpy.typing import NDArray

from windlab.reporting import (
    ReportingError,
    build_fixed_lead_series,
    save_test_prediction_figures,
)


def _arrays(
    window_count: int = 7,
    horizon: int = 24,
    airport_count: int = 4,
    target_count: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    size = window_count * horizon * airport_count * target_count
    predictions = np.arange(size, dtype=np.float64).reshape(
        window_count,
        horizon,
        airport_count,
        target_count,
    )
    targets = predictions + 1000.0
    timestamps = np.arange(window_count * horizon, dtype=np.int64).reshape(
        window_count,
        horizon,
    )
    return predictions, targets, timestamps


def test_save_test_prediction_figures_generates_expected_paths(tmp_path: Path) -> None:
    predictions, targets, timestamps = _arrays(window_count=5)
    figure_dir = tmp_path / "outputs" / "run" / "figures"

    paths = save_test_prediction_figures(
        predictions=predictions,
        targets=targets,
        target_timestamps=timestamps,
        output_dir=figure_dir,
        airport_labels=["ZGSZ", "ZGGG", "VHHH", "VMMC"],
        target_name="sknt",
    )

    expected_names = [
        "test_predictions_lead_1_full.png",
        "test_predictions_lead_1_first_300.png",
        "test_predictions_lead_6_full.png",
        "test_predictions_lead_6_first_300.png",
        "test_predictions_lead_12_full.png",
        "test_predictions_lead_12_first_300.png",
        "test_predictions_lead_24_full.png",
        "test_predictions_lead_24_first_300.png",
    ]
    assert [path.name for path in paths] == expected_names
    assert all(path.parent == figure_dir for path in paths)
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_fixed_lead_series_uses_requested_horizon_slice() -> None:
    predictions, targets, timestamps = _arrays(window_count=4)

    series = build_fixed_lead_series(
        predictions=predictions,
        targets=targets,
        target_timestamps=timestamps,
        lead=6,
    )

    assert np.array_equal(series.predictions, predictions[:, 5, :, 0])
    assert np.array_equal(series.targets, targets[:, 5, :, 0])
    assert np.array_equal(series.times, timestamps[:, 5])


def test_fixed_lead_series_uses_requested_mask_slice() -> None:
    predictions, targets, timestamps = _arrays(window_count=4)
    observed_mask = np.ones_like(predictions, dtype=bool)
    observed_mask[:, 5, :, 0] = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    series = build_fixed_lead_series(
        predictions=predictions,
        targets=targets,
        target_timestamps=timestamps,
        observed_target_mask=observed_mask,
        lead=6,
    )

    assert series.observed_mask is not None
    assert np.array_equal(series.observed_mask, observed_mask[:, 5, :, 0])


def test_save_test_prediction_figures_handles_less_than_300_points(
    tmp_path: Path,
) -> None:
    predictions, targets, timestamps = _arrays(window_count=3)

    paths = save_test_prediction_figures(
        predictions=predictions,
        targets=targets,
        target_timestamps=timestamps,
        output_dir=tmp_path / "figures",
    )

    assert len(paths) == 8
    assert all(path.is_file() for path in paths)


def test_save_test_prediction_figures_filters_to_observed_points(
    tmp_path: Path,
) -> None:
    predictions, targets, timestamps = _arrays(
        window_count=5,
        airport_count=1,
    )
    observed_mask = np.zeros_like(predictions, dtype=bool)
    observed_mask[:, 0, 0, 0] = np.array([True, False, True, True, False], dtype=bool)

    plot_calls: list[tuple[NDArray[np.int64], NDArray[np.float64], str | None]] = []
    original_plot = Axes.plot

    def spy_plot(self: Axes, *args: object, **kwargs: object) -> object:
        label: str | None
        raw_label = kwargs.get("label")
        label = raw_label if isinstance(raw_label, str) else None
        plot_calls.append(
            (
                np.asarray(args[0], dtype=np.int64).copy(),
                np.asarray(args[1], dtype=np.float64).copy(),
                label,
            )
        )
        return cast(Any, original_plot)(self, *args, **kwargs)

    with patch.object(Axes, "plot", new=spy_plot):
        save_test_prediction_figures(
            predictions=predictions,
            targets=targets,
            target_timestamps=timestamps,
            observed_target_mask=observed_mask,
            output_dir=tmp_path / "figures",
            airport_labels=["ZGSZ"],
            leads=[1],
        )

    observed_calls = [call for call in plot_calls if call[2] == "Observed"]
    prediction_calls = [call for call in plot_calls if call[2] == "Prediction"]
    expected_times = np.array([0, 48, 72], dtype=np.int64)
    expected_targets = targets[[0, 2, 3], 0, 0, 0]
    expected_predictions = predictions[[0, 2, 3], 0, 0, 0]

    assert len(observed_calls) == 2
    assert len(prediction_calls) == 2
    assert np.array_equal(observed_calls[0][0], expected_times)
    assert np.array_equal(observed_calls[0][1], expected_targets)
    assert np.array_equal(prediction_calls[0][0], expected_times)
    assert np.array_equal(prediction_calls[0][1], expected_predictions)


def test_fixed_lead_series_rejects_timestamp_length_mismatch() -> None:
    predictions, targets, timestamps = _arrays(window_count=4)

    with pytest.raises(ReportingError, match="Length mismatch"):
        build_fixed_lead_series(
            predictions=predictions,
            targets=targets,
            target_timestamps=timestamps[:-1],
            lead=1,
        )


def test_fixed_lead_series_rejects_prediction_target_shape_mismatch() -> None:
    predictions, targets, timestamps = _arrays(window_count=4)

    with pytest.raises(ReportingError, match="does not match"):
        build_fixed_lead_series(
            predictions=predictions,
            targets=targets[:-1],
            target_timestamps=timestamps,
            lead=1,
        )


def test_save_test_prediction_figures_does_not_mutate_arrays(tmp_path: Path) -> None:
    predictions, targets, timestamps = _arrays(window_count=4)
    original_predictions = predictions.copy()
    original_targets = targets.copy()

    save_test_prediction_figures(
        predictions=predictions,
        targets=targets,
        target_timestamps=timestamps,
        output_dir=tmp_path / "figures",
    )

    assert np.array_equal(predictions, original_predictions)
    assert np.array_equal(targets, original_targets)
