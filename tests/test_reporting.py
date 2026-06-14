from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
