from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from windlab.data.loaders import DatasetLoadError, load_dataset_root

from .helpers import create_ec_fixture, create_series_fixture


def test_load_series_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "series"
    create_series_fixture(dataset_root, source="series")
    loaded = load_dataset_root(dataset_root, "series")
    assert loaded.splits["train"].shape == (72, 4, 13)
    assert loaded.metadata["time_resolution"] == "1h"
    assert loaded.original_masks is None


@pytest.mark.parametrize("source", ["series_15min", "series_15min_cubic"])
def test_load_series_15min_like_dataset_with_masks(
    tmp_path: Path,
    source: str,
) -> None:
    dataset_root = tmp_path / source
    create_series_fixture(dataset_root, source=source)
    loaded = load_dataset_root(dataset_root, source)
    assert loaded.original_masks is not None
    assert loaded.original_masks["val"].dtype == bool
    assert loaded.original_masks["val"].shape == (56,)


def test_load_ec_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "EC"
    create_ec_fixture(dataset_root)
    loaded = load_dataset_root(dataset_root, "EC")
    assert loaded.splits["test"].shape == (36, 5, 9, 9)
    assert loaded.metadata["source"] == "EC"


@pytest.mark.parametrize("source", ["series_15min", "series_15min_cubic"])
def test_series_15min_like_rejects_bad_original_mask_shape(
    tmp_path: Path,
    source: str,
) -> None:
    dataset_root = tmp_path / source
    create_series_fixture(dataset_root, source=source)
    np.save(dataset_root / "train_original_mask.npy", np.ones((10, 1), dtype=bool))
    with pytest.raises(DatasetLoadError, match="shape \\[time\\]"):
        load_dataset_root(dataset_root, source)
