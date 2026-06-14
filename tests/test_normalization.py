from __future__ import annotations

from pathlib import Path

import numpy as np

from windlab.data.normalization import (
    apply_normalization,
    fit_normalization,
    load_normalization_state,
    save_normalization_state,
)


def test_fit_normalization_uses_train_statistics_only(tmp_path: Path) -> None:
    train = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    val = np.array(
        [
            [[100.0, 200.0], [300.0, 400.0]],
        ]
    )
    state = fit_normalization(train, ["a", "b"])
    normalized_train = apply_normalization(train, state)
    normalized_val = apply_normalization(val, state)
    assert np.allclose(normalized_train.mean(axis=(0, 1)), np.zeros(2))
    assert not np.allclose(normalized_val.mean(axis=(0, 1)), np.zeros(2))

    state_path = tmp_path / "normalization.npz"
    save_normalization_state(state_path, state)
    restored = load_normalization_state(state_path)
    restored_val = apply_normalization(val, restored)
    assert np.allclose(normalized_val, restored_val)
