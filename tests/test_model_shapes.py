from __future__ import annotations

import numpy as np

from windlab.models.gru import GRUModel


def test_gru_model_prediction_shape() -> None:
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(5, 24, 4, 13))
    targets = rng.normal(size=(5, 24, 4, 1))
    model = GRUModel(
        input_size=52,
        hidden_size=16,
        forecast_steps=24,
        airport_count=4,
        target_size=1,
        seed=7,
        ridge_lambda=0.001,
    )
    model.fit(inputs, targets)
    output = model.predict(inputs)
    assert output["prediction"].shape == (5, 24, 4, 1)
    assert output["aux"]["hidden_state"].shape == (5, 16)
