"""A small NumPy GRU baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from windlab.registry import MODELS

FloatArray = NDArray[np.float64]


def _sigmoid(values: FloatArray) -> FloatArray:
    return cast(FloatArray, 1.0 / (1.0 + np.exp(-values)))


@dataclass
class GRUModel:
    input_size: int
    hidden_size: int
    forecast_steps: int
    airport_count: int
    target_size: int
    seed: int
    ridge_lambda: float

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        hidden_scale = 1.0 / np.sqrt(max(self.hidden_size, 1))
        input_scale = 1.0 / np.sqrt(max(self.input_size, 1))
        self.w_z = rng.normal(0.0, input_scale, size=(self.input_size, self.hidden_size))
        self.u_z = rng.normal(0.0, hidden_scale, size=(self.hidden_size, self.hidden_size))
        self.b_z = np.zeros((self.hidden_size,), dtype=np.float64)
        self.w_r = rng.normal(0.0, input_scale, size=(self.input_size, self.hidden_size))
        self.u_r = rng.normal(0.0, hidden_scale, size=(self.hidden_size, self.hidden_size))
        self.b_r = np.zeros((self.hidden_size,), dtype=np.float64)
        self.w_h = rng.normal(0.0, input_scale, size=(self.input_size, self.hidden_size))
        self.u_h = rng.normal(0.0, hidden_scale, size=(self.hidden_size, self.hidden_size))
        self.b_h = np.zeros((self.hidden_size,), dtype=np.float64)
        self.readout: FloatArray | None = None

    @property
    def output_size(self) -> int:
        return self.forecast_steps * self.airport_count * self.target_size

    def _encode(self, inputs: FloatArray) -> FloatArray:
        batch_size, _, airport_count, feature_count = inputs.shape
        flattened = inputs.reshape(batch_size, inputs.shape[1], airport_count * feature_count)
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float64)

        for step in range(flattened.shape[1]):
            current = flattened[:, step, :]
            update_gate = _sigmoid(current @ self.w_z + hidden @ self.u_z + self.b_z)
            reset_gate = _sigmoid(current @ self.w_r + hidden @ self.u_r + self.b_r)
            candidate = np.tanh(
                current @ self.w_h + (reset_gate * hidden) @ self.u_h + self.b_h
            )
            hidden = (1.0 - update_gate) * candidate + update_gate * hidden
        return hidden

    def fit(self, inputs: FloatArray, targets: FloatArray) -> None:
        hidden = self._encode(inputs)
        design = np.concatenate(
            [hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        flattened_targets = targets.reshape(targets.shape[0], -1)
        gram = design.T @ design
        regularizer = np.eye(design.shape[1], dtype=np.float64) * self.ridge_lambda
        regularizer[-1, -1] = 0.0
        self.readout = np.linalg.solve(gram + regularizer, design.T @ flattened_targets)

    def predict(self, inputs: FloatArray) -> dict[str, Any]:
        if self.readout is None:
            raise RuntimeError("GRUModel must be fit before predict.")
        hidden = self._encode(inputs)
        design = np.concatenate(
            [hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        prediction = design @ self.readout
        reshaped = prediction.reshape(
            inputs.shape[0],
            self.forecast_steps,
            self.airport_count,
            self.target_size,
        )
        return {"prediction": reshaped, "aux": {"hidden_state": hidden}}

    def state_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "forecast_steps": self.forecast_steps,
            "airport_count": self.airport_count,
            "target_size": self.target_size,
            "seed": self.seed,
            "ridge_lambda": self.ridge_lambda,
            "w_z": self.w_z,
            "u_z": self.u_z,
            "b_z": self.b_z,
            "w_r": self.w_r,
            "u_r": self.u_r,
            "b_r": self.b_r,
            "w_h": self.w_h,
            "u_h": self.u_h,
            "b_h": self.b_h,
            "readout": self.readout,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "GRUModel":
        model = cls(
            input_size=int(state["input_size"]),
            hidden_size=int(state["hidden_size"]),
            forecast_steps=int(state["forecast_steps"]),
            airport_count=int(state["airport_count"]),
            target_size=int(state["target_size"]),
            seed=int(state["seed"]),
            ridge_lambda=float(state["ridge_lambda"]),
        )
        model.w_z = np.array(state["w_z"], dtype=np.float64)
        model.u_z = np.array(state["u_z"], dtype=np.float64)
        model.b_z = np.array(state["b_z"], dtype=np.float64)
        model.w_r = np.array(state["w_r"], dtype=np.float64)
        model.u_r = np.array(state["u_r"], dtype=np.float64)
        model.b_r = np.array(state["b_r"], dtype=np.float64)
        model.w_h = np.array(state["w_h"], dtype=np.float64)
        model.u_h = np.array(state["u_h"], dtype=np.float64)
        model.b_h = np.array(state["b_h"], dtype=np.float64)
        readout = state["readout"]
        model.readout = None if readout is None else np.array(readout, dtype=np.float64)
        return model


if "gru" not in MODELS.keys():
    MODELS.register("gru", GRUModel)
