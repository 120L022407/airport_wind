"""Generic training flow for configuration-driven experiments."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import pickle
from typing import Any, Callable, cast

import numpy as np

from windlab.config import ExperimentConfig, load_config
from windlab.data.normalization import (
    NormalizationState,
    apply_normalization,
    fit_normalization,
    save_normalization_state,
)
from windlab.data.series import PreparedSeriesData, PreparedSeriesSplit
from windlab.data.windows import WindowedData, build_windowed_data
from windlab.losses import mse_loss
from windlab.metrics import compute_metrics
from windlab.models.gru import GRUModel
from windlab.registry import DATA_BUILDERS, MODELS
from windlab.utils import create_run_dir, dump_json, dump_yaml, set_seed, timestamped_run_name

from . import models  # noqa: F401
from .data import series as _series_module  # noqa: F401

DataBuilderFn = Callable[[ExperimentConfig], PreparedSeriesData]


class Trainer:
    """One generic training flow for all experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def fit(self, output_root_override: str | None = None) -> Path:
        set_seed(self.config.experiment.seed)
        run_name = timestamped_run_name(self.config.run_name)
        output_root = output_root_override or self.config.runtime.output_root
        run_dir = create_run_dir(output_root, run_name)

        prepared = self._build_data()
        normalization_state = self._build_normalization_state(prepared)
        normalized_prepared = self._apply_normalization(prepared, normalization_state)
        windowed = build_windowed_data(normalized_prepared, self.config)

        model_class = cast(type[GRUModel], MODELS.get(self.config.model.name))
        model = model_class(
            input_size=windowed.train.inputs.shape[2] * windowed.train.inputs.shape[3],
            hidden_size=self.config.model.hidden_size,
            forecast_steps=self.config.data.forecast_steps,
            airport_count=len(self.config.data.airports),
            target_size=len(self.config.data.target_variables),
            seed=self.config.experiment.seed,
            ridge_lambda=self.config.trainer.ridge_lambda,
        )
        model.fit(windowed.train.inputs, windowed.train.targets)

        metrics_payload = self._collect_metrics(model, windowed)
        self._save_artifacts(run_dir, model, normalization_state, metrics_payload)
        return run_dir

    def _build_data(self) -> PreparedSeriesData:
        data_builder = cast(DataBuilderFn, DATA_BUILDERS.get(self.config.data.source))
        built = data_builder(self.config)
        if not isinstance(built, PreparedSeriesData):
            raise TypeError("Data builder must return PreparedSeriesData.")
        return built

    def _apply_normalization(
        self,
        prepared: PreparedSeriesData,
        state: NormalizationState,
    ) -> PreparedSeriesData:
        return PreparedSeriesData(
            source=prepared.source,
            train=self._normalize_split(prepared.train, state),
            val=self._normalize_split(prepared.val, state),
            test=self._normalize_split(prepared.test, state),
        )

    def _normalize_split(
        self,
        split: PreparedSeriesSplit,
        state: NormalizationState,
    ) -> PreparedSeriesSplit:
        if not self.config.normalization.enabled or not self.config.normalization.apply_to_inputs:
            return split
        normalized_values = apply_normalization(split.values, state)
        return replace(split, values=normalized_values)

    def _build_normalization_state(
        self,
        prepared: PreparedSeriesData,
    ) -> NormalizationState:
        if self.config.normalization.enabled:
            return fit_normalization(
                prepared.train.values,
                prepared.train.input_feature_names,
            )
        feature_count = prepared.train.values.shape[-1]
        identity = np.ones((1, 1, feature_count), dtype=np.float64)
        zeros = np.zeros((1, 1, feature_count), dtype=np.float64)
        return NormalizationState(
            mean=zeros,
            std=identity,
            feature_names=list(prepared.train.input_feature_names),
            axes=(0, 1),
        )

    def _collect_metrics(
        self,
        model: GRUModel,
        windowed: WindowedData,
    ) -> dict[str, Any]:
        val_output = model.predict(windowed.val.inputs)
        test_output = model.predict(windowed.test.inputs)
        val_mask = (
            windowed.val.observed_target_mask
            if self.config.evaluation.real_observation_only
            else None
        )
        test_mask = (
            windowed.test.observed_target_mask
            if self.config.evaluation.real_observation_only
            else None
        )
        val_metrics = compute_metrics(
            self.config.evaluation.metrics,
            val_output["prediction"],
            windowed.val.targets,
            val_mask,
        )
        test_metrics = compute_metrics(
            self.config.evaluation.metrics,
            test_output["prediction"],
            windowed.test.targets,
            test_mask,
        )
        val_metrics["mse_loss"] = mse_loss(
            val_output["prediction"],
            windowed.val.targets,
            val_mask,
        )
        return {
            "validation": val_metrics,
            "test": test_metrics,
            "real_observation_only": self.config.evaluation.real_observation_only,
            "metrics": list(self.config.evaluation.metrics),
        }

    def _save_artifacts(
        self,
        run_dir: Path,
        model: GRUModel,
        normalization_state: NormalizationState,
        metrics_payload: dict[str, Any],
    ) -> None:
        dump_yaml(run_dir / "config.yaml", asdict(self.config))
        dump_json(run_dir / "metrics.json", metrics_payload)
        save_normalization_state(run_dir / "normalization.npz", normalization_state)

        checkpoint_payload = {
            "model_name": self.config.model.name,
            "epoch": 0,
            "seed": self.config.experiment.seed,
            "best_validation_metric": metrics_payload["validation"][self.config.evaluation.metrics[0]],
            "model_state": model.state_dict(),
        }
        with (run_dir / "checkpoint.pt").open("wb") as handle:
            pickle.dump(checkpoint_payload, handle)


def train_from_config(
    config_path: str | Path,
    output_root_override: str | None = None,
) -> Path:
    config = load_config(config_path)
    trainer = Trainer(config)
    return trainer.fit(output_root_override=output_root_override)
