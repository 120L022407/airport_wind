"""Generic evaluation flow for saved run directories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from windlab.config import ExperimentConfig, load_config
from windlab.data.normalization import apply_normalization, load_normalization_state
from windlab.data.series import PreparedSeriesData, PreparedSeriesSplit
from windlab.data.torch_dataset import WindowedTorchDataset
from windlab.data.windows import WindowedData, WindowedSplit, build_windowed_data
from windlab.metrics import compute_metrics
from windlab.reporting import save_test_prediction_figures
from windlab.registry import DATA_BUILDERS, MODELS
from windlab.utils import dump_json

from . import models  # noqa: F401
from .data import series as _series_module  # noqa: F401

DataBuilderFn = Callable[[ExperimentConfig], PreparedSeriesData]


class Evaluator:
    """Evaluate one saved run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def evaluate(self) -> dict[str, Any]:
        config = load_config(self.run_dir / "config.yaml")
        device = self._resolve_device(config.trainer.device)
        prepared = self._build_data(config)
        normalization_state = load_normalization_state(self.run_dir / "normalization.npz")
        normalized = self._apply_normalization(
            prepared,
            normalization_state,
            normalization_enabled=config.normalization.enabled,
        )
        windowed = build_windowed_data(normalized, config)

        checkpoint = self._load_checkpoint(device)
        model = self._build_model(checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics_payload = self._collect_metrics(config, model, windowed, device)
        dump_json(self.run_dir / "metrics.json", metrics_payload)
        return metrics_payload

    def _build_model(self, checkpoint: dict[str, Any]) -> nn.Module:
        model_class = cast(type[nn.Module], MODELS.get(str(checkpoint["model_name"])))
        model = model_class(**checkpoint["model_init"])
        if not isinstance(model, nn.Module):
            raise TypeError("Registered model must be a torch.nn.Module.")
        return model

    def _load_checkpoint(self, device: torch.device) -> dict[str, Any]:
        best_path = self.run_dir / "best_checkpoint.pt"
        fallback_path = self.run_dir / "checkpoint.pt"
        checkpoint_path = best_path if best_path.is_file() else fallback_path
        return cast(dict[str, Any], torch.load(checkpoint_path, map_location=device))

    def _build_data(self, config: ExperimentConfig) -> PreparedSeriesData:
        data_builder = cast(DataBuilderFn, DATA_BUILDERS.get(config.data.source))
        built = data_builder(config)
        if not isinstance(built, PreparedSeriesData):
            raise TypeError("Data builder must return PreparedSeriesData.")
        return built

    def _apply_normalization(
        self,
        prepared: PreparedSeriesData,
        state: Any,
        normalization_enabled: bool,
    ) -> PreparedSeriesData:
        if not normalization_enabled:
            return prepared
        return PreparedSeriesData(
            source=prepared.source,
            train=self._normalize_split(prepared.train, state),
            val=self._normalize_split(prepared.val, state),
            test=self._normalize_split(prepared.test, state),
        )

    def _normalize_split(
        self,
        split: PreparedSeriesSplit,
        state: Any,
    ) -> PreparedSeriesSplit:
        return replace(split, values=apply_normalization(split.values, state))

    def _collect_metrics(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        windowed: WindowedData,
        device: torch.device,
    ) -> dict[str, Any]:
        val_prediction = self._predict_numpy(config, model, windowed.val, device)
        test_prediction = self._predict_numpy(config, model, windowed.test, device)
        val_mask = (
            windowed.val.observed_target_mask
            if config.evaluation.real_observation_only
            else None
        )
        test_mask = (
            windowed.test.observed_target_mask
            if config.evaluation.real_observation_only
            else None
        )
        return {
            "validation": compute_metrics(
                config.evaluation.metrics,
                val_prediction,
                windowed.val.targets,
                val_mask,
            ),
            "test": compute_metrics(
                config.evaluation.metrics,
                test_prediction,
                windowed.test.targets,
                test_mask,
            ),
            "real_observation_only": config.evaluation.real_observation_only,
            "metrics": list(config.evaluation.metrics),
            "figure_paths": [
                str(path)
                for path in save_test_prediction_figures(
                    predictions=test_prediction.astype(np.float64, copy=False),
                    targets=windowed.test.targets,
                    target_timestamps=windowed.test.target_time_index,
                    output_dir=self.run_dir / "figures",
                    airport_labels=windowed.test.airport_ids,
                    target_name=windowed.test.target_feature_names[0],
                )
            ],
        }

    def _predict_numpy(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        split: WindowedSplit,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        loader = DataLoader(
            WindowedTorchDataset(split),
            batch_size=config.trainer.batch_size,
            shuffle=False,
        )
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, _, _ in loader:
                output = model(inputs.to(device))
                predictions.append(output["prediction"].detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)

    def _resolve_device(self, configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(configured_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is not available.")
        return device


def evaluate_run_dir(run_dir: str | Path) -> dict[str, Any]:
    evaluator = Evaluator(run_dir)
    return evaluator.evaluate()
