"""Generic PyTorch training flow for configuration-driven experiments."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from windlab.config import ExperimentConfig, load_config
from windlab.data.normalization import (
    NormalizationState,
    apply_normalization,
    fit_normalization,
    save_normalization_state,
)
from windlab.data.series import PreparedSeriesData, PreparedSeriesSplit
from windlab.data.torch_dataset import WindowedTorchDataset
from windlab.data.windows import WindowedData, WindowedSplit, build_windowed_data
from windlab.metrics import compute_metrics
from windlab.registry import DATA_BUILDERS, LOSSES, MODELS
from windlab.utils import (
    create_run_dir,
    dump_json,
    dump_yaml,
    set_seed,
    timestamped_run_name,
)

from . import losses as _losses_module  # noqa: F401
from . import models  # noqa: F401
from .data import series as _series_module  # noqa: F401

DataBuilderFn = Callable[[ExperimentConfig], PreparedSeriesData]
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]
FloatArray = NDArray[np.float64]


class Trainer:
    """One generic training flow for all experiments."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.trainer.device)
        self.loss_fn = cast(LossFn, LOSSES.get("mse"))

    def fit(self, output_root_override: str | None = None) -> Path:
        set_seed(self.config.experiment.seed)
        run_name = timestamped_run_name(self.config.run_name)
        output_root = output_root_override or self.config.runtime.output_root
        run_dir = create_run_dir(output_root, run_name)

        prepared = self._build_data()
        normalization_state = self._build_normalization_state(prepared)
        normalized_prepared = self._apply_normalization(prepared, normalization_state)
        windowed = build_windowed_data(normalized_prepared, self.config)

        model = self._build_model(windowed).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.trainer.learning_rate,
            weight_decay=self.config.trainer.weight_decay,
        )

        config_payload = self._config_artifact_payload()
        dump_yaml(run_dir / "config.yaml", config_payload)
        dump_yaml(run_dir / "resolved_config.yaml", config_payload)
        save_normalization_state(run_dir / "normalization.npz", normalization_state)

        training_log = self._train_loop(run_dir, model, optimizer, windowed)
        dump_json(run_dir / "training_log.json", {"epochs": training_log})

        best_checkpoint = self._load_checkpoint(run_dir / "best_checkpoint.pt")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        metrics_payload = self._collect_metrics(model, windowed)
        dump_json(run_dir / "metrics.json", metrics_payload)
        shutil.copyfile(run_dir / "best_checkpoint.pt", run_dir / "checkpoint.pt")
        return run_dir

    def _train_loop(
        self,
        run_dir: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        windowed: WindowedData,
    ) -> list[dict[str, float | int]]:
        train_loader = DataLoader(
            WindowedTorchDataset(windowed.train),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
        )
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        training_log: list[dict[str, float | int]] = []

        for epoch in range(1, self.config.trainer.epochs + 1):
            train_loss = self._run_training_epoch(model, optimizer, train_loader)
            val_loss = self._evaluate_loss(model, windowed.val)
            log_row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            training_log.append(log_row)

            improved = val_loss < best_val_loss - self.config.trainer.min_delta
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(
                    run_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )
            else:
                epochs_without_improvement += 1

            self._save_checkpoint(
                run_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )

            if epochs_without_improvement >= self.config.trainer.patience:
                break

        return training_log

    def _run_training_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        model.train()
        losses: list[float] = []
        for inputs, targets, masks in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = self.loss_fn(output["prediction"], targets, masks)
            cast(Any, loss).backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    def _evaluate_loss(self, model: nn.Module, split: WindowedSplit) -> float:
        model.eval()
        loader = DataLoader(
            WindowedTorchDataset(split),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
        )
        losses: list[float] = []
        with torch.no_grad():
            for inputs, targets, masks in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                output = model(inputs)
                loss = self.loss_fn(output["prediction"], targets, masks)
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    def _build_model(self, windowed: WindowedData) -> nn.Module:
        model_class = cast(type[nn.Module], MODELS.get(self.config.model.name))
        model = model_class(
            input_size=windowed.train.inputs.shape[2] * windowed.train.inputs.shape[3],
            input_steps=windowed.train.inputs.shape[1],
            forecast_steps=self.config.data.forecast_steps,
            airport_count=len(self.config.data.airports),
            target_size=len(self.config.data.target_variables),
            **self.config.model.parameters,
        )
        if not isinstance(model, nn.Module):
            raise TypeError("Registered model must be a torch.nn.Module.")
        return model

    def _build_data(self) -> PreparedSeriesData:
        data_builder = cast(DataBuilderFn, DATA_BUILDERS.get(self.config.data.source))
        built = data_builder(self.config)
        if not isinstance(built, PreparedSeriesData):
            raise TypeError("Data builder must return PreparedSeriesData.")
        return built

    def _config_artifact_payload(self) -> dict[str, Any]:
        model_payload = {
            "name": self.config.model.name,
            **self.config.model.parameters,
        }
        return {
            "experiment": asdict(self.config.experiment),
            "runtime": asdict(self.config.runtime),
            "data": asdict(self.config.data),
            "normalization": asdict(self.config.normalization),
            "model": model_payload,
            "trainer": asdict(self.config.trainer),
            "evaluation": asdict(self.config.evaluation),
        }

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
        if (
            not self.config.normalization.enabled
            or not self.config.normalization.apply_to_inputs
        ):
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
        model: nn.Module,
        windowed: WindowedData,
    ) -> dict[str, Any]:
        val_prediction = self._predict_numpy(model, windowed.val)
        test_prediction = self._predict_numpy(model, windowed.test)
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
            val_prediction,
            windowed.val.targets,
            val_mask,
        )
        test_metrics = compute_metrics(
            self.config.evaluation.metrics,
            test_prediction,
            windowed.test.targets,
            test_mask,
        )
        val_metrics["mse_loss"] = self._evaluate_loss(model, windowed.val)
        return {
            "validation": val_metrics,
            "test": test_metrics,
            "real_observation_only": self.config.evaluation.real_observation_only,
            "metrics": list(self.config.evaluation.metrics),
        }

    def _predict_numpy(self, model: nn.Module, split: WindowedSplit) -> FloatArray:
        model.eval()
        loader = DataLoader(
            WindowedTorchDataset(split),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
        )
        predictions: list[FloatArray] = []
        with torch.no_grad():
            for inputs, _, _ in loader:
                output = model(inputs.to(self.device))
                prediction = output["prediction"].detach().cpu().numpy()
                predictions.append(prediction.astype(np.float64, copy=False))
        return cast(FloatArray, np.concatenate(predictions, axis=0))

    def _save_checkpoint(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
    ) -> None:
        init_kwargs = cast(Any, model).init_kwargs
        torch.save(
            {
                "model_name": self.config.model.name,
                "model_init": init_kwargs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_validation_loss": best_val_loss,
                "seed": self.config.experiment.seed,
            },
            path,
        )

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], torch.load(path, map_location=self.device))

    def _resolve_device(self, configured_device: str) -> torch.device:
        if configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(configured_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is not available.")
        return device


def train_from_config(
    config_path: str | Path,
    output_root_override: str | None = None,
) -> Path:
    config = load_config(config_path)
    trainer = Trainer(config)
    return trainer.fit(output_root_override=output_root_override)
