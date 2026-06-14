"""Generic evaluation flow for saved run directories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
from typing import Any, Callable, cast

from windlab.config import load_config
from windlab.data.normalization import apply_normalization, load_normalization_state
from windlab.data.series import PreparedSeriesData, PreparedSeriesSplit
from windlab.data.windows import build_windowed_data
from windlab.metrics import compute_metrics
from windlab.models.gru import GRUModel
from windlab.registry import DATA_BUILDERS, MODELS
from windlab.utils import dump_json

from . import models  # noqa: F401
from .data import series as _series_module  # noqa: F401

DataBuilderFn = Callable[[Any], PreparedSeriesData]


class Evaluator:
    """Evaluate one saved run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def evaluate(self) -> dict[str, Any]:
        config = load_config(self.run_dir / "config.yaml")
        prepared = self._build_data(config)
        normalization_state = load_normalization_state(self.run_dir / "normalization.npz")
        normalized = self._apply_normalization(
            prepared,
            normalization_state,
            normalization_enabled=config.normalization.enabled,
        )
        windowed = build_windowed_data(normalized, config)

        with (self.run_dir / "checkpoint.pt").open("rb") as handle:
            checkpoint = pickle.load(handle)

        model_class = cast(type[GRUModel], MODELS.get(str(checkpoint["model_name"])))
        model = model_class.from_state_dict(checkpoint["model_state"])
        val_output = model.predict(windowed.val.inputs)
        test_output = model.predict(windowed.test.inputs)
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
        metrics_payload = {
            "validation": compute_metrics(
                config.evaluation.metrics,
                val_output["prediction"],
                windowed.val.targets,
                val_mask,
            ),
            "test": compute_metrics(
                config.evaluation.metrics,
                test_output["prediction"],
                windowed.test.targets,
                test_mask,
            ),
            "real_observation_only": config.evaluation.real_observation_only,
            "metrics": list(config.evaluation.metrics),
        }
        dump_json(self.run_dir / "metrics.json", metrics_payload)
        return metrics_payload

    def _build_data(self, config: Any) -> PreparedSeriesData:
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

    def _normalize_split(self, split: PreparedSeriesSplit, state: Any) -> PreparedSeriesSplit:
        return replace(split, values=apply_normalization(split.values, state))


def evaluate_run_dir(run_dir: str | Path) -> dict[str, Any]:
    evaluator = Evaluator(run_dir)
    return evaluator.evaluate()
