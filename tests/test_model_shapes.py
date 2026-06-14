from __future__ import annotations

import pytest
import torch
from torch import nn

from windlab.models.dlinear import DLinearModel
from windlab.models.gru import GRUModel
from windlab.models.itransformer import ITransformerModel
from windlab.models.patchtst import PatchTSTModel
from windlab.registry import MODELS


def _common_kwargs() -> dict[str, int]:
    return {
        "input_size": 52,
        "input_steps": 24,
        "forecast_steps": 24,
        "airport_count": 4,
        "target_size": 1,
    }


MODEL_CASES = [
    (
        "gru",
        GRUModel,
        {"hidden_size": 16, "num_layers": 1, "dropout": 0.0},
    ),
    (
        "patchtst",
        PatchTSTModel,
        {
            "d_model": 16,
            "num_layers": 1,
            "n_heads": 4,
            "ff_dim": 32,
            "dropout": 0.0,
            "patch_len": 6,
            "stride": 3,
        },
    ),
    (
        "itransformer",
        ITransformerModel,
        {
            "d_model": 16,
            "num_layers": 1,
            "n_heads": 4,
            "ff_dim": 32,
            "dropout": 0.0,
        },
    ),
    (
        "dlinear",
        DLinearModel,
        {"moving_avg": 5, "individual": False},
    ),
]


@pytest.mark.parametrize(("registry_key", "model_class", "model_kwargs"), MODEL_CASES)
def test_model_forward_backward_shape(
    registry_key: str,
    model_class: type[nn.Module],
    model_kwargs: dict[str, int | float | bool],
) -> None:
    model = model_class(**_common_kwargs(), **model_kwargs)
    inputs = torch.randn(2, 24, 4, 13)
    targets = torch.randn(2, 24, 4, 1)

    output = model(inputs)
    assert output["prediction"].shape == (2, 24, 4, 1)
    assert MODELS.get(registry_key) is model_class

    loss = (output["prediction"] - targets).pow(2).mean()
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(("registry_key", "model_class", "model_kwargs"), MODEL_CASES)
def test_model_supports_batch_size_one(
    registry_key: str,
    model_class: type[nn.Module],
    model_kwargs: dict[str, int | float | bool],
) -> None:
    _ = registry_key
    model = model_class(**_common_kwargs(), **model_kwargs)
    output = model(torch.randn(1, 24, 4, 13))
    assert output["prediction"].shape == (1, 24, 4, 1)


def test_patchtst_rejects_patch_longer_than_input() -> None:
    with pytest.raises(ValueError, match="patch_len"):
        PatchTSTModel(
            **_common_kwargs(),
            d_model=16,
            num_layers=1,
            n_heads=4,
            ff_dim=32,
            dropout=0.0,
            patch_len=25,
            stride=1,
        )


def test_dlinear_rejects_even_moving_average() -> None:
    with pytest.raises(ValueError, match="moving_avg"):
        DLinearModel(**_common_kwargs(), moving_avg=4, individual=False)
