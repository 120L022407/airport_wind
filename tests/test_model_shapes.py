from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from windlab.config import load_config
from windlab.models.dlinear import DLinearModel
from windlab.models.gru import GRUModel
from windlab.models.hcan import HCANModel
from windlab.models.itransformer import ITransformerModel
from windlab.models.patchtst import PatchTSTModel
from windlab.models.tfps import TFPSModel
from windlab.models.timebridge import TimeBridgeModel
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
        "hcan",
        HCANModel,
        {
            "backbone_hidden_size": 16,
            "backbone_num_layers": 1,
            "backbone_dropout": 0.0,
            "hidden_dim": 8,
            "num_coarse": 4,
            "num_fine": 8,
            "lambda_cls": 1.0,
            "lambda_reg": 1.0,
            "lambda_acl": 1.0,
            "lambda_direct": 1.0,
        },
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
    (
        "timebridge",
        TimeBridgeModel,
        {
            "period": 6,
            "num_p": 2,
            "ia_layers": 1,
            "pd_layers": 1,
            "ca_layers": 1,
            "stable_len": 6,
            "input_feature_count": 13,
            "shared_time_feature_count": 5,
            "d_model": 16,
            "n_heads": 4,
            "d_ff": 32,
            "dropout": 0.0,
            "attn_dropout": 0.1,
            "activation": "gelu",
        },
    ),
    (
        "tfps",
        TFPSModel,
        {
            "d_model": 16,
            "num_layers": 1,
            "n_heads": 4,
            "ff_dim": 32,
            "dropout": 0.0,
            "patch_len": 6,
            "stride": 3,
            "time_num_experts": 4,
            "time_top_k": 2,
            "frequency_num_experts": 4,
            "frequency_top_k": 2,
            "expert_hidden_size": 32,
            "subspace_eta": 5.0,
            "use_time_domain": True,
            "use_frequency_domain": True,
            "use_pattern_identifier": True,
            "use_pattern_experts": True,
            "noisy_gating": False,
        },
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


@pytest.mark.parametrize(
    "config_path",
    [
        "config/patchtst/baseline_15min.yaml",
        "config/itransformer/baseline_15min.yaml",
        "config/dlinear/baseline_15min.yaml",
        "config/tfps/baseline_15min.yaml",
        "config/timebridge/baseline_15min.yaml",
    ],
)
def test_15min_baseline_configs_build_models_with_expected_shape(
    config_path: str,
) -> None:
    config = load_config(config_path)
    model_class = cast(type[nn.Module], MODELS.get(config.model.name))
    model = model_class(
        input_size=52,
        input_steps=96,
        forecast_steps=96,
        airport_count=1,
        target_size=1,
        **config.model.parameters,
    )

    output = model(torch.randn(2, 96, 4, 13))
    assert output["prediction"].shape == (2, 96, 1, 1)


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


def test_hcan_rejects_inconsistent_hierarchy_sizes() -> None:
    with pytest.raises(ValueError, match="num_fine"):
        HCANModel(
            **_common_kwargs(),
            backbone_hidden_size=16,
            backbone_num_layers=1,
            backbone_dropout=0.0,
            hidden_dim=8,
            num_coarse=4,
            num_fine=6,
            lambda_cls=1.0,
            lambda_reg=1.0,
            lambda_acl=1.0,
            lambda_direct=1.0,
        )


def test_dlinear_rejects_even_moving_average() -> None:
    with pytest.raises(ValueError, match="moving_avg"):
        DLinearModel(**_common_kwargs(), moving_avg=4, individual=False)


def test_timebridge_rejects_non_divisible_period() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TimeBridgeModel(
            **_common_kwargs(),
            period=5,
            num_p=2,
            ia_layers=1,
            pd_layers=1,
            ca_layers=1,
            stable_len=6,
            input_feature_count=13,
            shared_time_feature_count=5,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            attn_dropout=0.1,
            activation="gelu",
        )


def test_timebridge_rejects_invalid_shared_time_feature_count() -> None:
    with pytest.raises(ValueError, match="shared_time_feature_count"):
        TimeBridgeModel(
            **_common_kwargs(),
            period=6,
            num_p=2,
            ia_layers=1,
            pd_layers=1,
            ca_layers=1,
            stable_len=6,
            input_feature_count=13,
            shared_time_feature_count=13,
            d_model=16,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            attn_dropout=0.1,
            activation="gelu",
        )


def test_timebridge_aux_uses_signal_channels_only() -> None:
    model = TimeBridgeModel(
        **_common_kwargs(),
        period=6,
        num_p=2,
        ia_layers=1,
        pd_layers=1,
        ca_layers=1,
        stable_len=6,
        input_feature_count=13,
        shared_time_feature_count=5,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        attn_dropout=0.1,
        activation="gelu",
    )

    output = model(torch.randn(2, 24, 4, 13))

    assert output["prediction"].shape == (2, 24, 4, 1)
    aux = cast(dict[str, Any], output["aux"])
    assert aux["encoded"].shape == (2, 32, model.output_patch_count, 16)


def test_tfps_rejects_disabled_domains() -> None:
    with pytest.raises(ValueError, match="domain"):
        TFPSModel(
            **_common_kwargs(),
            d_model=16,
            num_layers=1,
            n_heads=4,
            ff_dim=32,
            dropout=0.0,
            patch_len=6,
            stride=3,
            time_num_experts=4,
            time_top_k=2,
            frequency_num_experts=4,
            frequency_top_k=2,
            expert_hidden_size=32,
            subspace_eta=5.0,
            use_time_domain=False,
            use_frequency_domain=False,
            use_pattern_identifier=True,
            use_pattern_experts=True,
            noisy_gating=False,
        )


def test_tfps_rejects_expert_top_k_mismatch() -> None:
    with pytest.raises(ValueError, match="time_top_k"):
        TFPSModel(
            **_common_kwargs(),
            d_model=16,
            num_layers=1,
            n_heads=4,
            ff_dim=32,
            dropout=0.0,
            patch_len=6,
            stride=3,
            time_num_experts=2,
            time_top_k=3,
            frequency_num_experts=4,
            frequency_top_k=2,
            expert_hidden_size=32,
            subspace_eta=5.0,
            use_time_domain=True,
            use_frequency_domain=True,
            use_pattern_identifier=True,
            use_pattern_experts=True,
            noisy_gating=False,
        )


def test_tfps_rejects_incompatible_subspace_size() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        TFPSModel(
            input_size=10,
            input_steps=24,
            forecast_steps=24,
            airport_count=2,
            target_size=1,
            d_model=16,
            num_layers=1,
            n_heads=4,
            ff_dim=32,
            dropout=0.0,
            patch_len=6,
            stride=3,
            time_num_experts=3,
            time_top_k=1,
            frequency_num_experts=2,
            frequency_top_k=1,
            expert_hidden_size=32,
            subspace_eta=5.0,
            use_time_domain=True,
            use_frequency_domain=False,
            use_pattern_identifier=True,
            use_pattern_experts=True,
            noisy_gating=False,
        )


def test_tfps_aux_affinity_uses_forecast_patch_count() -> None:
    model = TFPSModel(
        **_common_kwargs(),
        d_model=16,
        num_layers=1,
        n_heads=4,
        ff_dim=32,
        dropout=0.0,
        patch_len=6,
        stride=3,
        time_num_experts=4,
        time_top_k=2,
        frequency_num_experts=4,
        frequency_top_k=2,
        expert_hidden_size=32,
        subspace_eta=5.0,
        use_time_domain=True,
        use_frequency_domain=True,
        use_pattern_identifier=True,
        use_pattern_experts=True,
        noisy_gating=False,
    )

    output = model(torch.randn(2, 24, 4, 13))

    assert output["prediction"].shape == (2, 24, 4, 1)
    aux = cast(dict[str, Any], output["aux"])
    assert aux["time"]["affinity"].shape == (2, model.output_patch_count, 4)
    assert aux["frequency"]["affinity"].shape == (2, model.output_patch_count, 4)


def test_tfps_forecast_steps_control_prediction_length() -> None:
    model = TFPSModel(
        input_size=52,
        input_steps=24,
        forecast_steps=12,
        airport_count=4,
        target_size=1,
        d_model=16,
        num_layers=1,
        n_heads=4,
        ff_dim=32,
        dropout=0.0,
        patch_len=6,
        stride=3,
        time_num_experts=4,
        time_top_k=2,
        frequency_num_experts=4,
        frequency_top_k=2,
        expert_hidden_size=32,
        subspace_eta=5.0,
        use_time_domain=True,
        use_frequency_domain=True,
        use_pattern_identifier=True,
        use_pattern_experts=True,
        noisy_gating=False,
    )

    output = model(torch.randn(2, 24, 4, 13))

    assert output["prediction"].shape == (2, 12, 4, 1)
