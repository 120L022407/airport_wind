from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from windlab.losses import build_forecast_loss, masked_mse_loss


def _loss_config(*terms: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        name="composite",
        terms=[
            SimpleNamespace(
                name=term["name"],
                weight=term["weight"],
                params=term.get("params", {}),
            )
            for term in terms
        ],
    )


def test_masked_mse_uses_only_observed_points() -> None:
    prediction = torch.tensor([[[[1.0], [10.0]], [[3.0], [30.0]]]])
    target = torch.tensor([[[[2.0], [100.0]], [[1.0], [300.0]]]])
    mask = torch.tensor([[[[True], [False]], [[True], [False]]]])

    loss = masked_mse_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(((1.0 - 2.0) ** 2 + (3.0 - 1.0) ** 2) / 2)


def test_masked_mse_rejects_empty_mask() -> None:
    prediction = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)
    mask = torch.zeros(1, 2, 1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="selects no elements"):
        masked_mse_loss(prediction, target, mask)


def test_fourier_loss_fal_mode_respects_full_sequence_mask() -> None:
    prediction = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            [[[10.0]], [[20.0]], [[30.0]], [[40.0]]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [[[2.0]], [[3.0]], [[4.0]], [[5.0]]],
            [[[11.0]], [[21.0]], [[31.0]], [[41.0]]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [[[True]], [[True]], [[True]], [[True]]],
            [[[True]], [[False]], [[True]], [[True]]],
        ]
    )
    loss_fn = build_forecast_loss(
        _loss_config(
            {
                "name": "fourier_amplitude_correlation",
                "weight": 1.0,
                "params": {"mode": "fal"},
            }
        ),
        total_train_steps=8,
    )

    loss = loss_fn({"prediction": prediction}, target, mask)

    first_prediction = prediction[:1, :, :, :].permute(0, 2, 3, 1).reshape(1, 4)
    first_target = target[:1, :, :, :].permute(0, 2, 3, 1).reshape(1, 4)
    fft_prediction = torch.fft.fft(first_prediction, dim=-1, norm="ortho")
    fft_target = torch.fft.fft(first_target, dim=-1, norm="ortho")
    expected = torch.sqrt(torch.tensor(4.0)) * torch.nn.functional.mse_loss(
        fft_prediction.abs(),
        fft_target.abs(),
    )

    assert torch.allclose(loss, expected)
    cast(Any, loss).backward()
    assert prediction.grad is not None


def test_fourier_loss_fcl_mode_has_expected_shape_and_backward() -> None:
    prediction = torch.randn(2, 24, 1, 1, requires_grad=True)
    target = torch.randn(2, 24, 1, 1)
    loss_fn = build_forecast_loss(
        _loss_config(
            {
                "name": "fourier_amplitude_correlation",
                "weight": 1.0,
                "params": {"mode": "fcl"},
            }
        ),
        total_train_steps=10,
    )

    loss = loss_fn({"prediction": prediction}, target, None)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    cast(Any, loss).backward()
    assert prediction.grad is not None


def test_composite_loss_combines_mse_and_fourier_terms() -> None:
    prediction = torch.randn(2, 24, 1, 1, requires_grad=True)
    target = torch.randn(2, 24, 1, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_fn = build_forecast_loss(
        _loss_config(
            {"name": "mse", "weight": 1.0},
            {
                "name": "fourier_amplitude_correlation",
                "weight": 0.2,
                "params": {"mode": "fal"},
            },
        ),
        total_train_steps=12,
    )

    loss = loss_fn({"prediction": prediction}, target, mask)

    assert torch.isfinite(loss)
    cast(Any, loss).backward()
    assert prediction.grad is not None


def test_build_forecast_loss_rejects_invalid_fourier_params() -> None:
    with pytest.raises(ValueError, match="alpha"):
        build_forecast_loss(
            _loss_config(
                {
                    "name": "fourier_amplitude_correlation",
                    "weight": 1.0,
                    "params": {"mode": "paper_random"},
                }
            ),
            total_train_steps=6,
        )


def test_forecast_loss_adds_hcan_auxiliary_terms() -> None:
    target = torch.tensor([[[[0.2]], [[0.8]]]], dtype=torch.float32)
    prediction = torch.tensor(
        [[[[0.1]], [[0.7]]]], dtype=torch.float32, requires_grad=True
    )
    coarse_prediction = torch.zeros(
        1,
        2,
        1,
        1,
        2,
        dtype=torch.float32,
        requires_grad=True,
    )
    fine_prediction = torch.zeros(
        1,
        2,
        1,
        1,
        4,
        dtype=torch.float32,
        requires_grad=True,
    )
    coarse_logits = torch.zeros(
        1,
        2,
        1,
        1,
        2,
        dtype=torch.float32,
        requires_grad=True,
    )
    fine_logits = torch.zeros(
        1,
        2,
        1,
        1,
        4,
        dtype=torch.float32,
        requires_grad=True,
    )
    fine_logits_switch = torch.zeros(
        1,
        2,
        1,
        1,
        2,
        dtype=torch.float32,
        requires_grad=True,
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_fn = build_forecast_loss(
        _loss_config({"name": "hcan_auxiliary", "weight": 1.0}),
        train_targets=np.array([[[[0.0]]], [[[1.0]]]], dtype=np.float64),
        train_mask=np.ones((2, 1, 1, 1), dtype=bool),
        annealing_steps=2,
        total_train_steps=4,
    )

    loss = loss_fn(
        {
            "prediction": prediction,
            "aux": {
                "hcan": {
                    "coarse_prediction": coarse_prediction,
                    "coarse_logits": coarse_logits,
                    "fine_prediction": fine_prediction,
                    "fine_logits": fine_logits,
                    "fine_logits_switch": fine_logits_switch,
                    "num_coarse": 2,
                    "num_fine": 4,
                    "lambda_cls": 1.0,
                    "lambda_reg": 1.0,
                    "lambda_acl": 1.0,
                    "lambda_direct": 1.0,
                }
            },
        },
        target,
        mask,
    )

    assert torch.isfinite(loss)
    cast(Any, loss).backward()
    assert prediction.grad is not None
