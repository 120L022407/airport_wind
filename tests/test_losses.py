from __future__ import annotations

import pytest
import torch

from windlab.losses import masked_mse_loss


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
