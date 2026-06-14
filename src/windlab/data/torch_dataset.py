"""PyTorch dataset adapters for windowed data contracts."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from windlab.data.windows import WindowedSplit


class WindowedTorchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper around the public `WindowedSplit` contract."""

    def __init__(self, split: WindowedSplit) -> None:
        self.inputs = torch.as_tensor(split.inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(split.targets, dtype=torch.float32)
        self.masks = torch.as_tensor(split.observed_target_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index], self.masks[index]
