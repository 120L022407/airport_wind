"""Shared utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_dir(output_root: str | Path, run_name: str) -> Path:
    root = ensure_dir(output_root)
    candidate = root / run_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        candidate = root / f"{run_name}-{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def timestamped_run_name(base_name: str) -> str:
    return base_name
