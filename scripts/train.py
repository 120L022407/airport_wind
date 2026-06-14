#!/usr/bin/env python3
"""Single training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from windlab.trainer import train_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one configured experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the output root directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = train_from_config(args.config, output_root_override=args.output_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
