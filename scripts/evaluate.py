#!/usr/bin/env python3
"""Single evaluation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from windlab.evaluator import evaluate_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one saved run directory.")
    parser.add_argument("--run-dir", required=True, help="Path to the saved run dir.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = evaluate_run_dir(args.run_dir)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
