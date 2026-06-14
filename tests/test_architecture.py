from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_architecture_checks_pass() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_architecture.py"],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "passed" in result.stdout
