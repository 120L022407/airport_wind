PYTHON := python3
PYTEST := PYTHONPATH=src $(PYTHON) -m pytest
MYPY := $(PYTHON) -m mypy

.PHONY: test typecheck lint format-check architecture smoke check

test:
	$(PYTEST)

typecheck:
	PYTHONPATH=src $(MYPY) src tests scripts

lint:
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "ruff not installed; skipping lint"; fi

format-check:
	@if command -v ruff >/dev/null 2>&1; then ruff format --check .; else echo "ruff not installed; skipping format check"; fi

architecture:
	PYTHONPATH=src $(PYTHON) scripts/check_architecture.py

smoke:
	$(PYTEST) tests/test_smoke.py

check: lint format-check typecheck architecture test
