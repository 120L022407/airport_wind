from __future__ import annotations

import pytest

from windlab.registry import Registry, RegistryError


def test_registry_rejects_duplicate_key() -> None:
    registry: Registry[int] = Registry("demo")
    registry.register("item", 1)
    with pytest.raises(RegistryError, match="Duplicate"):
        registry.register("item", 2)


def test_registry_rejects_unknown_key() -> None:
    registry: Registry[int] = Registry("demo")
    with pytest.raises(RegistryError, match="Unknown"):
        registry.get("missing")
