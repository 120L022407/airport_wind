"""Lightweight registries for framework components."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a registry operation is invalid."""


class Registry(Generic[T]):  # noqa: UP046
    """Simple name-to-object registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T) -> None:
        if key in self._items:
            raise RegistryError(f"Duplicate {self.name} key: {key}")
        self._items[key] = item

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            raise RegistryError(f"Unknown {self.name} key: {key}") from exc

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


DATA_BUILDERS: Registry[object] = Registry("data_builder")
MODELS: Registry[object] = Registry("model")
LOSSES: Registry[object] = Registry("loss")
METRICS: Registry[object] = Registry("metric")
