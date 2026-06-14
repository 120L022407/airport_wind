#!/usr/bin/env python3
"""Focused architecture checks for the foundation repository."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TRAIN_ENTRY = PROJECT_ROOT / "scripts" / "train.py"
EXPECTED_EVAL_ENTRY = PROJECT_ROOT / "scripts" / "evaluate.py"

FORBIDDEN_IMPORT_PREFIXES: dict[Path, tuple[str, ...]] = {
    PROJECT_ROOT / "src" / "windlab" / "config.py": (
        "windlab.trainer",
        "windlab.evaluator",
        "windlab.models",
    ),
    PROJECT_ROOT / "src" / "windlab" / "data": (
        "windlab.trainer",
        "windlab.evaluator",
    ),
}


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if ".git" not in path.parts)


def _collect_import_names(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    return imports


def _check_entrypoints(errors: list[str]) -> None:
    train_candidates = [
        path
        for path in _iter_python_files(PROJECT_ROOT)
        if path.name == "train.py" or path.name.startswith("train_")
    ]
    eval_candidates = [
        path
        for path in _iter_python_files(PROJECT_ROOT)
        if path.name == "evaluate.py" or path.name.startswith("evaluate_")
    ]
    if train_candidates != [EXPECTED_TRAIN_ENTRY]:
        errors.append(
            "Training entry points must be exactly ['scripts/train.py'], got "
            + str([str(path.relative_to(PROJECT_ROOT)) for path in train_candidates])
        )
    if eval_candidates != [EXPECTED_EVAL_ENTRY]:
        errors.append(
            "Evaluation entry points must be exactly ['scripts/evaluate.py'], got "
            + str([str(path.relative_to(PROJECT_ROOT)) for path in eval_candidates])
        )


def _check_trainer_classes(errors: list[str]) -> None:
    trainer_classes: list[str] = []
    for path in _iter_python_files(PROJECT_ROOT / "src" / "windlab"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Trainer"):
                trainer_classes.append(str(path.relative_to(PROJECT_ROOT)))
    if trainer_classes != ["src/windlab/trainer.py"]:
        errors.append(
            "Trainer classes must exist only in src/windlab/trainer.py, got "
            + str(trainer_classes)
        )


def _matches_scope(scope: Path, path: Path) -> bool:
    if scope.is_dir():
        return scope in path.parents
    return scope == path


def _check_forbidden_imports(errors: list[str]) -> None:
    for path in _iter_python_files(PROJECT_ROOT / "src" / "windlab"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _collect_import_names(tree)
        for scope, forbidden_prefixes in FORBIDDEN_IMPORT_PREFIXES.items():
            if not _matches_scope(scope, path):
                continue
            for import_name in imports:
                if any(
                    import_name == prefix or import_name.startswith(prefix + ".")
                    for prefix in forbidden_prefixes
                ):
                    errors.append(
                        f"{path.relative_to(PROJECT_ROOT)} imports forbidden module {import_name}"
                    )


def main() -> int:
    errors: list[str] = []
    _check_entrypoints(errors)
    _check_trainer_classes(errors)
    _check_forbidden_imports(errors)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("architecture checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
