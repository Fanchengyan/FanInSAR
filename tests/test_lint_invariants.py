"""Import graph invariants for the greenfield package layout."""

from __future__ import annotations

import ast
from pathlib import Path

import faninsar

ROOT = Path(faninsar.__file__).resolve().parent


def _imports_in(package_dir: Path) -> set[str]:
    """Collect top-level ``faninsar.*`` import roots from Python files."""
    found: set[str] = set()
    if not package_dir.is_dir():
        return found
    for path in package_dir.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("faninsar."):
                        found.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("faninsar."):
                    found.add(node.module)
                elif node.level == 0 and node.module == "faninsar":
                    for alias in node.names:
                        found.add(f"faninsar.{alias.name}")
    return found


def _has_prefix(imports: set[str], prefix: str) -> bool:
    return any(imp == prefix or imp.startswith(prefix + ".") for imp in imports)


def test_core_does_not_import_missions_processing_timeseries_io_compute() -> None:
    """Core physical values must stay free of heavy package dependencies."""
    physical = ROOT / "core" / "physical.py"
    text = physical.read_text(encoding="utf-8")
    tree = ast.parse(text)
    pure_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "faninsar."
        ):
            pure_imports.add(node.module)
    forbidden = (
        "faninsar.missions",
        "faninsar.processing",
        "faninsar.timeseries",
        "faninsar.io",
        "faninsar.compute",
    )
    for prefix in forbidden:
        assert not _has_prefix(pure_imports, prefix), pure_imports


def test_retired_processing_stages_are_absent() -> None:
    """The retired generic processing aggregate is no longer importable."""
    import importlib.util

    retired = (
        "faninsar.processing.stages",
        "faninsar.stack.s1",
        "faninsar.stack.nisar",
        "faninsar.stack.nisar_provider",
        "faninsar.stack.stack_api",
        "faninsar.stack.network",
        "faninsar.processing.coordinates",
        "faninsar.processing.readers",
        "faninsar.processing.synthetic_slc",
        "faninsar.data.datasets.geobox",
    )
    assert all(importlib.util.find_spec(name) is None for name in retired)


def test_timeseries_does_not_import_missions() -> None:
    """Time-series algorithms must remain mission-neutral."""
    imports = _imports_in(ROOT / "timeseries")
    assert not _has_prefix(imports, "faninsar.missions")


def test_public_all_cap() -> None:
    """The root API stays intentionally small."""
    assert len(faninsar.__all__) <= 20
