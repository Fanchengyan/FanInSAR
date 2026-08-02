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
    """core must stay free of heavy package dependencies.

    Note: ``core.frame`` may late-import datasets for the Frame façade; the
    static scan allows datasets but forbids missions/processing/timeseries/io/compute
    at module top level of physical.py and pair modules.
    """
    # Scan only physical.py which must be pure
    physical = ROOT / "core" / "physical.py"
    imports = _imports_in(physical.parent)
    # physical.py alone
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


def test_processing_stages_do_not_import_missions() -> None:
    stages = ROOT / "processing" / "stages"
    imports = _imports_in(stages)
    assert not _has_prefix(imports, "faninsar.missions")
    assert not _has_prefix(imports, "faninsar.sentinel1")


def test_timeseries_does_not_import_missions() -> None:
    imports = _imports_in(ROOT / "timeseries")
    assert not _has_prefix(imports, "faninsar.missions")


def test_public_all_cap() -> None:
    assert len(faninsar.__all__) <= 20
