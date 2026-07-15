"""Tests for optional backend capability reporting."""

from __future__ import annotations

import sys
import tomllib
import zipfile
from pathlib import Path
from subprocess import run

from faninsar.capabilities import format_optional_backend_capabilities

PROJECT_ROOT = Path(__file__).parents[1]


def test_base_import_does_not_load_snaphu() -> None:
    """Given a base import, when inspected, then snaphu remains unloaded."""
    # Given / When: FanInSAR was imported while collecting this test module.

    # Then
    assert "snaphu" not in sys.modules


def test_capability_output_separates_availability_and_license_caveat() -> None:
    """Given capability output, then availability and licensing are distinct."""
    # Given / When
    output = format_optional_backend_capabilities()

    # Then
    assert "snaphu-py available:" in output
    assert "snaphu-py wrapper version: unavailable" in output
    assert "bundled SNAPHU version: unavailable" in output
    assert "snaphu-py license caveat:" in output
    assert "faninsar[snaphu]" in output


def test_snaphu_is_an_optional_dependency_only() -> None:
    """Given project metadata, then snaphu is present only in its extra."""
    # Given
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    # When
    base_dependencies = metadata["project"]["dependencies"]
    snaphu_dependencies = metadata["project"]["optional-dependencies"]["snaphu"]

    # Then
    assert all(not dependency.startswith("snaphu") for dependency in base_dependencies)
    assert snaphu_dependencies == ["snaphu>=0.4.1"]


def test_policy_artifacts_are_declared_for_wheel_installation() -> None:
    """Given package metadata, then policy artifacts install with the wheel."""
    # Given / When
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    installed_files = metadata["tool"]["setuptools"]["data-files"]

    # Then
    assert installed_files["share/faninsar/LICENSES"] == ["LICENSES/*"]
    assert installed_files["share/faninsar/docs"] == [
        "docs/development/processing_provenance.md"
    ]


def test_built_wheel_contains_installable_policy_artifacts(tmp_path: Path) -> None:
    """Given a wheel build, then required policy artifacts are included."""
    # Given
    wheel_directory = tmp_path / "dist"

    # When
    run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_directory)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(wheel_directory.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_members = wheel.namelist()

    # Then
    assert any(name.endswith("LICENSES/snaphu-py-NOTICE.md") for name in wheel_members)
    assert any(
        name.endswith("LICENSES/source_provenance.toml") for name in wheel_members
    )
    assert any(name.endswith("docs/processing_provenance.md") for name in wheel_members)
