"""Verify platform markers for dependencies without Windows wheels."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
WINDOWS_EXCLUSION_MARKER = "sys_platform != 'win32'"


def test_snaphu_is_excluded_from_windows_metadata() -> None:
    """The source and lock metadata both exclude snaphu on Windows."""
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    with (REPOSITORY_ROOT / "uv.lock").open("rb") as file:
        lock = tomllib.load(file)

    project_requirement = next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("snaphu")
    )
    assert WINDOWS_EXCLUSION_MARKER in project_requirement

    faninsar_lock = next(
        package for package in lock["package"] if package["name"] == "faninsar"
    )
    locked_requirement = next(
        requirement
        for requirement in faninsar_lock["metadata"]["requires-dist"]
        if requirement["name"] == "snaphu"
    )
    assert locked_requirement["marker"] == WINDOWS_EXCLUSION_MARKER

    locked_dependency = next(
        dependency
        for dependency in faninsar_lock["dependencies"]
        if dependency["name"] == "snaphu"
    )
    assert locked_dependency["marker"] == WINDOWS_EXCLUSION_MARKER
