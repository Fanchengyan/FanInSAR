"""Release qualification gates for the Python-native InSAR pipeline."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from faninsar.logging import setup_logger
from faninsar.processing.runtime.capabilities import (
    format_backend_capabilities,
    snaphu_capability,
)
from faninsar.processing.runtime.device_matrix import capability_matrix, probe_devices

logger = setup_logger(__name__)

FORBIDDEN_RUNTIME_MODULES: tuple[str, ...] = (
    "isce3",
    "isce2",
    "gmtsar",
)


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    """Machine-readable release qualification summary."""

    passed: bool
    checks: dict[str, bool]
    details: dict[str, Any]


def assert_no_external_sar_runtime() -> bool:
    """Return True when ISCE3/GMTSAR modules are absent from the import path."""
    present = [
        name for name in FORBIDDEN_RUNTIME_MODULES if find_spec(name) is not None
    ]
    if present:
        logger.error("Forbidden SAR runtime modules importable: %s", present)
        return False
    return True


def assert_core_imports() -> bool:
    """Return True when required FanInSAR processing entrypoints import cleanly."""
    modules = (
        "faninsar.missions.s1.processing",
        "faninsar.processing.unwrapping",
        "faninsar.missions.s1",
        "faninsar.processing.runtime.execution",
        "faninsar.processing.runtime.dask_exec",
        "faninsar.processing.runtime.device_matrix",
        "faninsar.missions.nisar",
    )
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            logger.exception("Failed to import %s", name)
            return False
    return True


def run_release_gates() -> ReleaseGateResult:
    """Execute lightweight release qualification gates.

    Returns
    -------
    ReleaseGateResult
        Aggregate pass/fail status with per-check results and diagnostics.

    """
    checks = {
        "no_external_sar_runtime": assert_no_external_sar_runtime(),
        "core_imports": assert_core_imports(),
        "capability_matrix_nonempty": len(capability_matrix()) > 0,
        "snaphu_capability_reports": snaphu_capability().display_name == "snaphu-py",
    }
    details: dict[str, Any] = {
        "devices": probe_devices(),
        "snaphu": format_backend_capabilities(),
        "kernels": [row.name for row in capability_matrix()],
        "forbidden_runtime_modules": list(FORBIDDEN_RUNTIME_MODULES),
    }
    passed = all(checks.values())
    if passed:
        logger.info("Release gates passed")
    else:
        logger.error("Release gates failed: %s", checks)
    return ReleaseGateResult(passed=passed, checks=checks, details=details)
