"""Regression tests for the native geo2rdr scalar ABI."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from faninsar.processing.geometry import public as geometry_public
from faninsar.processing.geometry.v2 import DeviceKey, Operation, SolverSettings


@pytest.mark.parametrize(
    ("device", "tail_length"),
    [(DeviceKey.cpu(), 6), (DeviceKey.cuda("test-device"), 5)],
    ids=["cpu", "cuda"],
)
def test_geo2rdr_native_scalar_abi_preserves_positive_tolerances(
    device: DeviceKey,
    tail_length: int,
) -> None:
    """Native calls pass time, range, and Doppler tolerances in ABI order."""
    captured: list[tuple[object, ...]] = []
    module = ModuleType("faninsar_native_geo2rdr_test")

    def geo2rdr(*values: object) -> list[object]:
        captured.append(values)
        return []

    setattr(
        module,
        "geo2rdr_cuda" if device.kind == "cuda" else "geo2rdr_cpu",
        geo2rdr,
    )
    solver = SolverSettings(
        max_iter=17,
        extra_iter=4,
        range_tolerance_m=0.25,
        doppler_tolerance_hz=2.5,
    )
    entry = geometry_public._resolve_native_entrypoint(
        module, Operation.GEO2RDR, device, solver
    )

    entry(
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        np.array([0.0, 1.0]),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.array([10.0, 0.002, 800000.0, 2.3, 0.0555]),
        True,
    )

    assert len(captured) == 1
    tail = captured[0][-tail_length:]
    expected = (17, 4, 1.0e-6, 0.25, 2.5)
    assert tuple(tail[:5]) == expected
    if device.kind == "cpu":
        assert tail[5] is True
