"""Contract and CPU-boundary tests for the native-v2 vertical slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.geometry.native_v2 import (
    NATIVE_RESULT_FIELDS,
    result_from_native_outputs,
)

SOURCE_ROOT = (
    Path(__file__).parents[3]
    / "faninsar"
    / "processing"
    / "geometry"
    / "native_v2"
)


def test_cpu_sources_export_both_operation_symbols_and_openmp_loop() -> None:
    """Both operation translation units contain the exact OpenMP seam."""
    binding = (SOURCE_ROOT / "bindings.cpp").read_text()
    for operation in ("geo2rdr", "rdr2geo"):
        source = (SOURCE_ROOT / f"{operation}.cpp").read_text()
        assert f"{operation}_cpu" in source
        assert "#pragma omp parallel for" in source
        assert "record_visit(point)" in source
        assert f'"{operation}_cpu"' in binding


def test_native_result_binding_centralizes_fourteen_field_validation() -> None:
    """Native outputs are converted to the exact public v2 result contract."""
    values: list[np.ndarray] = [
        np.zeros(2, dtype=np.float64) for _ in NATIVE_RESULT_FIELDS
    ]
    values[5] = np.ones(2, dtype=bool)
    values[6] = np.ones(2, dtype=np.int32)
    values[9] = np.ones(2, dtype=np.float64)
    values[10] = np.zeros(2, dtype=bool)
    values[11] = np.zeros(2, dtype=bool)
    result = result_from_native_outputs(values, operation="geo2rdr")

    assert result.fields == NATIVE_RESULT_FIELDS
    assert result.iterations.dtype == np.dtype(np.int32)
    assert result.converged.dtype == np.dtype(bool)


def test_native_result_binding_rejects_non_fourteen_field_abi() -> None:
    """The Python seam fails closed if an extension returns the wrong ABI."""
    with pytest.raises(ValueError, match="exactly fourteen"):
        result_from_native_outputs([np.zeros(1)] * 13, operation="rdr2geo")

