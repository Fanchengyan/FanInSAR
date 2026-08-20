"""Tests for PROPOSAL-0031 production geometry helper."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faninsar.processing.contracts import OrbitMetadata, OrbitStateVector
from faninsar.processing.geometry.backend_dispatch import DispatchError
from faninsar.processing.geometry.native_v2.builder import (
    NativeBackend,
    NativeBuilder,
    NativeBuildRequest,
    NativeOperation,
)
from faninsar.processing.geometry.orbit import OrbitInterpolator
from faninsar.processing.geometry.prepare_production import (
    prepare_production_geometry,
    run_geo2rdr,
    run_rdr2geo,
)
from faninsar.processing.geometry.transforms import RadarGeometryModel
from faninsar.processing.geometry.v2 import Operation


def _model() -> RadarGeometryModel:
    """Build a tiny circular-orbit radar model for helper tests."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    r_sat = 7_071_000.0
    v_sat = 7_000.0
    omega = v_sat / r_sat
    vectors = []
    for i in range(3):
        t = t0 + timedelta(seconds=(i - 1))
        angle = omega * (i - 1)
        pos = (r_sat * np.cos(angle), r_sat * np.sin(angle), 0.0)
        vel = (-v_sat * np.sin(angle), v_sat * np.cos(angle), 0.0)
        vectors.append(OrbitStateVector(time=t, position_m=pos, velocity_m_s=vel))
    orbit = OrbitMetadata(
        reference_frame="ECR",
        source="test",
        vectors=tuple(vectors),
    )
    return RadarGeometryModel(
        orbit=OrbitInterpolator.from_orbit(orbit),
        sensing_start=t0,
        azimuth_time_interval_s=1.0,
        starting_slant_range_m=700_000.0,
        range_spacing_m=10.0,
        wavelength_m=0.056,
        look_direction="right",
    )


def test_cuda_geo_plan_is_supported_for_build() -> None:
    """PROPOSAL-0031 flips CUDA geo2rdr/rdr2geo plan.supported for prepare."""
    builder = NativeBuilder()
    for operation in (NativeOperation.GEO2RDR, NativeOperation.RDR2GEO):
        plan = builder.plan(NativeBuildRequest(operation, NativeBackend.CUDA))
        assert plan.supported is True
        assert plan.unsupported_reason == ""


def test_prepare_production_geometry_requires_device() -> None:
    """Callers must pass device; there is no CPU default."""
    params = inspect.signature(prepare_production_geometry).parameters
    assert params["device"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="device"):
        prepare_production_geometry(  # type: ignore[misc]
            Operation.GEO2RDR,
            _model(),
            shape=(1, 1),
        )


def test_prepare_production_geometry_cpu_roundtrip() -> None:
    """CPU helper prepares and execute_geometry runs without UUID/executor."""
    model = _model()
    prepared = prepare_production_geometry(
        Operation.RDR2GEO,
        model,
        device="cpu",
        shape=(2, 2),
    )
    az = np.zeros((2, 2), dtype=np.float64)
    rg = np.zeros((2, 2), dtype=np.float64)
    result = run_rdr2geo(model, az, rg, device="cpu", max_iter=8)
    assert result.latitude_deg.shape == (2, 2)
    assert result.converged.dtype == bool
    geo = run_geo2rdr(
        model,
        result.latitude_deg,
        result.longitude_deg,
        np.where(np.isfinite(result.height_m), result.height_m, 0.0),
        device="cpu",
        max_iter=8,
    )
    assert geo.range_index.shape == (2, 2)
    _ = prepared


def test_prepare_production_geometry_rejects_mps() -> None:
    """MPS fails closed after Newton deletion."""
    pytest.importorskip("torch")
    import torch

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        with pytest.raises(DispatchError):
            prepare_production_geometry(
                Operation.GEO2RDR,
                _model(),
                device="mps",
                shape=(1,),
            )
        return
    with pytest.raises(DispatchError, match="mps"):
        prepare_production_geometry(
            Operation.GEO2RDR,
            _model(),
            device="mps",
            shape=(1,),
        )
