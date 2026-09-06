"""Unit tests for network misregistration invert (PROPOSAL-0017)."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.coregistration.geometric_phase import (
    apply_geometric_phase_from_range_offset,
    geometric_phase_from_range_offset,
    stage_topo,
)
from faninsar.processing.coregistration.geometry_coreg import combine_offset_fields
from faninsar.processing.coregistration.misreg_network import (
    MisregArc,
    invert_pair_misregistration,
)
from faninsar.processing.coregistration.offsets import OffsetFieldResult


def test_invert_pair_misregistration_closed_loop() -> None:
    """Synthetic date misregs recover from pair differences."""
    # Truth: m0=0, m1=0.2 az / 0.1 rg, m2=-0.1 az / 0.05 rg
    truth_az = {"20160101": 0.0, "20160113": 0.2, "20160125": -0.1}
    truth_rg = {"20160101": 0.0, "20160113": 0.1, "20160125": 0.05}
    pairs = [
        ("20160101", "20160113"),
        ("20160113", "20160125"),
        ("20160101", "20160125"),
    ]
    arcs = [
        MisregArc(
            primary=a,
            secondary=b,
            azimuth_shift_px=truth_az[b] - truth_az[a],
            range_shift_px=truth_rg[b] - truth_rg[a],
            azimuth_sigma_px=0.01,
            range_sigma_px=0.01,
            method="synthetic",
            n_valid=100,
        )
        for a, b in pairs
    ]
    result = invert_pair_misregistration(arcs, reference="20160101")
    assert result.azimuth_px["20160101"] == 0.0
    assert result.range_px["20160101"] == 0.0
    assert result.azimuth_px["20160113"] == pytest.approx(0.2, abs=1e-6)
    assert result.range_px["20160113"] == pytest.approx(0.1, abs=1e-6)
    assert result.azimuth_px["20160125"] == pytest.approx(-0.1, abs=1e-6)
    assert result.range_px["20160125"] == pytest.approx(0.05, abs=1e-6)


def test_invert_rejects_disconnected_network() -> None:
    """Dates with no arcs fail hard (default QC)."""
    arcs = [
        MisregArc(
            primary="20160101",
            secondary="20160113",
            azimuth_shift_px=0.1,
            range_shift_px=0.0,
            n_valid=10,
        ),
    ]
    with pytest.raises(Exception, match="disconnected"):
        invert_pair_misregistration(
            arcs,
            reference="20160101",
            dates=["20160101", "20160113", "20160201"],
        )


def test_combine_offset_fields_includes_misreg_constants() -> None:
    """Network constants enter the combined dense offset field."""
    geo = OffsetFieldResult(
        range_offset_px=np.ones((4, 4), dtype=np.float32),
        azimuth_offset_px=np.zeros((4, 4), dtype=np.float32),
        coverage=np.ones((4, 4), dtype=bool),
        uncertainty_px=np.zeros((4, 4), dtype=np.float32),
    )
    out = combine_offset_fields(
        geo,
        esd_azimuth_shift_px=0.1,
        amplitude_residual_rg=0.2,
        misreg_az_px=0.3,
        misreg_rg_px=0.4,
    )
    assert float(out.azimuth_offset_px[0, 0]) == pytest.approx(0.4)
    assert float(out.range_offset_px[0, 0]) == pytest.approx(1.6)


def test_geometric_phase_from_range_offset_scale() -> None:
    """Phase scales as 4π Δr / λ per range pixel."""
    spacing = 2.3
    wavelength = 0.055
    phase = geometric_phase_from_range_offset(
        1.0,
        range_spacing_m=spacing,
        wavelength_m=wavelength,
    )
    assert float(phase) == pytest.approx(4.0 * np.pi * spacing / wavelength)


def test_apply_geometric_phase_skips_when_already_removed() -> None:
    """already_removed flag is a no-op."""
    samples = np.ones((2, 2), dtype=np.complex64)
    out, mode = apply_geometric_phase_from_range_offset(
        samples,
        1.0,
        range_spacing_m=2.3,
        wavelength_m=0.055,
        already_removed=True,
    )
    assert mode == "none"
    assert np.allclose(out, samples)


def test_stage_topo_hard_deleted() -> None:
    """Public stage_topo raises (PROPOSAL-0017)."""
    with pytest.raises(RuntimeError, match="PROPOSAL-0017"):
        stage_topo()
