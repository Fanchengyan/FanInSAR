"""Tests for the snaphu backend gate and component reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.unwrapping import (
    SnaphuNotAvailableError,
    snaphu_available,
    snaphu_unwrap,
    unwrap,
    wrap_phase,
)
from faninsar.processing.unwrapping.reconcile import (
    align_components_to_reference,
    loop_closure_phase,
    reconcile_components,
)


def test_snaphu_backend_raises_when_not_installed() -> None:
    """Requesting snaphu without the optional extra fails explicitly."""
    if snaphu_available():
        pytest.skip("snaphu is installed in this environment")
    ifg = np.ones((8, 8), dtype=np.complex64)
    coh = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(SnaphuNotAvailableError, match="pip install faninsar"):
        snaphu_unwrap(ifg, coh)
    with pytest.raises(SnaphuNotAvailableError):
        unwrap(ifg, coh, method="snaphu")


def test_unwrap_dispatcher_irls_returns_spatial_result() -> None:
    """IRLS dispatcher returns the sole spatial result contract."""
    _, x = np.mgrid[0:12, 0:12]
    wrapped = wrap_phase(0.3 * x)
    result = unwrap(wrapped, method="irls")
    assert result.phase.shape == wrapped.shape
    assert result.component_labels.shape == wrapped.shape
    assert result.valid_mask.shape == wrapped.shape


def test_align_components_corrects_integer_cycle_offset() -> None:
    """Component alignment removes a synthetic 2π offset between islands."""
    phase = np.zeros((20, 20), dtype=np.float32)
    labels = np.ones((20, 20), dtype=np.int32)
    labels[:, 10:] = 2
    phase[:, 10:] = 2.0 * np.pi  # one cycle high
    corrected, corrections = align_components_to_reference(phase, labels)
    assert any(item.component_id == 2 and item.cycles == -1 for item in corrections)
    # medians should match after alignment
    med1 = float(np.median(corrected[labels == 1]))
    med2 = float(np.median(corrected[labels == 2]))
    assert abs(med1 - med2) < 1e-5


def test_loop_closure_and_reconcile_detect_improvement() -> None:
    """Closure residual is finite and reconciliation returns diagnostics."""
    shape = (16, 16)
    ab = np.zeros(shape, dtype=np.float32)
    bc = np.zeros(shape, dtype=np.float32)
    ac = np.zeros(shape, dtype=np.float32)
    # inject one-cycle error on ab right half component
    labels = np.ones(shape, dtype=np.int32)
    labels[:, 8:] = 2
    ab[:, 8:] = 2.0 * np.pi
    before = loop_closure_phase({"ab": ab, "bc": bc, "ac": ac}, ("ab", "bc", "ac"))
    assert before > 1.0
    result = reconcile_components(
        ab,
        labels,
        pair_phases_for_closure={"ab": ab, "bc": bc, "ac": ac},
        loop=("ab", "bc", "ac"),
    )
    assert result.corrections
    assert result.after_closure_rad <= result.before_closure_rad + 1e-6
