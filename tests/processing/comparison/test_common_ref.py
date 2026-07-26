"""Tests for common reference-pixel selection and residual subtraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faninsar.processing.comparison.common_ref import (
    load_pinned_ref_point,
    resolve_common_ref_point,
    select_common_ref_point,
    shared_residual_limits,
    subtract_ref_value,
)
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.unwrap import snaphu_available, snaphu_unwrap
from faninsar.processing.unwrap.snaphu_backend import SnaphuConfig


def test_select_common_ref_argmax_mean_coherence() -> None:
    """Selected pixel maximises mean of three coherences on triple-valid mask."""
    fan = np.full((8, 8), 0.2, dtype=np.float32)
    isce = np.full((8, 8), 0.2, dtype=np.float32)
    ins = np.full((8, 8), 0.2, dtype=np.float32)
    # Peak mean at (3, 5)
    fan[3, 5] = 0.9
    isce[3, 5] = 0.8
    ins[3, 5] = 0.85
    # Higher single-stack but lower mean elsewhere
    fan[0, 0] = 0.99
    isce[0, 0] = 0.2
    ins[0, 0] = 0.2
    valid = np.ones((8, 8), dtype=bool)
    ref = select_common_ref_point(
        {"fan": fan, "isce": isce, "insardev": ins},
        valid=valid,
        min_coherence=0.15,
        smooth_window=1,
    )
    assert (ref.row, ref.col) == (3, 5)
    assert ref.mean_coherence == pytest.approx((0.9 + 0.8 + 0.85) / 3.0)
    assert ref.coherences["fan"] == pytest.approx(0.9)


def test_select_common_ref_respects_valid_mask() -> None:
    """Invalid mask must exclude the absolute peak if marked invalid."""
    fan = np.full((6, 6), 0.3, dtype=np.float32)
    isce = fan.copy()
    ins = fan.copy()
    fan[1, 1] = 0.95
    isce[1, 1] = 0.95
    ins[1, 1] = 0.95
    fan[4, 4] = 0.7
    isce[4, 4] = 0.7
    ins[4, 4] = 0.7
    valid = np.ones((6, 6), dtype=bool)
    valid[1, 1] = False
    ref = select_common_ref_point(
        {"fan": fan, "isce": isce, "insardev": ins},
        valid=valid,
        min_coherence=0.15,
        smooth_window=1,
    )
    assert (ref.row, ref.col) == (4, 4)


def test_subtract_ref_value_zeros_reference() -> None:
    """Residual at the reference pixel is approximately zero."""
    unw = np.arange(20, dtype=np.float32).reshape(4, 5)
    ref = select_common_ref_point(
        {
            "a": np.ones((4, 5), dtype=np.float32) * 0.5,
            "b": np.ones((4, 5), dtype=np.float32) * 0.5,
        },
        min_coherence=0.1,
        smooth_window=1,
    )
    # Force known location
    from faninsar.processing.comparison.common_ref import CommonRefPoint

    ref = CommonRefPoint(row=2, col=3, mean_coherence=0.5, coherences={"a": 0.5, "b": 0.5})
    res = subtract_ref_value(unw, ref)
    assert res[2, 3] == pytest.approx(0.0)
    assert res[0, 0] == pytest.approx(float(unw[0, 0] - unw[2, 3]))


def test_shared_residual_limits_symmetric() -> None:
    """Shared limits are symmetric about zero and cover residual mass."""
    residuals = {
        "fan": np.array([1.0, -2.0, 0.5], dtype=np.float32),
        "isce": np.array([0.1, -0.2], dtype=np.float32),
    }
    vmin, vmax = shared_residual_limits(residuals, percentile=100.0)
    assert vmin == pytest.approx(-vmax)
    assert vmax >= 2.0 - 1e-6


def test_snaphu_unwrap_shipped_path_produces_unwrapped_span() -> None:
    """Real snaphu_unwrap entry returns method=snaphu and span beyond ±π."""
    if not snaphu_available():
        pytest.skip("snaphu not installed")
    y, x = np.mgrid[0:48, 0:48]
    true = 0.2 * x.astype(np.float64)
    ifg = np.exp(1j * true).astype(np.complex64)
    coh = np.full(true.shape, 0.85, dtype=np.float32)
    result = snaphu_unwrap(ifg, coh, config=SnaphuConfig(cost="defo", nlooks=1.0))
    assert result.method == "snaphu"
    unw = result.unwrapped_phase
    assert unw.shape == ifg.shape
    assert np.isfinite(unw).mean() > 0.9
    # Not still confined to wrapped domain only
    assert float(np.nanmax(unw) - np.nanmin(unw)) > 2.0 * np.pi - 0.5


def test_select_common_ref_raises_when_empty() -> None:
    """No valid pixel raises a typed processing error."""
    z = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(InvalidProcessingStateError):
        select_common_ref_point(
            {"a": z, "b": z},
            min_coherence=0.5,
            smooth_window=1,
        )


def test_load_pinned_ref_point_from_json(tmp_path: Path) -> None:
    """Pinned JSON fixes row/col; does not re-run argmax."""
    pin = tmp_path / "common_ref_point.json"
    pin.write_text(
        json.dumps(
            {
                "row": 175,
                "col": 2276,
                "mean_coherence": 0.71,
                "coherences": {"fan": 0.7, "isce": 0.5, "insardev": 0.9},
            }
        )
    )
    ref = load_pinned_ref_point(pin)
    assert (ref.row, ref.col) == (175, 2276)
    assert ref.mean_coherence == pytest.approx(0.71)


def test_resolve_prefers_pin_over_argmax(tmp_path: Path) -> None:
    """With pin present, resolve returns pin even if another pixel has higher γ."""
    pin = tmp_path / "common_ref_point.json"
    pin.write_text(json.dumps({"row": 1, "col": 2, "mean_coherence": 0.5}))
    # Argmax would pick (0, 0) with high coherence
    coh = np.full((4, 5), 0.2, dtype=np.float32)
    coh[0, 0] = 0.99
    coh[1, 2] = 0.4
    ref = resolve_common_ref_point(
        {"a": coh, "b": coh},
        pinned_path=pin,
        allow_auto=False,
        smooth_window=1,
    )
    assert (ref.row, ref.col) == (1, 2)


def test_resolve_refuses_auto_without_pin(tmp_path: Path) -> None:
    """Missing pin + allow_auto=False must not silently re-select."""
    missing = tmp_path / "no_such_pin.json"
    coh = np.ones((3, 3), dtype=np.float32) * 0.5
    with pytest.raises(InvalidProcessingStateError):
        resolve_common_ref_point(
            {"a": coh, "b": coh},
            pinned_path=missing,
            allow_auto=False,
        )


def test_resolve_allow_auto_falls_back_to_select(tmp_path: Path) -> None:
    """allow_auto=True with missing pin re-selects (explicit opt-in only)."""
    missing = tmp_path / "no_such_pin.json"
    coh = np.full((4, 4), 0.3, dtype=np.float32)
    coh[2, 3] = 0.9
    ref = resolve_common_ref_point(
        {"a": coh, "b": coh},
        pinned_path=missing,
        allow_auto=True,
        min_coherence=0.15,
        smooth_window=1,
    )
    assert (ref.row, ref.col) == (2, 3)


def test_campaign_pin_is_175_2276() -> None:
    """Repo campaign pin must stay at the three-way agreed pixel."""
    from faninsar.processing.comparison.common_ref import DEFAULT_PINNED_REF_JSON

    if not DEFAULT_PINNED_REF_JSON.is_file():
        pytest.skip("campaign pin file not present")
    ref = load_pinned_ref_point(DEFAULT_PINNED_REF_JSON)
    assert (ref.row, ref.col) == (175, 2276)
