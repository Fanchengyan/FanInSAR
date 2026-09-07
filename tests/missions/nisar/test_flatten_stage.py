"""Focused tests for Stack flattening-stage contracts (PROPOSAL-0035)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.config import StackConfig
from faninsar.missions.nisar.provider import _sanitize_phase_screen
from faninsar.io.storage.scene_store import (
    CoregisteredSceneStore,
    form_merged_scene_interferogram,
    write_scene_unit,
)

if TYPE_CHECKING:
    from pathlib import Path


def _scene(
    root: Path,
    date_id: str,
    *,
    stage: str = "coregistration",
    phase_screen: np.ndarray | None = None,
) -> CoregisteredSceneStore:
    """Write one minimal scene artifact and reopen its validated store."""
    samples = np.ones((2, 2), dtype=np.complex64)
    write_scene_unit(
        root,
        date_id=date_id,
        reference_id="20240101",
        domain="radar",
        tag="NISAR_b0",
        primary=samples,
        secondary=samples,
        row_origin=0,
        col_origin=0,
        phase_state={"flatten_stage": stage},
        phase_screen=phase_screen,
    )
    return CoregisteredSceneStore.open(root)


def test_stack_config_rejects_unknown_flatten_stage(tmp_path: Path) -> None:
    """Flatten-stage validation fails before any production work."""
    with pytest.raises(ValueError, match="flatten_stage"):
        StackConfig(
            work_dir=tmp_path,
            activation_mode="reference",
            flatten_stage="unknown",  # type: ignore[arg-type]
        )


def test_radar_interferogram_screen_sanitizes_invalid_mapping_lanes() -> None:
    """Invalid mapping lanes persist as neutral finite Radar screen values."""
    phase = np.array([[0.37, np.nan, np.inf, -np.inf]], dtype=np.float64)

    screen = _sanitize_phase_screen(phase)

    assert screen.dtype == np.float32
    np.testing.assert_array_equal(
        screen, np.array([[0.37, 0.0, 0.0, 0.0]], dtype=np.float32)
    )


def test_interferogram_stage_applies_screen_with_ifg_sign_and_preserves_amplitude(
    tmp_path: Path,
) -> None:
    """The persisted screen contributes exp(-j phase) before output formation."""
    phase = np.full((2, 2), 0.37, dtype=np.float32)
    reference = _scene(tmp_path / "reference", "20240101", stage="interferogram")
    secondary = _scene(
        tmp_path / "secondary",
        "20240113",
        stage="interferogram",
        phase_screen=phase,
    )
    product = form_merged_scene_interferogram(
        reference,
        secondary,
        primary_role="primary",
        secondary_role="secondary",
        multilook=(1, 1),
        flatten_stage="interferogram",
    )
    np.testing.assert_allclose(np.angle(product.complex_ifg), -phase, atol=1e-6)
    np.testing.assert_allclose(np.abs(product.complex_ifg), 1.0, atol=1e-6)


def test_default_coregistration_remains_unflattened_at_ifg_boundary(
    tmp_path: Path,
) -> None:
    """Default scene formation retains its historical complex product."""
    reference = _scene(tmp_path / "reference", "20240101")
    secondary = _scene(tmp_path / "secondary", "20240113")
    product = form_merged_scene_interferogram(reference, secondary, multilook=(1, 1))
    np.testing.assert_allclose(product.complex_ifg, 1.0 + 0.0j)


def test_mixed_flattening_stages_fail_closed(tmp_path: Path) -> None:
    """A pair cannot combine scene generations from different stages."""
    reference = _scene(tmp_path / "reference", "20240101")
    secondary = _scene(
        tmp_path / "secondary",
        "20240113",
        stage="interferogram",
        phase_screen=np.zeros((2, 2), dtype=np.float32),
    )
    with pytest.raises(InvalidProcessingStateError, match="flattening stages"):
        form_merged_scene_interferogram(
            reference,
            secondary,
            flatten_stage="coregistration",
        )
