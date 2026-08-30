"""Focused PROPOSAL-0040 tests for the radar projection seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from faninsar.processing.masking.radar_projection import (
    project_mask_to_radar,
    radar_projection_cache_key,
)


def _lut() -> SimpleNamespace:
    """Return a sparse lookup with valid and invalid source cells."""
    return SimpleNamespace(
        az_full=np.array([[0.1, 1.1], [2.0, np.nan]]),
        rg_full=np.array([[0.2, 1.1], [2.0, np.nan]]),
        valid=np.array([[True, True], [True, False]]),
        full_radar_shape=(3, 3),
    )


def test_geo_projection_is_canonical_and_leaves_holes_invalid() -> None:
    """Nearest LUT scatter preserves labels and does not fill holes."""
    result = project_mask_to_radar(
        np.array([[1, 0], [255, 1]], dtype=np.uint8),
        lut=_lut(),
        full_radar_shape=(3, 3),
    )

    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 1, 255}
    np.testing.assert_array_equal(
        result, [[1, 255, 255], [255, 0, 255], [255, 255, 255]]
    )


def test_target_validity_is_applied_after_projection() -> None:
    """Target invalidity wins over a valid source label."""
    result = project_mask_to_radar(
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
        lut=_lut(),
        full_radar_shape=(3, 3),
        target_validity=np.array([[False, True, True], [True, True, True], [True] * 3]),
    )
    assert result[0, 0] == 255


def test_invalid_source_label_remains_invalid() -> None:
    """Invalid source labels are not converted to keep or excluded."""
    result = project_mask_to_radar(
        np.array([[255, 0], [0, 0]], dtype=np.uint8),
        lut=_lut(),
        full_radar_shape=(3, 3),
    )
    assert result[0, 0] == 255


def test_noncanonical_input_is_rejected() -> None:
    """Arbitrary integer labels cannot silently become keep labels."""
    with pytest.raises(ValueError, match="canonical labels"):
        project_mask_to_radar(
            np.array([[2]], dtype=np.uint8), lut=_lut(), full_radar_shape=(3, 3)
        )


def test_multilook_uses_nearest_hard_label_without_aggregation() -> None:
    """Multilooking samples one deterministic hard label, never aggregates."""
    lut = SimpleNamespace(
        az_full=np.array([[0.0, 0.0]]),
        rg_full=np.array([[0.0, 1.0]]),
        valid=np.array([[True, True]]),
        full_radar_shape=(1, 2),
    )
    result = project_mask_to_radar(
        np.array([[0, 1]], dtype=np.uint8), lut=lut, multilook=(1, 2)
    )
    np.testing.assert_array_equal(result, [[0]])


def test_radar_mode_scatter_is_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked geometry mode scatters labels and keeps uncovered cells invalid."""
    import faninsar.processing.geometry.prepare_production as production

    def fake_geo2rdr(*args: object, **kwargs: object) -> SimpleNamespace:
        del kwargs
        latitude = np.asarray(args[1])
        return SimpleNamespace(
            azimuth_index=np.zeros(latitude.shape),
            range_index=np.zeros(latitude.shape),
            converged=np.ones(latitude.shape, dtype=bool),
        )

    monkeypatch.setattr(production, "run_geo2rdr", fake_geo2rdr)
    result = project_mask_to_radar(
        np.array([[1, 0]], dtype=np.uint8),
        geometry=object(),
        mask_transform=Affine.identity(),
        full_radar_shape=(2, 2),
        chunk_size=1,
    )
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 1, 255}
    assert result[1, 1] == 255


def test_cache_identity_binds_projection_inputs() -> None:
    """Changing any observable projection input changes the cache key."""
    base = {
        "source_mask_identity": "mask",
        "master_grid": "master",
        "target_grid": "target",
    }
    first = radar_projection_cache_key(
        **base, reference_scene="a", geometry="g", dem="d"
    )
    second = radar_projection_cache_key(
        **base, reference_scene="b", geometry="g", dem="d"
    )
    third = radar_projection_cache_key(
        **base, reference_scene="a", geometry="h", dem="d"
    )
    assert first != second
    assert first != third
