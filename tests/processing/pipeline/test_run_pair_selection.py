"""Unit tests for Stack burst-selection helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.pipeline.production import (
    _align_crop_end,
    _burst_index_list,
    _common_burst_indices,
    _normalize_burst_selection,
    _roi_burst_window,
    _select_bursts_by_roi,
    _window_crop_bounds,
    _window_halo_sufficient,
    _window_resample_halo_required,
)

SLC_ROOT = Path("/Volumes/DATA2/TEST_sentinel-1/sentinel-slc")
SLC_ROOT_RAW = Path("/Volumes/DATA2/TEST_sentinel-1/Raw Data/sentinel-slc")
SCENES = sorted(SLC_ROOT.glob("S1A_IW_SLC*.zip")) if SLC_ROOT.exists() else []
if not SCENES and SLC_ROOT_RAW.exists():
    SCENES = sorted(SLC_ROOT_RAW.glob("S1A_IW_SLC*.zip"))


def _swath(burst_times: list[datetime]) -> SimpleNamespace:
    bursts = [
        SimpleNamespace(index=index, azimuth_time=value)
        for index, value in enumerate(burst_times)
    ]
    return SimpleNamespace(
        bursts=bursts,
        lines_per_burst=1494,
        azimuth_time_interval_s=0.0020555563,
    )


def _counts() -> dict[tuple[int, str], int]:
    return {(0, "IW1"): 9, (0, "IW2"): 9, (1, "IW1"): 9, (1, "IW2"): 9}


def test_burst_index_list_accepts_all_range_and_list() -> None:
    """_burst_index_list resolves all supported selection spellings."""
    assert _burst_index_list("all", "IW1", 9) == list(range(9))
    assert _burst_index_list("2:5", "IW1", 9) == [2, 3, 4]
    assert _burst_index_list(range(2, 5), "IW1", 9) == [2, 3, 4]
    assert _burst_index_list([0, 2, 2, 4], "IW1", 9) == [0, 2, 4]


def test_burst_index_list_rejects_invalid_selections() -> None:
    """Invalid burst spellings raise before any processing starts."""
    with pytest.raises(InvalidProcessingStateError):
        _burst_index_list("bogus", "IW1", 9)
    with pytest.raises(InvalidProcessingStateError):
        _burst_index_list([], "IW1", 9)
    with pytest.raises(InvalidProcessingStateError):
        _burst_index_list([9], "IW1", 9)
    with pytest.raises(InvalidProcessingStateError):
        _burst_index_list([-1], "IW1", 9)


def test_normalize_burst_selection_none_selects_every_burst() -> None:
    """None resolves to all bursts of every frame and swath."""
    resolved = _normalize_burst_selection(None, 2, ("IW1", "IW2"), _counts())
    assert resolved[(0, "IW1")] == list(range(9))
    assert resolved[(1, "IW2")] == list(range(9))


def test_normalize_burst_selection_reuses_single_mapping() -> None:
    """One per-swath mapping applies to every frame."""
    resolved = _normalize_burst_selection(
        {"IW1": "2:5"}, 2, ("IW1", "IW2"), _counts()
    )
    assert resolved[(0, "IW1")] == [2, 3, 4]
    assert resolved[(1, "IW1")] == [2, 3, 4]
    assert resolved[(0, "IW2")] == list(range(9))


def test_normalize_burst_selection_accepts_per_frame_list() -> None:
    """A list of mappings addresses each frame individually."""
    resolved = _normalize_burst_selection(
        [{"IW1": [0]}, {"IW2": [1, 2]}], 2, ("IW1", "IW2"), _counts()
    )
    assert resolved[(0, "IW1")] == [0]
    assert resolved[(1, "IW1")] == list(range(9))
    assert resolved[(1, "IW2")] == [1, 2]


def test_normalize_burst_selection_rejects_wrong_frame_count() -> None:
    """Per-frame selections must match the number of input frames."""
    with pytest.raises(InvalidProcessingStateError):
        _normalize_burst_selection([{"IW1": [0]}], 2, ("IW1",), _counts())


def test_common_burst_indices_compares_time_of_day_across_dates() -> None:
    """Bursts on different acquisition dates align by their pass time."""
    reference = _swath([
        datetime(2016, 12, 7, 11, 18, 53, tzinfo=UTC),
        datetime(2016, 12, 7, 11, 18, 56, tzinfo=UTC),
    ])
    secondary = _swath([
        datetime(2016, 12, 31, 11, 18, 52, tzinfo=UTC),
        datetime(2016, 12, 31, 11, 18, 55, tzinfo=UTC),
    ])
    assert _common_burst_indices(reference, secondary) == [0, 1]


def test_common_burst_indices_rejects_shifted_pass_times() -> None:
    """A burst grid displaced by more than one burst window is not common."""
    reference = _swath([datetime(2016, 12, 7, 11, 18, 53, tzinfo=UTC)])
    secondary = _swath([datetime(2016, 12, 31, 11, 20, 0, tzinfo=UTC)])
    assert _common_burst_indices(reference, secondary) == []


@pytest.mark.slow
@pytest.mark.skipif(not SCENES, reason="need a local S1 ZIP scene")
def test_select_bursts_by_roi_uses_real_footprints() -> None:
    """An ROI around burst 0 selects it while a distant ROI selects nothing."""
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.query import BoundingBox

    product = open_safe_product(SCENES[0])
    first = product.swath("IW1").bursts[0]
    assert first.footprint is not None
    lon = [point[0] for point in first.footprint]
    lat = [point[1] for point in first.footprint]
    roi = BoundingBox(
        min(lon) - 0.05, min(lat) - 0.05, max(lon) + 0.05, max(lat) + 0.05
    )
    resolved = _select_bursts_by_roi(roi, [SCENES[0]], ("IW1",))
    assert 0 in resolved[(0, "IW1")]

    distant = BoundingBox(80.0, 20.0, 81.0, 21.0)
    resolved = _select_bursts_by_roi(distant, [SCENES[0]], ("IW1",))
    assert resolved[(0, "IW1")] == []


@pytest.mark.slow
@pytest.mark.skipif(not SCENES, reason="need a local S1 ZIP scene")
def test_roi_selection_ignores_explicit_swath_choice() -> None:
    """An ROI selects bursts from every swath, not just the requested one."""
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.query import BoundingBox

    product = open_safe_product(SCENES[0])
    first = product.swath("IW1").bursts[0]
    assert first.footprint is not None
    lon = [point[0] for point in first.footprint]
    lat = [point[1] for point in first.footprint]
    roi = BoundingBox(
        min(lon) - 0.05, min(lat) - 0.05, max(lon) + 0.05, max(lat) + 0.05
    )
    resolved = _select_bursts_by_roi(roi, [SCENES[0]], ("IW1", "IW2", "IW3"))
    assert resolved[(0, "IW2")] or resolved[(0, "IW3")]


@pytest.mark.slow
@pytest.mark.skipif(not SCENES, reason="need a local S1 ZIP scene")
def test_roi_burst_window_projects_onto_real_geometry() -> None:
    """The ROI radar window stays inside the burst extent."""
    from faninsar.missions.sentinel1.safe import open_safe_product
    from faninsar.processing.geometry import ConstantHeightDEM
    from faninsar.processing.pipeline.production import _radar_model
    from faninsar.query import BoundingBox

    product = open_safe_product(SCENES[0])
    swath = product.swath("IW1")
    burst = swath.bursts[0]
    shape = (swath.lines_per_burst, swath.samples_per_burst)
    geometry = _radar_model(swath, burst, shape=shape, row0=0, col0=0)
    roi = BoundingBox(80.0, 20.0, 81.0, 21.0)
    window = _roi_burst_window(
        roi, geometry, ConstantHeightDEM(0.0), shape, device="cpu"
    )
    assert window is None
    assert burst.footprint is not None
    lon = [point[0] for point in burst.footprint]
    lat = [point[1] for point in burst.footprint]
    roi = BoundingBox(
        min(lon) - 0.1, min(lat) - 0.1, max(lon) + 0.1, max(lat) + 0.1
    )
    window = _roi_burst_window(
        roi, geometry, ConstantHeightDEM(0.0), shape, device="cpu"
    )
    assert window is not None
    row0, row1, col0, col1 = window
    assert 0 <= row0 < row1 <= shape[0]
    assert 0 <= col0 < col1 <= shape[1]


def test_align_crop_end_matches_full_burst_control_grid() -> None:
    """A cropped trailing edge keeps ``end - 1`` on the stride grid."""
    assert _align_crop_end(40, 8, 8, 100) == 41
    assert _align_crop_end(47, 8, 8, 100) == 49
    assert _align_crop_end(48, 8, 8, 100) == 49
    assert _align_crop_end(90, 8, 8, 100) == 97
    assert _align_crop_end(100, 8, 8, 100) == 100


def test_window_crop_bounds_are_stride_aligned_and_cover_window() -> None:
    """The ROI crop is stride-aligned and keeps the window plus halo."""
    crop = _window_crop_bounds((100, 200, 200, 400), (512, 1024), stride=8, halo=64)
    cr0, cr1, cc0, cc1 = crop
    assert cr0 % 8 == 0
    assert cc0 % 8 == 0
    assert (cr1 - 1 - cr0) % 8 == 0 or cr1 == 512
    assert (cc1 - 1 - cc0) % 8 == 0 or cc1 == 1024
    assert cr0 <= 100
    assert cr1 >= 200
    assert cc0 <= 200
    assert cc1 >= 400
    assert cr1 - 200 >= 64
    assert cc1 - 400 >= 64
    assert 100 - cr0 >= 64
    assert 200 - cc0 >= 64


def test_window_resample_halo_required_matches_offset_extent() -> None:
    """The required halo covers the offset magnitude plus the Lanczos kernel."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult

    shape = (64, 128)
    offsets = OffsetFieldResult(
        range_offset_px=np.full(shape, 250.0, dtype=np.float32),
        azimuth_offset_px=np.zeros(shape, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    required = _window_resample_halo_required(
        offsets, (16, 48, 32, 96), (0, 0)
    )
    assert required == 255
    nan_offsets = OffsetFieldResult(
        range_offset_px=np.full(shape, np.nan, dtype=np.float32),
        azimuth_offset_px=np.full(shape, np.nan, dtype=np.float32),
        coverage=np.zeros(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    assert _window_resample_halo_required(
        nan_offsets, (16, 48, 32, 96), (0, 0)
    ) == 5


def test_window_halo_sufficient_exempts_burst_edges() -> None:
    """Burst-edge sides pass while interior shortfalls fail."""
    from faninsar.processing.coreg.offsets import OffsetFieldResult

    shape = (64, 128)
    offsets = OffsetFieldResult(
        range_offset_px=np.full(shape, 250.0, dtype=np.float32),
        azimuth_offset_px=np.zeros(shape, dtype=np.float32),
        coverage=np.ones(shape, dtype=bool),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )
    window = (16, 48, 32, 96)
    # Interior crop with 16 px margins: far too small for a 250 px offset.
    assert not _window_halo_sufficient(
        offsets, window, (8, 16), (8, 64, 16, 112), shape
    )
    # Crop touching every burst edge: identity is preserved by the edge
    # exemption (both runs read zero-filled samples past the burst).
    assert _window_halo_sufficient(
        offsets, window, (0, 0), (0, 64, 0, 128), shape
    )
