"""Slice D2 Stack-integration tests for the mask products (PROPOSAL-0039).

Covers TDD-plan items 19, 20, 22, and 23 plus the ionosphere opt-in:

19. IFG ``valid_mask`` intersection: the mask is intersected at formation
    (``valid_mask &= ~mask``), resampled with nearest-neighbour semantics
    when the mask grid differs from the IFG grid, never touches the
    ``PhaseFilter``, and a mask-absent run is unchanged.
20. Unwrap caller-supplied mask: the support rule persisted IFG mask ∧
    finite phase ∧ caller mask (the unwrapper owns the finite-value terms).
22. Radar projection: nearest-neighbour scatter + nearest fill through a
    dense ``Geo2RdrLUT`` (geo mode), chunked ``run_geo2rdr`` (radar mode),
    the ``(vector digest, buffer, resolution, DEM identity)`` cache, and the
    multilook ``any()`` reduction.
23. STAC ``mask`` asset: the buffered binary mask registers beside the Zarr
    store with its provenance keys.

Ionosphere: ``mask_apply_ionosphere=False`` (the default) leaves the
``estimate_ionosphere`` call arguments exactly unchanged; ``True`` injects
the mask plane through the existing ``valid_mask`` seam.

All tests are offline: fakes and synthetic fixtures only, no network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from affine import Affine
from shapely.geometry import box

import faninsar.processing.geometry.prepare_production as prepare_production_module
import faninsar.processing.masking.mask_manager as mask_manager_module
import faninsar.processing.masking.radar_projection as radar_projection_module
import faninsar.processing.stack.stack_api as stack_api_module
from faninsar import Pairs
from faninsar.processing.interferometry.phase_filter import (
    PhaseFilter,
    PhaseFilterResult,
)
from faninsar.processing.masking.mask import RasterMask
from faninsar.processing.masking.mask_manager import MaskManager, WaterLayer
from faninsar.processing.masking.radar_projection import (
    project_mask_to_radar,
    radar_projection_cache_key,
)
from faninsar.processing.pipeline.geo_lut import Geo2RdrLUT
from faninsar.processing.pipeline.products import (
    PairProductArrays,
    write_pair_stac_item,
)
from faninsar.processing.stack import Stack
from faninsar.processing.stack.config import StackConfig
from faninsar.processing.stack.ifg_store import (
    InterferogramArtifactStore,
    write_ifg_artifact,
)
from faninsar.processing.stack.scene_store import write_scene_unit
from faninsar.processing.unwrap.common import SpatialUnwrapper, SpatialUnwrapResult
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Callable

    from faninsar.processing.merge.grid import GeoGridSpec as GeoGridSpecType

from faninsar.processing.merge.grid import GeoGridSpec

# ---------------------------------------------------------------------------
# Shared synthetic geography (offline)
# ---------------------------------------------------------------------------

#: Geo IFG grid: 4 x 6 cells over lon [11.0, 11.018], lat [45.992, 46.0].
GEO_GRID = GeoGridSpec(
    crs="EPSG:4326",
    transform=(11.0, 0.003, 0.0, 46.0, 0.0, -0.002),
    width=6,
    height=4,
    resolution_m=(250.0, 222.0),
)
#: Coarser mask grid (2 x 3) covering the exact same bounds: one mask cell
#: spans 2 x 2 IFG cells, so the nearest-neighbour expansion is observable.
MASK_TRANSFORM = Affine(0.006, 0.0, 11.0, 0.0, -0.004, 46.0)
GRID_IDENTITY = hashlib.sha256(b"geo-grid").hexdigest()
DATES = ("20240101", "20240113")


def _water_raster_path(path: Path, *, water_column: int = 0) -> Path:
    """Write a synthetic uint8 mask GeoTIFF (1 = water, left band)."""
    import rasterio

    plane = np.zeros((2, 3), dtype=np.uint8)
    plane[:, water_column] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=MASK_TRANSFORM,
    ) as dataset:
        dataset.write(plane, 1)
    return path


def _expected_removed_plane() -> np.ndarray:
    """Nearest-neighbour expansion of the water column onto the geo grid.

    The mask water column 0 expands to IFG columns 0 and 1 on every row.
    """
    plane = np.zeros((4, 6), dtype=np.uint8)
    plane[:, :2] = 1
    return plane


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _KeepAllSampler:
    """Duck-typed MaskSampler that keeps everything."""

    def sample(self, latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
        """Return an all-keep plane."""
        del longitude_deg
        return np.ones(np.shape(latitude_deg), dtype=bool)


class _RecordingUnwrapper(SpatialUnwrapper):
    """Identity strategy recording the ``valid_mask`` it was handed."""

    def __init__(self) -> None:
        self.seen_valid_masks: list[np.ndarray] = []

    def unwrap(
        self,
        wrapped_phase: torch.Tensor,
        *,
        coherence: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> SpatialUnwrapResult:
        """Record the caller mask and return the supported input."""
        del coherence
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            self.seen_valid_masks.append(valid_mask.detach().cpu().numpy().copy())
            support = support & valid_mask
        labels = torch.where(
            support,
            torch.zeros_like(wrapped_phase, dtype=torch.int64),
            torch.full_like(wrapped_phase, -1, dtype=torch.int64),
        )
        return SpatialUnwrapResult(
            phase=torch.where(support, wrapped_phase, torch.nan),
            valid_mask=support,
            component_labels=labels,
            reference_values=torch.zeros(1, device=wrapped_phase.device),
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=0.0,
            failure_reason=None,
        )


class _RecordingFilter(PhaseFilter):
    """Identity filter recording exactly what the filter path received."""

    def __init__(self) -> None:
        self.seen_interferograms: list[np.ndarray] = []
        self.seen_supports: list[np.ndarray] = []

    def apply(
        self,
        interferogram: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> PhaseFilterResult:
        """Record the unmasked input and pass it through unchanged."""
        self.seen_interferograms.append(interferogram.detach().cpu().numpy().copy())
        support = torch.isfinite(interferogram.real) & torch.isfinite(
            interferogram.imag
        )
        if valid_mask is not None:
            self.seen_supports.append(valid_mask.detach().cpu().numpy().copy())
            support = support & valid_mask
        return PhaseFilterResult(
            interferogram=interferogram.clone(), valid_mask=support
        )


class _Watchdog:
    """Records watchdog sample labels (chunked-stage pattern)."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def sample(self, label: str) -> None:
        """Record one sample."""
        self.labels.append(label)


def _offline_stack(tmp_path: Path, **kwargs: Any) -> Stack:
    """Build a real Stack over empty SAFE directories (offline)."""
    paths = []
    for date in DATES:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date}T000000.SAFE"
        path.mkdir(exist_ok=True)
        paths.append(path)
    return Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
        **kwargs,
    )


def _write_geo_scene_units(stack: Stack, grid_shape: tuple[int, int]) -> None:
    """Persist geo-domain scene units for both dates on the shared grid."""
    aligned = np.exp(1j * 0.2 * np.ones(grid_shape)).astype(np.complex64)
    reference = np.ones(grid_shape, dtype=np.complex64)
    for date_id in DATES:
        root = stack.config.work_dir / "coreg" / date_id / "scenes"
        write_scene_unit(
            root,
            date_id=date_id,
            reference_id=DATES[0],
            domain="geo",
            tag="f0_IW1_b0",
            primary=reference,
            secondary=aligned,
            row_origin=0,
            col_origin=0,
            grid_shape=grid_shape,
            grid_identity=GRID_IDENTITY,
        )
        stack.coreg_paths[date_id] = root.parent


# ---------------------------------------------------------------------------
# Item 19 — IFG valid_mask intersection
# ---------------------------------------------------------------------------


class TestMaskRemovedPlane:
    """Nearest-neighbour rasterization of a mask onto an IFG geo grid."""

    def test_resamples_nearest_when_grids_differ(self, tmp_path: Path) -> None:
        """A coarser mask raster follows the finer IFG grid nearest-only."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        plane = _mask_plane_on_geo_grid(raster)
        np.testing.assert_array_equal(plane, _expected_removed_plane())

    def test_identical_grid_maps_onto_itself(self, tmp_path: Path) -> None:
        """An exactly matching target grid reproduces the source plane."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        mask = RasterMask(raster)
        from faninsar.processing.stack.session import _mask_removed_plane

        plane = _mask_removed_plane(mask, transform=MASK_TRANSFORM, shape=(2, 3))
        np.testing.assert_array_equal(
            plane, np.array([[1, 0, 0], [1, 0, 0]], dtype=np.uint8)
        )

    def test_sampler_path_samples_cell_centers(self) -> None:
        """Generic samplers are sampled at the target cell centres."""
        from faninsar.processing.stack.session import _mask_removed_plane

        plane = _mask_removed_plane(
            _KeepAllSampler(), transform=MASK_TRANSFORM, shape=(2, 3)
        )
        np.testing.assert_array_equal(plane, np.zeros((2, 3), dtype=np.uint8))


def _mask_plane_on_geo_grid(raster: Path) -> np.ndarray:
    """Sample a mask raster onto :data:`GEO_GRID` via the session helper."""
    from faninsar.processing.stack.session import _mask_removed_plane

    return _mask_removed_plane(
        RasterMask(raster),
        transform=_geo_grid_affine(),
        shape=GEO_GRID.shape,
    )


def _geo_grid_affine() -> Affine:
    """Return the affine of :data:`GEO_GRID` (GDAL-order conversion)."""
    gdal_x0, gdal_dx, gdal_rx, gdal_y0, gdal_ry, gdal_dy = GEO_GRID.transform
    return Affine(gdal_dx, gdal_rx, gdal_x0, gdal_ry, gdal_dy, gdal_y0)


class TestApplyMaskToValidMask:
    """``valid_mask &= ~mask`` semantics of the support intersection."""

    def test_intersection_removes_only_masked_cells(self, tmp_path: Path) -> None:
        """Water cells lose support; every other cell keeps it."""
        from faninsar.processing.stack.session import _apply_mask_to_valid_mask

        valid = np.ones((4, 6), dtype=bool)
        plane = _mask_plane_on_geo_grid(_water_raster_path(tmp_path / "m.tif"))
        result = _apply_mask_to_valid_mask(valid, plane)
        assert result is not None
        assert not result[:, :2].any()
        assert result[:, 2:].all()

    def test_mask_absent_returns_valid_mask_unchanged(self) -> None:
        """A mask-absent run returns the input support untouched."""
        from faninsar.processing.stack.session import _apply_mask_to_valid_mask

        valid = np.ones((2, 2), dtype=bool)
        assert _apply_mask_to_valid_mask(valid, None) is valid

    def test_nodata_cells_never_remove_support(self) -> None:
        """Plane value 255 (no mask data) does not delete data."""
        from faninsar.processing.stack.session import _apply_mask_to_valid_mask

        valid = np.ones((1, 3), dtype=bool)
        plane = np.array([[1, 255, 0]], dtype=np.uint8)
        result = _apply_mask_to_valid_mask(valid, plane)
        np.testing.assert_array_equal(result, [[False, True, True]])

    def test_valid_mask_none_yields_inverted_mask(self) -> None:
        """Without formation support the inverted mask becomes the support."""
        from faninsar.processing.stack.session import _apply_mask_to_valid_mask

        plane = np.array([[1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(
            _apply_mask_to_valid_mask(None, plane), [[False, True]]
        )


class TestFormationWiring:
    """The intersection is wired into ``form_interferograms`` persistence."""

    def test_persisted_valid_mask_intersects_mask_and_filter_is_unmasked(
        self, tmp_path: Path
    ) -> None:
        """The filter path sees the unmasked array; the mask hits only support."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        stack = _offline_stack(
            tmp_path,
            coregistration_grid="geo",
            geo_grid=GEO_GRID,
            mask=RasterMask(raster),
        )
        _write_geo_scene_units(stack, GEO_GRID.shape)
        phase_filter = _RecordingFilter()

        stack.form_interferograms(multilook=(1, 1), phase_filter=phase_filter)

        pair_dir = stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113"
        store = InterferogramArtifactStore.open(pair_dir)
        artifact = store.read()
        assert artifact.valid_mask is not None
        expected = np.ones(GEO_GRID.shape, dtype=bool)
        expected[:, :2] = False
        np.testing.assert_array_equal(artifact.valid_mask, expected)

        # The mask is a support input: the PhaseFilter received the
        # unmasked array and the numerics are bitwise unchanged.
        assert len(phase_filter.seen_interferograms) == 1
        filtered = phase_filter.seen_interferograms[0]
        supported = artifact.valid_mask
        np.testing.assert_array_equal(
            filtered[supported], artifact.complex_ifg[supported]
        )
        np.testing.assert_array_equal(
            phase_filter.seen_supports[0], np.ones(GEO_GRID.shape, dtype=bool)
        )
        store.close()

    def test_mask_disabled_leaves_support_unchanged(self, tmp_path: Path) -> None:
        """``mask=None`` persists the formation support (legacy path)."""
        _water_raster_path(tmp_path / "mask.tif")
        stack = _offline_stack(
            tmp_path,
            coregistration_grid="geo",
            geo_grid=GEO_GRID,
            mask=None,
        )
        _write_geo_scene_units(stack, GEO_GRID.shape)

        stack.form_interferograms(multilook=(1, 1))

        pair_dir = stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113"
        store = InterferogramArtifactStore.open(pair_dir)
        artifact = store.read()
        assert artifact.valid_mask is not None
        assert artifact.valid_mask.all()
        store.close()

    def test_auto_water_mask_degrades_without_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default water mask degrades to mask-absent (warning policy)."""
        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)
        stack = _offline_stack(tmp_path)

        sampler, record = stack._product_mask_resolution()

        assert sampler is None
        assert record["mask"] == "absent"

    def test_auto_water_mask_error_policy_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``mask_on_failure='error'`` raises on an unresolvable manager."""
        from faninsar.processing.errors import InvalidProcessingStateError

        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)
        stack = _offline_stack(tmp_path, mask_on_failure="error")

        with pytest.raises(
            InvalidProcessingStateError, match="FANINSAR_MASK_CACHE_DIR"
        ):
            stack._product_mask_resolution()


# ---------------------------------------------------------------------------
# Item 19 — mask_resolution_m consumption (override-grid rasterization)
# ---------------------------------------------------------------------------


def _write_vector_layer(path: Path, bounds: tuple[float, float, float, float]) -> Path:
    """Write a synthetic cached water layer GeoJSON with one polygon."""
    import geopandas as gpd

    frame = gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    return path


def _install_fake_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layer_path: Path
) -> None:
    """Route ``get_mask_manager`` to an offline fake serving ``layer_path``."""

    def fake_get_mask_manager(
        *, source: str | None = None, on_failure: str = "warning"
    ) -> MaskManager:
        return MaskManager(
            cache_dir=tmp_path / "mask-cache",
            source=source if source is not None else "water",
            on_failure=on_failure,  # type: ignore[arg-type]
        )

    layer = WaterLayer(
        path=layer_path,
        identity="testidentity",
        source_version="test-version",
        band=(-10.0, -10.0, 10.0, 10.0),
        from_cache=False,
        feature_count=1,
    )

    def fake_fetch(self: MaskManager, bounds: object) -> WaterLayer:
        del self, bounds
        return layer

    monkeypatch.setattr(mask_manager_module, "get_mask_manager", fake_get_mask_manager)
    monkeypatch.setattr(MaskManager, "get_water_layer", fake_fetch)


class TestMaskResolutionOverrideGrid:
    """``mask_resolution_m`` builds the override mask grid (D2 consumption)."""

    def test_rasterizes_buffered_water_onto_override_grid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The automatic mask product lands under work_dir with provenance."""
        layer_path = _write_vector_layer(
            tmp_path / "layer.geojson", (11.0, 45.99, 11.005, 46.0)
        )
        _install_fake_manager(monkeypatch, tmp_path, layer_path)
        stack = _offline_stack(
            tmp_path,
            roi=BoundingBox(11.0, 45.99, 11.018, 46.0),
            geo_grid=GEO_GRID,
            mask_buffer_km=0.0,
            mask_resolution_m=200.0,
        )

        sampler, record = stack._product_mask_resolution()

        assert isinstance(sampler, RasterMask)
        product = Path(str(record["mask_asset"]))
        assert product == stack.config.work_dir / "mask" / "water_mask.tif"
        assert product.is_file()
        assert record["mask"] == "present"
        assert record["mask_product"] == "water"
        assert record["mask_provider"] == "gsw"
        assert record["mask_threshold"] == 50
        assert record["mask_buffer_km"] == pytest.approx(0.0)
        # The D1 ROI subtraction keeps the right (unmasked) half: the run
        # never hits the structural empty-effective-ROI guard.
        assert record["mask_resolution_m"] == pytest.approx(200.0)
        assert record["mask_identity"] == "testidentity"

        import rasterio

        with rasterio.open(product) as dataset:
            assert dataset.crs.to_epsg() == 4326
            tags = dataset.tags()
        assert tags["mask_product"] == "water"
        assert tags["mask_identity"] == "testidentity"

        # The product feeds the IFG seam: the water half is removed on the
        # override grid sampled onto the geo IFG grid.
        plane = stack._ifg_mask_plane(
            domain="geo", looks=(1, 1), expected_shape=GEO_GRID.shape
        )
        assert plane is not None
        assert plane[:, :2].all()
        assert not plane[:, 2:].any()


# ---------------------------------------------------------------------------
# Item 20 — unwrap caller-supplied mask
# ---------------------------------------------------------------------------


class TestUnwrapCallerSuppliedMask:
    """The mask is the PROPOSAL-0038 caller mask at the unwrap seam."""

    def _stack_with_geo_ifg(self, tmp_path: Path, raster: Path) -> Stack:
        """Stack with one persisted geo-domain IFG and the mask configured."""
        stack = _offline_stack(
            tmp_path,
            coregistration_grid="geo",
            geo_grid=GEO_GRID,
            mask=RasterMask(raster),
        )
        phase = np.full(GEO_GRID.shape, 0.2, dtype=np.float32)
        persisted = np.ones(GEO_GRID.shape, dtype=bool)
        persisted[0, 5] = False  # a persisted invalid cell
        write_ifg_artifact(
            stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113",
            pair=DATES,
            looks=(1, 1),
            domain="geo",
            grid_identity=GRID_IDENTITY,
            filter_name="none",
            filter_parameters={},
            source_manifest_digests={"scenes": "a" * 64},
            complex_ifg=np.exp(1j * phase).astype(np.complex64),
            coherence=np.ones_like(phase),
            wrapped_phase=phase,
            amplitude=np.ones_like(phase),
            valid_mask=persisted,
        )
        stack.ifg_dirs = [
            stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113"
        ]
        return stack

    def test_unwrap_hands_persisted_and_caller_mask(self, tmp_path: Path) -> None:
        """valid_mask = persisted IFG mask ∧ caller mask (support rule)."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        stack = self._stack_with_geo_ifg(tmp_path, raster)
        unwrapper = _RecordingUnwrapper()

        stack.unwrap(unwrapper)

        assert len(unwrapper.seen_valid_masks) == 1
        seen = unwrapper.seen_valid_masks[0]
        expected = np.ones(GEO_GRID.shape, dtype=bool)
        expected[:, :2] = False  # caller mask (nearest-expanded water)
        expected[0, 5] = False  # persisted IFG mask
        np.testing.assert_array_equal(seen, expected)
        assert stack.analysis_ready

    def test_mask_absent_leaves_persisted_mask_as_caller_mask(
        self, tmp_path: Path
    ) -> None:
        """Without an applicable mask the persisted mask alone is handed on."""
        _water_raster_path(tmp_path / "mask.tif")
        stack = self._stack_with_geo_ifg(tmp_path, tmp_path / "mask.tif")
        stack.config.mask = None
        unwrapper = _RecordingUnwrapper()

        stack.unwrap(unwrapper)

        assert len(unwrapper.seen_valid_masks) == 1
        expected = np.ones(GEO_GRID.shape, dtype=bool)
        expected[0, 5] = False
        np.testing.assert_array_equal(unwrapper.seen_valid_masks[0], expected)


# ---------------------------------------------------------------------------
# Item 22 — radar projection
# ---------------------------------------------------------------------------


def _synthetic_lut() -> Geo2RdrLUT:
    """Small dense LUT over a 2 x 3 geo grid onto a 4 x 7 radar frame."""
    az_full = np.array([[0.1, 2.9, 1.5], [3.4, 0.6, np.nan]])
    rg_full = np.array([[1.2, 4.8, 3.3], [0.4, 5.6, 2.9]])
    valid = np.array([[True, True, True], [True, True, False]])
    return Geo2RdrLUT(
        az_full=az_full,
        rg_full=rg_full,
        valid=valid,
        full_radar_shape=(4, 7),
        height_m=0.0,
    )


def _synthetic_mask() -> np.ndarray:
    """uint8 mask with water at geo cells (0, 0) and (1, 1)."""
    plane = np.zeros((2, 3), dtype=np.uint8)
    plane[0, 0] = 1
    plane[1, 1] = 1
    return plane


class TestRadarProjectionGeoMode:
    """Scatter + nearest fill through the dense Geo2RdrLUT."""

    def test_scatter_and_nearest_fill(self) -> None:
        """Water lands at rint(LUT) indices; holes take the nearest value."""
        plane = project_mask_to_radar(
            _synthetic_mask(), lut=_synthetic_lut(), full_radar_shape=(4, 7)
        )
        # Direct scatters from the mask water cells (0, 0) and (1, 1):
        # (0, 0) -> az 0.1/rg 1.2 -> radar (0, 1); (1, 1) -> az 0.6/rg 5.6
        # -> radar (1, 6).
        assert plane[0, 1]
        assert plane[1, 6]
        # Land scatters stay land: (3, 5), (2, 3), and (3, 0).
        assert not plane[3, 5]
        assert not plane[2, 3]
        assert not plane[3, 0]
        # Nearest fill: uncovered pixels inherit the nearest covered value
        # (the water at (0, 1) is the nearest covered pixel of (0, 0)).
        assert plane[0, 0]
        # The invalid LUT cell (1, 2) contributes nothing.
        assert plane.shape == (4, 7)

    def test_scatter_wins_with_any_reduction(self) -> None:
        """Several geo cells mapping to one radar pixel: water wins."""
        lut = Geo2RdrLUT(
            az_full=np.array([[1.0, 1.2]]),
            rg_full=np.array([[2.0, 2.3]]),
            valid=np.array([[True, True]]),
            full_radar_shape=(3, 4),
            height_m=0.0,
        )
        mask = np.array([[0, 1]], dtype=np.uint8)
        plane = project_mask_to_radar(mask, lut=lut, full_radar_shape=(3, 4))
        assert plane[1, 2]

    def test_cache_hit_avoids_recomputation(self, tmp_path: Path) -> None:
        """The second call with the same identity loads the cached plane."""
        key = hashlib.sha256(
            radar_projection_cache_key(
                vector_digest="layer",
                buffer_km=1.0,
                resolution_m=30.0,
                dem_identity="dem",
            ).encode()
        ).hexdigest()
        calls = {"count": 0}
        original = radar_projection_module._scatter_lut_mask

        def counting(
            water_geo: np.ndarray, lut: Geo2RdrLUT, shape: tuple[int, int]
        ) -> np.ndarray:
            calls["count"] += 1
            return original(water_geo, lut, shape)

        lut = _synthetic_lut()
        first = project_mask_to_radar(
            _synthetic_mask(),
            lut=lut,
            full_radar_shape=(4, 7),
            cache_key=key,
            cache_dir=tmp_path,
        )
        radar_projection_module._scatter_lut_mask = counting  # type: ignore[assignment]
        try:
            second = project_mask_to_radar(
                _synthetic_mask(),
                lut=lut,
                full_radar_shape=(4, 7),
                cache_key=key,
                cache_dir=tmp_path,
            )
        finally:
            radar_projection_module._scatter_lut_mask = original  # type: ignore[assignment]
        assert calls["count"] == 0  # cache hit: no scatter at all
        np.testing.assert_array_equal(first, second)
        assert (tmp_path / f"{key}.npy").is_file()

    def test_changed_identity_recomputes(self, tmp_path: Path) -> None:
        """A different cache identity projects a fresh plane."""
        first_key = hashlib.sha256(b"identity-1").hexdigest()
        second_key = hashlib.sha256(b"identity-2").hexdigest()
        lut = _synthetic_lut()
        project_mask_to_radar(
            _synthetic_mask(),
            lut=lut,
            full_radar_shape=(4, 6),
            cache_key=first_key,
            cache_dir=tmp_path,
        )
        plane = project_mask_to_radar(
            _synthetic_mask(),
            lut=lut,
            full_radar_shape=(4, 7),
            cache_key=second_key,
            cache_dir=tmp_path,
        )
        assert plane[0, 1]
        assert (tmp_path / f"{second_key}.npy").is_file()

    def test_multilook_any_reduction(self, tmp_path: Path) -> None:
        """A look is removed when any contributing full-res pixel is water."""
        key = hashlib.sha256(b"multilook").hexdigest()
        full = project_mask_to_radar(
            _synthetic_mask(),
            lut=_synthetic_lut(),
            full_radar_shape=(4, 7),
            cache_key=key,
            cache_dir=tmp_path,
        )
        looked = project_mask_to_radar(
            _synthetic_mask(),
            lut=_synthetic_lut(),
            full_radar_shape=(4, 7),
            cache_key=key,
            cache_dir=tmp_path,
            multilook=(2, 2),
        )
        assert looked.shape == (2, 4)
        for row in range(2):
            for col in range(4):
                assert (
                    looked[row, col]
                    == full[2 * row : 2 * row + 2, 2 * col : 2 * col + 2].any()
                )


class TestRadarProjectionRadarMode:
    """Chunked run_geo2rdr projection with the memmap + watchdog pattern."""

    def test_chunked_projection_with_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mask rows run in chunks; converged indices scatter nearest."""
        chunks: list[int] = []

        def fake_geo2rdr(
            model: object,
            latitude_deg: np.ndarray,
            longitude_deg: np.ndarray,
            height_m: np.ndarray,
            *,
            device: str,
            **kwargs: object,
        ) -> Any:
            del model, longitude_deg, height_m, device, kwargs
            chunks.append(latitude_deg.shape[0])
            result = type("Result", (), {})()
            result.azimuth_index = np.full(latitude_deg.shape, 1.0)
            result.range_index = np.full(latitude_deg.shape, 2.0)
            result.converged = np.ones(latitude_deg.shape, dtype=bool)
            return result

        monkeypatch.setattr(prepare_production_module, "run_geo2rdr", fake_geo2rdr)
        watchdog = _Watchdog()

        plane = project_mask_to_radar(
            _synthetic_mask(),
            geometry=object(),
            mask_transform=MASK_TRANSFORM,
            full_radar_shape=(4, 6),
            chunk_size=1,
            watchdog=watchdog,
        )

        assert chunks == [1, 1]  # one tile per mask grid row
        assert len(watchdog.labels) == 2
        assert watchdog.labels[0].startswith("mask_radar_projection:0:")
        assert plane[1, 2]
        assert int(plane.sum()) == 1

    def test_out_of_bounds_and_nonconverged_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-converged and outside-frame indices never remove pixels."""

        def fake_geo2rdr(
            model: object,
            latitude_deg: np.ndarray,
            longitude_deg: np.ndarray,
            height_m: np.ndarray,
            *,
            device: str,
            **kwargs: object,
        ) -> Any:
            del model, longitude_deg, height_m, device, kwargs
            result = type("Result", (), {})()
            azimuth = np.full(latitude_deg.shape, -1.0)
            range_index = np.full(latitude_deg.shape, np.nan)
            converged = np.zeros(latitude_deg.shape, dtype=bool)
            converged[0, 0] = True
            range_index[0, 0] = 2.0
            result.azimuth_index = azimuth
            result.range_index = range_index
            result.converged = converged
            return result

        monkeypatch.setattr(prepare_production_module, "run_geo2rdr", fake_geo2rdr)

        plane = project_mask_to_radar(
            _synthetic_mask(),
            geometry=object(),
            mask_transform=MASK_TRANSFORM,
            full_radar_shape=(4, 6),
        )

        assert not plane.any()

    def test_cache_roundtrip_counts_chunks_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The radar-mode plane caches; a second call runs zero chunks."""
        calls = {"count": 0}

        def fake_geo2rdr(
            model: object,
            latitude_deg: np.ndarray,
            longitude_deg: np.ndarray,
            height_m: np.ndarray,
            *,
            device: str,
            **kwargs: object,
        ) -> Any:
            del model, longitude_deg, height_m, device, kwargs
            calls["count"] += 1
            result = type("Result", (), {})()
            result.azimuth_index = np.full(latitude_deg.shape, 1.0)
            result.range_index = np.full(latitude_deg.shape, 2.0)
            result.converged = np.ones(latitude_deg.shape, dtype=bool)
            return result

        monkeypatch.setattr(prepare_production_module, "run_geo2rdr", fake_geo2rdr)
        key = hashlib.sha256(
            radar_projection_cache_key(
                vector_digest="layer",
                buffer_km=1.0,
                resolution_m=None,
                dem_identity="dem",
            ).encode()
        ).hexdigest()

        first = project_mask_to_radar(
            _synthetic_mask(),
            geometry=object(),
            mask_transform=MASK_TRANSFORM,
            full_radar_shape=(4, 6),
            chunk_size=1,
            cache_key=key,
            cache_dir=tmp_path,
        )
        second = project_mask_to_radar(
            _synthetic_mask(),
            geometry=object(),
            mask_transform=MASK_TRANSFORM,
            full_radar_shape=(4, 6),
            chunk_size=1,
            cache_key=key,
            cache_dir=tmp_path,
        )

        assert calls["count"] == 2  # two mask grid rows, once
        np.testing.assert_array_equal(first, second)

    def test_requires_exactly_one_mode(self) -> None:
        """Both or neither projection source fails closed."""
        with pytest.raises(ValueError, match="exactly one"):
            project_mask_to_radar(_synthetic_mask(), full_radar_shape=(4, 6))
        with pytest.raises(ValueError, match="exactly one"):
            project_mask_to_radar(
                _synthetic_mask(),
                lut=_synthetic_lut(),
                geometry=object(),
                full_radar_shape=(4, 6),
            )


# ---------------------------------------------------------------------------
# Item 23 — STAC mask asset
# ---------------------------------------------------------------------------


def _pair_product(metadata: dict[str, Any]) -> PairProductArrays:
    """Build a minimal pair product for the STAC writer."""
    phase = np.full((4, 6), 0.2, dtype=np.float32)
    return PairProductArrays(
        pair_id="20240101_20240113",
        complex_ifg=np.exp(1j * phase).astype(np.complex64),
        coherence=np.ones_like(phase),
        wrapped_phase=phase,
        unwrapped_phase=phase.copy(),
        connected_components=np.zeros((4, 6), dtype=np.int32),
        metadata=metadata,
    )


class TestStacMaskAsset:
    """The buffered mask registers as a STAC asset with provenance."""

    def test_mask_asset_written_with_provenance(self, tmp_path: Path) -> None:
        """The item carries the mask href and faninsar:mask_* properties."""
        mask_path = _water_raster_path(tmp_path / "mask" / "water_mask.tif")
        product = _pair_product(
            {
                "unwrap_method": "stack_irls",
                "mask_href": str(mask_path),
                "mask_product": "water",
                "mask_provider": "gsw",
                "mask_retrieved": "2026-08-30",
                "mask_threshold": 50,
                "mask_buffer_km": 1.0,
            }
        )
        zarr_path = tmp_path / "pair.zarr"
        item_path = tmp_path / "pair.json"

        written = write_pair_stac_item(product, zarr_path, item_path)

        assert written == item_path
        item = json.loads(item_path.read_text(encoding="utf-8"))
        mask_asset = item["assets"]["mask"]
        assert mask_asset["href"] == str(mask_path)
        assert "mask" in mask_asset["roles"]
        assert "geotiff" in mask_asset["type"]
        properties = item["properties"]
        assert properties["faninsar:mask_product"] == "water"
        assert properties["faninsar:mask_provider"] == "gsw"
        assert properties["faninsar:mask_retrieved"] == "2026-08-30"
        assert properties["faninsar:mask_threshold"] == 50
        assert properties["faninsar:mask_buffer_km"] == 1.0
        assert item["assets"]["zarr"]["href"] == str(zarr_path)

    def test_without_mask_metadata_no_mask_asset(self, tmp_path: Path) -> None:
        """Unmasked products keep the original single-asset item."""
        product = _pair_product({"unwrap_method": "snaphu"})
        item_path = tmp_path / "pair.json"

        write_pair_stac_item(product, tmp_path / "pair.zarr", item_path)

        item = json.loads(item_path.read_text(encoding="utf-8"))
        assert set(item["assets"]) == {"zarr"}


# ---------------------------------------------------------------------------
# Ionosphere opt-in
# ---------------------------------------------------------------------------


class TestIonosphereOptIn:
    """``mask_apply_ionosphere`` consumes the existing valid_mask seam."""

    def _recording_api(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def fake_estimate_ionosphere(stack: Stack, **kwargs: Any) -> list[Any]:
            del stack
            calls.append(dict(kwargs))
            return []

        monkeypatch.setattr(
            stack_api_module, "estimate_ionosphere", fake_estimate_ionosphere
        )
        return calls

    def test_default_false_leaves_call_arguments_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default does not inject a valid_mask (and resolves nothing)."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        stack = _offline_stack(
            tmp_path,
            geo_grid=GEO_GRID,
            mask=RasterMask(raster),
        )
        calls = self._recording_api(monkeypatch)

        stack.estimate_ionosphere(range_sampling_rate_hz=6e7)

        assert len(calls) == 1
        assert "valid_mask" not in calls[0]

    def test_true_injects_the_mask_plane(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opting in feeds the geo-grid mask plane through the seam."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        stack = _offline_stack(
            tmp_path,
            geo_grid=GEO_GRID,
            mask=RasterMask(raster),
            mask_apply_ionosphere=True,
        )
        calls = self._recording_api(monkeypatch)

        stack.estimate_ionosphere(range_sampling_rate_hz=6e7)

        assert len(calls) == 1
        plane = calls[0]["valid_mask"]
        np.testing.assert_array_equal(plane, _expected_removed_plane())

    def test_caller_supplied_valid_mask_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit caller valid_mask is passed through untouched."""
        raster = _water_raster_path(tmp_path / "mask.tif")
        stack = _offline_stack(
            tmp_path,
            geo_grid=GEO_GRID,
            mask=RasterMask(raster),
            mask_apply_ionosphere=True,
        )
        calls = self._recording_api(monkeypatch)
        caller_mask = np.zeros((3, 3), dtype=bool)

        stack.estimate_ionosphere(range_sampling_rate_hz=6e7, valid_mask=caller_mask)

        assert len(calls) == 1
        assert calls[0]["valid_mask"] is caller_mask

    def test_mask_disabled_never_touches_the_seam(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``mask=None`` with opt-in still leaves the arguments unchanged."""
        stack = _offline_stack(
            tmp_path,
            geo_grid=GEO_GRID,
            mask=None,
            mask_apply_ionosphere=True,
        )
        calls = self._recording_api(monkeypatch)

        stack.estimate_ionosphere(range_sampling_rate_hz=6e7)

        assert len(calls) == 1
        assert "valid_mask" not in calls[0]


# ---------------------------------------------------------------------------
# Config guard
# ---------------------------------------------------------------------------


def test_stack_config_mask_surface_unchanged(tmp_path: Path) -> None:
    """The D2 work builds on the D1 mask surface verbatim."""
    config = StackConfig(work_dir=tmp_path, activation_mode="reference")
    assert config.mask == "water"
    assert config.mask_apply_ionosphere is False
    assert config.mask_resolution_m is None
