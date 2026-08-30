"""Slice D1 Stack-integration tests for the mask configuration (PROPOSAL-0039).

Covers TDD-plan items 18 and 21 against ``stack/config.py``,
``_stack_config.py``, and ``stack/session.py``:

18. ``StackConfig`` mask fields (default automatic water mask, ``None``
    disables), validation, and the matching ``run(config)`` keys with
    ``"none" -> None`` normalization.
21. ROI arithmetic: ``effective_ROI = ROI - water_beyond_buffer`` behind the
    unit-testable :func:`faninsar.processing.stack.session.\
_resolve_effective_roi` helper; the empty-ROI structured
    ``InvalidProcessingStateError`` is independent of ``mask_on_failure``;
    the antimeridian seam guard rejects near-seam ROIs before any fetch;
    ``mask=None`` leaves the ROI untouched (legacy path); the mask-absent
    lineage record is produced when resolution fails under ``warning``.

All tests are offline: the mask manager is built against a temporary cache
directory and its ``get_water_layer`` is faked to serve a synthetic GeoJSON
vector layer. No real Stack runs, no network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

import faninsar.processing.masking.mask_manager as mask_manager_module
import faninsar.processing.stack as stack_package
from faninsar.processing.masking.mask_manager import (
    MaskManager,
    MaskProviderUnavailableError,
    WaterLayer,
)
from faninsar.processing.stack.config import (
    AUTO_WATER_MASK,
    MASK_DISABLED,
    StackConfig,
)
from faninsar.processing.stack.session import Stack, _resolve_effective_roi
from faninsar.query import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Callable

#: ROI for the subtraction tests: a 2 deg x 1 deg box at the equator.
ROI = BoundingBox(0.0, 0.0, 2.0, 1.0)
#: Water polygon covering the left half of the ROI.
WATER_LEFT = (0.0, 0.0, 1.0, 1.0)
#: Guard message proving the water fetch is never reached.
FETCH_MUST_NOT_RUN = "water layer fetched despite the seam guard"


# ---------------------------------------------------------------------------
# Fakes (offline; no network)
# ---------------------------------------------------------------------------


class _SamplerStub:
    """Minimal duck-typed :class:`~faninsar.processing.masking.mask.MaskSampler`."""

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return an all-keep plane (never called in these tests)."""
        del longitude_deg
        return np.ones(np.shape(latitude_deg), dtype=bool)


def _write_vector_layer(path: Path, bounds: tuple[float, float, float, float]) -> Path:
    """Write a synthetic cached water layer GeoJSON with one polygon."""
    frame = gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    return path


def _fake_layer(path: Path, source_version: str = "test-version") -> WaterLayer:
    """Build a :class:`WaterLayer` pointing at a synthetic GeoJSON file."""
    return WaterLayer(
        path=path,
        identity="testidentity",
        source_version=source_version,
        band=(-10.0, -10.0, 10.0, 10.0),
        from_cache=False,
        feature_count=1,
    )


def _install_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    layer_path: Path,
    on_failure: str = "warning",
    buffer_km: float = 1.0,
    get_water_layer: Callable[..., WaterLayer] | None = None,
) -> list[tuple[str | None, str]]:
    """Route session mask resolution to an offline fake manager.

    Returns the recorded ``(source, on_failure)`` factory calls. The fake
    ``get_water_layer`` serves ``layer_path`` (or the provided callable), so
    the real fetch/vectorize pipeline is never exercised. ``buffer_km``
    configures the manager-owned land buffer under test.
    """
    calls: list[tuple[str | None, str]] = []

    def fake_get_mask_manager(
        *, source: str | None = None, on_failure: str = on_failure
    ) -> MaskManager:
        calls.append((source, on_failure))
        return MaskManager(
            cache_dir=tmp_path / "mask-cache",
            source=source if source is not None else "water",
            on_failure=on_failure,  # type: ignore[arg-type]
            buffer_km=buffer_km,
        )

    if get_water_layer is None:
        layer = _fake_layer(layer_path)

        def fake_fetch(self: MaskManager, bounds: object) -> WaterLayer:
            del self, bounds
            return layer

        get_water_layer = fake_fetch
    monkeypatch.setattr(mask_manager_module, "get_mask_manager", fake_get_mask_manager)
    monkeypatch.setattr(MaskManager, "get_water_layer", get_water_layer)
    return calls


def _outage_error(message: str = "provider outage") -> MaskProviderUnavailableError:
    """Build the structured outage error for the failure-policy tests."""
    return MaskProviderUnavailableError(
        message,
        product="water",
        provider="gsw",
        host="example.invalid",
        failure_class="upstream-outage",
        attempts=3,
    )


# ---------------------------------------------------------------------------
# StackConfig mask fields (TDD-plan item 18)
# ---------------------------------------------------------------------------


class TestStackConfigMaskFields:
    """Defaults and validation of the StackConfig mask surface."""

    def test_defaults_are_automatic_water_mask(self, tmp_path: Path) -> None:
        """The Stack default is the automatic water mask (PROPOSAL-0039 G5)."""
        config = StackConfig(work_dir=tmp_path, activation_mode="reference")
        assert config.mask == AUTO_WATER_MASK == "water"
        assert config.mask_source is None
        assert config.mask_on_failure == "warning"
        assert config.mask_apply_ionosphere is False

    def test_dropped_buffer_and_resolution_fields_are_gone(self) -> None:
        """Buffer and resolution ownership moved to the masking manager."""
        names = {field.name for field in dataclasses.fields(StackConfig)}
        for dropped in (
            "mask_buffer_km",
            "ocean_water_buffer_km",
            "inland_water_buffer_km",
            "mask_resolution_m",
        ):
            assert dropped not in names

    @pytest.mark.parametrize("disabled", [MASK_DISABLED, None], ids=["str", "none"])
    def test_mask_disabled_normalizes_to_none(
        self, tmp_path: Path, disabled: str | None
    ) -> None:
        """``mask='none'`` and ``mask=None`` both restore unmasked processing."""
        config = StackConfig(
            work_dir=tmp_path,
            activation_mode="reference",
            mask=disabled,
        )
        assert config.mask is None

    def test_mask_water_sentinel_is_idempotent(self, tmp_path: Path) -> None:
        """``mask='water'`` keeps the automatic-water sentinel."""
        config = StackConfig(
            work_dir=tmp_path,
            activation_mode="reference",
            mask="water",
        )
        assert config.mask == AUTO_WATER_MASK

    def test_mask_accepts_sampler_object(self, tmp_path: Path) -> None:
        """A user-supplied MaskSampler object is kept as the general mask."""
        sampler = _SamplerStub()
        config = StackConfig(
            work_dir=tmp_path,
            activation_mode="reference",
            mask=sampler,
        )
        assert config.mask is sampler

    @pytest.mark.parametrize("selection", ["raster", "auto", "WATER", ""])
    def test_mask_rejects_unknown_strings(self, tmp_path: Path, selection: str) -> None:
        """Unknown string selections fail closed at configuration time."""
        with pytest.raises(ValueError, match="mask"):
            StackConfig(
                work_dir=tmp_path,
                activation_mode="reference",
                mask=selection,
            )

    def test_mask_rejects_non_sampler_object(self, tmp_path: Path) -> None:
        """Objects without the MaskSampler shape fail closed."""
        with pytest.raises(ValueError, match="MaskSampler"):
            StackConfig(
                work_dir=tmp_path,
                activation_mode="reference",
                mask=object(),
            )

    def test_mask_on_failure_must_be_a_policy(self, tmp_path: Path) -> None:
        """The failure policy is validated against the three-value enum."""
        with pytest.raises(ValueError, match="mask_on_failure"):
            StackConfig(
                work_dir=tmp_path,
                activation_mode="reference",
                mask_on_failure="ignore",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("policy", ["error", "warning", "skip"])
    def test_mask_on_failure_accepts_policies(
        self, tmp_path: Path, policy: str
    ) -> None:
        """Every documented policy is accepted verbatim."""
        config = StackConfig(
            work_dir=tmp_path,
            activation_mode="reference",
            mask_on_failure=policy,  # type: ignore[arg-type]
        )
        assert config.mask_on_failure == policy


# ---------------------------------------------------------------------------
# run(config) keys (TDD-plan item 18)
# ---------------------------------------------------------------------------


class _RecordingStack:
    """Offline stand-in for the Stack execution front door."""

    captured: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Record the construction kwargs for assertions."""
        type(self).captured = dict(kwargs)

    @classmethod
    def from_safes(cls, paths: object, **kwargs: Any) -> _RecordingStack:
        """Record the from_safes kwargs exactly as ``run()`` forwards them."""
        return cls(paths=paths, **kwargs)

    def prepare_scenes(self) -> _RecordingStack:
        """Chain to the next stage."""
        return self

    def coregister_scenes(self) -> _RecordingStack:
        """Chain to the next stage."""
        return self

    def form_interferograms(self, *, overwrite: bool) -> _RecordingStack:
        """Record the overwrite flag and finish the default workflow."""
        type(self).captured["overwrite"] = overwrite
        return self


def _run_config(**extra: Any) -> dict[str, Any]:
    """Build a minimal valid ``run()`` mapping with mask keys."""
    config: dict[str, Any] = {
        "paths": ["/data/20200101.SAFE", "/data/20200113.SAFE"],
        "output": "/data/out",
    }
    config.update(extra)
    return config


class TestRunConfigMaskKeys:
    """``run(config)`` forwards and normalizes the mask keys."""

    def test_mask_keys_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every documented mask key reaches Stack construction verbatim."""
        monkeypatch.setattr(stack_package, "Stack", _RecordingStack)
        from faninsar._stack_config import run

        run(
            _run_config(
                mask="water",
                mask_source="water:worldcover",
                mask_on_failure="error",
                mask_apply_ionosphere=True,
            )
        )
        captured = _RecordingStack.captured
        assert captured["mask"] == "water"
        assert captured["mask_source"] == "water:worldcover"
        assert captured["mask_on_failure"] == "error"
        assert captured["mask_apply_ionosphere"] is True

    def test_dropped_buffer_and_resolution_keys_are_not_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Removed buffer/resolution keys no longer reach Stack construction."""
        monkeypatch.setattr(stack_package, "Stack", _RecordingStack)
        from faninsar._stack_config import run

        run(
            _run_config(
                mask_buffer_km=2.0,
                mask_resolution_m=30.0,
                inland_water_buffer_km=1.0,
            )
        )
        captured = _RecordingStack.captured
        for dropped in (
            "mask_buffer_km",
            "mask_resolution_m",
            "ocean_water_buffer_km",
            "inland_water_buffer_km",
        ):
            assert dropped not in captured

    def test_mask_none_normalizes_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``mask: none`` config spelling reaches the Stack as ``None``."""
        monkeypatch.setattr(stack_package, "Stack", _RecordingStack)
        from faninsar._stack_config import run

        run(_run_config(mask="none"))
        assert _RecordingStack.captured["mask"] is None

    def test_mask_absent_keeps_stack_auto_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absent ``mask`` key is not forwarded; the Stack default applies."""
        monkeypatch.setattr(stack_package, "Stack", _RecordingStack)
        from faninsar._stack_config import run

        run(_run_config())
        assert "mask" not in _RecordingStack.captured


# ---------------------------------------------------------------------------
# Effective-ROI arithmetic (TDD-plan item 21)
# ---------------------------------------------------------------------------


class TestResolveEffectiveRoi:
    """``_resolve_effective_roi``: ROI - buffered water, guards, lineage."""

    def test_subtracts_buffered_water_from_the_roi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Land beyond the buffered water is kept; water is removed."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path, buffer_km=0.1)

        result = _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure="warning",
        )

        assert result.roi is not ROI  # the ROI is reshaped, not aliased
        assert result.geometry is not None
        left, bottom, right, top = result.geometry.bounds
        # 0.1 km of land buffer is kept around the water edge.
        assert left == pytest.approx(1.0 + 0.1 / 111.32, abs=2e-3)
        assert bottom == pytest.approx(0.0, abs=2e-3)
        assert right == pytest.approx(2.0, abs=1e-9)
        assert top == pytest.approx(1.0, abs=1e-9)
        assert result.geometry.contains(Point(1.5, 0.5))
        assert not result.geometry.contains(Point(0.5, 0.5))

    def test_effective_roi_is_a_polygons_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The effective ROI feeds the burst-selection path as Polygons."""
        from faninsar.query import Polygons

        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path, buffer_km=0.1)

        result = _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure="warning",
        )

        assert isinstance(result.roi, Polygons)
        assert str(result.roi.crs) == "EPSG:4326"

    @pytest.mark.parametrize("policy", ["error", "warning", "skip"])
    def test_empty_effective_roi_always_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
    ) -> None:
        """An all-water ROI fails closed for every failure policy."""
        from faninsar.processing.errors import InvalidProcessingStateError

        layer_path = _write_vector_layer(
            tmp_path / "layer.geojson", (0.0, 0.0, 2.0, 1.0)
        )
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path, buffer_km=0.1)

        with pytest.raises(InvalidProcessingStateError, match="empty"):
            _resolve_effective_roi(
                ROI,
                mask=AUTO_WATER_MASK,
                mask_source=None,
                mask_on_failure=policy,  # type: ignore[arg-type]
            )

    def test_seam_guard_rejects_before_any_fetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A near-seam ROI raises structurally before the water fetch."""
        from faninsar.processing.errors import InvalidProcessingStateError

        def boom(self: MaskManager, bounds: object) -> WaterLayer:
            del self, bounds
            raise AssertionError(FETCH_MUST_NOT_RUN)

        _install_manager(
            monkeypatch,
            tmp_path,
            layer_path=Path("/unused"),
            get_water_layer=boom,
        )

        with pytest.raises(InvalidProcessingStateError, match="seam guard"):
            _resolve_effective_roi(
                BoundingBox(170.0, 8.0, 179.5, 10.0),
                mask=AUTO_WATER_MASK,
                mask_source=None,
                mask_on_failure="warning",
            )

    def test_mask_none_leaves_the_roi_untouched(self) -> None:
        """``mask=None`` is the bitwise-identical legacy path."""
        result = _resolve_effective_roi(
            ROI,
            mask=None,
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.roi is ROI
        assert result.geometry is None
        assert result.lineage == {}

    def test_mask_none_skips_the_seam_guard(self) -> None:
        """Unmasked processing never triggers the structural seam guard."""
        result = _resolve_effective_roi(
            BoundingBox(170.0, 8.0, 179.5, 10.0),
            mask=None,
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.roi is not None
        assert result.lineage == {}

    def test_missing_roi_defers_to_product_level_masking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an ROI the mask still applies at the product level (D2)."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        calls = _install_manager(monkeypatch, tmp_path, layer_path=layer_path)

        result = _resolve_effective_roi(
            None,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.roi is None
        assert result.lineage == {}
        assert calls == []  # no manager resolution without an ROI in D1

    @pytest.mark.parametrize("policy", ["warning", "skip"])
    def test_provider_failure_records_mask_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
    ) -> None:
        """A failed fetch under warning/skip records the mask-absent state."""

        def outage(self: MaskManager, bounds: object) -> WaterLayer:
            del self, bounds
            raise _outage_error()

        _install_manager(
            monkeypatch, tmp_path, layer_path=Path("/unused"), get_water_layer=outage
        )

        result = _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure=policy,  # type: ignore[arg-type]
        )
        assert result.roi is ROI  # degraded to the unmasked ROI
        assert result.geometry is None
        assert result.lineage["mask"] == "absent"
        assert "provider outage" in str(result.lineage["reason"])

    def test_provider_failure_under_error_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``error`` policy propagates the structured outage."""

        def outage(self: MaskManager, bounds: object) -> WaterLayer:
            del self, bounds
            raise _outage_error()

        _install_manager(
            monkeypatch,
            tmp_path,
            layer_path=Path("/unused"),
            get_water_layer=outage,
        )

        with pytest.raises(MaskProviderUnavailableError):
            _resolve_effective_roi(
                ROI,
                mask=AUTO_WATER_MASK,
                mask_source=None,
                mask_on_failure="error",
            )

    def test_missing_cache_dir_degrades_under_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable manager under warning records mask-absent."""
        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)

        result = _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.roi is ROI
        assert result.lineage["mask"] == "absent"
        assert "FANINSAR_MASK_CACHE_DIR" in str(result.lineage["reason"])

    def test_missing_cache_dir_raises_under_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``error`` policy fails closed on manager misconfiguration."""
        from faninsar.processing.errors import InvalidProcessingStateError

        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)

        with pytest.raises(InvalidProcessingStateError):
            _resolve_effective_roi(
                ROI,
                mask=AUTO_WATER_MASK,
                mask_source=None,
                mask_on_failure="error",
            )

    def test_present_lineage_records_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolved mask records identity, version, and buffer in lineage."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path, buffer_km=0.1)

        result = _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.lineage["mask"] == "present"
        assert result.lineage["mask_identity"] == "testidentity"
        assert result.lineage["mask_source_version"] == "test-version"
        assert result.lineage["buffer_km"] == pytest.approx(0.1)

    def test_user_sampler_defers_to_product_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-supplied MaskSampler does not reshape the ROI in D1."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        calls = _install_manager(monkeypatch, tmp_path, layer_path=layer_path)

        result = _resolve_effective_roi(
            ROI,
            mask=_SamplerStub(),
            mask_source=None,
            mask_on_failure="warning",
        )
        assert result.roi is ROI
        assert result.lineage == {}
        assert calls == []

    def test_manager_factory_receives_source_and_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manager is built with the configured source and policy."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        calls = _install_manager(monkeypatch, tmp_path, layer_path=layer_path)

        _resolve_effective_roi(
            ROI,
            mask=AUTO_WATER_MASK,
            mask_source="water:worldcover",
            mask_on_failure="error",
        )
        assert calls == [("water:worldcover", "error")]


# ---------------------------------------------------------------------------
# Session wiring (burst-selection seam and lineage record)
# ---------------------------------------------------------------------------


def _offline_stack(tmp_path: Path, roi: BoundingBox | None) -> Stack:
    """Build a real Stack over empty SAFE directories (offline)."""
    paths = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir(exist_ok=True)
        paths.append(path)
    stack = Stack.from_safes(
        paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    stack.config.roi = roi
    return stack


class TestSessionWiring:
    """The effective ROI feeds the burst-selection path; lineage is exposed."""

    def test_burst_kwargs_use_effective_roi(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_burst_kwargs`` passes the subtracted ROI to pair production."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path, buffer_km=0.1)
        stack = _offline_stack(tmp_path, BoundingBox(0.0, 0.0, 2.0, 1.0))

        kwargs = stack._burst_kwargs()

        assert not isinstance(kwargs["roi"], BoundingBox)
        assert kwargs["roi"].geometry.union_all().bounds[0] == pytest.approx(
            1.0 + 0.1 / 111.32, abs=2e-3
        )

    def test_burst_kwargs_untouched_when_resolution_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a resolvable mask the ROI is untouched (degraded run)."""
        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)
        stack = _offline_stack(tmp_path, ROI)

        kwargs = stack._burst_kwargs()

        assert kwargs["roi"] is ROI
        record = stack._mask_lineage_record()
        assert record["mask"] == "absent"
        assert "FANINSAR_MASK_CACHE_DIR" in str(record["reason"])

    def test_resolution_is_cached_per_stack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One Stack run resolves the mask once (no per-date refetch)."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        calls = _install_manager(monkeypatch, tmp_path, layer_path=layer_path)
        stack = _offline_stack(tmp_path, ROI)

        first = stack._burst_kwargs()
        second = stack._burst_kwargs()

        assert first["roi"] is second["roi"]
        assert len(calls) == 1

    def test_mask_lineage_record_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lineage record exposes the resolved-mask provenance."""
        layer_path = _write_vector_layer(tmp_path / "layer.geojson", WATER_LEFT)
        _install_manager(monkeypatch, tmp_path, layer_path=layer_path)
        stack = _offline_stack(tmp_path, ROI)

        record = stack._mask_lineage_record()

        assert record["mask"] == "present"
        assert record["mask_identity"] == "testidentity"

    def test_mask_lineage_record_is_json_serializable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record merges into the run-manifest JSON payload verbatim."""
        monkeypatch.delenv("FANINSAR_MASK_CACHE_DIR", raising=False)
        monkeypatch.delenv("FANINSAR_MASK_SOURCE", raising=False)
        stack = _offline_stack(tmp_path, ROI)

        payload = {"reference": stack.reference, **stack._mask_lineage_record()}

        assert json.loads(json.dumps(payload))["mask"] == "absent"
