"""Revision-2 burst footprint geometry (PROPOSAL-0084)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.scene_bursts import (
    actual_source_polygon,
    resolve_s1_scene_burst_footprints,
)
from faninsar.missions.s1.types import S1Product
from faninsar.network.readers.scene_bursts import SceneBurstObservation

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "sentinel1"
    / "minimal_annotation_iw1.xml"
)
BBOX = ((10.0, 20.0), (11.0, 20.0), (11.0, 21.0), (10.0, 21.0))
QUAD_A = np.array(
    [[1.0, 2.0], [1.5, 2.0], [1.5, 2.5], [1.0, 2.5]],
    dtype=np.float64,
)
QUAD_B = np.array(
    [[3.0, 4.0], [3.5, 4.0], [3.5, 4.5], [3.0, 4.5]],
    dtype=np.float64,
)


@dataclass
class _Catalog:
    """Minimal catalog exposing ``paths_for``."""

    paths: dict[str, tuple[Path, ...]]

    def paths_for(self, date_like: object) -> tuple[Path, ...]:
        """Return source paths for a date id."""
        return self.paths[str(date_like)]


@dataclass
class _Stack:
    """Minimal Stack for the S1 footprint resolver."""

    catalog: _Catalog
    reference: str = "20240101"
    _sensor_registry_key: str = "sentinel1"


def _swath(annotation_path: str = "mem://iw1.xml", *, name: str = "IW1") -> object:
    parsed = parse_annotation_xml(
        FIXTURE.read_text(encoding="utf-8"),
        annotation_path=annotation_path,
        measurement_path="mem://measurement.tiff",
    )
    if name == parsed.swath:
        return parsed
    return replace(parsed, swath=name)


def _product(path: Path, swath: object, *, absolute_orbit: int = 1) -> S1Product:
    return S1Product(
        product_id=path.stem,
        source_path=path,
        mission_id="S1A",
        product_type="SLC",
        mode="IW",
        absolute_orbit=absolute_orbit,
        polarizations=("VV",),
        swaths=(swath,),
    )


def _obs(date: str, burst: str, absolute: int | None = None) -> SceneBurstObservation:
    return SceneBurstObservation(
        date=date,
        burst=burst,
        track="88A",
        absolute_burst_id=absolute,
    )


def _rdr2geo_result(
    quads: Sequence[np.ndarray],
    *,
    converged: bool | Sequence[bool] = True,
) -> SimpleNamespace:
    """Build a stub rdr2geo result for one or more 4-corner groups."""
    lons: list[float] = []
    lats: list[float] = []
    flags: list[bool] = []
    if isinstance(converged, bool):
        conv_flags = [converged] * (4 * len(quads))
    else:
        conv_flags = [bool(flag) for flag in converged]
    for quad in quads:
        lons.extend(np.asarray(quad, dtype=np.float64)[:, 0].tolist())
        lats.extend(np.asarray(quad, dtype=np.float64)[:, 1].tolist())
    if len(conv_flags) < len(lons):
        conv_flags.extend([True] * (len(lons) - len(conv_flags)))
    flags = conv_flags[: len(lons)]
    return SimpleNamespace(
        longitude_deg=np.asarray(lons, dtype=np.float64),
        latitude_deg=np.asarray(lats, dtype=np.float64),
        converged=np.asarray(flags, dtype=bool),
    )


def _stub_rdr2geo(quads: Sequence[np.ndarray]) -> Callable[..., object]:
    """Return lon/lat by 4-corner group order, with all corners converged."""

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, slant, dem, device
        n = int(np.asarray(azimuth, dtype=np.float64).reshape(-1, 4).shape[0])
        chosen = [quads[index % len(quads)] for index in range(n)]
        return _rdr2geo_result(chosen)

    return run


def _prepare_source(tmp_path: Path, name: str, swath: object) -> tuple[Path, S1Product]:
    path = tmp_path / name
    path.mkdir()
    product = _product(path, replace(swath, annotation_path=str(path / "ann.xml")))
    return path, product


def test_secondary_observation_matches_across_parent_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspection binds the ANX-matching secondary, not the same frame slot."""
    from faninsar.missions.s1.scene_bursts import _bind_observation

    reference = _swath()
    reference = replace(
        reference, bursts=(replace(reference.bursts[0], azimuth_anx_time_s=100.0),)
    )
    wrong = replace(
        reference, bursts=(replace(reference.bursts[0], azimuth_anx_time_s=300.0),)
    )
    ref_product = _product(Path("tracks/88A/ref.SAFE"), reference)
    wrong_product = _product(Path("tracks/88A/wrong.SAFE"), wrong)
    matching_product = _product(Path("tracks/88A/matching.SAFE"), reference)
    monkeypatch.setattr(
        "faninsar.missions.s1.track_map.track_key_for_product", lambda _product: "88A"
    )
    stack = _Stack(_Catalog({}))
    cache = {"20240101": [ref_product], "20240113": [wrong_product, matching_product]}
    found = _bind_observation(stack, _obs("20240113", "f0_IW1_b0"), products=cache)
    assert found is not None
    assert found[0] is reference


def test_annotation_minmax_bbox_is_not_used_as_footprint_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Annotation bbox polygons are not accepted as source footprints."""
    swath = _swath()
    bursts = tuple(replace(burst, footprint=BBOX) for burst in swath.bursts)
    swath = replace(swath, bursts=bursts)
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.scene_bursts.run_rdr2geo",
        _stub_rdr2geo((QUAD_A, QUAD_B)),
    )
    stack = _Stack(_Catalog({"20240101": (path,)}))
    gdf = resolve_s1_scene_burst_footprints(
        stack,
        (_obs("20240101", "f0_IW1_b0"),),
    )
    geom = gdf.geometry.iloc[0]
    assert geom is not None
    xs = list(geom.exterior.coords)
    assert (10.0, 20.0) not in xs
    assert pytest.approx(xs[0][0]) == 1.0
    assert pytest.approx(xs[0][1]) == 2.0


def test_actual_source_polygon_is_explicitly_none_for_annotation_bbox() -> None:
    """Current S1 types have no persistable actual polygon distinct from the bbox."""
    burst = replace(_swath().bursts[0], footprint=BBOX)
    assert actual_source_polygon(burst) is None


def test_four_ordered_radar_corners_map_through_existing_rdr2geo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default geometry is the four ordered radar corners, dem=None."""
    calls: list[object] = []
    swath = _swath()
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        calls.append((dem, device, np.asarray(azimuth).shape))
        return _stub_rdr2geo((QUAD_A, QUAD_B))(
            model, azimuth, slant, dem, device=device
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    stack = _Stack(_Catalog({"20240101": (path,)}))
    gdf = resolve_s1_scene_burst_footprints(
        stack,
        (_obs("20240101", "f0_IW1_b0"),),
    )
    assert calls == [(None, "cpu", (4,))]
    geom = gdf.geometry.iloc[0]
    assert geom.geom_type == "Polygon"
    assert len(geom.exterior.coords) == 5


def test_partial_three_corner_success_is_null_not_triangle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fourth corner is null geometry, never a triangle or bbox."""
    swath = _swath()
    bursts = tuple(replace(burst, footprint=BBOX) for burst in swath.bursts)
    path, product = _prepare_source(tmp_path, "ref.SAFE", replace(swath, bursts=bursts))
    broken = QUAD_A.copy()
    broken[3] = np.nan

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, azimuth, slant, dem, device
        return SimpleNamespace(
            longitude_deg=broken[:, 0],
            latitude_deg=broken[:, 1],
            converged=np.array([True, True, True, True]),
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (_obs("20240101", "f0_IW1_b0"),),
    )
    assert gdf.geometry.isna().all()


def test_finite_unconverged_corners_are_null_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite lon/lat without converged=True is not a footprint."""
    swath = _swath()
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, azimuth, slant, dem, device
        return _rdr2geo_result((QUAD_A,), converged=[True, True, True, False])

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (_obs("20240101", "f0_IW1_b0"),),
    )
    assert gdf.geometry.isna().all()


def test_dateline_wrapping_quad_is_null_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dateline-wrapping ring is not published as a world-spanning polygon."""
    swath = _swath()
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)
    wrapped = np.array(
        [[179.5, 1.0], [-179.5, 1.0], [-179.5, 2.0], [179.5, 2.0]],
        dtype=np.float64,
    )

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, azimuth, slant, dem, device
        return _rdr2geo_result((wrapped,))

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (_obs("20240101", "f0_IW1_b0"),),
    )
    assert gdf.geometry.isna().all()


def test_geometry_is_per_burst_never_frame_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two bursts in one swath remain two polygons."""
    swath = _swath()
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.scene_bursts.run_rdr2geo",
        _stub_rdr2geo((QUAD_A, QUAD_B)),
    )
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (
            _obs("20240101", "f0_IW1_b0"),
            _obs("20240101", "f0_IW1_b1"),
        ),
    )
    assert len(gdf) == 2
    assert gdf.geometry.notna().all()
    a = np.array(gdf.geometry.iloc[0].exterior.coords)
    b = np.array(gdf.geometry.iloc[1].exterior.coords)
    assert not np.allclose(a, b)


def test_compatible_models_batch_rdr2geo_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same swath/orbit/DEM batches four-corner points in one solve."""
    swath = _swath()
    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)
    calls = {"n": 0, "points": 0}

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        calls["n"] += 1
        calls["points"] += int(np.asarray(azimuth).size)
        return _stub_rdr2geo((QUAD_A, QUAD_B))(
            model, azimuth, slant, dem, device=device
        )

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (
            _obs("20240101", "f0_IW1_b0"),
            _obs("20240101", "f0_IW1_b1"),
        ),
    )
    assert calls == {"n": 1, "points": 8}
    assert gdf.geometry.geom_type.ne("MultiPolygon").all()
    assert not any(
        getattr(geom, "geom_type", "") == "MultiPolygon" for geom in gdf.geometry
    )


def test_incompatible_models_fall_back_to_single_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different source swaths are not batched as one solve."""
    ref_swath = _swath("mem://ref.xml")
    sec_swath = _swath("mem://sec.xml")
    ref_path, ref_product = _prepare_source(tmp_path, "ref.SAFE", ref_swath)
    sec_path, sec_product = _prepare_source(tmp_path, "sec.SAFE", sec_swath)
    products = {ref_path: ref_product, sec_path: sec_product}
    calls = {"n": 0}

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, azimuth, slant, dem, device
        calls["n"] += 1
        quad = QUAD_A if calls["n"] == 1 else QUAD_B
        return _rdr2geo_result((quad,))

    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: products[Path(source)],  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (ref_path,), "20240113": (sec_path,)})),
        (
            _obs("20240101", "f0_IW1_b0"),
            _obs("20240113", "f0_IW1_b0"),
        ),
    )
    assert calls["n"] == 2
    assert gdf.geometry.notna().all()


def test_secondary_does_not_copy_reference_geometry_without_qualified_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unqualified secondary index matching yields null geometry."""
    ref_swath = _swath("mem://ref.xml")
    sec_swath = replace(_swath("mem://sec.xml"), bursts=())
    ref_path, ref_product = _prepare_source(tmp_path, "ref.SAFE", ref_swath)
    sec_path, sec_product = _prepare_source(tmp_path, "sec.SAFE", sec_swath)
    products = {ref_path: ref_product, sec_path: sec_product}
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: products[Path(source)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.scene_bursts.run_rdr2geo",
        _stub_rdr2geo((QUAD_A, QUAD_B)),
    )
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (ref_path,), "20240113": (sec_path,)})),
        (_obs("20240113", "f0_IW1_b0"),),
    )
    assert gdf.geometry.isna().all()


def test_missing_orbit_is_null_geometry_with_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing orbit keeps the row and does not fall back to a bbox."""
    swath = replace(_swath(), orbit=SimpleNamespace(vectors=()))
    bursts = tuple(replace(burst, footprint=BBOX) for burst in swath.bursts)
    path, product = _prepare_source(tmp_path, "ref.SAFE", replace(swath, bursts=bursts))
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )

    def fail_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        message = "rdr2geo must not run without an orbit"
        raise AssertionError(message)

    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", fail_run)
    with caplog.at_level("WARNING"):
        gdf = resolve_s1_scene_burst_footprints(
            _Stack(_Catalog({"20240101": (path,)})),
            (_obs("20240101", "f0_IW1_b0"),),
        )
    assert gdf.geometry.isna().all()
    assert any("orbit" in rec.message.lower() for rec in caplog.records)


def test_complementary_iw2_binds_via_merged_acquisition_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """f0_IW2 uses merged acquisition products, not raw catalog path 0."""
    iw1_path, iw1_product = _prepare_source(
        tmp_path, "iw1.SAFE", _swath("mem://iw1.xml")
    )
    iw2_path, iw2_product = _prepare_source(
        tmp_path, "iw2.SAFE", _swath("mem://iw2.xml", name="IW2")
    )
    products = {iw1_path: iw1_product, iw2_path: iw2_product}
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: products[Path(source)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.scene_bursts.run_rdr2geo",
        _stub_rdr2geo((QUAD_B,)),
    )
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (iw1_path, iw2_path)})),
        (_obs("20240101", "f0_IW2_b0"),),
    )
    assert gdf.geometry.notna().all()
    xs = list(gdf.geometry.iloc[0].exterior.coords)
    assert pytest.approx(xs[0][0]) == 3.0


def test_secondary_searches_all_merged_products_not_raw_frame_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secondary IW2 is found among complementary merged products, not paths[f]."""
    ref_iw1, ref_iw1_p = _prepare_source(
        tmp_path, "ref_iw1.SAFE", _swath("mem://r1.xml")
    )
    ref_iw2, ref_iw2_p = _prepare_source(
        tmp_path, "ref_iw2.SAFE", _swath("mem://r2.xml", name="IW2")
    )
    sec_iw1, sec_iw1_p = _prepare_source(
        tmp_path, "sec_iw1.SAFE", _swath("mem://s1.xml")
    )
    sec_iw2, sec_iw2_p = _prepare_source(
        tmp_path, "sec_iw2.SAFE", _swath("mem://s2.xml", name="IW2")
    )
    products = {
        ref_iw1: ref_iw1_p,
        ref_iw2: ref_iw2_p,
        sec_iw1: sec_iw1_p,
        sec_iw2: sec_iw2_p,
    }
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: products[Path(source)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "faninsar.missions.s1.scene_bursts.run_rdr2geo",
        _stub_rdr2geo((QUAD_A, QUAD_B)),
    )
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(
            _Catalog(
                {
                    "20240101": (ref_iw1, ref_iw2),
                    "20240113": (sec_iw1, sec_iw2),
                }
            )
        ),
        (_obs("20240113", "f0_IW2_b0"),),
    )
    assert gdf.geometry.notna().all()


def test_noncontiguous_burst_uses_azimuth_time_not_index_times_lpb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROI-kept bursts use burst.azimuth_time, not index * lines_per_burst."""
    swath = _swath()
    captured: list[np.ndarray] = []

    def run(
        model: object, azimuth: object, slant: object, dem: object, *, device: object
    ) -> object:
        del model, slant, dem, device
        captured.append(np.asarray(azimuth, dtype=np.float64).copy())
        return _rdr2geo_result((QUAD_B,))

    path, product = _prepare_source(tmp_path, "ref.SAFE", swath)
    monkeypatch.setattr(
        "faninsar.missions.s1.safe.open_safe_product",
        lambda source, **kwargs: product,  # noqa: ARG005
    )
    monkeypatch.setattr("faninsar.missions.s1.scene_bursts.run_rdr2geo", run)
    gdf = resolve_s1_scene_burst_footprints(
        _Stack(_Catalog({"20240101": (path,)})),
        (_obs("20240101", "f0_IW1_b1"),),
    )
    assert captured, "rdr2geo was not called"
    first_az = float(captured[0][0])
    index_line = float(swath.bursts[1].index * swath.lines_per_burst)
    assert first_az != pytest.approx(index_line)
    dt = float(swath.azimuth_time_interval_s)
    expected = (
        swath.bursts[1].azimuth_time - swath.bursts[0].azimuth_time
    ).total_seconds() / dt
    assert first_az == pytest.approx(expected, rel=1e-5, abs=1e-3)
    assert gdf.geometry.notna().all()
