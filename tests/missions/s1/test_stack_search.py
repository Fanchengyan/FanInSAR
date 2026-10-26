"""Focused tests for the Sentinel-1 Stack search seam."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from faninsar import remote
from faninsar.missions.s1.search import (
    catalog_for_product,
    complete_safe_asset,
    filter_burst_polarizations,
    normalize_polarizations,
    product_filters,
    provider_polarization_filter,
    safe_requires_aux_cal,
    select_orbit_bindings,
    validate_calibration_paths,
)
from faninsar.missions.s1.stack import S1Stack

if TYPE_CHECKING:
    from pathlib import Path


def _item(
    *assets: remote.RemoteAsset,
    raw_metadata: dict[str, object] | None = None,
) -> remote.CatalogItem:
    """Build one normalized catalog item for seam tests."""
    acquisition = remote.AcquisitionMetadata(
        acquisition_id="S1_TEST",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    )
    return remote.CatalogItem(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        acquisition=acquisition,
        assets={asset.key: asset for asset in assets},
        raw_metadata=raw_metadata or {},
    )


def _asset(
    key: str = "data", href: str = "https://datapool.asf.alaska.edu/S1.zip"
) -> remote.RemoteAsset:
    """Build one complete remote SAFE asset."""
    return remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="S1-SLC",
        item_id="S1_TEST",
        key=key,
        href=href,
        media_type="application/zip",
        representation="complete",
    )


def test_s1_polarization_defaults_and_dual_modes() -> None:
    """Single, dual, and all-channel requests map to local channels."""
    assert normalize_polarizations(None) == ("VV",)
    assert normalize_polarizations("VV+VH") == ("VV", "VH")
    assert normalize_polarizations(("HH", "HV")) == ("HH", "HV")
    assert normalize_polarizations("all") == "all"
    assert provider_polarization_filter(("VV", "VH")) is None
    assert provider_polarization_filter(("VV", "VH"), allow_dual=True) == "VV+VH"
    assert provider_polarization_filter("all") is None


def test_s1_product_type_maps_to_asf_dataset_and_level() -> None:
    """Burst is the default and scene selects full-scene SLC."""
    assert product_filters("burst", ("VV",)) == {
        "dataset": "SLC-BURST",
        "processingLevel": "BURST",
        "polarization": "VV",
    }
    assert "polarization" not in product_filters("burst", ("VV", "VH"))
    assert product_filters("scene", ("VV", "VH"))["polarization"] == "VV+VH"
    assert product_filters("scene", "all") == {
        "dataset": "SENTINEL-1",
        "processingLevel": "SLC",
    }


def test_complete_safe_asset_rejects_bare_burst_tiff() -> None:
    """A burst TIFF cannot cross the SAFE reader boundary as a complete file."""
    tiff = _asset(href="https://sentinel1.asf.alaska.edu/burst.tiff")
    with pytest.raises(ValueError, match="SAFE ZIP"):
        complete_safe_asset(_item(tiff))


def test_product_families_share_stable_asf_registration() -> None:
    """Dataset filters distinguish burst and scene under one S1 catalog."""
    assert catalog_for_product("burst") == "sentinel1-asf"
    assert catalog_for_product("scene") == "sentinel1-asf"


def test_asf_scene_registration_uses_canonical_collection_concept() -> None:
    """The scientific scene name maps to ASF's Sentinel-1 concept ID."""
    from faninsar.missions.s1.search import ensure_asf_catalog
    from faninsar.remote.catalog import _ADAPTERS

    name = ensure_asf_catalog(product_type="scene")
    assert _ADAPTERS[name].query_collection == "C1214470488-ASF"


def test_asf_registration_rejects_different_redirect_policy() -> None:
    """A stable catalog name cannot hide a different redirect policy."""
    from faninsar.missions.s1.search import ensure_asf_catalog
    from faninsar.remote.catalog import _ADAPTERS

    name = ensure_asf_catalog(product_type="burst")
    adapter = _ADAPTERS[name]
    original = adapter.redirect_origins
    adapter.redirect_origins = (*original, "https://unexpected.example")
    try:
        with pytest.raises(ValueError, match="incompatible ASF registration"):
            ensure_asf_catalog(product_type="scene")
    finally:
        adapter.redirect_origins = original


def test_orbit_selection_prefers_covering_poeorb(tmp_path: Path) -> None:
    """Official S1 priority selects POEORB before a covering RESORB."""
    body = """<Earth_Explorer_File><Data_Block><List_of_OSVs>
    <OSV><UTC>UTC=2023-12-31T23:00:00</UTC><X>1</X><Y>2</Y><Z>3</Z>
    <VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
    <OSV><UTC>UTC=2024-01-01T01:00:00</UTC><X>1</X><Y>2</Y><Z>3</Z>
    <VX>4</VX><VY>5</VY><VZ>6</VZ></OSV>
    </List_of_OSVs></Data_Block></Earth_Explorer_File>"""
    precise = tmp_path / "S1A_OPER_AUX_POEORB_TEST.EOF"
    restituted = tmp_path / "S1A_OPER_AUX_RESORB_TEST.EOF"
    precise.write_text(body)
    restituted.write_text(body)
    selected = select_orbit_bindings(
        [restituted, precise],
        {
            "20240101": (
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            )
        },
    )
    assert selected == {"20240101": precise}


def test_ipf_002_36_is_detected_from_safe_manifest(tmp_path: Path) -> None:
    """ISCE2's early-product condition is read from manifest.safe."""
    safe = tmp_path / "S1_TEST.SAFE"
    safe.mkdir()
    (safe / "manifest.safe").write_text(
        '<safe:processing xmlns:safe="http://www.esa.int/safe/sentinel-1.0">'
        '<safe:facility><safe:software version="002.36"/></safe:facility>'
        "</safe:processing>"
    )
    assert safe_requires_aux_cal(safe)


def test_burst_multi_polarization_filters_catalog_metadata() -> None:
    """Multi-channel burst searches retain only requested channels."""
    dual = _item(_asset(), raw_metadata={"Polarization": "VV+VH"})
    single = _item(_asset(), raw_metadata={"Polarization": "VV"})
    hh = _item(_asset(), raw_metadata={"Polarization": "HH"})
    assert filter_burst_polarizations([dual, single, hh], ("VV", "VH")) == [
        dual,
        single,
    ]
    with pytest.raises(ValueError, match="unambiguous polarization"):
        filter_burst_polarizations([_item(_asset())], ("VV", "VH"))


def test_orbit_selection_prefers_poeorb_then_resorb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Valid covering POEORB wins deterministically over RESORB."""
    poe = tmp_path / "S1_POEORB.EOF"
    res = tmp_path / "S1_RESORB.EOF"
    poe.touch()
    res.touch()
    vectors = [
        type("Vector", (), {"time": datetime(2024, 1, 1, tzinfo=UTC)})(),
        type("Vector", (), {"time": datetime(2024, 1, 2, tzinfo=UTC)})(),
    ]
    monkeypatch.setattr(
        "faninsar.missions.s1.orbit.read_eof_orbit",
        lambda _path: type("Orbit", (), {"vectors": vectors})(),
    )
    interval = {"20240101": (vectors[0].time, vectors[1].time)}
    assert select_orbit_bindings([res, poe], interval) == {"20240101": poe}
    assert select_orbit_bindings([res], interval) == {"20240101": res}


def test_orbit_selection_uses_embedded_vectors_without_candidates() -> None:
    """No explicit candidates preserve the SAFE embedded orbit vectors."""
    assert (
        select_orbit_bindings(
            None,
            {
                "20240101": (
                    datetime.min.replace(tzinfo=UTC),
                    datetime.max.replace(tzinfo=UTC),
                )
            },
        )
        == {}
    )


def test_orbit_selection_rejects_configured_gap(tmp_path: Path) -> None:
    """Configured orbit lookup failure is visible instead of silently embedded."""
    candidate = tmp_path / "S1_RESORB.EOF"
    candidate.touch()
    with pytest.raises(ValueError, match="do not cover"):
        select_orbit_bindings(
            [candidate],
            {
                "20240101": (
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                )
            },
        )


def test_calibration_bindings_require_existing_local_files(tmp_path: Path) -> None:
    """Explicit AUX_CAL bindings are validated and normalized to Paths."""
    calibration = tmp_path / "AUX_CAL.xml"
    calibration.touch()
    assert validate_calibration_paths([calibration]) == (calibration,)
    with pytest.raises(FileNotFoundError):
        validate_calibration_paths([tmp_path / "missing.xml"])


def test_from_search_downloads_one_selected_safe_and_prepares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constructor downloads selected complete assets before preparation."""
    item = _item(_asset())
    downloaded: list[Path] = []
    prepared: list[bool] = []

    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [item])
    )

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del asset, kwargs
        downloaded.append(destination)
        return destination

    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            prepared.append(True)

    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kwargs: FakeStack()),
    )
    result = S1Stack.from_roi(
        object(), work_dir=tmp_path, granularity="burst", provider="asf"
    )
    assert isinstance(result, FakeStack)
    assert downloaded == [tmp_path / "inputs" / "S1_TEST.zip"]
    assert prepared == [True]


def test_from_search_downloads_safes_concurrently_in_selection_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ready SAFE can finish while an earlier burst is still preparing."""
    slow = replace(_item(_asset("slow")), item_id="A_SLOW")
    fast = replace(_item(_asset("fast")), item_id="B_FAST")
    monkeypatch.setattr(
        S1Stack,
        "search",
        classmethod(lambda _cls, *_args, **_kwargs: [slow, fast]),
    )
    barrier = threading.Barrier(2)
    fast_done = threading.Event()
    completed: list[str] = []
    captured: list[tuple[Path, ...]] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del asset, kwargs
        barrier.wait(timeout=2)
        if destination.name.startswith("A_SLOW"):
            assert fast_done.wait(timeout=2)
        else:
            fast_done.set()
        completed.append(destination.name)
        return destination

    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            return None

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> FakeStack:
        del cls, kwargs
        captured.append(tuple(paths))
        return FakeStack()

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    S1Stack.from_roi(object(), work_dir=tmp_path, granularity="burst", provider="asf")

    assert completed == ["B_FAST.zip", "A_SLOW.zip"]
    assert captured == [
        (tmp_path / "inputs" / "A_SLOW.zip", tmp_path / "inputs" / "B_FAST.zip")
    ]


def test_from_search_sweeps_burst_probes_oldest_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stack retries due burst probes centrally, then downloads ready inputs."""
    from faninsar.remote.fetch.complete import _RemotePollObservation

    items = [
        _item(replace(_asset(f"burst-{index}"), item_id=f"S1_{index}"))
        for index in range(3)
    ]
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: items)
    )
    observations: dict[str, int] = {}
    probe_calls: list[tuple[str, int]] = []

    def probe(asset: Any, **kwargs: Any) -> Any:
        attempt = int(kwargs["attempt"])
        observations[asset.item_id] = attempt
        probe_calls.append((asset.item_id, attempt))
        if attempt == 0:
            return _RemotePollObservation("pending", kwargs["url"], 0.0)
        return _RemotePollObservation("ready", kwargs["url"])

    downloaded: list[str] = []

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        downloaded.append(asset.item_id)
        return destination

    monkeypatch.setattr(remote, "probe", probe)
    monkeypatch.setattr(remote, "download", download)

    class FakeStack:
        def prepare(self) -> None:
            return None

    monkeypatch.setattr(
        S1Stack,
        "from_safes",
        classmethod(lambda _cls, _paths, **_kwargs: FakeStack()),
    )
    S1Stack.from_roi(object(), work_dir=tmp_path, granularity="burst", provider="asf")

    assert sorted(index for index, attempt in probe_calls if attempt == 0) == [
        "S1_0",
        "S1_1",
        "S1_2",
    ]
    assert [index for index, attempt in probe_calls if attempt == 1] == [
        "S1_0",
        "S1_1",
        "S1_2",
    ]
    assert sorted(downloaded) == ["S1_0", "S1_1", "S1_2"]


def test_from_search_does_not_construct_stack_after_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One failed SAFE cancels the construction boundary for the whole search."""
    first = replace(_item(_asset("first")), item_id="S1_FIRST")
    second = replace(_item(_asset("second")), item_id="S1_SECOND")
    monkeypatch.setattr(
        S1Stack,
        "search",
        classmethod(lambda _cls, *_args, **_kwargs: [first, second]),
    )
    stack_constructed = False

    def download(asset: Any, destination: Path, **kwargs: Any) -> Path:
        del kwargs
        if asset.key == "second":
            message = "burst generation failed"
            raise RuntimeError(message)
        return destination

    monkeypatch.setattr(remote, "download", download)

    def from_safes(cls: Any, paths: Any, **kwargs: Any) -> NoReturn:
        del cls, paths, kwargs
        nonlocal stack_constructed
        stack_constructed = True
        message = "from_safes must not run after download failure"
        raise AssertionError(message)

    monkeypatch.setattr(S1Stack, "from_safes", classmethod(from_safes))
    with pytest.raises(RuntimeError, match="burst generation failed"):
        S1Stack.from_roi(
            object(), work_dir=tmp_path, granularity="burst", provider="asf"
        )
    assert not stack_constructed


def test_from_search_rejects_tiff_xml_without_burst_extractor_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TIFF/XML components never cross the complete SAFE input boundary."""
    measurement = remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="SLC-BURST",
        item_id="S1_TEST",
        key="measurement",
        href="https://sentinel1.asf.alaska.edu/burst.tiff",
        media_type="image/tiff",
        representation="component",
    )
    annotation = remote.RemoteAsset(
        provider="ASF",
        catalog="sentinel1-asf",
        collection="SLC-BURST",
        item_id="S1_TEST",
        key="annotation",
        href="https://sentinel1.asf.alaska.edu/burst.xml",
        media_type="application/xml",
        representation="component",
    )
    item = _item(measurement, annotation)
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [item])
    )

    downloads: list[object] = []
    monkeypatch.setattr(
        remote,
        "download",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="exactly one complete SAFE ZIP"):
        S1Stack.from_roi(
            object(), work_dir=tmp_path, granularity="burst", provider="asf"
        )

    assert downloads == []


def test_from_search_rejects_ambiguous_safe_zips_before_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extra TIFF/XML assets cannot hide two competing complete SAFE ZIPs."""
    item = _item(
        _asset("safe-a", "https://sentinel1.asf.alaska.edu/a.zip"),
        _asset("safe-b", "https://sentinel1.asf.alaska.edu/b.zip"),
        _asset("measurement", "https://sentinel1.asf.alaska.edu/burst.tiff"),
        _asset("annotation", "https://sentinel1.asf.alaska.edu/burst.xml"),
    )
    monkeypatch.setattr(
        S1Stack, "search", classmethod(lambda _cls, *_args, **_kwargs: [item])
    )
    downloads: list[object] = []
    monkeypatch.setattr(
        remote,
        "download",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="found 2"):
        S1Stack.from_roi(
            object(), work_dir=tmp_path, granularity="burst", provider="asf"
        )

    assert downloads == []
