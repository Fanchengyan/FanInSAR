"""Selected physical bursts bound ROI row demand (P0061/P0070/P0071)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from faninsar.data.query import BoundingBox
from faninsar.io import tiff
from faninsar.missions.s1 import acquisition, roi_transfer
from faninsar.missions.s1.annotation import parse_annotation_xml
from faninsar.missions.s1.errors import Sentinel1ProductError
from faninsar.remote import RemoteResourceBudget
from faninsar.remote.fetch import RangeSpan
from tests.missions.s1.test_from_roi import _parent_xml

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


@pytest.mark.parametrize("coreg_mode", ["geometry", "nesd"])
def test_roi_owner_keeps_only_hinted_burst_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, coreg_mode: str
) -> None:
    """A spatial envelope crossing neighbors cannot widen catalog selection."""
    xml = _parent_xml()
    parent = parse_annotation_xml(
        xml,
        annotation_path="annotation.xml",
        measurement_path="measurement.tiff",
        _original_parent=True,
    )
    monkeypatch.setattr(
        acquisition, "_roi_envelopes", lambda *_: [(18, 27, 4, 20), (31, 33, 4, 20)]
    )
    owner = roi_transfer.plan_owner(
        {
            "measurement_url": "https://example.test/measurement.tiff",
            "_asf_burst_hints": [(12345, parent.bursts[2].sensing_time)],
        },
        BoundingBox(100.0, 42.0, 100.4, 42.1, crs="EPSG:4326"),
        safe_dir=tmp_path / "selected.SAFE",
        fingerprint="selected",
        context=acquisition._RoiFetchContext(),
        coreg_mode=coreg_mode,
        granularity="roi",
        dem=object(),
        xml_text=xml,
        plan_spans=False,
        # Item-local hints must win over unrelated search-wide catalog hints.
        absolute_burst_ids=[67890],
        catalog_sensing_times=[parent.bursts[0].sensing_time],
    )
    assert owner is not None
    assert owner.listed == (2,)
    assert [burst.index for burst in owner.bursts] == [2]
    assert owner.absolute_burst_ids == {2: 12345}
    window = owner.bursts[0].windows[0]
    assert (window.parent_line0, window.parent_line1) == (20, 27)
    assert (window.col0, window.n_samples) == (4, 16)
    assert owner.bursts[0].overlaps == []
    assert len(owner.swath.bursts) == 6
    assert (
        owner.swath.bursts[2].first_valid_sample == parent.bursts[2].first_valid_sample
    )
    assert owner.swath.bursts[2].last_valid_sample == parent.bursts[2].last_valid_sample
    assert owner.xml_text == xml
    url = str(owner.item["measurement_url"])
    owner.context.layouts[url] = tiff.layout_from_tables(
        [4096 + row * 400 for row in range(60)], [400] * 60, 1, samples=100
    )
    roi_transfer._plan_owner_spans(owner)
    assert [(span.start, span.end) for span in owner.bursts[0].spans] == [
        (12096, 14895)
    ]


def test_roi_owner_skips_when_hinted_burst_misses_roi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An empty selection intersection skips instead of downloading neighbors."""
    xml = _parent_xml()
    parent = parse_annotation_xml(
        xml,
        annotation_path="annotation.xml",
        measurement_path="measurement.tiff",
        _original_parent=True,
    )
    monkeypatch.setattr(acquisition, "_roi_envelopes", lambda *_: [(21, 25, 4, 20)])
    owner = roi_transfer.plan_owner(
        {"_asf_burst_hints": [(12345, parent.bursts[0].sensing_time)]},
        BoundingBox(100.0, 42.0, 100.4, 42.1, crs="EPSG:4326"),
        safe_dir=Path(tmp_path) / "selected.SAFE",
        fingerprint="selected",
        context=acquisition._RoiFetchContext(),
        coreg_mode="geometry",
        granularity="roi",
        dem=object(),
        xml_text=xml,
        plan_spans=False,
    )
    assert owner is None
    assert "no selected physical burst intersects the ROI" in caplog.text


@pytest.mark.parametrize("hint_kind", ["empty", "missing_time", "unmatched"])
def test_roi_owner_rejects_invalid_physical_hints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hint_kind: str
) -> None:
    """Invalid catalog selection must never fall back to spatial neighbors."""
    xml = _parent_xml()
    parent = parse_annotation_xml(
        xml,
        annotation_path="annotation.xml",
        measurement_path="measurement.tiff",
        _original_parent=True,
    )
    hints = {
        "empty": [],
        "missing_time": [(12345, None)],
        "unmatched": [(12345, parent.bursts[2].sensing_time + timedelta(hours=1))],
    }[hint_kind]
    monkeypatch.setattr(acquisition, "_roi_envelopes", lambda *_: [(21, 25, 4, 20)])
    with pytest.raises(Sentinel1ProductError):
        roi_transfer.plan_owner(
            {"_asf_burst_hints": hints},
            BoundingBox(100.0, 42.0, 100.4, 42.1, crs="EPSG:4326"),
            safe_dir=tmp_path / "selected.SAFE",
            fingerprint="selected",
            context=acquisition._RoiFetchContext(),
            coreg_mode="geometry",
            granularity="roi",
            dem=object(),
            xml_text=xml,
            plan_spans=False,
        )


@pytest.mark.parametrize("workers", [4, 16])
def test_roi_fetcher_preserves_preparation_http_policy(
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """ROI sessions receive H4/H16, four URL slots, per-URL R4 and metadata M2."""
    budget = RemoteResourceBudget(
        max_workers=workers,
        max_active_files=4,
        max_ranges_per_file=4,
        max_metadata_files=2,
    )
    ledger = object()
    calls = []

    @asynccontextmanager
    async def provider_fetcher(
        spans: Sequence[RangeSpan], **policy: object
    ) -> AsyncIterator[object]:
        calls.append((spans, policy))
        yield object()

    monkeypatch.setattr(acquisition, "_cdse_range_fetcher", provider_fetcher)
    span = RangeSpan(
        "https://download.dataspace.copernicus.eu/odata/v1/annotation",
        0,
        10,
        "annotation-0",
    )

    async def run() -> None:
        token = acquisition._preparation_remote_policy.set((budget, ledger))
        try:
            async with acquisition._span_fetcher_cm([span]) as fetcher:
                await fetcher._ensure()
        finally:
            acquisition._preparation_remote_policy.reset(token)

    asyncio.run(run())
    assert calls == [((span,), {"budget": budget, "ledger": ledger})]
