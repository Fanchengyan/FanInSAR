"""COG export seam: labels, identity, reuse, preflight, and unsafe paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from faninsar.io.cog import (
    COG_FORMAT_VERSION,
    CogExportReport,
    ExportRequest,
    export_cog,
    export_identity,
    export_label,
    preflight_destinations,
    request_from_product,
    resolved_writer_profile,
)
from faninsar.io.raster import validate_cog, write_cog


def _request(**overrides: object) -> ExportRequest:
    """Return a baseline radar-domain export request."""
    fields: dict[str, object] = {
        "label": "unwrapped_phase",
        "asset_key": "unwrapped_phase",
        "source_write_id": "0" * 26,
        "domain": "radar",
        "looks": (4, 2),
        "date_token": "20240101-20240113",
    }
    fields.update(overrides)
    return ExportRequest(**fields)  # type: ignore[arg-type]


def test_stem_omits_the_look_marker_for_unit_looks() -> None:
    """A 1x1 product names no look factor; anything else does."""
    unit = _request(looks=(1, 1), domain="geo", date_token="20240101")
    assert unit.stem == "unwrapped_phase_20240101_geo"
    assert unit.filename == "unwrapped_phase_20240101_geo.cog.tif"

    coarse = _request()
    assert coarse.stem == "unwrapped_phase_ml4x2_20240101-20240113_radar"


def test_domain_is_the_last_filename_segment() -> None:
    """The coordinate domain is the trailing token so it stays readable."""
    assert _request().stem.endswith("_radar")
    assert _request(domain="geo").stem.endswith("_geo")


def test_ion_branch_gets_a_distinct_label_without_touching_the_asset_key() -> None:
    """Ordinary and ion-corrected branches cannot share a destination."""
    assert export_label("complex", product_kind="complex_ion") == "complex_ion"
    assert export_label("complex", product_kind="ifg") == "complex"
    assert export_label("unwrapped_phase", product_kind="unwrap_ion") == (
        "unwrapped_phase_ion"
    )


def test_identity_changes_with_every_output_affecting_input() -> None:
    """The digest covers source, selection, composition, mask, and writer."""
    base = _request()
    writer = resolved_writer_profile(dtype="float32")
    baseline = export_identity(base, mask=True, writer=writer)

    variants = [
        _request(source_write_id="1" * 26),
        _request(selection=(("unit", "f0_IW1_b0"),)),
        _request(composition="merge-on-read"),
        _request(contributing_write_ids=(("layout", "2" * 26),)),
        _request(asset_key="coherence"),
    ]
    for variant in variants:
        assert export_identity(variant, mask=True, writer=writer) != baseline

    assert export_identity(base, mask=False, writer=writer) != baseline
    assert (
        export_identity(
            base, mask=True, writer=resolved_writer_profile(dtype="float32", compress="LZW")
        )
        != baseline
    )
    assert export_identity(base, mask=True, writer=writer) == baseline


def test_requested_block_size_is_not_effective_block_size() -> None:
    """A clamped block size must not split the identity of identical files."""
    small = resolved_writer_profile(dtype="float32", blocksize=256)
    large = resolved_writer_profile(dtype="float32", blocksize=1024)
    assert small == large


def test_export_writes_valid_cog_and_reuses_on_identical_request(
    tmp_path: Path,
) -> None:
    """A repeated identical request reuses; a different one is refused."""
    request = _request()
    values = np.arange(64 * 64, dtype="float32").reshape(64, 64)

    first = export_cog(request, values, path=tmp_path)
    assert first.status == "written"
    assert first.path is not None
    assert validate_cog(first.path) == (True, ())

    second = export_cog(request, values, path=tmp_path)
    assert second.status == "reused"
    assert second.path == first.path

    changed_policy = export_cog(request, values, path=tmp_path, mask=False)
    assert changed_policy.status == "failed"
    assert "different export request" in changed_policy.reason

    replaced = export_cog(request, values, path=tmp_path, mask=False, overwrite=True)
    assert replaced.status == "written"


def test_export_identity_is_recorded_in_the_file(tmp_path: Path) -> None:
    """The digest travels with the COG, so reuse survives a process restart."""
    request = _request()
    item = export_cog(request, np.ones((32, 32), dtype="float32"), path=tmp_path)
    assert item.path is not None
    with rasterio.open(item.path) as dataset:
        tags = dataset.tags()
    assert tags["EXPORT_IDENTITY"] == item.identity
    assert tags["ASSET"] == "unwrapped_phase"
    assert tags["DOMAIN"] == "radar"
    assert tags["PAIR"] == "20240101-20240113"
    assert tags["RANGE_LOOKS"] == "4"
    assert tags["AZIMUTH_LOOKS"] == "2"
    assert tags["COG_FORMAT_VERSION"] == str(COG_FORMAT_VERSION)


def test_export_embeds_support_rather_than_inferring_it(tmp_path: Path) -> None:
    """A supported zero stays valid and an unsupported finite sample does not."""
    values = np.zeros((16, 16), dtype="float32")
    values[4:8, 4:8] = 5.0
    support = np.zeros((16, 16), dtype=bool)
    support[4:8, 4:8] = True
    support[0, 0] = True  # a supported zero

    request = _request()
    item = export_cog(request, values, path=tmp_path, mask=support)  # type: ignore[arg-type]
    assert item.path is not None
    with rasterio.open(item.path) as dataset:
        embedded = dataset.read_masks(1).astype(bool)
    np.testing.assert_array_equal(embedded, support)


def test_preflight_rejects_aliasing_destinations(tmp_path: Path) -> None:
    """Two names for one filesystem object are refused before any write."""
    existing = tmp_path / "taken.cog.tif"
    existing.write_bytes(b"x")
    alias = tmp_path / "TAKEN.cog.tif"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive; case aliasing does not apply")
    with pytest.raises(ValueError, match="one filesystem object"):
        preflight_destinations(tmp_path, [existing, alias])


def test_preflight_rejects_escape_and_symlinked_component(tmp_path: Path) -> None:
    """A destination outside the root, or through a symlink, is refused."""
    root = tmp_path / "export"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes the export root"):
        preflight_destinations(root, [tmp_path / "outside.cog.tif"])

    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preflight_destinations(root, [link / "file.cog.tif"])


def test_report_reports_partial_outcome(tmp_path: Path) -> None:
    """A batch that loses one target is partial, not silently complete."""
    from faninsar.io.cog import CogExportItem

    report = CogExportReport(
        requested=("a", "b"),
        items=(
            CogExportItem(target="a", path=tmp_path / "a.tif", status="written"),
            CogExportItem(target="b", path=None, status="failed", reason="boom"),
        ),
    )
    assert report.outcome == "partial"
    assert report.ready == ("a",)
    assert len(report.failed) == 1


def test_write_cog_reduces_a_volume_to_one_dataset_mask(tmp_path: Path) -> None:
    """A pixel is valid only when every band is finite."""
    values = np.ones((3, 24, 24), dtype="float32")
    values[1, 0, 0] = np.nan
    values[2, 5, 5] = np.nan

    path, profile = write_cog(values, tmp_path / "volume.cog.tif", None, mask=True)
    with rasterio.open(path) as dataset:
        embedded = dataset.read_masks(1).astype(bool)
    assert not embedded[0, 0]
    assert not embedded[5, 5]
    assert embedded[10, 10]
    assert profile["count"] == 3


def test_write_cog_rejects_a_mask_that_is_not_two_dimensional(tmp_path: Path) -> None:
    """An explicit mask must match the raster, not the volume."""
    values = np.ones((3, 8, 8), dtype="float32")
    with pytest.raises(ValueError, match="does not match the raster shape"):
        write_cog(values, tmp_path / "bad.cog.tif", None, mask=np.ones((3, 8, 8), bool))


def test_request_from_product_refuses_an_unknown_asset() -> None:
    """An export cannot claim an asset the product does not commit."""

    class Product:
        properties = {
            "faninsar:kind": "ifg",
            "faninsar:write_id": "0" * 26,
            "faninsar:pair": ["20240101", "20240113"],
        }
        assets = {"complex": {"href": "complex.zarr"}}

    with pytest.raises(ValueError, match="has no asset"):
        request_from_product(product=Product(), asset_key="coherence")

    request = request_from_product(product=Product(), asset_key="complex")
    assert request.date_token == "20240101-20240113"
    assert request.domain == "radar"


def _collection(tmp_path: Path, href: object) -> object:
    """Build a one-pair frame collection whose item records ``href``."""
    import json

    from faninsar.network.readers.interferogram import InterferogramCollection

    root = tmp_path / "frame" / "interferograms"
    pair = root / "20240101_20240113"
    pair.mkdir(parents=True)
    (pair / "unw_phase.cog.tif").write_bytes(b"payload")
    item = {
        "stac_version": "1.0.0",
        "type": "Feature",
        "id": "pair",
        "properties": {},
        "assets": {"unw_phase": {"href": href}},
    }
    (pair / "item.json").write_text(json.dumps(item), encoding="utf-8")
    return InterferogramCollection(root)


@pytest.mark.parametrize(
    "href",
    ["../escape.cog.tif", "/etc/passwd", "nested/unw_phase.cog.tif", ".", ".."],
)
def test_collection_path_rejects_an_unsafe_recorded_href(
    tmp_path: Path, href: str
) -> None:
    """A recorded href is data: it may not leave the collection or nest."""
    from faninsar.processing.errors import InvalidProcessingStateError

    collection = _collection(tmp_path, href)
    # ``reject`` raises the processing-state error for a rejected layout, which
    # is the same family the rest of the storage layer uses for unsafe paths.
    with pytest.raises((ValueError, TypeError, InvalidProcessingStateError)):
        collection.path("20240101_20240113", "unw_phase")


def test_collection_path_rejects_a_symlinked_pair_directory(tmp_path: Path) -> None:
    """A pair directory that is a symlink is refused before any read."""
    from faninsar.network.readers.interferogram import InterferogramCollection
    from faninsar.processing.errors import InvalidProcessingStateError

    root = tmp_path / "frame" / "interferograms"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "item.json").write_text("{}", encoding="utf-8")
    (root / "20240101_20240113").symlink_to(outside, target_is_directory=True)

    collection = InterferogramCollection(root)
    with pytest.raises((ValueError, InvalidProcessingStateError), match="symlink"):
        collection.path("20240101_20240113", "unw_phase")


def test_collection_path_honours_a_recorded_href(tmp_path: Path) -> None:
    """A committed href wins over the built-in table, keeping old layouts readable."""
    collection = _collection(tmp_path, "legacy_unw_phase.cog.tif")
    resolved = collection.path("20240101_20240113", "unw_phase")
    assert resolved.name == "legacy_unw_phase.cog.tif"

    # An item that records nothing still resolves through the table.
    plain = _collection(tmp_path / "plain", None)
    assert plain.path("20240101_20240113", "unw_phase").name == "unw_phase.cog.tif"


def test_stable_import_surface() -> None:
    """The raster primitives are reachable from ``faninsar.io`` and the shim."""
    from faninsar.io import read_geogrid, validate_cog, write_cog
    from faninsar.io.raster import RasterGridSpec
    from faninsar.network.readers import raster_io

    assert raster_io.write_cog is write_cog
    assert raster_io.validate_cog is validate_cog
    assert raster_io.read_geogrid is read_geogrid
    assert RasterGridSpec is not None
