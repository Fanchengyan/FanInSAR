"""Tests for the public TimeSeries export seam."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from faninsar.timeseries import SpatialMetadata, TimeSeries, write_timeseries_product
from faninsar.timeseries.export import ExporterRegistry, registry
from faninsar.timeseries.io import read_timeseries

if TYPE_CHECKING:
    from pathlib import Path


def _result(*, domain: str = "geo") -> TimeSeries:
    """Build a small deterministic result with structured grid metadata."""
    spatial = SpatialMetadata(
        domain=domain,  # type: ignore[arg-type]
        shape=(2, 3),
        grid_identity="grid-test",
        crs="EPSG:4326" if domain == "geo" else None,
        transform=(1.0, 0.0, 100.0, 0.0, -1.0, 20.0) if domain == "geo" else None,
        radar_coordinates={"range_spacing_m": 2.3} if domain == "radar" else None,
    )
    return TimeSeries(
        pair_ids=("20240101_20240113",),
        dates=("20240101", "20240113"),
        phase_increments_rad=np.ones((1, 2, 3), dtype=np.float32),
        residual_phase_pairs_rad=np.zeros((1, 2, 3), dtype=np.float32),
        phase_cumulative_rad=np.ones((2, 2, 3), dtype=np.float32),
        metadata={"phase_unit": "radian"},
        spatial=spatial,
    )


def test_spatial_metadata_requires_geo_reference() -> None:
    """Geographic results reject incomplete CRS/transform identity."""
    with pytest.raises(ValueError, match="CRS"):
        SpatialMetadata(domain="geo", shape=(2, 2), grid_identity="grid")


def test_default_export_is_cog_and_preserves_stac_metadata(tmp_path: Path) -> None:
    """The default geo export writes multiband COG assets and final STAC."""
    root = tmp_path / "result"

    exported = _result().export(root)

    assert exported == root
    item = json.loads((root / "item.json").read_text(encoding="utf-8"))
    assert item["properties"]["faninsar:domain"] == "geo"
    assert item["properties"]["faninsar:grid_identity"] == "grid-test"
    assert item["properties"]["faninsar:metadata"]["phase_unit"] == "radian"
    assert all(".export-" not in link["href"] for link in item["links"])
    assert (root / "cumulative.tif").is_file()
    assert (root / "increments.tif").is_file()
    with rasterio.open(root / "cumulative.tif") as dataset:
        assert dataset.count == 2
        assert dataset.descriptions == ("20240101", "20240113")
        assert dataset.tags(1)["faninsar_identity"] == "20240101"


def test_default_export_rejects_radar_without_fallback(tmp_path: Path) -> None:
    """The COG default never silently converts radar coordinates."""
    with pytest.raises(ValueError, match="geo"):
        _result(domain="radar").export(tmp_path / "result")


def test_stac_zarr_export_writes_canonical_asset_and_metadata(tmp_path: Path) -> None:
    """The explicit canonical exporter writes a Zarr group plus a STAC item."""
    root = tmp_path / "result"

    _result(domain="radar").export(root, exporter="stac-zarr")

    item = json.loads((root / "item.json").read_text(encoding="utf-8"))
    assert item["properties"]["faninsar:domain"] == "radar"
    assert (root / "timeseries.zarr").is_dir()
    assert read_timeseries(root).spatial.domain == "radar"


def test_canonical_product_writer_creates_and_updates_one_root(tmp_path: Path) -> None:
    """Canonical persistence supports repeated updates in place."""
    root = tmp_path / "canonical"
    first = _result(domain="radar")

    assert write_timeseries_product(first, root) == root
    first_item = json.loads((root / "item.json").read_text(encoding="utf-8"))
    assert first_item["properties"]["faninsar:metadata"] == {"phase_unit": "radian"}
    first_write_id = first_item["properties"]["faninsar:write_id"]

    second = replace(first, metadata={"phase_unit": "radian", "revision": 2})
    write_timeseries_product(second, root, overwrite=True)
    second_item = json.loads((root / "item.json").read_text(encoding="utf-8"))
    assert second_item["properties"]["faninsar:write_id"] != first_write_id
    assert second_item["properties"]["faninsar:metadata"]["revision"] == 2


def test_canonical_product_writer_requires_overwrite_to_replace(
    tmp_path: Path,
) -> None:
    """A second commit without ``overwrite`` is rejected, not silently applied."""
    from faninsar.processing.errors import InvalidProcessingStateError

    root = tmp_path / "canonical"
    write_timeseries_product(_result(domain="radar"), root)
    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        write_timeseries_product(_result(domain="radar"), root)


def test_canonical_path_is_a_validated_path_value() -> None:
    """TimeSeries rejects non-Path canonical identities."""
    with pytest.raises(ValueError, match="canonical_path"):
        TimeSeries(
            pair_ids=(),
            dates=(),
            phase_increments_rad=np.empty((0, 1, 1)),
            residual_phase_pairs_rad=np.empty((0, 1, 1)),
            phase_cumulative_rad=np.empty((0, 1, 1)),
            metadata={},
            canonical_path="not-a-path",  # type: ignore[arg-type]
        )


def test_netCDF_export_and_registered_class(tmp_path: Path) -> None:
    """A registered exporter class is instantiated at the public seam."""
    root = tmp_path / "result"

    _result().export(root, exporter="netcdf")

    assert (root / "timeseries.nc").is_file()
    assert (root / "item.json").is_file()


def test_custom_exporter_receives_path_and_overwrite(tmp_path: Path) -> None:
    """A direct configured callable remains a small extension escape hatch."""
    calls: list[tuple[Path, bool]] = []

    def custom(result: TimeSeries, path: Path, *, overwrite: bool) -> Path:
        del result
        calls.append((path, overwrite))
        path.mkdir()
        return path

    root = tmp_path / "result"
    assert _result().export(root, exporter=custom) == root
    assert calls == [(root, False)]


def test_existing_destination_requires_overwrite(tmp_path: Path) -> None:
    """Export never changes an existing root without explicit overwrite."""
    root = tmp_path / "result"
    root.mkdir()
    marker = root / "marker"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _result().export(root, exporter="stac-zarr")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_registry_accepts_zero_argument_exporter_classes() -> None:
    """Registry class registration is explicit and process-local."""

    class Exporter:
        def __call__(self, result: TimeSeries, path: Path, *, overwrite: bool) -> Path:
            del result, overwrite
            return path

    local = ExporterRegistry()
    local.register("test", Exporter)
    assert local.resolve("test") is Exporter
