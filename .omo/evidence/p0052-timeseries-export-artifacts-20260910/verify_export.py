"""Generate and verify retained Proposal 0052 time-series export artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import zarr

from faninsar.timeseries import (
    SpatialMetadata,
    TimeSeries,
    open_timeseries_zarr,
    write_timeseries_product,
)


ROOT = Path(__file__).parent
SPATIAL = SpatialMetadata(
    domain="geo",
    shape=(2, 3),
    grid_identity="grid-evidence",
    crs="EPSG:4326",
    transform=(1.0, 0.0, 100.0, 0.0, -1.0, 20.0),
)
RESULT = TimeSeries(
    pair_ids=("20240101_20240113",),
    dates=("20240101", "20240113"),
    phase_increments_rad=np.ones((1, 2, 3), dtype=np.float32),
    residual_phase_pairs_rad=np.zeros((1, 2, 3), dtype=np.float32),
    phase_cumulative_rad=np.ones((2, 2, 3), dtype=np.float32),
    metadata={"phase_unit": "radian"},
    spatial=SPATIAL,
)


def main() -> None:
    """Write all built-ins and verify their binary products."""
    RESULT.export(ROOT / "cog", overwrite=True)
    RESULT.export(ROOT / "zarr", exporter="stac-zarr", overwrite=True)
    RESULT.export(ROOT / "netcdf", exporter="netcdf", overwrite=True)

    item = json.loads((ROOT / "cog" / "item.json").read_text(encoding="utf-8"))
    assert set(item["assets"]) == {"increments", "cumulative", "residual_pairs"}
    assert item["properties"]["faninsar:domain"] == "geo"
    assert item["properties"]["faninsar:metadata"]["phase_unit"] == "radian"
    assert all(".export-" not in link["href"] for link in item["links"])
    with rasterio.open(ROOT / "cog" / "cumulative.tif") as dataset:
        assert dataset.count == 2
        assert dataset.descriptions == ("20240101", "20240113")
    generation = next((ROOT / "zarr" / "timeseries.zarr").glob(".timeseries_generations/*"))
    group = zarr.open_group(str(generation), mode="r")
    assert group.attrs["spatial"]["grid_identity"] == "grid-evidence"
    with xr.open_dataset(ROOT / "netcdf" / "timeseries.nc") as dataset:
        assert dataset["cumulative"].shape == (2, 2, 3)

    canonical = ROOT / "canonical"
    write_timeseries_product(RESULT, canonical)
    with open_timeseries_zarr(canonical / "timeseries.zarr") as pinned:
        first_generation = pinned.generation_id
    write_timeseries_product(
        replace(RESULT, metadata={"phase_unit": "radian", "revision": 2}),
        canonical,
    )
    with open_timeseries_zarr(canonical / "timeseries.zarr") as pinned:
        assert pinned.generation_id != first_generation
    canonical_item = json.loads(
        (canonical / "item.json").read_text(encoding="utf-8")
    )
    assert canonical_item["properties"]["faninsar:metadata"]["revision"] == 2
    print(f"verified {ROOT}")


if __name__ == "__main__":
    main()
