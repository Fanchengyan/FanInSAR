# Proposal 0052 time-series export verification

Date: 2026-09-10

## Automated scenarios

- Invocation: `uv run --offline pytest tests/timeseries/test_export.py tests/timeseries/test_results.py -q`
  - Observable: `23 passed`.
- Invocation: `uv run --offline pytest tests/timeseries -q`
  - Observable: `24 passed`.
- Invocation: `uv run --offline ruff check faninsar/timeseries/export.py faninsar/timeseries/results.py faninsar/timeseries/io.py faninsar/timeseries/__init__.py`
  - Observable: `All checks passed!`.
- Invocation: `uv run --offline ruff format --check faninsar/timeseries/export.py faninsar/timeseries/results.py faninsar/timeseries/io.py faninsar/timeseries/__init__.py tests/timeseries/test_export.py`
  - Observable: `5 files already formatted`.

## Retained product scenario

Invocation: `env -u PROJ_DATA -u PROJ_LIB -u GDAL_DATA -u GDAL_DRIVER_PATH uv run --offline python .omo/evidence/p0052-timeseries-export-artifacts-20260910/verify_export.py`.
The script creates one `TimeSeries` with geo
`SpatialMetadata(domain="geo", shape=(2, 3), crs="EPSG:4326", transform=(1, 0, 100, 0, -1, 20))`, then runs:

```text
result.export(root / "cog")
result.export(root / "zarr", exporter="stac-zarr")
result.export(root / "netcdf", exporter="netcdf")
```

The following offline reader assertions passed:

- `cog/item.json` contains the three scientific assets and
  `faninsar:domain == "geo"`; its structured `faninsar:metadata` retains
  `phase_unit == "radian"`, and has no staging-directory self-link.
- `cog/cumulative.tif` opens with rasterio as a two-band raster with band
  descriptions `("20240101", "20240113")`.
- `zarr/timeseries.zarr` contains a current generation whose Zarr attributes
  retain `spatial.grid_identity == "grid-evidence"`.
- `netcdf/timeseries.nc` opens with xarray and has cumulative shape `(2, 2, 3)`.
- `canonical/` is written twice through `write_timeseries_product`; its
  `timeseries.zarr/TIMESERIES_CURRENT` generation changes and its final STAC
  metadata reflects the second result.
- The focused failure scenario monkeypatches the STAC writer to raise after
  Zarr completion; the observable is a canonical Zarr generation with no
  `canonical/item.json`, proving STAC is published last.

Captured artifacts:

- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/cog/item.json`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/cog/cumulative.tif`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/zarr/item.json`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/zarr/timeseries.zarr/`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/netcdf/item.json`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/netcdf/timeseries.nc`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/canonical/item.json`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/canonical/timeseries.zarr/`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/verify_export.py`
