# Automatic DEM Management

`DEMManager` removes the manual DEM download step. Given a geographic extent, it
resolves which Copernicus GLO-30 raw tiles cover the area, reuses tiles that are
already present in a local cache, downloads any missing tiles from the public AWS
S3 bucket, and mosaics them into one GeoTIFF at a canonical output path. The
pipeline can do all of this by itself when `run_pair` is called with `dem=None`
and the environment is configured.

## Quick start

Point `FANINSAR_DEM_CACHE_DIR` at a folder of raw Copernicus GLO-30 tiles (an
empty folder works too - missing tiles are downloaded on demand), then ask the
manager for a combined DEM covering your ROI:

```python
import os
from pathlib import Path

from faninsar.processing.geometry import get_dem_manager
from faninsar.query import BoundingBox

os.environ["FANINSAR_DEM_CACHE_DIR"] = "./dem_cache"

manager = get_dem_manager()
roi = BoundingBox(97.5, 38.7, 101.4, 40.0)
dem_path = manager.fetch_dem(roi, Path("./output/dem.tif"))
```

- **`get_dem_manager()`** - builds a manager from the environment. Only
  `FANINSAR_DEM_CACHE_DIR` is required; `FANINSAR_DEM_SOURCE_URL` overrides the
  download source and `FANINSAR_DEM_NAME` renames the default output file.
- **`roi`** - a `BoundingBox`, or any `(min_lon, min_lat, max_lon, max_lat)`
  tuple in EPSG:4326. Both are accepted by every `DEMManager` method.
- **`fetch_dem(bounds, output_path)`** - the one-stop call: it queries the
  needed tiles, downloads anything missing into the cache, mosaics them, writes
  the combined GeoTIFF, and returns its path. If you omit `output_path`, the
  file lands at `<cache parent>/dem/<name>` (see [Output location](output-location)).

For the pipeline that calls `run_pair` for you, see
[Use it automatically in the pipeline](use-it-automatically-in-the-pipeline).

## What tiles cover my area?

Use `required_tiles` to see the Copernicus GLO-30 tile identifiers that
intersect an extent without touching the network or the cache:

```python
from pathlib import Path

from faninsar.processing.geometry import DEMManager

manager = DEMManager(Path("./dem_cache"))
tiles = manager.required_tiles((97.5, 38.7, 101.4, 40.0))
for tile_dir, filename in tiles:
    print(tile_dir, filename)
```

Each entry is a `(tile_dir, filename)` pair following the COG convention, for
example `("N38_E097", "Copernicus_DSM_COG_10_N38_00_E097_00_DEM.tif")`. Tiles
follow the one-degree Copernicus GLO-30 grid. The exported
`copernicus_tile_name(lat, lon)` helper returns the same pair for a single
coordinate.

## Cache behavior

The cache folder can use either of two layouts, and both are accepted at the
same time:

- flat files: `dem_cache/Copernicus_DSM_COG_10_N38_00_E097_00_DEM.tif`
- per-tile folders: `dem_cache/N38_E097/Copernicus_DSM_COG_10_N38_00_E097_00_DEM.tif`

```{note}
Tiles smaller than 1 MiB are treated as missing and downloaded again.
```

A tile already in the cache is never re-downloaded: `fetch_dem` logs a cache hit
and moves on. Missing tiles are downloaded into `cache/<tile_dir>/<filename>`
with up to three attempts. Downloads come from
`https://copernicus-dem-30m.s3.amazonaws.com` by default; pass `source_url=` to
`DEMManager` or set `FANINSAR_DEM_SOURCE_URL` to use a mirror.

(output-location)=

## Output location

`fetch_dem(bounds)` without an explicit path writes the mosaic to
`<cache parent>/dem/<name>`, where `<name>` defaults to `dem.tif` and can be
changed with `FANINSAR_DEM_NAME`:

```python
import os

os.environ["FANINSAR_DEM_CACHE_DIR"] = "./dem_cache"
os.environ["FANINSAR_DEM_NAME"] = "merged.tif"

from faninsar.processing.geometry import get_dem_manager

path = get_dem_manager().fetch_dem((97.5, 38.7, 101.4, 40.0))
print(path)  # <cache parent>/dem/merged.tif
```

The written GeoTIFF is float32, single-band, EPSG:4326, with `NaN` nodata, so it
can be opened directly by `RasterDEM` for sampling:

```python
from faninsar.processing.geometry import RasterDEM

dem = RasterDEM(path, interpolation="biquintic")
heights = dem.sample(latitudes, longitudes)
```

(use-it-automatically-in-the-pipeline)=

## Use it automatically in the pipeline

With `FANINSAR_DEM_CACHE_DIR` set, `run_pair(..., dem=None)` resolves the DEM by
itself: it determines the coverage from the ROI (or from the selected bursts
when no ROI is given), builds the mosaic at `<output_dir>/dem/<name>`, and uses
it for the pair.

```python
from faninsar.processing.pipeline import run_pair
from faninsar.query import BoundingBox

state = run_pair(
    reference,           # list of SAFE products (one or more frames)
    secondary,           # same frames for the secondary date
    output_dir="./pair",
    roi=BoundingBox(97.5, 38.7, 101.4, 40.0),  # optional
    dem=None,            # automatic DEM via the environment
)
```

```{tip}
Only the tile cache directory is required. When `FANINSAR_DEM_CACHE_DIR` is not
set, `run_pair` keeps its previous behavior.
```

The `faninsar frame` CLI exposes the same flow: pass `--dem name.tif` with a
bare file name to have it resolved (and built if missing) under
`<output>/dem/`, or omit `--dem` entirely and let `run_pair` resolve it:

```bash
faninsar frame --reference ref.zip --secondary sec.zip \
    --output ./pair --roi 97.5,38.7,101.4,40.0 --dem my_dem.tif
```

## Reference

| Symbol | Purpose |
| --- | --- |
| `DEMManager(cache_dir, source_url=...)` | Cache + query + download + mosaic in one class |
| `DEMManager.required_tiles(bounds)` | List `(tile_dir, filename)` pairs covering `bounds` |
| `DEMManager.fetch_dem(bounds, output_path=None)` | Ensure tiles, mosaic, return the output `Path` |
| `get_dem_manager()` | Manager from `FANINSAR_DEM_CACHE_DIR` (+ optional source/name vars) |
| `copernicus_tile_name(lat, lon)` | Tile identifier for one coordinate |
| `FANINSAR_DEM_CACHE_DIR` | Raw tile cache folder (required for the environment flow) |
| `FANINSAR_DEM_SOURCE_URL` | Download base URL override (default public AWS S3) |
| `FANINSAR_DEM_NAME` | Default output file name (default `dem.tif`) |

The module lives at `faninsar.processing.geometry.dem_manager`; see its
docstrings for the exact signatures.
