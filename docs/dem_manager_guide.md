# Multi-source DEM Manager Guide (PROPOSAL-0030)

This guide covers the automatic DEM manager introduced by PROPOSAL-0030:
fourteen selectable DEM products, a manual provider axis, bandwidth-saturating
parallel downloads, and the structured outage contract. All examples assume
`FANINSAR_DEM_CACHE_DIR` points at a writable directory (see
[Environment variables](#environment-variables)).

## The two axes: product and provider

A DEM is selected on **two independent axes**:

- **Product** — what the data is (Copernicus GLO-30, NASADEM, ArcticDEM, ...).
- **Provider** — where it is fetched from (AWS, Planetary Computer, Earthdata,
  JAXA FTP, ...).

Defaults are always **anonymous cloud channels** (AWS or Planetary Computer):
`DEMManager(cache_dir, source="glo30")` needs no account, no token, no API
key. Switching provider is **manual and explicit** — the manager never
silently switches product or provider, and `auto` never crosses the provider
axis. When a provider fails, you get a structured
`DEMProviderUnavailableError` (see [Outages](#outage-handling)) instead of a
hidden fallback.

## Quick start

```bash
export FANINSAR_DEM_CACHE_DIR=/path/to/dem-cache   # required for get_dem_manager
```

```python
from faninsar.processing.geometry.dem_manager import get_dem_manager

manager = get_dem_manager()                 # glo30 @ aws, the default
dem_path = manager.fetch_dem((99.0, 37.5, 101.0, 38.5))
# -> Path to a VRT/GeoTIFF mosaic covering the (min_lon, min_lat, max_lon, max_lat) bounds
```

Tiles are downloaded once into
`$FANINSAR_DEM_CACHE_DIR/<product>-<provider>/` and reused on every later
call; only missing tiles are fetched, with concurrent tile downloads and
ranged-chunk transfer for large files (see
[Performance knobs](#performance-knobs))).

To pick a different product, pass the selection explicitly:

```python
manager = get_dem_manager(source="nasadem")          # NASADEM @ Planetary Computer
manager = get_dem_manager(source="alos-dem")         # AW3D30 @ Planetary Computer
manager = get_dem_manager(source="arcticdem-2")      # 2 m ArcticDEM @ AWS (PGC quads)
```

## The fourteen selection names

`list_dem_sources()` returns all of them:

| Selection | Resolution | Vertical datum | Provider (default) | Auth | Notes |
|---|---|---|---|---|---|
| `glo30` | 30 m | EGM2008 | aws | none | **Default.** Copernicus GLO-30 COG tiles |
| `glo90` | 90 m | EGM2008 | aws | none | Copernicus GLO-90 |
| `auto` | 30 m | EGM2008 | aws | none | GLO-30 with same-family GLO-90 rescue for withheld cells |
| `nasadem` | 1″ | EGM96 | pc | none | NASADEM via Planetary Computer |
| `alos-dem` | 1″ | EGM96 | pc | none | ALOS World 3D-30m |
| `srtm-skadi` | 1″ | EGM96 | aws | none | SRTM `.hgt.gz` from AWS Open Data |
| `terrain-tiles` | ~38 m | mixed | aws | none | AWS terrain-tiles pyramid (derived; quality varies) |
| `arcticdem-10` / `-32` / `-2` | 10/32/2 m | ellipsoidal | aws | none | ArcticDEM tiers (PGC quads) |
| `rema-10` / `-32` / `-2` | 10/32/2 m | ellipsoidal | aws | none | REMA tiers (PGC quads) |
| `nisar-glo30` | 30 m | **ellipsoidal** | earthdata | Earthdata Login | NISAR mission DEM, re-referenced to WGS84 |

Datum matters: `DEMManager.vertical_datum` reports `"egm2008"`, `"egm96"`,
`"ellipsoidal"`, or `"mixed-derived"`. The pipeline entry point
`resolve_auto_dem` wraps orthometric sources into a geoid-adjusted sampler
automatically and **never** wraps ellipsoidal sources such as `nisar-glo30`,
`arcticdem-*`, or `rema-*`.

## Switching provider manually

Selection grammar is `"<product>"` or `"<product>:<provider>"`. v1 wires one
default channel per product plus three alternative identities:

```python
manager = get_dem_manager(source="nasadem:earthdata")  # NASADEM full-quality granules via Earthdata
manager = get_dem_manager(source="alos-dem:jaxa-ftp")  # AW3D30 via anonymous JAXA FTP zips
```

Provider prerequisites:

- **aws** — nothing; anonymous S3/HTTPS.
- **pc** — anonymous by default. Install the extra:
  `pip install "faninsar[pc]"` (adds `planetary-computer`, `pystac-client`).
  If anonymous throttling bites, Planetary Computer signing is applied
  automatically when available.
- **earthdata** — free NASA Earthdata Login account. Put credentials in
  `~/.netrc` (`machine urs.earthdata.nasa.gov login ... password ...`) or set
  `EARTHDATA_TOKEN`. A redirect-to-login page is detected and reported as a
  failure, never stored as data.
- **jaxa-ftp** — nothing; anonymous FTP with multi-block zips.

Unknown product/provider names fail closed at construction time — before any
network traffic.

## What `auto` does (and does not do)

`auto` = GLO-30 with a same-family rescue: when a GLO-30 cell is withheld
(e.g. the Caucasus N38–41 E045/046 404s), only the missing cells are refetched
from **GLO-90 on the same AWS channel**, and the mosaic is resampled back to
30 m. It never switches provider, never switches product family, and cannot
be combined with an explicit provider (`"auto:pc"` is rejected loudly).

## Python API summary

```python
from faninsar.processing.geometry.dem_manager import (
    DEMManager, get_dem_manager, default_dem_name,
)
from faninsar.processing.geometry.dem_sources import (
    list_dem_sources, parse_selection,
)

list_dem_sources()          # the 14 names
parse_selection("nasadem:earthdata")   # -> DemSource (product/provider/datum/...)

manager = DEMManager(
    cache_dir=Path("/path/to/cache"),  # explicit root; get_dem_manager reads the env instead
    source="glo90",                    # None -> FANINSAR_DEM_SOURCE -> "glo30"
    max_workers=8,                     # concurrent download streams
    chunked_threshold=4,               # below this many missing tiles, use ranged chunks
    base_url=None,                     # https mirror override (not for FTP/Earthdata)
)

manager.source_entry          # resolved DemSource (name, product, provider, datum, ...)
manager.vertical_datum        # "egm2000" / "egm96" / "ellipsoidal" / "mixed-derived"
dem_path = manager.fetch_dem(bounds, output_path=Path("out/dem.tif"))
```

Pipeline code should prefer the shared entry point, which applies the
datum-aware wrap rule for you:

```python
from faninsar.processing.pipeline.production import resolve_auto_dem

sampler = resolve_auto_dem(
    bounds,                       # (min_lon, min_lat, max_lon, max_lat)
    output_dir="out",             # mosaic lands at out/dem/<name>.tif
    geoid_correction=True,        # wrap orthometric sources; ellipsoidal stay unwrapped
    dem_source="nasadem",         # None -> FANINSAR_DEM_SOURCE -> glo30
)
```

## CLI

```bash
faninsar frame REF SEC --roi ... --dem auto-dem.tif --dem-source nasadem
```

`--dem-source` accepts any of the 14 names (including `auto`); unknown names
are rejected in a pre-gate before any pipeline work starts. The built mosaic
is pinned to the compute device by the pipeline (GPU biquintic when CUDA is
selected, per PROPOSAL-0032).

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `FANINSAR_DEM_CACHE_DIR` | Tile cache root (required for `get_dem_manager`) | — |
| `FANINSAR_DEM_SOURCE` | Default selection when none is passed | `glo30` |
| `FANINSAR_DEM_NAME` | Mosaic file name | `dem.tif` |
| `FANINSAR_DEM_SOURCE_URL` | https base-URL override for the selected source | provider default |
| `FANINSAR_OT_API_KEY` | OpenTopography API key (reserved; `ot` not wired in v1) | — |
| `EARTHDATA_TOKEN` | Earthdata bearer token alternative to `~/.netrc` | — |

An explicit argument always wins over the environment; the environment always
wins over the built-in default.

## Outage handling

A failing provider raises `DEMProviderUnavailableError` with structured
fields — no hidden cross-provider fallback:

```python
except DEMProviderUnavailableError as err:
    err.product           # "glo30"
    err.provider          # "aws"
    err.host              # the endpoint that failed
    err.failure_class     # "upstream-outage" | "forbidden" | ...
    err.attempts          # retry count actually spent
    err.alternatives      # {"nasadem:earthdata": "full-quality granules", ...}
```

`alternatives` lists the *manual* escape hatches you may switch to; nothing
switches on its own. Unwired alternatives (OpenTopography in v1) are excluded
from this list by design.

## Performance knobs

- `max_workers` (default 8) — concurrent streams for tile fan-outs. Raise it
  on fat links; the transport reuses one session per host and honors
  `Range:`/`Content-Range` per chunk.
- `chunked_threshold` (default 4) — when fewer than this many tiles are
  missing, the engine prefers ranged-chunk download of big single artifacts
  over many small requests.
- Cache partitions are per `<product>-<provider>`, so switching selections
  never invalidates other channels' tiles. Legacy flat GLO-30 layouts (from
  PROPOSAL-0013) are still recognized as `glo30@aws` hits.

## Origin

Design and contracts are governed by **PROPOSAL-0030** (registered Waymark;
extends PROPOSAL-0013). The fourteen-name matrix, provider axis, `auto`
same-family semantics, fail-closed selection, structured outage error, and
parallel transport budgets are normative — changes must go through a new
proposal.
