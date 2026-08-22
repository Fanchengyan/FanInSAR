# PROPOSAL-0030 TDD Plan (accepted revision, 2026-08-22)

Normative source: `waymark/proposals/PROPOSAL-0030-multi-source-parallel-dem-manager.md`
(revision after ROUND-01M0EH1AWZHVKBENK89VHA03AX; re-review ROUND-01M0EHXCBZ4J3YNDNB3FGASPE6
4/4 recommend-accept). Worktree: `/Users/fancy/Documents/GitHub/FanInSAR-stac-waymark-worktrees/PROPOSAL-0030/PROPOSAL-0030/multi-source-parallel-dem-manager`
branch `wm/0030/multi-source-parallel-dem-manager`.

## Amendment (decision-owner, 2026-08-22, sidecar commit e4bfbf6d)

`DemSource` separates a PRODUCT group (`product_kind`
`Literal["dsm","dtm","merged-derived"]`, `resolution_m`, `vertical_datum`,
`derived`, `coverage`, `raster_open`) from a PROVIDER group (default base_url,
`auth` `Literal["none","token","registration"]` default `"none"`, `layout_id`
e.g. `"copernicus-cog-stem"` / `"skadi-hgt-gz"` / `"terrain-zxy"`); selection
level = `name`, `description`, `tiles(bounds)`, `fallback`. Registry-metadata
unit tests pin: product_kind copernicus-30/90 = dsm, srtm-skadi = dsm,
terrain-tiles = merged-derived; auth == none for all v1 entries. Behavior
unchanged (same five names, URLs, transport semantics).

## Test-first slices

Each slice: write the failing tests first, then implement until green.

### S1 — Source registry (`faninsar/processing/geometry/dem_sources.py`)

Tests (`tests/processing/geometry/test_dem_sources.py`):
1. Registry metadata for the four wired sources: names `copernicus-30`, `copernicus-90`,
   `auto`, `srtm-skadi`, `terrain-tiles`; resolution_m; vertical_datum literal values
   (`egm2008`, `mixed-derived`); derived flags.
2. URL builders: GLO-30 stem layout unchanged from PROPOSAL-0013;
   **GLO-90 stem `Copernicus_DSM_COG_30_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM` at the
   bucket root — no directory prefix** (regression-pins the R1 feasibility finding);
   skadi `skadi/{N|S}{YY}/{N|S}{YY}{E|W}{XXX}.hgt.gz`;
   terrain-tiles `geotiff/12/{x}/{y}.tif` XYZ orientation pinned at a known cell.
3. Tile enumeration over bounds incl. degree boundaries and per-tile minimum-bytes floors
   (GLO ≥1 MiB, skadi much smaller with ~25.9 MB expected-decompressed metadata,
   terrain z12 floor).
4. Coverage fail-closed message for out-of-coverage latitude (skadi 56°S–60°N).
5. `get_dem_source(name)` fail-closed error listing valid names;
   `list_dem_sources()` returns all five.

### S2 — Parallel transport (`faninsar/processing/geometry/dem_transport.py`)

Tests (`tests/processing/geometry/test_dem_transport.py`) — all mocked transport:
1. Concurrent fetch dispatches through a bounded pool (>1 worker structural assertion).
2. Retry matrix: 429/500/502/503/504 + ConnectionError/Timeout/ChunkedEncodingError retried
   with backoff; certificate-verification SSLError fails fast (distinct from transient
   handshake/EOF truncation which retries) — unwrap the reason chain.
3. Ranged mode: fake HEAD (size, Accept-Ranges), parallel chunk GETs written via os.pwrite
   into ftruncate-preallocated .part; short-chunk tail; final size == Content-Length;
   sha256 equality vs single-stream reference; **200-to-a-Range-request hard failure**;
   **mismatched Content-Range hard failure**.
4. Windows branch forced via fake transport: per-thread handle seek+write path exercised on
   POSIX too; shared stream budget: tiles+chunks ≤ max_workers total in-flight.
5. Atomicity: unique `{target}.{pid}-{uuid}.part` → os.replace publish; .part unlinked on
   failure; age-based sweep threshold above worst-case in-flight download.
6. Transport-boundary guard: cache-relative paths containing `..` or absolute components are
   rejected before any I/O.

### S3 — DEMManager reshaping (`dem_manager.py`)

Tests (extend `tests/processing/geometry/test_dem_manager.py`):
1. Constructor: `source=` name or DemSource; legacy flat-cache hit still resolves first;
   new downloads land under `cache_dir/<source-name>/`.
2. fetch_dem passes explicit `res=` to rasterio.merge (registry resolution_m); mixed-
   resolution auto mosaic keeps primary resolution; fallback-tile warning logged; mosaic
   GeoTIFF stamped with source name(s)+datum tags; void pixels (nodata −32768) mask to NaN.
3. auto semantics: control tile is a live HEAD against the configured base bypassing local
   cache; probe required tiles until one 200s, else canonical out-of-ROI known-present cell
   or proceed all-fallback under fraction warning — never misreport mirror misconfiguration;
   fallback fraction >25% warning; FANINSAR_DEM_SOURCE_URL applies to primary only (auto
   fallback uses default base).
4. get_dem_manager env parsing: FANINSAR_DEM_SOURCE (default copernicus-30), https-enforced
   base override rejection of non-https scheme, cross-source override warning.
5. Resumable-not-transactional semantics stated by test: completed tiles persist when a later
   tile fails; permanent 404 without fallback fails loud after retries.
6. flatten.copernicus_glo30_dem delegates to the shared lookup (results unchanged).

### S4 — Pipeline & CLI integration

Tests (`tests/processing/pipeline/…`, `tests/cli/…`):
1. `_resolve_auto_dem` helper used by run_pair, _run_pair_sweep, AND cli/frame.py bare-name
   path (no third GeoidAdjustedDEM construction site remains).
2. Datum-aware wrap: ellipsoidal source ⇒ no GeoidAdjustedDEM wrap (unit-level via registry);
   CLI geoid_correction default pinned explicitly so the unified helper cannot drift per site.
3. run_pair dem=None smoke with FANINSAR_DEM_SOURCE=auto writes pair/dem/dem.tif at primary
   resolution (existing PROPOSAL-0013 smoke re-targeted).

## Real-network verification matrix (opt-in marks, after green units)

- GLO-30 tile; GLO-90 withheld-cell N38E045 (stem layout live-verified 200+206);
- skadi N34E094.hgt.gz via /vsigzip/: finite heights, void→NaN, sequential read note;
- terrain-tiles z12 tile WarpedVRT→4326 finite heights after masking;
- Caucasus auto ROI: control validation → GLO-90 fallback tile fetched, mosaic stays 30 m;
- ranged single-tile fetch sha256 == plain-fetch sha256;
- same-run throughput gate: serial-vs-parallel ratio ≥3x expected (structural pool
  assertion + <20 min absolute bound as backstops);
- run_pair(roi=..., dem=None) FANINSAR_DEM_SOURCE=auto pipeline smoke.

## Quality gates

ruff check + format clean on changed files; tests/processing suite green;
Windows-compat import smoke for dem_transport; every commit cites PROPOSAL-0030.
