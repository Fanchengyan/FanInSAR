# PROPOSAL-0030 TDD Plan v2 (provider-axis design, 2026-08-23)

Normative source: `waymark/proposals/PROPOSAL-0030-multi-source-parallel-dem-manager.md`
(ratified provider two-axis revision, checkpoints through 92ae7ae8; council MINUTES-r2).
Worktree: `/Users/fancy/Documents/GitHub/FanInSAR-stac-waymark-worktrees/PROPOSAL-0030/PROPOSAL-0030/multi-source-parallel-dem-manager`
branch `wm/0030/multi-source-parallel-dem-manager`.

NOTE: the worktree contains partial S1–S4 commits from the earlier (pre-reversion)
implementation pass. Reconcile: keep reusable scaffolding, re-align names and
structure to THIS plan; every test must pass against the current ratified design.

## Test-first slices

### S1 — Transport engine (`faninsar/processing/geometry/dem_transport.py`)

Tests (`tests/processing/geometry/test_dem_transport.py`, mocked transport):
1. FetchPlan union: TileSet(allowed_hosts) / Artifact(scheme https|ftp, members,
   member_pattern, expand, credential_ref); grammar rejects unknown fields.
2. URL host-pinning: every https Tile/Artifact URL host must be in allowed_hosts —
   mismatch is a hard error before connect (CMR-injection regression pin).
3. Cross-host redirect: followed WITHOUT credentials; auth attached cross-host is a
   hard error.
4. Retry matrix: 429/500/502/503/504 + ConnectionError/Timeout/ChunkedEncodingError/
   urllib3 ProtocolError retried with capped jitter; Retry-After honored;
   certificate-verification errors fail fast (distinct unit tests);
   401/403 are terminal loud errors (named test).
5. Ranged mode: fake HEAD Accept-Ranges → 8 MiB chunks, per-chunk status==206 AND
   matching Content-Range else hard error; short-chunk tail; os.pwrite assembly;
   Windows per-thread seek+write branch forced via fake transport (runs on POSIX);
   final size == HEAD Content-Length.
6. Ranged exclusion for Earthdata hosts (HEAD/GET divergence) — always whole-file.
7. FTP branch via urllib ftp://; no Content-Length → floor + post-extract checks.
8. Zip expand: central-directory CRC verified; resolved-path-within-staging
   containment on every member before write (zip-slip hostile payload named test);
   only members/member_pattern extracted; staging dir covered by age sweep.
9. Unique {target}.{pid}-{uuid}.part → os.replace publish; orphan sweep.
10. Credential hygiene (caplog assertions): no netrc/token/SAS sig/se content in
    logs, dumps, exceptions, cache paths, GeoTIFF tags.

### S2 — Source registry (`faninsar/processing/geometry/dem_sources.py`)

Tests (`tests/processing/geometry/test_dem_sources.py`):
1. Seven shape classes instantiate; abstract surface = plan() only; zero fetching.
2. Selection grammar "product" / "product:provider": fail closed on unknown product,
   unknown provider (lists valid), registered-but-unwired pair (cites wired status),
   hostile payloads (`glo30:../..`, trailing colon, homoglyphs, whitespace);
   charset validation on raw-DemSource name escape hatch.
3. Closed valid-pair matrix resolves exactly as the Goals table; dem_catalog()
   returns structured matrix (providers/default/auth/wired/datum/resolution) with a
   zero-network unit assertion (import/list/get/catalog perform no socket I/O).
4. LatLonGridSource: glo30/glo90 stem layouts (COG_10_/COG_30_, no directory prefix),
   skadi layout + small floors + ocean-404 skip flag scoped to skadi.
5. PgcQuadSource: QuadEnumerator protocol; v1 listing implementation loops until
   IsTruncated=false, reconciles bounds→expected-quad set, errors on gaps; quad-grid
   mapping pinned; coverage fail-closed outside polar bands.
6. TerrainPyramidSource: z12 XYZ orientation pin; derived=true registry-wide warning.
7. FtpZipSource: alos-dem@jaxa-ftp Artifact(scheme="ftp", expand="zip").
8. AuthenticatedGranuleSource: nasadem@earthdata (CMR C2763264762-LPCLOUD, EGM96,
   60N–56S coverage fail-closed) and nisar-glo30@earthdata (C3803703055-ASF,
   ellipsoidal no-wrap, unconditional EPSG4326 title filter + `-vrt` exclusion);
   URS-302→200-HTML magic-byte hard error (named test).
9. PcStacSource: sign_inplace modifier flow (mocked STAC); optional [pc] extra
   fail-closed with install guidance when planetary_computer/pystac_client absent.
10. MosaicRecipe dataclass drives all mosaic-side behavior (gdal_open prefix,
    source_crs/warp target/resampling/nodata/mask_to_nan) — manager reads recipe
    only; per-entry recipes for all 14 names asserted.

### S3 — DEMManager reshaping (`dem_manager.py`)

Tests (extend `tests/processing/geometry/test_dem_manager.py`):
1. Constructor source="glo30"/"glo30:pc"/DemSource; resolution kwarg > env >
   default (env FANINSAR_DEM_SOURCE carries compound value); auto + non-default
   provider rejected loudly; auto = glo30 with control-tile-guarded GLO-90 fallback
   (control independent of ROI withheld set; >25% fraction warning).
2. Cache partitions cache_dir/<product>-<provider>/; legacy flat probe bound to
   glo30@aws only; partition dirs opaque labels never parsed back.
3. fetch_dem: explicit merge res= (no silent downgrade), fallback warning,
   provenance tags product+provider+datum+date+host (advisory-only rule +
   contradiction-warning-use-after-warning semantics), resumable-not-transactional.
4. Structured DEMProviderUnavailableError with same-product alternatives
   (excluding unwired v1 providers) + cache-hit hints.
5. get_dem_manager env parsing incl. FANINSAR_DEM_SOURCE compound + SOURCE_URL
   override rules (https enforced, primary-only, non-default warning).
6. flatten.copernicus_glo30_dem delegates to shared lookup unchanged.

### S4 — Pipeline & CLI integration

1. Public shared helper (rename _resolve_auto_dem) used by run_pair,
   _run_pair_sweep, and cli/frame.py bare-name path; single datum-aware wrap rule
   everywhere; CLI geoid_correction default pinned.
2. CLI --dem-source accepts compound values; --dem-source glo30:ot fails closed
   citing unwired status (v1).

## Ordered landing

Transport engine first (S1+S2 core), then AWS shapes, then PcStacSource (+[pc]
extra in pyproject with requests>=2.33.0 runtime floor), then Earthdata/nisar-glo30
last. Real-network matrix (opt-in marks): one tile per wired source incl. GLO-90
withheld-cell N38E045 fallback staying 30m, PC glo30/glo90/nasadem/alos-dem signed
opens, ArcticDEM/REMA 32m quads reprojection, AW3D30 FTP zip extraction + ~1m
agreement vs GLO-30, ranged-vs-plain sha256 equality, same-run serial-vs-parallel
>=3x ratio (<20min absolute bound). Pipeline smoke deferred to verification
(Sentinel-1 data dependency). Quality gates: ruff clean, tests/processing green,
Windows import smoke; commits cite PROPOSAL-0030.
