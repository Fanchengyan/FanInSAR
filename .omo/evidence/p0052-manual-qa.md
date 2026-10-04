# P0052 independent manual QA

Date: 2026-09-10

Verdict: **PASS**. Post-fix focused and expanded suites pass, canonical scene/IFG/unwrap/TimeSeries artifacts round-trip through STAC+Zarr, Stack Network publication no longer uses NPY, and the three prior findings are resolved. No product code or Waymark lifecycle state was changed by this verification.

## surfaceEvidence

| Scenario | Criterion | Surface / exact invocation | Verdict | artifactRefs |
|---|---|---|---|---|
| S1 | 2, 3 | Python API: `network.invert()` and `network.invert(model=LinearModel, pairs=...)`; `uv run pytest tests/network/test_network_invert.py -q --override-ini='addopts='` | PASS | A1 |
| S2 | 6, 7, 9, 10, 11, 12 | Python API: `TimeSeries.export(path, exporter=...)`; `uv run pytest tests/timeseries/test_export.py -q --override-ini='addopts='` | PASS (managed test environment) | A1 |
| S3 | 4, 5, 14, 16 | Python API: scene/IFG/TimeSeries writes followed by validated open/readback; direct offline smoke | PASS for scene, IFG, and TimeSeries roots | A2 |
| S4 | 8, 13 | Filesystem/STAC inspection of readback roots: `item.json`, `stac-item.json`, `CURRENT`, Zarr generations and metadata | PASS | A2 |
| S5 | 1, 14, 16 | Stack API: `stack.unwrap(); stack.network.invert()` and affected stack tests | PASS for no-publish workflow | A1 |
| S6 | 4, 14, 16 | Stack API: `stack.publish_generation(timeseries_root)` in `tests/stack/test_session.py::test_publish_generation_exports_canonical_network_view` | PASS: Network publication uses `.zarr` assets and the test passes | A3 |
| S7 | 5, 16 | Scene API: `CoregisteredSceneStore.open()` after `write_scene_unit(... phase_screen=...)` in flatten-stage tests | PASS: phase-screen Zarr directory is validated and read back | A4 |
| S8 | 5 | NISAR geo tiled lifecycle: `uv run pytest tests/missions/nisar/test_provider.py::test_nisar_three_date_multitile_radar_and_projected_geo_lifecycle -q --override-ini='addopts='` | PASS: common scene transform is persisted; local tile geometry remains lineage | A5 |

## adversarialCases

| Scenario | Criterion | Adversarial class | Expected behavior | Verdict | artifactRefs |
|---|---|---|---|---|---|
| A1 | 7 | radar default export | `exporter=None` must select COG and reject radar without fallback | PASS | A1 |
| A2 | 9 | existing destination, `overwrite=False` | Raise `FileExistsError` before modification | PASS | A1 |
| A3 | 10 | existing destination, `overwrite=True` | Replace logical destination without user-visible publish/version choice | PASS | A1 |
| A4 | 12 | STAC-last failure | Failed STAC publication must not expose an item referencing incomplete assets | PASS | A1 |
| A5 | 5, 7 | incomplete geo metadata / guessed geometry | Reject COG unless geo CRS+transform are complete; never geocode or guess | PASS | A1 |
| A6 | 4 | NPY boundary | No NPY in canonical scene/IFG/unwrap/TimeSeries products | PASS | A3 |
| A7 | 16 | canonical re-open | Reopen persisted Zarr generation through validation and CURRENT | PASS | A2 |
| A8 | 6 | exporter registry | Built-ins resolve as `('netcdf', 'stac-cog', 'stac-zarr')`; custom class/callable receives path and overwrite | PASS | A1 |

## Findings

No actionable findings remain.

### Resolved P1 — Stack Network publication NPY boundary

The post-fix implementation at `faninsar/stack/session.py:3757-3895` copies validated `.zarr` payloads and records `.zarr` asset locations. The direct regression test now passes.

### Resolved P1 — Scene phase-screen Zarr readback

`faninsar/io/storage/scene_store.py:369-383` now validates a Zarr directory and reads it through `read_zarr_array()`. Flatten-stage tests and direct phase-screen readback pass.

### Resolved P2 — NISAR tiled Geo metadata

`faninsar/missions/nisar/provider.py:1569-1627` persists the common target CRS/transform and records local tile geometry in lineage. The three-date radar/projected-Geo lifecycle passes.

### Resolved P2 — Formatting

`uv run ruff check faninsar` and `uv run ruff format --check faninsar` both pass.

### Intentional non-canonical NPY references

The static audit finds only `_LEGACY_IFG_FILENAMES` in `faninsar/io/storage/ifg_store.py`, which is quarantine-only detection for rejecting old direct payload layouts, and NPY diagnostic assets in `faninsar/io/storage/ion_store.py`. Neither is a canonical scene, IFG, unwrap, Stack Network, or TimeSeries reopen/export boundary.

### Observation — legacy solver selector remains exposed

`faninsar/network/network.py:804-831` retains `analyze_time_series(solver="sbas"|"nsbas")` as a compatibility alias. The new public `Network.invert(model=...)` path correctly uses model presence as the discriminator, and all focused tests pass. This is not counted as a blocking finding because the accepted proposal explicitly preserves existing seams, but it is worth deprecating if the no-solver-selector rule is intended to be strict for all public entry points.

## Artifacts

| id | kind | description | path |
|---|---|---|---|
| A1 | test-log | Focused P0052 tests: 91 passed; includes inversion, exporter registry, default COG, radar rejection, overwrite and STAC-last | [.omo/evidence/p0052-verification-readback-20260910.txt](/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0052-verification-readback-20260910.txt) |
| A2 | readback-log | Real offline scene, IFG, canonical TimeSeries STAC+Zarr writes and validated generation readback | [.omo/evidence/p0052-verification-readback-20260910.txt](/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0052-verification-readback-20260910.txt) |
| A3 | test-log | Post-fix Stack Network publication and live `.zarr` asset-location assertions | [.omo/evidence/p0052-verification-readback-20260910.txt](/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0052-verification-readback-20260910.txt) |
| A4 | readback-log | Post-fix flatten-stage phase-screen Zarr directory readback | [.omo/evidence/p0052-verification-readback-20260910.txt](/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0052-verification-readback-20260910.txt) |
| A5 | test-log | Post-fix NISAR projected-geo common transform lifecycle | [.omo/evidence/p0052-verification-readback-20260910.txt](/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0052-verification-readback-20260910.txt) |
