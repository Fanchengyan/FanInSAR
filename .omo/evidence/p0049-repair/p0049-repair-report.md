# PROPOSAL-0049 revision 14 repair evidence

Date: 2026-09-10

Waymark lifecycle state was not changed by this repair.

## Scientific scenarios

1. Complete bounded candidate traversal
   - Invocation: `uv run pytest tests/remote -q --override-ini='addopts='`
   - Observable: `112 passed, 1 skipped`.
   - Specific coverage: `test_remote_search_fetches_terminal_page_before_applying_public_limit` proves a public limit of one still consumes ASF's terminal page; CMR now raises `incomplete_results` when its operation item budget is reached while a Search-After token remains.

2. Canonical ASF Sentinel-1 collection identity
   - Invocation: the combined focused test command below.
   - Observable: `test_asf_scene_registration_uses_canonical_collection_concept` resolves the scene registration to `C1214470488-ASF`; registration compares the complete local adapter identity before reuse.

3. Complete SAFE boundary
   - Invocation: the combined focused test command below.
   - Observable: complete Burst Extractor ZIP assets retain the download path. `test_from_search_rejects_tiff_xml_without_burst_extractor_safe` proves TIFF/XML-only input fails before any component download. `test_from_search_rejects_ambiguous_safe_zips_before_download` proves two complete ZIPs remain an error even when TIFF/XML components are present.

4. Sentinel-1 search and polarization identity
   - Invocation: the combined focused test command below.
   - Observable: S1 search normalizes and selects through `MissionSearchProfile`; selected product type and polarization are included in manifest processing identity and restored by `S1Stack.open()`.

5. Official Sentinel-1 orbit and early IPF behavior
   - Invocation: the combined focused test command below.
   - Observable: `test_orbit_selection_prefers_covering_poeorb` proves a covering POEORB is selected before RESORB. Unknown EOF types and non-covering configured candidates are rejected. `test_ipf_002_36_is_detected_from_safe_manifest` proves the ISCE2 condition is read from `manifest.safe`; `S1Stack` raises a clear unsupported error because EAP correction is not implemented, preventing scientifically incorrect silent processing.

6. Preparation failure and manifest identity
   - Invocation: the combined focused test command below.
   - Observable: mission preparation runs before shared preparation; failures leave `_prepared=False` and publish no manifest. Directory SAFE identities recursively hash relative names and bytes. Product, polarization, mission, ROI, grid, DEM, mask, reference, coregistration and multilook identities participate in reuse validation.

7. Ambiguous CMR representation
   - Invocation: the combined focused test command below.
   - Observable: `test_cmr_rejects_ambiguous_complete_data_representations` raises `ambiguous_representation`; duplicate links to the same canonical URL remain one representation.

8. Stable ASF registration
   - Invocation: the combined focused test command below.
   - Observable: burst and scene searches share the private `sentinel1-asf` name mapped to `C1214470488-ASF`; dataset and processing-level filters retain their product-family distinction.

## Revision 14 focused regression

Invocation:

`uv run pytest tests/remote/test_asf_search.py tests/remote/test_cmr_catalog.py tests/remote/test_complete_stream.py tests/remote/test_remote.py tests/stack/test_search.py tests/stack/test_input_manifest.py tests/missions/s1/test_stack_search.py -q --override-ini='addopts='`

Observable: `97 passed` in 1.84 seconds.

## Static checks

Invocation: scoped `ruff check` and `ruff format --check` over the seven revision-14 product files, followed by `git diff --check`.

Observable: Ruff reported `All checks passed!` and `7 files already formatted`; `git diff --check` produced no output.

## Evidence limitation

These are deterministic local tests. `/Volumes/DATA2/TEST_sentinel-1` is absent and the live CloudFront path ended with an SSL EOF in independent verification. This repair does not claim a successful real provider download or Verified status.
