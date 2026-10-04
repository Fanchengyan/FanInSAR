# P0049 independent manual QA

Date: 2026-09-10
Verifier: independent QA executor
Product worktree: `/Users/fancy/Documents/GitHub/FanInSAR-stac`
Proposal: `PROPOSAL-0049`

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| S1 | Verification plan: query tests and public Stack seam | Python API/tests | `uv run pytest tests/stack/test_search.py tests/stack/test_input_manifest.py tests/stack/test_lifecycle_0037.py tests/stack/test_session.py tests/remote/test_asf_search.py tests/remote/test_remote.py -q --override-ini='addopts='` | PASS (71 passed, 8 warnings) | A1 |
| S2 | Verification plan: S1 local preparation regression | Python tests | `uv run pytest tests/missions/s1/processing/test_production.py -q --override-ini='addopts='` | PASS (25 passed, 2 skipped, 1 warning in 3.13 s) | A2 |
| S2-live | Verification plan: S1 remote extraction regression | Python tests against the configured remote fixture | `uv run pytest tests/missions/s1/test_safe.py tests/missions/s1/test_orbit.py tests/missions/s1/test_remote_extract.py tests/missions/s1/test_burst_extract.py -q --override-ini='addopts='` | FAIL (the live burst extraction test stalled in a remote HTTPS range read; after 125.33 s, 3 passed and 2 skipped but no complete test run could be obtained, so this is not a PASS) | A10 |
| S3 | Verification plan: package lint for changed implementation | Static analysis | `uv run ruff check faninsar/stack faninsar/missions/s1 faninsar/remote` | FAIL (ASF provider has RET503 and E501; command also auto-fixed two import-order findings in pre-existing test files) | A3 |
| S4 | Verification plan: valid manifest reuse | Python API | Constructed `Stack._from_safes` with two SAFE-like directories, called `prepare()` twice against one work directory, and inspected `input_manifest.json` plus in-memory preparation fields. | FAIL | A4 |
| S5 | Verification plan: S1 reopen and manifest validation | Python API | Created a valid manifest, changed a source-directory binding digest, then called `S1Stack.open(work_dir)`. | FAIL | A5 |
| S6 | Verification plan: concrete S1 search profile | Python API | `S1Stack._search_profile()` and `S1Stack.search('not-a-query')` | PASS for profile presence; query invocation correctly rejected missing CRS | A6 |
| S7 | Public package importability | Python package API | `uv run python -c 'import faninsar; print("imported")'` and `uv run python -c 'from faninsar import remote; print("remote imported")'` | FAIL: both fresh-process imports raise a circular-import `ImportError` through `io.export.kmz -> processing -> missions.s1.search -> stack.session -> ... -> io.export.kmz`. | A11 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
| --- | --- | --- | --- | --- | --- |
| A1 | Manifest validation and re-use | Changed `reference`, `coreg_mode`, `multilook`, and ROI in a new Stack using the same work directory | Existing manifest must not be silently reused for a changed scientific request; preparation state should be rebuilt or a visible mismatch raised. | FAIL: old manifest was reused; file remained byte-identical and new Stack had `_mission_prepared is None`. | A4 |
| A2 | Manifest source identity | Changed contents inside a SAFE directory after manifest publication | Source identity must detect changed scientific input or explicitly require refresh. | FAIL: directory identity is only `{path, kind}`; changed file was accepted and manifest remained unchanged. | A4 |
| A3 | `S1Stack.open()` manifest contract | Tampered source checksum in a valid-shape manifest | Open must validate the manifest and reject the mismatch. | FAIL: `S1Stack.open()` accepted the tampered checksum because the recorded source was a directory and directory contents are not hashed. | A5 |
| A4 | `S1Stack.open()` lifecycle | Reopen a valid local manifest | Open should validate bindings and run the same preparation path, restoring a usable prepared Stack. | FAIL: returned Stack had `_prepared=False` and `_mission_prepared is None`. | A5 |
| A5 | Mission preparation seam | S1 SAFE preparation requiring external mission inputs | S1 hook should implement official POEORB/RESORB selection and applicable AUX_CAL/EAP preparation. | FAIL: current `_prepare_mission()` only reads SAFE footprints and returns `provenance={'mission': 'sentinel-1'}`; no orbit/calibration bindings are produced. | A7 |
| A6 | Stable ASF registration | Existing `sentinel1-asf` registration with a differing identity | Registry must compare the complete immutable identity and reject conflicts. | FAIL: `ensure_asf_catalog()` checks only provider name and returns an existing ASF registration without comparing collection/auth/delivery/endpoint identity. | A8 |
| A7 | Complete candidate traversal | Provider emits more records than public `limit` | Search must consume the complete bounded traversal before canonical ordering, limit, and mission selection. | FAIL by code-path inspection: `remote.search()` breaks when `len(results) >= limit`, and ASF adapter is called with `maxResults=limit`. | A9 |
| A8 | Public import boundary | Fresh interpreter import | The public `faninsar` root and `from faninsar import remote` must import successfully. | FAIL: circular import described in S7. | A11 |

## artifactRefs

| id | kind | description | path |
| --- | --- | --- | --- |
| A1 | test-output | Focused P0049/Stack/remote pytest run: 71 passed, 8 warnings. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A2 | test-output | Sentinel-1 production focused regression: 25 passed, 2 skipped. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A3 | lint-output | Changed implementation lint failed with ASF provider RET503 and E501; two unrelated import-order edits were made by the lint invocation. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A4 | manual-run | Same-workdir changed reference/config/ROI and SAFE-directory-content reproduction. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A5 | manual-run | S1 open with unknown-field and tampered-directory-binding checks; unknown field rejected, tampered directory binding accepted, reopened Stack unprepared. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A6 | manual-run | Concrete S1 profile exists; missing-CRS query rejected at the remote boundary. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A7 | source-inspection | Current S1 `_prepare_mission` implementation and returned bindings. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/faninsar/missions/s1/stack.py:318` |
| A8 | source-inspection | Current ASF catalog registration helper. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/faninsar/missions/s1/search.py:181` |
| A9 | source-inspection | Current remote result loop and ASF request cap. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/faninsar/remote/catalog.py:267` |
| A10 | test-output | Live S1 extraction stalled in `requests` redirect/range read after 125.33 s and was interrupted; no complete PASS verdict is claimed. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |
| A11 | test-output | Fresh-process package imports fail with circular-import traceback. | `/Users/fancy/Documents/GitHub/FanInSAR-stac/.omo/evidence/p0049-verification/p0049-manual-qa.md` |

## Evidence details

Focused test output:

```text
.......................................................................  [100%]
71 passed, 8 warnings in 4.62s
```

Changed-input reproduction:

```text
M1 initial_manifest True prepared True mission_state True
M2 changed_reference_config_roi_reused True reference 20240113 coreg_mode geometry multilook (1, 1) roi True mission_state False
M3_changed_directory_contents_reused True measurement_exists True manifest_unchanged True
M4_open_returns_unprepared False mission_state False manifest_exists True
```

The findings above are scientific stale-input and lifecycle defects, not hypothetical hostile-provider cases. No product code or Waymark status was changed by this verification.
