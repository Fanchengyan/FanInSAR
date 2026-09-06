# Processing provenance and dependency policy

This document defines the engineering provenance gate for FanInSAR processing
code. It is not legal advice. When a source has ambiguous or incompatible
terms, the conservative classification applies until the copyright holder or
qualified counsel provides written clearance.

The machine-readable authority for this policy is
`LICENSES/source_provenance.toml`. Every processing implementation must have a
ledger record before code is added. Reviewers must verify the record against
the source at the pinned revision; a permissive file header, a package-level
license, retyping, translation to another language, or a subprocess boundary
does not by itself establish permission.

## Status and implementation methods

- `permitted` means the stated implementation method may be used, subject to
  its attribution and testing obligations.
- `quarantined` means source text, structure, comments, constants, and close
  translations must not enter FanInSAR. Only black-box outputs may be retained
  as development oracles.
- `prohibited` means the component must not be copied, translated, vendored,
  imported, or made a runtime dependency of the MIT-licensed core.
- `audited_bsd_adaptation` requires a function-level provenance review and the
  BSD-3-Clause notice in `LICENSES/insardev_pygmtsar-BSD-3-Clause.txt`.
- `clean_room` means implementation from public specifications and papers by a
  contributor who does not consult quarantined implementation bodies while
  writing the code.
- `external_optional_dependency` means separately distributed code that is
  never vendored and is activated only by an explicit user choice.
- `oracle_only` permits black-box comparison in an isolated development
  environment; it never makes the oracle a normal test or runtime dependency.

## Mandatory quarantine list

The following sibling source is behavior-oracle material only. The complete
file/range list is repeated in the ledger so automated checks can enforce it.

- `core/insardev_pygmtsar/insardev_pygmtsar/utils_s1.py`: module lines 11-12;
  orbit/clock replacement lines 19-130; Doppler/orbit lines 133-146, 213 and
  258; burst parameter lines 316-337 and 362; and the additional
  GMTSAR-shaped conventions at 386, 397, 405, 469, 482, 490, 511, 782,
  800, 810, 817, 820, 866, 909, 921 and 1087.
- `core/insardev_pygmtsar/insardev_pygmtsar/utils_satellite.py`: radar/LLH
  geometry lines 1115, 1133 and 1194-1199; exact baseline algorithm lines
  1423-1424 and 1503-1655; `SAT_llt2rat` replacement lines 1922-1951; full
  baseline/phase algorithm marker at 2573; and retained compatibility code at
  3300 and 3326-3364.
- `core/insardev_pygmtsar/insardev_pygmtsar/PRM_gmtsar.py`: the
  `calc_dop_orb`/baseline family, especially lines 173-219, where the source
  states that it uses GMTSAR's exact algorithm.
- `core/insardev_pygmtsar/insardev_pygmtsar/utils_tidal.py`: lines 11-15 and
  the entire `solid_tide` translation derived from GMTSAR `solid_tide.c`.
- `core/insardev_pygmtsar/insardev_pygmtsar/S1_transform.py`: flat-earth and
  topographic-phase implementation at 1067-1073 and GMTSAR ordering/parameter
  behavior at 1139-1152.
- All algorithm bodies, comments, constants, and tests in `core/insardev`,
  including `utils_unwrap2d.py:1043-1162`,
  `Stack_unwrap2d.py:753-819`, `utils_goldstein.py:1-154`, and
  `Batch.py:1941-2035,2603-2679`. This component is under the
  InSAR.dev Source-Available License, not an OSI-approved license.

The list is intentionally broader than the minimum known markers. A future
audit may add paths immediately. Removing or narrowing an entry requires a
written provenance finding and ledger review.

## Allowed BSD adaptation

`insardev_pygmtsar` is distributed under BSD-3-Clause, but only a source file
and function with no quarantined lineage may be adapted. Before adaptation:

1. record the exact upstream repository, commit, file, symbol, and license;
2. search the symbol and its callees for translation, replacement, exact,
   compatibility, or copied-source markers;
3. reject the adaptation if any call path reaches a quarantined body;
4. retain the BSD copyright and license notice in source and binary
   distributions, and add a concise code-level attribution;
5. compare against an independent public specification as well as a frozen
   black-box oracle; and
6. update the ledger and pass the provenance checker before review.

Generic archive discovery and independently supportable metadata/container I/O
may qualify after that audit. Orbit/Doppler geometry, radar/geo transforms,
baselines, TOPS phase conventions, solid Earth tide, and flattening do not
currently qualify and must be clean-room implementations.

## Clean-room IRLS protocol

FanInSAR's internal IRLS unwrapper must be written from cited public literature,
not from `core/insardev`. The implementer records the papers/equations used and
works from an interface-and-behavior specification containing only inputs,
outputs, invariants, and tolerances. No `insardev` code, comments, identifiers,
control flow, constants, or tests may be consulted or reproduced during
implementation.

An independent verifier builds the test oracle from synthetic wrapped ramps,
residues, discontinuities, disconnected masks, low-coherence islands, and
black-box outputs. Required checks include gauge invariance, component
isolation, rewrap consistency modulo 2π, convergence/failure reporting,
pair-closure residuals, and CPU/accelerator equivalence. Oracle artifacts store
tool/version/configuration/checksums, never oracle source code. `insardev` is
absent from base and optional FanInSAR dependency graphs.

## Optional `snaphu-py` separation

`snaphu-py` is a separately distributed optional backend. Its Python wrapper is
offered under BSD-3-Clause OR Apache-2.0, while the bundled SNAPHU sources have
different terms and include portions that prohibit commercial use. FanInSAR
does not vendor either component, does not silently fall back to SNAPHU, and
does not represent installation of the wrapper as acceptance of the bundled
SNAPHU terms.

The base distribution must not depend on or import `snaphu`. A dedicated
`faninsar[snaphu]` extra may expose the backend only after packaging work adds
the dependency and the following disclosure surfaces are verified:

- install documentation shows the separate-license and commercial-use caveat
  before the command;
- backend discovery reports `available`, wrapper version, SNAPHU version, and
  `license_caveat` without importing it when absent;
- selecting the backend emits the caveat before processing and requires an
  explicit backend choice; and
- provenance attached to each result records wrapper/SNAPHU versions, cost
  mode, initialization, tiling, and configuration.

Until those packaging and capability checks exist, the ledger classification
is `quarantined`/`external_optional_dependency`: approved as a design, not yet
approved as a shipped dependency.

## Review and release gate

Every planned algorithm has `source`, `license`, `implementation_method`,
`attribution`, `test_oracle`, `status`, and `source_files` fields in the ledger.
A change is rejected if a field is absent, a quarantined path appears in added
implementation code, an oracle becomes a runtime dependency, required notices
are missing from built distributions, or runtime provenance cannot identify the
algorithm and backend actually used.

Release evidence must include a clean base-install dependency inspection, a
separate optional-backend inspection, the ledger checker result, wheel/sdist
notice inspection, and adversarial negative tests for missing fields,
incompatible licenses, forbidden paths, implicit fallback, and absent runtime
disclosure.

## Frozen radar/geo rebuild baseline

The executable authority for the radar/geo pipeline rebuild is
`tests/reference/pipeline_rebuild_manifest.yaml`. The manifest is deliberately
small and reviewable: it records the clean-room input corpus, the primary
specifications, behavior oracles, and quarantined diagnostic artifacts without
copying source code or generated products into the repository. Every local file
pin includes its absolute or checkout-relative path, byte count, and SHA-256.

The clean-room corpus is the fixed three-scene Sentinel-1 SLC set from
2016-12-07, 2016-12-31, and 2017-01-24, its three pairings, three precise
POEORB files, and twelve Copernicus DEM GLO-30 COG tiles. DEM input is
`EGM2008 orthometric height`; geometry consumers must use the corresponding
`WGS 84 ellipsoidal height` conversion with the EGM2008 geoid model. The
manifest also pins the Python, `uv`, PROJ, GDAL, SNAPHU, NumPy, rasterio,
pyproj, xarray, and Zarr environment used for the baseline.

Only the fixed campaign outputs are primary behavior oracles:

- ISCE2: `/Volumes/DATA2/TEST_sentinel-1/campaign/20161207_20161231_iw1_burst0/isce2/products`;
- InSAR.dev: `/Volumes/DATA2/TEST_sentinel-1/campaign/20161207_20161231_iw1_burst0/insardev/products`.

The former `out/oracle_isce2_from_slc/products` work directory is explicitly
forbidden because it was generated with a corrupt DEM VRT. The historical
flat/wrapped comparison identity is retained only as an unavailable quarantined
diagnostic; its bytes were already absent before PROPOSAL-0010 and it is not a
release oracle. A rebuild must reproduce the primary oracle artifacts by
behavior; it must never import, copy, or use oracle source code at runtime.

The baseline section preserves the starting Git identity and points at Waymark
`NOTE-0007`, the retirement inventory digest, and the encrypted archive digest.
The former execution store, session records, plans, and reports were retired by
PROPOSAL-0010 after all 584 entries were inventoried and a full quarantine
restore reproduced every byte. The manifest verifies the immutable retirement
identity rather than depending on deleted checkout-relative evidence paths.

The original out-of-scope inventory contained 1,025 entries. Its first recorded
digest was later invalidated because the underlying file had been overwritten;
that limitation remains historical context in the encrypted archive and is not
promoted into a current scientific assertion.

Validate the baseline through the public library driver before running a
processor:

```bash
uv run python -c "from pathlib import Path; from faninsar.io.storage.validation import validate_pipeline_rebuild_manifest; print(validate_pipeline_rebuild_manifest(Path('tests/reference/pipeline_rebuild_manifest.yaml')))"
```

The validator rejects malformed YAML, missing or changed pinned bytes, changed
Git HEAD/status artifacts, missing environment fields, stale oracle paths,
checksum mutations, and count mismatches. The focused regression surface is:

```bash
uv run pytest tests/reference/test_corpus.py tests/processing/test_provenance.py -q
```

Record the command output and the final out-of-scope hash comparison under the
Todo 0 evidence directory before claiming the baseline is frozen.
