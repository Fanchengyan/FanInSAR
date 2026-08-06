# PROPOSAL-0016 Gate Review — cf4a31c

## recommendation

APPROVE

## blockers

None.

## originalIntent

Make `run_pair` the single unified pair-processing entrypoint. It must preserve the legacy single-config radar path, accept one multilook pair as a tuple or integer list, accept an iterable of pairs as a sweep, run the expensive shared prefix once, support radar and geo coregistration, emit isolated look-qualified outputs, and return the contractually correct state/result type with complete lightweight outcome metadata.

## desiredOutcome

- A single pair returns `ProductionPairState`; an iterable, including a one-element iterable, returns `ProductionPairSweepResult`.
- Geo mode requires `geo_grid` and supports the proposal's geo/unwrap/resource parameters.
- Sweep outputs use `looks_{az}x{rg}/<pair_id>.zarr`, look-qualified STAC ids, completion manifests, per-config `nlooks`, cleanup, overwrite behavior, and inherited metadata.
- Runtime overloads and public pipeline exports expose the exact supported API.

## userOutcomeReview

The artifact at `cf4a31c` satisfies the stated user-visible and typing contracts. The implementation retains the legacy radar body when `_is_multilook_pair(multilook)` is true and the grid is radar. Sweep and geo calls route through `_run_pair_sweep`; that function normalizes iterable sweeps before SAFE/product I/O, removes only requested stale look subtrees under `overwrite=True`, executes `_archive_burst_ifgs` once, finalizes each configuration, returns a state for a tuple/list pair, and returns a lightweight result for an iterable. Geo calls require `geo_grid`. Each finalizer writes the look subtree and look-qualified STAC item and records the required metadata.

## checkedArtifactPaths

- Proposal: `/Users/fancy/Documents/GitHub/FanInSAR-stac-waymarks/waymark/proposals/PROPOSAL-0016-run-pair-sweep-geo.md`
- Inherited contract: `/Users/fancy/Documents/GitHub/FanInSAR-stac-waymarks/waymark/proposals/PROPOSAL-0015-multilook-sweep.md`
- Production: `faninsar/processing/pipeline/production.py`
- Public exports: `faninsar/processing/pipeline/__init__.py`
- STAC support delta: `faninsar/processing/pipeline/products.py`
- Regression coverage: `tests/processing/pipeline/test_multilook_sweep.py`, `test_production.py`, `test_run_pair_selection.py`, `test_geo_modes.py`, `test_pair_pipeline.py`, `tests/processing/geometry/test_dem_manager.py`
- Diffs inspected: `c5a506d..cf4a31c`, `fd4b391..cf4a31c`, and `f3ce818..cf4a31c`

## criterionEvidence

1. **Prior five blockers remain resolved.** Both radar and geo finalizers use `replace(snaphu_config, nlooks=float(az * rg))` for explicit configs. `_run_pair_sweep` assigns `archive["geo_prefix_state"]` to `SharedPairResources.prefix_state`; cleanup clears/deduplicates memmaps and owns temporary-directory cleanup. Overwrite removes existing requested subtrees and is a no-op for absent paths. A single tuple/list returns `single_state`, including geo. Both outcome finalizers record `multilook`, `multilook_sweep`, `wavelength_m`, `pair_id`, and `product_grid`. Regression tests covering these paths passed.
2. **Overload contract.** Source inspection found complete keyword-only signatures with defaults, no `**kwargs`, overload 1 `tuple[int, int] | list[int] = (2, 10)`, overload 2 required `Iterable[tuple[int, int]]`, and implementation annotation including `list[int]`. `typing.get_overloads(run_pair)` printed `['ProductionPairState', 'ProductionPairSweepResult']`. `inspect.signature` confirmed neither overload has `VAR_KEYWORD`, the first default is `(2, 10)`, and the second has no default.
3. **Public exports.** `PairSweepOutcome` and `ProductionPairSweepResult` are imported and listed in `faninsar.processing.pipeline.__all__`; direct package import succeeded.
4. **Proposal compliance.** The radar single-pair branch remains the existing body. Sweep results contain paths, shape, metadata, timings, and log rather than product arrays. `_archive_burst_ifgs` is called once outside the config loop. Geo validation precedes I/O. Finalizers use `looks_{az}x{rg}` and `{pair_id}__l{az}x{rg}`.
5. **Edge traces.** `_is_multilook_pair` accepts tuple and integer-list pairs and rejects iterable-of-pairs; default is `(2, 10)`. Thus tuple+geo and list+radar return state; a one-element iterable returns sweep result. `normalize_multilook_sweep` rejects empty, malformed, non-integer, non-positive, and duplicate configurations before `open_safe_product`. `overwrite=True` guards `rmtree` with `exists()`.

## reproducedVerification

- HEAD: `cf4a31c93eaa6ef9dec24fc5143461ede53edaca`.
- Ruff check: PASS, `All checks passed!`.
- Ruff format check: PASS, `3 files already formatted`.
- Required pytest matrix: PASS, `55 passed, 2 skipped, 5 deselected` in 8.14 s. Two warnings were non-failing: third-party `dominate` deprecation and a synthetic all-NaN coherence mean.
- `git diff --check c5a506d..cf4a31c`: PASS.

## directSlopAndProgrammingPass

The full delta and follow-up delta were reviewed directly under the `remove-ai-slops` and Python `programming` criteria. The round-2 change is a necessary public typing/export correction, not needless extraction or normalization. The new tests exercise dispatch semantics, outputs, cleanup, overwrite, metadata, and explicit Snaphu override behavior; they are not deletion-only, tautological, prose-pinning, or tests that merely verify requested removal. Mocks are concentrated at expensive SAFE/coregistration boundaries and assertions target observable contracts. No round-2 maintenance burden, false-confidence test, scope drift, dead export, or public `**kwargs` escape was found. `production.py` is oversized (3431 pure LOC), but that is a pre-existing architectural/style note and is not tied to a stated PROPOSAL-0016 success criterion, so it is non-blocking.

## exactEvidenceGaps

- No independent static type-checker command was required by the supplied verification checklist; runtime overload introspection and source signature inspection were reproduced instead.
- Real-data smoke evidence in the proposal record was not re-run in this one-shot lane; the required 55-test matrix and direct code-flow checks passed.
- No separate code-review report or manual-QA matrix path was supplied in the task payload. Their absence is not a stated success criterion, and direct gate evidence supports approval.

## notes

- The worktree contains an untracked `.omo/` evidence directory; no tracked source changes were made by this review.
- The proposal implementation record still describes the original `fd4b391` geo return-type deviation, but the current code and regression test at `cf4a31c` now satisfy the accepted single-config geo `ProductionPairState` contract.
