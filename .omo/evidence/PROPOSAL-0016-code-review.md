# Code review: PROPOSAL-0016 round-2 overload-contract fix

## Scope and independent evidence

- Reviewed `f3ce818..cf4a31c` at `cf4a31c93eaa6ef9dec24fc5143461ede53edaca`.
- Changed files: `faninsar/processing/pipeline/production.py` and `faninsar/processing/pipeline/__init__.py`.
- `git diff --check f3ce818..cf4a31c`: pass.
- Ruff check: pass (`All checks passed!`).
- Ruff format check: pass (`3 files already formatted`).
- Requested focused pytest command: pass (`55 passed, 2 skipped, 5 deselected` in 6.96 s). It produced two non-fatal warnings, including NumPy `Mean of empty slice` from the synthetic geo-wiring test.
- Requested overload introspection: pass: two overloads returning `ProductionPairState` and `ProductionPairSweepResult`, respectively.
- An AST signature comparison found all parameter names in both overloads match the implementation. The single overload has the implementation's `(2, 10)` default; the sweep overload correctly requires `multilook`. All shared keyword defaults match the implementation.
- `basedpyright` is not installed in the available environment, so a static type-checker pass is N/A; the requested runtime overload inspection succeeded.
- `omo ulw-loop status --json` reported `ULW_LOOP_PLAN_MISSING`; this artifact therefore uses the prescribed fallback path.

## Correctness and scope assessment

The prior MEDIUM finding is addressed: both overloads repeat the concrete keyword-only surface and no longer expose `**kwargs: Any` (`production.py:1980-2045`). The first overload accepts the runtime-supported `list[int]` single-pair form and has the same default as the implementation (`:1990`, `:2057`). The single-configuration return remains correct for geo because `_run_pair_sweep` returns `single_state` when `_is_multilook_pair` succeeds (`:2672`, `:2919-2925`).

The `pipeline` exports are alphabetized in both the import group and `__all__` (`__init__.py:13-28`, `:45-59`). They originate in `production`, which `pipeline.__init__` already imports, so this adds no new import edge or circular-import risk. The docstring return wording remains accurate for both radar and geo single runs and actual multi-config sweeps.

## Skill-perspective check

Ran: yes. I consulted `omo:remove-ai-slops` and `omo:programming` before reviewing test relevance and maintainability.

- Remove-AI-slops: no deletion-only tests, requested-removal checks, tautological tests, implementation-constant mirrors, or unnecessary production data extraction, parsing, or normalization were added. The repeated signatures are necessary to make the public overload contract exact, not needless production complexity.
- Programming: the old public `Any` keyword escape hatch is removed. The diff adds no untyped escape hatch, brittle prompt test, needless abstraction, or unnecessary validation/parsing. The long overload surface mirrors an already-existing public function and is warranted here; a `TypedDict`/`Unpack` abstraction would be additional indirection without a project precedent identified in review.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

1. `tests/processing/pipeline/test_multilook_sweep.py:696-841` — the geo memmap test supplies a pre-built `geo_prefix_state` from its `_archive_burst_ifgs` stub. It verifies the consumer-side ownership/cleanup and public geo return contract, but does not directly cover producer-side capture in `_archive_burst_ifgs` (`production.py:3123`, `:3164`). Suggestion: add a narrow producer-path regression when practical. This remains non-blocking because shared-resource cleanup and the real-data QA path provide adjacent coverage.

## Verdict

- `codeQualityStatus`: CLEAR
- `recommendation`: APPROVE
- `blockers`: none
