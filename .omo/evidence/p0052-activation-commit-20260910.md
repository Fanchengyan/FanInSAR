# P0052 activation commit evidence

Date: 2026-09-10

## Verification

| Scenario | Invocation | Observable |
|---|---|---|
| Affected regression suite | `uv run pytest tests/io/storage tests/network tests/timeseries tests/stack tests/missions/nisar/test_flatten_stage.py tests/missions/nisar/test_provider.py -q --override-ini='addopts='` | `232 passed, 17 warnings in 13.09s` |
| Ruff lint | `uv run ruff check faninsar` | `All checks passed!` |
| Ruff format | `uv run ruff format --check faninsar` | `264 files already formatted` |
| Patch integrity | `git diff --check` and `git diff --cached --check` | no output, exit status 0 |
| Activation commit | `git show --format='%H%n%s' --name-only HEAD` | commit `5efb023542f9dcd013707944a5e9aa6dd1170066`, 25 P0052 files |

The staged file list was limited to the P0052 product code, tests, and docs.
The `.omo/evidence/` files were intentionally not staged or committed.

## Captured artifacts

- `.omo/evidence/p0052-verification-readback-20260910.txt`
- `.omo/evidence/p0052-manual-qa.md`
- `.omo/evidence/p0052-docs-verification.md`
- `.omo/evidence/p0052-timeseries-export-artifacts-20260910/`

