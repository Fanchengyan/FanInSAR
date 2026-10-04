# AGENTS.md

This repository is a Python 3.11+ FanInSAR library. Keep changes small,
domain-owned, and evidence-backed. This guide records repository conventions;
`pyproject.toml`, `environment.yml`, the docs build files, and the Waymark
sidecar remain the sources of truth for configuration and design details.

## Before editing

- Inspect the current tree and the sources of truth for the files in scope.
- For Waymark work, read
  `/Users/fancy/Library/Application Support/Waymark/projects/FanInSAR-stac/project.json`.
  Resolve both `sidecar_root` and `workspace_root` from that file; the cached
  paths alone are not canonical.
- When a branch below applies, read the applicable registered Proposal files from
  `sidecar_root` before editing. Keep Proposal prose in the sidecar rather than
  copying it here.

### Proposal triggers

- **Geometry, native code, backend dispatch, preparation, benchmarks, packaging,
  or related CI:** read P0020, P0025, P0026, and any later applicable registered
  Proposal that governs the touched seam.
- **DEM selection, providers, fetching, materialization, or stack DEM wiring:**
  read P0030 and P0041.
- **Package layout, domain ownership, migration, Stack, Network, or storage
  boundaries:** read P0043 and P0050.
- **Remote catalogs, authentication, transfer, or automatic input preparation:**
  read P0044, P0045, P0047, and P0049.
- **Scientific result objects, inversion, persistence, or export semantics:**
  read P0052.

Identify the governing Proposal in the work record. An intentional contract
violation requires an amendment to the relevant Proposal or a new accepted
Proposal before implementation.

## Current domain ownership

Use the post-P0050 tree as the ownership map; place new code with the concept
that owns its behavior.

- `faninsar/core/` — shared value objects and foundational contracts.
- `faninsar/data/` — data objects, dataset representations, spatial queries,
  and samplers.
- `faninsar/stack/` — source-to-generation orchestration, Stack lifecycle,
  search, preparation, grids, and run configuration.
- `faninsar/network/` — persisted scientific products, Network views, readers,
  and product provenance.
- `faninsar/processing/` — scientific stages such as geometry, registration,
  interferometry, masking, mosaicking, unwrapping, and runtime execution.
- `faninsar/io/` — storage protocols, readers/writers, validation, and export.
- `faninsar/remote/` — provider-neutral catalog, authentication, transfer,
  cache, and remote-resource mechanics.
- `faninsar/timeseries/` — time-series result types, inversion, and export.
- `faninsar/missions/` — mission-owned behavior, including Sentinel-1 and NISAR.
- `faninsar/plotting/` — plotting, colormaps, and renderers.
- `faninsar/cli/` — command-line entry points and warm-up commands.

Keep scientific selection and interpretation in its domain; keep shared
transport in `remote`, persistence in `io`/`network`, and orchestration in
`stack`.

## Python standards

- Write code, comments, names, and user-facing developer text in English.
- Use Python 3.11+ type syntax: `|` unions, built-in generics, and
  `Literal`, `TypedDict`, `Protocol`, `TypeAlias`, `Self`, or `@overload` when
  they clarify the contract.
- Add `from __future__ import annotations` to new or modified implementation
  modules. Do not retrofit existing `__init__.py` files solely to satisfy this
  rule.
- Keep standard-library imports at module scope. Use `TYPE_CHECKING` for
  type-only heavy or circular imports; lazy-load optional heavy dependencies
  and raise a descriptive `ImportError` when a required optional dependency is
  missing.
- Use `pathlib.Path` internally and convert to `str` only for APIs that require
  it, such as native C++ interfaces.
- Give public modules, classes, functions, and methods NumPy-style docstrings;
  add Parameters, Returns, Raises, Examples, and relevant Sphinx references as
  the API requires.
- Use the project logger before raising an error with useful context, using
  `from faninsar.logging import setup_logger`; use logger methods for warnings.
- Use Conventional Commits when making commits (`feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, or `chore:`).
- Ruff rules, ignores, exclusions, and formatting behavior come from
  `pyproject.toml`; consult that file instead of duplicating its settings here.

## Environment, tests, and docs

Use the uv-managed environment for normal work:

```bash
uv sync --dev
```

The conda environment defined by `environment.yml` is `faninsar-dev`:

```bash
conda env create -f environment.yml
conda activate faninsar-dev
```

Check-only verification commands are:

```bash
uv run ruff check --no-fix .
uv run ruff format --check .
```

Mutating setup or formatting commands are separate:

```bash
uv run ruff format .
uv run pre-commit install
uv run pre-commit run --all-files
```

Run focused tests first, then the broader suite when the change warrants it.
Existing examples include:

```bash
uv run pytest --no-cov tests/processing/interferometry/test_pair.py
uv run pytest --no-cov tests/processing/interferometry/test_pair.py::test_identical_slcs_have_unit_coherence_and_zero_phase
uv run pytest -q --no-cov tests/processing/geometry/test_native_v2_cpu_sources.py::test_serial_cpu_extension_returns_validated_fourteen_field_result
uv run pytest --cov=faninsar
uv run pytest
```

The geometry native workflow job node is
`.github/workflows/geometry-native.yml#jobs.cpu`; its public-dispatch tests
are listed in that workflow and should be run with its `--no-cov` boundary.

```bash
uv run pytest -q --no-cov tests/processing/geometry/test_public_geometry_v2.py::test_public_selectors_execute_prepared_eager_and_native
```

Build docs with `cd docs && make html`; the generated index is
`docs/build/html/index.html`. For DEM behavior and user-facing examples, read
`docs/source/user_guide/dem.rst`.

## Temporary files

Never write temporary or scratch files to the system `/tmp` or the filesystem
root. Put them under the working directory instead: create a dedicated
`tmp/` directory there for temporary files when needed and keep all of them
inside it.

## Sentinel-1 scratch workspace

This policy is host-local. Before using it, check that `/Volumes/DATA2` is
mounted and that `/Volumes/DATA2/TEST_sentinel-1` is accessible. Put ad-hoc
Sentinel-1/InSAR scripts, campaign data, DEM/SLC caches, large outputs, and
comparisons there; keep oracle and multi-stack experiment drivers under
`/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/`. Keep shipping unit and
integration tests under `tests/`, and keep durable experimental findings in
the registered Waymark sidecar.

## Completion

A change is complete when the governing Proposal is identified where relevant,
the focused tests pass, both check-only Ruff commands pass, and the exact
commands and results are recorded with the work. Geometry/native changes also
need the relevant CI and Proposal qualification gates, including packaging and
public-dispatch evidence where applicable.
