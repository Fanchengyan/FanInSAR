# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Governing Proposals

The following registered Waymarks are the normative design baseline for geometry,
native acceleration, packaging, and their CI. The registered sidecar is the
source of truth for the full designs; this section records only the rules that
must hold for day-to-day development.

- **PROPOSAL-0020** — one NumPy-in/NumPy-out geometry contract with Native,
  prepared Compile, and Eager identities. Dispatch is native-first, then
  Compile, then same-device Eager; an admitted execution never silently
  switches backend. The retired finite-difference Newton path is forbidden.
- **PROPOSAL-0025** — `auto` is the prepared native/Compile/Eager production
  order. Native artifacts come from the packaged FanInSAR sources and the
  explicit preparation API; unqualified artifacts fail closed, and CPU serial
  packaging/dispatch smoke must remain green on Ubuntu, macOS, and Windows CI.
  A Native/OpenMP performance claim requires evidence from the actual geometry
  loop, not a separate probe; this CI is only serial/public-dispatch coverage.
- **PROPOSAL-0026** — `geo2rdr` and `rdr2geo` share the fourteen-field result,
  strict convergence and invalid-lane rules, variable-DEM semantics, and the
  integrated device-resident execution contract. All native inputs and results
  pass one centralized validator before ABI access. Native, Compile, and Eager
  benchmarks use identical fixture, DEM, iteration budget, and I/O boundaries;
  compute-only timings cannot replace the full public-call promotion boundary.
- **PROPOSAL-0030** — the DEM manager is a two-axis contract: fourteen
  selection names (thirteen products plus `auto`) on the product axis and a
  manual `product:provider` grammar on the provider axis. Defaults are
  anonymous cloud channels (AWS/Planetary Computer); the manager never
  silently switches product or provider, `auto` rescues withheld GLO-30 cells
  with same-family GLO-90 only, and provider failures raise the structured
  `DEMProviderUnavailableError` instead of a hidden fallback. Selection fails
  closed before any network traffic; user-facing usage is documented in
  `docs/dem_manager_guide.md`.

When a change would intentionally violate one of these rules, do not patch
around the Proposal. Amend the relevant Proposal or create a new accepted
Proposal before formal implementation. Every change touching
`faninsar/processing/geometry`, native sources, packaging, or the related CI
must cite the governing Proposal IDs and run the relevant regression tests.

## Build, Lint, and Test Commands

```bash
uv sync --dev                        # Install dev dependencies into the uv-managed .venv
# OR
conda env create -f environment.yml   # Use conda
conda activate faninsar
```

```bash
uv run ruff check .                 # Lint with Ruff
uv run ruff format .                # Format with Ruff
uv run pre-commit install           # Install pre-commit hooks
uv run pre-commit run --all-files
```

```bash
uv run pytest                                # Run all tests
uv run pytest tests/test_pairs.py            # Run single test file
uv run pytest tests/test_pairs.py::test_fn   # Run single test function
uv run pytest tests/test_pairs.py::TestCls   # Run single test class
uv run pytest --cov=faninsar                 # Run with coverage
uv run pytest tests/test_file.py -v --override-ini="addopts="  # Without coverage (ISCE2 env)
```

```bash
cd docs && make html              # Build HTML docs
open _build/html/index.html       # View docs (macOS)
```

## Architecture

### Module Hierarchy

- **`faninsar/`** — Main package
  - **`_core/`** — Core data structures and algorithms
    - `alg.py` — Core algorithms
    - `device.py` — Device/CPU/GPU management
    - `file_tools.py` — File I/O utilities
    - `geo/` — Geographic tools (`geo_tools.py`)
    - `render/` — Rendering: HTML/SVG export, formatting (`formatting*.py`, `svg_graph.py`, `html_component.py`)
    - `sar/` — SAR data structures: `Pairs`, `Acquisition`, `Baseline`, `loops`, `sar_missions`, `sar_property`, `sar_tools`
  - **`backends/`** — Backend utilities (`lazy_rasterio.py`)
  - **`constants/`** — Physical constants (`sar.py`)
  - **`datasets/`** — Dataset loaders: HyP3, LiCSAR, ARIA, GACOS, APS, base, geobox, hierarchical, ifg, xarray_dataset
  - **`isce2/`** — ISCE2 TOPS Sentinel-1 stack processing: workflow, config, command management, executors, sensors, workflows
  - **`isce3/`** — ISCE3 utilities: alignment, geocoding, geometry, GPU, topo, transform, S1_base, metadata
  - **`logging/`** — Logging utilities
  - **`NSBAS/`** — NSBAS time series algorithms: inversion, tsmodels, freeze_thaw_process
  - **`plots/`** — Visualization utilities: colorbars, formatters, hist_colorbar, utils
  - **`query/`** — Spatial/temporal query: `BoundingBox`, points, polygons, query
  - **`samplers/`** — Grid samplers: grid, collate
  - **`typing/`** — Type definitions: device, geo, logging, pairs, sar
  - **`uncertainties/`** — Uncertainty propagation (`uncertainty.py`)


## Sentinel-1 Ad-hoc Test Workspace

When running ad-hoc Sentinel-1 / InSAR experiments (scratch scripts, campaign runs, large outputs, DEM/SLC caches, GeoTIFF/PNG comparisons):

- **Put scripts and data under** `/Volumes/DATA2/TEST_sentinel-1` — not under the repository root.
- **Oracle / multi-stack compare drivers and former `processing.comparison` harness** live only under  
  `/Volumes/DATA2/TEST_sentinel-1/faninsar-experiments/` (scripts, lib, campaign reports).  
  Do **not** add compare/plot/ab_test/oracle drivers anywhere in the product tree.
- Experimental notes and durable findings belong in the registered external Waymark
  sidecar, not the product tree.
- Do **not** write large intermediate products, `out/`, campaign trees, or one-off test scripts into the repo working tree.
- Unit/integration tests that ship with the package stay in `tests/`; only heavy local data and throwaway experiment scripts go to DATA2.
- Prefer a dated or named subfolder, e.g. `/Volumes/DATA2/TEST_sentinel-1/<campaign-or-run-id>/`.

## Code Style Guidelines

### Language and Naming

- Write all code and comments in English
- Use descriptive English names for variables, functions, and classes
- Follow PEP 8 and PEP 257 standards

## Code Conventions

- **Python 3.11+** required
- **`from __future__ import annotations`** in every file
- **Type hints**: Use Python 3.11+ syntax (`str | None`, `dict[str, int]`). Use `Literal` for fixed value sets. Use `@overload` for functions with multiple calling patterns or polymorphic return types
- **Docstrings**: Use NumPy-style docstrings for all public modules, classes, functions, and methods. Include Parameters, Returns, Raises, and Examples where applicable, and use Sphinx reStructuredText markup such as :func:..., :class:..., and directives like .. note::, .. tip::, and .. warning:: when needed.
- **Logging**: Use `from faninsar.logging import setup_logger; logger = setup_logger(__name__)` — log before raising exceptions errors
- **Paths**: Use `pathlib.Path` internally; convert to `str` only when passing to C++ libraries or APIs that require strings
- **Type-checking imports**: Put heavy/circular imports inside `if TYPE_CHECKING:` blocks
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **Linter**: ruff only (no black, no flake8). Line length 88. Ruff excludes `tests/`, `docs/`, `examples/` directories.
