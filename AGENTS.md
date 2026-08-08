# AGENTS.md

This file provides guidance to agents when working with code in this repository.

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
