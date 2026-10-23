"""Configure a compatible geospatial test environment.

Network tests are opt-in: the suite skips ``network``-marked tests unless
``--run-network`` is passed, so a default run never downloads remote data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

import pytest

_PROJ_OVERRIDE_NAMES: Final = ("PROJ_DATA", "PROJ_LIB")
_PROJ_PROBE: Final = (
    sys.executable,
    "-c",
    "from pyproj import CRS as P; from rasterio.crs import CRS as R; "
    "raise SystemExit(P.from_epsg(4326).to_epsg() != 4326 or "
    "R.from_epsg(4326).to_epsg() != 4326)",
)
_PROJ_PROBE_TIMEOUT_SECONDS: Final = 5

for override_name in _PROJ_OVERRIDE_NAMES:
    override_value = os.environ.get(override_name)
    if override_value is None:
        continue
    if override_value == "":
        os.environ.pop(override_name)
        continue

    probe_environment = os.environ.copy()
    for candidate_name in _PROJ_OVERRIDE_NAMES:
        probe_environment.pop(candidate_name, None)
    probe_environment[override_name] = override_value

    try:
        probe_result = subprocess.run(
            _PROJ_PROBE,
            env=probe_environment,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_PROJ_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        os.environ.pop(override_name, None)
        continue

    if probe_result.returncode != 0:
        os.environ.pop(override_name, None)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in switch for remote-data tests."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked network (they download remote data)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip ``network``-marked tests unless ``--run-network`` is passed."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(
        reason="network test; pass --run-network to run it",
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
