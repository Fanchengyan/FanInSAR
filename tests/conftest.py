"""Configure a compatible geospatial test environment."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

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
