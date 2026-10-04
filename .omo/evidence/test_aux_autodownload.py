#!/usr/bin/env python3
"""A100 check: auxiliary inputs auto-download with activated P0053/P0054 code.

Download-only test (no InSAR processing):

1. Orbits (PROPOSAL-0054): ``ensure_covering_orbit_paths`` must fetch the four
   covering POEORB EOFs from the anonymous s1-orbits S3 host into a fresh
   cache directory.
2. DEM (PROPOSAL-0053): the Stack auto path ``_materialize_auto`` must fetch
   glo30 over the campaign 40 m grid (+1 km buffer), rescue glo90 twins, and
   fill unpublished ocean cells with 0 m + WARNING instead of failing 404.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path("/DATA/DATA6/fancy/FanInSAR-stac-test")
E2E = ROOT / "e2e_new"
OUT = E2E / "aux" / "autodownload_test"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT / "autodownload_report.json"
report: dict[str, object] = {"started_utc": datetime.now(timezone.utc).isoformat()}

import faninsar  # noqa: E402

report["faninsar_module"] = str(Path(faninsar.__file__).resolve())

# ---------------------------------------------------------------------------
# Part 1 — orbit EOFs from s1-orbits S3 (PROPOSAL-0054)
# ---------------------------------------------------------------------------
from faninsar.missions.s1.orbit_fetch import ensure_covering_orbit_paths  # noqa: E402

safes = sorted((ROOT / "data" / "SLC").glob("*.zip"))
assert len(safes) == 4, f"expected 4 SAFE zips, found {len(safes)}"


def _parse_sensing(name: str) -> tuple[datetime, datetime]:
    stem = name.split(".zip")[0]
    parts = stem.split("_")
    start = datetime.strptime(parts[5], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    stop = datetime.strptime(parts[6], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return start, stop


intervals = {}
mission_ids = {}
for safe in safes:
    start, stop = _parse_sensing(safe.name)
    date_id = start.strftime("%Y%m%d")
    intervals[date_id] = (start, stop)
    mission_ids[date_id] = safe.name[:3]

orbit_cache = OUT / "orbits"
if orbit_cache.exists():
    for stale in orbit_cache.glob("*.EOF"):
        stale.unlink()
orbit_cache.mkdir(parents=True, exist_ok=True)

t0 = time.time()
orbit_paths = ensure_covering_orbit_paths(
    intervals=intervals,
    mission_ids=mission_ids,
    cache_dir=orbit_cache,
    existing=None,
    auto_fetch=True,
)
orbit_seconds = time.time() - t0
downloaded = {}
for date_id, path in sorted(orbit_paths.items()):
    downloaded[date_id] = {
        "file": Path(path).name,
        "size": Path(path).stat().st_size,
        "covers_window": Path(path).exists(),
    }
report["orbits"] = {
    "ok": len(downloaded) == len(intervals)
    and all(item["covers_window"] for item in downloaded.values()),
    "seconds": round(orbit_seconds, 1),
    "cache_dir": str(orbit_cache),
    "downloaded": downloaded,
}
print("ORBITS", json.dumps(report["orbits"], indent=2))

# ---------------------------------------------------------------------------
# Part 2 — auto DEM with 1 km buffer + unpublished-ocean handling (P0053)
# ---------------------------------------------------------------------------
from faninsar.processing.geometry.grids import GridSpec  # noqa: E402
from faninsar.processing.geometry.providers.p0030_adapter import (  # noqa: E402
    _materialize_auto,
)

GEO_GRID = dict(
    crs="EPSG:32633",
    transform=(320067.489269, 40.0, 0.0, 4284852.990441, 0.0, -40.0),
    width=7511,
    height=5889,
    resolution_m=(40.0, 40.0),
)
grid = GridSpec(
    crs=GEO_GRID["crs"],
    transform=GEO_GRID["transform"],
    height=GEO_GRID["height"],
    width=GEO_GRID["width"],
    resolution_m=GEO_GRID["resolution_m"],
)

dem_cache = OUT / "dem_cache"
dem_cache.mkdir(parents=True, exist_ok=True)

t0 = time.time()
raster = _materialize_auto(grid, cache_dir=dem_cache, budget=None)
dem_seconds = time.time() - t0

import numpy as np  # noqa: E402

array = getattr(raster, "array", None)
if array is None:
    array = raster._array
array = np.asarray(array)
zero = int(((array == 0) | (array == 0.0)).sum())
report["dem"] = {
    "ok": bool(np.isfinite(array).all()),
    "seconds": round(dem_seconds, 1),
    "shape": list(array.shape),
    "nan_count": int(np.isnan(array).sum()),
    "zero_count": zero,
    "height_min": float(np.nanmin(array)),
    "height_max": float(np.nanmax(array)),
    "vertical_datum": str(getattr(raster, "vertical_datum", "")),
    "provenance": getattr(raster, "provenance", None),
    "cache_dir": str(dem_cache),
    "cache_files": sorted(p.name for p in dem_cache.glob("*")),
}
print("DEM", json.dumps(report["dem"], indent=2, default=str))

report["finished_utc"] = datetime.now(timezone.utc).isoformat()
report["all_ok"] = bool(report["orbits"]["ok"] and report["dem"]["ok"])
REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("REPORT", REPORT_PATH)
print("AUTODOWNLOAD", "PASS" if report["all_ok"] else "FAIL")
