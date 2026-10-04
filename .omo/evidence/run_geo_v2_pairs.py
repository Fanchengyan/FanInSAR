'''Fine-grid geo mode driver over the Stack API (one Stack per mode).

All processing runs through the Stack workflow: S1Stack.from_safes ->
prepare -> coregister_scenes (once per mode; date-level caches shared by
all pairs) -> form_interferograms per pair.  This driver only orchestrates:
it assembles the campaign inputs, exports each pair's 40 m layers for the
comparison figures, drops the consumed burst units to bound disk, and
records timings.  Exported pairs are skipped on relaunch, so the driver is
resumable.  Everything stays under /DATA/DATA6/fancy.
'''

from __future__ import annotations

import argparse
import json
import shutil
import shutil as _shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path("/DATA/DATA6/fancy/FanInSAR-stac-test")
E2E = ROOT / "e2e_new"
SLC = ROOT / "data" / "SLC"
ORBIT_DIR = E2E / "aux" / "orbits"
DEM_CACHE = E2E / "aux" / "dem" / "cache"
DEM_SOURCE_COPY = E2E / "aux" / "dem" / "dem_merged_wgs84_source.tif"

SWATHS = ("IW1", "IW2", "IW3")
ALL_BURSTS = {"IW1": "all", "IW2": "all", "IW3": "all"}
AZ_LOOKS, RG_LOOKS = 5, 2
REFERENCE = "20240611"
PAIR_NAMES = (
    "20240611_20240623",
    "20240611_20240705",
    "20240611_20240717",
    "20240623_20240705",
    "20240623_20240717",
    "20240705_20240717",
)

# Fine geographic grid: (5, 2) looks -> 40 m x 40 m output cells on the same
# origin as the historical 40 m study grid (40/5 = 8 m rows, 40/2 = 20 m cols).
GEO_GRID = dict(
    crs="EPSG:32633",
    transform=(320067.4892684915, 20.0, 0.0, 4284852.990440817, 0.0, -8.0),
    width=15022,
    height=29445,
    resolution_m=(20.0, 8.0),
)

MIN_WORK_ROOT_FREE_GB = 60
_SHM_BYTES_PER_GIB = 1024**3


def filled_on_grid_dem(stack) -> object:
    """Materialize the Stack DEM on its grid and fill void cells with 0 m."""
    from faninsar.processing.geometry.api import RasterDEM

    dem = stack.config.dem
    if not (
        isinstance(dem, RasterDEM)
        and dem.grid == stack.grid
        and dem.vertical_datum == "ellipsoidal"
    ):
        dem = dem.to_raster(stack.grid, vertical_datum="ellipsoidal")
    arr = np.asarray(dem.array, dtype=np.float64).copy()
    n_void = int((~np.isfinite(arr)).sum())
    arr[~np.isfinite(arr)] = 0.0
    print(f"[dem] materialized {arr.shape}, filled {n_void} void px", flush=True)
    return RasterDEM(array=arr, grid=dem.grid, vertical_datum="ellipsoidal")


def product_grid() -> dict:
    """Return the looks-scaled 40 m product grid for the comparison warp."""
    return {
        "crs": GEO_GRID["crs"],
        "transform": (
            GEO_GRID["transform"][0],
            GEO_GRID["transform"][1] * RG_LOOKS,
            0.0,
            GEO_GRID["transform"][3],
            0.0,
            GEO_GRID["transform"][5] * AZ_LOOKS,
        ),
        "width": GEO_GRID["width"] // RG_LOOKS,
        "height": GEO_GRID["height"] // AZ_LOOKS,
        "resolution_m": (
            GEO_GRID["resolution_m"][0] * RG_LOOKS,
            GEO_GRID["resolution_m"][1] * AZ_LOOKS,
        ),
    }


def resolve_work_root(requested: Path) -> Path:
    """Return the requested RAM-backed root when it has real headroom."""
    try:
        requested.mkdir(parents=True, exist_ok=True)
        usage = _shutil.disk_usage(requested)
    except OSError:
        usage = None
    if usage is None or usage.free < MIN_WORK_ROOT_FREE_GB * _SHM_BYTES_PER_GIB:
        fallback = E2E / "runs" / "v2_work"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"[work-root] {requested} unusable "
            f"(free={0 if usage is None else usage.free // _SHM_BYTES_PER_GIB}G); "
            f"falling back to {fallback}",
            flush=True,
        )
        return fallback
    print(
        f"[work-root] {requested} free {usage.free // _SHM_BYTES_PER_GIB}G",
        flush=True,
    )
    return requested


def stage_safes(work_root: Path, safes: list[Path]) -> list[Path]:
    """Copy the SAFE archives onto the work root once, with ENOSPC retries.

    The shared storage pool intermittently refuses even reads (SIGBUS on
    mapped access), so the run stages its inputs onto the RAM-backed root
    and reads them from there for the rest of the campaign.
    """
    from faninsar.missions.s1.processing import _archive_write_chunks

    stage = work_root / "SLC"
    stage.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for safe in safes:
        target = stage / safe.name
        if target.exists() and target.stat().st_size == safe.stat().st_size:
            staged.append(target)
            continue
        payload = safe.read_bytes()
        _archive_write_chunks(
            np.frombuffer(payload, dtype=np.uint8), str(target), safe.stem
        )
        staged.append(target)
        print(f"[stage] {safe.name} ({safe.stat().st_size // 1024 // 1024} MB)", flush=True)
    return staged


def copy_with_archive_recovery(source: Path, dest: Path, tag: str) -> None:
    """Copy one file to the persistent root through the ENOSPC-safe writer."""
    from faninsar.missions.s1.processing import _archive_write_chunks

    payload = np.frombuffer(source.read_bytes(), dtype=np.uint8)
    _archive_write_chunks(payload, str(dest), tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, choices=["geo_pair_v2", "geo_geometry_v2"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-root", default=str(E2E / "runs" / "v2_work"))
    args = parser.parse_args()
    coreg_mode = "pair" if args.mode == "geo_pair_v2" else "geometry"

    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.missions.s1 import S1Stack
    from faninsar.processing.geometry import DEM
    from faninsar.processing.mosaicking.grid import GeoGridSpec

    work_root = resolve_work_root(Path(args.work_root))
    export_root = E2E / f"faninsar_{args.mode}"
    export_root.mkdir(parents=True, exist_ok=True)
    safes = sorted(SLC.glob("*.zip"))
    if not safes:
        message = f"no SAFE zips under {SLC}"
        raise SystemExit(message)

    # One Stack per mode: prepare, DEM, and date-level coregistration are
    # pair-independent and run once; only merging/export is per pair.
    work = work_root / args.mode
    work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    stack = S1Stack.from_safes(
        safes,
        work_dir=work,
        reference=REFERENCE,
        polarizations=("VV",),
        swaths=SWATHS,
        bursts=ALL_BURSTS,
        dem=DEM.from_raster(DEM_SOURCE_COPY, vertical_datum="egm2008"),
        dem_cache_dir=DEM_CACHE,
        orbit_paths=sorted(ORBIT_DIR.glob("*.EOF")),
        executor="torch",
        device=args.device,
        coreg_mode=coreg_mode,
        coregistration_grid="geo",
        geo_grid=GeoGridSpec(**GEO_GRID),
        multilook=(AZ_LOOKS, RG_LOOKS),
    )
    # CUDA Ampcor requires device-local admission: n_jobs must stay 1.
    stack.config.n_jobs = 1
    stack.prepare()
    print(f"[{args.mode}] prepared", flush=True)
    if not (work / "coreg" / REFERENCE / "coreg_done.json").exists() or not all(
        (work / "coreg" / d / "coreg_done.json").exists()
        for d in ("20240623", "20240705", "20240717")
    ):
        stack.config.dem = filled_on_grid_dem(stack)
        t0 = time.perf_counter()
        stack.coregister_scenes()
        print(
            f"[{args.mode}] coregistered all dates "
            f"({time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    else:
        print(f"[{args.mode}] coreg caches present; skip", flush=True)

    timings: dict[str, object] = {}
    timing_path = export_root / "_timing.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())

    for pair_name in PAIR_NAMES:
        dest = export_root / pair_name
        if (dest / "wrapped_phase.npy").exists() and (
            dest / "coherence.npy"
        ).exists():
            print(f"[{args.mode}] {pair_name} already exported; skip", flush=True)
            continue

        t0 = time.perf_counter()
        stack.form_interferograms(
            multilook=(AZ_LOOKS, RG_LOOKS),
            pairs=__import__("faninsar").Pairs.from_names([pair_name]),
        )
        t_ifg = time.perf_counter() - t0
        print(f"[{args.mode}] {pair_name} IFG formed ({t_ifg:.1f}s)", flush=True)

        t0 = time.perf_counter()
        for ifg_dir in stack.ifg_dirs:
            if pair_name not in ifg_dir.as_posix():
                continue
            store = InterferogramArtifactStore.open(ifg_dir)
            art = store.read()
            pair = f"{store.pair[0]}_{store.pair[1]}"
            out = export_root / pair
            out.mkdir(parents=True, exist_ok=True)
            phase = np.asarray(art.wrapped_phase, dtype=np.float32)
            coh = np.asarray(art.coherence, dtype=np.float32)
            np.save(out / "wrapped_phase.npy", phase)
            np.save(out / "coherence.npy", coh)
            print(f"[{args.mode}] exported {pair} {phase.shape}", flush=True)
        t_export = time.perf_counter() - t0

        timings[pair_name] = {
            "ifg_s": round(t_ifg, 1),
            "export_s": round(t_export, 1),
        }
        timing_path.write_text(json.dumps(timings, indent=2, sort_keys=True))

        # Merge consumed the pair's burst units; drop them to bound disk.
        shutil.rmtree(work / "ifg" / f"ml_{AZ_LOOKS}x{RG_LOOKS}" / pair_name,
                      ignore_errors=True)
        print(f"[{args.mode}] {pair_name} done ({t_ifg + t_export:.1f}s)",
              flush=True)

    grid_path = E2E / "grid_source_v2.json"
    document = {"geo_grid": product_grid(), "looks": [AZ_LOOKS, RG_LOOKS]}
    grid_path.write_text(json.dumps(document, indent=2, sort_keys=True))
    print(
        f"[{args.mode}] all pairs done in "
        f"{round((time.perf_counter() - t_all) / 3600, 2)}h; "
        f"grid source written to {grid_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
