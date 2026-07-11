"""MintPy HDF5 export for frame products (inverse of :mod:`discovery.mintpy`).

Writes a MintPy-ready project directory::

    <output_dir>/
    └── inputs/
        ├── ifgramStack.h5      # all interferograms + coherence in one HDF5
        └── geometryGeo.h5      # geometry layers (dem, incidence, azimuth, ...)

The HDF5 layout follows the MintPy ``ifgramStack`` / ``geometry`` conventions
documented at
https://mintpy.readthedocs.io/en/latest/api/data_structure/. Root-level
metadata is written as HDF5 attributes (stringified, matching MintPy's own
``writefile.layout_hdf5``), and datasets use the canonical names/dtypes:

``ifgramStack.h5``::

    date             S8        (m, 2)      YYYYMMDD reference/secondary
    bperp            float32   (m,)        perpendicular baseline [m]
    dropIfgram       bool      (m,)        keep/drop flag
    unwrapPhase      float32   (m, L, W)   unwrapped phase [rad]
    coherence        float32   (m, L, W)   spatial coherence   (optional)
    connectComponent int16     (m, L, W)   connected components (optional)
    wrapPhase        float32   (m, L, W)   wrapped phase [rad]  (optional)

``geometryGeo.h5``::

    height           float32   (L, W)      DEM elevation [m]
    incidenceAngle   float32   (L, W)      incidence angle [deg]
    azimuthAngle     float32   (L, W)      azimuth angle [deg]   (optional)
    waterMask        bool      (L, W)      water mask            (optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger

from .metadata import save_json

if TYPE_CHECKING:
    from .geometry import FrameGeometry
    from .interferogram import FrameInterferogramCollection

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

# ifgramStack datasets that are always written vs. optional.
_IFGRAM_REQUIRED: dict[str, tuple[np.dtype, tuple[int | None, ...]]] = {
    "date": (np.dtype("S8"), (None, 2)),
    "bperp": (np.float32, (None,)),
    "dropIfgram": (np.bool_, (None,)),
}
_IFGRAM_3D: dict[str, np.dtype] = {
    "unwrapPhase": np.float32,
    "coherence": np.float32,
    "wrapPhase": np.float32,
    "connectComponent": np.int16,
}
# Map frame interferogram asset name -> mintpy dataset name.
_IFGRAM_ASSET_MAP: dict[str, str] = {
    "unw_phase": "unwrapPhase",
    "coherence": "coherence",
    "wrapped_phase": "wrapPhase",
    "conncomp": "connectComponent",
}
# Map frame geometry asset name -> mintpy dataset name.
_GEOM_ASSET_MAP: dict[str, str] = {
    "dem": "height",
    "incidence": "incidenceAngle",
    "azimuth": "azimuthAngle",
    "water_mask": "waterMask",
}
# Per-dataset UNIT attribute (mintpy convention).
_DS_UNIT: dict[str, str] = {
    "unwrapPhase": "radian",
    "wrapPhase": "radian",
    "coherence": "1",
    "connectComponent": "1",
    "bperp": "m",
    "height": "m",
    "incidenceAngle": "degree",
    "azimuthAngle": "degree",
    "waterMask": "1",
}


def _stringify_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    """Convert all metadata values to strings (MintPy convention)."""
    return {k: str(v) for k, v in attrs.items() if v is not None}


def _geo_attrs_from_geometry(geometry: FrameGeometry) -> dict[str, str]:
    """Derive geocoded HDF5 root attributes from a FrameGeometry."""
    meta = geometry._require_metadata()
    bounds = meta["bounds"]  # [left, bottom, right, top]
    width = int(meta["width"])
    height = int(meta["height"])
    res = meta["resolution"]  # [x_res, y_res]
    x_first = float(bounds[0])
    y_first = float(bounds[3])  # top
    x_step = float(abs(res[0]))
    y_step = float(-abs(res[1]))  # negative (north-up)

    attrs = {
        "FILE_TYPE": "geometry",
        "LENGTH": height,
        "WIDTH": width,
        "X_FIRST": x_first,
        "Y_FIRST": y_first,
        "X_STEP": x_step,
        "Y_STEP": y_step,
        "GEODETIC": "yes" if "4326" in str(meta.get("crs", "")) else "no",
    }
    return _stringify_attrs(attrs)


def _ifgram_attrs(
    geometry: FrameGeometry | None,
    length: int,
    width: int,
    *,
    project_name: str = "faninsar",
) -> dict[str, str]:
    """Build root attributes for ``ifgramStack.h5``."""
    attrs: dict[str, Any] = {
        "FILE_TYPE": "ifgramStack",
        "LENGTH": length,
        "WIDTH": width,
        "PROJECT_NAME": project_name,
        "PROCESSOR": "faninsar",
    }
    if geometry is not None:
        meta = geometry._require_metadata()
        res = meta.get("resolution")
        if res:
            attrs["AZIMUTH_PIXEL_SIZE"] = abs(float(res[1]))
            attrs["RANGE_PIXEL_SIZE"] = abs(float(res[0]))
        # MintPy reads X/Y_FIRST/STEP for geocoded stacks too.
        bounds = meta.get("bounds")
        if bounds:
            attrs["X_FIRST"] = float(bounds[0])
            attrs["Y_FIRST"] = float(bounds[3])
            if res:
                attrs["X_STEP"] = abs(float(res[0]))
                attrs["Y_STEP"] = -abs(float(res[1]))
    return _stringify_attrs(attrs)


def _coerce_dtype(name: str, arr: np.ndarray) -> np.ndarray:
    """Cast an array to the mintpy-canonical dtype for *name*."""
    if name == "waterMask":
        return arr.astype(np.bool_)
    if name in _IFGRAM_3D:
        return arr.astype(_IFGRAM_3D[name])
    if name in _GEOM_ASSET_MAP.values():
        return arr.astype(np.float32)
    return arr


def _write_ifgram_stack(
    path: Path,
    ifgs: FrameInterferogramCollection,
    geometry: FrameGeometry | None,
    *,
    overwrite: bool,
) -> None:
    """Write ``inputs/ifgramStack.h5`` from a FrameInterferogramCollection."""
    try:
        import h5py
    except ImportError as e:
        msg = (
            "h5py is required for to_mintpy(). Install it with: "
            "pip install h5py (or the 'hdf5' extra)."
        )
        raise ImportError(msg) from e

    if path.exists():
        if not overwrite:
            msg = f"{path} already exists. Pass overwrite=True to replace it."
            raise FileExistsError(msg)
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)

    pairs_obj = ifgs.pairs()
    pair_names: list[str] = pairs_obj.to_names().tolist()
    if not pair_names:
        msg = "No interferogram pairs found; cannot build ifgramStack.h5."
        raise ValueError(msg)

    # Determine common grid shape from the first available unwrapPhase.
    ref_grid = None
    for pname in pair_names:
        if ifgs.exists(pname, "unw_phase"):
            from .raster_io import read_geogrid

            ref_grid = read_geogrid(ifgs.path(pname, "unw_phase"))
            break
    if ref_grid is None:
        msg = "No unwrapped-phase asset found; cannot determine grid shape."
        raise ValueError(msg)
    length = ref_grid.height
    width = ref_grid.width

    m = len(pair_names)
    logger.info(
        "Writing ifgramStack.h5: %d pairs, shape=(%d, %d)",
        m,
        length,
        width,
    )

    # Collect per-pair metadata (dates, bperp).
    date_arr = np.empty((m, 2), dtype="S8")
    bperp_arr = np.zeros((m,), dtype=np.float32)
    for i, pname in enumerate(pair_names):
        parts = pname.split("_")
        if len(parts) != 2:
            msg = f"Pair name {pname!r} is not YYYYMMDD_YYYYMMDD."
            raise ValueError(msg)
        date_arr[i, 0] = parts[0].encode("ascii")
        date_arr[i, 1] = parts[1].encode("ascii")
        try:
            item = ifgs.item(pname)
            baseline = item.get("baseline")
            if baseline is not None:
                bperp_arr[i] = float(baseline)
        except Exception:
            logger.debug("No item.json baseline for pair %s; using 0.0", pname)

    # Decide which 3D datasets to write based on availability.
    asset_datasets: list[tuple[str, str]] = []
    for asset_name, ds_name in _IFGRAM_ASSET_MAP.items():
        if any(ifgs.exists(p, asset_name) for p in pair_names):
            asset_datasets.append((asset_name, ds_name))

    attrs = _ifgram_attrs(geometry, length, width)

    with h5py.File(path, "w") as f:
        # Required 1D/2D datasets.
        f.create_dataset("date", data=date_arr, chunks=True)
        f.create_dataset("bperp", data=bperp_arr, chunks=True)
        f.create_dataset("dropIfgram", data=np.ones((m,), dtype=np.bool_), chunks=True)

        # 3D datasets.
        for asset_name, ds_name in asset_datasets:
            comp = "lzf" if ds_name == "connectComponent" else "gzip"
            ds = f.create_dataset(
                ds_name,
                shape=(m, length, width),
                maxshape=(None, length, width),
                dtype=_IFGRAM_3D[ds_name],
                chunks=True,
                compression=comp,
            )
            if ds_name in _DS_UNIT:
                ds.attrs["UNIT"] = _DS_UNIT[ds_name]

            for i, pname in enumerate(pair_names):
                if not ifgs.exists(pname, asset_name):
                    logger.warning(
                        "Asset %r missing for pair %r; filling with NaN.",
                        asset_name,
                        pname,
                    )
                    ds[i] = np.nan
                    continue
                da = ifgs.open(pname, asset_name, masked=True, chunks=None)
                arr = np.asarray(da.values)
                if arr.shape != (length, width):
                    msg = (
                        f"Pair {pname!r} asset {asset_name!r} shape {arr.shape} "
                        f"!= expected ({length}, {width})."
                    )
                    raise ValueError(msg)
                ds[i] = _coerce_dtype(ds_name, arr)

        # Root attributes (stringified, mintpy convention).
        for key, value in attrs.items():
            f.attrs[key] = value

    logger.info("Wrote %s", path)


def _write_geometry(
    path: Path,
    geometry: FrameGeometry,
    *,
    overwrite: bool,
) -> None:
    """Write ``inputs/geometryGeo.h5`` from a FrameGeometry."""
    try:
        import h5py
    except ImportError as e:
        msg = (
            "h5py is required for to_mintpy(). Install it with: "
            "pip install h5py (or the 'hdf5' extra)."
        )
        raise ImportError(msg) from e

    if path.exists():
        if not overwrite:
            msg = f"{path} already exists. Pass overwrite=True to replace it."
            raise FileExistsError(msg)
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)

    attrs = _geo_attrs_from_geometry(geometry)
    meta = geometry._require_metadata()
    length = int(meta["height"])
    width = int(meta["width"])

    written: list[str] = []
    with h5py.File(path, "w") as f:
        for asset_name, ds_name in _GEOM_ASSET_MAP.items():
            if not geometry.exists(asset_name):
                continue
            da = geometry.open(asset_name, masked=True, chunks=None)
            arr = np.asarray(da.values)
            if arr.shape != (length, width):
                msg = (
                    f"Geometry asset {asset_name!r} shape {arr.shape} "
                    f"!= expected ({length}, {width})."
                )
                raise ValueError(msg)
            arr = _coerce_dtype(ds_name, arr)
            ds = f.create_dataset(ds_name, data=arr, chunks=True, compression="gzip")
            if ds_name in _DS_UNIT:
                ds.attrs["UNIT"] = _DS_UNIT[ds_name]
            written.append(ds_name)

        for key, value in attrs.items():
            f.attrs[key] = value

    logger.info("Wrote %s (datasets: %s)", path, ", ".join(written) or "none")


def build_mintpy(
    *,
    geometry: FrameGeometry | None,
    interferograms: FrameInterferogramCollection,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write a MintPy-ready project directory from frame components.

    This is the inverse of
    :class:`faninsar.datasets.frame.discovery.mintpy.MintPyDiscoverer`:
    it produces ``inputs/ifgramStack.h5`` and ``inputs/geometryGeo.h5`` ready
    for ``mintpy.load_data`` / ``mintpy.smallbaselineApp``.

    Parameters
    ----------
    geometry : FrameGeometry or None
        Geometry assets. When provided, ``inputs/geometryGeo.h5`` is written
        alongside the interferogram stack. May be *None* (geometry-free export).
    interferograms : FrameInterferogramCollection
        Interferogram collection. Must contain at least one pair with an
        ``unw_phase`` asset.
    output_dir : str or Path
        Output project directory. An ``inputs/`` subdirectory is created
        inside it.
    overwrite : bool
        If *True*, overwrite existing HDF5 files.

    Returns
    -------
    Path
        The *output_dir* path.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If no pairs or no unwrapped-phase assets are found.
    FileExistsError
        If a target file exists and *overwrite* is *False*.

    Examples
    --------
    >>> from faninsar.datasets.frame import build_mintpy
    >>> out = build_mintpy(
    ...     geometry=frame.geometry,
    ...     interferograms=frame.interferograms,
    ...     output_dir="mintpy_project",
    ...     overwrite=True,
    ... )

    """
    out = Path(output_dir)
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    _write_ifgram_stack(
        inputs / "ifgramStack.h5",
        interferograms,
        geometry,
        overwrite=overwrite,
    )

    if geometry is not None:
        _write_geometry(inputs / "geometryGeo.h5", geometry, overwrite=overwrite)

    # Minimal MintPy project metadata for traceability.
    project_meta = {
        "type": "MintPyProject",
        "processor": "faninsar",
        "inputs": ["ifgramStack.h5"]
        + (["geometryGeo.h5"] if geometry is not None else []),
        "pair_count": len(interferograms.pairs().to_names().tolist()),
    }
    save_json(project_meta, out / "mintpy_project.json")
    logger.info("MintPy project written to %s", out)
    return out
