"""Unified frame-level InSAR product combining geometry and interferograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger

from .discovery import discover_hyp3_geometry_product
from .geometry import FrameGeometry
from .interferogram import FrameInterferogramCollection
from .timeseries import FrameTimeSeries

if TYPE_CHECKING:
    from datetime import datetime

    import pystac

    from faninsar.datasets.frame.remote import RemoteFrame
    from faninsar.datasets.geogrid import GeoGrid

logger = setup_logger(__name__)


def _grid_from_geometry(geometry: FrameGeometry) -> GeoGrid | None:
    """Build a :class:`GeoGrid` from a FrameGeometry's metadata.

    Returns ``None`` if the metadata lacks the required fields.
    """
    meta = geometry.metadata
    if meta is None:
        return None
    bounds = meta.get("bounds")
    crs = meta.get("crs")
    height = meta.get("height")
    width = meta.get("width")
    if not all(v is not None for v in (bounds, crs, height, width)):
        return None
    from faninsar.datasets.geogrid import GeoGrid

    return GeoGrid.from_bounds(bounds, crs=crs, shape=(height, width), tight=True)


def _json_safe(obj: Any) -> Any:
    """Recursively convert a metadata dict to JSON-serialisable types.

    Needed because Zarr attrs must be JSON-safe (no tuples, no Path, no numpy
    scalars). Tuples become lists; Path/numpy scalars become str/float.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:  # pragma: no cover - defensive
        pass
    return obj


def _store_path(store: Any) -> str:
    """Best-effort extraction of a filesystem path from a store argument."""
    if isinstance(store, str | Path):
        return str(store)
    for attr in ("path", "root", "url", "base"):
        val = getattr(store, attr, None)
        if val is not None:
            return str(val)
    return "."


def _write_ifg_zarr_cube(
    ifgs_dir: Path,
    name: str,
    data: np.ndarray,
    pair_names: list[str],
    *,
    chunks: Any = "auto",
) -> None:
    """Write an interferogram stack array as a (pair, y, x) Zarr cube."""
    import xarray as xr

    da = xr.DataArray(
        data.astype(np.float32),
        dims=("pair", "y", "x"),
        coords={"pair": pair_names},
        name=name,
    )
    if chunks is not None:
        da = da.chunk(chunks)
    da.to_dataset(name=name).to_zarr(
        str(ifgs_dir / f"{name}.zarr"), mode="w", consolidated=False
    )


def _write_ifg_metadata(
    ifgs_dir: Path,
    pair_names: list[str],
    bperp: np.ndarray | None,
    length: int,
    width: int,
) -> None:
    """Write item.json per pair and interferograms_index.json."""
    from .metadata import (
        build_interferograms_index,
        build_item_metadata,
        resolve_geometry_href,
        save_json,
    )

    assets_by_pair: dict[str, list[str]] = {}
    for i, pname in enumerate(pair_names):
        parts = pname.split("_")
        ref_date = parts[0] if len(parts) >= 2 else ""
        sec_date = parts[1] if len(parts) >= 2 else ""
        pair_dir = ifgs_dir / pname
        pair_dir.mkdir(parents=True, exist_ok=True)
        assets_present = ["unw_phase"]
        if (ifgs_dir / "coherence.zarr").exists():
            assets_present.append("coherence")
        assets_by_pair[pname] = assets_present
        item = build_item_metadata(
            pair_name=pname,
            reference_date=ref_date,
            secondary_date=sec_date,
            grid={
                "crs": "EPSG:4326",
                "width": width,
                "height": length,
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, length],
                "bounds": [0.0, 0.0, width, length],
                "resolution": [1.0, 1.0],
            },
            assets={
                a: {"href": f"{a}.zarr", "dtype": "float32"} for a in assets_present
            },
            geometry_href=resolve_geometry_href(ifgs_dir),
            source_processor="mintpy",
            baseline=float(bperp[i]) if bperp is not None and i < len(bperp) else None,
            value_ranges={"unw_phase": (-3.14159265, 3.14159265)},
        )
        save_json(item, pair_dir / "item.json")

    index = build_interferograms_index(
        pair_count=len(pair_names),
        pairs=pair_names,
        assets_by_pair=assets_by_pair,
        common_grid=True,
        geometry_href=resolve_geometry_href(ifgs_dir),
    )
    save_json(index, ifgs_dir / "interferograms_index.json")


def _write_geometry_zarr(
    geom_dir: Path, geom_h5: Path, length: int, width: int
) -> None:
    """Write geometry assets as Zarr arrays from a MintPy geometryGeo.h5."""
    import h5py
    import xarray as xr

    field_map = {
        "incidenceAngle": "incidence",
        "azimuthAngle": "azimuth",
        "height": "dem",
        "waterMask": "water_mask",
    }
    assets_written: dict[str, dict[str, Any]] = {}
    with h5py.File(geom_h5, "r") as f:
        for src_name, dst_name in field_map.items():
            if src_name not in f:
                continue
            arr = f[src_name][:]
            da = xr.DataArray(
                arr.astype(np.float32),
                dims=("y", "x"),
                name=dst_name,
            )
            da.to_dataset(name=dst_name).to_zarr(
                str(geom_dir / f"{dst_name}.zarr"), mode="w", consolidated=False
            )
            assets_written[dst_name] = {"href": f"{dst_name}.zarr", "dtype": "float32"}

    if assets_written:
        from .metadata import build_geometry_metadata, save_json

        meta = build_geometry_metadata(
            crs="EPSG:4326",
            width=width,
            height=length,
            transform=[1.0, 0.0, 0.0, 0.0, -1.0, length],
            bounds=[0.0, 0.0, width, length],
            resolution=(1.0, 1.0),
            assets=assets_written,
            processing={"source": "mintpy", "format": "hdf5"},
        )
        save_json(meta, geom_dir / "geometry.json")


class Frame:
    """A standardized InSAR frame: geometry + interferogram collection.

    This class is a convenience facade that combines
    :class:`FrameGeometry` and :class:`FrameInterferogramCollection`
    into a single entry point.

    Parameters
    ----------
    root : str or Path
        Path to the frame directory containing ``geometry/`` and/or
        ``ifg/`` subdirectories.

    Examples
    --------
    Create from HyP3 products::

        frame = Frame.from_hyp3(out_dir="frame", root_dir="across_year")
        frame.to_stac(output_dir="stac_catalog")

    Reuse an existing geometry::

        frame = Frame.from_hyp3(
            out_dir="frame",
            root_dir="across_year",
            geometry=existing_geom,
        )

    Load an existing frame::

        frame = Frame("frame")
        frame.geometry.open("incidence")
        frame.interferograms.open_stack("coherence")

    """

    def __init__(self, root: str | Path) -> None:
        """Load an existing frame from *root*.

        Parameters
        ----------
        root : str or Path
            Frame directory (containing ``geometry/`` and/or ``ifg/``).

        """
        self._root = Path(root)
        if not self._root.is_dir():
            msg = f"Frame directory not found: {self._root}"
            raise FileNotFoundError(msg)

        self._geometry: FrameGeometry | None = None
        geom_dir = self._root / "geometry"
        if geom_dir.is_dir():
            self._geometry = FrameGeometry(geom_dir)

        self._interferograms: FrameInterferogramCollection | None = None
        # Prefer the new interferograms/ directory; fall back to legacy ifg/.
        ifgs_dir = self._root / "interferograms"
        if not ifgs_dir.is_dir():
            legacy = self._root / "ifg"
            if legacy.is_dir():
                ifgs_dir = legacy
        if ifgs_dir.is_dir():
            self._interferograms = FrameInterferogramCollection(ifgs_dir)

        self._timeseries: FrameTimeSeries | None = None
        ts_dir = self._root / "timeseries"
        if ts_dir.is_dir():
            self._timeseries = FrameTimeSeries(ts_dir)

    @property
    def root(self) -> Path:
        """Root directory of the frame."""
        return self._root

    @property
    def geometry(self) -> FrameGeometry | None:
        """Geometry assets, or *None* if ``geometry/`` is absent."""
        return self._geometry

    @property
    def interferograms(self) -> FrameInterferogramCollection | None:
        """Interferogram collection, or *None* if ``ifg/`` is absent."""
        return self._interferograms

    @property
    def timeseries(self) -> FrameTimeSeries | None:
        """Time-series product, or *None* if ``timeseries/`` is absent."""
        return self._timeseries

    def to_ifg_stack(self, *, pairs: Any = None) -> Any:
        """Build an :class:`~faninsar.processing.contracts.ifg.InterferogramStack`.

        This is the processing ↔ timeseries seam producer for Frame products.
        Unwrapped phase is stacked as ``(n_pair, n_pixel)`` from the
        ``unw_phase`` (or equivalent) assets.

        Parameters
        ----------
        pairs : Pairs, optional
            Subset of pairs; defaults to all pairs in the collection.

        Returns
        -------
        InterferogramStack
            Seam product accepted by :class:`~faninsar.timeseries.NSBASSolver`.

        """
        from faninsar.processing.contracts.ifg import InterferogramStack

        if self._interferograms is None:
            msg = "Frame has no interferogram collection"
            logger.error(msg)
            raise ValueError(msg)

        ifg = self._interferograms
        pair_obj = pairs if pairs is not None else ifg.pairs()
        names = list(pair_obj.names)

        # Prefer open_stack when available for dense arrays
        unw = None
        coh = None
        if hasattr(ifg, "open_stack"):
            try:
                unw_da = ifg.open_stack("unw_phase")
                unw = np.asarray(unw_da.values, dtype=np.float64)
                if unw.ndim == 3:
                    unw = unw.reshape(unw.shape[0], -1)
            except Exception:
                unw = None
            try:
                coh_da = ifg.open_stack("coherence")
                coh = np.asarray(coh_da.values, dtype=np.float64)
                if coh.ndim == 3:
                    coh = coh.reshape(coh.shape[0], -1)
            except Exception:
                coh = None

        if unw is None:
            msg = "could not load unwrapped phase stack from Frame"
            logger.error(msg)
            raise ValueError(msg)

        return InterferogramStack.from_unwrapped(
            stack_id=str(self._root),
            pairs=pair_obj,
            unwrapped=unw,
            coherence=coh,
        )

    @classmethod
    def from_hyp3(
        cls,
        out_dir: str | Path,
        root_dir: str | Path,
        *,
        geometry: FrameGeometry | None = None,
        assets: tuple[str, ...] = ("unw_phase", "coherence"),
        include_optional_assets: bool = False,
        pairs: Any = None,
        max_pairs: int | None = None,
        overwrite: bool = False,
    ) -> Frame:
        """Create a full frame from HyP3 GAMMA products.

        Discovers interferograms from *root_dir*. If *geometry* is not
        provided, geometry rasters are auto-discovered from the first
        product directory in *root_dir*.

        Parameters
        ----------
        out_dir : str or Path
            Output directory for the frame layout.
        root_dir : str or Path
            Root directory of HyP3 interferogram products.
        geometry : FrameGeometry, optional
            Pre-existing geometry to reuse. If *None*, geometry rasters
            are auto-discovered from *root_dir*.
        assets : tuple of str
            Interferogram assets to standardize.
        include_optional_assets : bool
            If *True*, discover optional HyP3 assets.
        pairs : Pairs, optional
            Subset of pairs to standardize.
        max_pairs : int, optional
            Limit the number of pairs.
        overwrite : bool
            If *True*, overwrite existing outputs.

        Returns
        -------
        Frame

        """
        if geometry is None:
            product_dir = discover_hyp3_geometry_product(root_dir)
            logger.info("Auto-discovered geometry from: %s", product_dir)
            geometry = FrameGeometry.from_hyp3(
                out_dir=out_dir,
                product_dir=product_dir,
                overwrite=overwrite,
            )

        ifgs = FrameInterferogramCollection.from_hyp3(
            out_dir=out_dir,
            root_dir=root_dir,
            geometry=geometry,
            pairs=pairs,
            assets=assets,
            include_optional_assets=include_optional_assets,
            max_pairs=max_pairs,
            overwrite=overwrite,
        )

        frame = cls(out_dir)
        frame._geometry = geometry
        frame._interferograms = ifgs
        return frame

    def to_mintpy(
        self,
        output_dir: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Export this frame to a MintPy-ready project directory.

        Writes the MintPy HDF5 layout that
        :class:`~faninsar.datasets.frame.discovery.mintpy.MintPyDiscoverer`
        reads back::

            <output_dir>/
            └── inputs/
                ├── ifgramStack.h5      # all interferograms + coherence
                └── geometryGeo.h5      # geometry layers (when available)

        The resulting directory can be loaded directly with
        ``mintpy.load_data`` / ``mintpy.smallbaselineApp``.

        Parameters
        ----------
        output_dir : str or Path, optional
            Output project directory. Defaults to ``<root>/mintpy/``.
        overwrite : bool
            If *True*, overwrite existing HDF5 files.

        Returns
        -------
        Path
            The output project directory.

        Raises
        ------
        FileNotFoundError
            If no interferogram collection is present on this frame.
        ImportError
            If ``h5py`` is not installed.

        Examples
        --------
        >>> frame = Frame("frame")
        >>> frame.to_mintpy(output_dir="mintpy_project", overwrite=True)

        """
        from .mintpy import build_mintpy

        if self._interferograms is None:
            msg = "No interferogram collection found; cannot export to MintPy."
            raise FileNotFoundError(msg)

        out = Path(output_dir) if output_dir is not None else self._root / "mintpy"
        return build_mintpy(
            geometry=self._geometry,
            interferograms=self._interferograms,
            output_dir=out,
            overwrite=overwrite,
        )

    @classmethod
    def from_mintpy(
        cls,
        mintpy_dir: str | Path,
        *,
        output_dir: str | Path | None = None,
        chunks: dict[str, int] | int | Literal["auto"] | None = "auto",
        overwrite: bool = False,
    ) -> Frame:
        """Load a MintPy project as a Frame.

        Reads ``inputs/ifgramStack.h5`` and ``inputs/geometryGeo.h5`` and
        produces the standard Frame on-disk layout (geometry/ +
        interferograms/) with Zarr cubes for the interferogram stacks — no
        per-pair COGs are materialised. The interferogram assets become lazy
        ``(pair, y, x)`` Zarr cubes, so :meth:`open_stack` returns
        dask-backed arrays.

        This is the reverse of :meth:`to_mintpy`, enabling the round-trip
        ``build_mintpy(frame)`` → ``from_mintpy(result)``.

        Parameters
        ----------
        mintpy_dir : str or Path
            Path to the MintPy project directory (containing ``inputs/``).
        output_dir : str or Path, optional
            Where to write the Frame layout. Defaults to a sibling ``frame/``
            directory under *mintpy_dir*.
        chunks : dict, int, ``"auto"``, or None
            Dask chunk spec for the written Zarr cubes. Default ``"auto"``.
        overwrite : bool
            Overwrite an existing output directory.

        Returns
        -------
        Frame
            A frame backed by lazy Zarr reads of the MintPy HDF5 data.

        Raises
        ------
        FileNotFoundError
            If ``inputs/ifgramStack.h5`` is missing.

        """
        import h5py

        mintpy_dir = Path(mintpy_dir)
        ifg_h5 = mintpy_dir / "inputs" / "ifgramStack.h5"
        geom_h5 = mintpy_dir / "inputs" / "geometryGeo.h5"
        if not ifg_h5.exists():
            msg = f"MintPy ifgramStack.h5 not found: {ifg_h5}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        out = Path(output_dir) if output_dir is not None else mintpy_dir / "frame"
        if out.exists() and overwrite:
            import shutil

            shutil.rmtree(out)
        ifgs_dir = out / "interferograms"
        geom_dir = out / "geometry"
        ifgs_dir.mkdir(parents=True, exist_ok=True)
        geom_dir.mkdir(parents=True, exist_ok=True)

        # --- Parse interferogram stack from ifgramStack.h5 ---
        with h5py.File(ifg_h5, "r") as f:
            date = f["date"][:]  # (m, 2) S8
            bperp = f["bperp"][:] if "bperp" in f else None
            has_coh = "coherence" in f
            length = int(f.attrs.get("LENGTH", f["unwrapPhase"].shape[1]))
            width = int(f.attrs.get("WIDTH", f["unwrapPhase"].shape[2]))

            pair_names: list[str] = []
            for i in range(date.shape[0]):
                ref = (
                    date[i, 0].decode()
                    if isinstance(date[i, 0], bytes)
                    else str(date[i, 0])
                )
                sec = (
                    date[i, 1].decode()
                    if isinstance(date[i, 1], bytes)
                    else str(date[i, 1])
                )
                pair_names.append(f"{ref}_{sec}")

            # Read arrays lazily-backed; h5py datasets support slicing.
            unw = f["unwrapPhase"][:]  # (m, L, W)
            coh = f["coherence"][:] if has_coh else None

        # Write interferogram Zarr cubes (no COGs).
        if unw.size > 0:
            _write_ifg_zarr_cube(ifgs_dir, "unw_phase", unw, pair_names, chunks=chunks)
        if coh is not None:
            _write_ifg_zarr_cube(ifgs_dir, "coherence", coh, pair_names, chunks=chunks)

        # Write per-pair metadata sidecars (item.json + index).
        _write_ifg_metadata(ifgs_dir, pair_names, bperp, length, width)

        # --- Parse geometry from geometryGeo.h5 (optional) ---
        if geom_h5.exists():
            _write_geometry_zarr(geom_dir, geom_h5, length, width)

        return cls(out)

    def to_zarr(
        self,
        store: str | Path | Any,
        *,
        overwrite: bool = False,
        chunks: dict[str, int] | None = None,
        assets: tuple[str, ...] = ("unw_phase",),
    ) -> Any:
        """Serialize the frame to a Zarr-based store.

        Writes interferogram stacks (per asset), geometry arrays, and the
        time-series displacement cube as Zarr arrays, with metadata
        (geometry.json, interferograms_index, timeseries.json) stored as Zarr
        group attributes. Supports direct cloud writes via
        :class:`zarr.storage.FSStore` / ``fsspec`` URLs (e.g. ``s3://...``)
        without GDAL.

        Parameters
        ----------
        store : str or Path or zarr.Store
            Target store. Local path, an ``s3://`` / ``gcs://`` URL, or an
            already-constructed zarr/fsspec store.
        overwrite : bool
            Overwrite an existing store.
        chunks : dict, optional
            Chunk spec for arrays. Defaults to sensible per-dimension chunks.
        assets : tuple of str
            Interferogram assets to include as ``(pair, y, x)`` cubes.

        Returns
        -------
        zarr store
            The store handle (for chaining / inspection).

        """
        import xarray as xr

        store_arg = self._resolve_zarr_store(store)
        if overwrite:
            self._zarr_clear(store_arg)

        attrs: dict[str, Any] = {"frame_type": "Frame", "frame_root": str(self._root)}

        if self._geometry is not None:
            meta = self._geometry.metadata
            if meta is not None:
                attrs["geometry_metadata"] = _json_safe(meta)
            # Write each geometry asset that exists as a 2-D Zarr array.
            geom = self._geometry
            for gname in ("incidence", "azimuth", "heading", "dem", "water_mask"):
                try:
                    geom.exists(gname)
                except Exception:
                    continue
                if not geom.exists(gname):
                    continue
                try:
                    gda = geom.open(gname, chunks=chunks or "auto")
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("Skipping geometry asset '%s': %s", gname, e)
                    continue
                if gda.ndim == 3:
                    gda = gda.squeeze("band", drop=True)
                # Write each asset to its own sub-store path to avoid xarray's
                # rejection of group= for store-like (FSMap) objects. Each asset
                # becomes a standalone Zarr group at geometry/<gname>.
                sub_store = self._resolve_substore(store, f"geometry/{gname}")
                gda.to_dataset(name=gname).to_zarr(
                    sub_store, mode="w", consolidated=False
                )

        if self._interferograms is not None:
            idx = self._interferograms.summary()
            attrs["interferograms_summary"] = _json_safe(idx)
            for asset_name in assets:
                try:
                    pn0 = self._interferograms.pairs().to_names()[0]
                except Exception:
                    continue
                if not self._interferograms.exists(pn0, asset_name):
                    continue
                try:
                    stack = self._interferograms.open_stack(asset_name, chunks="auto")
                except (ValueError, FileNotFoundError) as e:
                    logger.info("Skipping interferogram asset '%s': %s", asset_name, e)
                    continue
                if "band" in stack.dims:
                    stack = stack.squeeze("band", drop=True)
                chunk_spec = {
                    d: (1 if d == "pair" else 512)
                    for d in ("pair", "y", "x")
                    if d in stack.dims
                }
                if chunk_spec:
                    stack = stack.chunk(chunk_spec)
                sub_store = self._resolve_substore(
                    store, f"interferograms/{asset_name}"
                )
                stack.to_dataset(name=asset_name).to_zarr(
                    sub_store, mode="w", consolidated=False
                )

        if self._timeseries is not None:
            ts_meta = self._timeseries.metadata
            if ts_meta is not None:
                attrs["timeseries_metadata"] = _json_safe(ts_meta)
            disp = self._timeseries.root / "displacement.zarr"
            if disp.exists():
                attrs["timeseries_displacement_zarr"] = "timeseries/displacement.zarr"

        # Write top-level attrs to the root group.
        root_group = xr.Dataset(attrs=attrs)
        root_group.to_zarr(store_arg, mode="a", consolidated=False)
        return store_arg

    @staticmethod
    def _resolve_substore(store: str | Path | Any, sub: str) -> Any:
        """Return a sub-store handle at ``<store>/<sub>``.

        For local paths this is the joined path. For fsspec-backed URLs it is a
        fresh FSMap at the joined URL. For already-built stores it returns the
        original (callers must then use ``group=`` — not the FSMap case).
        """
        if isinstance(store, str | Path):
            sp = str(store)
            if "://" in sp:
                import fsspec

                fs, path = fsspec.url_to_fs(sp)
                joined = path.rstrip("/") + "/" + sub
                return fs.get_mapper(joined)
            return str(Path(sp) / sub)
        return store

    @staticmethod
    def _resolve_zarr_store(store: str | Path | Any) -> Any:
        """Normalise a store argument for xarray.to_zarr / zarr.open_group."""
        # An already-built store / mapper — pass through.
        if hasattr(store, "keys") or hasattr(store, "listdir"):
            return store
        sp = str(store)
        # fsspec-backed cloud URLs (s3://, gcs://, memory://, etc.).
        if "://" in sp:
            try:
                import fsspec

                fs, path = fsspec.url_to_fs(sp)
                return fs.get_mapper(path)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("fsspec mapping failed for '%s': %s", sp, e)
                return sp
        return sp

    @staticmethod
    def _zarr_clear(store: Any) -> None:
        """Best-effort removal of an existing zarr store for overwrite."""
        try:
            import zarr

            zarr.open_group(store, mode="w", storage_options={}).store.clear()
        except Exception:
            # For fsspec-backed stores, attempt path-based removal.
            try:
                sp = str(store)
                if "://" in sp:
                    import fsspec

                    fs, path = fsspec.url_to_fs(sp)
                    if fs.exists(path):
                        fs.rm(path, recursive=True)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Could not clear store %s: %s", store, e)

    @classmethod
    def from_zarr(cls, store: str | Path | Any) -> Frame:
        """Load a frame from a Zarr-based store.

        Reads arrays lazily (dask-backed) and reconstructs metadata from the
        group attributes. The returned :class:`Frame` points at the store; no
        COG files are materialised.

        Parameters
        ----------
        store : str or Path or zarr.Store
            Source store (local path, ``s3://`` URL, or store handle).

        Returns
        -------
        Frame
            A frame backed by lazy Zarr reads.

        """
        import tempfile

        import xarray as xr

        store_arg = cls._resolve_zarr_store(store)
        root_ds = xr.open_zarr(store_arg, consolidated=False)
        attrs = dict(root_ds.attrs)
        geom_meta = attrs.get("geometry_metadata")

        # Materialise a lightweight on-disk frame skeleton so the existing
        # Frame/FrameGeometry/FrameInterferogramCollection constructors (which
        # are path-oriented) can read the metadata sidecars. Arrays stay lazy
        # via the Zarr references; only JSON metadata is written.
        tmp = Path(tempfile.mkdtemp(prefix="frame_zarr_"))
        if geom_meta is not None:
            geom_dir = tmp / "geometry"
            geom_dir.mkdir(parents=True, exist_ok=True)
            from .metadata import save_json

            save_json(geom_meta, geom_dir / "geometry.json")

        ifg_summary = attrs.get("interferograms_summary")
        if ifg_summary is not None:
            ifgs_dir = tmp / "interferograms"
            ifgs_dir.mkdir(parents=True, exist_ok=True)
            from .metadata import build_interferograms_index, save_json

            pairs_list = ifg_summary.get("pairs", [])
            idx = build_interferograms_index(
                pair_count=ifg_summary.get("pair_count", len(pairs_list)),
                pairs=pairs_list,
                assets_by_pair=ifg_summary.get("assets_by_pair", {}),
                common_grid=ifg_summary.get("common_grid", False),
            )
            save_json(idx, ifgs_dir / "interferograms_index.json")
            # Create placeholder pair directories so the path-oriented
            # FrameInterferogramCollection.pairs() directory scan succeeds.
            for pname in pairs_list:
                (ifgs_dir / pname).mkdir(exist_ok=True)

        return cls(tmp)

    def plot_displacement(
        self,
        point: tuple[float, float] | None = None,
        *,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the displacement time series at a point (plots.frame wrapper)."""
        from faninsar.plots.frame import plot_displacement_timeseries

        if self._timeseries is None:
            msg = "Frame has no timeseries; cannot plot displacement."
            raise FileNotFoundError(msg)
        import xarray as xr

        disp_path = self._timeseries.root / "displacement.zarr"
        if not disp_path.exists():
            msg = f"displacement.zarr not found: {disp_path}"
            raise FileNotFoundError(msg)
        da = xr.open_zarr(str(disp_path), consolidated=False)["displacement"]
        return plot_displacement_timeseries(da, point=point, ax=ax, **kwargs)

    def plot_velocity(
        self,
        *,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Plot the velocity map (thin wrapper over plots.frame)."""
        from faninsar.plots.frame import plot_velocity

        if self._timeseries is None:
            msg = "Frame has no timeseries; cannot plot velocity."
            raise FileNotFoundError(msg)
        import xarray as xr

        out = self._timeseries.root / "displacement.zarr"
        if out.exists():
            ds = xr.open_zarr(str(out), consolidated=False)
            if "velocity" in ds:
                return plot_velocity(ds["velocity"], ax=ax, **kwargs)
        # Fall back to velocity COG if present.
        vel_path = self._timeseries.path("velocity")
        if vel_path.exists():
            import rioxarray

            da = rioxarray.open_rasterio(vel_path, masked=True)
            return plot_velocity(da.squeeze("band", drop=True), ax=ax, **kwargs)
        msg = "No velocity asset found in timeseries."
        raise FileNotFoundError(msg)

    def to_stac(
        self,
        *,
        catalog_id: str = "insar-frame",
        description: str = "",
        temporal_extent: tuple[datetime | None, datetime | None] | None = None,
        output_dir: str | Path | None = None,
        catalog_type: pystac.CatalogType | None = None,
    ) -> pystac.Catalog:
        """Generate a STAC Catalog from this frame.

        Parameters
        ----------
        catalog_id : str
            STAC Catalog id.
        description : str
            Catalog description.
        temporal_extent : tuple of (start, end), optional
            Temporal extent. If *None*, auto-detected.
        output_dir : str or Path, optional
            If provided, save the catalog here.
        catalog_type : pystac.CatalogType, optional
            STAC catalog type.

        Returns
        -------
        pystac.Catalog

        """
        if self._geometry is None:
            msg = "No geometry found; cannot build STAC catalog."
            raise FileNotFoundError(msg)

        return self._geometry.to_stac(
            ifgs=self._interferograms,
            catalog_id=catalog_id,
            description=description,
            temporal_extent=temporal_extent,
            output_dir=output_dir,
            catalog_type=catalog_type,
        )

    @classmethod
    def from_stac(
        cls,
        catalog_path: str | Path,
        *,
        frame_root: str | Path | None = None,
    ) -> Frame:
        """Reconstruct a Frame from a faninsar-generated local STAC catalog.

        Reverse of :meth:`to_stac`. The on-disk ``frame/geometry/`` and
        ``frame/interferograms/`` layout is the source of truth; the catalog
        is used only as an index to locate those directories. Assets are
        assumed to already be on disk — this is a metadata re-mount, not a
        download.

        Parameters
        ----------
        catalog_path : str or Path
            Path to the ``catalog.json`` written by ``Frame.to_stac()``.
        frame_root : str or Path, optional
            Explicit frame root directory. **Recommended** when the STAC
            catalog was written to a separate output directory
            (``to_stac(output_dir=...)`` with a different path than the
            frame root). When provided, ``geometry/`` and ``interferograms/``
            are looked up directly under *frame_root* — no href parsing.
            When *None*, the frame root is inferred from the catalog's
            asset hrefs (less robust if the catalog was normalised into a
            separate tree).

        Returns
        -------
        Frame
            A Frame whose :attr:`geometry` / :attr:`interferograms`
            properties point at the on-disk COG assets indexed by the
            catalog.

        Raises
        ------
        FileNotFoundError
            If the catalog or the referenced frame directories are missing.
        ValueError
            If the catalog does not look like a faninsar-standard frame
            catalog, or contains remote asset hrefs (use
            :meth:`open_remote` for those, M5).

        Examples
        --------
        >>> frame = Frame.from_hyp3(out_dir="frame", root_dir="data")
        >>> frame.to_stac(output_dir="stac_catalog")
        >>> loaded = Frame.from_stac("stac_catalog/catalog.json", frame_root="frame")

        """
        from .stac import _stac_to_frame_meta

        meta = _stac_to_frame_meta(catalog_path, frame_root=frame_root)
        frame_root_resolved = meta["frame_root"]
        if not frame_root_resolved.is_dir():
            msg = (
                f"Frame root resolved from STAC catalog does not exist: "
                f"{frame_root_resolved}"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        frame = cls(frame_root_resolved)

        geometry_dir = meta["geometry_dir"]
        if geometry_dir is not None and geometry_dir.is_dir():
            frame._geometry = FrameGeometry(geometry_dir)
        elif frame._geometry is None and geometry_dir is not None:
            logger.warning(
                "STAC catalog references geometry dir %s but it is missing.",
                geometry_dir,
            )

        interferograms_dir = meta["interferograms_dir"]
        if interferograms_dir is not None and interferograms_dir.is_dir():
            frame._interferograms = FrameInterferogramCollection(interferograms_dir)
        elif frame._interferograms is None and interferograms_dir is not None:
            logger.warning(
                "STAC catalog references interferograms dir %s but it is missing.",
                interferograms_dir,
            )

        return frame

    @classmethod
    def open_remote(
        cls,
        catalog_url: str,
        *,
        cache_dir: str | Path | None = None,
        anonymous: bool = True,
        **kwargs: Any,
    ) -> RemoteFrame:
        """Open a remote faninsar STAC catalog and lazily read assets over HTTP.

        Reads the catalog metadata over HTTP (via :mod:`pystac`) and returns
        a :class:`RemoteFrame` whose ``open()`` calls fetch COG windows on
        demand via GDAL's ``/vsicurl/`` — no full-file downloads.

        .. note::
            Requires the ``pystac`` optional dependency for catalog parsing
            and GDAL built with ``/vsicurl/`` support (standard in most
            distributions). Hugging Face Hub URLs are supported directly.

        Parameters
        ----------
        catalog_url : str
            HTTP(S) URL to a ``catalog.json``.
        cache_dir : str or Path, optional
            Local cache dir for catalog metadata. COG range reads are not
            cached by default.
        anonymous : bool
            If *True*, do not send credentials (public datasets). If *False*,
            uses the locally configured token / ``.netrc``.
        **kwargs
            Forwarded to :class:`RemoteFrame` constructor.

        Returns
        -------
        RemoteFrame
            A lazy view of the remote frame. Its ``geometry`` /
            ``interferograms`` sub-objects expose ``open()`` / ``open_stack()``
            calls that read COG windows over HTTP.

        Raises
        ------
        ImportError
            If ``pystac`` is not installed.

        """
        from .remote import RemoteFrame

        return RemoteFrame(
            catalog_url,
            cache_dir=cache_dir,
            anonymous=anonymous,
            **kwargs,
        )

    def validate(self) -> list[str]:
        """Cross-check frame-internal consistency.

        Returns
        -------
        list of str
            Human-readable issue strings. An empty list means the frame is
            internally consistent. Issues currently checked:

            - ``interferograms_index.json`` ``geometry_href`` (when present)
              matches what :func:`resolve_geometry_href` returns.
            - Every pair's ``item.json`` ``geometry_href`` is consistent.
            - When both ``geometry/`` and ``interferograms/`` exist, the
              geometry grid matches the first interferogram pair's grid
              (CRS + shape + alignment).

        This method never raises — it collects issues so callers can report
        them together.

        """
        from .metadata import resolve_geometry_href

        issues: list[str] = []

        ifgs = self._interferograms
        if ifgs is not None:
            expected_href = resolve_geometry_href(ifgs.root)

            # Collection-level index.
            idx = ifgs.index_metadata
            if idx is not None:
                actual = idx.get("geometry_href")
                if expected_href is None and actual not in (None, ""):
                    issues.append(
                        f"interferograms_index.json has geometry_href={actual!r} "
                        "but no sibling geometry/geometry.json exists."
                    )
                elif expected_href is not None and actual != expected_href:
                    issues.append(
                        f"interferograms_index.json geometry_href={actual!r} "
                        f"does not match expected {expected_href!r}."
                    )

            # Per-pair item.json.
            try:
                pairs_obj = ifgs.pairs()
                for pname in pairs_obj.to_names().tolist():
                    try:
                        item = ifgs.item(pname)
                    except Exception:
                        issues.append(f"item.json missing for pair {pname!r}.")
                        continue
                    actual = item.get("geometry_href")
                    if expected_href is not None and actual != expected_href:
                        issues.append(
                            f"item.json for pair {pname!r} geometry_href="
                            f"{actual!r} does not match expected "
                            f"{expected_href!r}."
                        )
            except Exception as e:  # pragma: no cover - defensive
                issues.append(f"Could not enumerate pairs for validation: {e}")

            # Grid alignment between geometry and interferograms.
            geom = self._geometry
            if geom is not None and geom.metadata is not None:
                try:
                    from .raster_io import validate_alignment

                    pairs_obj = ifgs.pairs()
                    names = pairs_obj.to_names().tolist()
                    if names and ifgs.exists(names[0], "unw_phase"):
                        unw_path = ifgs.path(names[0], "unw_phase")
                        if not validate_alignment(unw_path, _grid_from_geometry(geom)):
                            issues.append(
                                f"First pair {names[0]!r} unw_phase grid is "
                                "not aligned with the frame geometry grid."
                            )
                except Exception as e:  # pragma: no cover - defensive
                    issues.append(f"Grid alignment check failed: {e}")

        return issues

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the frame."""
        info: dict[str, Any] = {"root": str(self._root)}
        if self._geometry is not None:
            info["geometry"] = self._geometry.summary()
        if self._interferograms is not None:
            info["interferograms"] = self._interferograms.summary()
        if self._timeseries is not None:
            info["timeseries"] = self._timeseries.summary()
        return info

    def __repr__(self) -> str:
        """Return a short summary string."""
        parts = [f"Frame(root={self._root!r})"]
        if self._geometry is not None:
            parts.append(f"  geometry: {self._geometry.root}")
        if self._interferograms is not None:
            s = self._interferograms.summary()
            parts.append(f"  interferograms: {s['pair_count']} pairs")
        return "\n".join(parts)
