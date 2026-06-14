"""Standardized pair-level interferogram assets for one InSAR frame."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from faninsar._core.sar import Pairs
from faninsar.datasets.geogrid import GeoGrid
from faninsar.logging import setup_logger

from .exceptions import (
    GridMismatchError,
    MissingInterferogramAssetError,
    PairNotFoundError,
)
from .metadata import (
    CATEGORICAL_ASSETS,
    INTERFEROGRAM_ASSETS,
    InterferogramAssetName,
    build_interferograms_index,
    build_item_metadata,
    load_json,
    resolve_geometry_href,
    save_json,
)
from .raster_io import (
    read_geogrid,
    reproject_to_geogrid,
    write_cog,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Self

    import pystac

    from faninsar.datasets.frame.geometry import FrameGeometry
    from faninsar.datasets.ifg import InterferogramDataset

logger = setup_logger(__name__)


_PAIR_ASSET_FILENAMES: dict[str, str] = {
    name: f"{name}.cog.tif" for name in INTERFEROGRAM_ASSETS
}


def _pair_name(pair: Any) -> str:
    """Return the string name of a pair."""
    if isinstance(pair, str):
        return pair
    return str(pair)


def _index_to_records(index: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an interferograms_index dict into one record per pair.

    Each record has scalar columns (pair_name, common_grid, geometry_href,
    version, type) plus the pair's asset list serialised as a JSON string
    (parquet-friendly).
    """
    import json

    common_grid = bool(index.get("common_grid", False))
    geometry_href = index.get("geometry_href")
    version = index.get("version")
    type_ = index.get("type")
    assets_by_pair: dict[str, list[str]] = index.get("assets_by_pair", {})
    pairs: list[str] = index.get("pairs", [])
    return [
        {
            "pair_name": pname,
            "assets": json.dumps(assets_by_pair.get(pname, [])),
            "common_grid": common_grid,
            "geometry_href": geometry_href,
            "version": version,
            "type": type_,
        }
        for pname in pairs
    ]


def _records_to_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble an interferograms_index dict from per-pair records."""
    import json
    from datetime import UTC, datetime

    if not records:
        msg = "Cannot build index from empty records list."
        raise ValueError(msg)
    first = records[0]
    pairs = [r["pair_name"] for r in records]
    assets_by_pair = {r["pair_name"]: json.loads(r["assets"]) for r in records}
    return {
        "type": first.get("type", "FrameInterferogramIndex"),
        "version": first.get("version"),
        "pair_count": len(pairs),
        "pairs": pairs,
        "assets_by_pair": assets_by_pair,
        "common_grid": bool(first.get("common_grid", False)),
        "geometry_href": first.get("geometry_href"),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _load_index_parquet(path: str | Path) -> dict[str, Any]:
    """Load an interferograms_index.parquet and return the index dict.

    Raises
    ------
    ImportError
        If ``pyarrow`` is not installed.

    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised via fallback path
        msg = (
            "pyarrow is required to read interferograms_index.parquet. "
            "Install it with: pip install pyarrow (or the 'cloud' extra)."
        )
        raise ImportError(msg) from e

    path = Path(path)
    table = pq.read_table(path)
    records = table.to_pylist()
    return _records_to_index(records)


class FrameInterferogramCollection:
    """Standardized pair-level interferogram assets for one InSAR frame.

    Parameters
    ----------
    root : str or Path
        Path to the ``ifg`` directory (e.g. ``frame/ifg``).

    Examples
    --------
    Build from an existing FanInSAR dataset:

    >>> from faninsar.datasets import HyP3S1
    >>> ds = HyP3S1(root_dir="/path/to/hyp3_products")
    >>> ifgs = FrameInterferogramCollection.from_dataset(
    ...     out_dir="frame",
    ...     dataset=ds,
    ...     geometry=geom,
    ... )

    Build from HyP3 directly:

    >>> ifgs = FrameInterferogramCollection.from_hyp3(
    ...     out_dir="frame",
    ...     root_dir="/path/to/hyp3_products",
    ...     geometry=geom,
    ... )

    Open a single pair:

    >>> da = ifgs.open("20191115_20200314", "unw_phase")

    Stack a named asset across all pairs:

    >>> stack = ifgs.open_stack("coherence", chunks={"y": 512, "x": 512})

    """

    def __init__(self, root: str | Path) -> None:
        """Initialise from an existing ``interferograms`` directory."""
        self._root = Path(root)
        if not self._root.is_dir():
            msg = f"Interferogram directory not found: {self._root}"
            raise FileNotFoundError(msg)

        # Index path is resolved by _load_index() (prefers parquet, then JSON).
        self._index_path: Path = self._root / "interferograms_index.json"
        self._index: dict[str, Any] | None = None
        self._index_loaded = False
        self._load_index()

        self._pairs: Pairs | None = None

    def _load_index(self) -> None:
        """Load the collection index, preferring parquet over JSON.

        Sets ``self._index`` to the parsed dict (or ``None`` if no index
        file exists). Idempotent.
        """
        if self._index_loaded:
            return
        # Prefer parquet, then new JSON, then legacy ifg_index.json.
        parquet_path = self._root / "interferograms_index.parquet"
        if parquet_path.exists():
            try:
                self._index = _load_index_parquet(parquet_path)
            except ImportError:
                logger.warning(
                    "interferograms_index.parquet exists but pyarrow is not "
                    "installed; falling back to JSON."
                )
            else:
                self._index_path = parquet_path
                self._index_loaded = True
                return
        json_path = self._root / "interferograms_index.json"
        if not json_path.exists():
            legacy = self._root / "ifg_index.json"
            if legacy.exists():
                json_path = legacy
        self._index_path = json_path
        if json_path.exists():
            self._index = load_json(json_path)
        self._index_loaded = True

    @property
    def root(self) -> Path:
        """Root directory of the interferogram collection."""
        return self._root

    @property
    def index_metadata(self) -> dict[str, Any] | None:
        """Parsed collection index content (parquet or JSON), or None if absent."""
        return self._index

    def _require_index(self) -> dict[str, Any]:
        if self._index is None:
            if self._index_path.suffix == ".parquet":
                self._index = _load_index_parquet(self._index_path)
            else:
                self._index = load_json(self._index_path)
        return self._index

    def _discover_pair_dirs(self) -> list[Path]:
        """Return sorted list of pair subdirectories under root."""
        return sorted(
            p for p in self._root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def pairs(self) -> Pairs:
        """Return the :class:`Pairs` object for this collection."""
        if self._pairs is None:
            pair_dirs = self._discover_pair_dirs()
            pair_names = [d.name for d in pair_dirs]
            if pair_names:
                self._pairs = Pairs.from_names(pair_names)
            else:
                self._pairs = Pairs([])
        return self._pairs

    def pair_dir(self, pair: str) -> Path:
        """Return the directory for a given pair."""
        name = _pair_name(pair)
        d = self._root / name
        if not d.is_dir():
            raise PairNotFoundError(name)
        return d

    def path(self, pair: str, name: InterferogramAssetName) -> Path:
        """Return the file path for a pair asset."""
        if name not in _PAIR_ASSET_FILENAMES:
            msg = f"Unknown interferogram asset: {name}"
            raise ValueError(msg)
        return self.pair_dir(pair) / _PAIR_ASSET_FILENAMES[name]

    def exists(self, pair: str, name: InterferogramAssetName) -> bool:
        """Check if a pair asset file exists."""
        return self.path(pair, name).exists()

    def item(self, pair: str) -> dict[str, Any]:
        """Load and return the item.json for a pair."""
        item_path = self.pair_dir(pair) / "item.json"
        if not item_path.exists():
            raise PairNotFoundError(pair)
        return load_json(item_path)

    def summary(self) -> dict[str, Any]:
        """Return a collection-level summary."""
        pairs_obj = self.pairs()
        pair_list = pairs_obj.to_names().tolist() if len(pairs_obj) > 0 else []
        assets_by_pair: dict[str, list[str]] = {}
        for pname in pair_list:
            available: list[str] = [
                asset_name
                for asset_name in INTERFEROGRAM_ASSETS
                if self.exists(pname, asset_name)
            ]
            assets_by_pair[pname] = available

        common_grid = False
        if self._index is not None:
            common_grid = self._index.get("common_grid", False)

        return {
            "root": str(self._root),
            "pair_count": len(pair_list),
            "pairs": pair_list,
            "assets_by_pair": assets_by_pair,
            "common_grid": common_grid,
        }

    def to_parquet(self, path: str | Path | None = None) -> Path:
        """Write the collection index to ``interferograms_index.parquet``.

        Also refreshes the sibling ``interferograms_index.json`` if missing,
        so both representations stay consistent (double-write).

        Parameters
        ----------
        path : str or Path, optional
            Output parquet path. Defaults to
            ``<root>/interferograms_index.parquet``.

        Returns
        -------
        Path
            The resolved parquet path written.

        Raises
        ------
        ImportError
            If ``pyarrow`` is not installed.
        ValueError
            If the collection has no index to serialise.

        """
        try:
            import pyarrow as pa
        except ImportError as e:
            msg = (
                "pyarrow is required for to_parquet(). Install it with: "
                "pip install pyarrow (or the 'cloud' extra)."
            )
            raise ImportError(msg) from e

        index = self._require_index()
        records = _index_to_records(index)

        if path is not None:
            out_path = Path(path)
        else:
            out_path = self._root / "interferograms_index.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records)
        import pyarrow.parquet as pq

        pq.write_table(table, out_path)
        logger.info("Wrote interferograms index parquet: %s", out_path)

        # Double-write: refresh the JSON sibling if missing.
        json_sibling = self._root / "interferograms_index.json"
        if not json_sibling.exists():
            save_json(index, json_sibling)

        return out_path

    @classmethod
    def from_parquet(cls, parquet_path: str | Path) -> FrameInterferogramCollection:
        """Construct a collection rooted at the parquet's parent directory.

        Parameters
        ----------
        parquet_path : str or Path
            Path to an ``interferograms_index.parquet``.

        Returns
        -------
        FrameInterferogramCollection
            Collection whose root is the parquet's parent directory. The
            index is read from parquet (not JSON).

        Raises
        ------
        ImportError
            If ``pyarrow`` is not installed.

        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            msg = f"Parquet index not found: {parquet_path}"
            raise FileNotFoundError(msg)
        # Trigger pyarrow import early for a clear error.
        _load_index_parquet(parquet_path)
        return cls(parquet_path.parent)

    @classmethod
    def from_dataset(
        cls,
        out_dir: str | Path,
        dataset: InterferogramDataset,
        *,
        geometry: FrameGeometry | None = None,
        pairs: Pairs | None = None,
        assets: Sequence[InterferogramAssetName] = ("unw_phase", "coherence"),
        include_optional_hyp3_assets: bool = False,
        reference: str | Path | None = None,
        overwrite: bool = False,
        max_pairs: int | None = None,
    ) -> Self:
        """Standardize an existing InterferogramDataset into a frame layout.

        Parameters
        ----------
        out_dir : str or Path
            Output directory. An ``ifg`` subdirectory is created unless
            *out_dir* already points to a directory named ``ifg``.
        dataset : InterferogramDataset
            Source dataset providing unwrapped phase and coherence files.
        geometry : FrameGeometry, optional
            Geometry providing the reference grid.
        pairs : Pairs, optional
            Subset of pairs to standardize. If *None*, all valid pairs.
        assets : sequence of InterferogramAssetName
            Required assets to write.
        include_optional_hyp3_assets : bool
            If *True*, discover and copy optional HyP3 assets.
        reference : str or Path, optional
            Reference raster for the output grid.
        overwrite : bool
            If *True*, overwrite existing outputs.
        max_pairs : int, optional
            Limit the number of pairs to process.

        Returns
        -------
        FrameInterferogramCollection

        """
        import rasterio.enums

        out_dir = Path(out_dir)
        ifgs_dir = (
            out_dir / "interferograms" if out_dir.name != "interferograms" else out_dir
        )
        ifgs_dir.mkdir(parents=True, exist_ok=True)

        ref_grid: GeoGrid | None = None
        if reference is not None:
            ref_grid = read_geogrid(reference)
        elif geometry is not None:
            geom_meta = geometry.metadata
            if geom_meta is not None:
                ref_grid = GeoGrid.from_bounds(
                    geom_meta["bounds"],
                    crs=geom_meta["crs"],
                    shape=(geom_meta["height"], geom_meta["width"]),
                    tight=True,
                )

        ds_pairs = dataset.pairs
        if pairs is not None:
            mask = ds_pairs.where(pairs, return_type="mask")
            process_pairs = ds_pairs[mask]
        else:
            process_pairs = ds_pairs

        unw_valid = dataset.valid
        coh_valid = dataset.coh_dataset.valid
        both_valid = unw_valid & coh_valid

        if pairs is not None:
            pair_mask = ds_pairs.where(process_pairs, return_type="mask")
            both_valid = both_valid & pair_mask

        unw_paths = dataset.files.paths[both_valid]
        coh_paths = dataset.coh_dataset.files.paths[both_valid]
        active_pairs = ds_pairs[both_valid]

        # Apply max_pairs limit
        if max_pairs is not None and max_pairs < len(active_pairs):
            active_pairs = active_pairs[:max_pairs]
            unw_paths = unw_paths[:max_pairs]
            coh_paths = coh_paths[:max_pairs]

        common_grid = ref_grid is not None

        pair_names_list: list[str] = []
        assets_by_pair: dict[str, list[str]] = {}

        for idx in range(len(active_pairs)):
            pair_obj = active_pairs[idx]
            pname = str(pair_obj)
            pair_names_list.append(pname)

            pair_out = ifgs_dir / pname
            pair_out.mkdir(parents=True, exist_ok=True)

            unw_src = Path(unw_paths.iloc[idx])
            coh_src = Path(coh_paths.iloc[idx])

            pair_grid = read_geogrid(unw_src) if ref_grid is None else ref_grid

            written_assets: list[str] = []
            asset_records: dict[str, dict[str, Any]] = {}

            asset_sources: dict[str, Path] = {}
            for asset_name in assets:
                if asset_name == "unw_phase":
                    asset_sources[asset_name] = unw_src
                elif asset_name == "coherence":
                    asset_sources[asset_name] = coh_src
                elif asset_name == "wrapped_phase":
                    cand = unw_src.parent / unw_src.name.replace(
                        "_unw_phase", "_wrapped_phase"
                    )
                    if cand.exists():
                        asset_sources[asset_name] = cand
                elif asset_name == "los_disp":
                    cand = unw_src.parent / unw_src.name.replace(
                        "_unw_phase", "_los_disp"
                    )
                    if cand.exists():
                        asset_sources[asset_name] = cand
                elif asset_name == "vert_disp":
                    cand = unw_src.parent / unw_src.name.replace(
                        "_unw_phase", "_vert_disp"
                    )
                    if cand.exists():
                        asset_sources[asset_name] = cand
                elif asset_name == "amplitude":
                    cand = unw_src.parent / unw_src.name.replace("_unw_phase", "_amp")
                    if cand.exists():
                        asset_sources[asset_name] = cand

            if include_optional_hyp3_assets:
                suffix_map = {
                    "wrapped_phase": "_wrapped_phase",
                    "los_disp": "_los_disp",
                    "vert_disp": "_vert_disp",
                    "amplitude": "_amp",
                }
                for opt_name, suffix in suffix_map.items():
                    if opt_name not in asset_sources:
                        cand = unw_src.parent / unw_src.name.replace(
                            "_unw_phase", suffix
                        )
                        if cand.exists():
                            asset_sources[opt_name] = cand

            for asset_name, src in asset_sources.items():
                out_path = pair_out / _PAIR_ASSET_FILENAMES[asset_name]

                if out_path.exists() and not overwrite:
                    logger.info(
                        "Asset '%s' for pair '%s' exists, skipping.",
                        asset_name,
                        pname,
                    )
                    written_assets.append(asset_name)
                    import rasterio as _rio

                    with _rio.open(out_path) as ds:
                        asset_records[asset_name] = {
                            "href": _PAIR_ASSET_FILENAMES[asset_name],
                            "dtype": str(ds.dtypes[0]),
                            "nodata": ds.nodata,
                            "source_path": str(src),
                        }
                    continue

                is_cat = asset_name in CATEGORICAL_ASSETS
                if is_cat:
                    resampling = rasterio.enums.Resampling.nearest
                    out_dtype = "uint8"
                    out_nodata = 255
                else:
                    resampling = rasterio.enums.Resampling.bilinear
                    out_dtype = "float32"
                    out_nodata = -9999.0

                arr = reproject_to_geogrid(
                    src,
                    pair_grid,
                    resampling=resampling,
                    dst_dtype=out_dtype,
                    dst_nodata=out_nodata,
                )

                write_cog(
                    arr,
                    out_path,
                    pair_grid,
                    nodata=out_nodata,
                    dtype=out_dtype,
                    overwrite=overwrite,
                )

                written_assets.append(asset_name)
                asset_records[asset_name] = {
                    "href": _PAIR_ASSET_FILENAMES[asset_name],
                    "dtype": out_dtype,
                    "nodata": out_nodata,
                    "source_path": str(src),
                }

            assets_by_pair[pname] = written_assets

            bounds = pair_grid.bounds
            res = pair_grid.resolution
            grid_meta: dict[str, Any] = {
                "crs": str(pair_grid.crs),
                "width": pair_grid.width,
                "height": pair_grid.height,
                "transform": list(pair_grid.transform)[:6],
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "resolution": [abs(res.x), abs(res.y)],
            }

            parts = pname.split("_")
            ref_date = parts[0] if len(parts) >= 2 else ""
            sec_date = parts[1] if len(parts) >= 2 else ""

            item_meta = build_item_metadata(
                pair_name=pname,
                reference_date=ref_date,
                secondary_date=sec_date,
                grid=grid_meta,
                assets=asset_records,
                geometry_href=resolve_geometry_href(ifgs_dir),
                temporal_baseline_days=_compute_temporal_baseline(ref_date, sec_date),
            )
            save_json(item_meta, pair_out / "item.json")

        index_meta = build_interferograms_index(
            pair_count=len(pair_names_list),
            pairs=pair_names_list,
            assets_by_pair=assets_by_pair,
            common_grid=common_grid,
            geometry_href=resolve_geometry_href(ifgs_dir),
        )
        save_json(index_meta, ifgs_dir / "interferograms_index.json")

        # Double-write: emit the parquet sibling too (best-effort, optional dep).
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.info(
                "pyarrow not installed; skipping interferograms_index.parquet "
                "(install the 'cloud' extra to enable parquet double-write)."
            )
        else:
            try:
                records = _index_to_records(index_meta)
                import pyarrow as pa

                table = pa.Table.from_pylist(records)
                pq.write_table(table, ifgs_dir / "interferograms_index.parquet")
                logger.info("Wrote interferograms_index.parquet alongside JSON.")
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Failed to write parquet index: %s", e)

        logger.info(
            "FrameInterferogramCollection created at %s with %d pairs",
            ifgs_dir,
            len(pair_names_list),
        )
        return cls(ifgs_dir)

    @classmethod
    def from_hyp3(
        cls,
        out_dir: str | Path,
        root_dir: str | Path,
        *,
        geometry: FrameGeometry | None = None,
        pairs: Pairs | None = None,
        assets: Sequence[InterferogramAssetName] = ("unw_phase", "coherence"),
        include_optional_assets: bool = True,
        reference: str | Path | None = None,
        overwrite: bool = False,
        max_pairs: int | None = None,
        **dataset_kwargs: Any,
    ) -> Self:
        """Standardize HyP3 products into a frame layout.

        Parameters
        ----------
        out_dir : str or Path
            Output directory.
        root_dir : str or Path
            Root directory of the HyP3 products.
        geometry : FrameGeometry, optional
            Geometry providing the reference grid.
        pairs : Pairs, optional
            Subset of pairs.
        assets : sequence of InterferogramAssetName
            Required assets.
        include_optional_assets : bool
            If *True*, discover optional HyP3 assets.
        reference : str or Path, optional
            Reference raster for the output grid.
        overwrite : bool
            Overwrite existing outputs.
        max_pairs : int, optional
            Limit the number of pairs to process. Useful for demos or testing.
        **dataset_kwargs
            Extra keyword arguments passed to ``HyP3S1``.

        Returns
        -------
        FrameInterferogramCollection

        """
        from faninsar.datasets.hyp3 import HyP3S1

        dataset = HyP3S1(root_dir=root_dir, **dataset_kwargs)
        return cls.from_dataset(
            out_dir=out_dir,
            dataset=dataset,
            geometry=geometry,
            pairs=pairs,
            assets=assets,
            include_optional_hyp3_assets=include_optional_assets,
            reference=reference,
            overwrite=overwrite,
            max_pairs=max_pairs,
        )

    def open(
        self,
        pair: str,
        name: InterferogramAssetName,
        *,
        masked: bool = True,
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.DataArray:
        """Open a single pair asset as a lazy DataArray.

        Parameters
        ----------
        pair : str
            Pair name (e.g. ``"20191115_20200314"``).
        name : InterferogramAssetName
            Asset name.
        masked : bool
            Apply the nodata mask.
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays.

        Returns
        -------
        xarray.DataArray

        """
        import rioxarray

        asset_path = self.path(pair, name)
        if not asset_path.exists():
            raise MissingInterferogramAssetError(pair, name)

        da = rioxarray.open_rasterio(
            asset_path,
            masked=masked,
            chunks=chunks,  # type: ignore[arg-type]
        )
        if not isinstance(da, xr.DataArray):
            msg = f"Expected DataArray, got {type(da).__name__}"
            raise TypeError(msg)
        if da.ndim == 3:
            da = da.squeeze("band", drop=True)
        return da

    def open_stack(
        self,
        name: InterferogramAssetName,
        *,
        pairs: Pairs | None = None,
        chunks: dict[str, int] | int | Literal["auto"] | None = None,
    ) -> xr.DataArray:
        """Stack a named asset across all pairs.

        Parameters
        ----------
        name : InterferogramAssetName
            Asset name to stack.
        pairs : Pairs, optional
            Subset of pairs. If *None*, all pairs.
        chunks : dict, int, ``"auto"``, or None
            Chunk sizes for dask arrays.

        Returns
        -------
        xarray.DataArray
            DataArray with a ``pair`` dimension.

        Raises
        ------
        GridMismatchError
            If the selected rasters do not share the same grid.

        """
        if pairs is None:
            pairs = self.pairs()

        pair_names = pairs.to_names().tolist()
        arrays: list[xr.DataArray] = []
        ref_grid: GeoGrid | None = None

        for pname in pair_names:
            if not self.exists(pname, name):
                logger.warning(
                    "Asset '%s' missing for pair '%s', skipping.", name, pname
                )
                continue

            da = self.open(pname, name, masked=True, chunks=chunks)

            asset_grid = read_geogrid(self.path(pname, name))
            if ref_grid is None:
                ref_grid = asset_grid
            elif not asset_grid.is_aligned(ref_grid):
                msg = (
                    f"Pair '{pname}' asset '{name}' grid does not match "
                    f"the first pair's grid. Use a common reference when "
                    f"standardizing, or call from_dataset() with a geometry."
                )
                raise GridMismatchError(msg)

            da = da.expand_dims(pair=[pname])
            arrays.append(da)

        if not arrays:
            msg = f"No valid assets found for '{name}'."
            raise ValueError(msg)

        return xr.concat(arrays, dim="pair")

    def as_interferogram_dataset(self) -> InterferogramDataset:
        """Return a FanInSAR InterferogramDataset backed by the frame layout.

        This bridges the frame product to the existing FanInSAR query system.

        """
        from faninsar.datasets.ifg import InterferogramDataset

        pairs_obj = self.pairs()
        pair_names = pairs_obj.to_names().tolist() if len(pairs_obj) > 0 else []

        paths_unw: list[Path] = []
        paths_coh: list[Path] = []
        for pname in pair_names:
            unw_p = self.path(pname, "unw_phase")
            coh_p = self.path(pname, "coherence")
            if unw_p.exists():
                paths_unw.append(unw_p)
            if coh_p.exists():
                paths_coh.append(coh_p)

        return InterferogramDataset(
            root_dir=self._root,
            paths_unw=paths_unw if paths_unw else None,
            paths_coh=paths_coh if paths_coh else None,
            verbose=False,
        )

    def to_stac(
        self,
        geometry: FrameGeometry | None = None,
        *,
        catalog_id: str = "insar-interferograms",
        description: str = "",
        temporal_extent: tuple[datetime | None, datetime | None] | None = None,
        output_dir: str | Path | None = None,
        catalog_type: pystac.CatalogType | None = None,
    ) -> pystac.Catalog:
        """Generate a STAC Catalog from this interferogram collection.

        Parameters
        ----------
        geometry : FrameGeometry, optional
            If provided, include geometry assets as a sibling collection.
        catalog_id : str
            STAC Catalog id.
        description : str
            Catalog description.
        temporal_extent : tuple of (start, end), optional
            Temporal extent. If *None*, auto-detected from pair dates.
        output_dir : str or Path, optional
            If provided, save the catalog to this directory.
        catalog_type : pystac.CatalogType, optional
            STAC catalog type.

        Returns
        -------
        pystac.Catalog
            The generated STAC Catalog.

        """
        from .stac import build_stac_catalog

        return build_stac_catalog(
            geometry=geometry,
            ifgs=self,
            catalog_id=catalog_id,
            description=description,
            temporal_extent=temporal_extent,
            output_dir=output_dir,
            catalog_type=catalog_type,
        )


def _compute_temporal_baseline(ref_date: str, sec_date: str) -> int | None:
    """Compute the temporal baseline in days between two YYYYMMDD date strings."""
    from datetime import datetime

    try:
        d1 = datetime.strptime(ref_date, "%Y%m%d").replace(tzinfo=UTC)
        d2 = datetime.strptime(sec_date, "%Y%m%d").replace(tzinfo=UTC)
        return abs((d2 - d1).days)
    except (ValueError, TypeError):
        return None
