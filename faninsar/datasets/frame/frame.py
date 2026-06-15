"""Unified frame-level InSAR product combining geometry and interferograms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
