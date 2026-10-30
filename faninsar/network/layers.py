"""Lazy, typed layer views of committed Stack interferograms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.io.storage.product import Product, open_product
from faninsar.logging import setup_logger
from faninsar.network.products import ASSET_KIND_GEOCODE_KERNELS, AssetKind
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geocoding.geo_resample import GeocodeKind
    from faninsar.processing.geocoding.geocoder import GeocodedRaster
    from faninsar.processing.geocoding.product import GeocodeSession
    from faninsar.processing.geometry.grids import GridSpec

__all__ = ["InterferogramLayers", "ProductLayer"]

logger = setup_logger(__name__)

#: Canonical IFG layer names derived at read time from the owning complex
#: interferogram (``angle``/``abs``). Persistence ownership lives in
#: :mod:`faninsar.io.storage.ifg_store`, whose ``InterferogramArtifactStore.
#: read_layer`` computes them; no virtual payload is ever written.
_VIRTUAL_IFG_LAYERS = ("wrapped_phase", "amplitude")

#: Scene layer names served through the SCENE_KIND read branch — the persisted
#: per-unit ``slc`` zarr payloads plus the three read-time derivations and the
#: computed support mask. All five are virtual handles on the scene product.
SCENE_LAYER_NAMES = ("slc", "amplitude", "intensity", "phase", "valid_mask")


@dataclass(frozen=True, slots=True)
class ProductLayer:
    """One lazy raster bound to the exact commit that owns its values.

    Attributes
    ----------
    name : str
        Serialized asset name, independent of the short accessor spelling.
    asset_kind : AssetKind or None
        Physical type used to select the resampling kernel. Every layer
        produced by a Stack declares one; ``None`` only appears on a
        hand-built handle.
    full_radar_shape, multilook, origin : tuple[int, int]
        Full radar frame and the layer's sampling within it.
    crs : str or None
        Map CRS for geographic products, otherwise ``None``.

    Notes
    -----
    Obtain layers from ``stack.interferograms[pair]``; ``pair`` may be a
    name, a :class:`Pair`, or a :class:`Pairs` (the last yields a dict of
    views). Accessing a layer
    property reads no raster. :attr:`values` loads only the selected layer
    and returns a read-only array. Derived arrays belong in an in-memory
    ``RadarDomainProduct`` and do not inherit the committed source identity.
    Virtual layers (``wrapped_phase`` and ``amplitude`` of an IFG product)
    are derived from their owning complex IFG commit and expose
    :attr:`path` as ``None``.

    """

    _product: Product = field(repr=False)
    name: str
    asset_kind: AssetKind | None
    full_radar_shape: tuple[int, int]
    multilook: tuple[int, int] = (1, 1)
    origin: tuple[int, int] = (0, 0)
    crs: str | None = None
    extent: tuple[int, int] | None = None
    _scene_selection: frozenset[str] | None = field(
        default=None, repr=False, compare=False
    )
    _upstream: tuple[Product, ...] = field(default=(), repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)

    @property
    def source(self) -> Product:
        """Return the committed product that owns this layer."""
        return self._product

    @property
    def path(self) -> Path | None:
        """Return the asset path, or ``None`` for a virtual derived layer."""
        if self.name in self._product.assets:
            return self._product.asset_path(self.name)
        if self.name in _VIRTUAL_IFG_LAYERS:
            return None
        if self._product.kind == "scene" and self.name in SCENE_LAYER_NAMES:
            return None
        message = (
            f"layer {self.name!r} is not stored in {self._product.path}; it is "
            "neither a persisted asset nor a derivable virtual layer"
        )
        raise FileNotFoundError(message)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the declared layer shape without reading a raster."""
        if self.extent is not None:
            return self.extent
        shape = self._product.properties["faninsar:shape"]
        return int(shape[0]), int(shape[1])

    @property
    def geocode_kind(self) -> GeocodeKind:
        """Return the physical resampling kernel, including hard labels."""
        if self.asset_kind is None:
            reject_invalid_state(f"layer {self.name!r} has no physical type")
        return ASSET_KIND_GEOCODE_KERNELS[self.asset_kind]

    def geocode(
        self,
        grid: GridSpec | None = None,
        *,
        path: str | Path | None = None,
        overwrite: bool = False,
    ) -> GeocodedRaster:
        """Geocode this layer through the Stack session that produced it.

        The handle already carries the product and its physical type, so no
        product or kernel argument is accepted.

        Parameters
        ----------
        grid : GridSpec, optional
            Explicit destination grid. The owning Stack's authoritative grid
            is used when omitted.
        path : str or pathlib.Path, optional
            Complete output product directory. On the Stack grid, omitting it
            saves to ``<source>/geocoded/<asset>/``.
        overwrite : bool, optional
            Replace an existing geocoded child instead of reusing it.

        Returns
        -------
        GeocodedRaster
            Values, validity, and destination grid.

        Raises
        ------
        InvalidProcessingStateError
            If this handle was not obtained from a Stack session.

        """
        session = self._session
        if session is None:
            reject_invalid_state(
                f"layer {self.name!r} is not bound to a Stack session; "
                "use stack.geocode(layer, ...) for an unbound handle"
            )
        return session.geocode(self, grid=grid, path=path, overwrite=overwrite)

    def validate_source(self) -> None:
        """Reject a handle whose owning or upstream product was replaced."""
        for recorded in (self._product, *self._upstream):
            current = open_product(recorded.path, kind=recorded.kind)
            if current.write_id != recorded.write_id:
                reject_invalid_state(
                    f"layer {self.name!r} belongs to a replaced source at "
                    f"{recorded.path}; obtain a fresh layer from the Stack"
                )

    def validate_reference(self, write_id: str) -> None:
        """Reject geocoding against a different Stack Reference commit."""
        references = [
            product.write_id
            for product in self._upstream
            if product.properties.get("faninsar:role") == "reference"
        ]
        if references != [write_id]:
            reject_invalid_state("layer and Stack use different Reference products")

    def _read_virtual(self, name: str) -> np.ndarray:
        """Derive one virtual IFG layer through the owning artifact store."""
        from faninsar.io.storage.ifg_store import (
            IFG_KIND,
            InterferogramArtifactStore,
        )

        if self._product.kind != IFG_KIND:
            reject_invalid_state(
                f"layer {self.name!r} is virtual and requires an IFG product source"
            )
        return InterferogramArtifactStore.open(self._product.path).read_layer(name)

    def _read_scene_layer(self, name: str) -> np.ndarray:
        """Read one scene layer through its store's mosaic reader.

        ``slc`` mosaics the selected unit payloads onto their union bounding
        box (or the common grid for a ``.full()`` layer); ``amplitude``,
        ``intensity``, and ``phase`` derive from that mosaic; ``valid_mask``
        computes ``isfinite(real) & isfinite(imag) & (|slc| > 0)``.
        """
        from faninsar.io.storage.scene_store import (
            SCENE_KIND,
            CoregisteredSceneStore,
            valid_samples,
        )

        if self._product.kind != SCENE_KIND:
            reject_invalid_state(
                f"layer {self.name!r} is virtual and requires a scene product source"
            )
        store = CoregisteredSceneStore.open(self._product.path)
        if self._scene_selection is not None:
            store = store.select_units(self._scene_selection)
        mosaic = store.mosaic_slc(full=self.extent is None)
        if name == "slc":
            return mosaic
        if name == "amplitude":
            return np.abs(mosaic).astype(np.float32)
        if name == "intensity":
            return (np.abs(mosaic) ** 2).astype(np.float32)
        if name == "phase":
            return np.angle(mosaic).astype(np.float32)
        if name == "valid_mask":
            return valid_samples(mosaic)
        message = f"unknown scene layer: {name!r}"
        raise FileNotFoundError(message)

    def _read(self, name: str) -> np.ndarray:
        if name in self._product.assets:
            values = self._product.read(name)
        elif self._product.kind == "scene" and name in SCENE_LAYER_NAMES:
            values = self._read_scene_layer(name)
        elif name in _VIRTUAL_IFG_LAYERS:
            values = self._read_virtual(name)
        else:
            message = (
                f"layer {name!r} is not stored in {self._product.path}; it is "
                "neither a persisted asset nor a derivable virtual layer"
            )
            raise FileNotFoundError(message)
        if values.shape != self.shape:
            reject_invalid_state(f"layer {name!r} shape differs from its product")
        return values

    def read(self) -> np.ndarray:
        """Read a detached array after checking the source commit and dtype.

        Returns
        -------
        numpy.ndarray
            Newly loaded values that may be modified without changing this
            committed layer or its source identity. Virtual layers are
            derived from their owning complex IFG commit.

        """
        self.validate_source()
        values = self._read(self.name)
        kind = self.geocode_kind
        valid_dtype = {
            "complex": values.dtype.kind == "c",
            "phase": values.dtype.kind == "f",
            "continuous": values.dtype.kind == "f",
            "label": values.dtype.kind in "biu",
        }[kind]
        if not valid_dtype:
            reject_invalid_state(f"layer {self.name!r} has an incompatible dtype")
        return values

    @property
    def values(self) -> np.ndarray:
        """Read the selected layer as a read-only, detached array."""
        values = self.read()
        values.setflags(write=False)
        return values

    def to_xarray(self) -> Any:
        """Return a lazy ``xarray.DataArray`` view of this layer.

        ``dask.array.from_zarr`` opens each selected unit's persisted zarr
        lazily and a ``map_blocks`` step mosaics them onto the layer's frame
        (bounding-box when :attr:`extent` is set, the full common grid
        otherwise); the ``slc`` mosaic then derives ``amplitude``,
        ``intensity``, ``phase``, or ``valid_mask``. ``validate_source``
        runs once at build time; the returned ``DataArray`` does not
        re-validate on ``.compute()`` — callers needing a staleness re-check
        use :attr:`values` or :meth:`validate_source`.

        Returns
        -------
        xarray.DataArray
            Lazy array with dims ``("y", "x")`` and attrs ``shape``,
            ``multilook``, ``full_radar_shape``, ``origin``, ``asset_kind``,
            ``crs``, ``transform``, and ``source_write_id``.

        """
        import dask.array as da
        import xarray as xr

        self.validate_source()
        product = self._product
        if self._product.kind == "scene":
            from faninsar.io.storage._zarr_stac import read_zarr_array
            from faninsar.io.storage.scene_store import (
                CoregisteredSceneStore,
                valid_samples,
            )

            store = CoregisteredSceneStore.open(product.path)
            if self._scene_selection is not None:
                store = store.select_units(self._scene_selection)
            units = store.selected_units()
            # Lazy mosaic: each unit's zarr read is a dask.delayed block and
            # one delayed composite places them into the layer frame with the
            # first-valid-sample overlap rule (matching ``mosaic_slc``). The
            # whole frame is a single dask chunk; payloads load only when the
            # DataArray is computed.
            full = self.extent is None
            frame = store.grid_shape if full else store.selected_extent()
            origin = (0, 0) if full else store.selected_origin()
            layer_name = self.name

            def _compose(*blocks: np.ndarray) -> np.ndarray:
                out = np.full(frame, np.nan + 1j * np.nan, dtype=np.complex64)
                for block, unit in zip(blocks, units, strict=True):
                    r0 = unit.row_origin - origin[0]
                    c0 = unit.col_origin - origin[1]
                    win = out[r0 : r0 + unit.shape[0], c0 : c0 + unit.shape[1]]
                    invalid = ~valid_samples(win)
                    win[invalid] = block[invalid]
                if layer_name == "slc":
                    return out
                if layer_name == "amplitude":
                    return np.abs(out).astype(np.float32)
                if layer_name == "intensity":
                    return (np.abs(out) ** 2).astype(np.float32)
                if layer_name == "phase":
                    return np.angle(out).astype(np.float32)
                if layer_name == "valid_mask":
                    return valid_samples(out)
                return out

            from dask import delayed

            blocks = [delayed(read_zarr_array)(unit.payload_path) for unit in units]
            dtype = (
                np.complex64
                if layer_name == "slc"
                else np.bool_
                if layer_name == "valid_mask"
                else np.float32
            )
            data = da.from_delayed(delayed(_compose)(*blocks), shape=frame, dtype=dtype)
        else:
            data = da.from_zarr(str(self.path), chunks=self.shape)
            if self.name in _VIRTUAL_IFG_LAYERS:
                complex_block = data
                if self.name == "amplitude":
                    data = da.map_blocks(
                        lambda b: np.abs(b).astype(np.float32),
                        complex_block,
                        dtype=np.float32,
                    )
                elif self.name == "wrapped_phase":
                    data = da.map_blocks(
                        lambda b: np.angle(b).astype(np.float32),
                        complex_block,
                        dtype=np.float32,
                    )
        attrs = {
            "shape": self.shape,
            "multilook": self.multilook,
            "full_radar_shape": self.full_radar_shape,
            "origin": self.origin,
            "asset_kind": None if self.asset_kind is None else str(self.asset_kind),
            "crs": self.crs,
            "transform": product.properties.get("proj:transform"),
            "source_write_id": product.write_id,
        }
        return xr.DataArray(data, dims=("y", "x"), name=self.name, attrs=attrs)

    @property
    def valid_mask(self) -> np.ndarray | None:
        """Read authoritative support, including excluded unwrap components.

        Scene layers compute ``isfinite(real) & isfinite(imag) & (|slc| > 0)``
        from the mosaic on demand; persisted ``valid_mask`` assets on IFG and
        unwrap products are read through the usual branch.
        """
        self.validate_source()
        mask = None
        if self._product.kind == "scene":
            mask = self._read("valid_mask")
            if mask.dtype != np.bool_:
                reject_invalid_state("layer valid_mask must have boolean dtype")
            return mask
        if "valid_mask" in self._product.assets:
            mask = self._read("valid_mask")
            if mask.dtype != np.bool_:
                reject_invalid_state("layer valid_mask must have boolean dtype")
        if self._product.kind == "unwrap":
            components = self._read("connected_components")
            if components.dtype.kind not in "iu":
                reject_invalid_state("connected components must have integer dtype")
            supported = components > 0
            mask = supported if mask is None else mask & supported
        return mask

    def plot(
        self,
        ax: Any = None,
        *,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        masked: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        hist_colorbar: bool = True,
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        """Plot this layer with colormap and limits chosen from its asset kind.

        Phase layers use a cyclic colormap with +/- pi limits and a pi-ticked
        colorbar, ``coherence`` uses 0-1 grayscale, ``amplitude`` plots
        magnitude, and ``label`` uses a discrete map. A complex interferogram
        renders its in-phase and quadrature components as two side-by-side
        panels; its phase and magnitude are reached through the dedicated
        ``wrapped_phase`` and ``amplitude`` layers. Any default can be overridden
        per call.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure and axes are created when *None*.
            Ignored for a complex layer, which always builds a two-panel
            figure.
        cmap : str, optional
            Colormap name overriding the kind default.
        vmin, vmax : float, optional
            Data limits overriding the kind default. For a complex layer the
            same limits apply to both panels.
        colorbar : bool, optional
            Add a colorbar. Default True.
        masked : bool, optional
            Mask pixels outside :attr:`valid_mask` to NaN. Default True.
        figsize : tuple[float, float], optional
            Figure size when creating a new figure.
        title : str, optional
            Axes title. Defaults to the layer's public name; a complex layer
            suffixes ``I``/``Q`` on each panel.
        hist_colorbar : bool, optional
            Use FanInSAR :class:`~faninsar.plotting.HistColorbar` (histogram
            embedded beside the gradient) instead of a plain colorbar.
            Default True.
        **imshow_kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig, ax, im
            The figure, axes, and ``AxesImage`` so callers can keep
            customizing. For a complex layer *ax* and *im* are each a
            ``(I, Q)`` tuple; *fig* is the shared figure.

        """
        import matplotlib.pyplot as plt

        if self.asset_kind in (AssetKind.COMPLEX_INTERFEROGRAM, AssetKind.SLC):
            return self._plot_iq(
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                masked=masked,
                figsize=figsize,
                title=title,
                hist_colorbar=hist_colorbar,
                **imshow_kwargs,
            )

        values = np.asarray(self.values, dtype=np.result_type(self.values, np.float32))
        if masked:
            mask = self.valid_mask
            if mask is not None:
                values = np.where(mask, values, np.nan)

        kind = self.asset_kind
        if cmap is None:
            cmap = {
                AssetKind.WRAPPED_PHASE: "jet",
                AssetKind.UNWRAPPED_PHASE: "viridis",
                AssetKind.COHERENCE: "gray",
                AssetKind.AMPLITUDE: "gray",
                AssetKind.LABEL: "tab20",
            }.get(kind, "viridis")
        if vmin is None or vmax is None:
            default_limits: dict[AssetKind | None, tuple[float, float]] = {
                AssetKind.WRAPPED_PHASE: (-np.pi, np.pi),
                AssetKind.COHERENCE: (0.0, 1.0),
            }
            lo, hi = default_limits.get(kind, (np.nanmin(values), np.nanmax(values)))
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            if title is None:
                title = self.name
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_title(title if title is not None else "")
        if colorbar:
            self._add_colorbar(
                ax.figure,
                ax,
                im,
                values,
                phase=kind == AssetKind.WRAPPED_PHASE,
                hist=hist_colorbar,
            )
        return ax.figure, ax, im

    def _plot_iq(
        self,
        *,
        cmap: str | None,
        vmin: float | None,
        vmax: float | None,
        colorbar: bool,
        masked: bool,
        figsize: tuple[float, float] | None,
        title: str | None,
        hist_colorbar: bool,
        **imshow_kwargs: Any,
    ) -> tuple[Any, tuple[Any, Any], tuple[Any, Any]]:
        """Render a complex layer as side-by-side in-phase/quadrature panels."""
        import matplotlib.pyplot as plt

        values = np.asarray(self.values, dtype=np.complex64)
        mask = self.valid_mask if masked else None

        if cmap is None:
            cmap = "gray"
        if vmin is None or vmax is None:
            finite = np.concatenate(
                [
                    np.asarray(values.real)[np.isfinite(values.real)],
                    np.asarray(values.imag)[np.isfinite(values.imag)],
                ]
            )
            limit = float(np.abs(finite).max()) if finite.size else 1.0
            vmin = -limit if vmin is None else vmin
            vmax = limit if vmax is None else vmax

        base = title if title is not None else "complex"
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        ims = []
        for axis, component, suffix in (
            (axes[0, 0], values.real, "I"),
            (axes[0, 1], values.imag, "Q"),
        ):
            data = np.asarray(component, dtype=np.float32)
            if mask is not None:
                data = np.where(mask, data, np.nan)
            im = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
            axis.set_title(f"{base} ({suffix})")
            ims.append(im)
            if colorbar:
                self._add_colorbar(fig, axis, im, data, hist=hist_colorbar)
        return fig, (axes[0, 0], axes[0, 1]), (ims[0], ims[1])

    @staticmethod
    def _add_colorbar(
        fig: Any,
        ax: Any,
        im: Any,
        data: np.ndarray,
        *,
        phase: bool = False,
        hist: bool = True,
    ) -> None:
        """Attach a colorbar, FanInSAR HistColorbar when *hist* is set."""
        if hist:
            import faninsar.plotting  # noqa: F401  (installs Figure.hist_colorbar)

            hcb = fig.hist_colorbar(np.asarray(data).ravel(), mappable=im, ax=ax)
            if phase:
                from faninsar.plotting.formatters import setup_phase_axis

                setup_phase_axis(hcb.ax_cbar.yaxis)
            return
        cbar = fig.colorbar(im, ax=ax)
        if phase:
            from faninsar.plotting.formatters import setup_phase_axis

            setup_phase_axis(cbar.ax.yaxis)


@dataclass(frozen=True, slots=True)
class InterferogramLayers:
    """Flat per-Pair IFG view of lazy rasters and its bound unwrap product.

    Parameters
    ----------
    _product : Product
        Committed interferogram product.
    full_radar_shape : tuple[int, int]
        Authoritative full frame from the Stack Reference.
    multilook : tuple[int, int]
        Azimuth and range look factors.
    crs : str or None, optional
        Map CRS when this is already a geographic product.

    """

    _product: Product = field(repr=False)
    full_radar_shape: tuple[int, int]
    multilook: tuple[int, int]
    crs: str | None = None
    _upstream: tuple[Product, ...] = field(default=(), repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)

    @property
    def root(self) -> Path:
        """Return the source product directory."""
        return self._product.path

    @property
    def layers(self) -> list[str]:
        """List accessible public layer names, virtual layers included.

        Persisted optionals (``coherence``) and the per-Pair unwrap results
        (``unwrapped_phase``, ``connected_components``) appear only when they
        exist and the unwrap product is bound to this IFG commit; stale
        unwrap products of a replaced IFG are not accessible and are not
        listed. ``wrapped_phase`` and ``amplitude`` are always computable.

        """
        names = ["amplitude", "complex", "wrapped_phase"]
        if "coherence" in self._product.assets:
            names.append("coherence")
        from faninsar.io.storage.ifg_store import InterferogramArtifactStore

        if InterferogramArtifactStore.open(self.root).has_unwrapped():
            names.extend(["connected_components", "unwrapped_phase"])
        return sorted(names)

    def _layer(
        self,
        name: str,
        asset_kind: AssetKind,
        *,
        virtual: bool = False,
    ) -> ProductLayer:
        if not virtual and name not in self._product.assets:
            message = (
                f"IFG product {self._product.path} has no {name!r} layer; "
                f"available layers: {sorted(self._product.assets)}"
            )
            raise FileNotFoundError(message)
        return ProductLayer(
            self._product,
            name,
            asset_kind,
            self.full_radar_shape,
            self.multilook,
            crs=self.crs,
            _upstream=self._upstream,
            _session=self._session,
        )

    def _unwrap_layer(self, name: str, asset_kind: AssetKind | None) -> ProductLayer:
        """Return one layer of the unwrap product bound to this IFG commit."""
        from faninsar.io.storage.ifg_store import UNWRAP_KIND, unwrap_root

        root = unwrap_root(self.root)
        if not (root / "item.json").is_file():
            message = (
                f"Pair has no unwrapped result at {root}; run an unwrap stage first"
            )
            raise FileNotFoundError(message)
        product = open_product(root, kind=UNWRAP_KIND)
        if product.inputs.get("ifg") != self._product.write_id:
            reject_invalid_state("unwrap product belongs to a replaced IFG")
        if name not in product.assets:
            message = (
                f"unwrap product {root} has no {name!r} layer; "
                f"available layers: {sorted(product.assets)}"
            )
            raise FileNotFoundError(message)
        return ProductLayer(
            product,
            name,
            asset_kind,
            self.full_radar_shape,
            self.multilook,
            crs=self.crs,
            _upstream=(self._product, *self._upstream),
            _session=self._session,
        )

    @property
    def wrapped_phase(self) -> ProductLayer:
        """Return real wrapped radians, sampled as cyclic angles.

        The layer is virtual: it derives from the owning complex IFG commit
        and stores no payload.

        """
        return self._layer("wrapped_phase", AssetKind.WRAPPED_PHASE, virtual=True)

    @property
    def complex(self) -> ProductLayer:
        """Return the complex interferogram layer."""
        return self._layer("complex", AssetKind.COMPLEX_INTERFEROGRAM)

    @property
    def coherence(self) -> ProductLayer:
        """Return the coherence layer when it was estimated."""
        return self._layer("coherence", AssetKind.COHERENCE)

    @property
    def amplitude(self) -> ProductLayer:
        """Return the amplitude layer derived from the complex IFG commit."""
        return self._layer("amplitude", AssetKind.AMPLITUDE, virtual=True)

    @property
    def unwrapped_phase(self) -> ProductLayer:
        """Return real unwrapped radians from the bound unwrap product."""
        return self._unwrap_layer("unwrapped_phase", AssetKind.UNWRAPPED_PHASE)

    @property
    def connected_components(self) -> ProductLayer:
        """Return integer component labels sampled with nearest neighbors."""
        return self._unwrap_layer("connected_components", AssetKind.LABEL)
