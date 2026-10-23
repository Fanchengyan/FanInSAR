"""Lazy, typed layer views of committed Stack interferograms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from faninsar.io.storage.product import Product, open_product
from faninsar.logging import setup_logger
from faninsar.network.products import ASSET_KIND_GEOCODE_KERNELS, AssetKind
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.processing.geocoding.geo_resample import GeocodeKind

__all__ = ["InterferogramLayers", "ProductLayer"]

logger = setup_logger(__name__)

#: Canonical IFG layer names derived at read time from the owning complex
#: interferogram (``angle``/``abs``). Persistence ownership lives in
#: :mod:`faninsar.io.storage.ifg_store`, whose ``InterferogramArtifactStore.
#: read_layer`` computes them; no virtual payload is ever written.
_VIRTUAL_IFG_LAYERS = ("wrapped_phase", "amplitude")


@dataclass(frozen=True, slots=True)
class ProductLayer:
    """One lazy raster bound to the exact commit that owns its values.

    Attributes
    ----------
    name : str
        Serialized asset name, independent of the short accessor spelling.
    asset_kind : AssetKind or None
        Physical type used to select the resampling kernel.
    full_radar_shape, multilook, origin : tuple[int, int]
        Full radar frame and the layer's sampling within it.
    crs : str or None
        Map CRS for geographic products, otherwise ``None``.

    Notes
    -----
    Obtain layers from ``stack.interferograms[pair]``. Accessing a layer
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
    _kind: GeocodeKind | None = field(default=None, repr=False)
    _upstream: tuple[Product, ...] = field(default=(), repr=False)

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
        message = (
            f"layer {self.name!r} is not stored in {self._product.path}; it is "
            "neither a persisted asset nor a derivable virtual layer"
        )
        raise FileNotFoundError(message)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the declared layer shape without reading a raster."""
        shape = self._product.properties["faninsar:shape"]
        return int(shape[0]), int(shape[1])

    @property
    def geocode_kind(self) -> GeocodeKind:
        """Return the physical resampling kernel, including hard labels."""
        if self._kind is not None:
            return self._kind
        if self.asset_kind is None:
            reject_invalid_state(f"layer {self.name!r} has no physical type")
        return ASSET_KIND_GEOCODE_KERNELS[self.asset_kind]

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

    def _read(self, name: str) -> np.ndarray:
        if name in self._product.assets:
            values = self._product.read(name)
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

    @property
    def valid_mask(self) -> np.ndarray | None:
        """Read authoritative support, including excluded unwrap components."""
        self.validate_source()
        mask = None
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

    @property
    def root(self) -> Path:
        """Return the source product directory."""
        return self._product.path

    @property
    def layers(self) -> list[str]:
        """List accessible public layer names, virtual layers included.

        Persisted optionals (``coherence``) and the per-Pair unwrap results
        (``unwrapped``, ``connected_components``) appear only when they
        exist and the unwrap product is bound to this IFG commit; stale
        unwrap products of a replaced IFG are not accessible and are not
        listed. ``wrapped`` and ``amplitude`` are always computable.

        """
        names = ["amplitude", "complex_ifg", "wrapped"]
        if "coherence" in self._product.assets:
            names.append("coherence")
        from faninsar.io.storage.ifg_store import InterferogramArtifactStore

        if InterferogramArtifactStore.open(self.root).has_unwrapped():
            names.extend(["connected_components", "unwrapped"])
        return sorted(names)

    def _layer(
        self,
        name: str,
        asset_kind: AssetKind,
        *,
        kind: GeocodeKind | None = None,
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
            _kind=kind,
            _upstream=self._upstream,
        )

    def _unwrap_layer(
        self, name: str, asset_kind: AssetKind | None, kind: GeocodeKind | None = None
    ) -> ProductLayer:
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
            _kind=kind,
            _upstream=(self._product, *self._upstream),
        )

    @property
    def wrapped(self) -> ProductLayer:
        """Return real wrapped radians, sampled as cyclic angles.

        The layer is virtual: it derives from the owning complex IFG commit
        and stores no payload.

        """
        return self._layer("wrapped_phase", AssetKind.WRAPPED_PHASE, virtual=True)

    @property
    def complex_ifg(self) -> ProductLayer:
        """Return the complex interferogram layer."""
        return self._layer("complex_ifg", AssetKind.COMPLEX_INTERFEROGRAM)

    @property
    def coherence(self) -> ProductLayer:
        """Return the coherence layer when it was estimated."""
        return self._layer("coherence", AssetKind.COHERENCE)

    @property
    def amplitude(self) -> ProductLayer:
        """Return the amplitude layer derived from the complex IFG commit."""
        return self._layer("amplitude", AssetKind.AMPLITUDE, virtual=True)

    @property
    def unwrapped(self) -> ProductLayer:
        """Return real unwrapped radians from the bound unwrap product."""
        return self._unwrap_layer("unwrapped_phase", AssetKind.UNWRAPPED_PHASE)

    @property
    def connected_components(self) -> ProductLayer:
        """Return integer component labels sampled with nearest neighbors."""
        return self._unwrap_layer("connected_components", None, "label")
