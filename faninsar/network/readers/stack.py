"""Network-facing view over the committed products of a Stack (PROPOSAL-0059).

The collection reads ``ifg/ml_AxR/<pair>`` and ``unwrap/ml_AxR/<pair>``
products in place; nothing is copied into a second layout.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from faninsar.network.readers.interferogram import InterferogramCollection

if TYPE_CHECKING:
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.network.layers import InterferogramLayers
    from faninsar.stack.session import Stack

_ASSETS = ("complex_ifg", "unw_phase", "wrapped_phase", "amplitude", "coherence")
_READABLE_ASSETS = frozenset(_ASSETS)
#: Short public layer spellings accepted alongside the internal asset names.
_SHORT_ASSET_NAMES = {"wrapped": "wrapped_phase", "unwrapped": "unw_phase"}
#: Layers derived at read time from ``complex_ifg``; no file exists for them.
_VIRTUAL_ASSETS = ("wrapped_phase", "amplitude")


def _canonical_asset_name(name: str) -> str:
    """Map a short public name to its internal asset name and validate it."""
    canonical = _SHORT_ASSET_NAMES.get(str(name), str(name))
    if canonical not in _READABLE_ASSETS:
        message = f"unknown Stack interferogram asset: {name}"
        raise ValueError(message)
    return canonical


class StackInterferogramCollection(InterferogramCollection):
    """Read-only interferogram collection backed by a Stack work directory.

    Parameters
    ----------
    stack : Stack
        The Stack whose planned Pair network and multilook select the
        products.

    """

    def __init__(self, stack: Stack) -> None:
        """Bind the collection to a Stack."""
        self._stack = stack

    def __getitem__(self, pair: str) -> InterferogramLayers:
        """Return lazy, typed layers for one planned Pair.

        Parameters
        ----------
        pair : str
            Canonical date-pair identifier.

        Returns
        -------
        InterferogramLayers
            Flat view exposing ``wrapped``, ``complex_ifg``, ``coherence``,
            ``amplitude``, ``unwrapped``, ``connected_components``, and the
            accessible-name listing ``layers``. The ``wrapped`` and
            ``amplitude`` layers are virtual and derive from the committed
            complex IFG.

        Raises
        ------
        KeyError
            If the Pair is not part of this Stack.

        """
        from faninsar.io.storage.ifg_store import IFG_KIND, InterferogramArtifactStore
        from faninsar.io.storage.product import open_product
        from faninsar.io.storage.scene_store import SCENE_KIND, CoregisteredSceneStore
        from faninsar.network.layers import InterferogramLayers
        from faninsar.processing.errors import reject_invalid_state

        pair_id = str(pair)
        if pair_id not in self.pairs().to_names().tolist():
            raise KeyError(pair_id)
        store = InterferogramArtifactStore.open(self.root / pair_id)
        reference = self._stack.reference_scene
        if reference is None:
            reject_invalid_state(
                "the Reference scene is missing; run coregister_scenes"
            )
        if (
            store.domain != reference.domain
            or store.grid_identity != reference.grid_identity
        ):
            reject_invalid_state(
                "IFG and Reference scene use different coordinate grids"
            )
        upstream = [open_product(reference.root)]
        for role, date_id in zip(("primary", "secondary"), store.pair, strict=True):
            if date_id == self._stack.reference:
                expected = reference.write_id
            else:
                scene_product = open_product(
                    self._stack.config.work_dir / "coreg" / date_id, kind=SCENE_KIND
                )
                scene = CoregisteredSceneStore.open(scene_product.path)
                if scene.inputs.get("reference") != reference.write_id:
                    reject_invalid_state("IFG source is bound to a replaced Reference")
                expected = scene.write_id
                upstream.append(scene_product)
            if store.source_manifest_digests.get(role) != expected:
                reject_invalid_state(
                    "IFG source or Reference was replaced; recompute the IFG"
                )
        return InterferogramLayers(
            open_product(store.root, kind=IFG_KIND),
            reference.grid_shape,
            store.looks,
            store.crs,
            tuple(upstream),
            _session=self._stack,
        )

    @property
    def root(self) -> Path:
        """Return the IFG product directory of the configured multilook."""
        looks = self._stack.config.multilook
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    @property
    def index_metadata(self) -> dict[str, Any]:
        """Return the Network index metadata for the committed Pair set."""
        stores = self._stores()
        assets_by_pair: dict[str, list[str]] = {}
        for store in stores:
            assets = ["complex_ifg", "wrapped_phase", "amplitude"]
            if "coherence" in store._payloads:
                assets.append("coherence")
            if store.has_unwrapped():
                assets.append("unw_phase")
            assets_by_pair[_pair_id(store)] = assets
        return {
            "type": "NetworkInterferogramIndex",
            "pair_count": len(stores),
            "pairs": [_pair_id(store) for store in stores],
            "assets_by_pair": assets_by_pair,
            "common_grid": True,
        }

    @property
    def spatial_metadata(self) -> dict[str, Any]:
        """Return the common spatial metadata of the IFG products.

        Returns
        -------
        dict
            ``domain``, ``shape``, ``grid_identity``, and for geo products
            ``crs``/``transform``. Radar collections omit geographic metadata
            instead of fabricating it.

        """
        stores = self._stores()
        if not stores:
            message = "Stack has no interferogram products"
            raise ValueError(message)
        first = stores[0]
        values: dict[str, Any] = {
            "domain": first.domain,
            "shape": list(first.shape),
            "grid_identity": first.grid_identity,
        }
        if first.crs is not None:
            values["crs"] = first.crs
        if first.transform is not None:
            values["transform"] = list(first.transform)
        return values

    def pairs(self) -> Any:
        """Return the planned Stack Pair graph."""
        from faninsar.core import Pairs
        from faninsar.stack.session import _iter_pair_dates

        return Pairs.from_names(
            [
                f"{first}_{second}"
                for first, second in _iter_pair_dates(self._stack.pairs)
            ]
        )

    def _stores(self) -> list[InterferogramArtifactStore]:
        """Open the exact planned IFG product set."""
        return self._stack._pair_artifact_stores(looks=self._stack.config.multilook)

    def _pair_store(self, pair: str) -> InterferogramArtifactStore:
        """Return the IFG product store of one planned Pair.

        Only the requested Pair product is opened, and it is validated against
        its own planned Pair identity, looks, and domain. Other Pairs of the
        network are never consulted, so a missing or damaged product of another
        Pair cannot block this query.

        Raises
        ------
        KeyError
            If the Pair is not part of this Stack.
        InvalidProcessingStateError
            If the requested Pair product is missing, damaged, or inconsistent
            with the Stack configuration.

        """
        from faninsar.io.storage.ifg_store import InterferogramArtifactStore
        from faninsar.processing.errors import reject_invalid_state

        pair_id = str(pair)
        if pair_id not in self.pairs().to_names().tolist():
            raise KeyError(pair)
        store = InterferogramArtifactStore.open(self.root / pair_id)
        if _pair_id(store) != pair_id:
            reject_invalid_state(
                "Stack IFG artifact pair differs from the configured network"
            )
        azimuth_looks, range_looks = (
            int(self._stack.config.multilook[0]),
            int(self._stack.config.multilook[1]),
        )
        if store.looks != (azimuth_looks, range_looks):
            reject_invalid_state("Stack IFG artifacts use mixed or unexpected looks")
        if store.domain != self._stack.config.coregistration_grid:
            reject_invalid_state("Stack IFG artifacts use mixed or unexpected domains")
        return store

    def _array(self, store: InterferogramArtifactStore, name: str) -> np.ndarray:
        """Read one named layer of one Pair, loading only that layer."""
        canonical = _canonical_asset_name(name)
        if canonical == "unw_phase":
            if not store.has_unwrapped():
                raise FileNotFoundError(store.unwrap_root)
            return np.asarray(store.read_unwrapped_layer("unwrapped_phase"))
        if canonical == "coherence":
            if "coherence" not in store._payloads:
                message = "coherence is not present"
                raise FileNotFoundError(message)
            return np.asarray(store.read_layer("coherence"))
        if canonical in {"wrapped_phase", "amplitude", "complex_ifg"}:
            return np.asarray(store.read_layer(canonical))
        message = f"unknown Stack interferogram asset: {name}"
        raise ValueError(message)

    def open_stack(
        self,
        name: str,
        *,
        pairs: Any = None,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Open a named asset across the planned Pair network.

        Short public names (``wrapped``, ``unwrapped``) are accepted alongside
        the internal asset names (``wrapped_phase``, ``unw_phase``).

        An explicit ``pairs`` subset touches only the selected Pair products;
        a whole-network read keeps the strict exact planned-set check.
        """
        del chunks
        requested = (
            self.pairs().to_names().tolist()
            if pairs is None
            else pairs.to_names().tolist()
        )
        canonical = _canonical_asset_name(name)
        if pairs is None:
            by_pair = {_pair_id(store): store for store in self._stores()}
            stores = [by_pair[pair] for pair in requested]
        else:
            stores = [self._pair_store(pair) for pair in requested]
        arrays = [self._array(store, canonical) for store in stores]
        return xr.DataArray(
            np.stack(arrays, axis=0),
            dims=("pair", "y", "x"),
            coords={"pair": requested},
            name=canonical,
        )

    def open(self, pair: str, name: str, **kwargs: Any) -> xr.DataArray:
        """Open one named asset of one Pair, loading only that layer.

        The query is bound to the requested Pair product; other Pairs of the
        network are never read and cannot affect the result.
        """
        del kwargs
        canonical = _canonical_asset_name(name)
        return xr.DataArray(
            self._array(self._pair_store(str(pair)), canonical),
            dims=("y", "x"),
            name=canonical,
        )

    def exists(self, pair: str, name: str) -> bool:
        """Return whether a named asset is readable for a Pair.

        The check reads product metadata of the requested Pair only; no raster
        payload is loaded and other Pairs of the network are never consulted.

        Computable virtual layers (``wrapped_phase``, ``amplitude``) count as
        readable when their source ``complex_ifg`` is persisted; item/path
        adapters still refuse to name a file for them.
        """
        try:
            canonical = _canonical_asset_name(name)
            selected = self._pair_store(str(pair))
        except (FileNotFoundError, KeyError, ValueError):
            return False
        if canonical == "unw_phase":
            return selected.has_unwrapped()
        if canonical in _VIRTUAL_ASSETS:
            return "complex_ifg" in selected._payloads
        return canonical in selected._payloads

    def path(self, pair: str, name: str) -> Path:
        """Return the product path of one Pair asset.

        Virtual derived layers have no file; their paths raise
        :class:`FileNotFoundError` instead of inventing a payload location.

        Raises
        ------
        FileNotFoundError
            For ``wrapped_phase``/``amplitude`` (derived from the committed
            complex IFG), a missing unwrap product, or an optional payload
            that was never persisted.

        """
        canonical = _canonical_asset_name(name)
        if canonical not in _ASSETS:
            message = f"unknown Stack interferogram asset: {name}"
            raise ValueError(message)
        selected = self._pair_store(str(pair))
        if canonical in _VIRTUAL_ASSETS:
            message = (
                f"{canonical} is a virtual layer derived from complex_ifg; "
                f"no file exists for it in {selected.root}"
            )
            raise FileNotFoundError(message)
        if canonical == "unw_phase":
            if not selected.has_unwrapped():
                raise FileNotFoundError(selected.unwrap_root)
            return selected.unwrap_root / "unwrapped_phase.zarr"
        payload = selected._payloads.get(canonical)
        if payload is None:
            message = f"{canonical} is not present in the IFG product {selected.root}"
            raise FileNotFoundError(message)
        return selected.root / str(payload["href"])

    def item(self, pair: str) -> dict[str, Any]:
        """Return an item-like record for one Stack Pair.

        Assets enumerate real payloads only: virtual derived layers are
        omitted rather than given a fabricated href.
        """
        if str(pair) not in self.pairs().to_names().tolist():
            raise KeyError(pair)
        assets: dict[str, Any] = {}
        for name in _ASSETS:
            try:
                assets[name] = {"href": str(self.path(str(pair), name))}
            except FileNotFoundError:
                continue
        return {
            "id": str(pair),
            "pair": str(pair),
            "assets": assets,
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the committed Stack products."""
        index = self.index_metadata
        return {
            "root": str(self.root),
            "pair_count": index["pair_count"],
            "pairs": index["pairs"],
            "assets_by_pair": index["assets_by_pair"],
            "common_grid": True,
        }

    def unwrap_write_ids(self) -> dict[str, str]:
        """Return the ``write_id`` of each Pair's current unwrap product."""
        return {
            _pair_id(store): store.unwrap_product().write_id for store in self._stores()
        }

    def to_zarr_stack(
        self,
        name: str = "unw_phase",
        *,
        zarr_path: str | Path,
        chunks: tuple[int, int, int] = (1, 512, 512),
        overwrite: bool = False,
    ) -> Path:
        """Export one asset across the Pair network as an xarray Zarr cube.

        Parameters
        ----------
        name : str, optional
            Asset to export.
        zarr_path : str or pathlib.Path
            Destination outside the Stack product tree.
        chunks : tuple[int, int, int], optional
            ``(pair, y, x)`` chunking.
        overwrite : bool, optional
            Replace an existing destination.

        Returns
        -------
        pathlib.Path
            The written Zarr store.

        """
        out = Path(zarr_path)
        if out.exists() and not overwrite:
            raise FileExistsError(out)
        if out.exists():
            shutil.rmtree(out)
        data = self.open_stack(name).chunk(
            {"pair": chunks[0], "y": chunks[1], "x": chunks[2]}
        )
        data.to_dataset(name=name).to_zarr(str(out), mode="w", consolidated=False)
        return out


def _pair_id(store: InterferogramArtifactStore) -> str:
    """Return the canonical Pair id of one IFG product."""
    return f"{store.pair[0]}_{store.pair[1]}"
