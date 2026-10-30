"""Network-facing view over the committed products of a Stack (PROPOSAL-0059).

The collection reads ``ifg/ml_AxR/<pair>`` and ``unwrap/ml_AxR/<pair>``
products in place; nothing is copied into a second layout.

``sel(looks=, pair=, pairs=)`` returns an immutable view bound to a
committed multilook tree or a subset of the planned Pair network
(PROPOSAL-0064); the default view binds ``StackConfig.multilook`` and the
full network.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from faninsar.core import Pair, Pairs
from faninsar.network.readers.interferogram import InterferogramCollection

if TYPE_CHECKING:
    from faninsar.core.acquisition import Acquisition
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.io.storage.scene_store import (
        CoregisteredSceneStore,
        SceneLayers,
    )
    from faninsar.network.layers import InterferogramLayers
    from faninsar.stack.session import Stack

_ASSETS = ("complex", "unw_phase", "wrapped_phase", "amplitude", "coherence")
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


@dataclass(frozen=True, slots=True)
class _Selection:
    """Internal selection carried by one collection view (PROPOSAL-0064).

    ``looks`` picks a committed ``ifg/ml_AxR`` tree; ``pairs`` narrows the
    planned Pair network to a subset. ``None`` means the Stack defaults —
    ``StackConfig.multilook`` and the full planned network.
    """

    looks: tuple[int, int] | None = None
    pairs: Pairs | None = None

    def __post_init__(self) -> None:
        """Validate looks shape; subset-of-plan is checked by ``sel``."""
        if self.looks is not None and (
            not isinstance(self.looks, tuple)
            or len(self.looks) != 2
            or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1
                for v in self.looks
            )
        ):
            message = f"looks must be two positive integers; got {self.looks!r}"
            raise ValueError(message)


class StackInterferogramCollection(InterferogramCollection):
    """Read-only interferogram collection backed by a Stack work directory.

    Parameters
    ----------
    stack : Stack
        The Stack whose planned Pair network and committed multilook trees
        back the view.
    selection : _Selection, optional
        ``looks``/``pairs`` bound to this view. ``None`` selects
        ``stack.config.multilook`` and the whole planned network. Build
        constrained views with :meth:`sel`, never by mutating the
        collection in place.

    """

    def __init__(self, stack: Stack, selection: _Selection | None = None) -> None:
        """Bind the collection to a Stack and an immutable selection."""
        self._stack = stack
        self._selection = selection or _Selection()

    @property
    def looks(self) -> tuple[int, int]:
        """Return the multilook this view reads.

        ``sel(looks=)`` overrides it; otherwise it is
        ``stack.config.multilook``.
        """
        if self._selection.looks is not None:
            return self._selection.looks
        looks = self._stack.config.multilook
        return (int(looks[0]), int(looks[1]))

    @property
    def available_looks(self) -> list[tuple[int, int]]:
        """Return the committed multilook views under ``work_dir/ifg``.

        The list enumerates ``ml_AxR`` directories on disk, so every entry
        names a committed product view regardless of completeness.
        """
        ifg_root = self._stack.config.work_dir / "ifg"
        if not ifg_root.is_dir():
            return []
        views: list[tuple[int, int]] = []
        for path in ifg_root.iterdir():
            if not path.is_dir() or not path.name.startswith("ml_"):
                continue
            try:
                az_str, rg_str = path.name[3:].split("x", 1)
                views.append((int(az_str), int(rg_str)))
            except ValueError:
                continue
        return sorted(set(views))

    def sel(
        self,
        *,
        looks: tuple[int, int] | None = None,
        pairs: Pairs | Pair | str | None = None,
    ) -> StackInterferogramCollection | InterferogramLayers:
        """Return a view bound to a committed multilook or Pair subset.

        Parameters
        ----------
        looks : tuple[int, int], optional
            ``(azimuth, range)`` look factors of a committed
            ``ifg/ml_AxR`` view. The view must exist on disk; reading it
            still enforces per-pair validation. ``None`` keeps the current
            ``self.looks``.
        pairs : Pairs, Pair, or str, optional
            A :class:`Pairs` selects a subset of the planned network and
            returns a new :class:`StackInterferogramCollection`; a scalar
            :class:`Pair` or canonical pair id demotes the collection to
            that pair's :class:`InterferogramLayers` — the same object
            ``collection[pair]`` returns. Mirrors the xarray ``sel``
            scalar-vs-list demotion rule.

        Returns
        -------
        StackInterferogramCollection or InterferogramLayers
            A new collection when ``pairs`` is a :class:`Pairs`, or the
            pair's layer view when ``pairs`` is a scalar.

        Raises
        ------
        TypeError
            If ``pairs`` is neither :class:`Pairs`, :class:`Pair`, nor
            ``str``.
        KeyError
            If a scalar ``pairs`` is not part of the planned network.
        ValueError
            If ``looks`` is not two positive integers, ``pairs`` is not a
            non-empty subset of the planned network, or no dims select
            anything.

        """
        if looks is not None and (
            not isinstance(looks, tuple)
            or len(looks) != 2
            or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in looks
            )
        ):
            message = f"looks must be two positive integers; got {looks!r}"
            raise ValueError(message)
        selection = _Selection(
            looks=self._selection.looks if looks is None else looks,
            pairs=self._selection.pairs,
        )
        if pairs is None:
            if looks is None:
                message = "sel requires at least one of looks= or pairs="
                raise ValueError(message)
            return StackInterferogramCollection(self._stack, selection)
        if isinstance(pairs, Pairs):
            planned = set(self._planned_pairs().to_names().tolist())
            requested = pairs.to_names().tolist()
            if not requested or not set(requested).issubset(planned):
                message = "pairs must be a non-empty subset of the planned network"
                raise ValueError(message)
            # Copy so a caller retaining the input Pairs cannot mutate this
            # view's selection via Pairs.sort(inplace=True) (PROPOSAL-0064
            # review: immutable views must not alias a mutable caller object).
            selection = _Selection(looks=selection.looks, pairs=pairs.copy())
            return StackInterferogramCollection(self._stack, selection)
        if isinstance(pairs, Pair):
            pair_id = pairs.name
        elif isinstance(pairs, str):
            pair_id = pairs
        else:
            message = f"pairs must be Pairs, Pair, or str; got {type(pairs).__name__}"
            raise TypeError(message)
        if pair_id not in self.pairs.to_names().tolist():
            raise KeyError(pairs)
        view = StackInterferogramCollection(self._stack, selection)
        return view._load_layers(pair_id)

    def __getitem__(
        self, pair: str | Pair | Pairs
    ) -> InterferogramLayers | dict[str, InterferogramLayers]:
        """Return lazy, typed layers for one planned Pair, or a subset.

        Parameters
        ----------
        pair : str, Pair, or Pairs
            Canonical date-pair identifier, a single :class:`Pair`, or a
            :class:`Pairs` selection. A ``str`` or ``Pair`` returns one
            layer view; a ``Pairs`` returns a ``dict`` keyed on the
            canonical pair names in the input's order.

        Returns
        -------
        InterferogramLayers or dict[str, InterferogramLayers]
            Flat view exposing ``wrapped_phase``, ``complex``, ``coherence``,
            ``amplitude``, ``unwrapped_phase``, ``connected_components``, and
            the accessible-name listing ``layers``. The ``wrapped_phase`` and
            ``amplitude`` layers are virtual and derive from the committed
            complex IFG. A ``Pairs`` input yields one such view per pair.

        Raises
        ------
        KeyError
            If any requested Pair is not part of this view's selection.

        """
        if isinstance(pair, Pairs):
            return {name: self._load_layers(name) for name in pair.to_names().tolist()}
        pair_id = pair.name if isinstance(pair, Pair) else str(pair)
        return self._load_layers(pair_id)

    def _load_layers(self, pair_id: str) -> InterferogramLayers:
        """Build the validated lazy-layer view for one committed Pair product."""
        from faninsar.io.storage.ifg_store import IFG_KIND, InterferogramArtifactStore
        from faninsar.io.storage.product import open_product
        from faninsar.io.storage.scene_store import SCENE_KIND, CoregisteredSceneStore
        from faninsar.network.layers import InterferogramLayers
        from faninsar.processing.errors import reject_invalid_state

        if pair_id not in self.pairs.to_names().tolist():
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
        """Return the IFG product directory of this view's multilook."""
        looks = self.looks
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    @property
    def index_metadata(self) -> dict[str, Any]:
        """Return the Network index metadata for this view's Pair set."""
        stores = self._selected_stores()
        assets_by_pair: dict[str, list[str]] = {}
        for store in stores:
            assets = ["complex", "wrapped_phase", "amplitude"]
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
        stores = self._selected_stores()
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

    def _planned_pairs(self) -> Pairs:
        """Return the planned Stack Pair graph (unselected)."""
        from faninsar.stack.session import _iter_pair_dates

        return Pairs.from_names(
            [
                f"{first}_{second}"
                for first, second in _iter_pair_dates(self._stack.pairs)
            ]
        )

    @property
    def pairs(self) -> Pairs:
        """Return this view's Pair graph.

        The planned network on the default view, or the selected subset on
        a ``sel(pairs=)`` view.
        """
        if self._selection.pairs is not None:
            return self._selection.pairs
        return self._planned_pairs()

    def _stores(self) -> list[InterferogramArtifactStore]:
        """Open the exact planned IFG product set of this view's multilook.

        The strict check runs against the *planned* network regardless of a
        ``sel(pairs=)`` subset: network-integrity discovery never shrinks
        (PROPOSAL-0059 AC#6). Use :meth:`_selected_stores` for the subset
        view's reads.
        """
        return self._stack._pair_artifact_stores(looks=self.looks)

    def _selected_stores(self) -> list[InterferogramArtifactStore]:
        """Open the IFG product stores of this view's selected pairs.

        On the default view this is the strict full-network set. On a
        ``sel(pairs=)`` subset it opens only the selected products; each
        still passes :meth:`_pair_store`'s per-pair validation.
        """
        if self._selection.pairs is None:
            return self._stores()
        return [self._pair_store(pair_id) for pair_id in self.pairs.to_names()]

    def _pair_store(self, pair: str) -> InterferogramArtifactStore:
        """Return the IFG product store of one planned Pair.

        Only the requested Pair product is opened, and it is validated against
        its own planned Pair identity, looks, and domain. Other Pairs of the
        network are never consulted, so a missing or damaged product of another
        Pair cannot block this query.

        Raises
        ------
        KeyError
            If the Pair is not part of this view's selection.
        InvalidProcessingStateError
            If the requested Pair product is missing, damaged, or inconsistent
            with the Stack configuration.

        """
        from faninsar.io.storage.ifg_store import InterferogramArtifactStore
        from faninsar.processing.errors import reject_invalid_state

        pair_id = str(pair)
        if pair_id not in self.pairs.to_names().tolist():
            raise KeyError(pair)
        store = InterferogramArtifactStore.open(self.root / pair_id)
        if _pair_id(store) != pair_id:
            reject_invalid_state(
                "Stack IFG artifact pair differs from the configured network"
            )
        azimuth_looks, range_looks = (int(self.looks[0]), int(self.looks[1]))
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
        if canonical in {"wrapped_phase", "amplitude", "complex"}:
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
        """Open a named asset across this view's Pair network.

        Short public names (``wrapped``, ``unwrapped``) are accepted alongside
        the internal asset names (``wrapped_phase``, ``unw_phase``).

        On the default view a whole-network read keeps the strict exact
        planned-set check; an explicit ``pairs`` argument further narrows
        within this view's selection.
        """
        del chunks
        requested = (
            self.pairs.to_names().tolist()
            if pairs is None
            else pairs.to_names().tolist()
        )
        canonical = _canonical_asset_name(name)
        if pairs is None and self._selection.pairs is None:
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
            return "complex" in selected._payloads
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
        if str(pair) not in self.pairs.to_names().tolist():
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
        """Return a summary of this view's committed Stack products."""
        index = self.index_metadata
        return {
            "root": str(self.root),
            "pair_count": index["pair_count"],
            "pairs": index["pairs"],
            "assets_by_pair": index["assets_by_pair"],
            "common_grid": True,
        }

    def unwrap_write_ids(self) -> dict[str, str]:
        """Return the ``write_id`` of each selected Pair's current unwrap product."""
        return {
            _pair_id(store): store.unwrap_product().write_id
            for store in self._selected_stores()
        }

    def to_zarr_stack(
        self,
        name: str = "unw_phase",
        *,
        zarr_path: str | Path,
        chunks: tuple[int, int, int] = (1, 512, 512),
        overwrite: bool = False,
    ) -> Path:
        """Export one asset across this view's Pair network as a Zarr cube.

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


class StackSceneCollection(dict):
    """Dict of committed scene products with session-bound layer views.

    ``stack.scenes`` returns this ``dict`` subclass purely to overload
    ``__getitem__`` for :class:`Acquisition` input — ``["date"]`` still
    returns a :class:`CoregisteredSceneStore`, but bound to the Stack so
    its ``.layers`` view carries ``_session``/``_upstream``. An
    ``Acquisition`` index returns ``dict[date, SceneLayers]`` keyed on
    dates.
    """

    def __init__(
        self,
        stack: Stack,
        stores: dict[str, CoregisteredSceneStore],
    ) -> None:
        """Bind the collection to a Stack and its committed scene stores."""
        super().__init__(stores)
        self._stack = stack

    def _bound(self, store: CoregisteredSceneStore) -> CoregisteredSceneStore:
        """Return the store re-bound to this Stack's session."""
        from dataclasses import replace

        from faninsar.io.storage.product import open_product

        reference = self._stack.reference_scene
        upstream = (open_product(reference.root),) if reference is not None else ()
        return replace(store, _session=self._stack, _upstream=upstream)

    def __getitem__(
        self, key: str | Acquisition
    ) -> CoregisteredSceneStore | dict[str, SceneLayers]:
        """Return one scene store, or per-date ``SceneLayers`` for an index.

        Parameters
        ----------
        key : str or Acquisition
            A ``str`` date id or a single ``Acquisition`` record returns the
            committed :class:`CoregisteredSceneStore`; an ``Acquisition``
            index (``pd.DatetimeIndex``) returns ``dict[date, SceneLayers]``
            keyed on the dates it names.

        """
        from faninsar.core.acquisition import Acquisition

        if isinstance(key, Acquisition):
            if hasattr(key, "_physical_key"):
                date_id = key.strftime("%Y%m%d")
                return self._bound(super().__getitem__(date_id))
            return {
                date: self._bound(store).layers
                for date, store in self.items()
                if date in {item.strftime("%Y%m%d") for item in key}
            }
        return self._bound(super().__getitem__(key))
