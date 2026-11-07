"""Network-facing view over the committed products of a Stack (PROPOSAL-0059).

The collection reads ``ifg/ml_AxR/<pair>`` and ``unwrap/ml_AxR/<pair>``
products in place; nothing is copied into a second layout.

``sel(looks=, pairs=, polygon=, track=, direction=, pol=, burst=)`` returns
an immutable view bound to a committed multilook tree, a subset of the
planned Pair network, one ROI-polygon subtree, or a track-axis date subset
(PROPOSAL-0064); the default view binds ``StackConfig.multilook`` and the
full network.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from faninsar.core import Pair, Pairs
from faninsar.network.readers.interferogram import InterferogramCollection

if TYPE_CHECKING:
    from faninsar.core.acquisition import Acquisition
    from faninsar.core.baseline import Baselines
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.io.storage.product import Product
    from faninsar.io.storage.scene_store import (
        CoregisteredSceneStore,
        SceneLayers,
    )
    from faninsar.network.layers import InterferogramLayers, ProductLayer
    from faninsar.processing.mosaicking import MergeLayoutWindow
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
    planned Pair network to a subset; ``polygon`` reads one ``p<i>``
    subtree of a multi-polygon ROI; ``track``/``pol`` hold the resolved
    track label and polarization axis selectors. ``None`` means the Stack
    defaults — ``StackConfig.multilook`` and the full planned network.
    """

    looks: tuple[int, int] | None = None
    pairs: Pairs | None = None
    polygon: int | None = None
    track: str | None = None
    pol: tuple[str, ...] | None = None
    representation: Literal["merged", "burst"] = "merged"
    burst_ids: tuple[str, ...] | None = None
    merge_on_read: bool = False

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
        polygon: int | None = None,
        track: str | None = None,
        direction: str | None = None,
        pol: str | list[str] | tuple[str, ...] | None = None,
        burst: int | list[int] | tuple[int, ...] | None = None,
        representation: Literal["merged", "burst"] | None = None,
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
        polygon : int, optional
            Index of the effective-ROI polygon whose ``p<i>`` product subtree
            to read. Valid only when the Stack ROI decomposes to more than
            one polygon (``stack.roi_polygon_count > 1``); a
            :class:`BoundingBox`, a single polygon, or no ROI rejects it.
            On a ``sel(track=)`` view the index is checked against
            :attr:`Stack.polygon_tracks` — a polygon that falls on another
            track is rejected with an error naming the correct track.
            ``None`` keeps the current ``self._selection.polygon``.
        track : str, optional
            Track label of the ``"88A"`` form (or a bare relative-orbit
            number, selecting both direction letters). Keeps only the
            planned pairs whose two acquisition dates lie on the selected
            track; cross-track pairs are never part of the narrowed view.
            ``None`` keeps the current ``self._selection.track``.
        direction : str, optional
            ``"ASC"``/``"DESC"`` (or the full ``"ASCENDING"``/
            ``"DESCENDING"``). Narrows the date axis the same way
            ``track=`` does; composes with it.
        pol : str or sequence of str, optional
            Polarization channel(s) (``"VV"``, ``"VH"``, ``"HH"``,
            ``"HV"``). Keeps pairs whose acquisitions all carry the
            requested channels; the value is stored on the view so
            consumers see the polarization constraint.
        burst : int or sequence of int, optional
            Absolute burst id(s). When the Stack's per-date burst metadata
            is known the date axis is narrowed to acquisitions covering
            the requested ids; unknown metadata fails closed with
            :class:`ValueError` rather than silently ignoring the
            selector. With ``representation="burst"`` the same values select
            burst IFG products under ``ifg/bursts/``.
        representation : {"merged", "burst"}, optional
            Product representation to read. ``"merged"`` (the default) opens
            ``ifg/ml_AxR/<pair>[/pN]``. ``"burst"`` opens
            ``ifg/bursts/ml_AxR/<pair>/<burst_id>[/pN]`` and requires
            ``burst=`` to name the burst product. ``None`` keeps the current
            view's representation.

        Returns
        -------
        StackInterferogramCollection or InterferogramLayers
            A new collection for non-scalar selections, or the pair's
            layer view when ``pairs`` is a scalar.

        Raises
        ------
        TypeError
            If ``pairs`` is neither :class:`Pairs`, :class:`Pair`, nor
            ``str``.
        KeyError
            If a scalar ``pairs`` is not part of the planned network.
        ValueError
            If ``looks`` is not two positive integers, ``pairs`` is not a
            non-empty subset of the planned network, ``polygon`` is given
            for a single-polygon or absent ROI or names a polygon on
            another track, ``track``/``direction``/``pol``/``burst``
            select nothing or cannot be resolved, ``representation`` is
            invalid, or no dims select anything.

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

        resolved_track = self._resolve_track_selector(track)
        resolved_pol = self._resolve_pol_selector(pol)
        resolved_pairs = self._selection.pairs
        if resolved_track is not None or direction is not None:
            track_dates = self._stack._catalog_dates_for_track(
                track=resolved_track,
                direction=direction,
            )
            narrowed = self._stack._select_pairs_for_track(track_dates)
            resolved_pairs = (
                narrowed
                if resolved_pairs is None
                else resolved_pairs.intersect(narrowed)
            )
            if len(resolved_pairs) == 0:
                label = resolved_track or direction
                message = (
                    f"sel(track={label!r}) selects no pairs of the planned "
                    "network; both endpoints of a pair must lie on the "
                    "selected track"
                )
                raise ValueError(message)
        if resolved_pol is not None:
            pol_dates = self._stack._catalog_dates_for_track(pol=resolved_pol)
            narrowed = self._stack._select_pairs_for_track(pol_dates)
            resolved_pairs = (
                narrowed
                if resolved_pairs is None
                else resolved_pairs.intersect(narrowed)
            )
            if len(resolved_pairs) == 0:
                message = (
                    f"sel(pol={resolved_pol!r}) selects no pairs; not every "
                    "selected date carries the requested channels"
                )
                raise ValueError(message)
        effective_track = (
            resolved_track if resolved_track is not None else self._selection.track
        )
        resolved_polygon = self._resolve_polygon(polygon, effective_track)
        resolved_representation = self._resolve_representation(representation)
        resolved_burst_ids = self._resolve_burst_ids(burst)
        if burst is not None and resolved_representation != "burst":
            resolved_pairs = self._resolve_burst_selector(burst, resolved_pairs)

        selection = _Selection(
            looks=self._selection.looks if looks is None else looks,
            pairs=resolved_pairs,
            polygon=self._selection.polygon if polygon is None else resolved_polygon,
            track=(
                resolved_track if resolved_track is not None else self._selection.track
            ),
            pol=resolved_pol if resolved_pol is not None else self._selection.pol,
            representation=resolved_representation,
            burst_ids=resolved_burst_ids,
            merge_on_read=self._selection.merge_on_read,
        )
        if pairs is None:
            if all(
                value is None
                for value in (
                    looks,
                    polygon,
                    track,
                    direction,
                    pol,
                    burst,
                    representation,
                )
            ):
                message = (
                    "sel requires at least one of looks=, pairs=, polygon=, "
                    "track=, direction=, pol=, burst=, or representation="
                )
                raise ValueError(message)
            return StackInterferogramCollection(self._stack, selection)
        if isinstance(pairs, Pairs):
            # A Pairs argument narrows this view's pair set. It must stay
            # inside the planned network *and* — when the view is
            # track-narrowed — inside the selected track's pair set so a
            # sel(track=).sel(pairs=) chain cannot smuggle a cross-track
            # pair back in.
            planned = set(self._planned_pairs().to_names().tolist())
            requested = pairs.to_names().tolist()
            if not requested or not set(requested).issubset(planned):
                message = "pairs must be a non-empty subset of the planned network"
                raise ValueError(message)
            if selection.pairs is not None and not set(requested).issubset(
                set(selection.pairs.to_names().tolist())
            ):
                message = (
                    "pairs must be a subset of this view's selected network; "
                    "the view is narrowed by track/direction/pol/burst"
                )
                raise ValueError(message)
            # Copy so a caller retaining the input Pairs cannot mutate this
            # view's selection via Pairs.sort(inplace=True) (PROPOSAL-0064
            # review: immutable views must not alias a mutable caller object).
            selection = _Selection(
                looks=selection.looks,
                pairs=pairs.copy(),
                polygon=selection.polygon,
                track=selection.track,
                pol=selection.pol,
                representation=selection.representation,
                burst_ids=selection.burst_ids,
                merge_on_read=selection.merge_on_read,
            )
            return StackInterferogramCollection(self._stack, selection)
        if isinstance(pairs, Pair):
            pair_id = pairs.name
        elif isinstance(pairs, str):
            pair_id = pairs
        else:
            message = f"pairs must be Pairs, Pair, or str; got {type(pairs).__name__}"
            raise TypeError(message)
        # Membership is checked against the view's selected pair set (the
        # track-/pol-/burst-narrowed network), not the whole plan.
        view = StackInterferogramCollection(self._stack, selection)
        if pair_id not in view.pairs.to_names().tolist():
            raise KeyError(pairs)
        return view._load_layers(pair_id)

    def _resolve_track_selector(
        self,
        track: str | None,
    ) -> str | None:
        """Normalize ``track=`` against the Stack's known track labels."""
        return self._stack._normalize_track_selector(track)

    def _resolve_pol_selector(
        self, pol: str | list[str] | tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Retain the channel constraint; mission policy validates it."""
        if pol is None:
            return None
        return (pol,) if isinstance(pol, str) else tuple(pol)

    def _resolve_burst_selector(
        self,
        burst: int | list[int] | tuple[int, ...],
        pairs: Pairs | None,
    ) -> Pairs:
        """Narrow pairs using the Stack's normalized acquisition accessor."""
        dates = self._stack._catalog_dates_for_track(burst=burst)
        narrowed = self._stack._select_pairs_for_track(dates)
        resolved = narrowed if pairs is None else pairs.intersect(narrowed)
        if len(resolved) == 0:
            message = f"sel(burst={burst!r}) selects no pairs of the planned network"
            raise ValueError(message)
        return resolved

    def _resolve_polygon(
        self,
        polygon: int | None,
        track: str | None,
    ) -> int | None:
        """Validate ``polygon=`` against ROI arity and the selected track."""
        if polygon is None:
            return None
        count = int(getattr(self._stack, "roi_polygon_count", 0))
        if count <= 1:
            message = (
                "sel(polygon=) requires a multi-polygon ROI; the effective "
                f"ROI has {count} polygon(s)"
            )
            raise ValueError(message)
        if (
            not isinstance(polygon, int)
            or isinstance(polygon, bool)
            or polygon < 0
            or polygon >= count
        ):
            message = f"polygon must be an integer in [0, {count}); got {polygon!r}"
            raise ValueError(message)
        if track is not None:
            mapping = self._stack.polygon_tracks
            actual = mapping.get(polygon)
            if actual is not None and actual != track:
                message = (
                    f"polygon {polygon} falls on track {actual!r}, not "
                    f"{track!r}; select sel(track={actual!r}) instead"
                )
                raise ValueError(message)
        return polygon

    def _resolve_representation(
        self, representation: Literal["merged", "burst"] | None
    ) -> Literal["merged", "burst"]:
        """Validate ``representation=`` against the allowed product kinds."""
        if representation is None:
            return self._selection.representation
        if representation not in {"merged", "burst"}:
            message = (
                f"representation must be 'merged' or 'burst'; got {representation!r}"
            )
            raise ValueError(message)
        return representation

    def _resolve_burst_ids(
        self, burst: int | list[int] | tuple[int, ...] | None
    ) -> tuple[str, ...] | None:
        """Record burst product identities selected by ``burst=``."""
        if burst is None:
            return self._selection.burst_ids
        values = (burst,) if isinstance(burst, int) else tuple(burst)
        return tuple(str(value) for value in values)

    def merge_bursts(self) -> StackInterferogramCollection:
        """Return a lazy per-pair merged view of burst IFG products.

        Each pair is merged independently through the shared
        :func:`~faninsar.processing.mosaicking.merge_bursts` kernel at read
        time. Source burst products are not rewritten and pairs are never
        merged together.

        Returns
        -------
        StackInterferogramCollection
            The same collection type, with ``merge_on_read`` set so layer
            reads apply the bounded kernel.

        """
        selection = _Selection(
            looks=self._selection.looks,
            pairs=self._selection.pairs,
            polygon=self._selection.polygon,
            track=self._selection.track,
            pol=self._selection.pol,
            representation="merged",
            burst_ids=self._selection.burst_ids,
            merge_on_read=True,
        )
        return StackInterferogramCollection(self._stack, selection)

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
        if self._selection.merge_on_read:
            return self._load_merged_layers(pair_id)
        store = InterferogramArtifactStore.open(self._pair_dir(pair_id))
        reference = self._stack._reference_scene_for(self._selection.polygon)
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
                    self._stack._scene_dir(date_id, self._selection.polygon),
                    kind=SCENE_KIND,
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

    def _load_merged_layers(self, pair_id: str) -> InterferogramLayers:
        """Build a lazy merged view from burst IFG products of one pair."""
        from faninsar.io.storage.ifg_store import IFG_KIND
        from faninsar.io.storage.product import open_product
        from faninsar.network.layers import InterferogramLayers
        from faninsar.processing.errors import reject_invalid_state

        burst_dirs = self._burst_product_dirs(pair_id)
        if not burst_dirs:
            message = (
                f"pair {pair_id!r} has no burst IFG products under "
                "ifg/bursts; form_interferograms(merge=False) first"
            )
            reject_invalid_state(message)
        burst_products = tuple(open_product(path, kind=IFG_KIND) for path in burst_dirs)
        for product in burst_products:
            representation = product.properties.get("faninsar:representation", "burst")
            if representation != "burst":
                reject_invalid_state(
                    f"burst IFG {product.path} is not representation='burst'"
                )
        handle = burst_products[0]
        reference = self._stack._reference_scene_for(self._selection.polygon)
        if reference is None:
            reject_invalid_state(
                "the Reference scene is missing; run coregister_scenes"
            )
        grid = handle.properties.get(
            "faninsar:grid_shape", handle.properties["faninsar:shape"]
        )
        shape = (int(grid[0]), int(grid[1]))
        layout, layout_product = _opened_merge_layout(
            self._stack, polygon=self._selection.polygon
        )
        return InterferogramLayers(
            handle,
            shape,
            self.looks,
            handle.properties.get("proj:code"),
            (open_product(reference.root),),
            _session=self._stack,
            _merge_on_read=True,
            _merge_layout=layout,
            _layout_product=layout_product,
            _burst_products=burst_products,
        )

    @property
    def root(self) -> Path:
        """Return the IFG product directory of this view's multilook."""
        looks = self.looks
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    def _pair_dir(self, pair_id: str) -> Path:
        """Return the product directory of one Pair for this view's polygon."""
        if self._selection.representation == "burst":
            burst_ids = self._selection.burst_ids or ()
            if len(burst_ids) != 1:
                message = (
                    "sel(representation='burst') requires burst= to name one "
                    f"burst IFG product; got {burst_ids!r}"
                )
                raise ValueError(message)
            return self._burst_pair_dir(pair_id, burst_ids[0])
        base = self.root / pair_id
        if self._selection.polygon is None:
            return base
        return base / f"p{self._selection.polygon}"

    def _burst_pair_dir(self, pair_id: str, burst_id: str) -> Path:
        """Return the burst IFG product directory of one pair and burst."""
        looks = self.looks
        base = (
            self._stack.config.work_dir
            / "ifg"
            / "bursts"
            / f"ml_{looks[0]}x{looks[1]}"
            / pair_id
            / str(burst_id)
        )
        if self._selection.polygon is None:
            return base
        return base / f"p{self._selection.polygon}"

    def _burst_product_dirs(self, pair_id: str) -> list[Path]:
        """Discover burst IFG product directories of one pair."""
        if self._selection.burst_ids:
            return [
                self._burst_pair_dir(pair_id, burst_id)
                for burst_id in self._selection.burst_ids
            ]
        looks = self.looks
        parent = (
            self._stack.config.work_dir
            / "ifg"
            / "bursts"
            / f"ml_{looks[0]}x{looks[1]}"
            / pair_id
        )
        if not parent.is_dir():
            return []
        paths: list[Path] = []
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            candidate = (
                child
                if self._selection.polygon is None
                else child / f"p{self._selection.polygon}"
            )
            if (candidate / "item.json").is_file():
                paths.append(candidate)
        return paths

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

    @property
    def baseline(self) -> Baselines:
        """Return the signed perpendicular baseline of this view's network.

        Each acquisition's value is the scene baseline persisted on its
        coreg product at coregistration (``faninsar:baseline_perpendicular_m``,
        relative to the Stack Reference — signed via the ``velocity x look``
        cross-track convention, PROPOSAL-0065). Because the values are already
        reference-relative, :class:`~faninsar.core.baseline.Baselines` is
        populated directly per date rather than by ``from_pair_wise``
        inversion; per-pair values come from
        :meth:`Baselines.to_pair_wise` on the view's :attr:`pairs`.

        Scenes committed before the field existed read ``NaN``; a Reference
        product without the field reads ``0.0``.

        Returns
        -------
        Baselines
            Signed perpendicular baseline per acquisition date of this
            view's Pair network.

        """
        from faninsar.core.baseline import Baselines

        scenes = self._stack.scenes
        dates = list(self.pairs.dates)
        values = [
            scenes[date.strftime("%Y%m%d")].baseline["perpendicular_m"]
            for date in dates
        ]
        return Baselines(dates, np.asarray(values, dtype=np.float32))

    def _stores(self) -> list[InterferogramArtifactStore]:
        """Open the exact planned IFG product set of this view's multilook.

        The strict check runs against the *planned* network regardless of a
        ``sel(pairs=)`` subset: network-integrity discovery never shrinks
        (PROPOSAL-0059 AC#6). Use :meth:`_selected_stores` for the subset
        view's reads.
        """
        return self._stack._pair_artifact_stores(
            looks=self.looks, polygon=self._selection.polygon
        )

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
        store = InterferogramArtifactStore.open(self._pair_dir(pair_id))
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

    def __str__(self) -> str:
        """Return a one-line summary of this view."""
        return (
            f"StackInterferogramCollection({len(self.pairs)} pairs, "
            f"looks={self.looks[0]}x{self.looks[1]})"
        )

    def __repr__(self) -> str:
        """Return a header plus the Pairs table of this view."""
        looks = self.looks
        header = (
            f"StackInterferogramCollection({len(self.pairs)} pairs, "
            f"looks={looks[0]}x{looks[1]}, root={self.root!r})"
        )
        return f"{header}\n{self.pairs._to_frame().__repr__()}"


def _pair_id(store: InterferogramArtifactStore) -> str:
    """Return the canonical Pair id of one IFG product."""
    return f"{store.pair[0]}_{store.pair[1]}"


def _merge_layout_dir(stack: Stack, polygon: int | None = None) -> Path:
    """Return the layout directory published for this Reference.

    Mission coregistration writes ``{work_dir}/layout/{reference}``. A
    multi-polygon subtree may also place the product at ``.../pN`` or,
    when the scene lives at ``coreg/<date>/pN``, beside that coreg tree
    using the same walk-up as ``publish_reference_merge_layout``.
    """
    reference = str(stack.reference)
    canonical = stack.config.work_dir / "layout" / reference
    if polygon is None:
        return canonical
    nested = canonical / f"p{polygon}"
    if (nested / "item.json").is_file():
        return nested
    if (canonical / "item.json").is_file():
        return canonical
    parent = stack._scene_dir(reference, polygon).parent
    root = parent.parent if parent.name == "coreg" else parent
    return root / "layout" / reference


def _opened_merge_layout(
    stack: Stack, polygon: int | None = None
) -> tuple[MergeLayoutWindow | None, Product | None]:
    """Open ``work_dir/layout/<reference>`` when that product exists.

    Collection views never call :func:`build_merge_layout` to invent owner
    labels from burst bounding boxes. Missing layouts stay ``None`` so
    ``merge_bursts()`` can still return a lazy collection; the read path
    rejects the missing layout.
    """
    from faninsar.io.storage.merge_layout import LAYOUT_KIND, open_merge_layout
    from faninsar.io.storage.product import open_product

    path = _merge_layout_dir(stack, polygon)
    if not (path / "item.json").is_file():
        return None, None
    product = open_product(path, kind=LAYOUT_KIND)
    return open_merge_layout(product), product


class _MergeBoundSceneStore:
    """Scene store proxy whose ``.layers`` apply merge-on-read."""

    def __init__(
        self,
        store: CoregisteredSceneStore,
        layout: MergeLayoutWindow | None,
        session: Stack,
        upstream: tuple[Any, ...],
        layout_product: Product | None = None,
    ) -> None:
        self._store = store
        self._layout = layout
        self._layout_product = layout_product
        self._session = session
        self._upstream = upstream

    def __getattr__(self, name: str) -> Any:
        """Forward store attributes that this proxy does not override."""
        return getattr(self._store, name)

    @property
    def layers(self) -> Any:
        """Return the lazy merged scene layer view."""
        from faninsar.io.storage.product import open_product
        from faninsar.io.storage.scene_store import SCENE_KIND
        from faninsar.network.layers import MergedBurstLayers

        product = open_product(self._store.root, kind=SCENE_KIND)
        return MergedBurstLayers(
            product,
            self._store.grid_shape,
            origin=(0, 0),
            crs=self._store.crs,
            extent=self._store.grid_shape,
            _upstream=self._upstream,
            _session=self._session,
            _merge_layout=self._layout,
            _layout_product=self._layout_product,
        )


class StackSceneCollection(dict):
    """Dict of committed scene products with session-bound layer views.

    ``stack.scenes`` returns this ``dict`` subclass — ``["date"]`` still
    returns a :class:`CoregisteredSceneStore`, but bound to the Stack so
    its ``.layers`` view carries ``_session``/``_upstream``. Beyond the
    dict behaviour it mirrors :class:`StackInterferogramCollection`:
    :attr:`acquisitions` exposes the date axis as an :class:`Acquisition`
    index and :meth:`sel` returns immutable date-subset views (or demotes
    to one date's :class:`SceneLayers` on a scalar key).
    """

    def __init__(
        self,
        stack: Stack,
        stores: dict[str, CoregisteredSceneStore],
        selected: frozenset[str] | None = None,
        *,
        merge_on_read: bool = False,
        merge_layout: MergeLayoutWindow | None = None,
        layout_product: Product | None = None,
    ) -> None:
        """Bind the collection to a Stack and its committed scene stores.

        Parameters
        ----------
        stack : Stack
            The Stack whose committed ``coreg/<date>`` products back the
            view.
        stores : dict[str, CoregisteredSceneStore]
            Every committed scene product, keyed on its canonical date id.
        selected : frozenset[str], optional
            The date subset this view exposes. ``None`` selects all
            committed dates. Build constrained views with :meth:`sel`,
            never by mutating the collection in place.
        merge_on_read : bool, optional
            When True, scene layer reads merge selected bursts through the
            shared mosaicking kernel instead of mosaicking in place.
        merge_layout : MergeLayoutWindow, optional
            Persisted merge layout bound to this view. ``None`` leaves
            merge-on-read views lazy; a read without a committed layout is
            rejected rather than assembled from burst windows.
        layout_product : Product, optional
            The ``merge_layout`` product whose ``write_id`` is re-checked
            on each merged read.

        """
        super().__init__(stores)
        self._stack = stack
        self._selected = selected
        self._merge_on_read = merge_on_read
        self._merge_layout = merge_layout
        self._layout_product = layout_product

    # ---- date axis -----------------------------------------------------

    @property
    def dates(self) -> list[str]:
        """Return this view's selected date ids in sorted order."""
        committed = dict.keys(self)
        if self._selected is None:
            return sorted(committed)
        return sorted(self._selected & committed)

    @property
    def acquisitions(self) -> Acquisition:
        """Return this view's date axis as an :class:`Acquisition` index."""
        from faninsar.core.acquisition import Acquisition

        return Acquisition(pd.to_datetime(self.dates, format="%Y%m%d"))

    @property
    def reference(self) -> str | None:
        """Return the Reference acquisition date, when it is committed."""
        return self._stack.reference

    # ---- Reference-bound geometry sugar (PROPOSAL-0066) ------------------

    def _reference_layers(self) -> SceneLayers:
        """Return the Reference scene's bound layer view.

        Geometry and LOS layers are Reference-bound, so they resolve only on
        the Reference scene. A Stack with no committed Reference has no
        radar-geometry frame to read.
        """
        from faninsar.processing.errors import reject_invalid_state

        reference = self._stack.reference_scene
        if reference is None:
            reject_invalid_state(
                "the Stack Reference scene is not committed; run "
                "coregister_scenes before reading geometry layers"
            )
        return self._bound(reference).layers

    def _los_layer(self, name: str) -> ProductLayer:
        """Return one cached virtual LOS unit-vector layer.

        The cache is a plain session-scoped dict keyed on the
        ``(incidence_write_id, azimuth_write_id)`` pair of the Reference
        scene item: a re-materialized Reference invalidates the entry, and a
        fresh ``stack.scenes`` collection drops it entirely.
        """
        from faninsar.io.storage.product import open_product

        layers = self._reference_layers()
        # The incidence/azimuth payloads are committed on the Reference item
        # under the scene write_id, so that id is the cache key for every
        # derived LOS layer. A re-materialized Reference invalidates it.
        product = open_product(layers._store.root)
        key = (product.write_id, product.write_id, name)
        cache = self._los_cache
        if key not in cache:
            cache[key] = getattr(layers, name)
        return cache[key]

    @property
    def _los_cache(self) -> dict[tuple[str, str, str], ProductLayer]:
        """Return this collection's session-scoped LOS layer cache."""
        cache = getattr(self, "_los_cache_dict", None)
        if cache is None:
            cache = {}
            self._los_cache_dict = cache
        return cache

    @property
    def latitude(self) -> ProductLayer:
        """Return the Reference geodetic latitude field (degrees)."""
        return self._reference_layers().latitude

    @property
    def longitude(self) -> ProductLayer:
        """Return the Reference geodetic longitude field (degrees)."""
        return self._reference_layers().longitude

    @property
    def height(self) -> ProductLayer:
        """Return the Reference ellipsoidal height field (metres)."""
        return self._reference_layers().height

    @property
    def incidence(self) -> ProductLayer:
        """Return the Reference incidence-angle field (degrees)."""
        return self._reference_layers().incidence

    @property
    def azimuth(self) -> ProductLayer:
        """Return the Reference look-azimuth field (degrees)."""
        return self._reference_layers().azimuth

    @property
    def geom_flags(self) -> ProductLayer:
        """Return the Reference layover/shadow bitmask layer."""
        return self._reference_layers().geom_flags

    @property
    def los_east(self) -> ProductLayer:
        """Return the virtual east LOS unit-vector component."""
        return self._los_layer("los_east")

    @property
    def los_north(self) -> ProductLayer:
        """Return the virtual north LOS unit-vector component."""
        return self._los_layer("los_north")

    @property
    def los_up(self) -> ProductLayer:
        """Return the virtual up LOS unit-vector component."""
        return self._los_layer("los_up")

    def merge_bursts(self) -> StackSceneCollection:
        """Return a lazy per-date merged view of burst scene units.

        Each acquisition is merged independently through the shared
        :func:`~faninsar.processing.mosaicking.merge_bursts` kernel at read
        time. Persisted burst scenes are not rewritten and dates are never
        merged together.

        Returns
        -------
        StackSceneCollection
            The same collection type, with merge-on-read layer views.

        """
        layout, layout_product = _opened_merge_layout(self._stack)
        return StackSceneCollection(
            self._stack,
            dict(self),
            selected=self._selected,
            merge_on_read=True,
            merge_layout=layout,
            layout_product=layout_product,
        )

    # ---- read views ----------------------------------------------------

    def _selected_stores(self) -> dict[str, CoregisteredSceneStore]:
        """Return committed stores of this view's selected dates."""
        return {date: self[date] for date in self.dates if date in self}

    def sel(
        self,
        *,
        dates: Acquisition | slice | list[str] | tuple[str, ...] | str | None = None,
        track: str | None = None,
        direction: str | None = None,
        pol: str | list[str] | tuple[str, ...] | None = None,
        burst: int | list[int] | tuple[int, ...] | None = None,
    ) -> StackSceneCollection | SceneLayers:
        """Return a view narrowed to a date subset.

        Parameters
        ----------
        dates : Acquisition, slice, list of str, or str, optional
            A :class:`Acquisition` index, a date-``str`` list, or a
            ``slice`` of date bounds selects a subset and returns a new
            :class:`StackSceneCollection`; a scalar date string or a
            record-shaped ``Acquisition`` demotes the collection to that
            scene's :class:`SceneLayers` — the same object
            ``collection[date].layers`` returns. Mirrors the xarray
            ``sel`` scalar-vs-list demotion rule used by
            :meth:`StackInterferogramCollection.sel`.
        track : str, optional
            Track label (``"88A"``, or a bare relative-orbit number
            selecting both direction letters). Keeps the committed dates
            that lie on the selected track; cross-track dates drop out so
            the view stays single-track.
        direction : str, optional
            ``"ASC"``/``"DESC"`` (or ``"ASCENDING"``/``"DESCENDING"``);
            narrows the date axis the same way ``track=`` does.
        pol : str or sequence of str, optional
            Polarization channel(s); keeps dates whose acquisitions carry
            every requested channel.
        burst : int or sequence of int, optional
            Absolute burst id(s); keeps dates whose acquisitions cover
            every requested id. Fails closed when the Stack's per-date
            burst metadata is unavailable.

        Returns
        -------
        StackSceneCollection or SceneLayers
            A new collection for multi-date selections, or the scene's
            layer view for a scalar date selection.

        Raises
        ------
        TypeError
            If ``dates`` is not a supported selection type.
        KeyError
            If ``dates`` names an uncommitted scene product.
        ValueError
            If the selection is empty or a track-axis selector cannot be
            resolved.

        """
        from faninsar.core.acquisition import Acquisition

        track_subset = self._track_selected_dates(
            track=track, direction=direction, pol=pol, burst=burst
        )
        if dates is None:
            if track_subset is None:
                message = (
                    "sel requires at least one of dates=, track=, direction=, "
                    "pol=, or burst="
                )
                raise ValueError(message)
            return StackSceneCollection(
                self._stack,
                dict(self),
                selected=frozenset(track_subset),
                merge_on_read=self._merge_on_read,
                merge_layout=self._merge_layout,
                layout_product=self._layout_product,
            )

        scalar_id: str | None = None
        if isinstance(dates, str):
            scalar_id = dates
        elif isinstance(dates, Acquisition) and hasattr(dates, "_physical_key"):
            scalar_id = str(dates.strftime("%Y%m%d")[0])
        if scalar_id is not None:
            if scalar_id not in self:
                raise KeyError(dates)
            if track_subset is not None and scalar_id not in track_subset:
                message = (
                    f"date {scalar_id!r} is not part of the track-axis "
                    "selection; sel(track=/direction=/pol=/burst=) excludes it"
                )
                raise KeyError(message)
            return self[scalar_id].layers

        if isinstance(dates, slice):
            lo = self._date_bound(dates.start, min)
            hi = self._date_bound(dates.stop, max)
            subset = {date for date in self.dates if lo <= date <= hi}
        elif isinstance(dates, Acquisition):
            subset = set(
                self._normalize_dates(item.strftime("%Y%m%d") for item in dates)
            )
        elif isinstance(dates, Iterable) and not isinstance(dates, str):
            subset = set(self._normalize_dates(dates))
        else:
            message = (
                "dates must be an Acquisition, a date str, a list of date "
                f"strings, or a slice; got {type(dates).__name__}"
            )
            raise TypeError(message)
        if track_subset is not None:
            subset &= track_subset
        unknown = sorted(subset - set(dict.keys(self)))
        if unknown:
            message = f"scene products not committed for dates: {unknown!r}"
            raise KeyError(message)
        if not subset:
            message = "sel produced an empty date selection"
            raise ValueError(message)
        return StackSceneCollection(
            self._stack,
            dict(self),
            selected=frozenset(subset),
            merge_on_read=self._merge_on_read,
            merge_layout=self._merge_layout,
            layout_product=self._layout_product,
        )

    def _track_selected_dates(
        self,
        *,
        track: str | None,
        direction: str | None,
        pol: str | list[str] | tuple[str, ...] | None,
        burst: int | list[int] | tuple[int, ...] | None,
    ) -> set[str] | None:
        """Return committed dates matching the track-axis selectors.

        ``None`` means no selector was given; a set means the committed
        dates surviving every selector (empty allowed — the caller decides
        whether emptiness is an error).
        """
        if all(value is None for value in (track, direction, pol, burst)):
            return None
        normalized_track = self._stack._normalize_track_selector(track)
        selected = self._stack._catalog_dates_for_track(
            track=normalized_track, direction=direction, pol=pol, burst=burst
        )
        return set(selected) & set(dict.keys(self))

    def _date_bound(self, bound: object, default: Any) -> str:
        """Normalize one slice bound to a canonical ``%Y%m%d`` date id."""
        if bound is None:
            return default(self.dates, default="")
        if isinstance(bound, str):
            try:
                return pd.to_datetime(bound).strftime("%Y%m%d")
            except ValueError:
                return bound
        try:
            return pd.to_datetime(bound).strftime("%Y%m%d")
        except (KeyError, TypeError, ValueError) as error:
            message = f"unsupported date bound {bound!r}"
            raise TypeError(message) from error

    @staticmethod
    def _normalize_dates(values: Iterable[str]) -> list[str]:
        """Parse date-like values into canonical ``%Y%m%d`` ids."""
        normalized: list[str] = []
        for value in values:
            try:
                normalized.append(pd.to_datetime(value).strftime("%Y%m%d"))
            except (TypeError, ValueError):
                normalized.append(str(value))
        return normalized

    # ---- mapping --------------------------------------------------------

    def _bound(self, store: CoregisteredSceneStore) -> CoregisteredSceneStore:
        """Return the store re-bound to this Stack's session."""
        from dataclasses import replace

        from faninsar.io.storage.product import open_product

        reference = self._stack.reference_scene
        upstream = (open_product(reference.root),) if reference is not None else ()
        bound = replace(store, _session=self._stack, _upstream=upstream)
        if not self._merge_on_read:
            return bound
        return _MergeBoundSceneStore(
            bound,
            self._merge_layout,
            self._stack,
            upstream,
            layout_product=self._layout_product,
        )

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
                date_id = str(key.strftime("%Y%m%d")[0])
                return self._bound(dict.__getitem__(self, date_id))
            return {
                date: self._bound(dict.__getitem__(self, date)).layers
                for date in self.dates
                if date in {item.strftime("%Y%m%d") for item in key} and date in self
            }
        if isinstance(key, Iterable) and not isinstance(key, str):
            return {
                date: self._bound(dict.__getitem__(self, date))
                for date in self._normalize_dates(key)
            }
        return self._bound(dict.__getitem__(self, str(key)))

    def __iter__(self) -> Iterator[str]:
        """Iterate over this view's selected date ids."""
        return iter(self.dates)

    def keys(self) -> list[str]:
        """Return this view's selected date ids."""
        return self.dates

    def values(self) -> list[CoregisteredSceneStore]:
        """Return the committed stores of this view's selected dates."""
        return [self[date] for date in self.dates]

    def items(self) -> list[tuple[str, CoregisteredSceneStore]]:
        """Return ``(date, store)`` pairs of this view's selected dates."""
        return [(date, self[date]) for date in self.dates]

    def get(
        self, key: str, default: CoregisteredSceneStore | None = None
    ) -> CoregisteredSceneStore | None:
        """Return the scene store of *key*, honouring the view's selection."""
        if self._selected is not None and str(key) not in self._selected:
            return default
        store = dict.get(self, str(key))
        if store is None:
            return default
        return self._bound(store)

    # ---- display --------------------------------------------------------

    def _frame(self) -> pd.DataFrame:
        """Return this view's committed scene metadata as a DataFrame."""
        rows = [
            {
                "date": date,
                "role": store.role,
                "units": len(store.units),
                "shape": "x".join(str(size) for size in store.grid_shape),
            }
            for date, store in (
                (date, self[date]) for date in self.dates if date in self
            )
        ]
        if not rows:
            return pd.DataFrame(columns=["date", "role", "units", "shape"])
        return pd.DataFrame(rows).set_index("date")

    def __str__(self) -> str:
        """Return a one-line summary of this view."""
        return (
            f"StackSceneCollection({len(self.dates)} scenes, "
            f"reference={self.reference!r})"
        )

    def __repr__(self) -> str:
        """Return a Pairs-like tabular representation of this view."""
        header = (
            f"StackSceneCollection({len(self.dates)} scenes, "
            f"reference={self.reference!r}, "
            f"root={self._stack.config.work_dir / 'coreg'!r})"
        )
        return f"{header}\n{self._frame().__repr__()}"
