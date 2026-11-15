"""Lazy, typed layer views of committed Stack interferograms."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.core.sar_tools import Looks, coerce_looks
from faninsar.io.storage.product import Product, open_product
from faninsar.logging import reject, setup_logger
from faninsar.network.products import ASSET_KIND_GEOCODE_KERNELS, AssetKind
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from faninsar.processing.geocoding.geo_resample import GeocodeKind
    from faninsar.processing.geocoding.geocoder import GeocodedRaster
    from faninsar.processing.geocoding.product import GeocodeSession
    from faninsar.processing.geometry.grids import GridSpec
    from faninsar.processing.mosaicking import MergeLayoutWindow

__all__ = ["InterferogramLayers", "MergedBurstLayers", "ProductLayer"]

#: Half-open raster window ``(r0, r1, c0, c1)`` in row/column coordinates.
Window = tuple[int, int, int, int]

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

#: Virtual LOS unit-vector layers on a Reference scene product. They derive
#: from the stored ``incidence``/``azimuth`` geometry assets at read time and
#: persist no payload (PROPOSAL-0066).
_SCENE_LOS_LAYERS = ("los_east", "los_north", "los_up")

#: Committed Reference geometry assets served directly off the scene product
#: item.json. Unlike the per-unit ``slc`` mosaics these are single-frame
#: rasters, so they read through the asset branch — but they must not fall
#: through to the scene-mosaic branch when absent.
_SCENE_GEOMETRY_ASSETS = (
    "latitude",
    "longitude",
    "height",
    "incidence",
    "azimuth",
    "geom_flags",
)

_REFERENCE_SUBTRACT_KINDS = frozenset(
    {AssetKind.UNWRAPPED_PHASE, AssetKind.DISPLACEMENT}
)

#: Displayed width/height ratio at or above which ``colorbar_location="auto"``
#: places the colorbar below the image instead of to its right.
_AUTO_COLORBAR_ASPECT_THRESHOLD = 2.0


def _as_window(window: object) -> Window:
    """Validate a half-open ``(r0, r1, c0, c1)`` window."""
    if (
        not isinstance(window, tuple)
        or len(window) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in window
        )
    ):
        message = (
            f"window must be a half-open (r0, r1, c0, c1) integer tuple; got {window!r}"
        )
        raise TypeError(message)
    row0, row1, col0, col1 = window
    if row0 > row1 or col0 > col1:
        message = f"window bounds are not half-open: {window!r}"
        raise ValueError(message)
    return row0, row1, col0, col1


def _window_shape(window: Window) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a half-open window."""
    row0, row1, col0, col1 = window
    return row1 - row0, col1 - col0


def _overlap_window(first: Window, second: Window) -> Window | None:
    """Return the overlapping half-open window, or ``None``."""
    row0 = max(first[0], second[0])
    row1 = min(first[1], second[1])
    col0 = max(first[2], second[2])
    col1 = min(first[3], second[3])
    if row0 >= row1 or col0 >= col1:
        return None
    return row0, row1, col0, col1


def _tight_support_window(mask: np.ndarray, origin: Window) -> Window | None:
    """Return the grid window of True samples, preserving holes and gaps.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean support sampled on ``origin``.
    origin : tuple of int
        Grid window that *mask* occupies.

    Returns
    -------
    tuple of int or None
        Tight bounding window in grid coordinates, or ``None`` when empty.

    """
    if mask.shape[0] == 0 or mask.shape[1] == 0 or not np.any(mask):
        return None
    rows, cols = np.nonzero(mask)
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1
    return (
        origin[0] + row0,
        origin[0] + row1,
        origin[2] + col0,
        origin[2] + col1,
    )


def _unit_grid_box(unit: Any) -> Window:
    """Return a scene unit's footprint on the shared grid."""
    row0 = int(unit.row_origin)
    col0 = int(unit.col_origin)
    return row0, row0 + int(unit.shape[0]), col0, col0 + int(unit.shape[1])


def _phase_convention(value: object) -> str:
    """Return a non-empty phase-convention label for a BurstWindow."""
    if isinstance(value, str) and value:
        return value
    return "primary_minus_secondary"


def _scene_burst_id(unit: Any) -> str:
    """Return the persisted physical burst id, never a local unit tag."""
    burst_id = unit.burst_id
    if not isinstance(burst_id, str) or not burst_id:
        reject_invalid_state(
            f"scene unit {unit.tag!r} requires faninsar:burst_id; "
            "layout source_ids are physical identities, not local tags"
        )
    return burst_id


def _ifg_burst_id(product: Product) -> str:
    """Return the persisted IFG burst id, never a directory basename fallback."""
    burst_id = product.properties.get("faninsar:burst_id")
    if not isinstance(burst_id, str) or not burst_id:
        reject_invalid_state(f"burst IFG {product.path} requires faninsar:burst_id")
    return burst_id


def _unwrap_payload(payload: object) -> np.ndarray:
    """Return the array from a store payload that may be ``(array, unit)``."""
    if isinstance(payload, tuple):
        payload = payload[0]
    return np.asarray(payload)


def _crop_committed_layout(layout: Any, grid_window: Window) -> Any:
    """Crop a committed layout to *grid_window* without reassembling owners.

    ``build_merge_layout`` is an assembler only and does not infer physical
    Geo midpoints from ROI boxes. Collection reads must open a published
    ``merge_layout`` and slice it; they must not rebuild owner labels from
    burst footprints.
    """
    from faninsar.processing.mosaicking import LayoutSourceWindow, MergeLayoutWindow

    if not isinstance(layout, MergeLayoutWindow):
        reject_invalid_state(
            "merged layer read requires a committed merge_layout at "
            "work_dir/layout/<reference>; owner labels are not assembled "
            "from burst bounding boxes"
        )
    layout_window = _as_window(layout.window)
    overlap = _overlap_window(layout_window, grid_window)
    if overlap is None:
        reject_invalid_state(
            f"read window {grid_window!r} does not intersect merge layout "
            f"{layout_window!r}"
        )
    if overlap == layout_window:
        return layout
    row0 = overlap[0] - layout_window[0]
    row1 = overlap[1] - layout_window[0]
    col0 = overlap[2] - layout_window[2]
    col1 = overlap[3] - layout_window[2]
    sources: list[Any] = []
    for source in layout.source_windows:
        source_window = _as_window(source.window)
        source_overlap = _overlap_window(source_window, overlap)
        if source_overlap is None:
            continue
        src_row0 = source_overlap[0] - source_window[0]
        src_row1 = source_overlap[1] - source_window[0]
        src_col0 = source_overlap[2] - source_window[2]
        src_col1 = source_overlap[3] - source_window[2]
        sources.append(
            LayoutSourceWindow(
                burst_id=source.burst_id,
                window=source_overlap,
                scientific_support=np.asarray(source.scientific_support)[
                    src_row0:src_row1, src_col0:src_col1
                ],
                swath=source.swath,
            )
        )
    return MergeLayoutWindow(
        scope=layout.scope,
        window=overlap,
        source_ids=tuple(layout.source_ids),
        owner=np.asarray(layout.owner, dtype=np.uint16)[row0:row1, col0:col1],
        scientific_support=np.asarray(layout.scientific_support, dtype=bool)[
            row0:row1, col0:col1
        ],
        seams=tuple(layout.seams),
        source_windows=tuple(sources),
    )


def _layer_placement_properties(layer: ProductLayer) -> dict[str, Any]:
    """Return IFG placement metadata, falling back to an upstream IFG product.

    Unwrap siblings do not copy ``row_origin``/``look_origin``. Those fields
    live on the parent IFG and combine as in
    :func:`faninsar.io.storage.ifg_store.ifg_sample_placement`.
    """
    properties = dict(layer.source.properties)
    if "faninsar:row_origin" in properties or "faninsar:look_origin" in properties:
        return properties
    for parent in layer._upstream:
        parent_props = getattr(parent, "properties", None)
        if isinstance(parent_props, dict) and (
            "faninsar:row_origin" in parent_props
            or "faninsar:look_origin" in parent_props
        ):
            return dict(parent_props)
    return properties


def _layer_native_origin(layer: ProductLayer) -> tuple[int, int]:
    """Return the native index of layer pixel ``(0, 0)``'s first sample."""
    properties = _layer_placement_properties(layer)
    looks = coerce_looks(layer.multilook)
    raw_range_looks = properties.get("faninsar:range_looks")
    raw_azimuth_looks = properties.get("faninsar:azimuth_looks")
    if raw_range_looks is not None and raw_azimuth_looks is not None:
        looks = Looks(int(raw_range_looks), int(raw_azimuth_looks))
    raw_look_origin = properties.get("faninsar:look_origin", (0, 0))
    look_origin = (int(raw_look_origin[0]), int(raw_look_origin[1]))
    row_origin = int(properties.get("faninsar:row_origin", layer.origin[0]))
    col_origin = int(properties.get("faninsar:col_origin", layer.origin[1]))
    return (
        row_origin + look_origin[0] * looks.azimuth,
        col_origin + look_origin[1] * looks.range,
    )


def _resolve_plot_ref_points(
    layer: ProductLayer, ref_points: object
) -> tuple[object, str]:
    """Resolve inherit / None / explicit points for one plot call."""
    from faninsar.data.reference import is_inherit, resolve_ref_points_argument

    session = layer._session
    if is_inherit(ref_points) and session is None:
        return None, "common"
    inherited_points = (
        getattr(session, "ref_points", None) if session is not None else None
    )
    inherited_policy = (
        getattr(session, "ref_policy", "common") if session is not None else "common"
    )
    return resolve_ref_points_argument(ref_points, inherited_points, inherited_policy)


def _map_layer_reference_points(
    layer: ProductLayer, points: object
) -> tuple[object, str]:
    """Map points onto this layer's scientific grid."""
    from faninsar.data.query.points import Points
    from faninsar.data.reference import map_geographic_points

    if not isinstance(points, Points):
        message = "ref_points must be a faninsar.data.query.Points instance"
        reject(message, TypeError, logger=logger)
    properties = layer.source.properties
    domain = properties.get("faninsar:domain")
    if domain is None:
        domain = "geo" if layer.crs is not None else "radar"
    if domain == "geo":
        transform = properties.get("proj:transform")
        if transform is None:
            message = "geographic reference-point mapping requires a layer transform"
            reject(message, ValueError, logger=logger)
        return (
            map_geographic_points(
                points,
                transform=transform,
                shape=layer.shape,
                crs=layer.crs or properties.get("proj:code"),
            ),
            "geo",
        )
    mapper = getattr(layer._session, "map_reference_points_to_radar", None)
    if layer._session is None or mapper is None:
        message = (
            "radar reference-point mapping requires geometry context; "
            "use a Stack-bound layer or pass geographic points on a "
            "geographic layer"
        )
        reject(message, ValueError, logger=logger)
    looks = coerce_looks(layer.multilook)
    raw_range_looks = properties.get("faninsar:range_looks")
    raw_azimuth_looks = properties.get("faninsar:azimuth_looks")
    if raw_range_looks is not None and raw_azimuth_looks is not None:
        looks = Looks(int(raw_range_looks), int(raw_azimuth_looks))
    return (
        mapper(
            points,
            origin=_layer_native_origin(layer),
            looks=looks,
            shape=layer.shape,
        ),
        "radar",
    )


def _reference_display_xy(
    layer: ProductLayer,
    points: object,
    mapping: object,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return marker coordinates in the plotted extent's coordinate system."""
    if domain == "geo":
        from faninsar.data.reference import coordinates_in_crs

        crs = layer.crs or layer.source.properties.get("proj:code")
        if crs is not None and getattr(points, "crs", None) is not None:
            return coordinates_in_crs(points, crs)
        return (
            np.asarray(points.x, dtype=np.float64),
            np.asarray(points.y, dtype=np.float64),
        )
    return (
        np.asarray(mapping.col_frac, dtype=np.float64),
        np.asarray(mapping.row_frac, dtype=np.float64),
    )


def _apply_plot_reference(
    layer: ProductLayer,
    display_values: np.ndarray,
    ref_points: object,
    *,
    subtract: bool,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Subtract a reference mean from a display copy of the full layer."""
    from faninsar.data.reference import (
        mean_offsets,
        sample_at_pixels,
        select_valid_subsets,
        subtract_offsets,
    )

    points, policy = _resolve_plot_ref_points(layer, ref_points)
    if points is None:
        return display_values, None
    mapping, domain = _map_layer_reference_points(layer, points)
    raw = np.asarray(layer.read())
    if np.iscomplexobj(raw):
        raw = np.asarray(raw.real)
    sampled = sample_at_pixels(raw, mapping, layer._support_mask())
    valid = np.isfinite(sampled)
    xs, ys = _reference_display_xy(layer, points, mapping, domain)
    markers = (xs, ys, valid)
    if not subtract:
        return display_values, markers
    selected = select_valid_subsets({"layer": valid}, policy)
    offsets = mean_offsets({"layer": sampled}, selected)
    referenced = subtract_offsets({"layer": display_values}, offsets)
    return referenced["layer"], markers


def _scatter_reference_markers(
    ax: Any,
    markers: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    extent: tuple[float, float, float, float],
) -> None:
    """Draw requested points that fall inside the displayed extent."""
    if markers is None:
        return
    xs, ys, valid = markers
    left, right, bottom, top = extent
    x_lo, x_hi = (left, right) if left <= right else (right, left)
    y_lo, y_hi = (bottom, top) if bottom <= top else (top, bottom)
    inside = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
    shown_valid = valid & inside
    shown_invalid = (~valid) & inside
    if np.any(shown_valid):
        ax.scatter(
            xs[shown_valid],
            ys[shown_valid],
            marker="+",
            c="k",
            s=40,
            zorder=5,
        )
    if np.any(shown_invalid):
        ax.scatter(
            xs[shown_invalid],
            ys[shown_invalid],
            marker="x",
            c="0.5",
            s=30,
            zorder=5,
        )
    n_outside = int(np.count_nonzero(~inside))
    if n_outside:
        logger.info("%s reference points lie outside the displayed extent", n_outside)


def _shift_affine(
    transform: object, row0: int, col0: int
) -> tuple[float, float, float, float, float, float] | None:
    """Return a 6-parameter affine whose origin is the window's (0, 0)."""
    try:
        values = tuple(float(item) for item in transform)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 6:
        return None
    a, b, c, d, e, f = values
    return (
        a,
        b,
        c + a * col0 + b * row0,
        d,
        e,
        f + d * col0 + e * row0,
    )


def _displayed_width_height_ratio(ax: Any) -> float | None:
    """Return the displayed data-box width/height ratio, or None if undefined."""
    aspect = ax.get_aspect()
    if aspect == "auto":
        return None
    aspect_value = 1.0 if aspect == "equal" else float(aspect)
    data_ratio = abs(ax.get_data_ratio())
    if not data_ratio or aspect_value <= 0:
        return None
    ratio = 1.0 / (aspect_value * data_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    return float(ratio)


def _displayed_fills_cell_width(ax: Any) -> bool | None:
    """Whether the displayed data box reaches the axes cell's horizontal edges."""
    ratio = _displayed_width_height_ratio(ax)
    if ratio is None:
        return None
    cell = ax.get_position(original=True)
    fig = ax.get_figure(root=False)
    fig_w, fig_h = fig.get_size_inches()
    if cell.width <= 0 or cell.height <= 0 or fig_w <= 0:
        return None
    cell_ratio = (cell.width * fig_w) / (cell.height * fig_h)
    return bool(ratio >= cell_ratio)


def _resolve_colorbar_location(ax: Any, location: str) -> str:
    """Resolve ``"auto"`` to a concrete colorbar side from the data box."""
    if location != "auto":
        return location
    engine = ax.get_figure(root=False).get_layout_engine()
    if engine is not None and not engine.colorbar_gridspec:
        # constrained_layout repositions horizontal colorbars relative to the
        # full axes cell, so bottom/top matching is only approximate there.
        return "right"
    ratio = _displayed_width_height_ratio(ax)
    if ratio is not None and ratio >= _AUTO_COLORBAR_ASPECT_THRESHOLD:
        return "bottom"
    return "right"


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
    full_radar_shape, origin : tuple[int, int]
        Full radar frame and the layer's ``(row, col)`` sampling origin.
    multilook : Looks or tuple[int, int]
        Range-first ``(range, azimuth)`` look factors, normalized to
        :class:`~faninsar.core.sar_tools.Looks`.
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
    multilook: Looks | tuple[int, int] = field(default_factory=lambda: Looks(1, 1))
    origin: tuple[int, int] = (0, 0)
    crs: str | None = None
    extent: tuple[int, int] | None = None
    _scene_selection: frozenset[str] | None = field(
        default=None, repr=False, compare=False
    )
    _upstream: tuple[Product, ...] = field(default=(), repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)
    _merge_on_read: bool = field(default=False, repr=False, compare=False)
    _merge_layout: MergeLayoutWindow | None = field(
        default=None, repr=False, compare=False
    )
    _layout_product: Product | None = field(default=None, repr=False, compare=False)
    _burst_products: tuple[Product, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize the look factors to range-first :class:`Looks`."""
        object.__setattr__(self, "multilook", coerce_looks(self.multilook))

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
        if self._product.kind == "scene" and self.name in (
            SCENE_LAYER_NAMES + _SCENE_LOS_LAYERS
        ):
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

    def to_cog(
        self,
        path: str | Path | None = None,
        *,
        mask: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """Export this layer as one Cloud Optimized GeoTIFF.

        Format conversion only: the samples, the grid, and the coordinate
        domain are unchanged. Use :meth:`geocode` when a different frame is
        wanted.

        Values come from :attr:`read`, so the exported file holds exactly what
        this handle yields; the view selection still enters the export identity,
        so two selections of one source never reuse each other's file.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination. A ``.tif``/``.tiff`` path names the file; anything
            else is treated as a directory. ``None`` writes to
            ``<product>/cog/<generated name>``.
        mask : bool, optional
            Embed validity as the file's internal mask. The plane comes from
            the persisted support, not from the sample values, so a supported
            ``0`` stays valid and an unsupported finite sample stays masked.
        overwrite : bool, optional
            Replace an existing destination written by a different export
            request.

        Returns
        -------
        pathlib.Path
            The written or reused file.

        Raises
        ------
        InvalidProcessingStateError
            If the owning product was replaced.
        RuntimeError
            If the export failed; the reason names the destination.

        Examples
        --------
        >>> ifg = stack.interferograms["20240101_20240113"]
        >>> path = ifg.unwrapped_phase.to_cog()  # doctest: +SKIP

        """
        from faninsar.io.cog import CogExportItem, export_cog, request_from_product

        self.validate_source()
        product = self._product
        contributing: list[tuple[str, str]] = [
            (f"upstream_{index}", upstream.write_id)
            for index, upstream in enumerate(self._upstream)
        ]
        if self._layout_product is not None:
            contributing.append(("layout", self._layout_product.write_id))
        for index, burst in enumerate(self._burst_products):
            contributing.append((f"burst_{index}", burst.write_id))
        selection: list[tuple[str, str]] = []
        if self._scene_selection is not None:
            selection.extend(("unit", unit) for unit in sorted(self._scene_selection))
        selection.append(("merge_on_read", "true" if self._merge_on_read else "false"))
        selection.append(("name", self.name))

        request = request_from_product(
            product=product,
            asset_key=self.name,
            composition="merge-on-read" if self._merge_on_read else "identity",
            selection=selection,
            contributing=contributing,
            origin=self.origin,
            looks=(self.multilook.range, self.multilook.azimuth),
        )

        values = self.read()
        grid = None
        if self.crs is not None:
            # A geographic layer carries its frame on the product: proj:code,
            # proj:transform, and the declared shape. Rebuilding the grid from
            # those is exact, whereas opening one payload would only work for
            # a layer that happens to have a stored raster.
            from affine import Affine

            from faninsar.data.datasets.geogrid import GeoGrid

            properties = self._product.properties
            raw_transform = properties.get("proj:transform")
            raw_shape = properties.get("faninsar:shape")
            if raw_transform is None or raw_shape is None:
                message = (
                    f"geo layer {self.name!r} cannot be exported: its product "
                    "records no proj:transform or faninsar:shape"
                )
                raise RuntimeError(message)
            grid = GeoGrid(
                shape=(int(raw_shape[0]), int(raw_shape[1])),
                transform=Affine(*[float(value) for value in raw_transform]),
                crs=self.crs,
            )

        item: CogExportItem = export_cog(
            request,
            values,
            path=path,
            grid=grid,
            mask=mask,
            overwrite=overwrite,
        )
        if item.status == "failed" or item.path is None:
            message = f"COG export of layer {self.name!r} failed: {item.reason}"
            raise RuntimeError(message)
        return item.path

    def validate_source(self) -> None:
        """Reject a handle whose owning, layout, or upstream product was replaced."""
        recorded_products = (self._product, *self._upstream, *self._burst_products)
        if self._layout_product is not None:
            recorded_products = (*recorded_products, self._layout_product)
        for recorded in recorded_products:
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

    def _grid_box(self) -> Window:
        """Return this layer's footprint on the source grid."""
        row0, col0 = self.origin
        rows, cols = self.shape
        return row0, row0 + rows, col0, col0 + cols

    def _localize_window(self, window: Window | None) -> Window | None:
        """Translate a source-grid window into this layer's local array."""
        if window is None:
            return None
        overlap = _overlap_window(_as_window(window), self._grid_box())
        if overlap is None:
            reject_invalid_state(
                f"layer {self.name!r} window {window!r} does not overlap "
                f"origin={self.origin} shape={self.shape}"
            )
        row0, col0 = self.origin
        return (
            overlap[0] - row0,
            overlap[1] - row0,
            overlap[2] - col0,
            overlap[3] - col0,
        )

    def _grid_from_local(self, window: Window | None) -> Window:
        """Return the source-grid window of a local array window."""
        if window is None:
            return self._grid_box()
        row0, col0 = self.origin
        return (
            window[0] + row0,
            window[1] + row0,
            window[2] + col0,
            window[3] + col0,
        )

    def _read_product_array(self, name: str, window: Window | None) -> np.ndarray:
        """Read one product asset, forwarding a local window to storage."""
        if window is None:
            return self._product.read(name)
        return self._product.read(name, window=window)

    def _read_virtual(self, name: str, window: Window | None = None) -> np.ndarray:
        """Derive one virtual IFG layer through the owning artifact store."""
        from faninsar.io.storage.ifg_store import (
            COMPLEX_ION_KIND,
            IFG_KIND,
            InterferogramArtifactStore,
        )

        if self._product.kind == IFG_KIND:
            return InterferogramArtifactStore.open(self._product.path).read_layer(
                name, window=window
            )
        if self._product.kind != COMPLEX_ION_KIND:
            reject_invalid_state(
                f"layer {self.name!r} is virtual and requires an IFG or "
                "complex_ion product source"
            )
        complex_ifg = self._read_product_array("complex", window)
        valid_mask = (
            self._read_product_array("valid_mask", window)
            if "valid_mask" in self._product.assets
            else None
        )
        if name == "wrapped_phase":
            derived = np.asarray(np.angle(complex_ifg))
        else:
            derived = np.asarray(np.abs(complex_ifg))
        if valid_mask is not None:
            derived[~np.asarray(valid_mask, dtype=bool)] = np.nan
        return derived

    def _read_scene_layer(self, name: str, window: Window | None = None) -> np.ndarray:
        """Read one scene layer through its store's mosaic reader.

        ``slc`` mosaics the selected unit payloads onto their union bounding
        box (or the common grid for a ``.full()`` layer); ``amplitude``,
        ``intensity``, and ``phase`` derive from that mosaic; ``valid_mask``
        uses persisted ``{tag}.valid`` assets when present.
        """
        from faninsar.io.storage.scene_store import (
            SCENE_KIND,
            CoregisteredSceneStore,
        )

        if self._product.kind != SCENE_KIND:
            reject_invalid_state(
                f"layer {self.name!r} is virtual and requires a scene product source"
            )
        store = CoregisteredSceneStore.open(self._product.path)
        if self._scene_selection is not None:
            store = store.select_units(self._scene_selection)
        full_local = (0, self.shape[0], 0, self.shape[1])
        if (window is None or window == full_local) and not self._merge_on_read:
            mosaic = store.mosaic_slc(full=self.extent is None)
        else:
            mosaic = self._compose_scene_window(store, window)
        if name == "slc":
            return mosaic
        if name == "amplitude":
            return np.abs(mosaic).astype(np.float32)
        if name == "intensity":
            return (np.abs(mosaic) ** 2).astype(np.float32)
        if name == "phase":
            return np.angle(mosaic).astype(np.float32)
        if name == "valid_mask":
            # Persisted support is authoritative. Deriving it from the mosaic
            # would report a finite-but-invalid sample as valid and a supported
            # ``0j`` as invalid, the opposite of what the product committed.
            # The windowed path already reads the committed planes; the full
            # path takes the same route instead of the mosaic shortcut.
            return self._compose_scene_valid_mask(store, window, mosaic)
        message = f"unknown scene layer: {name!r}"
        raise FileNotFoundError(message)

    def _compose_scene_window(
        self,
        store: Any,
        window: Window | None,
    ) -> np.ndarray:
        """Compose a local scene window, using the merge kernel when requested."""
        local = window if window is not None else (0, self.shape[0], 0, self.shape[1])
        if self._merge_on_read:
            return self._merge_scene_window(store, local)
        return self._mosaic_scene_window(store, local)

    def _mosaic_scene_window(self, store: Any, local: Window) -> np.ndarray:
        """Place intersecting unit windows onto one local frame."""
        from faninsar.io.storage.scene_store import valid_samples

        rows, cols = _window_shape(local)
        out = np.full((rows, cols), np.nan + 1j * np.nan, dtype=np.complex64)
        grid_window = self._grid_from_local(local)
        for unit in store.selected_units():
            overlap = _overlap_window(grid_window, _unit_grid_box(unit))
            if overlap is None:
                continue
            unit_window = (
                overlap[0] - unit.row_origin,
                overlap[1] - unit.row_origin,
                overlap[2] - unit.col_origin,
                overlap[3] - unit.col_origin,
            )
            payload = store.read(unit.tag, window=unit_window)
            if isinstance(payload, tuple):
                payload = payload[0]
            dest = out[
                overlap[0] - grid_window[0] : overlap[1] - grid_window[0],
                overlap[2] - grid_window[2] : overlap[3] - grid_window[2],
            ]
            invalid = ~valid_samples(dest)
            dest[invalid] = payload[invalid]
        return out

    def _compose_scene_valid_mask(
        self,
        store: Any,
        window: Window | None,
        mosaic: np.ndarray,
    ) -> np.ndarray:
        """Read per-unit valid masks, falling back to mosaic support."""
        from faninsar.io.storage.scene_store import valid_samples

        local = window if window is not None else (0, self.shape[0], 0, self.shape[1])
        rows, cols = _window_shape(local)
        out = np.zeros((rows, cols), dtype=np.bool_)
        grid_window = self._grid_from_local(local)
        read_valid = getattr(store, "read_valid_mask", None)
        filled = False
        for unit in store.selected_units():
            overlap = _overlap_window(grid_window, _unit_grid_box(unit))
            if overlap is None:
                continue
            unit_window = (
                overlap[0] - unit.row_origin,
                overlap[1] - unit.row_origin,
                overlap[2] - unit.col_origin,
                overlap[3] - unit.col_origin,
            )
            mask = None
            if read_valid is not None:
                mask = read_valid(unit.tag, window=unit_window)
            else:
                asset = f"{unit.tag}.valid"
                if asset in self._product.assets:
                    mask = self._read_product_array(asset, unit_window)
            if mask is None:
                continue
            dest = out[
                overlap[0] - grid_window[0] : overlap[1] - grid_window[0],
                overlap[2] - grid_window[2] : overlap[3] - grid_window[2],
            ]
            dest |= np.asarray(mask, dtype=np.bool_)
            filled = True
        if filled:
            return out
        return valid_samples(mosaic)

    def _merge_scene_window(self, store: Any, local: Window) -> np.ndarray:
        """Apply the shared merge kernel to bounded burst windows."""
        merged = self._merge_window(store, local, kind="scene")
        if merged is None:
            return np.zeros(_window_shape(local), dtype=np.complex64)
        return self._place_merged_array(
            np.asarray(merged.complex), merged.window, local, fill=np.complex64(0)
        )

    def _merge_window(
        self,
        store: Any,
        local: Window,
        *,
        kind: Literal["scene", "ifg"],
    ) -> Any:
        """Run :func:`merge_bursts` on one local window, or ``None`` if empty."""
        from faninsar.processing.mosaicking import merge_bursts

        layout = self._layout_for_window(local)
        products = self._burst_windows_for_read(store, local, kind=kind, layout=layout)
        if not products:
            return None
        return merge_bursts(products, layout=layout)

    def _burst_windows_for_read(
        self,
        store: Any,
        local: Window,
        *,
        kind: Literal["scene", "ifg"],
        layout: Any,
    ) -> list[Any]:
        """Build bounded BurstWindow inputs for one local read."""
        grid_window = self._grid_from_local(local)
        if kind == "ifg":
            windows = [
                self._ifg_burst_window(product, grid_window, layout)
                for product in self._burst_products
            ]
            return [item for item in windows if item is not None]
        windows = [
            self._scene_burst_window(store, unit, grid_window, layout)
            for unit in store.selected_units()
        ]
        return [item for item in windows if item is not None]

    def _ifg_burst_window(
        self, product: Product, grid_window: Window, layout: Any
    ) -> Any | None:
        """Return one IFG BurstWindow overlapping *grid_window*, or ``None``."""
        from faninsar.processing.mosaicking import BurstWindow

        properties = product.properties
        row0 = int(properties.get("faninsar:row_origin", 0))
        col0 = int(properties.get("faninsar:col_origin", 0))
        shape = properties["faninsar:shape"]
        box = (row0, row0 + int(shape[0]), col0, col0 + int(shape[1]))
        overlap = _overlap_window(grid_window, box)
        if overlap is None:
            return None
        unit_window = (
            overlap[0] - row0,
            overlap[1] - row0,
            overlap[2] - col0,
            overlap[3] - col0,
        )
        payload, mask = self._read_burst_payload(
            product, "complex", "valid_mask", unit_window
        )
        pair = properties.get("faninsar:pair", ())
        acquisitions = (
            tuple(str(item) for item in pair)
            if isinstance(pair, list | tuple) and pair
            else (str(product.path.parent.name),)
        )
        return BurstWindow(
            scope=layout.scope,
            burst_id=_ifg_burst_id(product),
            acquisitions=acquisitions,
            write_ids=(product.write_id,),
            phase_convention=_phase_convention(
                properties.get("faninsar:phase_convention")
            ),
            window=overlap,
            complex=payload,
            valid_mask=np.asarray(mask, dtype=np.bool_),
        )

    def _scene_burst_window(
        self, store: Any, unit: Any, grid_window: Window, layout: Any
    ) -> Any | None:
        """Return one scene BurstWindow overlapping *grid_window*, or ``None``."""
        from faninsar.processing.mosaicking import BurstWindow

        overlap = _overlap_window(grid_window, _unit_grid_box(unit))
        if overlap is None:
            return None
        unit_window = (
            overlap[0] - unit.row_origin,
            overlap[1] - unit.row_origin,
            overlap[2] - unit.col_origin,
            overlap[3] - unit.col_origin,
        )
        payload = self._read_scene_unit_payload(store, unit, unit_window)
        mask = self._read_scene_unit_mask(store, unit, unit_window, payload)
        return BurstWindow(
            scope=layout.scope,
            burst_id=_scene_burst_id(unit),
            acquisitions=(str(store.date_id),),
            write_ids=(store.write_id,),
            phase_convention=_phase_convention(
                (unit.phase_state or {}).get("convention")
            ),
            window=overlap,
            complex=payload,
            valid_mask=np.asarray(mask, dtype=np.bool_),
        )

    def _read_burst_payload(
        self,
        product: Product,
        complex_name: str,
        mask_name: str,
        unit_window: Window,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one burst IFG payload through the public windowed product seam."""
        payload = product.read(complex_name, window=unit_window)
        mask = (
            product.read(mask_name, window=unit_window)
            if mask_name in product.assets
            else np.ones(_window_shape(unit_window), dtype=np.bool_)
        )
        return payload, np.asarray(mask, dtype=np.bool_)

    def _read_scene_unit_payload(
        self, store: Any, unit: Any, unit_window: Window
    ) -> np.ndarray:
        """Read one scene unit through the public windowed store seam."""
        return _unwrap_payload(store.read(unit.tag, window=unit_window))

    def _read_scene_unit_mask(
        self,
        store: Any,
        unit: Any,
        unit_window: Window,
        payload: np.ndarray,
    ) -> np.ndarray:
        """Read one scene unit's support mask through the windowed store seam."""
        from faninsar.io.storage.scene_store import valid_samples

        read_valid = getattr(store, "read_valid_mask", None)
        if read_valid is not None:
            mask = read_valid(unit.tag, window=unit_window)
            if mask is not None:
                return np.asarray(mask, dtype=np.bool_)
        asset = f"{unit.tag}.valid"
        if asset in self._product.assets:
            return np.asarray(
                self._read_product_array(asset, unit_window), dtype=np.bool_
            )
        return valid_samples(payload)

    def _layout_for_window(self, local: Window) -> Any:
        """Return the committed merge layout cropped to the local read window."""
        return _crop_committed_layout(self._merge_layout, self._grid_from_local(local))

    def _place_merged_array(
        self,
        field: np.ndarray,
        merged_window: Window,
        local: Window,
        *,
        fill: object,
    ) -> np.ndarray:
        """Paste a MergedWindow field into the local read canvas."""
        canvas = np.full(_window_shape(local), fill, dtype=field.dtype)
        grid = self._grid_from_local(local)
        overlap = _overlap_window(_as_window(tuple(merged_window)), grid)
        if overlap is None:
            return canvas
        dest = (
            slice(overlap[0] - grid[0], overlap[1] - grid[0]),
            slice(overlap[2] - grid[2], overlap[3] - grid[2]),
        )
        src = (
            slice(overlap[0] - merged_window[0], overlap[1] - merged_window[0]),
            slice(overlap[2] - merged_window[2], overlap[3] - merged_window[2]),
        )
        canvas[dest] = field[src]
        return canvas

    def _scene_store(self) -> Any:
        """Open the scene store honoring the current unit selection."""
        from faninsar.io.storage.scene_store import CoregisteredSceneStore

        store = CoregisteredSceneStore.open(self._product.path)
        if self._scene_selection is not None:
            store = store.select_units(self._scene_selection)
        return store

    def _read_scene_los(self, name: str) -> np.ndarray:
        """Derive one LOS unit-vector layer from the Reference geometry assets.

        Opens the scene store through the product path so the lazy
        materialization trigger in
        :func:`~faninsar.io.storage.scene_geometry.los_unit_vector` runs the
        rdr2geo solve and commits ``incidence``/``azimuth`` on first access.
        """
        from faninsar.io.storage.scene_geometry import los_unit_vector
        from faninsar.io.storage.scene_store import (
            SCENE_KIND,
            CoregisteredSceneStore,
        )

        if self._product.kind != SCENE_KIND:
            reject_invalid_state(
                f"layer {name!r} is virtual and requires a scene product source"
            )
        store = CoregisteredSceneStore.open(self._product.path)
        return los_unit_vector(store, name)

    def _read(self, name: str, window: Window | None = None) -> np.ndarray:
        if self._merge_on_read and name in (
            *SCENE_LAYER_NAMES,
            "complex",
            "wrapped_phase",
            "amplitude",
            "valid_mask",
        ):
            values = self._read_merged(name, window)
        elif name in self._product.assets:
            values = self._read_product_array(name, window)
        elif self._product.kind == "scene" and name in _SCENE_LOS_LAYERS:
            values = self._read_scene_los(name)
            if window is not None:
                values = values[window[0] : window[1], window[2] : window[3]]
        elif self._product.kind == "scene" and name in SCENE_LAYER_NAMES:
            values = self._read_scene_layer(name, window)
        elif name in _VIRTUAL_IFG_LAYERS:
            values = self._read_virtual(name, window)
        else:
            message = (
                f"layer {name!r} is not stored in {self._product.path}; it is "
                "neither a persisted asset nor a derivable virtual layer"
            )
            raise FileNotFoundError(message)
        expected = self.shape if window is None else _window_shape(window)
        if values.shape != expected:
            reject_invalid_state(f"layer {name!r} shape differs from its product")
        return values

    def _read_merged(self, name: str, window: Window | None) -> np.ndarray:
        """Read one merged layer through the shared numerical kernel."""
        local = window if window is not None else (0, self.shape[0], 0, self.shape[1])
        kind: Literal["scene", "ifg"] = (
            "scene" if self._product.kind == "scene" else "ifg"
        )
        store = self._scene_store() if kind == "scene" else None
        merged = self._merge_window(store, local, kind=kind)
        if merged is None:
            return self._empty_merged_values(name, local)
        mosaic = self._place_merged_array(
            np.asarray(merged.complex), merged.window, local, fill=np.complex64(0)
        )
        values: np.ndarray | None
        if name in {"slc", "complex"}:
            values = mosaic
        elif name == "amplitude":
            values = np.abs(mosaic).astype(np.float32)
        elif name == "intensity":
            values = (np.abs(mosaic) ** 2).astype(np.float32)
        elif name in {"phase", "wrapped_phase"}:
            values = np.angle(mosaic).astype(np.float32)
        elif name == "valid_mask":
            values = self._place_merged_array(
                np.asarray(merged.valid_mask, dtype=np.bool_),
                merged.window,
                local,
                fill=False,
            )
        elif name == "coherence" and merged.coherence is not None:
            values = self._place_merged_array(
                np.asarray(merged.coherence),
                merged.window,
                local,
                fill=np.float32(np.nan),
            )
        else:
            values = None
        if values is None:
            message = f"unknown merged layer: {name!r}"
            raise FileNotFoundError(message)
        return values

    def _empty_merged_values(self, name: str, local: Window) -> np.ndarray:
        """Return an empty local canvas when no burst overlaps the window."""
        shape = _window_shape(local)
        if name == "valid_mask":
            return np.zeros(shape, dtype=np.bool_)
        if name in {"slc", "complex"}:
            return np.zeros(shape, dtype=np.complex64)
        return np.full(shape, np.nan, dtype=np.float32)

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read a detached array after checking the source commit and dtype.

        Parameters
        ----------
        window : tuple of int, optional
            Half-open source-grid window ``(r0, r1, c0, c1)``. Coordinates use
            the same origin as :attr:`origin`; they are translated to local
            array indices before the storage ``read(..., window=)`` seam.
            ``None`` reads the full layer.

        Returns
        -------
        numpy.ndarray
            Newly loaded values that may be modified without changing this
            committed layer or its source identity. Virtual layers are
            derived from their owning complex IFG commit. A windowed read
            materializes only the requested local rectangle.

        """
        self.validate_source()
        local = self._localize_window(window)
        values = self._read(self.name, window=local)
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

        Merge-on-read layers and virtual IFG layers (``wrapped_phase``,
        ``amplitude``) compute through :meth:`read`. Other scene layers open
        each selected unit's persisted zarr lazily and mosaic with the
        first-valid-sample overlap rule (matching ``mosaic_slc``); the
        ``slc`` mosaic then derives ``amplitude``, ``intensity``, ``phase``,
        or ``valid_mask``. ``validate_source`` runs at build time. Paths that
        compute through :meth:`read` re-validate on ``.compute()``; other
        lazy mosaics do not — callers needing a staleness re-check use
        :attr:`values` or :meth:`validate_source`.

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
        is_scene = self._product.kind == "scene"
        if self._merge_on_read or self.name in _VIRTUAL_IFG_LAYERS:
            from dask import delayed

            data = da.from_delayed(
                delayed(self.read)(),
                shape=self.shape,
                dtype=self._lazy_dtype(),
            )
        elif is_scene and self.name in _SCENE_LOS_LAYERS:
            # Virtual LOS: derive eagerly at build time from the stored
            # incidence/azimuth assets (which materialize lazily on demand).
            data = da.from_array(self._read_scene_los(self.name), chunks=self.shape)
        elif is_scene and self.name in product.assets:
            # Committed Reference geometry assets are single-frame rasters on
            # the scene item, not per-unit mosaics; read the zarr directly.
            data = da.from_zarr(str(self.path), chunks=self.shape)
        elif is_scene and self.name not in SCENE_LAYER_NAMES:
            message = (
                f"layer {self.name!r} is not stored in {self._product.path}; "
                "it is neither a persisted asset nor a derivable virtual layer"
            )
            raise FileNotFoundError(message)
        elif is_scene:
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
        attrs = {
            "shape": self.shape,
            "multilook": tuple(self.multilook),
            "full_radar_shape": self.full_radar_shape,
            "origin": self.origin,
            "asset_kind": None if self.asset_kind is None else str(self.asset_kind),
            "crs": self.crs,
            "transform": product.properties.get("proj:transform"),
            "source_write_id": product.write_id,
        }
        return xr.DataArray(data, dims=("y", "x"), name=self.name, attrs=attrs)

    def _lazy_dtype(self) -> np.dtype[Any]:
        """Return the dask dtype for a lazy full-layer read."""
        if self.name == "valid_mask":
            return np.dtype(np.bool_)
        if self.name in {"slc", "complex"}:
            return np.dtype(np.complex64)
        return np.dtype(np.float32)

    @property
    def valid_mask(self) -> np.ndarray | None:
        """Read the persisted finite/invalid support mask.

        Scene layers prefer persisted ``{tag}.valid`` assets; otherwise they
        compute support from the mosaic. IFG and unwrap products read the
        stored ``valid_mask`` asset. Support is never inferred from sample
        values (phase, amplitude, or connected-component labels); a supported
        ``0j`` or label-``0`` sample stays valid, and an unsupported pixel
        stays invalid even if a label is non-negative.
        """
        self.validate_source()
        return self._support_mask()

    def _support_mask(self, window: Window | None = None) -> np.ndarray | None:
        """Read support, optionally windowed in local layer coordinates."""
        mask = None
        local = self._localize_window(window) if window is not None else None
        if self._product.kind == "scene":
            mask = self._read("valid_mask", window=local)
            if mask.dtype != np.bool_:
                reject_invalid_state("layer valid_mask must have boolean dtype")
            return mask
        if "valid_mask" in self._product.assets:
            mask = self._read("valid_mask", window=local)
            if mask.dtype != np.bool_:
                reject_invalid_state("layer valid_mask must have boolean dtype")
        return mask

    def _candidate_window(self) -> Window:
        """Return the coarse layout or footprint window for a cropped read.

        A selected unmerged burst uses its full ROI-qualified footprint, not
        an owner==burst midpoint trim. Merged polygon views may intersect
        layout source windows when a layout is bound.
        """
        box = self._grid_box()
        layout = self._merge_layout
        if self._is_selected_burst() or layout is None:
            return box
        if not layout.source_windows:
            overlap = _overlap_window(box, layout.window)
            return box if overlap is None else overlap
        union: Window | None = None
        for source in layout.source_windows:
            overlap = _overlap_window(box, source.window)
            if overlap is None:
                continue
            if union is None:
                union = overlap
            else:
                union = (
                    min(union[0], overlap[0]),
                    max(union[1], overlap[1]),
                    min(union[2], overlap[2]),
                    max(union[3], overlap[3]),
                )
        return box if union is None else union

    def _is_selected_burst(self) -> bool:
        """Return whether this layer is a single selected unmerged burst."""
        return self._scene_selection is not None and len(self._scene_selection) == 1

    def _crop_window(self) -> Window | None:
        """Return the tight valid-support window, or ``None`` when empty."""
        candidate = self._candidate_window()
        mask_window = None if candidate == self._grid_box() else candidate
        mask = self._support_mask(window=mask_window)
        if mask is None:
            return candidate
        origin = self._grid_box() if mask_window is None else mask_window
        return _tight_support_window(mask, origin)

    def _imshow_extent(
        self, window: Window | None
    ) -> tuple[float, float, float, float]:
        """Return ``imshow`` extent ``(left, right, bottom, top)`` for a window."""
        if window is None:
            window = self._grid_box()
        row0, row1, col0, col1 = window
        transform = self._product.properties.get("proj:transform")
        shifted = None if transform is None else _shift_affine(transform, row0, col0)
        if shifted is not None:
            a, b, c, d, e, f = shifted
            left = c
            top = f
            right = c + a * (col1 - col0) + b * (row1 - row0)
            bottom = f + d * (col1 - col0) + e * (row1 - row0)
            return left, right, bottom, top
        return (col0 - 0.5, col1 - 0.5, row1 - 0.5, row0 - 0.5)

    def _plot_nodata(
        self,
        ax: Any,
        *,
        figsize: tuple[float, float] | None,
        title: str | None,
        iq: bool,
    ) -> tuple[Any, Any, Any]:
        """Draw a no-data panel instead of fabricating a zero image."""
        import matplotlib.pyplot as plt

        label = title if title is not None else self.name
        if iq:
            fig, axes = plt.subplots(
                1, 2, figsize=figsize, sharex=True, sharey=True, squeeze=False
            )
            images: list[Any] = []
            for axis, suffix in ((axes[0, 0], "I"), (axes[0, 1], "Q")):
                axis.set_title(f"{label} ({suffix})")
                axis.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                images.append(None)
            return fig, (axes[0, 0], axes[0, 1]), (images[0], images[1])
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.set_title(label)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax, None

    def plot(
        self,
        ax: Any = None,
        *,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = True,
        colorbar_location: Literal["auto", "right", "left", "top", "bottom"] = "auto",
        colorbar_kwargs: Mapping[str, Any] | None = None,
        masked: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        hist_colorbar: bool = True,
        crop: bool = True,
        ref_points: object = "inherit",
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

        ``ref_points="inherit"`` uses the parent Stack defaults when this
        layer carries a session. Inherit without a session is a no-op.
        ``None`` disables markers and display referencing. Explicit
        :class:`~faninsar.data.query.Points` override locations and keep the
        Stack policy, or ``"common"`` when there is no Stack. Unwrapped phase
        and displacement subtract the valid-point arithmetic mean from a
        temporary copy before automatic ``vmin``/``vmax``; wrapped phase,
        coherence, labels, and amplitude only overlay markers. Sampling uses
        the full scientific layer even when ``crop`` or ``masked`` changes
        the displayed buffer. Layer ``read`` results are not modified.
        Subtracting a mean does not independently align unwrap components.

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
            same limits apply to both panels. Automatic limits for unwrapped
            phase and displacement are computed after reference subtraction.
        colorbar : bool, optional
            Add a colorbar. Default True.
        colorbar_location : {'auto', 'right', 'left', 'top', 'bottom'}, optional
            Colorbar side relative to each displayed image. ``"auto"`` chooses
            ``"bottom"`` for wide data boxes (displayed width/height ratio at
            or above 2.0) and ``"right"`` otherwise. Under
            ``constrained_layout``, ``"auto"`` stays on the right because the
            layout engine repositions horizontal colorbars only approximately.
            A warning is emitted when the chosen side cannot match exactly:
            horizontal bars may be shorter and offset, while vertical bars may
            be offset horizontally when the image is narrower than the axes
            cell. Default ``"auto"``.
        colorbar_kwargs : mapping, optional
            Extra keyword arguments forwarded to the colorbar constructor. The
            keys ``data``, ``mappable``, ``ax``, ``cax``, ``location``, and
            ``orientation`` are reserved; use ``colorbar_location`` for
            placement. ``match_extent=False`` restores the old full-cell
            colorbar length instead of matching the displayed image extent.
        masked : bool, optional
            Mask pixels outside :attr:`valid_mask` to NaN. Default True.
            Reference-point validity always consults the support mask.
        figsize : tuple[float, float], optional
            Figure size when creating a new figure.
        title : str, optional
            Axes title. Defaults to the layer's public name; a complex layer
            suffixes ``I``/``Q`` on each panel.
        hist_colorbar : bool, optional
            Use FanInSAR :class:`~faninsar.plotting.HistColorbar` (histogram
            embedded beside the gradient) instead of a plain colorbar.
            Default True.
        crop : bool, optional
            Bound the read to actual product-valid support inside the layout
            or footprint candidate window (default True for Polygon layers).
            ``crop=False`` keeps the original scientific view extent. Empty
            support reports no data rather than a fabricated zero image.
            Interior holes and disjoint distances are preserved. Reference
            offsets do not change with crop.
        ref_points : Points, None, ``"inherit"``, or Parameter.INHERIT, optional
            Inherited Stack points, an explicit override, or ``None`` to
            disable. Standalone geographic layers may resolve explicit
            geographic points without a Stack; standalone radar layers
            need geometry context.
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

        plot_window = self._grid_box()
        read_window: Window | None = None
        if crop:
            plot_window = self._crop_window()
            if plot_window is None:
                return self._plot_nodata(
                    ax,
                    figsize=figsize,
                    title=title,
                    iq=self.asset_kind
                    in (AssetKind.COMPLEX_INTERFEROGRAM, AssetKind.SLC),
                )
            if plot_window != self._grid_box():
                read_window = plot_window
        if self.asset_kind in (AssetKind.COMPLEX_INTERFEROGRAM, AssetKind.SLC):
            return self._plot_iq(
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                colorbar_location=colorbar_location,
                colorbar_kwargs=colorbar_kwargs,
                masked=masked,
                figsize=figsize,
                title=title,
                hist_colorbar=hist_colorbar,
                window=read_window,
                plot_window=plot_window,
                ref_points=ref_points,
                **imshow_kwargs,
            )

        raw = self.read(window=read_window)
        values = np.asarray(raw, dtype=np.result_type(raw, np.float32))
        if masked:
            mask = self._support_mask(window=read_window)
            if mask is not None:
                values = np.where(mask, values, np.nan)
        subtract = self.asset_kind in _REFERENCE_SUBTRACT_KINDS
        values, markers = _apply_plot_reference(
            self, values, ref_points, subtract=subtract
        )

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
            finite = values[np.isfinite(values)]
            fallback = (
                (float(np.nanmin(finite)), float(np.nanmax(finite)))
                if finite.size
                else (0.0, 1.0)
            )
            lo, hi = default_limits.get(kind, fallback)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            if title is None:
                title = self.name
        extent = self._imshow_extent(plot_window)
        im = ax.imshow(
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            **imshow_kwargs,
        )
        ax.set_title(title if title is not None else "")
        _scatter_reference_markers(ax, markers, extent)
        if colorbar:
            self._add_colorbar(
                ax.figure,
                ax,
                im,
                values,
                phase=kind == AssetKind.WRAPPED_PHASE,
                hist=hist_colorbar,
                location=colorbar_location,
                colorbar_kwargs=colorbar_kwargs,
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
        colorbar_location: Literal["auto", "right", "left", "top", "bottom"] = "auto",
        colorbar_kwargs: Mapping[str, Any] | None = None,
        window: Window | None = None,
        plot_window: Window | None = None,
        ref_points: object = "inherit",
        **imshow_kwargs: Any,
    ) -> tuple[Any, tuple[Any, Any], tuple[Any, Any]]:
        """Render a complex layer as side-by-side in-phase/quadrature panels.

        ``colorbar_location`` and ``colorbar_kwargs`` are resolved separately
        for each panel, so ``"auto"`` follows that panel's displayed data box.
        """
        import matplotlib.pyplot as plt

        values = np.asarray(self.read(window=window), dtype=np.complex64)
        mask = self._support_mask(window=window) if masked else None
        _, markers = _apply_plot_reference(
            self, np.asarray(values.real), ref_points, subtract=False
        )

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
        extent = self._imshow_extent(
            plot_window
            if plot_window is not None
            else (window if window is not None else self._grid_box())
        )
        ims = []
        for axis, component, suffix in (
            (axes[0, 0], values.real, "I"),
            (axes[0, 1], values.imag, "Q"),
        ):
            data = np.asarray(component, dtype=np.float32)
            if mask is not None:
                data = np.where(mask, data, np.nan)
            im = axis.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, **imshow_kwargs
            )
            axis.set_title(f"{base} ({suffix})")
            _scatter_reference_markers(axis, markers, extent)
            ims.append(im)
            if colorbar:
                self._add_colorbar(
                    fig,
                    axis,
                    im,
                    data,
                    hist=hist_colorbar,
                    location=colorbar_location,
                    colorbar_kwargs=colorbar_kwargs,
                )
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
        location: Literal["auto", "right", "left", "top", "bottom"] = "auto",
        colorbar_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach a colorbar, FanInSAR HistColorbar when *hist* is set."""
        allowed = {"auto", "right", "left", "top", "bottom"}
        if location not in allowed:
            msg = (
                f"colorbar_location must be one of {sorted(allowed)}; got {location!r}"
            )
            reject(msg, ValueError, logger=logger)
        extra = dict(colorbar_kwargs or {})
        reserved = {"data", "mappable", "ax", "cax", "location", "orientation"}
        overlap = reserved.intersection(extra)
        if overlap:
            msg = (
                "colorbar_kwargs may not override "
                f"{sorted(overlap)}; use colorbar_location for placement."
            )
            reject(msg, ValueError, logger=logger)
        resolved = _resolve_colorbar_location(ax, location)
        engine = fig.get_layout_engine()
        if engine is not None and not engine.colorbar_gridspec:
            if resolved in ("bottom", "top"):
                warnings.warn(
                    f"colorbar_location={resolved!r} under constrained_layout is "
                    "approximate: the layout engine anchors horizontal colorbars "
                    "to the axes cell, so the bar may be shorter than the visible "
                    "data width and offset from the image. Use "
                    "colorbar_location='right' or a non-constrained layout for "
                    "exact matching.",
                    UserWarning,
                    stacklevel=3,
                )
            elif _displayed_fills_cell_width(ax) is False:
                warnings.warn(
                    f"colorbar_location={resolved!r} under constrained_layout may "
                    "be offset horizontally: the layout engine anchors the bar to "
                    "the axes cell while the visible image is narrower. Its length "
                    "still matches the image; use a non-constrained layout for "
                    "exact positioning.",
                    UserWarning,
                    stacklevel=3,
                )
        if hist:
            import faninsar.plotting  # noqa: F401  (installs Figure.hist_colorbar)

            hcb = fig.hist_colorbar(
                np.asarray(data).ravel(),
                mappable=im,
                ax=ax,
                location=resolved,
                **extra,
            )
            if phase:
                from faninsar.plotting.formatters import setup_phase_axis

                axis = (
                    hcb.ax_cbar.yaxis
                    if hcb.orientation == "vertical"
                    else hcb.ax_cbar.xaxis
                )
                setup_phase_axis(axis)
            return
        match_extent = bool(extra.pop("match_extent", True))
        if match_extent:
            from faninsar.plotting.colorbar import _compute_extent_match_factor

            orientation = "vertical" if resolved in ("left", "right") else "horizontal"
            fraction = float(extra.get("fraction", 0.15))
            if extra.get("pad") is None:
                extra["pad"] = 0.15 if orientation == "horizontal" else 0.05
            match_factor = _compute_extent_match_factor(
                ax,
                orientation=orientation,
                fraction=fraction,
                pad=float(extra["pad"]),
            )
            extra["shrink"] = float(extra.get("shrink", 1.0)) * match_factor
        cbar = fig.colorbar(im, ax=ax, location=resolved, **extra)
        if phase:
            from faninsar.plotting.formatters import setup_phase_axis

            axis = cbar.ax.yaxis if cbar.orientation == "vertical" else cbar.ax.xaxis
            setup_phase_axis(axis)


@dataclass(frozen=True, slots=True)
class MergedBurstLayers:
    """Lazy merged burst view of one scene or pair. Not a persisted product.

    Collection ``merge_bursts()`` returns per-date or per-pair views of this
    type's layer handles. Reads apply
    :func:`faninsar.processing.mosaicking.merge_bursts` to bounded burst
    windows and do not rewrite source products.
    """

    _product: Product = field(repr=False)
    full_radar_shape: tuple[int, int]
    multilook: Looks | tuple[int, int] = field(default_factory=lambda: Looks(1, 1))
    origin: tuple[int, int] = (0, 0)
    crs: str | None = None
    extent: tuple[int, int] | None = None
    _scene_selection: frozenset[str] | None = field(default=None, repr=False)
    _upstream: tuple[Product, ...] = field(default=(), repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)
    _merge_layout: MergeLayoutWindow | None = field(
        default=None, repr=False, compare=False
    )
    _layout_product: Product | None = field(default=None, repr=False, compare=False)
    _burst_products: tuple[Product, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize the look factors to range-first :class:`Looks`."""
        object.__setattr__(self, "multilook", coerce_looks(self.multilook))

    def _layer(self, name: str, asset_kind: AssetKind) -> ProductLayer:
        """Return one merged lazy layer bound to this view."""
        return ProductLayer(
            self._product,
            name,
            asset_kind,
            self.full_radar_shape,
            self.multilook,
            origin=self.origin,
            crs=self.crs,
            extent=self.extent,
            _scene_selection=self._scene_selection,
            _upstream=self._upstream,
            _session=self._session,
            _merge_on_read=True,
            _merge_layout=self._merge_layout,
            _layout_product=self._layout_product,
            _burst_products=self._burst_products,
        )

    @property
    def slc(self) -> ProductLayer:
        """Return the merged complex SLC layer."""
        return self._layer("slc", AssetKind.SLC)

    @property
    def amplitude(self) -> ProductLayer:
        """Return merged amplitude."""
        return self._layer("amplitude", AssetKind.AMPLITUDE)

    @property
    def intensity(self) -> ProductLayer:
        """Return merged intensity."""
        return self._layer("intensity", AssetKind.AMPLITUDE)

    @property
    def phase(self) -> ProductLayer:
        """Return merged wrapped phase."""
        name = "phase" if self._product.kind == "scene" else "wrapped_phase"
        return self._layer(name, AssetKind.WRAPPED_PHASE)

    @property
    def complex(self) -> ProductLayer:
        """Return the merged complex interferogram layer."""
        return self._layer("complex", AssetKind.COMPLEX_INTERFEROGRAM)

    @property
    def wrapped_phase(self) -> ProductLayer:
        """Return merged wrapped phase of an IFG view."""
        return self._layer("wrapped_phase", AssetKind.WRAPPED_PHASE)


@dataclass(frozen=True, slots=True)
class InterferogramLayers:
    """Flat per-Pair IFG view of lazy rasters and its bound unwrap product.

    Parameters
    ----------
    _product : Product
        Committed interferogram product.
    full_radar_shape : tuple[int, int]
        Authoritative full frame from the Stack Reference.
    multilook : Looks or tuple[int, int]
        Range-first look factors; a two-sequence is read as
        ``(range, azimuth)`` and normalized to
        :class:`~faninsar.core.sar_tools.Looks`.
    crs : str or None, optional
        Map CRS when this is already a geographic product.

    """

    _product: Product = field(repr=False)
    full_radar_shape: tuple[int, int]
    multilook: Looks | tuple[int, int]
    crs: str | None = None
    _upstream: tuple[Product, ...] = field(default=(), repr=False)
    _session: GeocodeSession | None = field(default=None, repr=False, compare=False)
    _merge_on_read: bool = field(default=False, repr=False, compare=False)
    _merge_layout: MergeLayoutWindow | None = field(
        default=None, repr=False, compare=False
    )
    _layout_product: Product | None = field(default=None, repr=False, compare=False)
    _burst_products: tuple[Product, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize the look factors to range-first :class:`Looks`."""
        object.__setattr__(self, "multilook", coerce_looks(self.multilook))

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
        if not virtual and not self._merge_on_read and name not in self._product.assets:
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
            origin=(0, 0),
            crs=self.crs,
            extent=self.full_radar_shape if self._merge_on_read else None,
            _upstream=self._upstream,
            _session=self._session,
            _merge_on_read=self._merge_on_read,
            _merge_layout=self._merge_layout,
            _layout_product=self._layout_product,
            _burst_products=self._burst_products,
        )

    def _unwrap_layer(
        self,
        name: str,
        asset_kind: AssetKind | None,
        *,
        source: str = "complex",
    ) -> ProductLayer:
        """Return one layer of the unwrap product bound to this IFG commit."""
        from faninsar.io.storage.ifg_store import (
            InterferogramArtifactStore,
            unwrap_ion_root,
            unwrap_root,
        )

        root = (
            unwrap_ion_root(self.root)
            if source == "complex_ion"
            else unwrap_root(self.root)
        )
        if not (root / "item.json").is_file():
            message = (
                f"Pair has no unwrapped result at {root}; run an unwrap stage first"
            )
            raise FileNotFoundError(message)
        store = InterferogramArtifactStore.open(self.root)
        product = (
            store.unwrap_ion_product()
            if source == "complex_ion"
            else store.unwrap_product()
        )
        if name not in product.assets:
            message = (
                f"unwrap product {product.path} has no {name!r} layer; "
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

    def _complex_ion_layer(self, name: str, asset_kind: AssetKind) -> ProductLayer:
        """Return one layer of the current complex_ion sibling."""
        from faninsar.io.storage.ifg_store import (
            InterferogramArtifactStore,
            complex_ion_root,
        )

        root = complex_ion_root(self.root)
        if not (root / "item.json").is_file():
            message = (
                f"Pair has no complex_ion result at {root}; "
                "run apply_ionosphere_correction first"
            )
            raise FileNotFoundError(message)
        product = InterferogramArtifactStore.open(self.root).complex_ion_product()
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

    @property
    def complex_ion(self) -> ProductLayer:
        """Return the ion-corrected complex interferogram sibling."""
        return self._complex_ion_layer("complex", AssetKind.COMPLEX_INTERFEROGRAM)

    @property
    def wrapped_ion(self) -> ProductLayer:
        """Return wrapped phase derived from the complex_ion sibling."""
        return self._complex_ion_layer("wrapped_phase", AssetKind.WRAPPED_PHASE)

    @property
    def amplitude_ion(self) -> ProductLayer:
        """Return amplitude derived from the complex_ion sibling."""
        return self._complex_ion_layer("amplitude", AssetKind.AMPLITUDE)

    @property
    def unwrapped_ion(self) -> ProductLayer:
        """Return unwrapped phase of the ion-corrected complex IFG."""
        return self._unwrap_layer(
            "unwrapped_phase", AssetKind.UNWRAPPED_PHASE, source="complex_ion"
        )

    @property
    def connected_components_ion(self) -> ProductLayer:
        """Return component labels of the ion-corrected unwrap product."""
        return self._unwrap_layer(
            "connected_components", AssetKind.LABEL, source="complex_ion"
        )

    def __repr__(self) -> str:
        """Return a concise summary of this per-Pair IFG layer view."""
        return (
            f"InterferogramLayers(pair={self._product.path.name!r}, "
            f"looks={self.multilook.range}x{self.multilook.azimuth}, "
            f"layers={self.layers})"
        )
