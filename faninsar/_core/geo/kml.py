"""KML and KMZ exporters for array-backed geospatial overlays."""

from __future__ import annotations

import io
import math
import posixpath
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from matplotlib import ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from pykml.factory import KML_ElementMaker as KML
from pyproj.crs import CRS

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from os import PathLike

    from faninsar.query.bbox import BoundingBox

logger = setup_logger(__name__)


def _ensure_bounds_in_wgs84(
    bounds: tuple[float, ...],
) -> tuple[float, float, float, float]:
    """Ensure the bounds are in WGS84 coordinate system."""
    if len(bounds) != 4:
        msg = (
            f"bounds should be a tuple of (west, south, east, north), but got {bounds}"
        )
        logger.error(msg)
        raise ValueError(msg)
    west, south, east, north = bounds
    if west < -180 or east > 180 or south < -90 or north > 90:
        msg = (
            "bounds should be in WGS84 coordinate system, "
            f"but got [{west}, {south}, {east}, {north}]"
        )
        logger.error(msg)
        raise ValueError(msg)
    return west, south, east, north


def _normalize_kml_bounds(
    bounds: tuple[float, float, float, float] | BoundingBox,
) -> BoundingBox:
    """Normalize bounds to a WGS84 :class:`BoundingBox`.

    Parameters
    ----------
    bounds : tuple[float, float, float, float] | BoundingBox
        Bounds in ``(west, south, east, north)`` order or a bounding box instance.

    Returns
    -------
    BoundingBox
        Bounds normalized to WGS84.

    Raises
    ------
    TypeError
        If ``bounds`` is not an iterable of four floats or a bounding box.

    """
    from faninsar.query.bbox import BoundingBox

    wgs84 = CRS.from_epsg(4326)
    if isinstance(bounds, BoundingBox):
        bounds_wgs84 = bounds
    elif isinstance(bounds, Iterable):
        bounds_wgs84 = BoundingBox(*_ensure_bounds_in_wgs84(tuple(bounds)), crs=wgs84)
    else:
        msg = (
            "bounds should be a BoundingBox or an iterable of "
            "(west, south, east, north)"
        )
        logger.error(msg)
        raise TypeError(msg)

    if bounds_wgs84.crs != wgs84:
        bounds_wgs84 = bounds_wgs84.to_crs(wgs84)

    return bounds_wgs84


def _normalize_image_kwargs(
    img_kwargs: dict[str, Any] | None,
    *,
    interpolation: str,
) -> dict[str, Any]:
    """Normalize image rendering keyword arguments.

    Parameters
    ----------
    img_kwargs : dict[str, Any] | None
        Keyword arguments passed to :func:`matplotlib.axes.Axes.imshow`.
    interpolation : str
        Default interpolation method to use when not provided by the caller.

    Returns
    -------
    dict[str, Any]
        A shallow copy of the input keyword arguments with defaults applied.

    """
    img_kwargs_new = {} if img_kwargs is None else dict(img_kwargs)
    img_kwargs_new.setdefault("interpolation", interpolation)
    return img_kwargs_new


def _render_array_to_rgba(
    arr: np.ndarray,
    *,
    img_kwargs: dict[str, Any] | None = None,
    render_scale: float = 1.0,
) -> tuple[Figure, ScalarMappable, np.ndarray]:
    """Render an array to an exact-pixel RGBA image.

    Parameters
    ----------
    arr : np.ndarray
        Array to render with :func:`matplotlib.axes.Axes.imshow`.
    img_kwargs : dict[str, Any] | None, optional
        Keyword arguments for :func:`matplotlib.axes.Axes.imshow`.
    render_scale : float, optional
        Scale factor applied to the output width and height.

    Returns
    -------
    tuple[Figure, ScalarMappable, np.ndarray]
        Figure used for rendering, the ``imshow`` mappable, and the rendered
        RGBA image as ``uint8``.

    Raises
    ------
    ValueError
        If ``render_scale`` is not positive.

    """
    if render_scale <= 0:
        msg = f"render_scale should be positive, but got {render_scale}"
        logger.error(msg)
        raise ValueError(msg)

    if arr.ndim < 2:
        msg = f"arr should have at least 2 dimensions, but got shape {arr.shape}"
        logger.error(msg)
        raise ValueError(msg)

    render_height = max(1, round(arr.shape[0] * render_scale))
    render_width = max(1, round(arr.shape[1] * render_scale))
    dpi = 100

    fig = Figure(
        figsize=(render_width / dpi, render_height / dpi),
        dpi=dpi,
        frameon=False,
    )
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    mappable = ax.imshow(arr, **(img_kwargs or {}))
    ax.set_aspect("auto")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    return fig, mappable, rgba


def _render_colorbar_to_png_bytes(
    mappable: ScalarMappable,
    **kwargs: Any,
) -> bytes:
    """Render a colorbar to PNG bytes.

    Parameters
    ----------
    mappable : ScalarMappable
        The scalar mappable used to build the colorbar.
    **kwargs : Any
        Keyword arguments accepted by :func:`save_colorbar`, excluding
        ``out_file`` and ``mappable``.

    Returns
    -------
    bytes
        Encoded PNG data.

    """
    buffer = io.BytesIO()
    colorbar_mappable = ScalarMappable(norm=mappable.norm, cmap=mappable.cmap)
    array = mappable.get_array()
    if array is not None:
        colorbar_mappable.set_array(np.asarray(array))
    colorbar_mappable.set_clim(*mappable.get_clim())
    save_colorbar(buffer, colorbar_mappable, **kwargs)
    return buffer.getvalue()


def _rgba_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode an RGBA array to PNG bytes.

    Parameters
    ----------
    arr : np.ndarray
        RGBA array to encode.

    Returns
    -------
    bytes
        Encoded PNG data.

    """
    buffer = io.BytesIO()
    plt.imsave(buffer, arr, format="png")
    return buffer.getvalue()


def _resize_rgba_nearest(
    arr: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Resize an RGBA array with nearest-neighbor sampling.

    Parameters
    ----------
    arr : np.ndarray
        Input RGBA image.
    target_height : int
        Target height in pixels.
    target_width : int
        Target width in pixels.

    Returns
    -------
    np.ndarray
        Resized RGBA image.

    """
    if arr.shape[0] == target_height and arr.shape[1] == target_width:
        return arr

    row_idx = np.linspace(0, arr.shape[0] - 1, target_height).round().astype(int)
    col_idx = np.linspace(0, arr.shape[1] - 1, target_width).round().astype(int)
    return arr[row_idx][:, col_idx]


def _pixel_window_to_bounds(
    image_bounds: BoundingBox,
    image_width: int,
    image_height: int,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> BoundingBox:
    """Convert a pixel window to geographic bounds.

    Parameters
    ----------
    image_bounds : BoundingBox
        Full image bounds in WGS84.
    image_width : int
        Full image width in pixels.
    image_height : int
        Full image height in pixels.
    x0, x1, y0, y1 : int
        Pixel window coordinates where ``x1`` and ``y1`` are exclusive.

    Returns
    -------
    BoundingBox
        Geographic bounds for the requested pixel window.

    """
    from faninsar.query.bbox import BoundingBox

    lon_span = image_bounds.right - image_bounds.left
    lat_span = image_bounds.top - image_bounds.bottom
    west = image_bounds.left + lon_span * x0 / image_width
    east = image_bounds.left + lon_span * x1 / image_width
    north = image_bounds.top - lat_span * y0 / image_height
    south = image_bounds.top - lat_span * y1 / image_height
    return BoundingBox(west, south, east, north, crs=image_bounds.crs)


def _node_lod(
    *,
    is_leaf: bool,
    tile_size: int,
) -> tuple[int, int]:
    """Compute region LOD limits for a tile node.

    Parameters
    ----------
    is_leaf : bool
        Whether the node is a leaf tile.
    tile_size : int
        Maximum tile size in pixels.

    Returns
    -------
    tuple[int, int]
        ``(min_lod_pixels, max_lod_pixels)`` for the node overlay.

    """
    return (0, -1 if is_leaf else tile_size * 2)


def _kml_latlon_box(bounds: BoundingBox) -> Any:
    """Create a KML LatLonBox element.

    Parameters
    ----------
    bounds : BoundingBox
        Tile bounds in WGS84.

    Returns
    -------
    Any
        A ``pykml`` LatLonBox element.

    """
    return KML.LatLonBox(
        KML.north(bounds.top),
        KML.south(bounds.bottom),
        KML.east(bounds.right),
        KML.west(bounds.left),
    )


def _kml_region(
    bounds: BoundingBox,
    *,
    min_lod_pixels: int,
    max_lod_pixels: int,
) -> Any:
    """Create a KML Region element.

    Parameters
    ----------
    bounds : BoundingBox
        Region bounds in WGS84.
    min_lod_pixels : int
        Minimum number of screen pixels before the region loads.
    max_lod_pixels : int
        Maximum number of screen pixels before the region unloads.

    Returns
    -------
    Any
        A ``pykml`` Region element.

    """
    return KML.Region(
        KML.LatLonAltBox(
            KML.north(bounds.top),
            KML.south(bounds.bottom),
            KML.east(bounds.right),
            KML.west(bounds.left),
        ),
        KML.Lod(
            KML.minLodPixels(str(min_lod_pixels)),
            KML.maxLodPixels(str(max_lod_pixels)),
        ),
    )


def _relative_kml_href(source_path: str, target_path: str) -> str:
    """Build a relative KML href.

    Parameters
    ----------
    source_path : str
        Source file path inside the KMZ.
    target_path : str
        Target file path inside the KMZ.

    Returns
    -------
    str
        Relative POSIX path from ``source_path`` to ``target_path``.

    """
    source_dir = posixpath.dirname(source_path) or "."
    return posixpath.relpath(target_path, start=source_dir)


@dataclass(slots=True)
class _TiledKmzNode:
    """Internal quadtree node used to build a tiled KMZ."""

    level: int
    x: int
    y: int
    x0: int
    x1: int
    y0: int
    y1: int
    bounds: BoundingBox
    children: list[_TiledKmzNode] = field(default_factory=list)

    @property
    def width(self) -> int:
        """Width of the node window in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Height of the node window in pixels."""
        return self.y1 - self.y0

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no children."""
        return not self.children

    @property
    def kml_path(self) -> str:
        """KMZ-internal KML path for the node."""
        return f"tiles/{self.level}/{self.x}/{self.y}.kml"

    @property
    def png_path(self) -> str:
        """KMZ-internal PNG path for the node."""
        return f"tiles/{self.level}/{self.x}/{self.y}.png"


def _build_tiled_kmz_tree(
    *,
    image_width: int,
    image_height: int,
    bounds: BoundingBox,
    tile_size: int,
) -> _TiledKmzNode:
    """Build a quadtree over a rendered overlay image.

    Parameters
    ----------
    image_width : int
        Rendered image width in pixels.
    image_height : int
        Rendered image height in pixels.
    bounds : BoundingBox
        Image bounds in WGS84.
    tile_size : int
        Maximum tile size in pixels.

    Returns
    -------
    _TiledKmzNode
        Root node of the tile pyramid.

    """

    def build(
        level: int,
        x: int,
        y: int,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> _TiledKmzNode:
        node_bounds = _pixel_window_to_bounds(
            bounds,
            image_width,
            image_height,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )
        node = _TiledKmzNode(level, x, y, x0, x1, y0, y1, node_bounds)
        should_split = (
            (node.width > tile_size or node.height > tile_size)
            and node.width > 1
            and node.height > 1
        )
        if not should_split:
            return node

        x_mid = x0 + node.width // 2
        y_mid = y0 + node.height // 2
        child_windows = [
            (x0, x_mid, y0, y_mid),
            (x_mid, x1, y0, y_mid),
            (x0, x_mid, y_mid, y1),
            (x_mid, x1, y_mid, y1),
        ]
        for child_x_offset, child_y_offset, child_window in zip(
            (0, 1, 0, 1),
            (0, 0, 1, 1),
            child_windows,
            strict=True,
        ):
            child_x0, child_x1, child_y0, child_y1 = child_window
            if child_x0 == child_x1 or child_y0 == child_y1:
                continue
            node.children.append(
                build(
                    level + 1,
                    x * 2 + child_x_offset,
                    y * 2 + child_y_offset,
                    child_x0,
                    child_x1,
                    child_y0,
                    child_y1,
                )
            )
        return node

    return build(0, 0, 0, 0, image_width, 0, image_height)


def _iter_tiled_kmz_nodes(node: _TiledKmzNode) -> Iterator[_TiledKmzNode]:
    """Iterate through a tiled KMZ quadtree in depth-first order.

    Parameters
    ----------
    node : _TiledKmzNode
        Root or intermediate node.

    Yields
    ------
    _TiledKmzNode
        Each node in the tree.

    """
    yield node
    for child in node.children:
        yield from _iter_tiled_kmz_nodes(child)


def _tile_image_for_node(
    rgba: np.ndarray,
    node: _TiledKmzNode,
    *,
    tile_size: int,
) -> np.ndarray:
    """Extract and downsample the tile image for a quadtree node.

    Parameters
    ----------
    rgba : np.ndarray
        Full rendered RGBA image.
    node : _TiledKmzNode
        Node describing the source window.
    tile_size : int
        Maximum tile size in pixels.

    Returns
    -------
    np.ndarray
        Tile RGBA image.

    """
    tile = rgba[node.y0 : node.y1, node.x0 : node.x1]
    scale = max(tile.shape[1] / tile_size, tile.shape[0] / tile_size, 1.0)
    target_width = max(1, math.ceil(tile.shape[1] / scale))
    target_height = max(1, math.ceil(tile.shape[0] / scale))
    return _resize_rgba_nearest(tile, target_height, target_width)


def _tiled_kmz_tile_kml(
    node: _TiledKmzNode,
    *,
    tile_size: int,
    min_lod_pixels: int,
) -> bytes:
    """Build a tile KML document for a tiled KMZ node.

    Parameters
    ----------
    node : _TiledKmzNode
        Tile node to serialize.
    tile_size : int
        Maximum tile size in pixels.
    min_lod_pixels : int
        Minimum LOD threshold for child network links.

    Returns
    -------
    bytes
        Encoded KML document.

    """
    node_min_lod, node_max_lod = _node_lod(is_leaf=node.is_leaf, tile_size=tile_size)
    document = KML.Document(
        KML.GroundOverlay(
            KML.name(f"tile_{node.level}_{node.x}_{node.y}"),
            _kml_region(
                node.bounds,
                min_lod_pixels=node_min_lod,
                max_lod_pixels=node_max_lod,
            ),
            KML.Icon(KML.href(posixpath.basename(node.png_path))),
            _kml_latlon_box(node.bounds),
        )
    )

    for child in node.children:
        document.append(
            KML.NetworkLink(
                KML.name(f"tile_{child.level}_{child.x}_{child.y}"),
                _kml_region(
                    child.bounds,
                    min_lod_pixels=min_lod_pixels,
                    max_lod_pixels=-1,
                ),
                KML.Link(
                    KML.href(_relative_kml_href(node.kml_path, child.kml_path)),
                    KML.viewRefreshMode("onRegion"),
                ),
            )
        )

    return etree.tostring(
        KML.kml(document),
        pretty_print=True,
        encoding="utf-8",
        xml_declaration=True,
    )


def _tiled_kmz_root_kml(
    root: _TiledKmzNode,
    bounds: BoundingBox,
    *,
    colorbar_path: str,
) -> bytes:
    """Build the root KML document for a tiled KMZ.

    Parameters
    ----------
    root : _TiledKmzNode
        Root node of the tile pyramid.
    bounds : BoundingBox
        Full overlay bounds in WGS84.
    colorbar_path : str
        KMZ-internal path to the colorbar image.

    Returns
    -------
    bytes
        Encoded root KML document.

    """
    document = KML.Document(
        KML.NetworkLink(
            KML.name("root"),
            _kml_region(bounds, min_lod_pixels=0, max_lod_pixels=-1),
            KML.Link(
                KML.href(root.kml_path),
                KML.viewRefreshMode("onRegion"),
            ),
        ),
        KML.ScreenOverlay(
            KML.name("Color bar"),
            KML.Icon(KML.href(colorbar_path)),
            KML.overlayXY(x="1", y="0", xunits="fraction", yunits="fraction"),
            KML.screenXY(x="1", y="0", xunits="fraction", yunits="fraction"),
            KML.size(x="0", y="500", xunits="pixel", yunits="pixel"),
        ),
    )
    return etree.tostring(
        KML.kml(document),
        pretty_print=True,
        encoding="utf-8",
        xml_declaration=True,
    )


def save_colorbar(
    out_file: PathLike,
    mappable: ScalarMappable,
    figsize: tuple[float, float] = (0.18, 3.6),
    label: str | None = None,
    nbins: int | None = None,
    alpha: float = 0.5,
    **kwargs,
) -> None:
    """Save the colorbar to a file.

    Parameters
    ----------
    out_file: str or Path
        the path of the output colorbar file.
    mappable: ScalarMappable
        the ScalarMappable object to be used for the colorbar.
    figsize: tuple
        the size of the colorbar figure. Default is (0.18, 3.6).
    label: str
        the label of the colorbar. Default is None.
    nbins: int
        the number of bins of the colorbar. Default is None.
    alpha: float
        the transparency of the colorbar figure. Default is 0.5.
    kwargs: dict
        the keyword arguments for :func:`matplotlib.pyplot.colorbar` function.

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    kwargs.update({"fraction": 1})
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)

    # update colorbar label and ticks
    if label:
        cbar.set_label(label, fontsize=12)
    if nbins:
        cbar.locator = ticker.MaxNLocator(nbins=nbins)
        cbar.update_ticks()

    cbar.ax.tick_params(which="both", labelsize=12)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(alpha)
    fig.savefig(out_file, bbox_inches="tight", dpi=300)


def array2kml(
    arr: np.ndarray,
    out_file: PathLike,
    bounds: tuple[float, float, float, float] | BoundingBox,
    img_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    verbose: bool = True,
) -> None:
    """Write a numpy array into a kml file.

    Parameters
    ----------
    arr: numpy.ndarray
        the numpy array to be written into kml file.
    out_file: str or Path
        the path of the kml file.
    bounds: tuple or BoundingBox
        the bounds of image in [west, south, east, north] order in WGS84.
    img_kwargs: dict
        the keyword arguments for :func:`matplotlib.pyplot.imshow` function.
    cbar_kwargs: dict
        the keyword arguments for :func:`save_colorbar` function, except for
        the out_file and mappable argument.
    verbose: bool
        whether to print the information of the kml file. Default is verbose.

    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    img_kwargs = _normalize_image_kwargs(img_kwargs, interpolation="lanczos")
    bounds = _normalize_kml_bounds(bounds)

    out_file = Path(out_file)
    if out_file.suffix != ".kml":
        out_file = out_file.parent / (out_file.stem + ".kml")
    img_file = out_file.parent / (out_file.stem + ".png")
    cbar_file = out_file.parent / (out_file.stem + "_cbar.png")

    # plot image
    figsize = (arr.shape[1] / 100, arr.shape[0] / 100)
    plt.figure(figsize=figsize)
    im = plt.imshow(arr, **img_kwargs)
    plt.axis("off")
    plt.savefig(img_file, bbox_inches="tight", pad_inches=0, dpi=100, transparent=True)
    plt.close()

    # plot colorbar
    save_colorbar(cbar_file, im, **cbar_kwargs)

    # write kml file
    kml_doc = KML.Document()
    img_overlay = KML.GroundOverlay(
        KML.Icon(
            KML.href(img_file.name),
            KML.viewBoundScale(1),  # Set viewBoundScale to 1.
            KML.scale(1),  # Set scale to 1.
            KML.size(
                x=str(arr.shape[1]),
                y=str(arr.shape[0]),
                xunits="pixels",
                yunits="pixels",
            ),
        ),
        KML.LatLonBox(
            KML.north(bounds[3]),
            KML.south(bounds[1]),
            KML.east(bounds[2]),
            KML.west(bounds[0]),
        ),
    )
    kml_doc.append(img_overlay)

    # colorbar overlay
    cbar_overlay = KML.ScreenOverlay(
        KML.name("Color bar"),
        KML.Icon(KML.href(cbar_file.name)),
        KML.overlayXY(x="1", y="0", xunits="fraction", yunits="fraction"),
        KML.screenXY(x="1", y="0", xunits="fraction", yunits="fraction"),
        KML.size(x="0", y="500", xunits="pixel", yunits="pixel"),
    )
    kml_doc.append(cbar_overlay)

    kml = KML.kml(kml_doc)
    Path(out_file).write_text(
        etree.tostring(kml, pretty_print=True).decode("utf8"), encoding="utf-8"
    )
    if verbose:
        info = f"write kml file to {out_file}"
        logger.info(info)


def array2kmz(
    arr: np.ndarray,
    out_file: PathLike,
    bounds: tuple[float, float, float, float] | BoundingBox,
    img_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    keep_kml: bool = False,
    verbose: bool = True,
) -> None:
    """Write a numpy array into a kmz file.

    Parameters
    ----------
    arr: numpy.ndarray
        the numpy array to be written into kml file.
    out_file: str or Path
        the path of the kmz file.
    bounds: tuple or BoundingBox
        the bounds of image in [west, south, east, north] order in WGS84
    img_kwargs: dict
        the keyword arguments for :func:`matplotlib.pyplot.imshow` function.
    cbar_kwargs: dict
        the keyword arguments for :func:`save_colorbar` function, except for
        the out_file and mappable argument.
    keep_kml: bool
        whether to keep the kml file. Default is False.
    verbose: bool
        whether to print the information of the kmz file. Default is verbose.

    """
    if cbar_kwargs is None:
        cbar_kwargs = {}
    img_kwargs = _normalize_image_kwargs(img_kwargs, interpolation="lanczos")
    out_file = Path(out_file)
    if out_file.suffix != ".kmz":
        out_file = out_file.parent / (out_file.stem + ".kmz")
    img_file = out_file.parent / (out_file.stem + ".png")
    cbar_file = out_file.parent / (out_file.stem + "_cbar.png")

    kml_file = out_file.parent / (out_file.stem + ".kml")
    array2kml(arr, kml_file, bounds, img_kwargs, cbar_kwargs, verbose=False)
    with zipfile.ZipFile(out_file, "w") as kmz:
        kmz.write(kml_file, kml_file.name)
        kmz.write(img_file, img_file.name)
        kmz.write(cbar_file, cbar_file.name)
    if not keep_kml:
        img_file.unlink()
        cbar_file.unlink()
        kml_file.unlink()
    if verbose:
        info = f"write kmz file to {out_file}"
        logger.info(info)


def array2tiled_kmz(
    arr: np.ndarray,
    out_file: PathLike,
    bounds: tuple[float, float, float, float] | BoundingBox,
    img_kwargs: dict | None = None,
    cbar_kwargs: dict | None = None,
    tile_size: int = 256,
    min_lod_pixels: int = 128,
    render_scale: float = 1.0,
    verbose: bool = True,
) -> None:
    """Write an array into a tiled KMZ tile pyramid.

    Parameters
    ----------
    arr : np.ndarray
        Array to export. The primary supported use case is a 2D scalar raster.
    out_file : PathLike
        Output KMZ path.
    bounds : tuple[float, float, float, float] | BoundingBox
        Bounds of the image in ``(west, south, east, north)`` order in WGS84.
    img_kwargs : dict | None, optional
        Keyword arguments forwarded to :func:`matplotlib.axes.Axes.imshow`.
    cbar_kwargs : dict | None, optional
        Keyword arguments forwarded to :func:`save_colorbar`, excluding
        ``out_file`` and ``mappable``.
    tile_size : int, optional
        Maximum tile size in pixels for each tile image.
    min_lod_pixels : int, optional
        Minimum screen-space threshold used by child ``NetworkLink`` regions.
    render_scale : float, optional
        Scale factor applied to the rendered image size before tiling.
    verbose : bool, optional
        Whether to log the output file path.

    Raises
    ------
    ValueError
        If any tiling parameter is invalid.

    Notes
    -----
    This exporter writes a self-contained KMZ that uses a KML SuperOverlay
    structure internally. The source data are still limited by the resolution
    of ``arr``.

    """
    if tile_size <= 0:
        msg = f"tile_size should be positive, but got {tile_size}"
        logger.error(msg)
        raise ValueError(msg)
    if min_lod_pixels < 0:
        msg = f"min_lod_pixels should be non-negative, but got {min_lod_pixels}"
        logger.error(msg)
        raise ValueError(msg)

    bounds = _normalize_kml_bounds(bounds)
    img_kwargs_norm = _normalize_image_kwargs(img_kwargs, interpolation="nearest")
    cbar_kwargs_norm = {} if cbar_kwargs is None else dict(cbar_kwargs)

    fig, mappable, rgba = _render_array_to_rgba(
        arr,
        img_kwargs=img_kwargs_norm,
        render_scale=render_scale,
    )
    try:
        colorbar_png = _render_colorbar_to_png_bytes(mappable, **cbar_kwargs_norm)
    finally:
        plt.close(fig)

    root = _build_tiled_kmz_tree(
        image_width=rgba.shape[1],
        image_height=rgba.shape[0],
        bounds=bounds,
        tile_size=tile_size,
    )

    out_file = Path(out_file)
    if out_file.suffix != ".kmz":
        out_file = out_file.parent / (out_file.stem + ".kmz")

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr(
            "doc.kml",
            _tiled_kmz_root_kml(
                root,
                bounds,
                colorbar_path="legend/colorbar.png",
            ),
        )
        kmz.writestr("legend/colorbar.png", colorbar_png)

        for node in _iter_tiled_kmz_nodes(root):
            kmz.writestr(
                node.png_path,
                _rgba_to_png_bytes(
                    _tile_image_for_node(rgba, node, tile_size=tile_size)
                ),
            )
            kmz.writestr(
                node.kml_path,
                _tiled_kmz_tile_kml(
                    node,
                    tile_size=tile_size,
                    min_lod_pixels=min_lod_pixels,
                ),
            )

    if verbose:
        info = f"write tiled kmz file to {out_file}"
        logger.info(info)
