"""COG export seam: one committed layer or product to one Cloud Optimized GeoTIFF.

Export is format conversion only. It never geocodes, resamples, or reprojects:
a caller who needs a different grid or a geographic frame uses
:meth:`faninsar.stack.session.Stack.geocode` instead.

Reuse is governed by an identity digest rather than a filename. A filename is a
human label; two requests that would write different bytes must never share one
file, and a request whose bytes cannot be proven identical is refused rather
than assumed stale. Only layers whose values are owned by a committed product
are exported, because that product is what makes the identity provable.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.io.raster import write_cog
from faninsar.logging import reject, setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

logger = setup_logger(__name__)

__all__ = [
    "COG_FORMAT_VERSION",
    "CogExportItem",
    "CogExportReport",
    "ExportRequest",
    "export_cog",
    "export_identity",
    "export_label",
    "preflight_destinations",
    "resolved_writer_profile",
]

COG_FORMAT_VERSION = 1
"""Schema marker for the COG writer's output layout.

Reuse does not depend on this constant alone: the identity digest also covers
the resolved writer profile, so a changed block size or compression invalidates
reuse without a manual bump. This marker exists for coarse schema changes.
"""

_DOMAIN_BY_KIND: dict[str, str] = {
    "scene": "radar",
    "ifg": "radar",
    "unwrap": "radar",
    "complex_ion": "radar",
    "unwrap_ion": "radar",
}

_ION_ASSETS: frozenset[str] = frozenset({"complex_ion", "unwrap_ion", "ion_corrected"})


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """One export target resolved from a layer handle.

    Attributes
    ----------
    label : str
        Human-facing label written into the filename, derived from the owning
        product kind and the committed asset key. It is not the storage key.
    asset_key : str
        The committed asset key the export reads.
    source_write_id : str
        Write id of the product that owns the exported bytes.
    contributing_write_ids : tuple of (str, str)
        Every other product whose commit contributed to the exported bytes,
        as ``(role, write_id)`` pairs. A merge-on-read layer depends on its
        layout product and on each participating burst product.
    composition : str
        Normalized read mode, which decides overlap handling.
    selection : tuple of (str, str)
        Normalized view selection: the selected units, track, or pair.
    domain : str
        ``"radar"`` or ``"geo"``.
    looks : tuple of (int, int)
        Range-first look factors.
    date_token : str
        ``YYYYMMDD`` or ``<ref>-<sec>``.

    """

    label: str
    asset_key: str
    source_write_id: str
    domain: str
    looks: tuple[int, int]
    date_token: str
    contributing_write_ids: tuple[tuple[str, str], ...] = ()
    composition: str = "identity"
    selection: tuple[tuple[str, str], ...] = ()
    default_root: str = "cog"

    @property
    def stem(self) -> str:
        """Return the filename stem without the ``.cog.tif`` suffix."""
        parts = [self.label]
        if self.looks != (1, 1):
            parts.append(f"ml{self.looks[0]}x{self.looks[1]}")
        parts.append(self.date_token)
        parts.append(self.domain)
        return "_".join(part for part in parts if part)

    @property
    def filename(self) -> str:
        """Return the export filename."""
        return f"{self.stem}.cog.tif"


@dataclass(frozen=True, slots=True)
class CogExportItem:
    """One written COG and its outcome.

    Attributes
    ----------
    target : str
        Stable label of the request that produced this file.
    path : pathlib.Path or None
        Written file, or ``None`` when the export failed.
    status : {"written", "reused", "failed"}
        Outcome.
    identity : str
        Export identity digest, empty for a failed export.
    reason : str
        Failure detail, empty otherwise.

    """

    target: str
    path: Path | None
    status: Literal["written", "reused", "failed"]
    identity: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CogExportReport:
    """Requested, written, reused, and failed exports.

    The vocabulary mirrors :class:`~faninsar.stack.preparation.PreparationReport`
    without importing it: this is an operation receipt, not a scientific
    product.

    Attributes
    ----------
    requested : tuple of str
        Targets the caller asked for, in request order.
    items : tuple of CogExportItem
        One entry per target.
    root : pathlib.Path or None
        Export root, when the request named a directory.

    """

    requested: tuple[str, ...]
    items: tuple[CogExportItem, ...] = ()
    root: Path | None = None

    @property
    def ready(self) -> tuple[str, ...]:
        """Return targets that produced a file, written or reused."""
        return tuple(
            item.target for item in self.items if item.status in {"written", "reused"}
        )

    @property
    def failed(self) -> tuple[CogExportItem, ...]:
        """Return the items that failed."""
        return tuple(item for item in self.items if item.status == "failed")

    @property
    def reused(self) -> tuple[str, ...]:
        """Return targets served from an existing matching file."""
        return tuple(item.target for item in self.items if item.status == "reused")

    @property
    def outcome(self) -> Literal["complete", "partial", "failed"]:
        """Return completeness independently of usable subset readiness."""
        if not self.ready:
            return "failed"
        return "partial" if self.failed else "complete"

    @property
    def complete(self) -> bool:
        """Return whether every requested target produced a file."""
        return self.outcome == "complete"


def export_identity(
    request: ExportRequest,
    *,
    mask: str | bool,
    writer: Mapping[str, object],
) -> str:
    """Return a stable digest over every input that decides output bytes.

    The digest covers the owning product, every contributing product, the
    committed asset key, the normalized view selection, the read composition
    mode, the mask policy, and the **resolved** writer profile. A requested
    block size that GDAL clamps to its own grid does not change the digest,
    because the resolved profile is what a reader sees.

    Parameters
    ----------
    request : ExportRequest
        Resolved export target.
    mask : str or bool
        Mask policy: ``True``/``"derived"`` for finiteness, ``"explicit"`` for a
        caller-supplied support plane, ``False``/``"none"`` for no mask. An
        explicit plane is recorded as a policy, not as its contents, because
        the plane is derivable from the same committed product.
    writer : mapping
        Resolved writer profile returned by
        :func:`faninsar.io.raster.write_cog`, or the profile the request would
        resolve to when deciding reuse before writing.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    """
    payload = {
        "asset_key": request.asset_key,
        "composition": request.composition,
        "contributing": [list(pair) for pair in request.contributing_write_ids],
        "format_version": COG_FORMAT_VERSION,
        "mask": mask if isinstance(mask, str) else bool(mask),
        "selection": [list(pair) for pair in request.selection],
        "source_write_id": request.source_write_id,
        "writer": {str(key): _json_safe(value) for key, value in writer.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolved_writer_profile(
    *,
    dtype: object,
    nodata: float | None = None,
    georeferenced: bool = False,
    compress: str = "DEFLATE",
    blocksize: int = 512,
    overview_resampling: str = "average",
) -> dict[str, object]:
    """Predict the writer profile GDAL will produce for a COG.

    Identity has to be known before the file exists and again when a later
    request asks whether it may reuse that file. Both need one deterministic
    answer, and rewriting tags on a published COG is not an option: opening a
    finalized COG for update raises a layout-break error, and forcing it
    through invalidates the file (verified). So this function normalizes the
    request to what the driver actually emits.

    The COG driver clamps block size to its own grid — a requested 256 and 512
    both produce 512x512 blocks (verified) — and stores overview resampling in
    the ``IMAGE_STRUCTURE`` domain rather than in ``profile``.

    Parameters
    ----------
    dtype : str or numpy.dtype
        Written sample dtype.
    nodata : float or None, optional
        NoData value written into the file. It is part of the identity because
        a file written with one value serves different bytes to a reader than
        the same samples written with another.
    georeferenced : bool, optional
        Whether a grid was supplied. An unreferenced file and a georeferenced
        one with identical samples are not interchangeable.
    compress : str, optional
        Requested compression.
    blocksize : int, optional
        Requested tile size.
    overview_resampling : str, optional
        Requested overview kernel.

    Returns
    -------
    dict
        Normalized profile used for the identity digest.

    """
    return {
        "driver": "COG",
        "dtype": str(np.dtype(dtype)),
        "compress": compress.casefold(),
        "blocksize": int(blocksize),
        "nodata": None if nodata is None else str(float(nodata)),
        "georeferenced": bool(georeferenced),
        "overview_resampling": overview_resampling.casefold(),
    }


def _json_safe(value: object) -> object:
    """Convert one value into a JSON-stable form."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def export_label(asset_key: str, *, product_kind: str) -> str:
    """Return the export label for one committed asset.

    The label distinguishes the ion-corrected branch from the ordinary one
    while keeping the committed asset key untouched: a ``complex_ion`` product
    stores its payload under ``complex``, and renaming that key would break
    every reader that resolves it against the committed item.

    Parameters
    ----------
    asset_key : str
        Committed asset key.
    product_kind : str
        Kind of the product that owns the asset.

    Returns
    -------
    str
        Export label, safe as one filename component.

    """
    if product_kind in _ION_ASSETS and not asset_key.endswith("_ion"):
        return f"{asset_key}_ion"
    return asset_key


def _filesystem_key(path: Path) -> tuple[str, int] | str:
    """Return a key that is equal for two paths naming one filesystem object.

    ``Path.resolve()`` is not sufficient on a case-insensitive volume: two
    differently-cased names resolve differently yet name the same file. For an
    existing target the inode identifies it; for a target that does not exist
    yet, the case-folded resolved path is the best available proxy, which is
    what this workspace's filesystem will collapse.

    """
    try:
        stat = path.stat()
    except OSError:
        return ("absent", str(path.resolve()).casefold())
    return ("present", int(stat.st_dev), int(stat.st_ino))


def preflight_destinations(root: Path, destinations: Sequence[Path]) -> None:
    """Reject unsafe or colliding destinations before any write.

    Three conditions are checked in one pass, because each is cheap and each
    must hold before the first payload or Item is written:

    - every destination stays inside ``root`` once resolved;
    - no destination path component is a symlink, so a planted link cannot
      redirect the write outside the export tree;
    - no two destinations name one filesystem object, including case-variant
      names and hardlinks.

    Parameters
    ----------
    root : pathlib.Path
        Export root the destinations must stay under.
    destinations : sequence of pathlib.Path
        Every path the request will write.

    Raises
    ------
    ValueError
        If a destination escapes the root, a component is a symlink, or two
        destinations alias.

    """
    resolved_root = root.resolve()
    seen: dict[object, Path] = {}
    for destination in destinations:
        for parent in [*destination.parents, destination]:
            if parent.is_symlink():
                reject(
                    f"destination path component is a symlink: {parent}",
                    ValueError,
                )
        resolved = destination.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            reject(
                f"destination escapes the export root: {destination}",
                ValueError,
            )
        key = _filesystem_key(destination)
        if key in seen:
            reject(
                "two export targets name one filesystem object: "
                f"{seen[key]} and {destination}",
                ValueError,
            )
        seen[key] = destination


def _tags_for(request: ExportRequest) -> dict[str, object]:
    """Return the allowlisted dataset tags for one export."""
    tags: dict[str, object] = {
        "ASSET": request.asset_key,
        "DOMAIN": request.domain,
        "COG_FORMAT_VERSION": COG_FORMAT_VERSION,
        "SOURCE_WRITE_ID": request.source_write_id,
        "COMPOSITION": request.composition,
    }
    date_token = request.date_token.replace("-", "_")
    if "-" in request.date_token:
        tags["PAIR"] = request.date_token
    else:
        tags["DATE"] = date_token
    if request.looks != (1, 1):
        tags["RANGE_LOOKS"] = request.looks[0]
        tags["AZIMUTH_LOOKS"] = request.looks[1]
    for role, write_id in request.contributing_write_ids:
        tags[f"INPUT_{role.upper()}"] = write_id
    return tags


def export_cog(
    request: ExportRequest,
    values: np.ndarray,
    *,
    path: str | Path | None = None,
    grid: object | None = None,
    mask: np.ndarray | bool = True,
    nodata: float | None = None,
    overwrite: bool = False,
) -> CogExportItem:
    """Write one resolved export request as a single COG.

    Reuse is decided by the identity digest recorded in the file, so a caller
    that changes the source, the selection, the mask policy, or an effective
    writer setting gets a recompute rather than another request's bytes.

    Parameters
    ----------
    request : ExportRequest
        Resolved export target.
    values : numpy.ndarray
        2-D samples to write.
    path : str or pathlib.Path, optional
        Destination file, or a directory to receive the generated filename.
    grid : RasterGridSpec, optional
        Destination grid; ``None`` writes an unreferenced COG.
    mask : bool or numpy.ndarray, optional
        ``True`` derives validity from finiteness; a boolean ``(height, width)``
        plane embeds that support directly, which is what a caller holding the
        committed mask passes; ``False`` embeds none.
    nodata : float, optional
        NoData value written alongside the mask.
    overwrite : bool, optional
        Replace an existing destination whose identity differs.

    Returns
    -------
    CogExportItem
        Outcome of this one export.

    """
    if path is None:
        # The documented default: ``<product>/cog/<generated name>``. A bare
        # filename here would resolve against the process working directory,
        # which is not the product tree.
        target = Path(request.default_root) / request.filename
    else:
        target = Path(path)
        if target.suffix.lower() not in {".tif", ".tiff"}:
            target = target / request.filename

    root = target.parent
    try:
        preflight_destinations(root, [target])
    except ValueError as error:
        return CogExportItem(
            target=request.stem,
            path=None,
            status="failed",
            reason=str(error),
        )

    tags = _tags_for(request)
    mask_policy = "derived" if mask is True else "none" if mask is False else "explicit"
    writer_profile = resolved_writer_profile(
        dtype=np.asarray(values).dtype,
        nodata=nodata,
        georeferenced=grid is not None,
    )
    identity = export_identity(request, mask=mask_policy, writer=writer_profile)
    tags["EXPORT_IDENTITY"] = identity

    if target.exists() and not overwrite:
        recorded = _recorded_identity(target)
        if recorded == identity:
            return CogExportItem(
                target=request.stem,
                path=target,
                status="reused",
                identity=identity,
            )
        return CogExportItem(
            target=request.stem,
            path=None,
            status="failed",
            reason=(
                f"destination {target} was written by a different export "
                "request; pass overwrite=True to replace it"
            ),
        )

    try:
        written, _resolved = write_cog(
            values,
            target,
            grid,  # type: ignore[arg-type]
            nodata=nodata,
            mask=mask,
            band_descriptions=[request.label],
            tags=tags,
            overwrite=True,
        )
    except Exception as error:
        logger.warning("COG export failed for %s: %s", request.stem, error)
        return CogExportItem(
            target=request.stem,
            path=None,
            status="failed",
            identity=identity,
            reason=str(error),
        )

    return CogExportItem(
        target=request.stem,
        path=written,
        status="written",
        identity=identity,
    )


def _recorded_identity(path: Path) -> str | None:
    """Return the export identity recorded in an existing COG, if any."""
    import rasterio

    try:
        with rasterio.open(path) as dataset:
            return dataset.tags().get("EXPORT_IDENTITY")
    except Exception:
        return None


_VIRTUAL_DERIVATIONS: dict[str, tuple[str, ...]] = {
    "wrapped_phase": ("complex", "complex_ion"),
    "amplitude": ("complex", "complex_ion"),
    "wrapped": ("complex", "complex_ion"),
    "intensity": ("complex", "complex_ion"),
    "phase": ("complex", "complex_ion"),
}


def _is_virtual_for(kind: str, asset_key: str, assets: Mapping[str, object]) -> bool:
    """Return whether a layer is derived from a committed asset this product has.

    A scene derives ``slc``/``amplitude``/``intensity``/``phase`` from its unit
    payloads, which are named after burst tags rather than after the layer, so
    any committed unit asset makes those derivations reachable.
    """
    sources = _VIRTUAL_DERIVATIONS.get(asset_key, ())
    if any(source in assets for source in sources):
        return True
    if kind == "scene":
        return asset_key in {"slc", "amplitude", "intensity", "phase"} and any(
            not str(name).endswith(".valid") for name in assets
        )
    return False


def request_from_product(
    *,
    product: object,
    asset_key: str,
    label: str | None = None,
    composition: str = "identity",
    selection: Iterable[tuple[str, str]] = (),
    contributing: Iterable[tuple[str, str]] = (),
    origin: tuple[int, int] = (0, 0),
    looks: tuple[int, int] = (1, 1),
) -> ExportRequest:
    """Build an export request from a committed product and one asset key.

    Every identity field is read from the product's own commit, so a later
    export cannot claim a provenance it does not have. A product that does not
    record a date, pair, or domain is refused rather than given a guessed
    token.

    Parameters
    ----------
    product : faninsar.io.storage.product.Product
        Committed product that owns the bytes.
    asset_key : str
        Asset key to export.
    label : str, optional
        Export label; defaults to the branch-aware label for this product kind.
    composition : str, optional
        Normalized read mode that decides overlap handling.
    selection : iterable of (str, str), optional
        Normalized view selection.
    contributing : iterable of (str, str), optional
        ``(role, write_id)`` for every other product whose commit contributes.
    origin : tuple of int, optional
        ``(row, col)`` crop origin, recorded because it shifts the frame.
    looks : tuple of int, optional
        Range-first look factors.

    Returns
    -------
    ExportRequest
        Fully identified export target.

    Raises
    ------
    ValueError
        If the product records no date, pair, or domain, or the asset is absent.

    """
    properties = dict(getattr(product, "properties", {}) or {})
    kind = str(properties.get("faninsar:kind", ""))
    assets = dict(getattr(product, "assets", {}) or {})
    # A virtual layer (``wrapped_phase``, ``amplitude``) is derived from a
    # committed complex asset and stores no payload of its own. It is a valid
    # export target, but only when the derivation source is committed.
    if asset_key not in assets and not _is_virtual_for(kind, asset_key, assets):
        reject(
            f"product {getattr(product, 'path', '?')} has no asset "
            f"{asset_key!r}; available: {sorted(assets)}",
            ValueError,
        )

    domain = properties.get("faninsar:domain")
    if domain not in {"radar", "geo"}:
        domain = _DOMAIN_BY_KIND.get(kind)
    if domain not in {"radar", "geo"}:
        reject(f"product kind {kind!r} records no radar/geo domain", ValueError)

    raw_date = properties.get("faninsar:date")
    raw_pair = properties.get("faninsar:pair")
    if isinstance(raw_pair, (list, tuple)) and len(raw_pair) == 2:
        date_token = f"{raw_pair[0]}-{raw_pair[1]}"
    elif isinstance(raw_date, str) and raw_date:
        date_token = raw_date
    else:
        reject(f"product kind {kind!r} records no date or pair", ValueError)

    write_id = str(properties.get("faninsar:write_id", ""))
    if not write_id:
        reject("product records no write id", ValueError)

    product_path = getattr(product, "path", None)
    default_root = str(Path(product_path) / "cog") if product_path else "cog"

    return ExportRequest(
        label=label or export_label(asset_key, product_kind=kind),
        asset_key=asset_key,
        source_write_id=write_id,
        domain=domain,
        looks=tuple(int(value) for value in looks),  # type: ignore[arg-type]
        date_token=date_token,
        contributing_write_ids=tuple(contributing),
        composition=composition
        if origin == (0, 0)
        else f"{composition}@{origin[0]},{origin[1]}",
        selection=tuple(selection),
        default_root=default_root,
    )
