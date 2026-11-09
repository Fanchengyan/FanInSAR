"""Stack work-directory layout conventions.

Track-grouped workspaces stage acquisition SAFEs under
``tracks/{trackKey}/acquisitions/`` where ``trackKey`` is the
:func:`~faninsar.missions.s1.types.track_group_key` label (``"88A"``);
shared inputs flatten to the work-directory root:

    <work_dir>/
        tracks/<trackKey>/acquisitions/<YYYYMMDD>/<parentBox>/<name>.SAFE
        tmp/s1-transfer/<canvasStem>/<fingerprint>/  CDSE range scratch
        orbits/                                     precise orbit files
        aux_cal/                                    Sentinel-1 AUX_CAL files
        dem.tif                                     WGS84 ellipsoidal DEM
        dem.json                                    DEM sidecar identity
        dem-cache/                                  source DEM tile cache
        inputs/                                     legacy staging root

``<YYYYMMDD>`` groups one observing day.  ``<parentBox>`` is the full ESA
parent-scene stem, not a recomputed short id: complementary ``-IW1-VV``
canvases of one product share that folder, and a later along-track frame
is a new sibling.  The folders are a bookshelf; burst pairing still uses
identity.

``inputs/`` remains the fallback for acquisitions staged before the
track-aware layout existed; :func:`acquisition_staging_dir` resolves the
legacy root when no ``track_key`` is given so older stacks reopen.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_SWATH_CANVAS_SUFFIX = re.compile(
    r"[-_](IW[1-3]|EW[1-5])-(VV|VH|HH|HV)$",
    re.IGNORECASE,
)
_DATE_FIELD = re.compile(r"(?<![0-9])(\d{8})(?:T\d{6})?")
_UNDATED_KEY = "undated"

TRACKS_DIRNAME = "tracks"
"""Workspace subtree holding per-track staged acquisitions."""

ACQUISITIONS_DIRNAME = "acquisitions"
"""Per-track SAFE staging subdirectory name."""

LEGACY_INPUTS_DIRNAME = "inputs"
"""Pre-track staging root, still honored on reopen."""

ORBITS_DIRNAME = "orbits"
"""Top-level precise-orbit directory (was ``inputs/orbits``)."""

AUX_CAL_DIRNAME = "aux_cal"
"""Top-level Sentinel-1 AUX_CAL directory (EAP phase correction)."""

DEM_RASTER_NAME = "dem.tif"
"""Top-level WGS84 ellipsoidal DEM raster (was ``dem/stack.tif``)."""

DEM_SIDECAR_NAME = "dem.json"
"""Top-level DEM sidecar record (was ``dem/stack.json``)."""

DEM_WGS84_NAME = "dem_wgs84.tif"
"""Legacy geographic sibling; new workspaces do not create this file."""

DEM_CACHE_DIRNAME = "dem-cache"
"""Top-level source DEM tile cache."""

TMP_DIRNAME = "tmp"
"""Workspace-local scratch (never the system ``/tmp``)."""

S1_TRANSFER_DIRNAME = "s1-transfer"
"""CDSE range-fetch scratch under ``<work_dir>/tmp/s1-transfer``."""

_PATHISH_KEYS = frozenset({"path", "cache_dir", "dem_cache_dir"})
"""JSON keys whose string values are filesystem paths in ``stack.json``."""

_PATHISH_MAP_KEYS = frozenset({"orbit_paths", "aux_cal_paths"})
"""JSON maps whose values are per-date filesystem paths in ``stack.json``."""

_LEGACY_DEM_RASTER = ("dem", "stack.tif")
"""Pre-flattening Stack-grid DEM raster location."""

_LEGACY_DEM_SIDECAR = ("dem", "stack.json")
"""Pre-flattening DEM sidecar location."""

_LEGACY_DEM_WGS84 = ("dem", "stack_wgs84.tif")
"""Pre-flattening geographic DEM location."""


def _product_stem(name: str | Path) -> str:
    """Return a SAFE/zip name with container suffixes stripped."""
    text = Path(name).name
    lower = text.lower()
    if lower.endswith(".safe.zip"):
        return text[: -len(".SAFE.zip")]
    if lower.endswith(".zip"):
        return text[: -len(".zip")]
    if lower.endswith(".safe"):
        return text[: -len(".SAFE")]
    return text


def parent_box_stem(name: str | Path) -> str:
    """Return the parent-scene folder for one staged SAFE or zip name.

    Canvas names ``<parent>-IW1-VV.SAFE`` (or ``_IW1-VV``) share ``<parent>``.
    A name without that suffix is its own box.  The stem is the ESA product
    id as written, not a shortened or recomputed label, so a later frame
    is a new folder rather than a rename.

    Parameters
    ----------
    name : str or pathlib.Path
        File or directory name, optionally with ``.SAFE`` / ``.zip``.

    Returns
    -------
    str
        Parent-box directory name (no ``.SAFE`` suffix).

    """
    text = _product_stem(name)
    match = _SWATH_CANVAS_SUFFIX.search(text)
    return text[: match.start()] if match else text


def acquisition_date_key(name: str | Path) -> str:
    """Return the ``YYYYMMDD`` bookshelf folder encoded in a product name.

    Reads the first eight-digit date already present in the ESA stem
    (``20260221T053526`` or ``_20260221_``).  Names without a date land
    under ``undated``.  The date is not recomputed from burst sensing
    times, so adding another frame never rewrites existing folders.

    """
    match = _DATE_FIELD.search(_product_stem(name))
    return match.group(1) if match else _UNDATED_KEY


def s1_transfer_root(work_dir: str | Path, safe_name: str | Path) -> Path:
    """Return ``<work_dir>/tmp/s1-transfer/<canvasStem>``.

    Incomplete CDSE canvases keep ``<fingerprint>/*.part`` here.  Completing
    a canvas deletes that fingerprint directory so the published SAFE holds
    only ``coverage.json``, not transfer scratch.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Stack work directory.
    safe_name : str or pathlib.Path
        Canvas SAFE directory name (``…-IW1-VV.SAFE``).

    Returns
    -------
    pathlib.Path
        Per-canvas transfer root (not created).

    """
    return Path(work_dir) / TMP_DIRNAME / S1_TRANSFER_DIRNAME / _product_stem(safe_name)


def staged_acquisition_path(staging_root: str | Path, name: str | Path) -> Path:
    """Return ``<staging_root>/<YYYYMMDD>/<parentBox>/<name>``.

    Parameters
    ----------
    staging_root : str or pathlib.Path
        ``tracks/<key>/acquisitions`` or a legacy ``inputs/`` directory.
    name : str or pathlib.Path
        SAFE or zip file name, including its suffix.

    Returns
    -------
    pathlib.Path
        Nested destination (not created).

    """
    filename = Path(name).name
    parent = parent_box_stem(filename)
    return Path(staging_root) / acquisition_date_key(parent) / parent / filename


def _is_date_key(name: str) -> bool:
    """Return whether ``name`` is a date (or undated) bookshelf folder."""
    return name == _UNDATED_KEY or (len(name) == 8 and name.isdigit())


def _is_staged_product(path: Path) -> bool:
    """Return whether ``path`` is a SAFE directory or zip at this layout."""
    return path.name.lower().endswith((".safe", ".zip"))


def _relocate(source: Path, dest: Path) -> bool:
    """Move ``source`` to ``dest`` when the destination is free."""
    if source.resolve() == dest.resolve() or dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    source.rename(dest)
    return True


def migrate_acquisition_staging(staging_root: str | Path) -> list[tuple[Path, Path]]:
    """Move leftover SAFEs and parent-box folders into ``<date>/<parent>/``.

    Accepts a flat ``*.SAFE`` tree, a previous ``<parentBox>/*.SAFE`` tree,
    and the current date-grouped layout.  SAFE contents are not rewritten.

    Parameters
    ----------
    staging_root : str or pathlib.Path
        One track's ``acquisitions`` directory (or legacy ``inputs/``).

    Returns
    -------
    list of tuple of pathlib.Path
        ``(old, new)`` pairs that were renamed.

    """
    root = Path(staging_root)
    if not root.is_dir():
        return []
    moved: list[tuple[Path, Path]] = []

    def _move_product(source: Path) -> None:
        dest = staged_acquisition_path(root, source.name)
        if _relocate(source, dest):
            moved.append((source, dest))

    for child in sorted(root.iterdir()):
        if _is_staged_product(child):
            _move_product(child)
            continue
        if not child.is_dir() or _is_date_key(child.name):
            continue
        dest_box = root / acquisition_date_key(child.name) / child.name
        if child.resolve() == dest_box.resolve():
            continue
        products = [item for item in child.iterdir() if _is_staged_product(item)]
        if _relocate(child, dest_box):
            moved.extend((item, dest_box / item.name) for item in products)
            continue
        dest_box.mkdir(parents=True, exist_ok=True)
        for item in products:
            dest = dest_box / item.name
            if _relocate(item, dest):
                moved.append((item, dest))
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()

    for date_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not _is_date_key(date_dir.name):
            continue
        for child in sorted(date_dir.iterdir()):
            if _is_staged_product(child):
                _move_product(child)
    return moved


def nested_acquisition_moves(staging_root: str | Path) -> list[tuple[Path, Path]]:
    """Infer old-layout paths from an already date-nested staging tree.

    After :func:`migrate_acquisition_staging` the previous flat and
    parent-box paths no longer exist, but identity documents may still
    record them.  Walking the nested tree recovers those aliases.
    """
    root = Path(staging_root)
    if not root.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir() or not _is_date_key(date_dir.name):
            continue
        for box in sorted(date_dir.iterdir()):
            if not box.is_dir() or _is_staged_product(box):
                continue
            for child in sorted(box.iterdir()):
                if not _is_staged_product(child):
                    continue
                if parent_box_stem(child.name) != box.name:
                    continue
                pairs.append((root / child.name, child))
                pairs.append((root / box.name / child.name, child))
    return pairs


def identity_workspace_root(identity_dir: str | Path) -> Path:
    """Return the workspace root that owns one identity document.

    ``tracks/<key>/stack.json`` is anchored at the track directory; its
    workspace is two parents up.  A root ``stack.json`` is the workspace.
    """
    path = Path(identity_dir)
    if path.parent.name == TRACKS_DIRNAME:
        return path.parent.parent
    return path


def encode_identity_path(path: str | Path, identity_dir: str | Path) -> str:
    """Record ``path`` relative to the identity file's directory.

    Paths inside the workspace become POSIX relatives (``acquisitions/...``,
    ``../../dem.tif``).  Paths outside the workspace stay absolute so a
    shared SAFE or orbit store does not break when the workspace is copied.
    """
    raw = Path(path)
    identity = Path(identity_dir).resolve()
    workspace = identity_workspace_root(identity).resolve()
    resolved = raw.resolve() if raw.is_absolute() else (identity / raw).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return raw.as_posix() if raw.is_absolute() else Path(path).as_posix()
    return Path(os.path.relpath(resolved, identity)).as_posix()


def decode_identity_path(recorded: str, identity_dir: str | Path) -> Path:
    """Resolve one recorded identity path against the identity directory.

    Absolute strings are returned unchanged so legacy ``stack.json`` files
    keep working.  Relative strings join the identity directory.
    """
    path = Path(recorded)
    if path.is_absolute():
        return path
    return (Path(identity_dir) / path).resolve()


def relativize_identity_paths(
    document: dict[str, object], identity_dir: str | Path
) -> None:
    """Rewrite filesystem path fields in ``document`` to identity-relative form."""
    _rewrite_pathish_fields(
        document,
        lambda recorded: encode_identity_path(recorded, identity_dir),
    )


def resolve_identity_paths(
    document: dict[str, object], identity_dir: str | Path
) -> None:
    """Rewrite filesystem path fields in ``document`` to absolute paths."""
    _rewrite_pathish_fields(
        document,
        lambda recorded: str(decode_identity_path(recorded, identity_dir)),
    )


def _rewrite_pathish_fields(node: object, transform: object) -> None:
    """Apply ``transform`` to recorded filesystem path strings in a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _PATHISH_KEYS and isinstance(value, str) and value:
                node[key] = transform(value)  # type: ignore[operator]
            elif key in _PATHISH_MAP_KEYS and isinstance(value, dict):
                for date, entry in value.items():
                    if isinstance(entry, str) and entry:
                        value[date] = transform(entry)  # type: ignore[operator]
                    else:
                        _rewrite_pathish_fields(entry, transform)
            else:
                _rewrite_pathish_fields(value, transform)
    elif isinstance(node, list):
        for item in node:
            _rewrite_pathish_fields(item, transform)


def rewrite_identity_paths(
    identity_path: str | Path,
    moved: Sequence[tuple[Path, Path]],
) -> int:
    """Rewrite recorded ``path`` entries after an acquisition rename.

    Parameters
    ----------
    identity_path : str or pathlib.Path
        ``stack.json`` (root or ``tracks/<key>/stack.json``).
    moved : sequence of tuple of pathlib.Path
        ``(old, new)`` pairs from :func:`migrate_acquisition_staging`.

    Returns
    -------
    int
        Number of path strings rewritten.

    """
    path = Path(identity_path)
    if not path.is_file() or not moved:
        return 0
    identity_dir = path.parent
    mapping = {old.resolve(): new.resolve() for old, new in moved}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    count = _rewrite_moved_path_fields(document, mapping, identity_dir)
    if count:
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return count


def migrate_workspace_acquisitions(work_dir: str | Path) -> list[tuple[Path, Path]]:
    """Nest flat acquisitions under every track of one workspace.

    Moves leftover ``*.SAFE`` trees into ``<YYYYMMDD>/<parentBox>/`` and
    rewrites matching ``path`` fields in each track ``stack.json`` and the
    root identity.  Legacy ``inputs/`` is migrated the same way.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Stack work directory.

    Returns
    -------
    list of tuple of pathlib.Path
        All ``(old, new)`` renames.

    """
    root = Path(work_dir)
    moved: list[tuple[Path, Path]] = []
    identities = [root / "stack.json"]
    staging_roots: list[Path] = [root / LEGACY_INPUTS_DIRNAME]
    tracks = root / TRACKS_DIRNAME
    if tracks.is_dir():
        for track_dir in sorted(tracks.iterdir()):
            if not track_dir.is_dir():
                continue
            staging_roots.append(track_dir / ACQUISITIONS_DIRNAME)
            identities.append(track_dir / "stack.json")
    for staging in staging_roots:
        moved.extend(migrate_acquisition_staging(staging))
    mapping = list(moved)
    for staging in staging_roots:
        mapping.extend(nested_acquisition_moves(staging))
    for identity in identities:
        rewrite_identity_paths(identity, mapping)
    return moved


def _rewrite_moved_path_fields(
    node: object,
    mapping: dict[Path, Path],
    identity_dir: Path,
) -> int:
    """Replace matching path strings after an acquisition rename."""
    count = 0
    if isinstance(node, dict):
        value = node.get("path")
        if isinstance(value, str) and value:
            recorded = decode_identity_path(value, identity_dir)
            lookup = recorded.resolve()
            if lookup in mapping:
                node["path"] = encode_identity_path(mapping[lookup], identity_dir)
                count += 1
        count += sum(
            _rewrite_moved_path_fields(item, mapping, identity_dir)
            for item in node.values()
        )
    elif isinstance(node, list):
        count += sum(
            _rewrite_moved_path_fields(item, mapping, identity_dir) for item in node
        )
    return count


def track_acquisitions_dir(work_dir: str | Path, track_key: str) -> Path:
    """Return ``<work_dir>/tracks/<track_key>/acquisitions``.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Stack work directory.
    track_key : str
        ``"<relOrbit><A|D>"`` grouping label.

    Returns
    -------
    pathlib.Path
        Per-track SAFE staging directory (not created).

    """
    return Path(work_dir) / TRACKS_DIRNAME / track_key / ACQUISITIONS_DIRNAME


def acquisition_staging_dir(
    work_dir: str | Path,
    track_key: str | None = None,
) -> Path:
    """Return the SAFE staging root for one acquisition group.

    ``track_key`` selects the track-grouped layout; ``None`` keeps the
    legacy ``<work_dir>/inputs`` root so older workspaces reopen.
    """
    if track_key:
        return track_acquisitions_dir(work_dir, track_key)
    return Path(work_dir) / LEGACY_INPUTS_DIRNAME


def orbits_dir(work_dir: str | Path) -> Path:
    """Return the precise-orbit directory (``<work_dir>/orbits``)."""
    return Path(work_dir) / ORBITS_DIRNAME


def workspace_orbits_dir(work_dir: str | Path) -> Path:
    """Return the shared orbit cache for a Stack work directory.

    Track-scoped workspaces keep acquisitions under ``tracks/<key>/`` but
    covering EOF files stay at the workspace root so every track reuses
    the same s1-orbits cache.
    """
    path = Path(work_dir)
    if path.parent.name == TRACKS_DIRNAME:
        path = path.parent.parent
    return orbits_dir(path)


def resolve_orbit_cache_dir(work_dir: str | Path) -> Path:
    """Prefer the shared workspace EOF cache, then track-local, then legacy."""
    root = Path(work_dir)
    shared = workspace_orbits_dir(root)
    local = orbits_dir(root)
    legacy = legacy_orbits_dir(root)
    for candidate in (shared, local, legacy):
        if not candidate.is_dir():
            continue
        if any(
            path.is_file() and path.suffix.upper() == ".EOF"
            for path in candidate.iterdir()
        ):
            return candidate
    return shared


def aux_cal_dir(work_dir: str | Path) -> Path:
    """Return the AUX_CAL directory (``<work_dir>/aux_cal``)."""
    return Path(work_dir) / AUX_CAL_DIRNAME


def legacy_orbits_dir(work_dir: str | Path) -> Path:
    """Return the pre-flattening orbit directory (``inputs/orbits``)."""
    return Path(work_dir) / LEGACY_INPUTS_DIRNAME / ORBITS_DIRNAME


def dem_raster_path(work_dir: str | Path) -> Path:
    """Return the durable WGS84 ellipsoidal DEM raster path.

    Prefers the flat ``dem.tif``; falls back to ``dem/stack.tif`` when a
    legacy workspace holds only the nested file. The canonical raster is
    geographic and independent of the Stack product grid.
    """
    root = Path(work_dir)
    flat = root / DEM_RASTER_NAME
    legacy = root.joinpath(*_LEGACY_DEM_RASTER)
    if flat.is_file() or not legacy.is_file():
        return flat
    return legacy


def dem_sidecar_path(work_dir: str | Path) -> Path:
    """Return the DEM identity sidecar (``dem.json`` or legacy ``dem/stack.json``)."""
    root = Path(work_dir)
    flat = root / DEM_SIDECAR_NAME
    legacy = root.joinpath(*_LEGACY_DEM_SIDECAR)
    if flat.is_file() or not legacy.is_file():
        return flat
    return legacy


def dem_wgs84_path(work_dir: str | Path) -> Path:
    """Return a legacy geographic sibling path if one already exists.

    New workspaces persist only ``dem.tif``. This helper locates an older
    ``dem_wgs84.tif`` or ``dem/stack_wgs84.tif`` without creating it.
    """
    root = Path(work_dir)
    flat = root / DEM_WGS84_NAME
    legacy = root.joinpath(*_LEGACY_DEM_WGS84)
    if flat.is_file() or not legacy.is_file():
        return flat
    return legacy


def dem_cache_dir(work_dir: str | Path) -> Path:
    """Return the source DEM tile cache (``<work_dir>/dem-cache``)."""
    return Path(work_dir) / DEM_CACHE_DIRNAME


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "AUX_CAL_DIRNAME",
    "DEM_CACHE_DIRNAME",
    "DEM_RASTER_NAME",
    "DEM_SIDECAR_NAME",
    "DEM_WGS84_NAME",
    "LEGACY_INPUTS_DIRNAME",
    "ORBITS_DIRNAME",
    "S1_TRANSFER_DIRNAME",
    "TMP_DIRNAME",
    "TRACKS_DIRNAME",
    "acquisition_date_key",
    "acquisition_staging_dir",
    "aux_cal_dir",
    "dem_cache_dir",
    "dem_raster_path",
    "dem_sidecar_path",
    "dem_wgs84_path",
    "legacy_orbits_dir",
    "migrate_acquisition_staging",
    "migrate_workspace_acquisitions",
    "nested_acquisition_moves",
    "orbits_dir",
    "parent_box_stem",
    "resolve_orbit_cache_dir",
    "rewrite_identity_paths",
    "s1_transfer_root",
    "staged_acquisition_path",
    "track_acquisitions_dir",
    "workspace_orbits_dir",
]
