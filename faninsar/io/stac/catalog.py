"""Stage and atomically replace the portable derived STAC catalog (P0089).

Persistence is deliberately separate from scientific commit truth: it writes
only ``<work_dir>/catalog.json`` and ``<work_dir>/stac/``, protects unowned
artifacts, stages a complete generation, swaps the tree, and replaces the root
catalog last so a partial tree is never advertised.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from faninsar.io.stac.render import (
    LinkTargets,
    render_collection,
    render_family_catalog,
    render_item,
    render_root_catalog,
)
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from faninsar.io.stac.index import StacIndex

__all__ = [
    "CatalogPersistenceReport",
    "write_portable_catalog",
]

logger = setup_logger(__name__)

_GENERATED_BY = "faninsar.stac_serve"
_MARKER_NAME = ".faninsar-stac.json"
_MAX_OWNER_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CatalogPersistenceReport:
    """Outcome of one optional portable-catalog persistence attempt."""

    requested: bool
    persisted: bool
    generation: str
    catalog_path: Path | None
    error: str | None


def write_portable_catalog(
    index: StacIndex,
    *,
    work_dir: str | Path,
) -> CatalogPersistenceReport:
    """Write one complete portable catalog generation.

    The derived tree is ``<work_dir>/catalog.json`` and
    ``<work_dir>/stac/``. A complete generation is staged under
    ``<work_dir>/.stac-staging-<generation>/``, the ``stac/`` directory is
    swapped, and ``catalog.json`` is replaced last. Unowned or symlinked
    artifacts are never overwritten.

    Parameters
    ----------
    index : StacIndex
        In-memory generation to persist.
    work_dir : str or path-like
        Stack workspace root.

    Returns
    -------
    CatalogPersistenceReport
        Whether the write succeeded, the generation, and any refusal or error.

    """
    root = Path(work_dir).expanduser().resolve()
    catalog_path = root / "catalog.json"
    if not root.is_dir():
        return _failure(index, f"workspace is not a directory: {root}")
    try:
        _protect_unowned(root, catalog_path)
    except _UnownedArtifactError as error:
        return _failure(index, str(error))

    staging = root / f".stac-staging-{index.generation}"
    previous = root / f".stac-previous-{index.generation}"
    try:
        _clean_stale(root)
        _remove(staging)
        _remove(previous)
        staging.mkdir()
        root_document = _write_views(index, root, staging)
        _fsync_tree(staging)
        _swap(
            root=root,
            staging=staging,
            previous=previous,
            catalog_path=catalog_path,
            root_document=root_document,
        )
    except Exception as error:
        _remove(staging)
        _remove(previous)
        logger.warning("portable STAC catalog was not persisted: %s", error)
        return _failure(index, f"portable catalog write failed: {error}")
    return CatalogPersistenceReport(
        requested=True,
        persisted=True,
        generation=index.generation,
        catalog_path=catalog_path,
        error=None,
    )


class _UnownedArtifactError(Exception):
    """Internal refusal to replace a catalog artifact this service does not own."""


def _reject_unowned(description: str, path: Path) -> NoReturn:
    """Raise an ownership refusal with a deterministic diagnostic."""
    message = f"refusing to overwrite unowned {description}: {path}"
    raise _UnownedArtifactError(message)


def _failure(index: StacIndex, error: str) -> CatalogPersistenceReport:
    """Return a non-persisted report."""
    return CatalogPersistenceReport(
        requested=True,
        persisted=False,
        generation=index.generation,
        catalog_path=None,
        error=error,
    )


def _protect_unowned(root: Path, catalog_path: Path) -> None:
    """Refuse to replace an unowned root catalog or ``stac`` directory."""
    if catalog_path.exists() or catalog_path.is_symlink():
        if catalog_path.is_symlink() or not catalog_path.is_file():
            _reject_unowned("catalog artifact", catalog_path)
        if not _has_owner_marker(catalog_path):
            _reject_unowned("catalog artifact", catalog_path)
    stac_dir = root / "stac"
    if stac_dir.exists() or stac_dir.is_symlink():
        if stac_dir.is_symlink() or not stac_dir.is_dir():
            _reject_unowned("stac directory", stac_dir)
        marker = stac_dir / _MARKER_NAME
        if marker.is_symlink() or not marker.is_file():
            _reject_unowned("stac directory", stac_dir)
        if not _has_owner_marker(marker):
            _reject_unowned("stac directory", stac_dir)


def _has_owner_marker(path: Path) -> bool:
    """Return whether a bounded JSON document carries our ownership marker."""
    try:
        if path.stat().st_size > _MAX_OWNER_BYTES:
            return False
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return (
        isinstance(document, dict)
        and document.get("faninsar:generated_by") == _GENERATED_BY
    )


def _clean_stale(root: Path) -> None:
    """Remove our own leftover staging, retirement, and root temp files."""
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        message = f"workspace cannot be listed: {error}"
        raise OSError(message) from error
    for entry in entries:
        name = entry.name
        if name.startswith((".stac-staging-", ".stac-previous-", ".stac-catalog-")):
            _remove(entry)


def _write_views(index: StacIndex, root: Path, staging: Path) -> dict[str, Any]:
    """Write every non-root view into staging and return the root document."""
    root_path = root / "catalog.json"
    targets = LinkTargets.for_disk(work_dir=root, view_path=root_path)
    root_document = render_root_catalog(index, targets)
    root_document["faninsar:generated_by"] = _GENERATED_BY
    _write_document(staging / _MARKER_NAME, _marker_document(index))
    for family in index.families():
        path = root / "stac" / "catalogs" / f"{family}.json"
        document = render_family_catalog(
            index,
            family,
            LinkTargets.for_disk(work_dir=root, view_path=path),
        )
        _write_document(_staged(staging, root, path), document)
    for collection in index.collections:
        path = root / "stac" / "collections" / f"{collection.collection_id}.json"
        document = render_collection(
            collection,
            index,
            LinkTargets.for_disk(work_dir=root, view_path=path),
        )
        _write_document(_staged(staging, root, path), document)
    for record in index.records:
        path = root / "stac" / "items" / f"{record.public_id}.json"
        document = render_item(
            record,
            LinkTargets.for_disk(work_dir=root, view_path=path),
        )
        _write_document(_staged(staging, root, path), document)
    return root_document


def _marker_document(index: StacIndex) -> dict[str, Any]:
    """Return the ownership marker document for one generation."""
    return {
        "faninsar:generated_by": _GENERATED_BY,
        "generation": index.generation,
    }


def _staged(staging: Path, root: Path, final_path: Path) -> Path:
    """Map one final ``stac`` view path to its staging path."""
    return staging / final_path.relative_to(root / "stac")


def _write_document(path: Path, document: dict[str, Any]) -> None:
    """Write one strict-JSON document and flush it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    """Flush every staged file and directory, deepest directory last."""
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            message = f"staging tree contains a symlink: {path}"
            raise OSError(message)
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            message = f"staging tree contains a non-file: {path}"
            raise OSError(message)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    directories.append(root)
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _swap(
    *,
    root: Path,
    staging: Path,
    previous: Path,
    catalog_path: Path,
    root_document: dict[str, Any],
) -> None:
    """Swap the complete staged tree and replace the root catalog last."""
    stac_dir = root / "stac"
    previous_catalog: bytes | None = None
    if stac_dir.exists() or stac_dir.is_symlink():
        stac_dir.rename(previous)
    installed = False
    catalog_replaced = False
    try:
        staging.rename(stac_dir)
        installed = True
        if catalog_path.is_file() and not catalog_path.is_symlink():
            previous_catalog = catalog_path.read_bytes()
        generation = staging.name.removeprefix(".stac-staging-")
        temporary = root / f".stac-catalog-{generation}.json"
        _remove(temporary)
        _write_document(temporary, root_document)
        temporary.replace(catalog_path)
        catalog_replaced = True
        _fsync_directory(root)
    except BaseException:
        if catalog_replaced:
            _restore_catalog(root, catalog_path, previous_catalog)
        if installed:
            _remove(stac_dir)
        if previous.exists():
            previous.rename(stac_dir)
        raise
    try:
        _remove(previous)
    except OSError as error:
        logger.warning("could not remove previous derived catalog tree: %s", error)


def _restore_catalog(
    root: Path,
    catalog_path: Path,
    previous: bytes | None,
) -> None:
    """Best-effort restore of the previous root catalog bytes."""
    if previous is None:
        catalog_path.unlink(missing_ok=True)
        return
    temporary = root / ".stac-catalog-restore.json"
    with temporary.open("wb") as stream:
        stream.write(previous)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(catalog_path)


def _remove(path: Path) -> None:
    """Remove one path without following symlinks; ignore absence."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
