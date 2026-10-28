"""One storage rule for every Stack product (PROPOSAL-0059).

A product is a directory holding its payloads and one STAC ``item.json``.
Payloads are written into a sibling staging directory, ``item.json`` is
written last, and the directory is renamed into place, so a present
``item.json`` is the only completion signal. Every commit stamps a fresh
``faninsar:write_id``; dependents record the write ids of the upstream
products they consumed and become stale when an upstream is replaced.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from faninsar.io.storage._zarr_stac import digest_path, read_zarr_array
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np

logger = setup_logger(__name__)

ITEM_NAME = "item.json"
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

__all__ = [
    "ITEM_NAME",
    "Product",
    "ProductWriter",
    "commit_product",
    "discover_products",
    "new_write_id",
    "open_product",
    "remove_product",
    "verify_product",
]


def new_write_id() -> str:
    """Return a fresh 26-character ULID used as a product write id."""
    timestamp = int(time.time() * 1000) << 80
    value = timestamp | int.from_bytes(secrets.token_bytes(10), "big")
    characters = []
    for _ in range(26):
        characters.append(_ULID_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(characters))


def is_write_id(value: object) -> bool:
    """Return whether ``value`` has the shape of a product write id (ULID)."""
    return (
        isinstance(value, str)
        and len(value) == 26
        and all(character in _ULID_ALPHABET for character in value)
    )


def _basename(href: object) -> str:
    """Return ``href`` when it is a plain relative basename, else reject."""
    if not isinstance(href, str):
        reject_invalid_state("product asset href must be a basename string")
    candidate = Path(href)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != href
        or href in {"", ".", ".."}
    ):
        reject_invalid_state(f"product asset href must be a basename: {href!r}")
    return href


def _strict_json(payload: Mapping[str, Any]) -> str:
    """Serialize product metadata, rejecting NaN, infinity, and unknown types."""
    try:
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"product metadata is not strict JSON: {error}")


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry so a rename survives a crash."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_sibling(path: Path) -> None:
    """Delete one kernel-generated sibling without following symlinks."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        path.unlink()
        return
    shutil.rmtree(path)


def _stale_siblings(target: Path) -> list[Path]:
    """Return staging and aside siblings this kernel may have left for ``target``."""
    parent = target.parent
    if not parent.is_dir():
        return []
    prefix_tmp = f".{target.name}.tmp"
    prefix_aside = f".{target.name}.aside-"
    return [
        entry
        for entry in parent.iterdir()
        if entry.name == prefix_tmp or entry.name.startswith(prefix_aside)
    ]


@dataclass(frozen=True, slots=True)
class Product:
    """A committed product: its directory and parsed ``item.json``."""

    path: Path
    item: Mapping[str, Any] = field(repr=False)

    @property
    def properties(self) -> Mapping[str, Any]:
        """Return the STAC ``properties`` mapping."""
        return self.item["properties"]

    @property
    def assets(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the STAC ``assets`` mapping."""
        return self.item["assets"]

    @property
    def kind(self) -> str:
        """Return the product kind (``faninsar:kind``)."""
        return str(self.properties["faninsar:kind"])

    @property
    def write_id(self) -> str:
        """Return the write id stamped at commit."""
        return str(self.properties["faninsar:write_id"])

    @property
    def inputs(self) -> dict[str, str]:
        """Return the upstream write ids this product consumed."""
        return dict(self.properties.get("faninsar:inputs", {}))

    def asset_path(self, name: str) -> Path:
        """Return the absolute path of one asset."""
        if name not in self.assets:
            reject_invalid_state(f"product {self.path} has no asset {name!r}")
        return self.path / _basename(self.assets[name]["href"])

    def read(self, name: str) -> np.ndarray:
        """Read one Zarr asset."""
        return read_zarr_array(self.asset_path(name))

    def stale_inputs(self, current: Mapping[str, Product | str]) -> list[str]:
        """Return input names whose recorded write id differs from ``current``.

        Parameters
        ----------
        current : mapping of str to Product or str
            Current upstream products (or their write ids) keyed like
            :attr:`inputs`. A recorded input missing from ``current`` is stale.

        """
        recorded = self.inputs
        stale = []
        for name, write_id in sorted(recorded.items()):
            upstream = current.get(name)
            upstream_id = (
                upstream.write_id if isinstance(upstream, Product) else upstream
            )
            if upstream_id != write_id:
                stale.append(name)
        return stale


class ProductWriter:
    """Stage one product's payloads, then publish it with :meth:`commit`.

    Payloads are written into :attr:`staging` (a sibling of the target).
    Nothing is visible until :meth:`commit` writes ``item.json`` and renames
    the staging directory into place. Used as a context manager, an
    exception before commit removes the staging directory.

    Parameters
    ----------
    target : path-like
        Final product directory.
    kind : str
        Product kind recorded as ``faninsar:kind``.
    overwrite : bool, optional
        Replace an existing product on commit. Otherwise an existing product
        is rejected immediately.

    """

    def __init__(
        self, target: str | Path, *, kind: str, overwrite: bool = False
    ) -> None:
        """Create the staging directory for ``target``."""
        self.target = Path(target)
        self.kind = kind
        self.overwrite = overwrite
        self.target.parent.mkdir(parents=True, exist_ok=True)
        for stale in _stale_siblings(self.target):
            _remove_sibling(stale)
        if self.target.exists() and not overwrite:
            reject_invalid_state(
                f"product already exists: {self.target}; pass overwrite=True"
            )
        self.staging = self.target.parent / f".{self.target.name}.tmp"
        self.staging.mkdir()
        self._assets: dict[str, dict[str, Any]] = {}
        self._closed = False

    def add_asset(self, name: str, href: str, **fields: Any) -> Path:
        """Register one asset already written (or about to be) under staging."""
        if name in self._assets:
            reject_invalid_state(f"product asset {name!r} registered twice")
        self._assets[str(name)] = {"href": _basename(href), **fields}
        return self.staging / href

    def commit(
        self,
        properties: Mapping[str, Any],
        *,
        inputs: Mapping[str, Product | str] | None = None,
    ) -> Product:
        """Write ``item.json`` and publish the product atomically."""
        if self._closed:
            reject_invalid_state("product writer is already closed")
        try:
            write_id = new_write_id()
            assets = {}
            for name, asset in self._assets.items():
                entry = dict(asset)
                entry.setdefault("roles", ["data"])
                entry["file:checksum"] = digest_path(self.staging / entry["href"])
                assets[name] = entry
            item = {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": f"{self.kind}-{self.target.name}",
                "geometry": None,
                "bbox": None,
                "properties": {
                    **dict(properties),
                    "datetime": None,
                    "faninsar:kind": self.kind,
                    "faninsar:write_id": write_id,
                    "faninsar:inputs": {
                        str(name): (
                            value.write_id if isinstance(value, Product) else str(value)
                        )
                        for name, value in (inputs or {}).items()
                    },
                },
                "assets": assets,
                "links": [],
            }
            text = _strict_json(item)
            with (self.staging / ITEM_NAME).open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            aside = None
            if self.target.exists():
                if not self.overwrite:
                    reject_invalid_state(f"product already exists: {self.target}")
                aside = self.target.parent / f".{self.target.name}.aside-{write_id}"
                self.target.rename(aside)
            self.staging.rename(self.target)
            _fsync_directory(self.target.parent)
            if aside is not None:
                _remove_sibling(aside)
        except BaseException:
            self.abort()
            raise
        self._closed = True
        return Product(path=self.target, item=item)

    def abort(self) -> None:
        """Remove the staging directory without publishing anything."""
        self._closed = True
        if self.staging.exists() or self.staging.is_symlink():
            _remove_sibling(self.staging)

    def __enter__(self) -> Self:
        """Return the writer."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Abort an uncommitted writer when the block raised."""
        if not self._closed:
            self.abort()


def commit_product(
    target: str | Path,
    *,
    kind: str,
    properties: Mapping[str, Any],
    write_payloads: Callable[[Path], Mapping[str, Mapping[str, Any]]],
    inputs: Mapping[str, Product | str] | None = None,
    overwrite: bool = False,
) -> Product:
    """Write one product atomically and return it.

    Parameters
    ----------
    target : path-like
        Final product directory.
    kind : str
        Product kind recorded as ``faninsar:kind``.
    properties : mapping
        Additional STAC properties; custom keys use the ``faninsar:`` prefix.
    write_payloads : callable
        Receives the staging directory, writes every payload into it, and
        returns the asset mapping (name to a dict with at least ``href``).
    inputs : mapping, optional
        Upstream products (or write ids) this product consumed.
    overwrite : bool, optional
        Replace an existing product. Otherwise an existing one is an error.

    Returns
    -------
    Product
        The committed product.

    Raises
    ------
    InvalidProcessingStateError
        If the product exists without ``overwrite``, an href is not a plain
        basename, or the metadata is not strict JSON.

    """
    with ProductWriter(target, kind=kind, overwrite=overwrite) as writer:
        for name, asset in write_payloads(writer.staging).items():
            fields = {key: value for key, value in asset.items() if key != "href"}
            writer.add_asset(str(name), asset.get("href"), **fields)  # type: ignore[arg-type]
        return writer.commit(properties, inputs=inputs)


def open_product(path: str | Path, *, kind: str | None = None) -> Product:
    """Open a committed product without re-hashing its payloads.

    Parameters
    ----------
    path : path-like
        Product directory.
    kind : str, optional
        Required product kind.

    Raises
    ------
    InvalidProcessingStateError
        If ``item.json`` is missing or invalid, the kind differs, or an asset
        is missing, symlinked, or outside the product directory.

    """
    root = Path(path)
    item_path = root / ITEM_NAME
    if not item_path.is_file() or item_path.is_symlink() or root.is_symlink():
        reject_invalid_state(f"product item.json is missing: {root}")
    try:
        item = json.loads(item_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        reject_invalid_state(f"product item.json cannot be read: {error}")
    properties = item.get("properties") if isinstance(item, dict) else None
    if (
        not isinstance(properties, dict)
        or not isinstance(item.get("assets"), dict)
        or "faninsar:kind" not in properties
        or "faninsar:write_id" not in properties
    ):
        reject_invalid_state(
            f"{item_path} is not a FanInSAR product item; rebuild this work "
            "directory with the current FanInSAR version"
        )
    if kind is not None and properties["faninsar:kind"] != kind:
        reject_invalid_state(
            f"product {root} has kind {properties['faninsar:kind']!r}, "
            f"expected {kind!r}"
        )
    for name, asset in item["assets"].items():
        asset_path = root / _basename(asset.get("href"))
        if asset_path.is_symlink():
            reject_invalid_state(f"product asset {name!r} is a symbolic link")
        if not asset_path.exists():
            reject_invalid_state(f"product asset {name!r} is missing: {asset_path}")
    return Product(path=root, item=item)


def discover_products(root: str | Path, *, kind: str) -> list[Product]:
    """Return committed child products of ``root`` with the given kind.

    Staging and aside directories (names starting with ``.``) are ignored.
    """
    parent = Path(root)
    if not parent.is_dir():
        return []
    found = []
    for candidate in sorted(parent.iterdir()):
        name = candidate.name
        if name.startswith(".") or candidate.is_symlink() or not candidate.is_dir():
            continue
        if not (candidate / ITEM_NAME).is_file():
            continue
        product = open_product(candidate)
        if product.kind == kind:
            found.append(product)
    return found


def remove_product(path: str | Path) -> None:
    """Delete a product directory without following symlinks."""
    _remove_sibling(Path(path))


def verify_product(path: str | Path) -> list[str]:
    """Return asset names whose bytes differ from their write-time checksum."""
    product = open_product(path)
    return [
        name
        for name, asset in sorted(product.assets.items())
        if digest_path(product.asset_path(name)) != asset.get("file:checksum")
    ]
