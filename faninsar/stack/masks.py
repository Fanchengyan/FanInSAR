"""Persisted mask products and the ``stack.masks`` facade (PROPOSAL-0067).

Every :class:`~faninsar.stack.mask_plan.MaskPlan` registry entry can be
materialized on the target grid of each stage that references it — and on
user request — and committed as a flat product
``work_dir/masks/<name>/<domain>/item.json + mask.zarr`` through the
PROPOSAL-0059 kernel. The product is bound to ``(mask_name,
target_grid_identity)``; radar-domain products additionally record the
Reference ``write_id`` they were projected against, and every product
records its ``mask_recipe_identity``.

``stack.masks`` is a read-only :class:`~collections.abc.Mapping` facade:
``stack.masks["water"]`` and ``stack.masks.water`` return a
:class:`StackMaskView` exposing ``.radar`` / ``.geo`` / ``.roi`` /
``.sel(domain=)`` reads over the committed rasters. ``combined`` is a
reserved virtual name: it is never persisted and resolves, per stage, to
the union of masks that stage actually referenced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal, NoReturn

import numpy as np

from faninsar.logging import setup_logger
from faninsar.stack.mask_plan import (
    RESERVED_MASK_NAMES,
    MaskDefinition,
    StageName,
)

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.io.storage.product import Product
    from faninsar.processing.masking.mask import GridSpec, Mask, RasterMask
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)

MASK_KIND = "mask"
COMBINED_NAME = "combined"
_MASK_ASSET = "mask.zarr"
MaskDomain = Literal["radar", "geo"]

#: User-facing domain selectors that fold into the two persisted domains.
_DOMAIN_ALIASES: dict[str, MaskDomain] = {
    "radar": "radar",
    "geo": "geo",
    "roi": "geo",
}

#: Stages whose mask target is always the geographic grid.
_GEO_STAGES: frozenset[StageName] = frozenset({"roi", "ionosphere"})

_SAFE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


def _fail(message: str, error_type: type[Exception] = ValueError) -> NoReturn:
    """Log and raise a mask-facade error."""
    from faninsar.logging import reject

    reject(message, error_type, logger=logger)


def validate_mask_product_name(name: object) -> str:
    """Return ``name`` when it is one safe product-directory component."""
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or any(character not in _SAFE_NAME_CHARS for character in name)
    ):
        _fail(f"mask product name must be one safe path component; got {name!r}")
    return name


def mask_product_root(stack: Stack, name: str, domain: str) -> Path:
    """Return the product directory of one committed mask."""
    return (
        stack.config.work_dir
        / "masks"
        / validate_mask_product_name(name)
        / validate_mask_product_name(domain)
    )


def mask_recipe_identity(definition: MaskDefinition | Mask) -> str:
    """Return the canonical recipe/materialization identity of one registry value."""
    from faninsar.processing.masking.mask import Mask

    if isinstance(definition, Mask):
        return definition.identity
    payload = {"name": definition.name, **definition.identity_payload}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def grid_identity(target: GridSpec) -> str:
    """Return the canonical identity of one mask target grid.

    This is the same digest ``Stack._materialize_stage_mask`` uses for its
    in-memory cache key, so a product bound to ``(name, grid_identity)``
    covers exactly one materialization result.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "crs": str(target.crs),
                "transform": tuple(float(value) for value in target.transform),
                "shape": tuple(int(value) for value in target.shape),
                "validity": (
                    None
                    if target.validity is None
                    else np.asarray(target.validity, dtype=bool).tobytes().hex()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def stage_domain(stack: Stack, stage: StageName) -> MaskDomain:
    """Return the persisted domain of one stage's mask target grid.

    ``roi`` folds into ``geo`` (the ROI recipe's vector source is rasterized
    onto the geographic grid); ``ionosphere`` is ``geo``;
    ``interferogram``/``unwrap`` follow ``config.coregistration_grid``.
    """
    if stage in _GEO_STAGES:
        return "geo"
    return "geo" if stack.config.coregistration_grid == "geo" else "radar"


def is_water_definition(definition: object) -> bool:
    """Return whether one registry value is a water-recipe mask."""
    from faninsar.processing.masking.mask import Mask

    if isinstance(definition, Mask):
        return getattr(definition, "_water_recipe", None) is not None
    return getattr(definition, "kind", None) == "water"


def water_mask_names(stack: Stack) -> tuple[str, ...]:
    """Return the registered water-recipe mask names, in registry order."""
    return tuple(
        name
        for name, definition in stack.config.mask_plan.masks.items()
        if is_water_definition(definition)
    )


def _reference_write_id(stack: Stack) -> str | None:
    """Return the committed Reference scene write id, when one exists."""
    from faninsar.processing.errors import InvalidProcessingStateError

    try:
        reference = stack.reference_scene
    except InvalidProcessingStateError:
        return None
    return None if reference is None else reference.write_id


def _product_inputs(stack: Stack, domain: str) -> dict[str, str]:
    """Return upstream bindings for one mask product.

    Radar-domain products bind ``reference=<ref write_id>`` so a replaced
    Reference invalidates the projection; geo-domain products bind nothing.
    """
    inputs: dict[str, str] = {}
    if domain == "radar":
        reference_id = _reference_write_id(stack)
        if reference_id is not None:
            inputs["reference"] = reference_id
    return inputs


def _product_properties(
    stack: Stack,
    *,
    name: str,
    domain: str,
    target: GridSpec,
    recipe_identity: str,
) -> dict[str, object]:
    """Return the strict-JSON property block of one mask product."""
    properties: dict[str, object] = {
        "faninsar:mask_name": name,
        "faninsar:mask_domain": domain,
        "faninsar:target_grid_identity": grid_identity(target),
        "faninsar:mask_recipe_identity": recipe_identity,
        "faninsar:grid": {
            "crs": str(target.crs),
            "transform": [float(value) for value in target.transform],
            "shape": [int(target.height), int(target.width)],
        },
    }
    if domain == "radar":
        reference_id = _reference_write_id(stack)
        if reference_id is not None:
            properties["faninsar:reference_write_id"] = reference_id
    return properties


def _matches_product(
    product: Product,
    stack: Stack,
    *,
    name: str,
    domain: str,
    target: GridSpec,
    recipe_identity: str,
) -> bool:
    """Return whether a committed product still identifies this materialization."""
    properties = product.properties
    if (
        properties.get("faninsar:mask_name") != name
        or properties.get("faninsar:mask_domain") != domain
        or properties.get("faninsar:target_grid_identity") != grid_identity(target)
        or properties.get("faninsar:mask_recipe_identity") != recipe_identity
    ):
        return False
    if domain == "radar":
        recorded = properties.get("faninsar:reference_write_id")
        current = _reference_write_id(stack)
        return recorded == current
    return True


def _write_mask_payloads(mask: RasterMask):  # noqa: ANN202 - closure factory
    """Return the ``write_payloads`` callback persisting one uint8 plane."""
    from faninsar.io.storage._zarr_stac import write_zarr_array

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        write_zarr_array(staging / _MASK_ASSET, np.asarray(mask.data, dtype=np.uint8))
        return {
            "mask": {
                "href": _MASK_ASSET,
                "type": "application/vnd+zarr",
            }
        }

    return write_payloads


def persist_mask_product(
    stack: Stack,
    *,
    name: str,
    definition: MaskDefinition | Mask,
    domain: MaskDomain,
    target: GridSpec,
    mask: RasterMask,
) -> Product:
    """Commit one materialized mask under ``masks/<name>/<domain>/``.

    An already-committed product whose identity still matches is reused; a
    mismatched one is replaced (``overwrite=True`` keeps the flat layout
    free of duplicates).

    """
    from faninsar.io.storage.product import open_product, publish_product

    recipe_identity = mask_recipe_identity(definition)
    target_dir = mask_product_root(stack, name, domain)
    if (target_dir / "item.json").is_file():
        product = open_product(target_dir, kind=MASK_KIND)
        if _matches_product(
            product,
            stack,
            name=name,
            domain=domain,
            target=target,
            recipe_identity=recipe_identity,
        ):
            return product
        logger.info("replacing stale mask product %s (identity changed)", target_dir)
    return publish_product(
        target_dir,
        kind=MASK_KIND,
        properties=_product_properties(
            stack,
            name=name,
            domain=domain,
            target=target,
            recipe_identity=recipe_identity,
        ),
        write_payloads=_write_mask_payloads(mask),
        inputs=_product_inputs(stack, domain),
        overwrite=True,
    )


def read_mask_product(stack: Stack, *, name: str, domain: MaskDomain) -> np.ndarray:
    """Read the committed mask plane for ``(name, domain)``.

    Raises
    ------
    KeyError
        If no product is committed for that pair.

    """
    from faninsar.io.storage.product import open_product

    target_dir = mask_product_root(stack, name, domain)
    if not (target_dir / "item.json").is_file():
        message = (
            f"mask {name!r} has no committed {domain!r}-domain product; "
            f"read stack.masks[{name!r}].{domain} first to materialize it"
        )
        raise KeyError(message)
    product = open_product(target_dir, kind=MASK_KIND)
    return np.asarray(product.read("mask"), dtype=np.uint8)


def committed_mask_product(
    stack: Stack, *, name: str, domain: MaskDomain
) -> Product | None:
    """Return the committed product for ``(name, domain)``, if present."""
    from faninsar.io.storage.product import open_product

    target_dir = mask_product_root(stack, name, domain)
    if not (target_dir / "item.json").is_file():
        return None
    return open_product(target_dir, kind=MASK_KIND)


def discover_mask_products(stack: Stack) -> list[Product]:
    """Return every committed mask product under ``work_dir/masks/``.

    The flat layout keeps one directory per product; ``<name>/<domain>``
    levels are plain grouping directories without their own ``item.json``,
    so the two levels are enumerated explicitly.
    """
    root = stack.config.work_dir / "masks"
    found: list[Product] = []
    if not root.is_dir():
        return found
    from faninsar.io.storage.product import ITEM_NAME, open_product

    for name_dir in sorted(root.iterdir()):
        if (
            name_dir.name.startswith(".")
            or name_dir.is_symlink()
            or not name_dir.is_dir()
        ):
            continue
        for domain_dir in sorted(name_dir.iterdir()):
            if (
                domain_dir.name.startswith(".")
                or domain_dir.is_symlink()
                or not domain_dir.is_dir()
                or not (domain_dir / ITEM_NAME).is_file()
            ):
                continue
            product = open_product(domain_dir)
            if product.kind == MASK_KIND:
                found.append(product)
    return found


class StackMaskView:
    """Read view over the persisted products of one registered mask.

    Parameters
    ----------
    stack : Stack
        Owning session.
    name : str
        Registry name in ``stack.config.mask_plan``.

    """

    def __init__(self, stack: Stack, name: str) -> None:
        """Bind the view; no materialization happens here."""
        self._stack = stack
        self._name = name

    @property
    def name(self) -> str:
        """Return the registry name of this mask."""
        return self._name

    def _definition(self) -> MaskDefinition | Mask:
        registry = self._stack.config.mask_plan.masks
        if self._name not in registry:
            message = f"unknown mask {self._name!r}"
            raise KeyError(message)
        return registry[self._name]

    @property
    def definition(self) -> MaskDefinition | Mask:
        """Return the registry recipe (or concrete Mask) for this mask."""
        return self._definition()

    def sel(self, domain: str) -> np.ndarray:
        """Return the committed plane for an explicit domain.

        Parameters
        ----------
        domain : {"radar", "geo", "roi"}
            ``"roi"`` folds into the geo-domain raster. Any other value is
            rejected; the view never silently returns the other domain's
            raster.

        """
        if not isinstance(domain, str) or domain not in _DOMAIN_ALIASES:
            _fail(
                f"unknown mask domain {domain!r}; expected one of "
                f"{sorted(_DOMAIN_ALIASES)}"
            )
        resolved = _DOMAIN_ALIASES[domain]
        return self._stack._mask_domain_plane(
            self._name, self._definition(), domain=resolved
        )

    @property
    def radar(self) -> np.ndarray:
        """Return the radar-domain raster (``coregistration_grid='radar'``)."""
        return self.sel("radar")

    @property
    def geo(self) -> np.ndarray:
        """Return the geographic-domain raster (EPSG:4326 target)."""
        return self.sel("geo")

    @property
    def roi(self) -> np.ndarray:
        """Alias for :attr:`geo` — the ROI recipe rasterizes geographically."""
        return self.sel("roi")

    def __repr__(self) -> str:
        """Return a concise view summary."""
        return f"StackMaskView(name={self._name!r})"


class CombinedMaskView:
    """Virtual per-stage union view; never persisted (decision 3).

    ``.radar`` unions the masks the interferogram/unwrap stage actually
    referenced on the radar coregistration grid; ``.geo`` unions what the
    geo-domain stages referenced. ``.sel(domain=)`` accepts an optional
    ``stage`` selector for explicit per-stage reads.
    """

    def __init__(self, stack: Stack) -> None:
        """Bind the virtual view."""
        self._stack = stack

    def sel(self, domain: str, *, stage: StageName | None = None) -> np.ndarray:
        """Return the union of masks one stage referenced on its grid.

        Parameters
        ----------
        domain : {"radar", "geo", "roi"}
            Domain selector; ``"roi"`` selects the ``roi`` stage.
        stage : {"roi", "interferogram", "unwrap", "ionosphere"}, optional
            Explicit stage. When omitted, ``radar`` uses the interferogram
            stage references and ``geo`` the ionosphere references (``roi``
            forces the ``roi`` stage).

        """
        if not isinstance(domain, str) or domain not in _DOMAIN_ALIASES:
            _fail(
                f"unknown mask domain {domain!r}; expected one of "
                f"{sorted(_DOMAIN_ALIASES)}"
            )
        if stage is not None and stage not in StageName.__args__:
            _fail(f"unknown mask stage {stage!r}")
        if stage is None:
            stage = (
                "roi"
                if domain == "roi"
                else (
                    "interferogram"
                    if _DOMAIN_ALIASES[domain] == "radar"
                    else "ionosphere"
                )
            )
        resolved = _DOMAIN_ALIASES[domain]
        return self._stack._combined_mask_plane(
            stage, self._stack.config.mask_plan.references(stage), domain=resolved
        )

    @property
    def radar(self) -> np.ndarray:
        """Return the union the interferogram/unwrap stage referenced."""
        return self.sel("radar")

    @property
    def geo(self) -> np.ndarray:
        """Return the union the geo-domain stages referenced."""
        return self.sel("geo")

    @property
    def roi(self) -> np.ndarray:
        """Return the union the ROI stage referenced (geo domain)."""
        return self.sel("roi")

    def __repr__(self) -> str:
        """Return a concise view summary."""
        return "CombinedMaskView(virtual)"


class StackMasks(Mapping[str, StackMaskView]):
    """Read-only Mapping facade over the Stack's registered masks.

    Iteration and ``len`` cover the MaskPlan registry only — ``combined``
    is reserved and virtual, reachable by name or attribute but never part
    of the registry keys.

    """

    def __init__(self, stack: Stack) -> None:
        """Bind the facade to one Stack session."""
        self._stack = stack

    def __getitem__(self, name: str) -> StackMaskView | CombinedMaskView:
        """Return the per-mask view, or the virtual ``combined`` union."""
        if name in RESERVED_MASK_NAMES:
            return CombinedMaskView(self._stack)
        registry = self._stack.config.mask_plan.masks
        if name not in registry:
            raise KeyError(name)
        return StackMaskView(self._stack, name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered mask names."""
        return iter(self._stack.config.mask_plan.masks)

    def __len__(self) -> int:
        """Return the number of registered masks."""
        return len(self._stack.config.mask_plan.masks)

    def __getattr__(self, name: str) -> StackMaskView | CombinedMaskView:
        """Return the view bound to ``stack.masks.<name>``."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __repr__(self) -> str:
        """Return the registered mask names."""
        names = ", ".join(sorted(self._stack.config.mask_plan.masks))
        return f"StackMasks({names})"


__all__ = [
    "COMBINED_NAME",
    "MASK_KIND",
    "CombinedMaskView",
    "MaskDomain",
    "StackMaskView",
    "StackMasks",
    "committed_mask_product",
    "discover_mask_products",
    "grid_identity",
    "is_water_definition",
    "mask_product_root",
    "mask_recipe_identity",
    "persist_mask_product",
    "read_mask_product",
    "stage_domain",
    "validate_mask_product_name",
    "water_mask_names",
]
