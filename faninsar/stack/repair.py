"""Repair and rescan for cross-version workspace recovery (PROPOSAL-0069).

A work directory whose ``stack.json`` can no longer be decoded after a
FanInSAR upgrade is recovered in two distinct layers:

- the *identity* layer (``stack.json`` + ``item.json`` metadata) is
  rewritable: :func:`rescan` diagnoses what survives on disk and
  :func:`repair` rebuilds a fresh ``faninsar.stack.v1`` identity from the
  recovered science configuration plus committed-product inference;
- the *product* layer (zarr payloads + ``write_id`` lineage) is never
  rewritten — products whose upstream ``write_id`` no longer matches are
  surfaced as ``stale`` through the existing :meth:`Stack.status` machinery
  and must be reprocessed, not repaired in place.

``Stack.open`` is deliberately *tolerant*: a readable-but-drifted
``stack.json`` still yields a usable Stack so ``status()``/``scenes``/
products remain reachable, and the unrecovered fields are reported under
``stack.repair_report``.  Only a missing ``stack.json`` still raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.config import StackConfig
from faninsar.stack.identity import (
    UNSAVED_FIELDS,
    decode_value,
    read_stack_identity_lenient,
)

if TYPE_CHECKING:
    from faninsar.io.storage.product import Product
    from faninsar.stack.session import Stack

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Audit of one recovery pass over a work directory.

    Populated by :func:`rescan`; :func:`repair` attaches it to the returned
    Stack as ``stack.repair_report`` so the degraded state is visible through
    :meth:`Stack.status` under ``report["identity"]["repair"]``.
    """

    work_dir: Path
    """The recovered work directory."""

    schema_found: str | None
    """The ``schema`` value recorded in ``stack.json`` (``None`` when unreadable)."""

    inferred_reference: str | None
    """Reference date inferred from products (``faninsar:role == "reference"``)."""

    science_recovered: dict[str, Any] = field(default_factory=dict)
    """``StackConfig`` field names → decoded values recovered from ``science``."""

    science_dropped: list[str] = field(default_factory=list)
    """``science`` keys present but not decodable into the current StackConfig."""

    science_defaulted: list[str] = field(default_factory=list)
    """``StackConfig`` science fields absent from ``science`` (defaults used)."""

    inputs_ok: list[str] = field(default_factory=list)
    """Input records (sources/orbits/calibrations) that pass ``_check_input``."""

    inputs_changed: list[str] = field(default_factory=list)
    """Input records that fail ``_check_input`` (missing or size-changed)."""

    inferred_pairs: list[str] = field(default_factory=list)
    """Pair names inferred from committed ``ifg/ml_*`` product directories."""

    inferred_misreg_pairs: list[str] = field(default_factory=list)
    """Pair names inferred from ``unwrap/ml_*`` and the ``misreg`` product."""

    products_found: dict[str, list[str]] = field(default_factory=dict)
    """Committed product ids grouped by kind (``scene``, ``ifg``, ``unwrap``, …)."""

    stale_products: list[str] = field(default_factory=list)
    """Products flagged stale by :meth:`Stack.status` after repair."""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe projection for :meth:`Stack.status`."""
        return {
            "work_dir": str(self.work_dir),
            "schema_found": self.schema_found,
            "science_recovered": sorted(self.science_recovered),
            "science_dropped": list(self.science_dropped),
            "science_defaulted": list(self.science_defaulted),
            "inputs_ok": list(self.inputs_ok),
            "inputs_changed": list(self.inputs_changed),
            "inferred_pairs": list(self.inferred_pairs),
            "inferred_misreg_pairs": list(self.inferred_misreg_pairs),
            "inferred_reference": self.inferred_reference,
            "products_found": {
                kind: sorted(ids) for kind, ids in self.products_found.items()
            },
            "stale_products": list(self.stale_products),
        }


def rescan(work_dir: str | Path) -> RepairReport:
    """Diagnose one work directory without writing anything.

    Reads ``stack.json`` leniently, runs ``_check_input`` on each recorded
    input, and infers the planned Pair network and Reference from the
    committed products.  The result tells the caller exactly what a
    subsequent :func:`repair` would recover versus leave defaulted.

    Parameters
    ----------
    work_dir : str or pathlib.Path
        Work directory to inspect.

    Returns
    -------
    RepairReport
        The recovery audit.

    """
    root = Path(work_dir)
    document = read_stack_identity_lenient(root)
    schema_found = None if document is None else document.get("schema")

    science_recovered, science_dropped, science_defaulted = _recover_science(document)
    inputs_ok, inputs_changed = _check_inputs(document)
    products_found, inferred_pairs, inferred_misreg_pairs = _discover(root)
    inferred_reference = _infer_reference(root)

    return RepairReport(
        work_dir=root,
        schema_found=None if schema_found is None else str(schema_found),
        science_recovered=science_recovered,
        science_dropped=science_dropped,
        science_defaulted=science_defaulted,
        inputs_ok=inputs_ok,
        inputs_changed=inputs_changed,
        inferred_pairs=inferred_pairs,
        inferred_misreg_pairs=inferred_misreg_pairs,
        inferred_reference=inferred_reference,
        products_found=products_found,
    )


def _recover_science(
    document: dict[str, object] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Decode the ``science`` section into recovered/dropped/defaulted.

    ``StackConfig`` field names drive classification: a science key matching
    a current field that decodes cleanly is recovered; one that fails to
    decode (or a ``faninsar:`` payload that round-trips to the wrong type) is
    dropped; a saved field absent from the document is defaulted.  Keys that
    do not correspond to a current ``StackConfig`` field are ignored here —
    ``Stack.open`` already logs them separately.
    """
    from faninsar.stack.identity import fields as config_fields

    known = {f.name for f in config_fields(StackConfig)} - UNSAVED_FIELDS
    raw_science = (document or {}).get("science", {})
    science_recovered: dict[str, Any] = {}
    science_dropped: list[str] = []
    if isinstance(raw_science, dict):
        for name, value in raw_science.items():
            if name not in known:
                continue
            try:
                science_recovered[name] = decode_value(value)
            except Exception:  # a drifted payload must not abort recovery
                science_dropped.append(name)
                logger.warning(
                    "stack.json science field %r could not be decoded and is defaulted",
                    name,
                )
    science_defaulted = sorted(known - set(science_recovered))
    return science_recovered, sorted(science_dropped), science_defaulted


def _check_inputs(document: dict[str, object] | None) -> tuple[list[str], list[str]]:
    """Run ``_check_input`` over each recorded input record."""
    from faninsar.stack.session import _check_input

    inputs = (document or {}).get("inputs", {})
    records: list[object] = []
    if isinstance(inputs, dict):
        for entries in inputs.get("sources", {}).values():
            if isinstance(entries, list):
                records.extend(entries)
        records.extend(inputs.get("orbits", {}).values())
        calibrations = inputs.get("calibrations", [])
        if isinstance(calibrations, list):
            records.extend(calibrations)

    inputs_ok: list[str] = []
    inputs_changed: list[str] = []
    for entry in records:
        try:
            _check_input(entry)
        except InvalidProcessingStateError:
            path = entry.get("path") if isinstance(entry, dict) else None
            inputs_changed.append(str(path) if path is not None else "<unknown>")
        else:
            path = entry.get("path") if isinstance(entry, dict) else None
            inputs_ok.append(str(path) if path is not None else "<unknown>")
    return sorted(inputs_ok), sorted(inputs_changed)


def _discover(
    root: Path,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Discover committed products and infer the planned networks."""
    from faninsar.io.storage.ifg_store import IFG_KIND, UNWRAP_KIND
    from faninsar.io.storage.product import discover_products
    from faninsar.io.storage.scene_store import SCENE_KIND

    products_found: dict[str, list[str]] = {}
    inferred_pairs: list[str] = []
    inferred_misreg_pairs: list[str] = []

    def _scan(base: Path, kind: str, label: str) -> None:
        if not base.is_dir():
            return
        try:
            products = discover_products(base, kind=kind)
        except InvalidProcessingStateError:
            logger.warning("could not scan %s products under %s", label, base)
            return
        ids = [_product_label(product, label) for product in products]
        products_found[label] = sorted(ids)

    _scan(root / "coreg", SCENE_KIND, "scene")

    # IFG products live under ifg/ml_AxR/<pair>; unwrap mirrors the same
    # pair under unwrap/ml_AxR/<pair>.  Scan every committed multilook.
    ifg_root = root / "ifg"
    if ifg_root.is_dir():
        for looks_dir in sorted(ifg_root.glob("ml_*")):
            if not looks_dir.is_dir():
                continue
            _scan(looks_dir, IFG_KIND, "ifg")
            for product in _products(looks_dir, IFG_KIND):
                pair = _product_label(product, "ifg")
                if pair not in inferred_pairs:
                    inferred_pairs.append(pair)

    unwrap_base = root / "unwrap"
    if unwrap_base.is_dir():
        for looks_dir in sorted(unwrap_base.glob("ml_*")):
            if not looks_dir.is_dir():
                continue
            _scan(looks_dir, UNWRAP_KIND, "unwrap")
            for product in _products(looks_dir, UNWRAP_KIND):
                pair = _product_label(product, "unwrap")
                if pair not in inferred_misreg_pairs:
                    inferred_misreg_pairs.append(pair)

    # The single misreg product binds every measured arc; its recorded arcs
    # are the authoritative misreg network.
    misreg_root = root / "misreg"
    if (misreg_root / "item.json").is_file():
        try:
            from faninsar.stack.misreg import read_misreg

            record = read_misreg(misreg_root)
        except Exception:  # an unreadable misreg is reported, not fatal
            logger.warning("misreg product under %s could not be read", misreg_root)
        else:
            if record is not None:
                products_found["misreg"] = [record.write_id]
                for arc in record.arcs:
                    name = f"{arc.primary}_{arc.secondary}"
                    if name not in inferred_misreg_pairs:
                        inferred_misreg_pairs.append(name)

    timeseries_root = root / "timeseries"
    if (timeseries_root / "item.json").is_file():
        products_found["timeseries"] = [str(timeseries_root)]

    return (
        products_found,
        sorted(inferred_pairs),
        sorted(inferred_misreg_pairs),
    )


def _products(base: Path, kind: str) -> list[Product]:
    """Return discovered products, swallowing scan errors."""
    from faninsar.io.storage.product import discover_products

    try:
        return discover_products(base, kind=kind)
    except InvalidProcessingStateError:
        return []


def _product_label(product: Product, kind: str) -> str:
    """Return a stable display id for one committed product."""
    if kind in {"ifg", "unwrap"}:
        pair = product.properties.get("faninsar:pair")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return f"{pair[0]}_{pair[1]}"
    date = product.properties.get("faninsar:date")
    if isinstance(date, str):
        return date
    return product.write_id


def _infer_reference(root: Path) -> str | None:
    """Infer the Reference from the scene product carrying the role tag."""
    from faninsar.io.storage.product import discover_products
    from faninsar.io.storage.scene_store import SCENE_KIND

    try:
        scenes = discover_products(root / "coreg", kind=SCENE_KIND)
    except InvalidProcessingStateError:
        scenes = []
    for product in scenes:
        if product.properties.get("faninsar:role") == "reference":
            date = product.properties.get("faninsar:date")
            if isinstance(date, str):
                return date
    # Fall back to the earliest committed scene date.
    dates = [
        str(product.properties.get("faninsar:date"))
        for product in scenes
        if isinstance(product.properties.get("faninsar:date"), str)
    ]
    return min(dates) if dates else None


def repair(
    cls: type[Stack],
    work_dir: str | Path,
    *,
    pairs: object = None,
    misreg_pairs: object = None,
    overwrite_science: dict[str, Any] | None = None,
    **runtime: Any,
) -> Stack:
    """Rebuild ``stack.json`` and reopen the Stack.

    Runs :func:`rescan`, rebuilds a :class:`StackConfig` from recovered
    science plus defaults, reconstructs the catalog and Pair networks from
    committed products, writes ``stack.json`` via
    :func:`~faninsar.stack.identity.save_stack_identity` (still schema
    ``faninsar.stack.v1``), then runs the same open tail as
    :meth:`Stack.open`.  ``inputs_changed`` never blocks repair.

    Parameters
    ----------
    cls : type of Stack
        The concrete Stack subclass to reopen as (``S1Stack``,
        ``NISARStack``); the mission recorded in ``stack.json`` wins when
        readable.
    work_dir : str or pathlib.Path
        Work directory to repair.
    pairs, misreg_pairs : Pairs, optional
        Explicit networks; when ``None`` they are inferred from committed
        products (or read literally from a readable ``stack.json``).
    overwrite_science : dict, optional
        Science fields to override after recovery.
    **runtime : Any
        Runtime-only :class:`StackConfig` values (``device``, ``executor``,
        ``n_jobs``, …), identical to :meth:`Stack.open`.

    Returns
    -------
    Stack
        The reopened Stack with ``stack.repair_report`` attached.

    """
    # Deferred to avoid a session↔repair import cycle.
    from faninsar import Pairs
    from faninsar.stack.identity import save_stack_identity
    from faninsar.stack.session import _mission_class

    root = Path(work_dir)
    report = rescan(root)
    document = read_stack_identity_lenient(root) or {}

    # Mission: readable stack.json wins, else the concrete subclass invoked.
    mission = document.get("mission")
    stack_cls = _mission_class(str(mission)) if mission else cls

    science = dict(report.science_recovered)
    if overwrite_science:
        science.update(overwrite_science)
    science["work_dir"] = root
    science.update({name: v for name, v in runtime.items() if name != "scene_provider"})
    # Runtime fields are never persisted; the recovered extras merge back so
    # orbit/calibration paths recorded in stack.json survive the rebuild.
    science["extra"] = _recovered_extra(document)
    config = StackConfig(**science)

    catalog = _rebuild_catalog(document, root)
    reference = _literal(document, "reference", None) or report.inferred_reference
    if reference is None:
        dates = sorted(catalog.paths)
        reference = dates[0] if dates else ""
    stack = stack_cls(
        catalog=catalog,
        config=config,
        pairs=Pairs.from_names(
            pairs or _literal(document, "pairs", report.inferred_pairs)
        ),
        misreg_pairs=Pairs.from_names(
            misreg_pairs
            or _literal(document, "misreg_pairs", report.inferred_misreg_pairs)
        ),
        reference=str(reference),
    )
    stack.scene_provider = (
        runtime.get("scene_provider") or stack_cls._default_scene_provider()
    )
    save_stack_identity(stack)
    stack.prepare_scenes()
    stack._capture_dem_source()
    stack._ensure_raster_dem()
    stack._prepared = True
    stack._refresh_network()
    stack.repair_report = report
    return stack


def _recovered_extra(document: dict[str, object]) -> dict[str, Any]:
    """Decode ``science_extra`` plus re-persisted orbit/calibration inputs."""
    extra = {
        key: decode_value(value)
        for key, value in (document.get("science_extra") or {}).items()
    }
    inputs = document.get("inputs", {})
    if isinstance(inputs, dict):
        orbits = inputs.get("orbits")
        if isinstance(orbits, dict) and orbits:
            extra["orbit_paths"] = {
                date: Path(str(entry["path"]))
                for date, entry in orbits.items()
                if isinstance(entry, dict) and "path" in entry
            }
        calibrations = inputs.get("calibrations")
        if isinstance(calibrations, list) and calibrations:
            extra["calibration_paths"] = [
                Path(str(entry["path"]))
                for entry in calibrations
                if isinstance(entry, dict) and "path" in entry
            ]
    return extra


def _rebuild_catalog(document: dict[str, object], root: Path) -> Any:
    """Rebuild the SceneCatalog from recorded inputs or committed scenes."""
    from faninsar.stack.catalog import SceneCatalog

    inputs = document.get("inputs", {})
    sources = inputs.get("sources") if isinstance(inputs, dict) else None
    paths: list[Path] = []
    if isinstance(sources, dict):
        for date in sorted(sources):
            paths.extend(
                Path(str(entry["path"]))
                for entry in sources[date]
                if isinstance(entry, dict) and "path" in entry
            )
    if not paths:
        # No readable source records: derive the catalog dates from the
        # committed scene products so the reopened Stack still has a date
        # axis to organize its network.
        try:
            from faninsar.io.storage.product import discover_products
            from faninsar.io.storage.scene_store import SCENE_KIND

            products = discover_products(root / "coreg", kind=SCENE_KIND)
        except InvalidProcessingStateError:
            products = []
        paths = [product.path for product in products]
    return SceneCatalog.from_paths(paths)


def _literal(
    document: dict[str, object],
    key: str,
    inferred: object,
) -> Any:
    """Return ``document[key]`` when present and non-empty, else ``inferred``."""
    value = document.get(key)
    if isinstance(value, list) and value:
        return value
    if isinstance(value, str) and value:
        return value
    return inferred


__all__ = ["RepairReport", "repair", "rescan"]
