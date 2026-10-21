"""Interferogram and unwrap products of a Stack (PROPOSAL-0059).

An interferogram product is ``ifg/ml_AxR/<pair>/`` with Zarr layers and one
``item.json`` (kind ``ifg``) that binds both scene write ids. The
spatially unwrapped phase of that Pair is a separate product at
``unwrap/ml_AxR/<pair>/`` (kind ``unwrap``) bound to the IFG ``write_id``.
Both are written through :mod:`faninsar.io.storage.product`; reads validate
the Item, shapes, and dtypes without re-hashing payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.io.storage._zarr_stac import (
    read_zarr_array,
    validate_spatial_metadata,
    write_zarr_array,
)
from faninsar.io.storage.product import (
    Product,
    commit_product,
    is_write_id,
    open_product,
)
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = setup_logger(__name__)

IFG_KIND = "ifg"
UNWRAP_KIND = "unwrap"
_IFG_LAYERS = ("complex_ifg", "coherence", "wrapped_phase", "amplitude", "valid_mask")
_REQUIRED_IFG_LAYERS = ("complex_ifg", "wrapped_phase", "amplitude")
_UNWRAP_LAYERS = ("unwrapped_phase", "connected_components")


def _canonical_parameters(value: object, field_name: str) -> dict[str, Any]:
    """Validate string-keyed strict-JSON metadata."""
    import json

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        reject_invalid_state(f"artifact {field_name} must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        reject_invalid_state(f"artifact {field_name} is not strict JSON: {error}")
    return dict(value)


def _is_sha256(value: object) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_source_identity(value: object) -> bool:
    """Return whether a value names an upstream (write id or SHA-256)."""
    return is_write_id(value) or _is_sha256(value)


def unwrap_root(ifg_root: str | Path) -> Path:
    """Return the unwrap product directory paired with an IFG product.

    ``<work>/ifg/ml_AxR/<pair>`` maps to ``<work>/unwrap/ml_AxR/<pair>``.
    """
    root = Path(ifg_root)
    if root.parent.parent.name != "ifg":
        reject_invalid_state(
            f"IFG product is not under an ifg/ml_AxR/<pair> layout: {root}"
        )
    return root.parent.parent.parent / "unwrap" / root.parent.name / root.name


def _validate_phase_screen_digests(value: object) -> dict[str, dict[str, str]]:
    """Validate the role- and unit-keyed phase-screen source table."""
    if not isinstance(value, dict) or any(
        role not in {"primary", "secondary"} or not isinstance(entries, dict)
        for role, entries in value.items()
    ):
        reject_invalid_state(
            "artifact phase_screen_digests must map primary/secondary to unit ids"
        )
    result: dict[str, dict[str, str]] = {}
    for role, entries in value.items():
        if any(
            not isinstance(tag, str) or not tag or not _is_source_identity(digest)
            for tag, digest in entries.items()
        ):
            reject_invalid_state("artifact phase-screen digests are invalid")
        result[str(role)] = {str(tag): str(digest) for tag, digest in entries.items()}
    return result


@dataclass(frozen=True, slots=True)
class InterferogramArtifact:
    """Layers of one interferogram product.

    Attributes
    ----------
    complex_ifg, wrapped_phase, amplitude : numpy.ndarray
        Product layers sharing one grid.
    coherence : numpy.ndarray or None
        Optional coherence layer.
    valid_mask : numpy.ndarray or None
        Optional authoritative support mask.

    """

    complex_ifg: np.ndarray
    coherence: np.ndarray | None
    wrapped_phase: np.ndarray
    amplitude: np.ndarray
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class UnwrappedArtifact:
    """Layers of one unwrap product bound to its interferogram.

    Attributes
    ----------
    unwrapped_phase, connected_components : numpy.ndarray
        Unwrap result layers.
    method : str
        Unwrapping method name.
    method_parameters : dict[str, object]
        Strict-JSON method configuration.
    ifg_manifest_digest : str
        ``write_id`` of the interferogram the unwrap consumed.
    write_id : str
        ``write_id`` of the unwrap product.

    """

    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    method: str
    method_parameters: dict[str, Any]
    ifg_manifest_digest: str
    write_id: str = ""


@dataclass(frozen=True, slots=True)
class InterferogramArtifactStore:
    """Read-only view of one committed interferogram product.

    Attributes
    ----------
    root : pathlib.Path
        Product directory ``ifg/ml_AxR/<pair>``.
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors.
    filter_name : str
        Applied filter name.
    filter_parameters : dict[str, object]
        Filter configuration.
    source_manifest_digests : dict[str, str]
        Upstream identities: ``primary``/``secondary`` scene write ids and
        the ``runtime`` fingerprint.
    shape : tuple[int, int]
        Output grid shape.
    manifest_digest : str
        The product ``write_id`` (dependents bind it).

    """

    root: Path
    pair: tuple[str, str]
    domain: str
    wavelength_m: float | None
    grid_identity: str
    looks: tuple[int, int]
    filter_name: str
    filter_parameters: dict[str, Any]
    source_manifest_digests: dict[str, str]
    shape: tuple[int, int]
    manifest_digest: str
    _payloads: dict[str, dict[str, Any]]
    flatten_stage: str = "coregistration"
    phase_screen_model: str | None = None
    phase_screen_digests: dict[str, dict[str, str]] = field(default_factory=dict)
    phase_screen_domain: str = "radar"
    phase_screen_grid_identity: str = ""
    mask_plan_identity: str = ""
    mask_identity: str | None = None
    crs: str | None = None
    transform: tuple[float, ...] | None = None

    @property
    def write_id(self) -> str:
        """Return the product write id."""
        return self.manifest_digest

    @property
    def generation_id(self) -> str:
        """Return the product write id (one product, no generations)."""
        return self.manifest_digest

    @property
    def generation_root(self) -> Path:
        """Return the directory holding the payload layers."""
        return self.root

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open one committed interferogram product.

        Parameters
        ----------
        root : str or pathlib.Path
            Product directory containing ``item.json``.

        Returns
        -------
        InterferogramArtifactStore
            Validated, read-only product metadata.

        Raises
        ------
        InvalidProcessingStateError
            If the directory is not a committed IFG product or its metadata
            is inconsistent.

        """
        product = open_product(root, kind=IFG_KIND)
        p = product.properties
        raw_pair = p.get("faninsar:pair")
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(not isinstance(value, str) or not value for value in raw_pair)
        ):
            reject_invalid_state("artifact pair must contain two non-empty ids")
        raw_looks = p.get("faninsar:looks")
        if (
            not isinstance(raw_looks, list)
            or len(raw_looks) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in raw_looks
            )
        ):
            reject_invalid_state("artifact looks must contain two positive integers")
        raw_shape = p.get("faninsar:shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 2:
            reject_invalid_state("artifact shape is invalid")
        domain, shape, grid_identity, crs, transform = validate_spatial_metadata(
            p.get("faninsar:domain"),
            (int(raw_shape[0]), int(raw_shape[1])),
            p.get("faninsar:grid_identity"),
            crs=p.get("proj:code"),
            transform=p.get("proj:transform"),
        )
        flatten_stage = p.get("faninsar:flatten_stage", "coregistration")
        if flatten_stage not in {"coregistration", "interferogram"}:
            reject_invalid_state("artifact flatten_stage is invalid")
        phase_screen_model = p.get("faninsar:phase_screen_model")
        if phase_screen_model not in {None, "nisar_ellipsoidal_v1"}:
            reject_invalid_state("artifact phase_screen_model is invalid")
        phase_screen_digests = _validate_phase_screen_digests(
            p.get("faninsar:phase_screen_digests", {})
        )
        raw_filter = p.get("faninsar:filter")
        if not isinstance(raw_filter, dict) or not raw_filter.get("name"):
            reject_invalid_state("artifact filter metadata is invalid")
        sources = {
            **dict(product.inputs),
            **{str(k): str(v) for k, v in dict(p.get("faninsar:runtime", {})).items()},
        }
        payloads: dict[str, dict[str, Any]] = {}
        for name, asset in product.assets.items():
            if name in _IFG_LAYERS:
                payloads[name] = dict(asset)
        missing = [name for name in _REQUIRED_IFG_LAYERS if name not in payloads]
        if missing:
            reject_invalid_state(f"IFG product is missing layers: {missing}")
        raw_wavelength = p.get("faninsar:wavelength_m")
        return cls(
            root=product.path,
            pair=(raw_pair[0], raw_pair[1]),
            domain=str(domain),
            wavelength_m=None if raw_wavelength is None else float(raw_wavelength),
            grid_identity=str(grid_identity),
            crs=crs,
            transform=transform,
            looks=(raw_looks[0], raw_looks[1]),
            filter_name=str(raw_filter["name"]),
            filter_parameters=_canonical_parameters(
                raw_filter.get("parameters", {}), "filter parameters"
            ),
            source_manifest_digests=sources,
            flatten_stage=str(flatten_stage),
            phase_screen_model=phase_screen_model,
            phase_screen_digests=phase_screen_digests,
            phase_screen_domain=str(domain),
            phase_screen_grid_identity=str(grid_identity),
            mask_plan_identity=str(p.get("faninsar:mask_plan_identity", "")),
            mask_identity=p.get("faninsar:mask_identity"),
            shape=shape,
            manifest_digest=product.write_id,
            _payloads=payloads,
        )

    def close(self) -> None:
        """Release nothing; products are not leased."""

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release nothing."""

    def _layer(self, name: str) -> np.ndarray:
        path = self.root / str(self._payloads[name]["href"])
        try:
            array = read_zarr_array(path)
        except (OSError, ValueError, EOFError) as error:
            reject_invalid_state(f"IFG layer {name} cannot be loaded: {error}")
        if array.shape != self.shape:
            reject_invalid_state(f"IFG layer {name} shape differs from the Item")
        return array

    def read(self) -> InterferogramArtifact:
        """Read all IFG layers after validating shapes and dtypes."""
        arrays = {name: self._layer(name) for name in self._payloads}
        if (
            arrays["complex_ifg"].dtype.kind != "c"
            or any(
                arrays[name].dtype.kind != "f"
                for name in ("wrapped_phase", "amplitude")
            )
            or ("coherence" in arrays and arrays["coherence"].dtype.kind != "f")
        ):
            reject_invalid_state("artifact IFG layer dtypes are incompatible")
        valid_mask = arrays.get("valid_mask")
        if valid_mask is not None and valid_mask.dtype != np.bool_:
            reject_invalid_state("artifact IFG valid_mask dtype is incompatible")
        return InterferogramArtifact(
            complex_ifg=arrays["complex_ifg"],
            coherence=arrays.get("coherence"),
            wrapped_phase=arrays["wrapped_phase"],
            amplitude=arrays["amplitude"],
            valid_mask=valid_mask,
        )

    @property
    def unwrap_root(self) -> Path:
        """Return the unwrap product directory for this Pair."""
        return unwrap_root(self.root)

    def has_unwrapped(self) -> bool:
        """Return whether a current unwrap product is bound to this IFG."""
        item = self.unwrap_root / "item.json"
        if not item.is_file():
            return False
        product = open_product(self.unwrap_root, kind=UNWRAP_KIND)
        return product.inputs.get("ifg") == self.write_id

    def unwrap_product(self) -> Product:
        """Open the unwrap product bound to this IFG without reading layers.

        Raises
        ------
        InvalidProcessingStateError
            If no unwrap product exists or it is bound to another IFG.

        """
        product = open_product(self.unwrap_root, kind=UNWRAP_KIND)
        if product.inputs.get("ifg") != self.write_id:
            reject_invalid_state("unwrap artifact is bound to a different IFG")
        return product

    def read_unwrapped_layer(self, name: str) -> np.ndarray:
        """Read one named layer of the unwrap product bound to this IFG."""
        product = self.unwrap_product()
        if name not in product.assets:
            reject_invalid_state(f"unwrap product has no layer {name!r}")
        return read_zarr_array(product.asset_path(name))

    def read_unwrapped(self) -> UnwrappedArtifact:
        """Read the unwrap product bound to this exact IFG.

        Raises
        ------
        InvalidProcessingStateError
            If no unwrap product exists or it is bound to another IFG.

        """
        product = self.unwrap_product()
        arrays: dict[str, np.ndarray] = {}
        for name in _UNWRAP_LAYERS:
            if name not in product.assets:
                reject_invalid_state(f"unwrap product is missing layer {name}")
            array = read_zarr_array(product.asset_path(name))
            if array.shape != self.shape:
                reject_invalid_state("unwrap artifact shape differs from its IFG")
            arrays[name] = array
        if (
            arrays["unwrapped_phase"].dtype.kind != "f"
            or arrays["connected_components"].dtype.kind not in "iu"
        ):
            reject_invalid_state("unwrap artifact layer dtypes are incompatible")
        method = product.properties.get("faninsar:method")
        if not isinstance(method, dict) or not method.get("name"):
            reject_invalid_state("unwrap method metadata is invalid")
        return UnwrappedArtifact(
            unwrapped_phase=arrays["unwrapped_phase"],
            connected_components=arrays["connected_components"],
            method=str(method["name"]),
            method_parameters=_canonical_parameters(
                method.get("parameters", {}), "unwrap method parameters"
            ),
            ifg_manifest_digest=self.write_id,
            write_id=product.write_id,
        )


def write_ifg_artifact(
    root: str | Path,
    *,
    pair: tuple[str, str],
    looks: tuple[int, int],
    domain: str = "radar",
    wavelength_m: float | None = None,
    grid_identity: str | None = None,
    crs: object = None,
    transform: object = None,
    filter_name: str,
    filter_parameters: Mapping[str, Any],
    source_manifest_digests: Mapping[str, str],
    mask_plan_identity: str = "",
    mask_identity: str | None = None,
    flatten_stage: str = "coregistration",
    phase_screen_model: str | None = None,
    phase_screen_digests: Mapping[str, Mapping[str, str]] | None = None,
    phase_screen_domain: str | None = None,
    phase_screen_grid_identity: str | None = None,
    complex_ifg: np.ndarray,
    coherence: np.ndarray | None,
    wrapped_phase: np.ndarray,
    amplitude: np.ndarray,
    valid_mask: np.ndarray | None = None,
    replace_existing: bool = False,
) -> InterferogramArtifactStore:
    """Commit one interferogram product.

    Parameters
    ----------
    root : str or pathlib.Path
        Product directory ``ifg/ml_AxR/<pair>``.
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors.
    domain : {"radar", "geo"}, optional
        Output coordinate domain.
    wavelength_m : float, optional
        Radar wavelength.
    grid_identity : str, optional
        Source scene grid identity; derived from domain and shape when
        omitted (radar only).
    crs, transform : object, optional
        Geographic metadata (geo only).
    filter_name : str
        Applied filter name, or ``"none"``.
    filter_parameters : mapping
        Strict-JSON filter configuration.
    source_manifest_digests : mapping[str, str]
        Upstream identities. ``primary`` and ``secondary`` are the scene
        write ids recorded as Item inputs; other keys (``runtime``) are
        recorded as properties.
    mask_plan_identity, mask_identity : str, optional
        Mask lineage.
    flatten_stage : {"coregistration", "interferogram"}, optional
        Stage at which flattening was applied.
    phase_screen_model : {None, "nisar_ellipsoidal_v1"}, optional
        Phase-screen convention.
    phase_screen_digests : mapping, optional
        Source role to unit to scene write id of the applied screens.
    phase_screen_domain, phase_screen_grid_identity : str, optional
        Must equal the IFG domain and grid when given.
    complex_ifg, wrapped_phase, amplitude : numpy.ndarray
        Required layers.
    coherence, valid_mask : numpy.ndarray, optional
        Optional layers.
    replace_existing : bool, optional
        Replace an existing product.

    Returns
    -------
    InterferogramArtifactStore
        The committed product.

    """
    arrays: dict[str, np.ndarray] = {
        "complex_ifg": np.asarray(complex_ifg),
        "wrapped_phase": np.asarray(wrapped_phase),
        "amplitude": np.asarray(amplitude),
    }
    if coherence is not None:
        arrays["coherence"] = np.asarray(coherence)
    if valid_mask is not None:
        arrays["valid_mask"] = np.asarray(valid_mask)
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        reject_invalid_state("IFG artifact layers must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or any(size <= 0 for size in shape):
        reject_invalid_state("IFG artifact layers must be non-empty 2-D arrays")
    if (
        arrays["complex_ifg"].dtype.kind != "c"
        or any(
            arrays[name].dtype.kind != "f" for name in ("wrapped_phase", "amplitude")
        )
        or (coherence is not None and arrays["coherence"].dtype.kind != "f")
    ):
        reject_invalid_state("IFG artifact layer dtypes are incompatible")
    if valid_mask is not None and arrays["valid_mask"].dtype != np.bool_:
        reject_invalid_state("IFG artifact valid_mask must be boolean")
    if (
        len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        or len(looks) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in looks
        )
        or not isinstance(filter_name, str)
        or not filter_name
        or domain not in {"radar", "geo"}
        or flatten_stage not in {"coregistration", "interferogram"}
    ):
        reject_invalid_state("IFG artifact processing metadata is invalid")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("IFG artifact wavelength_m is invalid")
    if domain == "geo" and grid_identity is None:
        reject_invalid_state("Geo IFG publication requires an explicit grid_identity")
    if grid_identity is None:
        from faninsar.io.storage.scene_store import scene_grid_identity

        grid_identity = scene_grid_identity("radar", (int(shape[0]), int(shape[1])))
    domain, shape, grid_identity, resolved_crs, resolved_transform = (
        validate_spatial_metadata(
            domain,
            (int(shape[0]), int(shape[1])),
            grid_identity,
            crs=crs,
            transform=transform,
        )
    )
    if phase_screen_domain is not None and phase_screen_domain != domain:
        reject_invalid_state("IFG phase-screen domain differs from IFG domain")
    if (
        phase_screen_grid_identity is not None
        and phase_screen_grid_identity != grid_identity
    ):
        reject_invalid_state("IFG phase-screen grid differs from IFG grid")
    screens = _validate_phase_screen_digests(dict(phase_screen_digests or {}))
    if phase_screen_model not in {None, "nisar_ellipsoidal_v1"}:
        reject_invalid_state("IFG phase_screen_model is invalid")
    if flatten_stage == "coregistration" and (phase_screen_model or screens):
        reject_invalid_state(
            "coregistration IFG artifacts must not carry a phase screen"
        )
    if flatten_stage == "interferogram" and (
        phase_screen_model != "nisar_ellipsoidal_v1" or not any(screens.values())
    ):
        reject_invalid_state(
            "interferogram IFG artifacts require non-empty NISAR phase-screen lineage"
        )
    sources = dict(source_manifest_digests)
    if not sources or any(
        not isinstance(name, str) or not name or not _is_source_identity(value)
        for name, value in sources.items()
    ):
        reject_invalid_state("IFG artifact source manifest digests are invalid")
    inputs = {
        name: value
        for name, value in sources.items()
        if name in {"primary", "secondary"}
    }
    runtime = {name: value for name, value in sources.items() if name not in inputs}
    properties: dict[str, object] = {
        "faninsar:pair": list(pair),
        "faninsar:looks": [int(looks[0]), int(looks[1])],
        "faninsar:domain": domain,
        "faninsar:shape": [int(shape[0]), int(shape[1])],
        "faninsar:wavelength_m": wavelength_m,
        "faninsar:grid_identity": grid_identity,
        "faninsar:flatten_stage": flatten_stage,
        "faninsar:phase_screen_model": phase_screen_model,
        "faninsar:phase_screen_digests": screens,
        "faninsar:filter": {
            "name": filter_name,
            "parameters": _canonical_parameters(
                dict(filter_parameters), "filter parameters"
            ),
        },
        "faninsar:runtime": runtime,
        "faninsar:mask_plan_identity": mask_plan_identity,
        "faninsar:mask_identity": mask_identity,
    }
    if resolved_crs is not None:
        properties["proj:code"] = resolved_crs
        properties["proj:transform"] = list(resolved_transform or ())

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        assets = {}
        for name in _IFG_LAYERS:
            if name in arrays:
                write_zarr_array(staging / f"{name}.zarr", arrays[name])
                assets[name] = {"href": f"{name}.zarr"}
        return assets

    product = commit_product(
        root,
        kind=IFG_KIND,
        properties=properties,
        write_payloads=write,
        inputs=inputs,
        overwrite=replace_existing,
    )
    logger.info("Committed IFG product %s for pair %s", product.path, pair)
    return InterferogramArtifactStore.open(product.path)


def write_unwrapped_artifact(
    root: str | Path,
    *,
    unwrapped_phase: np.ndarray,
    connected_components: np.ndarray,
    method: str,
    method_parameters: Mapping[str, Any],
    ifg_manifest_digest: str,
    extra_layers: Mapping[str, np.ndarray] | None = None,
    properties: Mapping[str, object] | None = None,
    replace_existing: bool = False,
) -> Product:
    """Commit the unwrap product of one interferogram.

    Parameters
    ----------
    root : str or pathlib.Path
        The interferogram product directory; the unwrap product is written
        to the paired ``unwrap/ml_AxR/<pair>`` directory.
    unwrapped_phase : numpy.ndarray
        Floating-point unwrapped phase in radians.
    connected_components : numpy.ndarray
        Integer connected-component labels.
    method : str
        Unwrapping method name.
    method_parameters : mapping
        Strict-JSON method configuration.
    ifg_manifest_digest : str
        ``write_id`` of the interferogram being unwrapped.
    extra_layers : mapping, optional
        Further same-shape layers (for example ``valid_mask``); 1-D arrays
        such as per-component ``reference_values`` are stored as given.
    properties : mapping, optional
        Extra ``faninsar:`` Item properties (for example mask lineage).
        Geographic metadata is inherited from the IFG.
    replace_existing : bool, optional
        Replace an existing unwrap product (and its ``ion_corrected``
        child).

    Returns
    -------
    Product
        The committed unwrap product.

    """
    store = InterferogramArtifactStore.open(root)
    if ifg_manifest_digest != store.write_id:
        reject_invalid_state("unwrap artifact IFG binding is invalid")
    phase = np.asarray(unwrapped_phase)
    components = np.asarray(connected_components)
    if (
        phase.shape != store.shape
        or components.shape != store.shape
        or phase.dtype.kind != "f"
        or components.dtype.kind not in "iu"
    ):
        reject_invalid_state("unwrap layers must match the IFG shape and dtypes")
    if not isinstance(method, str) or not method:
        reject_invalid_state("unwrap method must not be empty")
    item_properties: dict[str, object] = {
        "faninsar:pair": list(store.pair),
        "faninsar:looks": list(store.looks),
        "faninsar:domain": store.domain,
        "faninsar:shape": list(store.shape),
        "faninsar:grid_identity": store.grid_identity,
        "faninsar:method": {
            "name": method,
            "parameters": _canonical_parameters(
                dict(method_parameters), "unwrap method parameters"
            ),
        },
    }
    extras = {
        name: np.asarray(value) for name, value in dict(extra_layers or {}).items()
    }
    for name in extras:
        if name in _UNWRAP_LAYERS or not name.isidentifier():
            reject_invalid_state(f"unwrap extra layer name is invalid: {name!r}")
    item_properties.update(dict(properties or {}))
    if store.crs is not None:
        item_properties["proj:code"] = store.crs
        item_properties["proj:transform"] = list(store.transform or ())

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        layers = {
            "unwrapped_phase": phase,
            "connected_components": components,
            **extras,
        }
        for name, array in layers.items():
            write_zarr_array(staging / f"{name}.zarr", array)
        return {name: {"href": f"{name}.zarr"} for name in layers}

    target = store.unwrap_root
    target.parent.mkdir(parents=True, exist_ok=True)
    product = commit_product(
        target,
        kind=UNWRAP_KIND,
        properties=item_properties,
        write_payloads=write,
        inputs={"ifg": store.write_id},
        overwrite=replace_existing,
    )
    logger.info("Committed unwrap product %s with method %s", target, method)
    return product


__all__ = [
    "IFG_KIND",
    "UNWRAP_KIND",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "UnwrappedArtifact",
    "unwrap_root",
    "write_ifg_artifact",
    "write_unwrapped_artifact",
]
