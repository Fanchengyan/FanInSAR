"""Ionosphere products of a Stack (PROPOSAL-0036, PROPOSAL-0059).

A per-Pair ionosphere estimate is the product ``ion/ml_AxR/<pair>/`` (kind
``ion``) bound to both scene write ids. The ionosphere-corrected unwrapped
phase is the child product ``unwrap/ml_AxR/<pair>/ion_corrected/`` (kind
``ion_corrected``) bound to the unwrap and ion write ids, so replacing the
unwrap removes it and replacing the ion estimate makes it stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.io.storage._zarr_stac import read_zarr_array, write_zarr_array
from faninsar.io.storage.ifg_store import (
    UNWRAP_KIND,
    InterferogramArtifactStore,
    _canonical_parameters,
    _is_sha256,
    _is_source_identity,
)
from faninsar.io.storage.product import Product, commit_product, open_product
from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)

ION_KIND = "ion"
ION_CORRECTED_KIND = "ion_corrected"
ION_CORRECTED_NAME = "ion_corrected"
_ION_LAYERS = ("ionosphere_phase", "nondispersive_phase", "weight")


@dataclass(frozen=True, slots=True)
class IonosphereArtifact:
    """Layers of one ionosphere product.

    Attributes
    ----------
    ionosphere_phase : numpy.ndarray
        Dispersive ionospheric phase screen in radians. The screen models
        ``ion[primary] - ion[secondary]``; corrections subtract it from the
        pair's unwrapped interferometric phase.
    nondispersive_phase : numpy.ndarray
        Non-dispersive component retained as a diagnostic layer.
    weight : numpy.ndarray
        Final per-pixel inverse-variance weights in ``[0, 1]``; larger is
        more confident. NaN marks unconstrained pixels.

    """

    ionosphere_phase: np.ndarray
    nondispersive_phase: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True, slots=True)
class IonosphereArtifactStore:
    """Read-only view of one committed ionosphere product.

    Attributes
    ----------
    root : pathlib.Path
        Product directory ``ion/ml_AxR/<pair>``.
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors of the ion grid.
    domain : str
        Coordinate domain of the source scenes.
    wavelength_m : float | None
        Radar wavelength echoed from the source scenes.
    grid_identity : str
        Source scene coordinate-grid identity.
    method_name : str
        Estimation method.
    method_parameters : dict[str, object]
        Strict-JSON echo of the lane configuration.
    degraded : bool
        Whether the estimate was committed as degraded.
    degradation_reason : str | None
        Reason code required when ``degraded`` is true.
    runtime_fingerprint : str
        Stack runtime fingerprint bound at commit.
    devices : tuple[str, ...]
        Devices used by the estimation lane.
    source_manifest_digests : dict[str, str]
        Upstream scene write ids (``primary``/``secondary``).
    shape : tuple[int, int]
        Ion-grid shape.
    manifest_digest : str
        The product ``write_id`` (dependents bind it).

    """

    root: Path
    pair: tuple[str, str]
    looks: tuple[int, int]
    domain: str
    wavelength_m: float | None
    grid_identity: str
    method_name: str
    method_parameters: dict[str, Any]
    degraded: bool
    degradation_reason: str | None
    runtime_fingerprint: str
    devices: tuple[str, ...]
    source_manifest_digests: dict[str, str]
    shape: tuple[int, int]
    manifest_digest: str

    @property
    def write_id(self) -> str:
        """Return the product write id."""
        return self.manifest_digest

    @classmethod
    def open(cls, root: str | Path) -> Self:
        """Open one committed ionosphere product.

        Raises
        ------
        InvalidProcessingStateError
            If the directory is not a committed ion product or its metadata
            is inconsistent.

        """
        product = open_product(root, kind=ION_KIND)
        p = product.properties
        raw_pair = p.get("faninsar:pair")
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or any(not isinstance(value, str) or not value for value in raw_pair)
        ):
            reject_invalid_state("ion artifact pair must contain two non-empty ids")
        raw_looks = p.get("faninsar:looks")
        if (
            not isinstance(raw_looks, list)
            or len(raw_looks) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in raw_looks
            )
        ):
            reject_invalid_state("ion artifact looks must be two positive integers")
        raw_shape = p.get("faninsar:shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 2
            or any(not isinstance(value, int) or value <= 0 for value in raw_shape)
        ):
            reject_invalid_state("ion artifact shape is invalid")
        raw_method = p.get("faninsar:method")
        if not isinstance(raw_method, dict) or not raw_method.get("name"):
            reject_invalid_state("ion artifact method metadata is invalid")
        degraded = p.get("faninsar:degraded")
        if not isinstance(degraded, dict) or not isinstance(degraded.get("flag"), bool):
            reject_invalid_state("ion artifact degraded metadata is invalid")
        missing = [name for name in _ION_LAYERS if name not in product.assets]
        if missing:
            reject_invalid_state(f"ion product is missing layers: {missing}")
        raw_wavelength = p.get("faninsar:wavelength_m")
        reason = degraded.get("reason")
        return cls(
            root=product.path,
            pair=(raw_pair[0], raw_pair[1]),
            looks=(raw_looks[0], raw_looks[1]),
            domain=str(p.get("faninsar:domain")),
            wavelength_m=None if raw_wavelength is None else float(raw_wavelength),
            grid_identity=str(p.get("faninsar:grid_identity")),
            method_name=str(raw_method["name"]),
            method_parameters=_canonical_parameters(
                raw_method.get("parameters", {}), "ion method parameters"
            ),
            degraded=bool(degraded["flag"]),
            degradation_reason=None if reason is None else str(reason),
            runtime_fingerprint=str(p.get("faninsar:runtime_fingerprint")),
            devices=tuple(str(value) for value in p.get("faninsar:devices", [])),
            source_manifest_digests=dict(product.inputs),
            shape=(raw_shape[0], raw_shape[1]),
            manifest_digest=product.write_id,
        )

    def close(self) -> None:
        """Release nothing; products are not leased."""

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release nothing."""

    def read(self) -> IonosphereArtifact:
        """Read all ionosphere layers after validating shapes and dtypes."""
        product = open_product(self.root, kind=ION_KIND)
        arrays = {
            name: read_zarr_array(product.asset_path(name)) for name in _ION_LAYERS
        }
        if any(
            array.shape != self.shape or array.dtype.kind != "f"
            for array in arrays.values()
        ):
            reject_invalid_state("ion artifact layer shapes or dtypes are incompatible")
        return IonosphereArtifact(**arrays)


def write_ionosphere_artifact(
    root: str | Path,
    *,
    pair: tuple[str, str],
    looks: tuple[int, int],
    domain: str,
    wavelength_m: float | None,
    grid_identity: str,
    method_name: str,
    method_parameters: Mapping[str, Any],
    degraded: bool,
    degradation_reason: str | None,
    runtime_fingerprint: str,
    devices: Sequence[str],
    source_manifest_digests: Mapping[str, str],
    ionosphere_phase: np.ndarray,
    nondispersive_phase: np.ndarray,
    weight: np.ndarray,
    replace_existing: bool = False,
) -> IonosphereArtifactStore:
    """Commit one ionosphere product.

    Parameters
    ----------
    root : str or pathlib.Path
        Product directory ``ion/ml_AxR/<pair>``.
    pair : tuple[str, str]
        Primary and secondary acquisition identifiers.
    looks : tuple[int, int]
        Applied azimuth and range look factors of the ion grid.
    domain : {"radar", "geo"}
        Coordinate domain of the source scenes.
    wavelength_m : float, optional
        Radar wavelength echoed from the source scenes.
    grid_identity : str
        Source scene coordinate-grid identity (SHA-256).
    method_name : str
        Estimation method name (``"ionosphere_split_spectrum"``).
    method_parameters : mapping
        Strict-JSON echo of the lane configuration.
    degraded : bool
        Whether this estimate is marked degraded.
    degradation_reason : str, optional
        Required non-empty reason when ``degraded`` is true; forbidden
        otherwise.
    runtime_fingerprint : str
        Stack runtime fingerprint (SHA-256).
    devices : sequence of str
        Devices used by the estimation lane.
    source_manifest_digests : mapping[str, str]
        Upstream scene write ids, recorded as Item inputs.
    ionosphere_phase, nondispersive_phase, weight : numpy.ndarray
        Matching two-dimensional float layers.
    replace_existing : bool, optional
        Replace an existing product.

    Returns
    -------
    IonosphereArtifactStore
        The committed product.

    """
    arrays = {
        "ionosphere_phase": np.asarray(ionosphere_phase),
        "nondispersive_phase": np.asarray(nondispersive_phase),
        "weight": np.asarray(weight),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        reject_invalid_state("ion artifact layers must share one shape")
    shape = next(iter(shapes))
    if len(shape) != 2 or any(size <= 0 for size in shape):
        reject_invalid_state("ion artifact layers must be non-empty 2-D arrays")
    if any(array.dtype.kind != "f" for array in arrays.values()):
        reject_invalid_state("ion artifact layer dtypes must be floating point")
    devices_tuple = tuple(str(value) for value in devices)
    if (
        len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        or len(looks) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in looks
        )
        or domain not in {"radar", "geo"}
        or not isinstance(method_name, str)
        or not method_name
        or not devices_tuple
        or any(not value for value in devices_tuple)
    ):
        reject_invalid_state("ion artifact processing metadata is invalid")
    if wavelength_m is not None and (
        not np.isfinite(wavelength_m) or wavelength_m <= 0.0
    ):
        reject_invalid_state("ion artifact wavelength_m is invalid")
    if not _is_sha256(grid_identity):
        reject_invalid_state("ion artifact grid_identity is invalid")
    if not _is_sha256(runtime_fingerprint):
        reject_invalid_state("ion artifact runtime fingerprint is invalid")
    if degraded and (not isinstance(degradation_reason, str) or not degradation_reason):
        reject_invalid_state("degraded ion artifacts require a degradation reason")
    if not degraded and degradation_reason is not None:
        reject_invalid_state(
            "qualified ion artifacts must not carry a degradation reason"
        )
    sources = dict(source_manifest_digests)
    if not sources or any(
        not isinstance(name, str) or not name or not _is_source_identity(value)
        for name, value in sources.items()
    ):
        reject_invalid_state("ion artifact source manifest digests are invalid")
    properties = {
        "faninsar:pair": list(pair),
        "faninsar:looks": [int(looks[0]), int(looks[1])],
        "faninsar:domain": domain,
        "faninsar:shape": [int(shape[0]), int(shape[1])],
        "faninsar:wavelength_m": wavelength_m,
        "faninsar:grid_identity": grid_identity,
        "faninsar:method": {
            "name": method_name,
            "parameters": _canonical_parameters(
                dict(method_parameters), "ion method parameters"
            ),
        },
        "faninsar:degraded": {"flag": bool(degraded), "reason": degradation_reason},
        "faninsar:runtime_fingerprint": runtime_fingerprint,
        "faninsar:devices": list(devices_tuple),
    }

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        for name, array in arrays.items():
            write_zarr_array(staging / f"{name}.zarr", array)
        return {name: {"href": f"{name}.zarr"} for name in arrays}

    product = commit_product(
        root,
        kind=ION_KIND,
        properties=properties,
        write_payloads=write,
        inputs=sources,
        overwrite=replace_existing,
    )
    logger.info("Committed ion product %s for pair %s", product.path, pair)
    return IonosphereArtifactStore.open(product.path)


def ion_dir(work_dir: str | Path, pair_id: str, looks: tuple[int, int]) -> Path:
    """Return the ion product directory ``work_dir/ion/ml_AxR/<pair>``."""
    return Path(work_dir) / "ion" / f"ml_{int(looks[0])}x{int(looks[1])}" / pair_id


def _current_unwrap(store: InterferogramArtifactStore) -> Product:
    """Open the unwrap product bound to this exact IFG."""
    unwrap = open_product(store.unwrap_root, kind=UNWRAP_KIND)
    if unwrap.inputs.get("ifg") != store.write_id:
        reject_invalid_state("unwrap artifact is bound to a different IFG")
    return unwrap


def write_ion_correction_artifact(
    root: str | Path,
    *,
    unwrapped_phase: np.ndarray,
    unwrap_write_id: str,
    ion_write_id: str,
    ion_looks: tuple[int, int],
    degraded_consumed: bool,
    replace_existing: bool = False,
) -> Product:
    """Commit the ionosphere-corrected unwrapped phase of one Pair.

    The product is the ``ion_corrected`` child of the Pair's unwrap product;
    upstream IFG, unwrap, and ion layers are never rewritten.

    Parameters
    ----------
    root : str or pathlib.Path
        The interferogram product directory ``ifg/ml_AxR/<pair>``.
    unwrapped_phase : numpy.ndarray
        Corrected unwrapped phase on the unwrap grid.
    unwrap_write_id : str
        ``write_id`` of the unwrap product that was corrected.
    ion_write_id : str
        ``write_id`` of the ion product that was subtracted.
    ion_looks : tuple[int, int]
        Look factors of the consumed ion product.
    degraded_consumed : bool
        Whether a degraded ion product was consumed under explicit admission.
    replace_existing : bool, optional
        Replace an existing correction.

    Returns
    -------
    Product
        The committed correction product.

    """
    store = InterferogramArtifactStore.open(root)
    unwrap = _current_unwrap(store)
    phase = np.asarray(unwrapped_phase)
    if phase.shape != store.shape or phase.dtype.kind != "f":
        reject_invalid_state(
            "corrected phase must match the IFG shape as a float array"
        )
    if unwrap_write_id != unwrap.write_id or not _is_source_identity(ion_write_id):
        reject_invalid_state("ion correction binding is invalid")
    properties = {
        "faninsar:pair": list(store.pair),
        "faninsar:shape": list(store.shape),
        "faninsar:ion_looks": [int(ion_looks[0]), int(ion_looks[1])],
        "faninsar:degraded_consumed": bool(degraded_consumed),
    }

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        write_zarr_array(
            staging / "unwrapped_phase.zarr", phase.astype(np.float32, copy=False)
        )
        return {"unwrapped_phase": {"href": "unwrapped_phase.zarr"}}

    product = commit_product(
        store.unwrap_root / ION_CORRECTED_NAME,
        kind=ION_CORRECTED_KIND,
        properties=properties,
        write_payloads=write,
        inputs={"unwrap": unwrap.write_id, "ion": ion_write_id},
        overwrite=replace_existing,
    )
    logger.info("Committed ion correction %s", product.path)
    return product


def open_ion_correction(root: str | Path) -> Product:
    """Open the ionosphere correction bound to the current unwrap of an IFG.

    Parameters
    ----------
    root : str or pathlib.Path
        The interferogram product directory ``ifg/ml_AxR/<pair>``.

    Raises
    ------
    InvalidProcessingStateError
        If the correction is missing or bound to another unwrap.

    """
    store = InterferogramArtifactStore.open(root)
    unwrap = _current_unwrap(store)
    corrected = open_product(
        store.unwrap_root / ION_CORRECTED_NAME, kind=ION_CORRECTED_KIND
    )
    if corrected.inputs.get("unwrap") != unwrap.write_id:
        reject_invalid_state("ion correction is bound to a different unwrap")
    return corrected


def read_ion_correction_artifact(root: str | Path) -> np.ndarray:
    """Read the ionosphere-corrected unwrapped phase of one Pair."""
    corrected = open_ion_correction(root)
    phase = read_zarr_array(corrected.asset_path("unwrapped_phase"))
    expected = tuple(int(size) for size in corrected.properties["faninsar:shape"])
    if phase.shape != expected or phase.dtype.kind != "f":
        reject_invalid_state("ion correction layer shape or dtype is incompatible")
    return phase


__all__ = [
    "ION_CORRECTED_KIND",
    "ION_KIND",
    "IonosphereArtifact",
    "IonosphereArtifactStore",
    "ion_dir",
    "open_ion_correction",
    "read_ion_correction_artifact",
    "write_ion_correction_artifact",
    "write_ionosphere_artifact",
]
