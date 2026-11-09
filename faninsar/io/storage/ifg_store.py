"""Interferogram, unwrap, and ion-corrected products of a Stack.

An interferogram product is ``ifg/ml_AxR/<pair>/`` with Zarr layers and one
``item.json`` (kind ``ifg``) that binds both scene write ids. The persisted
layers are the canonical ``complex_ifg`` plus optional ``coherence`` and
``valid_mask``; ``wrapped_phase`` and ``amplitude`` are virtual layers derived
as ``angle(complex_ifg)`` and ``abs(complex_ifg)`` at read time and never
materialize separate Zarr stores. Ordinary unwrap is the pair sibling
``ifg/ml_AxR/<pair>/unwrap`` (kind ``unwrap``). Ion estimates live under
``ifg/ion/ml_AxR/<pair>``. The ion-corrected complex IFG is the pair sibling
``complex_ion/`` (kind ``complex_ion``) written before any full-band unwrap;
its unwrap is ``unwrap_ion/`` (kind ``unwrap_ion``). Products are written
through :mod:`faninsar.io.storage.product`; reads validate the Item, shapes,
and dtypes without re-hashing payloads (PROPOSAL-0059, PROPOSAL-0080).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from faninsar.io.storage._zarr_stac import (
    read_zarr_array,
    require_basename,
    validate_spatial_metadata,
    write_zarr_array,
)
from faninsar.io.storage.product import (
    Product,
    is_write_id,
    open_product,
    publish_product,
)
from faninsar.logging import setup_logger
from faninsar.processing.errors import (
    InvalidProcessingStateError,
    reject_invalid_state,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)

IFG_KIND = "ifg"
UNWRAP_KIND = "unwrap"
COMPLEX_ION_KIND = "complex_ion"
UNWRAP_ION_KIND = "unwrap_ion"
MERGED_REPRESENTATION = "merged"
BURST_REPRESENTATION = "burst"
#: Persisted IFG layers; ``wrapped_phase`` and ``amplitude`` are virtual.
_IFG_LAYERS = ("complex", "coherence", "valid_mask")
_REQUIRED_IFG_LAYERS = ("complex",)
#: Virtual layers derived from the canonical ``complex_ifg``.
_VIRTUAL_IFG_LAYERS = ("wrapped_phase", "amplitude")
_UNWRAP_LAYERS = ("unwrapped_phase", "connected_components")
_MASKED_BILINEAR = "masked_bilinear_unwrapped"
_UNWRAP_SOURCES = ("complex", "complex_ion")
#: Public layer name → (branch, real asset, derivation).
_LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "complex": ("ifg", "complex", "identity"),
    "wrapped": ("ifg", "complex", "angle"),
    "wrapped_phase": ("ifg", "complex", "angle"),
    "amplitude": ("ifg", "complex", "abs"),
    "coherence": ("ifg", "coherence", "identity"),
    "valid_mask": ("ifg", "valid_mask", "identity"),
    "unwrapped": ("unwrap", "unwrapped_phase", "identity"),
    "unw_phase": ("unwrap", "unwrapped_phase", "identity"),
    "unwrapped_phase": ("unwrap", "unwrapped_phase", "identity"),
    "connected_components": ("unwrap", "connected_components", "identity"),
    "complex_ion": ("complex_ion", "complex", "identity"),
    "wrapped_ion": ("complex_ion", "complex", "angle"),
    "amplitude_ion": ("complex_ion", "complex", "abs"),
    "unwrapped_ion": ("unwrap_ion", "unwrapped_phase", "identity"),
    "connected_components_ion": ("unwrap_ion", "connected_components", "identity"),
}


def _derive_virtual_layer(
    name: str,
    complex_ifg: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    """Derive one virtual IFG layer from the canonical complex IFG.

    ``wrapped_phase`` is ``angle(complex_ifg)`` and ``amplitude`` is
    ``abs(complex_ifg)``, keeping the complex real precision. Samples where a
    persisted ``valid_mask`` is False are NaN in both derived layers; NaN
    complex samples propagate NaN naturally.

    Parameters
    ----------
    name : {"wrapped_phase", "amplitude"}
        Virtual layer to derive.
    complex_ifg : numpy.ndarray
        Canonical complex IFG samples.
    valid_mask : numpy.ndarray or None
        Persisted support mask; ``None`` derives without masking.

    Returns
    -------
    numpy.ndarray
        Derived floating-point layer.

    """
    if name == "wrapped_phase":
        derived = np.asarray(np.angle(complex_ifg))
    else:
        derived = np.asarray(np.abs(complex_ifg))
    if valid_mask is not None:
        derived[~valid_mask] = np.nan
    return derived


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


def ifg_artifact_dir(
    work_dir: str | Path,
    looks: tuple[int, int],
    pair: tuple[str, str],
    *,
    representation: str = MERGED_REPRESENTATION,
    burst_id: str | None = None,
    polygon: int | None = None,
) -> Path:
    """Return the product directory of one merged or burst interferogram.

    Merged products keep ``<work>/ifg/ml_AxR/<pair>[/pN]``. Burst products
    live at ``<work>/ifg/bursts/ml_AxR/<pair>/<burst_id>[/pN]``.
    """
    if representation not in {MERGED_REPRESENTATION, BURST_REPRESENTATION}:
        reject_invalid_state("IFG representation must be merged or burst")
    pair_id = f"{pair[0]}_{pair[1]}"
    looks_name = f"ml_{int(looks[0])}x{int(looks[1])}"
    root = Path(work_dir) / "ifg"
    if representation == BURST_REPRESENTATION:
        if burst_id is None:
            reject_invalid_state("burst IFG path requires a burst_id")
        require_basename(burst_id, "IFG burst_id")
        target = root / "bursts" / looks_name / pair_id / burst_id
    else:
        target = root / looks_name / pair_id
    if polygon is not None:
        target = target / f"p{int(polygon)}"
    return target


def unwrap_root(
    ifg_root: str | Path,
    *,
    allow_burst: bool = False,
    representation: str | None = None,
) -> Path:
    """Return the unwrap product directory paired with an IFG product.

    ``<work>/ifg/ml_AxR/<pair>`` maps to ``<work>/ifg/ml_AxR/<pair>/unwrap``.
    Burst products are rejected unless ``allow_burst=True``, in which case
    unwrap is the sibling ``unwrap/`` under the burst IFG root.
    """
    root = _validated_ifg_root(
        ifg_root, allow_burst=allow_burst, representation=representation
    )
    return root / "unwrap"


def complex_ion_root(ifg_root: str | Path) -> Path:
    """Return the ion-corrected complex product directory of an IFG pair."""
    return Path(ifg_root) / "complex_ion"


def unwrap_ion_root(ifg_root: str | Path) -> Path:
    """Return the unwrap product directory of the ion-corrected complex IFG."""
    return Path(ifg_root) / "unwrap_ion"


def _validated_ifg_root(
    ifg_root: str | Path,
    *,
    allow_burst: bool = False,
    representation: str | None = None,
) -> Path:
    """Reject burst unwrap unless explicitly allowed; return the IFG root."""
    root = Path(ifg_root)
    resolved = representation
    item = root / "item.json"
    if resolved is None and item.is_file():
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            payload = {}
        properties = payload.get("properties") if isinstance(payload, dict) else None
        if isinstance(properties, dict):
            resolved = str(
                properties.get("faninsar:representation", MERGED_REPRESENTATION)
            )
    if "bursts" in root.parts or resolved == BURST_REPRESENTATION:
        if not allow_burst:
            reject_invalid_state(
                "unwrap consumes merged IFGs; burst products require allow_burst=True"
            )
        if root.parent.parent.parent.name != "bursts":
            reject_invalid_state(
                "burst IFG product is not under "
                f"ifg/bursts/ml_AxR/<pair>/<burst>: {root}"
            )
        return root
    if root.parent.parent.name != "ifg":
        reject_invalid_state(
            f"IFG product is not under an ifg/ml_AxR/<pair> layout: {root}"
        )
    return root


def ifg_sample_placement(
    store: InterferogramArtifactStore,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return native-pixel first-sample center and spacing of an IFG grid.

    Native pixel centers are integers. The first contributing sample of the
    first block is ``row_origin + look_origin * looks``; the published center
    is that origin plus ``(L - 1) / 2`` with spacing ``L``.
    """
    from faninsar.processing.atmosphere.grid_map import native_block_center_anchor

    native_origin = (
        int(store.row_origin) + int(store.look_origin[0]) * int(store.looks[0]),
        int(store.col_origin) + int(store.look_origin[1]) * int(store.looks[1]),
    )
    return native_block_center_anchor(native_origin, store.looks), (
        float(store.looks[0]),
        float(store.looks[1]),
    )


@dataclass(frozen=True, slots=True)
class ResolvedIfgLayer:
    """Public IFG layer mapped onto one product, real asset, and derivation.

    Attributes
    ----------
    product_root : pathlib.Path
        Directory of the product that owns the real asset.
    real_asset : str
        Persisted asset name inside that product.
    derivation : {"identity", "angle", "abs"}
        How the public layer is obtained from ``real_asset``.
    branch : {"ifg", "unwrap", "complex_ion", "unwrap_ion"}
        Owning product kind relative to the pair IFG.

    """

    product_root: Path
    real_asset: str
    derivation: str
    branch: str


def resolve_ifg_layer(ifg_root: str | Path, name: str) -> ResolvedIfgLayer:
    """Map a public IFG layer name onto its owning product and derivation.

    Ordinary names read the IFG or its ``unwrap/`` sibling. Ion names
    (``complex_ion``, ``wrapped_ion``, ``amplitude_ion``, ``unwrapped_ion``,
    ``connected_components_ion``) never fall back to the ordinary branch.

    Parameters
    ----------
    ifg_root : str or pathlib.Path
        Interferogram product directory.
    name : str
        Public layer name, including short aliases ``wrapped`` and
        ``unwrapped``.

    Returns
    -------
    ResolvedIfgLayer
        Product root, real asset, and derivation.

    Raises
    ------
    InvalidProcessingStateError
        If ``name`` is not a known public IFG layer.

    """
    spec = _LAYER_SPECS.get(str(name))
    if spec is None:
        reject_invalid_state(f"unknown IFG layer: {name!r}")
    branch, real_asset, derivation = spec
    root = Path(ifg_root)
    if branch == "ifg":
        product_root = root
    elif branch == "unwrap":
        product_root = unwrap_root(root)
    elif branch == "complex_ion":
        product_root = complex_ion_root(root)
    else:
        product_root = unwrap_ion_root(root)
    return ResolvedIfgLayer(
        product_root=product_root,
        real_asset=real_asset,
        derivation=derivation,
        branch=branch,
    )


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
    complex_ifg : numpy.ndarray
        Canonical complex IFG layer sharing one grid with the other layers.
    wrapped_phase, amplitude : numpy.ndarray
        Virtual layers derived from ``complex_ifg`` at read time as
        ``angle`` and ``abs``; samples masked by ``valid_mask`` are NaN.
    coherence : numpy.ndarray or None
        Optional coherence layer.
    valid_mask : numpy.ndarray or None
        Optional authoritative support mask.

    """

    complex: np.ndarray
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
    representation: str = MERGED_REPRESENTATION
    burst_id: str | None = None
    row_origin: int = 0
    col_origin: int = 0
    grid_shape: tuple[int, int] | None = None
    look_origin: tuple[int, int] = (0, 0)
    layout_id: str = ""
    polygon_identity: str = ""

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
        representation = str(p.get("faninsar:representation", MERGED_REPRESENTATION))
        if representation not in {MERGED_REPRESENTATION, BURST_REPRESENTATION}:
            reject_invalid_state("artifact representation must be merged or burst")
        raw_burst_id = p.get("faninsar:burst_id")
        burst_id = (
            require_basename(raw_burst_id, "IFG burst_id")
            if raw_burst_id is not None
            else None
        )
        if representation == BURST_REPRESENTATION and burst_id is None:
            reject_invalid_state("burst IFG artifacts require faninsar:burst_id")
        raw_grid_shape = p.get("faninsar:grid_shape", list(shape))
        if not isinstance(raw_grid_shape, list) or len(raw_grid_shape) != 2:
            reject_invalid_state("artifact grid_shape is invalid")
        parent_grid = (int(raw_grid_shape[0]), int(raw_grid_shape[1]))
        raw_look_origin = p.get("faninsar:look_origin", [0, 0])
        if not isinstance(raw_look_origin, list) or len(raw_look_origin) != 2:
            reject_invalid_state("artifact look_origin is invalid")
        look_origin = (int(raw_look_origin[0]), int(raw_look_origin[1]))
        layout_id = str(
            p.get("faninsar:layout_id") or product.inputs.get("layout") or ""
        )
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
            representation=representation,
            burst_id=burst_id,
            row_origin=int(p.get("faninsar:row_origin", 0)),
            col_origin=int(p.get("faninsar:col_origin", 0)),
            grid_shape=parent_grid,
            look_origin=look_origin,
            layout_id=layout_id,
            polygon_identity=str(p.get("faninsar:polygon_identity", "")),
        )

    def close(self) -> None:
        """Release nothing; products are not leased."""

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release nothing."""

    def _layer(
        self, name: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        path = self.root / str(self._payloads[name]["href"])
        try:
            array = (
                read_zarr_array(path)
                if window is None
                else read_zarr_array(path, window=window)
            )
        except (OSError, ValueError, EOFError) as error:
            reject_invalid_state(f"IFG layer {name} cannot be loaded: {error}")
        expected = (
            self.shape
            if window is None
            else (window[1] - window[0], window[3] - window[2])
        )
        if array.shape != expected:
            reject_invalid_state(f"IFG layer {name} shape differs from the Item")
        return array

    def _check_layer_dtype(self, name: str, array: np.ndarray) -> None:
        """Validate the dtype of one persisted IFG layer."""
        if name == "complex" and array.dtype.kind != "c":
            reject_invalid_state("artifact IFG layer dtypes are incompatible")
        if name == "coherence" and array.dtype.kind != "f":
            reject_invalid_state("artifact IFG layer dtypes are incompatible")
        if name == "valid_mask" and array.dtype != np.bool_:
            reject_invalid_state("artifact IFG valid_mask dtype is incompatible")

    def _read_persisted_layer(
        self, name: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Read and validate one persisted IFG layer."""
        if name not in _IFG_LAYERS:
            reject_invalid_state(f"unknown IFG layer: {name!r}")
        if name not in self._payloads:
            reject_invalid_state(f"IFG product has no layer {name!r}")
        array = self._layer(name, window=window)
        self._check_layer_dtype(name, array)
        return array

    def read_layer(
        self, name: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Read one canonical IFG layer by name.

        Parameters
        ----------
        name : str
            Public layer name. Ordinary names include ``complex``,
            ``wrapped``/``wrapped_phase``, ``amplitude``, ``coherence``, and
            ``valid_mask``. Ion names ``complex_ion``, ``wrapped_ion``, and
            ``amplitude_ion`` read the ``complex_ion`` sibling; they never
            fall back to the ordinary complex IFG. ``wrapped_phase`` and
            ``amplitude`` (and their ion counterparts) are virtual
            ``angle``/``abs`` derivations and are never persisted. Where a
            persisted ``valid_mask`` is False, derived samples are NaN.
        window : tuple of int, optional
            Local half-open ``(r0, r1, c0, c1)`` in the layer array.

        Returns
        -------
        numpy.ndarray
            Layer samples in ``(azimuth, range)`` order; derived layers keep
            the complex real precision.

        Raises
        ------
        InvalidProcessingStateError
            If the layer name is unknown, the named branch is missing or
            stale, the persisted layer is absent or damaged, or its dtype or
            shape is unexpected.

        """
        resolved = resolve_ifg_layer(self.root, name)
        if resolved.branch == "ifg":
            if resolved.derivation == "angle":
                return self._virtual_ifg_layer("wrapped_phase", window=window)
            if resolved.derivation == "abs":
                return self._virtual_ifg_layer("amplitude", window=window)
            return self._read_persisted_layer(resolved.real_asset, window=window)
        if resolved.branch == "complex_ion":
            return self._read_complex_ion_layer(resolved, window=window)
        product = (
            self.unwrap_product()
            if resolved.branch == "unwrap"
            else self._require_unwrap_ion()
        )
        if resolved.real_asset not in product.assets:
            reject_invalid_state(
                f"{resolved.branch} product has no layer {resolved.real_asset!r}"
            )
        array = read_zarr_array(product.asset_path(resolved.real_asset), window=window)
        expected = (
            self.shape
            if window is None
            else (window[1] - window[0], window[3] - window[2])
        )
        if array.shape != expected:
            reject_invalid_state(
                f"{resolved.branch} layer {resolved.real_asset} "
                "shape differs from the IFG"
            )
        return array

    def _virtual_ifg_layer(
        self, name: str, window: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Derive one ordinary virtual IFG layer from the canonical complex."""
        complex_ifg = self._read_persisted_layer("complex", window=window)
        valid_mask = (
            self._read_persisted_layer("valid_mask", window=window)
            if "valid_mask" in self._payloads
            else None
        )
        return _derive_virtual_layer(name, complex_ifg, valid_mask)

    def _read_complex_ion_layer(
        self,
        resolved: ResolvedIfgLayer,
        window: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Read one layer of the current, non-stale complex_ion product."""
        product = self._require_complex_ion()
        if resolved.real_asset not in product.assets:
            reject_invalid_state("complex_ion product is missing the complex layer")
        array = product.read(resolved.real_asset, window=window)
        valid_mask = (
            product.read("valid_mask", window=window)
            if "valid_mask" in product.assets
            else None
        )
        if resolved.derivation == "identity":
            return array
        derived_name = (
            "wrapped_phase" if resolved.derivation == "angle" else "amplitude"
        )
        return _derive_virtual_layer(derived_name, array, valid_mask)

    def _require_complex_ion(self) -> Product:
        """Open the complex_ion sibling bound to this IFG and the current ion."""
        root = complex_ion_root(self.root)
        if not (root / "item.json").is_file():
            reject_invalid_state(f"IFG product {self.root} has no complex_ion sibling")
        product = open_product(root, kind=COMPLEX_ION_KIND)
        if product.inputs.get("ifg") != self.write_id:
            reject_invalid_state("complex_ion is bound to a different IFG")
        ion_write_id = product.inputs.get("ion")
        if not isinstance(ion_write_id, str) or not ion_write_id:
            reject_invalid_state("complex_ion is missing its ion binding")
        current_ion = self._current_ion_write_id(product)
        if current_ion != ion_write_id:
            reject_invalid_state("complex_ion is bound to a replaced ion product")
        return product

    def _current_ion_write_id(self, complex_ion: Product) -> str | None:
        """Return the write id of the ion product this complex_ion named."""
        from faninsar.io.storage.ion_store import ION_KIND, ion_dir

        raw_looks = complex_ion.properties.get("faninsar:ion_looks")
        if (
            not isinstance(raw_looks, list)
            or len(raw_looks) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in raw_looks
            )
        ):
            reject_invalid_state("complex_ion ion_looks metadata is invalid")
        pair_id = f"{self.pair[0]}_{self.pair[1]}"
        ion_looks = (int(raw_looks[0]), int(raw_looks[1]))
        ion_path = ion_dir(self.root.parent.parent.parent, pair_id, ion_looks)
        if not (ion_path / "item.json").is_file():
            return None
        return open_product(ion_path, kind=ION_KIND).write_id

    def _require_unwrap_ion(self) -> Product:
        """Open unwrap_ion bound to the current complex_ion of this IFG."""
        root = unwrap_ion_root(self.root)
        if not (root / "item.json").is_file():
            reject_invalid_state(f"IFG product {self.root} has no unwrap_ion sibling")
        product = open_product(root, kind=UNWRAP_ION_KIND)
        if product.inputs.get("ifg") != self.write_id:
            reject_invalid_state("unwrap_ion is bound to a different IFG")
        complex_ion = self._require_complex_ion()
        if product.inputs.get("complex_ion") != complex_ion.write_id:
            reject_invalid_state(
                "unwrap_ion is bound to a replaced complex_ion product"
            )
        if product.inputs.get("ion") != complex_ion.inputs.get("ion"):
            reject_invalid_state("unwrap_ion is bound to a replaced ion product")
        return product

    def read(
        self, window: tuple[int, int, int, int] | None = None
    ) -> InterferogramArtifact:
        """Read all IFG layers, deriving wrapped phase and amplitude.

        The canonical ``complex_ifg`` is loaded once; ``wrapped_phase`` and
        ``amplitude`` are derived from it and are NaN where a persisted
        ``valid_mask`` is False.

        Returns
        -------
        InterferogramArtifact
            Validated layers sharing one grid.

        Raises
        ------
        InvalidProcessingStateError
            If any persisted layer is damaged or has an unexpected dtype or
            shape.

        """
        complex_ifg = self._read_persisted_layer("complex", window=window)
        coherence = (
            self._read_persisted_layer("coherence", window=window)
            if "coherence" in self._payloads
            else None
        )
        valid_mask = (
            self._read_persisted_layer("valid_mask", window=window)
            if "valid_mask" in self._payloads
            else None
        )
        return InterferogramArtifact(
            complex=complex_ifg,
            coherence=coherence,
            wrapped_phase=_derive_virtual_layer(
                "wrapped_phase", complex_ifg, valid_mask
            ),
            amplitude=_derive_virtual_layer("amplitude", complex_ifg, valid_mask),
            valid_mask=valid_mask,
        )

    @property
    def unwrap_root(self) -> Path:
        """Return the unwrap product directory for this Pair."""
        return unwrap_root(self.root, representation=self.representation)

    def has_unwrapped(self) -> bool:
        """Return whether a current unwrap product is bound to this IFG.

        Burst IFGs return ``False`` without traversing the unwrap root.
        Unwrap consumes merged products; ``unwrap_root`` still rejects
        burst paths unless ``allow_burst=True``.
        """
        if self.representation == BURST_REPRESENTATION:
            return False
        item = self.unwrap_root / "item.json"
        if not item.is_file():
            return False
        product = open_product(self.unwrap_root, kind=UNWRAP_KIND)
        return product.inputs.get("ifg") == self.write_id

    def has_unwrap_ion(self) -> bool:
        """Return whether a current unwrap_ion product is bound to this IFG.

        Stale siblings bound to a replaced IFG, complex_ion, or ion estimate
        count as absent so Network admission cannot mix branches.
        """
        if self.representation == BURST_REPRESENTATION:
            return False
        if not (unwrap_ion_root(self.root) / "item.json").is_file():
            return False
        try:
            self._require_unwrap_ion()
        except InvalidProcessingStateError:
            return False
        return True

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

    def complex_ion_product(self) -> Product:
        """Open the ion-corrected complex sibling bound to this IFG."""
        return self._require_complex_ion()

    def unwrap_ion_product(self) -> Product:
        """Open the unwrap_ion sibling bound to the current complex_ion."""
        return self._require_unwrap_ion()

    def read_unwrapped_layer(self, name: str, *, source: str = "complex") -> np.ndarray:
        """Read one named layer of the unwrap product bound to this IFG."""
        product = (
            self.unwrap_product() if source == "complex" else self._require_unwrap_ion()
        )
        if name not in product.assets:
            reject_invalid_state(f"unwrap product has no layer {name!r}")
        return read_zarr_array(product.asset_path(name))

    def read_unwrapped(self, *, source: str = "complex") -> UnwrappedArtifact:
        """Read the unwrap product bound to this exact IFG.

        Parameters
        ----------
        source : {"complex", "complex_ion"}, optional
            Ordinary unwrap (default) or the unwrap of the ion-corrected
            complex IFG. A missing or stale named branch raises; it never
            falls back to the other branch.

        Raises
        ------
        InvalidProcessingStateError
            If the named unwrap product is missing, stale, or bound to
            another IFG.

        """
        if source not in _UNWRAP_SOURCES:
            reject_invalid_state("unwrap source must be 'complex' or 'complex_ion'")
        product = (
            self.unwrap_product() if source == "complex" else self._require_unwrap_ion()
        )
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
    coherence: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    replace_existing: bool = False,
    representation: str = MERGED_REPRESENTATION,
    burst_id: str | None = None,
    row_origin: int = 0,
    col_origin: int = 0,
    grid_shape: tuple[int, int] | None = None,
    look_origin: tuple[int, int] = (0, 0),
    layout_id: str | None = None,
    polygon_identity: str | None = None,
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
    complex_ifg : numpy.ndarray
        Required canonical complex layer.
    coherence, valid_mask : numpy.ndarray, optional
        Optional layers. ``wrapped_phase`` and ``amplitude`` are virtual
        layers derived from ``complex_ifg`` at read time and are never
        persisted.
    replace_existing : bool, optional
        Replace an existing product.
    representation : {"merged", "burst"}, optional
        Product kind. Merged is the default unwrap/network input.
    burst_id : str, optional
        Basename burst identifier; required when ``representation="burst"``.
    row_origin, col_origin : int, optional
        Unit placement on the parent scene grid, in SLC pixels.
    grid_shape : tuple[int, int], optional
        Parent looked grid shape. Defaults to this product's array shape.
    look_origin : tuple[int, int], optional
        Look-grid origin of this product on the parent looked grid.
    layout_id : str, optional
        Write id of the separate ``merge_layout`` product, if any.
    polygon_identity : str, optional
        Multi-polygon ROI identity recorded on the Item.

    Returns
    -------
    InterferogramArtifactStore
        The committed product.

    """
    arrays: dict[str, np.ndarray] = {"complex": np.asarray(complex_ifg)}
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
    if arrays["complex"].dtype.kind != "c" or (
        coherence is not None and arrays["coherence"].dtype.kind != "f"
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
    if representation not in {MERGED_REPRESENTATION, BURST_REPRESENTATION}:
        reject_invalid_state("IFG representation must be merged or burst")
    resolved_burst_id = (
        require_basename(burst_id, "IFG burst_id") if burst_id is not None else None
    )
    if representation == BURST_REPRESENTATION and resolved_burst_id is None:
        reject_invalid_state("burst IFG artifacts require a burst_id")
    if representation == MERGED_REPRESENTATION and resolved_burst_id is not None:
        reject_invalid_state("merged IFG artifacts must not carry a burst_id")
    parent_grid = (
        (int(shape[0]), int(shape[1]))
        if grid_shape is None
        else (int(grid_shape[0]), int(grid_shape[1]))
    )
    if len(look_origin) != 2:
        reject_invalid_state("IFG look_origin must be a 2-tuple")
    resolved_look_origin = (int(look_origin[0]), int(look_origin[1]))
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
        if name in {"primary", "secondary", "layout"}
    }
    if layout_id:
        inputs["layout"] = str(layout_id)
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
        "faninsar:representation": representation,
        "faninsar:row_origin": int(row_origin),
        "faninsar:col_origin": int(col_origin),
        "faninsar:grid_shape": [int(parent_grid[0]), int(parent_grid[1])],
        "faninsar:look_origin": [resolved_look_origin[0], resolved_look_origin[1]],
        "faninsar:layout_id": str(layout_id or ""),
        "faninsar:polygon_identity": str(polygon_identity or ""),
    }
    if resolved_burst_id is not None:
        properties["faninsar:burst_id"] = resolved_burst_id
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

    product = publish_product(
        root,
        kind=IFG_KIND,
        properties=properties,
        write_payloads=write,
        inputs=inputs,
        overwrite=replace_existing,
    )
    logger.info("Committed IFG product %s for pair %s", product.path, pair)
    return InterferogramArtifactStore.open(product.path)


def _placement_pair(value: Sequence[float], name: str) -> list[float]:
    """Validate a two-element finite placement pair for product metadata."""
    try:
        pair = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        reject_invalid_state(f"{name} must be two finite numbers")
    if len(value) != 2 or not np.isfinite(pair).all():
        reject_invalid_state(f"{name} must be two finite numbers")
    return [pair[0], pair[1]]


def write_complex_ion_artifact(
    root: str | Path,
    *,
    complex_ifg: np.ndarray,
    valid_mask: np.ndarray | None = None,
    ion_write_id: str,
    ion_looks: tuple[int, int],
    interpolation: str,
    source_anchor: Sequence[float],
    source_spacing: Sequence[float],
    target_anchor: Sequence[float],
    target_spacing: Sequence[float],
    replace_existing: bool = False,
) -> Product:
    """Commit the ion-corrected complex IFG sibling of one pair.

    The product is ``<ifg>/complex_ion`` (kind ``complex_ion``) with a
    persisted ``complex.zarr`` written before any full-band unwrap. It binds
    the IFG write id and the ion write id; replacing the parent IFG removes
    it, replacing the shared ion makes it stale.

    Parameters
    ----------
    root : str or pathlib.Path
        Interferogram product directory.
    complex_ifg : numpy.ndarray
        Corrected complex samples on the IFG grid.
    valid_mask : numpy.ndarray, optional
        Support of the correction (IFG validity ∩ mapped ion support).
    ion_write_id : str
        ``write_id`` of the consumed ion product.
    ion_looks : tuple[int, int]
        Look factors of that ion product.
    interpolation : str
        Interpolation policy; must be ``masked_bilinear_unwrapped``.
    source_anchor, source_spacing, target_anchor, target_spacing : sequence
        Native-pixel placement of the ion grid and the target IFG grid.
    replace_existing : bool, optional
        Replace an existing ``complex_ion`` sibling. The ordinary unwrap
        sibling is left in place.

    Returns
    -------
    Product
        The committed ``complex_ion`` product.

    """
    store = InterferogramArtifactStore.open(root)
    array = np.asarray(complex_ifg)
    if array.shape != store.shape or array.dtype.kind != "c":
        reject_invalid_state("complex_ion must match the IFG shape as a complex array")
    mask = None if valid_mask is None else np.asarray(valid_mask)
    if mask is not None and (mask.shape != store.shape or mask.dtype != np.bool_):
        reject_invalid_state("complex_ion valid_mask must match the IFG as a boolean")
    if interpolation != _MASKED_BILINEAR:
        reject_invalid_state(
            "complex_ion interpolation must be 'masked_bilinear_unwrapped'"
        )
    if (
        len(ion_looks) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in ion_looks
        )
        or not _is_source_identity(ion_write_id)
    ):
        reject_invalid_state("complex_ion ion binding is invalid")
    item_properties: dict[str, object] = {
        "faninsar:pair": list(store.pair),
        "faninsar:looks": list(store.looks),
        "faninsar:domain": store.domain,
        "faninsar:shape": list(store.shape),
        "faninsar:grid_identity": store.grid_identity,
        "faninsar:ion_looks": [int(ion_looks[0]), int(ion_looks[1])],
        "faninsar:interpolation": interpolation,
        "faninsar:sample_center_anchor": _placement_pair(
            target_anchor, "target_anchor"
        ),
        "faninsar:sample_spacing": _placement_pair(target_spacing, "target_spacing"),
        "faninsar:source_sample_center_anchor": _placement_pair(
            source_anchor, "source_anchor"
        ),
        "faninsar:source_sample_spacing": _placement_pair(
            source_spacing, "source_spacing"
        ),
    }
    if store.crs is not None:
        item_properties["proj:code"] = store.crs
        item_properties["proj:transform"] = list(store.transform or ())

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        write_zarr_array(staging / "complex.zarr", array)
        assets: dict[str, dict[str, Any]] = {"complex": {"href": "complex.zarr"}}
        if mask is not None:
            write_zarr_array(staging / "valid_mask.zarr", mask)
            assets["valid_mask"] = {"href": "valid_mask.zarr"}
        return assets

    product = publish_product(
        complex_ion_root(store.root),
        kind=COMPLEX_ION_KIND,
        properties=item_properties,
        write_payloads=write,
        inputs={"ifg": store.write_id, "ion": ion_write_id},
        overwrite=replace_existing,
    )
    logger.info(
        "Committed complex_ion product %s for pair %s", product.path, store.pair
    )
    return product


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
    source: str = "complex",
) -> Product:
    """Commit the unwrap product of one interferogram.

    Parameters
    ----------
    root : str or pathlib.Path
        The interferogram product directory; ordinary unwrap is written to
        the pair sibling ``unwrap/``, and ion-corrected unwrap to
        ``unwrap_ion/``.
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
        Replace an existing unwrap product of the selected ``source``.
        Sibling products of the other branch are left in place.
    source : {"complex", "complex_ion"}, optional
        Ordinary unwrap of the original IFG, or unwrap of the
        ``complex_ion`` sibling. ``complex_ion`` binds that product and the
        transitive ion write id.

    Returns
    -------
    Product
        The committed unwrap product.

    """
    store = InterferogramArtifactStore.open(root)
    if ifg_manifest_digest != store.write_id:
        reject_invalid_state("unwrap artifact IFG binding is invalid")
    if source not in _UNWRAP_SOURCES:
        reject_invalid_state("unwrap source must be 'complex' or 'complex_ion'")
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
        "faninsar:source": source,
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
    if source == "complex":
        target = store.unwrap_root
        kind = UNWRAP_KIND
        inputs: dict[str, str] = {"ifg": store.write_id}
    else:
        complex_ion = store._require_complex_ion()
        target = unwrap_ion_root(store.root)
        kind = UNWRAP_ION_KIND
        inputs = {
            "ifg": store.write_id,
            "complex_ion": complex_ion.write_id,
            "ion": str(complex_ion.inputs["ion"]),
        }

    def write(staging: Path) -> dict[str, dict[str, Any]]:
        layers = {
            "unwrapped_phase": phase,
            "connected_components": components,
            **extras,
        }
        for name, array in layers.items():
            write_zarr_array(staging / f"{name}.zarr", array)
        return {name: {"href": f"{name}.zarr"} for name in layers}

    target.parent.mkdir(parents=True, exist_ok=True)
    product = publish_product(
        target,
        kind=kind,
        properties=item_properties,
        write_payloads=write,
        inputs=inputs,
        overwrite=replace_existing,
    )
    logger.info("Committed unwrap product %s with method %s", target, method)
    return product


__all__ = [
    "COMPLEX_ION_KIND",
    "IFG_KIND",
    "UNWRAP_ION_KIND",
    "UNWRAP_KIND",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "ResolvedIfgLayer",
    "UnwrappedArtifact",
    "complex_ion_root",
    "ifg_artifact_dir",
    "ifg_sample_placement",
    "resolve_ifg_layer",
    "unwrap_ion_root",
    "unwrap_root",
    "write_complex_ion_artifact",
    "write_ifg_artifact",
    "write_unwrapped_artifact",
]
