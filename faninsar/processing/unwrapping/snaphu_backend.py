"""snaphu-py unwrapping backend with lazy import and no silent fallback."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.core.sar_tools import Looks, coerce_looks
from faninsar.logging import reject, setup_logger
from faninsar.processing.errors import ProcessingContractError, reject_invalid_state
from faninsar.processing.runtime.capabilities import snaphu_capability
from faninsar.processing.unwrapping.common import (
    SpatialUnwrapper,
    SpatialUnwrapResult,
)
from faninsar.processing.unwrapping.errors import NoValidSupportError

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = setup_logger(__name__)

SnaphuCostMode = Literal["defo", "smooth"]

_AUTO_TILE_SIZE = 2048
_AUTO_TILE_OVERLAP = 64
_AUTO_MAX_PROCESSES = 4


class SnaphuNotAvailableError(ProcessingContractError):
    """Raised when the required snaphu-py package is not installed."""


@dataclass(frozen=True, slots=True)
class SnaphuConfig:
    """Typed configuration for the snaphu-py backend.

    ``None`` on ``nlooks``, ``ntiles``, ``nproc``, or ``tile_overlap`` means
    the value is unresolved and is filled for one invocation from the image
    shape, interferogram looks, and Stack divisor. Explicit zeros and
    nonpositive ``nproc`` are preserved. New fields are keyword-only so
    ``SnaphuConfig("defo", 10.0, (2, 2), -1)`` still means ``nproc=-1``.

    Parameters
    ----------
    cost : {"defo", "smooth"}, optional
        SNAPHU statistical-cost mode. Default ``"defo"``.
    nlooks : float, optional
        Effective number of looks forwarded to SNAPHU. ``None`` is resolved
        from interferogram looks and ``nlooks_divisor``, or ``1.0`` for a
        standalone array call without look metadata.
    ntiles : tuple of int, optional
        Tile counts ``(azimuth, range)``. ``None`` uses tiles of 2048
        samples. Explicit ``(1, 1)`` disables tiling.
    nproc : int, optional
        Process count. ``None`` uses a bounded automatic rule. Explicit
        nonpositive values keep snaphu-py's all-CPU meaning.
    tile_overlap : int or tuple of int, optional
        Tile overlap in samples. A scalar is applied to both axes.
        ``None`` uses 64 samples on tiled axes and zero on untiled axes.
        Explicit zero is preserved. Keyword-only.
    single_tile_reoptimize : bool, optional
        Forwarded to snaphu-py. Default ``True``. Keyword-only.

    """

    cost: SnaphuCostMode = "defo"
    nlooks: float | None = None
    ntiles: tuple[int, int] | None = None
    nproc: int | None = None
    tile_overlap: int | tuple[int, int] | None = field(default=None, kw_only=True)
    single_tile_reoptimize: bool = field(default=True, kw_only=True)


def _require_positive_finite(value: object, name: str) -> float:
    """Return ``value`` as a finite number strictly greater than zero."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = float("nan")
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
        reject(f"{name} must be a finite number greater than zero", ValueError)
    return number


def _require_int(value: object, name: str) -> int:
    """Return a non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        reject(f"{name} must be an integer", ValueError)
    return int(value)


def _as_tile_pair(value: object, *, name: str) -> tuple[int, int]:
    """Normalize an int or length-2 sequence into a pair of integers."""
    if isinstance(value, bool):
        reject(f"{name} must be an int or a pair of ints", ValueError)
    if isinstance(value, int):
        item = int(value)
        return item, item
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return _require_int(value[0], name), _require_int(value[1], name)
    return reject(f"{name} must be an int or a pair of ints", ValueError)


def _resolve_nlooks(
    config: SnaphuConfig,
    *,
    looks: Looks | None,
    nlooks_divisor: float | None,
) -> float:
    """Return the SNAPHU ``nlooks`` argument for one invocation."""
    if config.nlooks is not None:
        nlooks = float(config.nlooks)
    elif looks is not None:
        divisor = _require_positive_finite(nlooks_divisor, "nlooks_divisor")
        nlooks = float(looks.nlooks) / divisor
    else:
        nlooks = 1.0
    if not math.isfinite(nlooks) or nlooks < 1.0:
        reject("resolved SNAPHU nlooks must be finite and at least 1", ValueError)
    return nlooks


def _nlooks_diagnostics(
    config: SnaphuConfig,
    *,
    looks: Looks | tuple[int, int] | None,
    nlooks_divisor: float | None,
    nlooks: float,
) -> dict[str, Any]:
    """Return unwrap-product diagnostics for the nlooks resolution path.

    These fields are recorded on the unwrap Item, not in compared method
    identity. An explicit ``SnaphuConfig.nlooks`` bypasses the divisor for
    reuse while still recording the applicable coefficient and looks. The
    ``looks`` field records both factors in range-first order.
    """
    resolved_looks = None if looks is None else coerce_looks(looks)
    if config.nlooks is not None:
        source: Literal["explicit", "multilook_divisor", "default"] = "explicit"
    elif looks is not None:
        source = "multilook_divisor"
    else:
        source = "default"
    payload: dict[str, Any] = {
        "nlooks": float(nlooks),
        "source": source,
        "nlooks_divisor": (None if nlooks_divisor is None else float(nlooks_divisor)),
        "looks": (
            None
            if resolved_looks is None
            else [resolved_looks.range, resolved_looks.azimuth]
        ),
    }
    return json.loads(json.dumps(payload, allow_nan=False))


def _nominal_tile_extent(length: int, n_tiles: int) -> int:
    """Return the nominal extent of one tile along an axis."""
    return max(1, math.ceil(length / n_tiles))


def _capped_overlap(overlap: int, extent: int) -> int:
    """Cap overlap strictly below half of ``extent``."""
    limit = max(0, (extent - 1) // 2)
    return min(overlap, limit)


def _resolve_snaphu_config(
    config: SnaphuConfig,
    *,
    shape: tuple[int, int],
    looks: Looks | tuple[int, int] | None = None,
    nlooks_divisor: float | None = None,
    process_limit: int | None = None,
) -> SnaphuConfig:
    """Fill unresolved SNAPHU execution fields for one image shape.

    Only ``None`` fields are resolved. Explicit ``ntiles=(1, 1)``, overlap
    ``0``, and nonpositive ``nproc`` are forwarded unchanged. Automatic
    overlap is zero on untiled axes and capped below half the nominal tile
    extent; explicit overlap is a nonnegative pair with no invented half
    bound. Automatic ``nproc`` is at least one and at most the tile count,
    CPU count, the private cap, and ``process_limit`` when supplied. The
    2048/64/4 tile rule is a modest heuristic, not a runtime or memory
    guarantee.
    """
    if len(shape) != 2 or any(int(axis) < 1 for axis in shape):
        reject("SNAPHU input shape must be two positive integers", ValueError)
    height, width = int(shape[0]), int(shape[1])
    resolved_looks = None if looks is None else coerce_looks(looks)
    nlooks = _resolve_nlooks(
        config, looks=resolved_looks, nlooks_divisor=nlooks_divisor
    )

    if config.ntiles is None:
        tile_size = _AUTO_TILE_SIZE
        ntiles = (
            max(1, math.ceil(height / tile_size)),
            max(1, math.ceil(width / tile_size)),
        )
    else:
        ntiles = _as_tile_pair(config.ntiles, name="ntiles")
        if ntiles[0] < 1 or ntiles[1] < 1:
            reject("ntiles values must be at least 1", ValueError)
        if ntiles[0] > height or ntiles[1] > width:
            reject("ntiles cannot exceed the image shape", ValueError)

    if config.tile_overlap is None:
        overlap_az = 0 if ntiles[0] == 1 else _AUTO_TILE_OVERLAP
        overlap_rg = 0 if ntiles[1] == 1 else _AUTO_TILE_OVERLAP
        tile_overlap = (
            _capped_overlap(overlap_az, _nominal_tile_extent(height, ntiles[0])),
            _capped_overlap(overlap_rg, _nominal_tile_extent(width, ntiles[1])),
        )
    else:
        tile_overlap = _as_tile_pair(config.tile_overlap, name="tile_overlap")
        if tile_overlap[0] < 0 or tile_overlap[1] < 0:
            reject("tile_overlap values must be non-negative", ValueError)

    if config.nproc is None:
        tile_count = ntiles[0] * ntiles[1]
        if tile_count == 1:
            nproc = 1
        else:
            cpus = os.cpu_count() or 1
            caps = [tile_count, int(cpus), _AUTO_MAX_PROCESSES]
            if process_limit is not None:
                caps.append(_require_int(process_limit, "process_limit"))
            nproc = max(1, min(caps))
    else:
        nproc = _require_int(config.nproc, "nproc")

    if not isinstance(config.single_tile_reoptimize, bool):
        reject("single_tile_reoptimize must be a bool", ValueError)

    return replace(
        config,
        nlooks=nlooks,
        ntiles=ntiles,
        nproc=nproc,
        tile_overlap=tile_overlap,
        single_tile_reoptimize=bool(config.single_tile_reoptimize),
    )


def _snaphu_json_parameters(config: SnaphuConfig) -> dict[str, Any]:
    """Return strict-JSON method parameters for an effective SNAPHU config."""
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _pair_prefix_token(pair_id: str | None) -> str:
    """Return a filesystem-safe Pair token, or empty when none is supplied."""
    if pair_id is None or pair_id == "":
        return ""
    return Path(str(pair_id).replace("\\", "/")).name


@contextmanager
def _snaphu_scratch_directory(work_dir: Path, pair_id: str | None) -> Iterator[str]:
    """Yield one invocation-owned scratch directory under ``work_dir/tmp/snaphu``."""
    scratch_root = Path(work_dir) / "tmp" / "snaphu"
    scratch_root.mkdir(parents=True, exist_ok=True)
    token = _pair_prefix_token(pair_id)
    prefix = "snaphu-" if not token else f"snaphu-{token}-"
    scratch = TemporaryDirectory(dir=scratch_root, prefix=prefix)
    try:
        yield scratch.name
    finally:
        try:
            scratch.cleanup()
        except Exception:
            logger.exception("SNAPHU scratch cleanup failed for %s", scratch.name)
            raise


class Snaphu(SpatialUnwrapper):
    """Adapt the SNAPHU spatial backend to the Torch spatial seam.

    Parameters
    ----------
    config : SnaphuConfig, optional
        Typed configuration passed to ``snaphu-py``.
    work_dir : str or pathlib.Path, optional
        Workspace used for invocation-owned scratch. ``None`` uses
        :func:`pathlib.Path.cwd`.

    Notes
    -----
    The adapter accepts Torch tensors and performs the only NumPy boundary in
    this strategy around the external SNAPHU package. It never falls back to
    another unwrapper. Input tensors are returned on their original device.
    ``work_dir`` and the private Pair prefix are execution context, not
    scientific method identity.

    """

    def __init__(
        self,
        config: SnaphuConfig | None = None,
        *,
        work_dir: str | Path | None = None,
    ) -> None:
        """Store the explicit SNAPHU configuration and scratch workspace."""
        self.config = config or SnaphuConfig()
        self._work_dir = None if work_dir is None else Path(work_dir)
        self._pair_id: str | None = None

    def unwrap(
        self,
        wrapped_phase: Any,
        *,
        coherence: Any | None = None,
        valid_mask: Any | None = None,
    ) -> SpatialUnwrapResult:
        """Unwrap one 2-D Torch phase tensor with SNAPHU.

        Parameters
        ----------
        wrapped_phase : torch.Tensor
            Wrapped phase in radians, ordered ``(azimuth, range)``.
        coherence : torch.Tensor, optional
            Finite values in ``[0, 1]``. Nonfinite values remove support.
        valid_mask : torch.Tensor, optional
            Boolean support mask matching ``wrapped_phase``.

        Returns
        -------
        SpatialUnwrapResult
            The parsed SNAPHU phase and deterministic spatial diagnostics.

        Raises
        ------
        NoValidSupportError
            If no input pixel is supported.
        SnaphuNotAvailableError
            If the external SNAPHU package is unavailable.
        ValueError
            If tensor inputs or SNAPHU configuration violate the contract.

        """
        import torch

        self._validate_tensors(wrapped_phase, coherence, valid_mask)
        support = torch.isfinite(wrapped_phase)
        if valid_mask is not None:
            support &= valid_mask
        if coherence is not None:
            finite = torch.isfinite(coherence)
            finite_values = coherence[finite]
            if torch.any(finite_values < 0) or torch.any(finite_values > 1):
                reject_invalid_state("coherence finite values must be within [0, 1]")
            support &= finite
            backend_coherence = torch.nan_to_num(coherence, nan=0.0)
        else:
            backend_coherence = torch.ones_like(wrapped_phase)
        if not bool(torch.any(support)):
            reject("SNAPHU input has no valid support", NoValidSupportError)

        phase = torch.remainder(wrapped_phase + torch.pi, 2 * torch.pi) - torch.pi
        if coherence is None:
            coherence = backend_coherence
        resolved = _resolve_snaphu_config(
            self.config,
            shape=(int(wrapped_phase.shape[0]), int(wrapped_phase.shape[1])),
        )
        try:
            backend_phase, _backend_components = _snaphu_numpy(
                torch.exp(1j * torch.nan_to_num(phase)).detach().cpu().numpy(),
                backend_coherence.detach().cpu().numpy(),
                config=resolved,
                work_dir=self._work_dir,
                pair_id=self._pair_id,
            )
        except SnaphuNotAvailableError:
            raise
        except Exception:
            logger.exception("SNAPHU backend failed")
            return self._failed_result(phase, support)

        output = torch.as_tensor(
            backend_phase,
            dtype=phase.dtype,
            device=phase.device,
        )
        output_valid = support & torch.isfinite(output)
        if tuple(output.shape) != tuple(phase.shape) or not bool(
            torch.all(torch.isfinite(output[output_valid]))
        ):
            logger.error("SNAPHU returned malformed or nonfinite output")
            return self._failed_result(phase, support)
        labels, references, active_h, active_v = self._labels_and_edges(
            phase, output_valid, coherence
        )
        residual = self._residual_norm(phase, output, active_h, active_v)
        output = output.clone()
        output[~output_valid] = torch.nan
        return SpatialUnwrapResult(
            phase=output,
            valid_mask=output_valid,
            component_labels=labels,
            reference_values=references,
            converged=True,
            iterations=0,
            pcg_iterations=0,
            residual_norm=residual,
            failure_reason=None,
        )

    @staticmethod
    def _validate_tensors(phase: Any, coherence: Any, valid_mask: Any) -> None:
        """Validate Torch tensor shape, dtype, and device constraints."""
        import torch

        if not isinstance(phase, torch.Tensor) or phase.ndim != 2:
            reject_invalid_state("wrapped_phase must be a 2-D torch tensor")
        if not phase.is_floating_point() or phase.is_complex():
            reject_invalid_state("wrapped_phase must be a real floating tensor")
        for name, value in (("coherence", coherence), ("valid_mask", valid_mask)):
            if value is not None and not isinstance(value, torch.Tensor):
                reject_invalid_state(f"{name} must be a torch tensor")
            if value is not None and value.shape != phase.shape:
                reject_invalid_state(f"{name} must match wrapped_phase shape")
            if value is not None and value.device != phase.device:
                reject_invalid_state(f"{name} must share wrapped_phase device")
        if coherence is not None and not coherence.is_floating_point():
            reject_invalid_state("coherence must be a floating tensor")
        if valid_mask is not None and valid_mask.dtype is not torch.bool:
            reject_invalid_state("valid_mask must have dtype torch.bool")

    @staticmethod
    def _labels_and_edges(
        phase: Any,
        support: Any,
        coherence: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Build row-major support labels and positive-quality edge masks."""
        import torch

        height, width = phase.shape
        active_h = support[:, :-1] & support[:, 1:]
        active_v = support[:-1, :] & support[1:, :]
        if coherence is not None:
            active_h &= coherence[:, :-1] > 0
            active_h &= coherence[:, 1:] > 0
            active_v &= coherence[:-1, :] > 0
            active_v &= coherence[1:, :] > 0
        parent = list(range(height * width))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for row, col in zip(*torch.where(active_h), strict=True):
            union(int(row) * width + int(col), int(row) * width + int(col) + 1)
        for row, col in zip(*torch.where(active_v), strict=True):
            union(int(row) * width + int(col), int(row) * width + int(col) + width)
        labels = torch.full_like(support, -1, dtype=torch.int64)
        anchors: list[int] = []
        root_labels: dict[int, int] = {}
        for index in range(height * width):
            row, col = divmod(index, width)
            if not bool(support[row, col]):
                continue
            root = find(index)
            label = root_labels.setdefault(root, len(root_labels))
            labels[row, col] = label
            if label == len(anchors):
                anchors.append(index)
        anchor_tensor = torch.as_tensor(anchors, device=phase.device)
        return labels, phase.flatten()[anchor_tensor], active_h, active_v

    @staticmethod
    def _residual_norm(phase: Any, output: Any, active_h: Any, active_v: Any) -> float:
        """Compute active-edge wrapped-gradient residual norm."""
        import torch

        def wrap(values: Any) -> Any:
            return torch.remainder(values + torch.pi, 2 * torch.pi) - torch.pi

        residual_h = output[:, 1:] - output[:, :-1] - wrap(phase[:, 1:] - phase[:, :-1])
        residual_v = output[1:, :] - output[:-1, :] - wrap(phase[1:, :] - phase[:-1, :])
        values = torch.cat((residual_h[active_h], residual_v[active_v]))
        return float(torch.linalg.vector_norm(values)) if values.numel() else 0.0

    @staticmethod
    def _failed_result(phase: Any, support: Any) -> SpatialUnwrapResult:
        """Return a closed failed result after an admitted backend failure."""
        import torch

        labels = torch.full_like(phase, -1, dtype=torch.int64)
        output = torch.full_like(phase, torch.nan)
        return SpatialUnwrapResult(
            phase=output,
            valid_mask=torch.zeros_like(support, dtype=torch.bool),
            component_labels=labels,
            reference_values=torch.empty(0, dtype=phase.dtype, device=phase.device),
            converged=False,
            iterations=0,
            pcg_iterations=0,
            residual_norm=float("inf"),
            failure_reason="backend_failure",
        )


def snaphu_available() -> bool:
    """Return whether the snaphu package is importable."""
    return find_spec("snaphu") is not None


def require_snaphu() -> Any:
    """Import snaphu lazily or raise a typed capability error.

    Returns
    -------
    module
        The imported ``snaphu`` module.

    Raises
    ------
    SnaphuNotAvailableError
        If the required ``snaphu-py`` package is not installed.

    """
    capability = snaphu_capability()
    if not capability.available:
        message = (
            "snaphu-py is not installed. Reinstall FanInSAR with "
            "`pip install faninsar` "
            f"and review the license caveat: {capability.license_caveat}"
        )
        reject(message, SnaphuNotAvailableError)
    return import_module("snaphu")


def _snaphu_numpy(
    complex_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    config: SnaphuConfig | None = None,
    work_dir: str | Path | None = None,
    pair_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Unwrap a complex interferogram with the snaphu-py API.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Complex interferogram (not phase-only).
    coherence : numpy.ndarray
        Coherence in ``[0, 1]`` matching the interferogram shape.
    config : SnaphuConfig, optional
        Typed snaphu configuration.
    work_dir : str or pathlib.Path, optional
        Workspace for invocation scratch. ``None`` uses the current working
        directory.
    pair_id : str, optional
        Private library-normalized Pair token used only in the scratch prefix.

    Raises
    ------
    SnaphuNotAvailableError
        If snaphu is not installed.
    InvalidProcessingStateError
        If inputs are invalid.

    Notes
    -----
    This backend never silently falls back to IRLS. Callers must choose the
    method explicitly. SNAPHU license terms are surfaced via capabilities.

    """
    if complex_ifg.ndim != 2 or not np.iscomplexobj(complex_ifg):
        reject_invalid_state("snaphu unwrap requires a 2-D complex interferogram")
    if coherence.shape != complex_ifg.shape:
        reject_invalid_state("coherence must match the interferogram shape")

    cfg = _resolve_snaphu_config(
        config or SnaphuConfig(),
        shape=(int(complex_ifg.shape[0]), int(complex_ifg.shape[1])),
    )
    nlooks = cfg.nlooks
    ntiles = cfg.ntiles
    nproc = cfg.nproc
    tile_overlap = cfg.tile_overlap
    if nlooks is None or ntiles is None or nproc is None or tile_overlap is None:
        reject("internal SNAPHU config resolution left a field unset", RuntimeError)
    snaphu = require_snaphu()
    capability = snaphu_capability()

    kwargs: dict[str, Any] = {
        "nlooks": float(nlooks),
        "cost": cfg.cost,
        "nproc": int(nproc),
        "ntiles": (int(ntiles[0]), int(ntiles[1])),
        "tile_overlap": (int(tile_overlap[0]), int(tile_overlap[1])),
        "single_tile_reoptimize": bool(cfg.single_tile_reoptimize),
        "delete_scratch": False,
    }
    valid = (
        np.isfinite(complex_ifg.real)
        & np.isfinite(complex_ifg.imag)
        & np.isfinite(coherence)
        & (coherence > 0.0)
    )
    kwargs["mask"] = valid.astype(np.uint8)

    logger.info(
        "Calling snaphu-py unwrap (wrapper=%s, bundled=%s, cost=%s)",
        capability.wrapper_version,
        capability.bundled_snaphu_version,
        cfg.cost,
    )
    resolved_work_dir = Path.cwd() if work_dir is None else Path(work_dir)
    with _snaphu_scratch_directory(resolved_work_dir, pair_id) as scratchdir:
        unwrapped, conncomp = snaphu.unwrap(
            np.asarray(complex_ifg),
            np.asarray(coherence, dtype=np.float32),
            scratchdir=scratchdir,
            **kwargs,
        )
        return (
            np.asarray(unwrapped, dtype=np.float32),
            np.asarray(conncomp, dtype=np.int32),
        )


def snaphu_unwrap(
    complex_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    config: SnaphuConfig | None = None,
    work_dir: str | Path | None = None,
    _pair_id: str | None = None,
) -> SpatialUnwrapResult:
    """Unwrap a complex interferogram and return the spatial result contract.

    Parameters
    ----------
    complex_ifg : numpy.ndarray
        Two-dimensional complex interferogram.
    coherence : numpy.ndarray
        Coherence in ``[0, 1]`` matching ``complex_ifg``.
    config : SnaphuConfig, optional
        Explicit SNAPHU configuration.
    work_dir : str or pathlib.Path, optional
        Workspace for invocation scratch. ``None`` uses the current working
        directory.

    Returns
    -------
    SpatialUnwrapResult
        Torch result on CPU. Use :class:`Snaphu` to preserve a caller's
        existing Torch device.

    """
    import torch

    if not isinstance(complex_ifg, np.ndarray):
        complex_ifg = np.asarray(complex_ifg)
    if not isinstance(coherence, np.ndarray):
        coherence = np.asarray(coherence)
    phase = np.angle(complex_ifg).astype(np.float32)
    strategy = Snaphu(config, work_dir=work_dir)
    strategy._pair_id = _pair_id
    return strategy.unwrap(
        torch.as_tensor(phase),
        coherence=torch.as_tensor(coherence, dtype=torch.float32),
    )
