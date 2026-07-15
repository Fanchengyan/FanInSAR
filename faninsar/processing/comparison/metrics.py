"""Reference comparison metrics for FanInSAR pair products vs external processors.

This module provides machine-readable metrics for comparing FanInSAR
production pair outputs against reference products from external
processors such as pygmtsar, ISCE2, or ISCE3.  No external processor
runtime dependency is required; only NumPy (and optionally Zarr) are
needed.

Expected reference layers
-----------------------
When comparing against pygmtsar/ISCE-style products, the caller should
supply arrays or Zarr groups containing the following layers:

- ``wrapped_phase`` — wrapped interferometric phase in radians
- ``coherence`` — coherence magnitude in [0, 1]
- ``unwrapped_phase`` — unwrapped phase in radians (optional)
- ``complex_ifg`` — complex interferogram (optional)
- ``lon`` / ``lat`` — geolocation grids in degrees (optional)

pygmtsar naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~
pygmtsar typically writes:

- ``phasefilt.grd`` → wrapped phase
- ``corr.grd`` → coherence
- ``unwrap.grd`` → unwrapped phase
- ``intf.grd`` → complex interferogram

ISCE2 naming conventions
~~~~~~~~~~~~~~~~~~~~~~~
ISCE2 ``topsApp.py`` outputs:

- ``merged/filt_topophase.unw.geo`` (unwrapped)
- ``merged/filt_topophase.flat.geo`` (wrapped)
- ``merged/phsig.cor.geo`` → coherence
- ``merged/lon.rdr`` / ``merged/lat.rdr`` → geolocation grids

ISCE3 naming conventions
~~~~~~~~~~~~~~~~~~~~~~~
ISCE3 ``insar`` app outputs:

- ``unwrapped.geo`` / ``wrapped.geo``
- ``coherence.geo``
- ``lon.geo`` / ``lat.geo``

All functions accept in-memory :class:`numpy.ndarray` objects or paths
to Zarr stores.  The high-level :func:`compare_pair_products` entry
point writes a JSON report suitable for CI gates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = setup_logger(__name__)

FloatArray: type[NDArray[np.float64]] = NDArray[np.float64]


class PairComparisonError(ValueError):
    """Raised when comparison inputs are invalid or incompatible."""

    def __init__(self, detail: str) -> None:
        """Initialize with actionable validation detail."""
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        """Return the boundary-validation detail."""
        return self.detail


def _fail(detail: str) -> None:
    logger.exception(detail)
    raise PairComparisonError(detail)


def _to_array(value: np.ndarray | str | Path, layer_name: str) -> np.ndarray:
    """Resolve an array-like input to a NumPy array."""
    if isinstance(value, (str, Path)):
        try:
            import zarr
        except ImportError as exc:
            message = (
                f"zarr is required to load layer {layer_name!r} from path: {value}"
            )
            logger.exception(message)
            raise PairComparisonError(message) from exc
        root = zarr.open_group(str(value), mode="r")
        return np.asarray(root[layer_name])
    return np.asarray(value)


def _paired_finite(
    reference: FloatArray,
    candidate: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Validate that two arrays share shape and contain only finite values."""
    if reference.shape != candidate.shape:
        _fail(
            f"reference and candidate shape mismatch: "
            f"{reference.shape} != {candidate.shape}"
        )
    if reference.size == 0:
        _fail("metric arrays must not be empty")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        _fail("metric arrays must contain only finite values")
    return reference, candidate


def _rmse(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


@dataclass(frozen=True, slots=True)
class PhaseComparisonResult:
    """Circular and linear phase residual summary."""

    valid_count: int
    circular_rmse_rad: float
    linear_rmse_rad: float
    mean_bias_rad: float
    max_absolute_error_rad: float
    rewrap_count: int

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoherenceComparisonResult:
    """Coherence absolute error summary."""

    valid_count: int
    mean_absolute_error: float
    max_absolute_error: float
    median_absolute_error: float
    rmse: float

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewrapResidualResult:
    """Statistics on the residual after re-wrapping unwrapped phase."""

    valid_count: int
    residual_mean_rad: float
    residual_std_rad: float
    residual_rmse_rad: float
    residual_max_rad: float
    fraction_within_half_cycle: float

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeolocationResidualResult:
    """Horizontal and vertical geolocation residual summary."""

    valid_count: int
    horizontal_rmse_m: float
    horizontal_max_m: float
    vertical_rmse_m: float
    vertical_max_m: float

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairComparisonReport:
    """Machine-readable JSON report for a pair-product comparison."""

    pair_id: str
    reference_source: str
    candidate_source: str
    phase: PhaseComparisonResult | None
    coherence: CoherenceComparisonResult | None
    rewrap_residual: RewrapResidualResult | None
    geolocation: GeolocationResidualResult | None
    versions: dict[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "pair_id": self.pair_id,
            "reference_source": self.reference_source,
            "candidate_source": self.candidate_source,
            "phase": self.phase.to_json_dict() if self.phase is not None else None,
            "coherence": (
                self.coherence.to_json_dict() if self.coherence is not None else None
            ),
            "rewrap_residual": (
                self.rewrap_residual.to_json_dict()
                if self.rewrap_residual is not None
                else None
            ),
            "geolocation": (
                self.geolocation.to_json_dict()
                if self.geolocation is not None
                else None
            ),
            "versions": self.versions,
        }

    def write_json(self, path: str | Path) -> Path:
        """Persist the report to a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSON path.

        Returns
        -------
        pathlib.Path
            Path to the written file.

        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote pair comparison report: %s", out)
        return out


def phase_circular_rmse(
    reference: FloatArray,
    candidate: FloatArray,
) -> PhaseComparisonResult:
    """Compute circular phase RMSE and related statistics.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference wrapped phase in radians.
    candidate : numpy.ndarray
        Candidate wrapped phase in radians.

    Returns
    -------
    PhaseComparisonResult
        Circular RMSE, linear RMSE, bias, max error, and rewrap count.

    Raises
    ------
    PairComparisonError
        If arrays have mismatched shapes or non-finite values.

    Examples
    --------
    >>> ref = np.angle(np.exp(1j * np.ones((3, 3))))
    >>> cand = np.angle(np.exp(1j * (np.ones((3, 3)) + 0.1)))
    >>> result = phase_circular_rmse(ref, cand)
    >>> result.circular_rmse_rad
    0.1

    """
    reference, candidate = _paired_finite(reference, candidate)
    linear_residual = candidate - reference
    circular_residual = np.angle(np.exp(1j * linear_residual))
    rewrap_count = int(np.sum(np.abs(linear_residual) > np.pi))
    return PhaseComparisonResult(
        valid_count=reference.size,
        circular_rmse_rad=_rmse(circular_residual),
        linear_rmse_rad=_rmse(linear_residual),
        mean_bias_rad=float(np.mean(circular_residual)),
        max_absolute_error_rad=float(np.max(np.abs(circular_residual))),
        rewrap_count=rewrap_count,
    )


def coherence_absolute_error(
    reference: FloatArray,
    candidate: FloatArray,
) -> CoherenceComparisonResult:
    """Compute coherence absolute error statistics.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference coherence in [0, 1].
    candidate : numpy.ndarray
        Candidate coherence in [0, 1].

    Returns
    -------
    CoherenceComparisonResult
        MAE, max AE, median AE, and RMSE.

    Raises
    ------
    PairComparisonError
        If values lie outside [0, 1] or arrays are incompatible.

    Examples
    --------
    >>> ref = np.full((4, 4), 0.8)
    >>> cand = np.full((4, 4), 0.75)
    >>> result = coherence_absolute_error(ref, cand)
    >>> result.mean_absolute_error
    0.05

    """
    reference, candidate = _paired_finite(reference, candidate)
    if np.any((reference < 0) | (reference > 1) | (candidate < 0) | (candidate > 1)):
        _fail("coherence values must lie in [0, 1]")
    residual = candidate - reference
    abs_residual = np.abs(residual)
    return CoherenceComparisonResult(
        valid_count=residual.size,
        mean_absolute_error=float(np.mean(abs_residual)),
        max_absolute_error=float(np.max(abs_residual)),
        median_absolute_error=float(np.median(abs_residual)),
        rmse=_rmse(residual),
    )


def rewrap_residual_stats(
    reference_unwrapped: FloatArray,
    candidate_unwrapped: FloatArray,
) -> RewrapResidualResult:
    r"""Compute statistics on the residual after re-wrapping unwrapped phase.

    The residual is defined as the circular difference between the two
    unwrapped fields after wrapping them back to :math:`[-\pi, \pi)`.
    A small residual indicates that the unwrapped phases agree up to
    an integer number of cycles.

    Parameters
    ----------
    reference_unwrapped : numpy.ndarray
        Reference unwrapped phase in radians.
    candidate_unwrapped : numpy.ndarray
        Candidate unwrapped phase in radians.

    Returns
    -------
    RewrapResidualResult
        Mean, std, RMSE, max of the rewrap residual, and the fraction
        of pixels within a half cycle (:math:`|residual| < \pi/2`).

    Raises
    ------
    PairComparisonError
        If arrays have mismatched shapes or non-finite values.

    Examples
    --------
    >>> ref = np.zeros((3, 3))
    >>> cand = np.full((3, 3), 2 * np.pi)
    >>> result = rewrap_residual_stats(ref, cand)
    >>> result.residual_mean_rad
    0.0

    """
    reference, candidate = _paired_finite(reference_unwrapped, candidate_unwrapped)
    linear_residual = candidate - reference
    circular_residual = np.angle(np.exp(1j * linear_residual))
    half_cycle = np.pi / 2.0
    within_half = np.abs(circular_residual) < half_cycle
    return RewrapResidualResult(
        valid_count=reference.size,
        residual_mean_rad=float(np.mean(circular_residual)),
        residual_std_rad=float(np.std(circular_residual)),
        residual_rmse_rad=_rmse(circular_residual),
        residual_max_rad=float(np.max(np.abs(circular_residual))),
        fraction_within_half_cycle=float(np.mean(within_half)),
    )


def geolocation_residual(
    reference_lon: FloatArray,
    reference_lat: FloatArray,
    candidate_lon: FloatArray,
    candidate_lat: FloatArray,
) -> GeolocationResidualResult:
    """Compute horizontal geolocation residuals from lon/lat grids.

    Parameters
    ----------
    reference_lon : numpy.ndarray
        Reference longitude grid in degrees.
    reference_lat : numpy.ndarray
        Reference latitude grid in degrees.
    candidate_lon : numpy.ndarray
        Candidate longitude grid in degrees.
    candidate_lat : numpy.ndarray
        Candidate latitude grid in degrees.

    Returns
    -------
    GeolocationResidualResult
        Horizontal RMSE and max error in metres, computed with a
        simple spherical Earth approximation (111 195 m per degree).

    Raises
    ------
    PairComparisonError
        If arrays have mismatched shapes or non-finite values.

    """
    ref_lon, cand_lon = _paired_finite(reference_lon, candidate_lon)
    ref_lat, cand_lat = _paired_finite(reference_lat, candidate_lat)
    if ref_lon.shape != ref_lat.shape:
        _fail(
            f"lon/lat shape mismatch in reference: {ref_lon.shape} != {ref_lat.shape}"
        )
    if cand_lon.shape != cand_lat.shape:
        _fail(
            f"lon/lat shape mismatch in candidate: {cand_lon.shape} != {cand_lat.shape}"
        )
    # Approximate metres per degree at the mean latitude
    mean_lat = float(np.mean(ref_lat))
    metres_per_degree_lat = 111_132.0 - 559.82 * np.cos(2.0 * np.deg2rad(mean_lat))
    metres_per_degree_lon = 111_132.0 * np.cos(np.deg2rad(mean_lat))
    dlon_m = (cand_lon - ref_lon) * metres_per_degree_lon
    dlat_m = (cand_lat - ref_lat) * metres_per_degree_lat
    horizontal = np.sqrt(np.square(dlon_m) + np.square(dlat_m))
    return GeolocationResidualResult(
        valid_count=ref_lon.size,
        horizontal_rmse_m=_rmse(horizontal),
        horizontal_max_m=float(np.max(horizontal)),
        vertical_rmse_m=0.0,
        vertical_max_m=0.0,
    )


def compare_pair_products(
    faninsar_zarr: str | Path | Mapping[str, np.ndarray],
    reference_zarr_or_arrays: str | Path | Mapping[str, np.ndarray],
    out_json: str | Path,
    *,
    pair_id: str = "unknown-pair",
    reference_source: str = "external",
    candidate_source: str = "faninsar",
    layers: Sequence[str] = ("wrapped_phase", "coherence", "unwrapped_phase"),
    lon_lat_layers: tuple[str, str] | None = ("lon", "lat"),
) -> PairComparisonReport:
    """Compare FanInSAR pair products against a reference and write a JSON report.

    Parameters
    ----------
    faninsar_zarr : str, pathlib.Path, or collections.abc.Mapping
        FanInSAR product Zarr path or mapping of layer names to arrays.
    reference_zarr_or_arrays : str, pathlib.Path, or collections.abc.Mapping
        Reference product Zarr path or mapping of layer names to arrays.
    out_json : str or pathlib.Path
        Output JSON report path.
    pair_id : str, optional
        Identifier for the pair being compared.
    reference_source : str, optional
        Human-readable reference processor name.
    candidate_source : str, optional
        Human-readable candidate processor name.
    layers : sequence of str, optional
        Layer names to compare.  Defaults to wrapped phase, coherence,
        and unwrapped phase.
    lon_lat_layers : tuple of str or None, optional
        Names of the longitude and latitude layers for geolocation
        residual computation.  Set to ``None`` to skip.

    Returns
    -------
    PairComparisonReport
        The populated report object.

    Raises
    ------
    PairComparisonError
        If required layers are missing or arrays are incompatible.

    Examples
    --------
    >>> report = compare_pair_products(
    ...     "faninsar_pair.zarr",
    ...     "isce_pair.zarr",
    ...     "report.json",
    ...     pair_id="S1A_20240101_20240113",
    ...     reference_source="ISCE2 topsApp",
    ... )
    >>> report.phase is not None
    True

    """
    # Resolve inputs to mappings
    def _resolve(
        src: str | Path | Mapping[str, np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        if isinstance(src, (str, Path)):
            try:
                import zarr
            except ImportError as exc:
                message = f"zarr is required to open Zarr store: {src}"
                logger.exception(message)
                raise PairComparisonError(message) from exc
            root = zarr.open_group(str(src), mode="r")
            return {name: np.asarray(root[name]) for name in root.array_keys()}
        return src

    faninsar = _resolve(faninsar_zarr)
    reference = _resolve(reference_zarr_or_arrays)

    phase_result: PhaseComparisonResult | None = None
    coherence_result: CoherenceComparisonResult | None = None
    rewrap_result: RewrapResidualResult | None = None
    geo_result: GeolocationResidualResult | None = None

    if "wrapped_phase" in layers:
        if "wrapped_phase" not in faninsar or "wrapped_phase" not in reference:
            _fail("wrapped_phase layer missing from one or both inputs")
        phase_result = phase_circular_rmse(
            np.asarray(reference["wrapped_phase"]),
            np.asarray(faninsar["wrapped_phase"]),
        )

    if "coherence" in layers:
        if "coherence" not in faninsar or "coherence" not in reference:
            _fail("coherence layer missing from one or both inputs")
        coherence_result = coherence_absolute_error(
            np.asarray(reference["coherence"]),
            np.asarray(faninsar["coherence"]),
        )

    if "unwrapped_phase" in layers:
        if "unwrapped_phase" not in faninsar or "unwrapped_phase" not in reference:
            logger.warning(
                "unwrapped_phase layer missing from one or both inputs; "
                "skipping rewrap residual"
            )
        else:
            rewrap_result = rewrap_residual_stats(
                np.asarray(reference["unwrapped_phase"]),
                np.asarray(faninsar["unwrapped_phase"]),
            )

    if lon_lat_layers is not None:
        lon_name, lat_name = lon_lat_layers
        has_lon = lon_name in faninsar and lon_name in reference
        has_lat = lat_name in faninsar and lat_name in reference
        if has_lon and has_lat:
            geo_result = geolocation_residual(
                np.asarray(reference[lon_name]),
                np.asarray(reference[lat_name]),
                np.asarray(faninsar[lon_name]),
                np.asarray(faninsar[lat_name]),
            )
        else:
            logger.warning(
                "lon/lat layers missing from one or both inputs; "
                "skipping geolocation residual"
            )

    report = PairComparisonReport(
        pair_id=pair_id,
        reference_source=reference_source,
        candidate_source=candidate_source,
        phase=phase_result,
        coherence=coherence_result,
        rewrap_residual=rewrap_result,
        geolocation=geo_result,
        versions={"numpy": np.__version__},
    )
    report.write_json(out_json)
    return report
