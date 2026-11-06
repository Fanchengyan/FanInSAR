"""Sentinel-1 AUX_CAL parsing and the IPF 002.36 EAP phase correction.

Products processed by Sentinel-1 IPF 002.36 (S1A, before March 2015) were
focused without compensating the phase of the two-way elevation antenna
pattern (EAP).  Interferograms that mix such a product with a later IPF
show a range-dependent phase error and a sub-swath phase jump.  ESA's fix,
implemented by ISCE2 (``isceobj.Sensor.TOPS.Sentinel1``), divides every
burst by ``exp(j * arg G_EAP(theta))``, where ``G_EAP`` is the complex
pattern published in the ``AUX_CAL`` auxiliary product and ``theta`` is the
elevation angle of each range sample.

The correction is applied when burst samples are loaded
(:func:`correct_eap_phase`); it is never a separate processing stage.
This module ports ISCE2's formulation verbatim, including the nominal roll
and height laws (``s1_anx2roll`` / ``s1_anx2Height``).
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import reject, setup_logger
from faninsar.missions.s1.errors import reject_product
from faninsar.missions.s1.types import S1SourcePhase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar.missions.s1.io.read import BurstArray
    from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath

logger = setup_logger(__name__)

EAP_CORRECTION_IPF_VERSIONS = frozenset({"002.36"})
"""IPF versions whose SLC phase lacks the elevation antenna pattern."""

EAP_CORRECTION_VERSION = "isce2-eap-division-v1"
"""Version of ``samples / exp(1j * angle(G_EAP))`` at source load."""

_AUX_CAL_NAME = re.compile(
    r"(?P<mission>S1[ABCD])_AUX_CAL_V(?P<validity>\d{8}T\d{6})_"
    r"G(?P<generation>\d{8}T\d{6})",
    re.IGNORECASE,
)
_MISSIONS = frozenset({"S1A", "S1B", "S1C", "S1D"})

# Nominal Sentinel-1 roll and altitude laws (ISCE2 ``s1_anx2roll`` and
# ``s1_anx2Height``, "straight from S1A documentation").
_ROLL_REFERENCE_HEIGHT_KM = 711.700
_ROLL_REFERENCE_BORESIGHT_DEG = 29.450
_ROLL_PER_KM_DEG = 0.0566
_HEIGHT_MEAN_M = 707714.8
_HEIGHT_AMPLITUDES_M = (8351.5, 8947.0, 23.32, 11.74)
_HEIGHT_PHASES_RAD = (3.1495, -1.5655, -3.1297, 4.7222)
_ORBIT_PERIOD_S = (12 * 24 * 60 * 60) / 175.0


@dataclass(frozen=True, slots=True)
class AuxCalName:
    """Parsed ``S1x_AUX_CAL_V<validity>_G<generation>`` identity."""

    mission_id: str
    validity_start: datetime
    generation: datetime
    product_name: str


@dataclass(frozen=True, slots=True)
class ElevationAntennaPattern:
    """Complex two-way elevation antenna pattern of one swath/polarization.

    ``gain[i]`` is sampled at antenna-frame elevation angle
    ``(i - (N - 1) / 2) * elevation_angle_increment_deg``.
    """

    swath: str
    polarization: str
    elevation_angle_increment_deg: float
    gain: np.ndarray


def _compact_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def parse_aux_cal_name(value: str | Path) -> AuxCalName | None:
    """Parse the AUX_CAL product identity embedded in a file or path name.

    Parameters
    ----------
    value : str or pathlib.Path
        Product name, ``.SAFE``/``.SAFE.zip`` name, or any path containing
        one (for example ``.../S1A_AUX_CAL_V..._G....SAFE/data/s1a-aux-cal.xml``).

    Returns
    -------
    AuxCalName or None
        Parsed identity, or ``None`` when no AUX_CAL name is present.

    Examples
    --------
    >>> name = parse_aux_cal_name("S1A_AUX_CAL_V20140908T000000_G20240327T101157")
    >>> name.mission_id, name.validity_start.year, name.generation.year
    ('S1A', 2014, 2024)

    """
    match = None
    for part in reversed(Path(str(value)).parts):
        match = _AUX_CAL_NAME.search(part)
        if match is not None:
            break
    if match is None:
        return None
    try:
        validity = _compact_utc(match["validity"])
        generation = _compact_utc(match["generation"])
    except ValueError:
        return None
    mission = match["mission"].upper()
    return AuxCalName(
        mission_id=mission,
        validity_start=validity,
        generation=generation,
        product_name=(f"{mission}_AUX_CAL_V{match['validity']}_G{match['generation']}"),
    )


def normalize_aux_cal_path(path: str | Path) -> Path:
    """Return the AUX_CAL file to bind for one user-supplied location.

    ``.SAFE`` directories resolve to their ``data/s1?-aux-cal.xml``; zip
    archives and XML files are returned unchanged.

    Raises
    ------
    FileNotFoundError
        If the location does not exist or a ``.SAFE`` has no data XML.

    """
    source = Path(path)
    if source.is_dir():
        candidates = sorted((source / "data").glob("s1?-aux-cal.xml"))
        if len(candidates) != 1:
            message = f"AUX_CAL SAFE has no single data/s1?-aux-cal.xml: {source}"
            reject(message, FileNotFoundError, logger=logger)
        return candidates[0]
    if not source.is_file():
        message = f"configured S1 AUX_CAL file is missing: {source}"
        reject(message, FileNotFoundError, logger=logger)
    return source


def _aux_cal_xml_bytes(path: Path) -> bytes:
    """Return the ``s1?-aux-cal.xml`` document of a zip, SAFE, or XML path."""
    source = normalize_aux_cal_path(path)
    if source.suffix.casefold() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [
                name
                for name in archive.namelist()
                if re.search(r"/data/s1[abcd]-aux-cal\.xml$", name, re.IGNORECASE)
            ]
            if len(names) != 1:
                reject_product(f"AUX_CAL zip has no single data XML: {source}")
            return archive.read(names[0])
    return source.read_bytes()


def read_elevation_antenna_pattern(
    path: str | Path,
    *,
    swath: str,
    polarization: str,
) -> ElevationAntennaPattern:
    """Read one swath/polarization elevation antenna pattern from AUX_CAL.

    Parameters
    ----------
    path : str or pathlib.Path
        AUX_CAL ``.SAFE.zip``, ``.SAFE`` directory, or ``s1?-aux-cal.xml``.
    swath : str
        Sub-swath name such as ``"IW1"``.
    polarization : str
        Channel such as ``"VV"``.

    Returns
    -------
    ElevationAntennaPattern
        Complex pattern de-interleaved from the ``I Q I Q ...`` values.

    Raises
    ------
    Sentinel1ProductError
        If the file has no matching ``calibrationParams`` record or the
        record is malformed.

    """
    root = ET.fromstring(_aux_cal_xml_bytes(Path(path)))
    params = root.find("calibrationParamsList")
    wanted_swath = str(swath).strip().upper()
    wanted_pol = str(polarization).strip().upper()
    for record in params if params is not None else ():
        record_swath = (record.findtext("swath") or "").strip().upper()
        record_pol = (record.findtext("polarisation") or "").strip().upper()
        if record_swath != wanted_swath or record_pol != wanted_pol:
            continue
        increment = record.findtext("elevationAntennaPattern/elevationAngleIncrement")
        values = record.findtext("elevationAntennaPattern/values")
        if increment is None or values is None:
            break
        iq = np.asarray(values.split(), dtype=np.float64)
        if iq.size < 4 or iq.size % 2:
            break
        return ElevationAntennaPattern(
            swath=wanted_swath,
            polarization=wanted_pol,
            elevation_angle_increment_deg=float(increment),
            gain=iq[0::2] + 1j * iq[1::2],
        )
    return reject_product(
        f"AUX_CAL {path} has no elevation antenna pattern for "
        f"{wanted_swath}/{wanted_pol}"
    )


def nominal_height_m(delta_anx_s: float) -> float:
    """Return the nominal Sentinel-1 altitude ``delta_anx_s`` after ANX.

    Port of ISCE2 ``s1_anx2Height``: a four-term harmonic series in the
    orbital angular rate.
    """
    omega = 2.0 * np.pi / _ORBIT_PERIOD_S
    height = _HEIGHT_MEAN_M
    for order, (amplitude, phase) in enumerate(
        zip(_HEIGHT_AMPLITUDES_M, _HEIGHT_PHASES_RAD, strict=True), start=1
    ):
        height += amplitude * np.sin(order * omega * delta_anx_s + phase)
    return float(height)


def nominal_roll_deg(delta_anx_s: float) -> float:
    """Return the nominal platform roll (off-nadir boresight) in degrees.

    Port of ISCE2 ``s1_anx2roll``:
    ``29.450 - 0.0566 * (h_km - 711.700)``.
    """
    height_km = nominal_height_m(delta_anx_s) / 1000.0
    return _ROLL_REFERENCE_BORESIGHT_DEG - _ROLL_PER_KM_DEG * (
        height_km - _ROLL_REFERENCE_HEIGHT_KM
    )


def requires_eap_correction(product: S1Product) -> bool:
    """Return whether ``product`` needs the elevation antenna phase correction."""
    version = str(getattr(product, "ipf_version", None) or "").strip()
    return version in EAP_CORRECTION_IPF_VERSIONS


def source_phase_binding(
    product: S1Product,
    *,
    aux_cal_path: str | Path | None,
) -> S1SourcePhase:
    """Describe the qualified source-load result without reading SLC samples.

    Parameters
    ----------
    product : S1Product
        Materialized raw-phase source and its IPF applicability metadata.
    aux_cal_path : path or None
        Bound calibration product, required for affected IPF versions.

    Returns
    -------
    S1SourcePhase
        Correction identity and resulting source phase. The digest covers the
        calibration XML, independent of ZIP packaging or the local cache path.

    Raises
    ------
    Sentinel1ProductError
        If an applicable source has no calibration binding.

    """
    applicable = requires_eap_correction(product)
    identity = None
    digest = None
    if applicable:
        if aux_cal_path is None:
            reject_product(
                f"{product.product_id} was processed by IPF {product.ipf_version} and "
                "needs the AUX_CAL elevation antenna phase correction, but no "
                "AUX_CAL file is bound; run Stack.prepare_data() "
                "to download it or pass "
                "aux_cal_paths="
            )
        path = normalize_aux_cal_path(aux_cal_path)
        name = parse_aux_cal_name(path)
        identity = name.product_name if name is not None else path.name
        digest = hashlib.sha256(_aux_cal_xml_bytes(path)).hexdigest()
    return S1SourcePhase(
        product_id=product.product_id,
        ipf_version=product.ipf_version,
        eap_applicable=applicable,
        aux_cal_id=identity,
        aux_cal_digest=digest,
        correction_version=EAP_CORRECTION_VERSION,
        phase_state="eap_corrected" if applicable else "raw",
    )


def eap_phase_vector(
    swath: S1Swath,
    burst: S1Burst,
    pattern: ElevationAntennaPattern,
) -> np.ndarray:
    """Return the unit-modulus EAP phasor for every range sample of a burst.

    Mirrors ISCE2 ``computeElevationAntennaPatternCorrection``: the
    annotation elevation record nearest the burst mid time maps slant-range
    time to elevation angle, the AUX_CAL pattern is placed on the geometric
    frame by the nominal roll at the burst start, and the complex gain is
    linearly interpolated at each sample's elevation angle.

    Parameters
    ----------
    swath : S1Swath
        Parsed sub-swath (needs ``ascending_node_time`` and
        ``elevation_angle_records``).
    burst : S1Burst
        Burst whose samples will be corrected.
    pattern : ElevationAntennaPattern
        AUX_CAL pattern of the burst swath/polarization.

    Returns
    -------
    numpy.ndarray
        ``complex128`` vector of length ``swath.samples_per_burst``,
        ``exp(j * arg G_EAP)`` at absolute range samples ``0..N-1``.

    Raises
    ------
    Sentinel1ProductError
        If the annotation lacks the ascending-node time or elevation
        records.

    """
    if swath.ascending_node_time is None:
        reject_product(
            f"{swath.annotation_path} has no ascendingNodeTime; "
            "cannot apply the EAP phase correction"
        )
    if not swath.elevation_angle_records:
        reject_product(
            f"{swath.annotation_path} has no antennaPattern records; "
            "cannot apply the EAP phase correction"
        )
    sensing_mid = burst.azimuth_time + timedelta(
        seconds=(burst.lines - 1) * swath.azimuth_time_interval_s / 2.0
    )
    record = min(
        swath.elevation_angle_records,
        key=lambda item: abs((item.azimuth_time - sensing_mid).total_seconds()),
    )
    tau_sub = np.asarray(record.slant_range_time_s, dtype=np.float64)
    theta_sub = np.asarray(record.elevation_angle_deg, dtype=np.float64)

    n_elements = pattern.gain.shape[0]
    theta_antenna = (
        np.arange(n_elements, dtype=np.float64) - (n_elements - 1) / 2.0
    ) * pattern.elevation_angle_increment_deg
    delta_anx = (burst.azimuth_time - swath.ascending_node_time).total_seconds()
    theta_eap = theta_antenna + nominal_roll_deg(delta_anx)

    n_samples = int(swath.samples_per_burst)
    tau = swath.slant_range_time_s + np.arange(n_samples) / swath.range_sampling_rate_hz
    theta = np.interp(tau, tau_sub, theta_sub)
    if theta.min() < theta_eap[0] or theta.max() > theta_eap[-1]:
        reject_product(
            f"{swath.swath} elevation angles [{theta.min():.3f}, "
            f"{theta.max():.3f}] deg fall outside the AUX_CAL pattern "
            f"[{theta_eap[0]:.3f}, {theta_eap[-1]:.3f}] deg"
        )
    gain = np.interp(theta, theta_eap, pattern.gain.real) + 1j * np.interp(
        theta, theta_eap, pattern.gain.imag
    )
    return np.exp(1j * np.angle(gain))


def _apply_eap_phase_correction(
    array: BurstArray,
    swath: S1Swath,
    pattern: ElevationAntennaPattern,
) -> BurstArray:
    """Divide burst samples by the EAP phasor of their absolute range columns."""
    burst = swath.bursts[array.burst_index]
    phasor = eap_phase_vector(swath, burst, pattern)
    width = array.samples.shape[1]
    columns = phasor[array.col0 : array.col0 + width]
    if columns.shape[0] != width:
        reject_product(
            f"burst {burst.index} range window [{array.col0}, "
            f"{array.col0 + width}) exceeds {swath.samples_per_burst} samples"
        )
    corrected = (array.samples / columns[None, :]).astype(np.complex64)
    return replace(
        array,
        samples=np.where(array.valid_mask, corrected, 0),
        phase_state="eap_corrected",
    )


def correct_eap_phase(
    product: S1Product,
    swath: S1Swath,
    arrays: Sequence[BurstArray],
    *,
    aux_cal_path: str | Path | None,
) -> list[BurstArray]:
    """Apply the IPF 002.36 EAP phase correction to freshly read bursts.

    Products from any other IPF version are returned unchanged.  This is
    the single load-time seam used by every Sentinel-1 SLC loader.

    Parameters
    ----------
    product : S1Product
        Product the bursts were read from (supplies ``ipf_version``).
    swath : S1Swath
        Sub-swath the bursts belong to.
    arrays : sequence of BurstArray
        Raw burst arrays from :func:`~faninsar.missions.s1.io.read.read_full_burst`.
    aux_cal_path : path, optional
        AUX_CAL file bound to the product's acquisition date.

    Returns
    -------
    list of BurstArray
        Corrected (or untouched) arrays in input order.

    Raises
    ------
    Sentinel1ProductError
        If the product needs the correction and no AUX_CAL is bound.

    """
    if not requires_eap_correction(product):
        return list(arrays)
    if aux_cal_path is None:
        reject_product(
            f"{product.product_id} was processed by IPF {product.ipf_version} and "
            "needs the AUX_CAL elevation antenna phase correction, but no "
            "AUX_CAL file is bound; run Stack.prepare_data() to download it or pass "
            "aux_cal_paths="
        )
    pattern = read_elevation_antenna_pattern(
        aux_cal_path, swath=swath.swath, polarization=swath.polarization
    )
    logger.info(
        "Applying IPF %s EAP phase correction to %s %s with %s",
        product.ipf_version,
        product.product_id,
        swath.swath,
        Path(aux_cal_path).name,
    )
    return [_apply_eap_phase_correction(array, swath, pattern) for array in arrays]


def select_aux_cal_bindings(
    candidates: Mapping[str, object] | Sequence[str | Path] | None,
    acquisitions: Mapping[str, tuple[str, datetime]],
    *,
    require_coverage: bool = False,
) -> dict[str, Path]:
    """Bind one AUX_CAL file to each acquisition date.

    Among the mission's candidates whose validity starts no later than the
    sensing time, the latest validity start wins and the latest generation
    breaks ties (the newest active calibration for that date).

    Parameters
    ----------
    candidates : mapping or sequence, optional
        Date-keyed pins or a shared candidate pool of AUX_CAL zips, ``.SAFE``
        directories, or ``s1?-aux-cal.xml`` files.
    acquisitions : mapping
        ``date_id -> (mission_id, sensing_time)`` for the dates to bind.
    require_coverage : bool, optional
        Raise when a date has no qualifying candidate.

    Returns
    -------
    dict
        Date-to-file bindings for the covered dates.

    Raises
    ------
    FileNotFoundError
        If a candidate does not exist.
    ValueError
        If ``require_coverage`` is true and a date is uncovered.

    """
    if candidates is None:
        return {}
    if isinstance(candidates, (str, Path)):
        candidates = [candidates]
    if isinstance(candidates, Mapping):
        by_date = {str(key): _as_paths(value) for key, value in candidates.items()}
    else:
        shared = _as_paths(candidates)
        by_date = dict.fromkeys(acquisitions, shared)
    selected: dict[str, Path] = {}
    for date_id, (mission, sensing) in acquisitions.items():
        ranked: list[tuple[datetime, datetime, str, Path]] = []
        for candidate in by_date.get(date_id, ()):
            path = normalize_aux_cal_path(candidate)
            parsed = parse_aux_cal_name(path)
            if parsed is None:
                logger.warning("ignoring S1 AUX_CAL candidate without a name: %s", path)
                continue
            if mission in _MISSIONS and parsed.mission_id != mission:
                continue
            if parsed.validity_start > sensing:
                continue
            ranked.append(
                (parsed.validity_start, parsed.generation, path.as_posix(), path)
            )
        if not ranked:
            if require_coverage:
                message = f"no configured S1 AUX_CAL file covers acquisition {date_id}"
                reject(message, ValueError, logger=logger)
            continue
        selected[date_id] = max(ranked)[3]
    return selected


def _as_paths(value: object) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, (list, tuple)):
        return tuple(Path(item) for item in value)
    message = f"unsupported S1 AUX_CAL binding: {value!r}"
    return reject(message, TypeError, logger=logger)


__all__ = [
    "EAP_CORRECTION_IPF_VERSIONS",
    "AuxCalName",
    "ElevationAntennaPattern",
    "correct_eap_phase",
    "eap_phase_vector",
    "nominal_height_m",
    "nominal_roll_deg",
    "normalize_aux_cal_path",
    "parse_aux_cal_name",
    "read_elevation_antenna_pattern",
    "requires_eap_correction",
    "select_aux_cal_bindings",
]
