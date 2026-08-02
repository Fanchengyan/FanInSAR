"""EGM96 geoid undulation model (ISCE2-compatible).

Evaluates the EGM96 geoid undulation from the spherical-harmonic
coefficient set distributed with ISCE2 (``correct_geoid_i2_srtm/
egm96geoid.dat``).  The algorithm is a faithful port of ISCE2's
Fortran ``geoid_hgt`` routine so FanInSAR converts orthometric DEM
heights to ellipsoidal heights exactly like ISCE2's ``verifyDEM``
geoid-correction step.

To keep runtime acceptable, the undulation is evaluated once on a
0.1-degree grid covering the sampled region (the same sampling ISCE2
uses) and bilinearly interpolated for every requested coordinate.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = setup_logger(__name__)

# WGS84 (G873) constants used by ISCE2's geoid_hgt.
_SEMI_MAJOR_AXIS_M = 6_378_137.0
_ECCENTRICITY_SQUARED = 0.00669437999013
_EQUATORIAL_GRAVITY_M_S2 = 9.7803253359
_GRAVITY_CONSTANT = 0.00193185265246
_GM_M3_S2 = 0.3986004418e15
_J2 = 0.108262982131e-2
_J4 = -0.237091120053e-5
_J6 = 0.608346498882e-8
_J8 = -0.142681087920e-10
_J10 = 0.121439275882e-13
_MAX_DEGREE = 360
_COEFFICIENT_COUNT = (_MAX_DEGREE + 1) * (_MAX_DEGREE + 2) // 2
_GEOID_SAMPLE_DEG = 0.1
_GEOID_PAD_DEG = 1.5 * _GEOID_SAMPLE_DEG


@dataclass(frozen=True, slots=True)
class _EGM96Coefficients:
    """Fully-normalized EGM96 spherical-harmonic coefficients.

    ``cc``/``cs`` are the correction coefficients used for the height
    anomaly-to-undulation term, and ``hc``/``hs`` are the model
    coefficients used for the main potential sum (with the WGS84
    even-degree reference zonal terms restored).
    """

    cc: np.ndarray
    cs: np.ndarray
    hc: np.ndarray
    hs: np.ndarray


def default_egm96_path() -> Path | None:
    """Return the default EGM96 coefficient file path, or ``None``.

    Resolution order:

    1. ``FANINSAR_EGM96_FILE`` environment variable.
    2. ``faninsar/data/egm96/egm96geoid.dat`` bundled package data.
    3. The ISCE2 source tree copy (``contrib/demUtils/
       correct_geoid_i2_srtm/egm96geoid.dat``) for local development.
    """
    env_path = os.environ.get("FANINSAR_EGM96_FILE")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning("FANINSAR_EGM96_FILE points to missing file: %s", path)
    package_candidates = [
        Path(__file__).resolve().parents[3] / "data" / "egm96" / "egm96geoid.dat",
        Path(__file__).resolve().parents[3] / "data" / "egm96geoid.dat",
    ]
    for candidate in package_candidates:
        if candidate.exists():
            return candidate
    isce2_candidates = [
        Path.home()
        / "Documents"
        / "GitHub"
        / "isce2"
        / "contrib"
        / "demUtils"
        / "correct_geoid_i2_srtm"
        / "egm96geoid.dat",
    ]
    for candidate in isce2_candidates:
        if candidate.exists():
            return candidate
    return None


def load_egm96_coefficients(path: str | Path) -> _EGM96Coefficients:
    """Read the ISCE2-format EGM96 coefficient file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``egm96geoid.dat``.

    Returns
    -------
    _EGM96Coefficients
        Parsed coefficient arrays.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has an unexpected record count.

    """
    path = Path(path)
    if not path.exists():
        message = f"EGM96 coefficient file does not exist: {path}"
        logger.error(message)
        raise FileNotFoundError(message)
    raw = path.read_bytes()
    if len(raw) % 24 != 0:
        message = f"EGM96 coefficient file size not a multiple of 24: {path}"
        logger.error(message)
        raise ValueError(message)
    record_count = len(raw) // 24
    if record_count < 2 * _COEFFICIENT_COUNT:
        message = (
            f"EGM96 coefficient file {path} has {record_count} records, "
            f"expected at least {2 * _COEFFICIENT_COUNT}"
        )
        logger.error(message)
        raise ValueError(message)

    cc = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    cs = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    hc = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    hs = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    for index in range(record_count):
        offset = index * 24
        n = int.from_bytes(raw[offset : offset + 4], "little", signed=True)
        m = int.from_bytes(raw[offset + 4 : offset + 8], "little", signed=True)
        c = float(np.frombuffer(raw[offset + 8 : offset + 16], dtype="<f8")[0])
        s = float(np.frombuffer(raw[offset + 16 : offset + 24], dtype="<f8")[0])
        location = (n * (n + 1)) // 2 + m
        if index < _COEFFICIENT_COUNT:
            cc[location] = c
            cs[location] = s
        else:
            hc[location] = c
            hs[location] = s

    # Restore the WGS84 even-degree reference zonal terms (ISCE2 dhcsin).
    hc[3] += _J2 / math.sqrt(5.0)
    hc[10] += _J4 / 3.0
    hc[21] += _J6 / math.sqrt(13.0)
    hc[36] += _J8 / math.sqrt(17.0)
    hc[55] += _J10 / math.sqrt(21.0)
    logger.info("Loaded EGM96 coefficients from %s", path)
    return _EGM96Coefficients(cc=cc, cs=cs, hc=hc, hs=hs)


def _legendre_functions(m: int, theta: float, nmax: int) -> np.ndarray:
    """Return fully-normalized Legendre functions for one order.

    Port of ISCE2's ``legfdn`` (Colombo recursion).  ``theta`` is the
    colatitude in radians.  The returned array is indexed by degree
    ``n`` (``result[n]`` is ``Pbar_{n,m}``); degrees below ``m`` are
    zero.
    """
    nmax1 = nmax + 1
    nmax2p = 2 * nmax + 1
    m1 = m + 1
    m2 = m + 2
    m3 = m + 3
    drts = np.sqrt(np.arange(1, nmax2p + 1, dtype=np.float64))
    dirt = 1.0 / drts
    cothet = math.cos(theta)
    sithet = math.sin(theta)

    rlnn = np.zeros(nmax1 + 1, dtype=np.float64)
    rleg = np.zeros(nmax1 + 1, dtype=np.float64)
    rlnn[1] = 1.0
    if nmax1 >= 2:
        rlnn[2] = sithet * drts[2]
    for n1 in range(3, m1 + 1):
        n = n1 - 1
        n2 = 2 * n
        rlnn[n1] = drts[n2] * dirt[n2 - 1] * sithet * rlnn[n1 - 1]

    if m <= 1:
        if m == 0:
            rleg[1] = 1.0
            if nmax1 >= 2:
                rleg[2] = cothet * drts[2]
        else:
            rleg[2] = rlnn[2]
            if nmax1 >= 3:
                rleg[3] = drts[4] * cothet * rleg[2]
    if m1 <= nmax1:
        rleg[m1] = rlnn[m1]
        if m2 <= nmax1:
            rleg[m2] = drts[2 * m1] * cothet * rleg[m1]
            if m3 <= nmax1:
                for n1 in range(m3, nmax1 + 1):
                    n = n1 - 1
                    if not ((m == 0 and n < 2) or (m == 1 and n < 3)):
                        n2 = 2 * n
                        rleg[n1] = (
                            drts[n2]
                            * dirt[n + m - 1]
                            * dirt[n - m - 1]
                            * (
                                drts[n2 - 2] * cothet * rleg[n1 - 1]
                                - drts[n + m - 2]
                                * drts[n - m - 2]
                                * dirt[n2 - 4]
                                * rleg[n1 - 2]
                            )
                        )
    return rleg


def _geoid_undulation_point(
    latitude_deg: float,
    longitude_deg: float,
    coefficients: _EGM96Coefficients,
) -> float:
    """Evaluate EGM96 geoid undulation at one geodetic coordinate."""
    flat = float(latitude_deg)
    flon = float(longitude_deg)
    if flon < 0.0:
        flon += 360.0
    flatr = math.radians(flat)
    flonr = math.radians(flon)
    height_m = 0.0

    # radgra: geocentric radius, geocentric latitude, normal gravity.
    sin_flat = math.sin(flatr)
    t1 = sin_flat**2
    radius_n = _SEMI_MAJOR_AXIS_M / math.sqrt(1.0 - _ECCENTRICITY_SQUARED * t1)
    t2 = (radius_n + height_m) * math.cos(flatr)
    x = t2 * math.cos(flonr)
    y = t2 * math.sin(flonr)
    z = (radius_n * (1.0 - _ECCENTRICITY_SQUARED) + height_m) * sin_flat
    radius_geo = math.sqrt(x * x + y * y + z * z)
    latitude_geo = math.atan(z / math.sqrt(x * x + y * y))
    gravity = (
        _EQUATORIAL_GRAVITY_M_S2
        * (1.0 + _GRAVITY_CONSTANT * t1)
        / math.sqrt(1.0 - _ECCENTRICITY_SQUARED * t1)
    )

    # Fully-normalized Legendre functions for every order at this latitude.
    theta = math.pi / 2.0 - latitude_geo
    p = np.zeros(_COEFFICIENT_COUNT, dtype=np.float64)
    for j in range(1, _MAX_DEGREE + 2):
        order = j - 1
        rleg = _legendre_functions(order, theta, _MAX_DEGREE)
        for i in range(j, _MAX_DEGREE + 2):
            degree = i - 1
            location = (degree * (degree + 1)) // 2 + order
            p[location] = rleg[i]

    # dscml: sin/cos of m * longitude via recurrence.
    sin_ml = np.zeros(_MAX_DEGREE + 1, dtype=np.float64)
    cos_ml = np.zeros(_MAX_DEGREE + 1, dtype=np.float64)
    sin_lon = math.sin(flonr)
    cos_lon = math.cos(flonr)
    sin_ml[1] = sin_lon
    cos_ml[1] = cos_lon
    sin_ml[2] = 2.0 * cos_lon * sin_lon
    cos_ml[2] = 2.0 * cos_lon * cos_lon - 1.0
    for order in range(3, _MAX_DEGREE + 1):
        sin_ml[order] = 2.0 * cos_lon * sin_ml[order - 1] - sin_ml[order - 2]
        cos_ml[order] = 2.0 * cos_lon * cos_ml[order - 1] - cos_ml[order - 2]

    cc = coefficients.cc
    cs = coefficients.cs
    hc = coefficients.hc
    hs = coefficients.hs
    ar = _SEMI_MAJOR_AXIS_M / radius_geo
    arn = ar
    ac = 0.0
    undulation_sum = 0.0
    kk = 2
    for degree in range(2, _MAX_DEGREE + 1):
        arn = arn * ar
        kk += 1
        main_sum = p[kk] * hc[kk]
        corr_sum = p[kk] * cc[kk]
        for order in range(1, degree + 1):
            kk += 1
            corr_term = cc[kk] * cos_ml[order] + cs[kk] * sin_ml[order]
            main_term = hc[kk] * cos_ml[order] + hs[kk] * sin_ml[order]
            corr_sum += p[kk] * corr_term
            main_sum += p[kk] * main_term
        ac += corr_sum
        undulation_sum += main_sum * arn
    ac += cc[0] + p[1] * cc[1] + p[2] * (cc[2] * cos_ml[1] + cs[2] * sin_ml[1])
    height_anomaly_correction = ac / 100.0
    undulation = (
        undulation_sum * _GM_M3_S2 / (gravity * radius_geo)
        + height_anomaly_correction
        - 0.53
    )
    return float(undulation)


class EGM96Geoid:
    """Sample EGM96 geoid undulation at geodetic coordinates.

    The undulation is evaluated once on a 0.1-degree grid covering the
    requested region (with a 0.15-degree pad, matching ISCE2's geoid
    correction) and bilinearly interpolated for all subsequent samples.
    Heights are returned in metres; add the undulation to orthometric
    heights to obtain ellipsoidal heights.
    """

    def __init__(
        self,
        *,
        coefficient_path: str | Path | None = None,
        sample_degree: float = _GEOID_SAMPLE_DEG,
    ) -> None:
        """Initialise the geoid model.

        Parameters
        ----------
        coefficient_path : str or pathlib.Path, optional
            EGM96 coefficient file. Defaults to :func:`default_egm96_path`.
        sample_degree : float, optional
            Grid spacing in degrees for the cached undulation grid.
            Default 0.1 (matches ISCE2).

        Raises
        ------
        FileNotFoundError
            If no coefficient file can be located.

        """
        if coefficient_path is None:
            resolved = default_egm96_path()
            if resolved is None:
                message = (
                    "EGM96 coefficient file not found; set FANINSAR_EGM96_FILE "
                    "or install the bundled egm96geoid.dat"
                )
                logger.error(message)
                raise FileNotFoundError(message)
            coefficient_path = resolved
        self.coefficient_path = Path(coefficient_path)
        self.sample_degree = float(sample_degree)
        self._coefficients: _EGM96Coefficients | None = None
        self._grid_latitude: np.ndarray | None = None
        self._grid_longitude: np.ndarray | None = None
        self._grid_values: np.ndarray | None = None
        self._grid_bbox: tuple[float, float, float, float] | None = None
        self._interpolator: Callable[[np.ndarray], np.ndarray] | None = None

    def _load_coefficients(self) -> _EGM96Coefficients:
        if self._coefficients is None:
            self._coefficients = load_egm96_coefficients(self.coefficient_path)
        return self._coefficients

    def _evaluate_grid(
        self, latitude_deg: np.ndarray, longitude_deg: np.ndarray
    ) -> None:
        """Evaluate undulation on a 0.1-degree grid covering the samples."""
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        finite = np.isfinite(lat) & np.isfinite(lon)
        if not finite.any():
            return
        lat_min = float(lat[finite].min())
        lat_max = float(lat[finite].max())
        lon_min = float(lon[finite].min())
        lon_max = float(lon[finite].max())
        if self._grid_bbox is not None:
            g_lat_min, g_lat_max, g_lon_min, g_lon_max = self._grid_bbox
            if (
                lat_min >= g_lat_min
                and lat_max <= g_lat_max
                and lon_min >= g_lon_min
                and lon_max <= g_lon_max
            ):
                return

        pad = _GEOID_PAD_DEG
        lat0 = math.floor((lat_min - pad) / self.sample_degree) * self.sample_degree
        lat1 = math.ceil((lat_max + pad) / self.sample_degree) * self.sample_degree
        lon0 = math.floor((lon_min - pad) / self.sample_degree) * self.sample_degree
        lon1 = math.ceil((lon_max + pad) / self.sample_degree) * self.sample_degree
        latitudes = np.arange(lat0, lat1 + self.sample_degree / 2, self.sample_degree)
        longitudes = np.arange(lon0, lon1 + self.sample_degree / 2, self.sample_degree)
        values = np.full((latitudes.size, longitudes.size), np.nan, dtype=np.float64)
        coefficients = self._load_coefficients()
        for i, lat_i in enumerate(latitudes):
            for j, lon_j in enumerate(longitudes):
                values[i, j] = _geoid_undulation_point(
                    float(lat_i), float(lon_j), coefficients
                )
        self._grid_latitude = latitudes
        self._grid_longitude = longitudes
        self._grid_values = values
        self._grid_bbox = (float(lat0), float(lat1), float(lon0), float(lon1))
        self._interpolator = None
        logger.info(
            "EGM96 grid evaluated: %s x %s over (%.2f..%.2f, %.2f..%.2f)",
            latitudes.size,
            longitudes.size,
            lat0,
            lat1,
            lon0,
            lon1,
        )

    def sample(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
    ) -> np.ndarray:
        """Return geoid undulation (m) at each latitude/longitude sample.

        Parameters
        ----------
        latitude_deg, longitude_deg : numpy.ndarray
            Geodetic coordinates in degrees.

        Returns
        -------
        numpy.ndarray
            Undulation in metres, NaN for out-of-grid samples.

        """
        lat, lon = np.broadcast_arrays(
            np.asarray(latitude_deg, dtype=np.float64),
            np.asarray(longitude_deg, dtype=np.float64),
        )
        self._evaluate_grid(lat, lon)
        if self._interpolator is None:
            from scipy.interpolate import RegularGridInterpolator

            self._interpolator = RegularGridInterpolator(
                (self._grid_latitude, self._grid_longitude),
                self._grid_values,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        points = np.column_stack([lat.ravel(), lon.ravel()])
        values = self._interpolator(points)
        return np.asarray(values, dtype=np.float64).reshape(lat.shape)


__all__ = [
    "EGM96Geoid",
    "default_egm96_path",
    "load_egm96_coefficients",
]
