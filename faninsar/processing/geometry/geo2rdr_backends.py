"""Backend selection and accelerated implementations for ``geo2rdr``.

The transform ships with three interchangeable backends:

* ``numpy`` -- the original vectorised Newton solver (always available);
* ``torch`` -- a pure-PyTorch compiled solver with a scene-local orbit and an
  analytic Doppler derivative (default when Torch is importable);
* ``cpp`` -- an optional C++/OpenMP kernel loaded through :mod:`ctypes`
  (built by the package build script; falls back to Torch when the shared
  library or compiler is unavailable).

All backends expose the same numpy-in / numpy-out contract and are validated
against the native ISCE3 Geo2Rdr oracle in the FanInSAR-benchmarks repository.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.geometry.ellipsoid import llh_to_ecef

if TYPE_CHECKING:
    from faninsar.processing.geometry.transforms import RadarGeometryModel

logger = setup_logger(__name__)

BackendName = Literal["auto", "numpy", "torch", "cpp"]

_CPP_LIB_DIR = Path(__file__).resolve().parent / "cpp" / "lib"
_DEFAULT_HALF_SPAN_S = 600.0
_DEFAULT_TIME_TOL_S = 1e-8
_DEFAULT_MAX_ITER = 4


def _pack_orbit_uniform(
    times_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Pack Hermite coefficients for a uniformly sampled orbit."""
    dt = float(np.diff(times_s)[0])
    slope = np.diff(position_m, axis=0) / dt
    coefficients = np.empty((4, times_s.size - 1, 3), dtype=np.float64)
    coefficients[0] = (velocity_m_s[:-1] + velocity_m_s[1:] - 2.0 * slope) / dt**2
    coefficients[1] = (3.0 * slope - 2.0 * velocity_m_s[:-1] - velocity_m_s[1:]) / dt
    coefficients[2] = velocity_m_s[:-1]
    coefficients[3] = position_m[:-1]
    return coefficients, float(times_s[0]), dt


def _pack_orbit_general(
    times_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack Hermite coefficients for an arbitrarily sampled orbit."""
    interval = np.diff(times_s)[:, None]
    slope = np.diff(position_m, axis=0) / interval
    coefficients = np.empty((4, times_s.size - 1, 3), dtype=np.float64)
    coefficients[0] = (velocity_m_s[:-1] + velocity_m_s[1:] - 2.0 * slope) / interval**2
    coefficients[1] = (
        3.0 * slope - 2.0 * velocity_m_s[:-1] - velocity_m_s[1:]
    ) / interval
    coefficients[2] = velocity_m_s[:-1]
    coefficients[3] = position_m[:-1]
    return coefficients, np.asarray(times_s, dtype=np.float64)


def _trim_orbit(
    times_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    sensing_start_s: float,
    half_span_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only orbit states within ``sensing_start +/- half_span``."""
    low, high = sensing_start_s - half_span_s, sensing_start_s + half_span_s
    keep = (times_s >= low) & (times_s <= high)
    return times_s[keep], position_m[keep], velocity_m_s[keep]


def _orbit_inputs(
    model: RadarGeometryModel,
    *,
    half_span_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return scene-local orbit arrays plus sensing start in seconds."""
    orbit = model.orbit
    times_s = np.asarray(orbit.times_s, dtype=np.float64)
    positions = np.stack(
        [spline(times_s) for spline in orbit.trajectory_splines],
        axis=-1,
    )
    velocities = np.stack(
        [spline(times_s, 1) for spline in orbit.trajectory_splines],
        axis=-1,
    )
    sensing_offset_s = float((model.sensing_start - orbit.epoch).total_seconds())
    return *_trim_orbit(
        times_s,
        positions,
        velocities,
        sensing_start_s=sensing_offset_s,
        half_span_s=half_span_s,
    ), sensing_offset_s


def _llh_to_ecef_torch(
    latitude_deg: Any,
    longitude_deg: Any,
    height_m: Any,
) -> Any:
    """Convert geodetic coordinates to geocentric ECEF in double precision."""
    import torch

    radius = 6_378_137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = flattening * (2.0 - flattening)
    latitude = torch.deg2rad(latitude_deg)
    longitude = torch.deg2rad(longitude_deg)
    prime_vertical = radius / torch.sqrt(
        1.0 - eccentricity_squared * torch.sin(latitude) ** 2
    )
    x = (prime_vertical + height_m) * torch.cos(latitude) * torch.cos(longitude)
    y = (prime_vertical + height_m) * torch.cos(latitude) * torch.sin(longitude)
    z = (prime_vertical * (1.0 - eccentricity_squared) + height_m) * torch.sin(latitude)
    return torch.stack((x, y, z), dim=-1)


def _evaluate_orbit_interval(
    dt: Any,
    coefficients: Any,
    interval: Any,
) -> tuple[Any, Any, Any]:
    """Evaluate Hermite position, velocity, and acceleration by interval."""
    c0, c1, c2, c3 = (coefficients[index, interval, :] for index in range(4))
    position = ((c0 * dt[:, None] + c1) * dt[:, None] + c2) * dt[:, None] + c3
    velocity = (3.0 * c0 * dt[:, None] + 2.0 * c1) * dt[:, None] + c2
    acceleration = 6.0 * c0 * dt[:, None] + 2.0 * c1
    return position, velocity, acceleration


def _evaluate_orbit_searchsorted(
    times_s: Any,
    knot_times_s: Any,
    coefficients: Any,
) -> tuple[Any, Any, Any]:
    """Evaluate Hermite using a binary-search interval lookup."""
    import torch

    interval = torch.searchsorted(knot_times_s, times_s, right=True) - 1
    interval = interval.clamp(0, knot_times_s.numel() - 2)
    return _evaluate_orbit_interval(
        times_s - knot_times_s[interval], coefficients, interval
    )


def _along_track_seed_uniform(
    targets: Any,
    coefficients: Any,
    orbit_t0: float,
    orbit_dt: float,
    n_intervals: int,
    sensing_offset_s: float,
) -> Any:
    """Return the along-track projection seed for a uniform orbit."""
    import torch

    interval0 = int((sensing_offset_s - orbit_t0) / orbit_dt)
    interval0 = min(max(interval0, 0), n_intervals - 1)
    dt0 = sensing_offset_s - (orbit_t0 + interval0 * orbit_dt)
    c0, c1, c2, c3 = (coefficients[index, interval0, :] for index in range(4))
    pos0 = ((c0 * dt0 + c1) * dt0 + c2) * dt0 + c3
    vel0 = (3.0 * c0 * dt0 + 2.0 * c1) * dt0 + c2
    speed2 = torch.sum(vel0 * vel0)
    seed = sensing_offset_s + torch.sum((targets - pos0) * vel0, dim=-1) / speed2
    return seed.clamp(orbit_t0, orbit_t0 + n_intervals * orbit_dt)


def _along_track_seed_general(
    targets: Any,
    knot_times_s: Any,
    coefficients: Any,
    sensing_offset_s: float,
    device: str,
) -> Any:
    """Return the along-track projection seed for an arbitrary orbit."""
    import torch

    seed_times = torch.full(
        (targets.shape[0],), sensing_offset_s, dtype=torch.float64, device=device
    )
    satellite, velocity, _ = _evaluate_orbit_searchsorted(
        seed_times, knot_times_s, coefficients
    )
    speed2 = torch.sum(velocity * velocity, dim=-1)
    usable = torch.isfinite(speed2) & (speed2 > 0.0)
    along = torch.zeros_like(seed_times)
    along[usable] = (
        torch.sum((targets[usable] - satellite[usable]) * velocity[usable], dim=-1)
        / speed2[usable]
    )
    seed = sensing_offset_s + along
    return seed.clamp(knot_times_s[0], knot_times_s[-1])


def torch_geo2rdr(
    model: RadarGeometryModel,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    *,
    max_iter: int = _DEFAULT_MAX_ITER,
    time_tol_s: float = _DEFAULT_TIME_TOL_S,
    device: str | None = None,
    half_span_s: float = _DEFAULT_HALF_SPAN_S,
) -> dict[str, np.ndarray]:
    """Run the pure-Torch geo2rdr solver and return numpy arrays.

    Parameters
    ----------
    model : RadarGeometryModel
        Orbit and radar timing model.
    latitude_deg, longitude_deg, height_m : numpy.ndarray
        Geodetic coordinates in degrees and metres, broadcastable shapes.
    max_iter : int, optional
        Fixed Newton iteration budget. Four iterations are the validated
        safety ceiling across DEM extremes and strong Doppler.
    time_tol_s : float, optional
        Azimuth-time convergence tolerance in seconds.
    device : str or None, optional
        Torch device string (``"cpu"``, ``"cuda"``, or None for default).
    half_span_s : float, optional
        Scene-local orbit trimming window around sensing start.

    Returns
    -------
    dict of numpy.ndarray
        Keys ``range_index``, ``azimuth_index``, ``converged``,
        ``residual_doppler_hz``, ``residual_range_m``.

    """
    try:
        import torch
    except ImportError as error:
        message = (
            "Torch backend requires the 'torch' package; install it or use "
            "backend='numpy'."
        )
        logger.exception(message)
        raise ImportError(message) from error

    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    height = np.asarray(height_m, dtype=np.float64)
    lat_b, lon_b, h_b = np.broadcast_arrays(lat, lon, height)
    shape = lat_b.shape

    times_s, position_m, velocity_m_s, sensing_offset_s = _orbit_inputs(
        model,
        half_span_s=half_span_s,
    )
    uniform = np.allclose(np.diff(times_s), np.diff(times_s)[0])
    if uniform:
        coefficients, orbit_t0, orbit_dt = _pack_orbit_uniform(
            times_s, position_m, velocity_m_s
        )
        n_intervals = int(coefficients.shape[1])
    else:
        coefficients, knot_times = _pack_orbit_general(
            times_s, position_m, velocity_m_s
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lat_t = torch.as_tensor(lat_b.reshape(-1), dtype=torch.float64, device=device)
    lon_t = torch.as_tensor(lon_b.reshape(-1), dtype=torch.float64, device=device)
    hae_t = torch.as_tensor(h_b.reshape(-1), dtype=torch.float64, device=device)
    targets = _llh_to_ecef_torch(lat_t, lon_t, hae_t)
    finite = torch.isfinite(targets).all(dim=-1)

    coeff_t = torch.as_tensor(coefficients, dtype=torch.float64, device=device)
    if uniform:
        seed = _along_track_seed_uniform(
            targets,
            coeff_t,
            orbit_t0,
            orbit_dt,
            n_intervals,
            sensing_offset_s,
        )
    else:
        knot_t = torch.as_tensor(knot_times, dtype=torch.float64, device=device)
        seed = _along_track_seed_general(
            targets,
            knot_t,
            coeff_t,
            sensing_offset_s,
            device,
        )

    times = torch.where(finite, seed, torch.full_like(lat_t, sensing_offset_s))
    solved = torch.zeros_like(lat_t, dtype=torch.bool)
    doppler_out = torch.full_like(lat_t, torch.nan)
    slant_range = torch.zeros_like(lat_t)

    for _ in range(max_iter):
        if uniform:
            interval = ((times - orbit_t0) / orbit_dt).floor().long()
            interval = interval.clamp(0, n_intervals - 1)
            dt = times - (orbit_t0 + orbit_dt * interval.to(torch.float64))
            satellite, velocity, acceleration = _evaluate_orbit_interval(
                dt, coeff_t, interval
            )
        else:
            satellite, velocity, acceleration = _evaluate_orbit_searchsorted(
                times, knot_t, coeff_t
            )
        look = targets - satellite
        slant_range = torch.linalg.vector_norm(look, dim=-1)
        unit_look = look / slant_range[:, None]
        current_doppler = torch.sum(velocity * unit_look, dim=-1)
        velocity_squared = torch.sum(velocity * velocity, dim=-1)
        acceleration_along_look = torch.sum(acceleration * unit_look, dim=-1)
        derivative = acceleration_along_look + (
            current_doppler**2 - velocity_squared
        ) / slant_range
        usable = (
            torch.isfinite(derivative)
            & (torch.abs(derivative) >= 1e-12)
            & torch.isfinite(slant_range)
            & (slant_range > 0.0)
        )
        step = torch.where(
            usable,
            -current_doppler / derivative,
            torch.full_like(current_doppler, torch.nan),
        )
        times = torch.where(usable, times + step, times)
        doppler_out = torch.where(usable, current_doppler, doppler_out)
        solved = solved | (usable & (torch.abs(step) < time_tol_s))

    solved &= finite
    range_index = (slant_range - model.starting_slant_range_m) / model.range_spacing_m
    azimuth_index = (times - sensing_offset_s) / model.azimuth_time_interval_s
    range_index = torch.where(solved, range_index, torch.full_like(range_index, np.nan))
    azimuth_index = torch.where(
        solved, azimuth_index, torch.full_like(azimuth_index, np.nan)
    )
    residual_range = torch.zeros_like(range_index)
    residual_range = torch.where(
        solved, residual_range, torch.full_like(residual_range, np.nan)
    )
    doppler_out = torch.where(
        solved, doppler_out, torch.full_like(doppler_out, np.nan)
    )

    return {
        "range_index": range_index.reshape(shape).cpu().numpy(),
        "azimuth_index": azimuth_index.reshape(shape).cpu().numpy(),
        "converged": solved.reshape(shape).cpu().numpy(),
        "residual_doppler_hz": doppler_out.reshape(shape).cpu().numpy(),
        "residual_range_m": residual_range.reshape(shape).cpu().numpy(),
    }


@lru_cache(maxsize=1)
def cpp_library_path() -> Path | None:
    """Return the cached C++ shared-library path, if one is present."""
    explicit = os.environ.get("FANINSAR_GEO2RDR_LIB")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(_CPP_LIB_DIR / "libgeo2rdr.so")
    candidates.append(_CPP_LIB_DIR / "libgeo2rdr.dylib")
    candidates.append(_CPP_LIB_DIR / "geo2rdr.dll")
    candidates.append(
        Path.home() / ".cache" / "faninsar" / "geo2rdr" / "libgeo2rdr.so"
    )
    candidates.append(
        Path.home() / ".cache" / "faninsar" / "geo2rdr" / "libgeo2rdr.dylib"
    )
    candidates.append(
        Path.home() / ".cache" / "faninsar" / "geo2rdr" / "geo2rdr.dll"
    )
    for candidate in candidates:
        if candidate.is_file():
            logger.info("Using geo2rdr C++ library: %s", candidate)
            return candidate
    return None


def _load_cpp_library() -> Any | None:
    """Load the C++ shared library or return None when unavailable."""
    path = cpp_library_path()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path.resolve()))
    except OSError:
        logger.warning("Failed to load geo2rdr C++ library: %s", path)
        return None
    lib.geo2rdr_batch.restype = ctypes.c_int
    lib.geo2rdr_batch.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    return lib


def cpp_geo2rdr(
    model: RadarGeometryModel,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    *,
    max_iter: int = _DEFAULT_MAX_ITER,
    time_tol_s: float = _DEFAULT_TIME_TOL_S,
) -> dict[str, np.ndarray]:
    """Run the optional C++/OpenMP geo2rdr kernel.

    Raises
    ------
    RuntimeError
        If the shared library is not built or cannot be loaded.

    """
    lib = _load_cpp_library()
    if lib is None:
        message = (
            "C++ geo2rdr library is unavailable; build it with "
            "`uv run scripts/build_geo2rdr_cpp.py` or use backend='torch'."
        )
        logger.error(message)
        raise RuntimeError(message)

    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    height = np.asarray(height_m, dtype=np.float64).reshape(-1)
    lat, lon, height = np.broadcast_arrays(lat, lon, height)
    lat = np.ascontiguousarray(lat, dtype=np.float64)
    lon = np.ascontiguousarray(lon, dtype=np.float64)
    height = np.ascontiguousarray(height, dtype=np.float64)
    n = int(lat.size)

    times_s = np.asarray(model.orbit.times_s, dtype=np.float64)
    positions = np.stack(
        [spline(times_s) for spline in model.orbit.trajectory_splines],
        axis=-1,
    )
    velocities = np.stack(
        [spline(times_s, 1) for spline in model.orbit.trajectory_splines],
        axis=-1,
    )
    sensing_offset_s = float((model.sensing_start - model.orbit.epoch).total_seconds())
    az_out = np.empty(n, dtype=np.float64)
    range_out = np.empty(n, dtype=np.float64)
    conv_out = np.empty(n, dtype=np.int32)

    lib.geo2rdr_batch(
        lat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        height.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n),
        times_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positions.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        velocities.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(int(times_s.size)),
        ctypes.c_double(sensing_offset_s),
        ctypes.c_double(model.azimuth_time_interval_s),
        ctypes.c_double(model.starting_slant_range_m),
        ctypes.c_double(model.range_spacing_m),
        ctypes.c_double(model.wavelength_m),
        ctypes.c_double(0.0),
        ctypes.c_double(time_tol_s),
        ctypes.c_int(max_iter),
        ctypes.c_int(15),
        ctypes.c_int(1),
        ctypes.c_int(1),
        az_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        range_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        conv_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    target_shape = np.broadcast_shapes(
        np.asarray(latitude_deg).shape,
        np.asarray(longitude_deg).shape,
        np.asarray(height_m).shape,
    )
    converged = conv_out.astype(bool)
    doppler = np.full(int(np.prod(target_shape)), np.nan, dtype=np.float64)
    solved_indices = np.flatnonzero(converged)
    if solved_indices.size:
        solved_times = (
            sensing_offset_s
            + az_out[solved_indices] * model.azimuth_time_interval_s
        )
        solved_times = np.clip(
            solved_times,
            model.orbit.t_min_s,
            model.orbit.t_max_s,
        )
        satellite, velocity = model.orbit.evaluate_array(solved_times)
        target_x, target_y, target_z = llh_to_ecef(
            lat[solved_indices], lon[solved_indices], height[solved_indices]
        )
        targets = np.stack([target_x, target_y, target_z], axis=-1)
        look = targets - satellite
        range_m = np.linalg.norm(look, axis=1)
        unit_look = look / range_m[:, None]
        doppler[solved_indices] = np.einsum("ij,ij->i", velocity, unit_look)
    return {
        "range_index": range_out.reshape(target_shape),
        "azimuth_index": az_out.reshape(target_shape),
        "converged": converged.reshape(target_shape),
        "residual_doppler_hz": doppler.reshape(target_shape),
        "residual_range_m": np.where(converged.reshape(target_shape), 0.0, np.nan),
    }


def available_backends() -> tuple[str, ...]:
    """Return backend names usable in the current environment."""
    backends: list[str] = ["numpy"]
    try:
        import torch  # noqa: F401

        backends.append("torch")
    except ImportError:
        pass
    if _load_cpp_library() is not None:
        backends.append("cpp")
    return tuple(backends)


def resolve_backend(backend: BackendName) -> str:
    """Resolve a backend request to a concrete available backend."""
    available = available_backends()
    if backend != "auto":
        if backend in available:
            return backend
        logger.warning(
            "Requested geo2rdr backend '%s' is unavailable; available: %s",
            backend,
            available,
        )
        return "torch" if "torch" in available else "numpy"
    for candidate in ("cpp", "torch", "numpy"):
        if candidate in available:
            return candidate
    return "numpy"
