# ruff: noqa: TRY003, EM102
"""Canonical vertical-datum graph for DEM materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = setup_logger(__name__)

VerticalDatum = Literal["ellipsoidal", "egm96", "egm2008"]


def validate_datum(value: str) -> VerticalDatum:
    """Validate the closed target datum vocabulary."""
    if value not in {"ellipsoidal", "egm96", "egm2008"}:
        raise ValueError(f"unsupported vertical datum: {value}")
    return value  # type: ignore[return-value]


def conversion_models(source: VerticalDatum, target: VerticalDatum) -> tuple[str, ...]:
    """Return required geoid models in canonical conversion order.

    Same-datum conversion is empty.  A geoid-to-geoid conversion traverses
    ellipsoidal height and therefore requires both independent models.
    """
    validate_datum(source)
    validate_datum(target)
    if source == target:
        return ()
    if source == "ellipsoidal":
        return (target,)
    if target == "ellipsoidal":
        return (source,)
    return (source, target)


def requires_fetch(source: VerticalDatum, target: VerticalDatum) -> bool:
    """Return whether conversion needs at least one geoid model."""
    return bool(conversion_models(source, target))


def fetch_required(
    source: VerticalDatum,
    target: VerticalDatum,
    fetch: callable,
) -> tuple[object, ...]:
    """Fetch required models in graph order and return their loaded handles."""
    return tuple(fetch(model) for model in conversion_models(source, target))


def convert_heights(
    heights: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    source: VerticalDatum | str,
    target: VerticalDatum | str,
    *,
    fetch: object | None = None,
    samplers: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Convert DEM heights through the canonical vertical-datum graph.

    Parameters
    ----------
    heights : numpy.ndarray
        Source-datum elevations in metres.  Non-finite source values remain
        invalid in the result.
    longitude_deg, latitude_deg : numpy.ndarray
        WGS84 target-centre coordinates.  Longitude is periodic and latitude
        is checked by the selected geoid sampler.
    source, target : {"ellipsoidal", "egm96", "egm2008"}
        Source and requested output vertical datums.
    fetch : object, optional
        A :class:`~faninsar.processing.geometry.Fetch` instance.  It is passed to
        the lazy geoid loader and is therefore only used when a conversion
        actually requires a model.
    samplers : mapping, optional
        Internal test seam mapping model names to sampler objects exposing
        ``sample(latitude_deg, longitude_deg)``.  It does not alter public
        resource selection.

    Returns
    -------
    numpy.ndarray
        Float64 converted heights with NaN for invalid source or geoid cells.

    Notes
    -----
    EGM-to-EGM conversion explicitly follows ``h = H + N`` and
    ``H = h - N`` via WGS84 ellipsoidal height.  Conversion is pointwise and
    does not perform terrain or SAR-image resampling.

    """
    source_datum = validate_datum(str(source).strip().lower())
    target_datum = validate_datum(str(target).strip().lower())
    values, longitudes, latitudes = np.broadcast_arrays(
        np.asarray(heights, dtype=np.float64),
        np.asarray(longitude_deg, dtype=np.float64),
        np.asarray(latitude_deg, dtype=np.float64),
    )
    if np.any(~np.isfinite(longitudes)) or np.any(~np.isfinite(latitudes)):
        message = "datum conversion coordinates must be finite"
        logger.error(message)
        raise ValueError(message)
    if source_datum == target_datum:
        return np.array(values, dtype=np.float64, copy=True)

    model_samplers: dict[str, object] = dict(samplers or {})

    def model_sample(model: str) -> np.ndarray:
        sampler = model_samplers.get(model)
        if sampler is None:
            from .geoid import load_geoid

            sampler = load_geoid(model, fetch=fetch)  # type: ignore[arg-type]
        sample_method = getattr(sampler, "sample", None)
        if not callable(sample_method):
            message = f"geoid sampler for {model!r} has no sample method"
            logger.error(message)
            raise TypeError(message)
        undulation = np.asarray(sample_method(latitudes, longitudes), dtype=np.float64)
        if undulation.shape != values.shape:
            undulation = np.broadcast_to(undulation, values.shape)
        return np.asarray(undulation, dtype=np.float64)

    result = np.array(values, dtype=np.float64, copy=True)
    valid = np.isfinite(result)
    if source_datum != "ellipsoidal":
        undulation = model_sample(source_datum)
        valid &= np.isfinite(undulation)
        result = result + np.where(valid, undulation, 0.0)
    if target_datum != "ellipsoidal":
        undulation = model_sample(target_datum)
        valid &= np.isfinite(undulation)
        result = result - np.where(valid, undulation, 0.0)
    result[~valid] = np.nan
    return result


__all__ = [
    "VerticalDatum",
    "conversion_models",
    "convert_heights",
    "fetch_required",
    "requires_fetch",
    "validate_datum",
]
