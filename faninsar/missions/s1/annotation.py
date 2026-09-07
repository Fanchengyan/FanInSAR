"""Parse Sentinel-1 Level-1 annotation XML into typed metadata."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.logging import setup_logger
from faninsar.missions.protocols import DopplerCentroidPolynomial

from .errors import reject_product
from .types import S1Burst, S1Swath

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: ET.Element, name: str) -> ET.Element:
    for child in parent:
        if _local(child.tag) == name:
            return child
    message = f"missing required annotation element: {name}"
    reject_product(message)
    raise AssertionError(message)


def _text(parent: ET.Element, name: str) -> str:
    value = (_child(parent, name).text or "").strip()
    if not value:
        reject_product(f"empty annotation field: {name}")
    return value


def _optional_text(parent: ET.Element, name: str) -> str | None:
    for child in parent:
        if _local(child.tag) == name:
            value = (child.text or "").strip()
            return value or None
    return None


def _parse_time(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split())


def _parse_burst_footprints(
    root: ET.Element,
    lines_per_burst: int,
) -> list[tuple[tuple[float, float], ...]] | None:
    grid = None
    for child in root:
        if _local(child.tag) == "geolocationGrid":
            grid = child
            break
    if grid is None:
        return None
    points: list[tuple[int, float, float]] = []
    for gp in grid.iter():
        if _local(gp.tag) != "geolocationGridPoint":
            continue
        line = int(_text(gp, "line"))
        lat = float(_text(gp, "latitude"))
        lon = float(_text(gp, "longitude"))
        points.append((line, lat, lon))
    if not points:
        return None
    max_line = max(p[0] for p in points)
    n_bursts = max_line // lines_per_burst + 1
    footprints: list[tuple[tuple[float, float], ...]] = []
    for index in range(n_bursts):
        low = index * lines_per_burst
        high = (
            max_line + 1
            if index == n_bursts - 1
            else min((index + 1) * lines_per_burst, max_line + 1)
        )
        rows = [p for p in points if low <= p[0] < high]
        if not rows:
            mid = (low + high - 1) / 2.0
            rows = [min(points, key=lambda p: abs(p[0] - mid))]
        lats = [p[1] for p in rows]
        lons = [p[2] for p in rows]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        footprints.append(
            (
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
            )
        )
    return footprints


def parse_annotation_xml(
    xml_text: str,
    *,
    annotation_path: str,
    measurement_path: str,
) -> S1Swath:
    """Parse one sub-swath annotation document.

    Parameters
    ----------
    xml_text : str
        Annotation XML document body.
    annotation_path : str
        Logical path of the annotation asset.
    measurement_path : str
        Logical path of the paired measurement GeoTIFF.

    Returns
    -------
    S1Swath
        Typed sub-swath metadata with bursts, orbit, and Doppler terms.

    """
    root = ET.fromstring(xml_text)
    ads = _child(root, "adsHeader")
    general = _child(root, "generalAnnotation")
    product_info = _child(general, "productInformation")
    image_info = _child(_child(root, "imageAnnotation"), "imageInformation")
    swath_timing = _child(root, "swathTiming")
    doppler_root = _child(root, "dopplerCentroid")

    swath_name = _text(ads, "swath")
    polarization = _text(ads, "polarisation")
    lines_per_burst = int(_text(swath_timing, "linesPerBurst"))
    samples_per_burst = int(_text(swath_timing, "samplesPerBurst"))

    footprints = _parse_burst_footprints(root, lines_per_burst)
    bursts: list[S1Burst] = []
    burst_list = _child(swath_timing, "burstList")
    for index, burst_el in enumerate(
        child for child in burst_list if _local(child.tag) == "burst"
    ):
        first = _ints(_text(burst_el, "firstValidSample"))
        last = _ints(_text(burst_el, "lastValidSample"))
        if len(first) != lines_per_burst or len(last) != lines_per_burst:
            reject_product(
                f"burst {index} valid-sample length mismatch in {annotation_path}"
            )
        bursts.append(
            S1Burst(
                index=index,
                azimuth_time=_parse_time(_text(burst_el, "azimuthTime")),
                sensing_time=_parse_time(_text(burst_el, "sensingTime")),
                azimuth_anx_time_s=float(_text(burst_el, "azimuthAnxTime")),
                byte_offset=int(_text(burst_el, "byteOffset")),
                lines=lines_per_burst,
                samples=samples_per_burst,
                first_valid_sample=first,
                last_valid_sample=last,
                footprint=(
                    footprints[index]
                    if footprints is not None and index < len(footprints)
                    else None
                ),
            )
        )
    if not bursts:
        reject_product(f"no bursts found in {annotation_path}")

    vectors: list[OrbitStateVector] = []
    orbit_list = _child(general, "orbitList")
    for orbit_el in (child for child in orbit_list if _local(child.tag) == "orbit"):
        position = _child(orbit_el, "position")
        velocity = _child(orbit_el, "velocity")
        vectors.append(
            OrbitStateVector(
                time=_parse_time(_text(orbit_el, "time")),
                position_m=(
                    float(_text(position, "x")),
                    float(_text(position, "y")),
                    float(_text(position, "z")),
                ),
                velocity_m_s=(
                    float(_text(velocity, "x")),
                    float(_text(velocity, "y")),
                    float(_text(velocity, "z")),
                ),
            )
        )
    if len(vectors) < 2:
        reject_product(f"need >=2 orbit vectors in {annotation_path}")

    doppler: list[DopplerCentroidPolynomial] = []
    dc_list = _child(doppler_root, "dcEstimateList")
    for dc_el in (child for child in dc_list if _local(child.tag) == "dcEstimate"):
        poly_text = _optional_text(dc_el, "dataDcPolynomial") or _text(
            dc_el,
            "geometryDcPolynomial",
        )
        doppler.append(
            DopplerCentroidPolynomial(
                coefficients_hz=_floats(poly_text),
                reference_range_m=float(_text(dc_el, "t0")) * 299_792_458.0 / 2.0,
                reference_time_s=0.0,
            )
        )
    if not doppler:
        reject_product(f"missing Doppler centroid estimates in {annotation_path}")

    fm_rates: list[tuple[datetime, float, tuple[float, ...]]] = []
    # Scan children (namespace-safe).
    for child in general:
        if _local(child.tag) != "azimuthFmRateList":
            continue
        for fm_el in (item for item in child if _local(item.tag) == "azimuthFmRate"):
            poly = _optional_text(fm_el, "azimuthFmRatePolynomial")
            if poly is None:
                # some products nest coefficients differently
                continue
            fm_rates.append(
                (
                    _parse_time(_text(fm_el, "azimuthTime")),
                    float(_text(fm_el, "t0")),
                    _floats(poly),
                )
            )

    range_sampling_rate_hz = float(_text(product_info, "rangeSamplingRate"))
    # ``rangePixelSpacing`` is rounded to six decimal places in Sentinel-1
    # annotation.  That is adequate for image display, but not for InSAR
    # geometry: the accumulated range error across an IW burst becomes a
    # sub-millimetre path error and therefore a radian-scale flattening phase
    # error.  ISCE2 derives the spacing from the sampling rate for this reason.
    range_pixel_spacing_m = SPEED_OF_LIGHT_M_S / (2.0 * range_sampling_rate_hz)

    return S1Swath(
        swath=swath_name,  # type: ignore[arg-type]
        polarization=polarization,
        annotation_path=annotation_path,
        measurement_path=measurement_path,
        lines=int(_text(image_info, "numberOfLines")),
        samples=int(_text(image_info, "numberOfSamples")),
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        range_pixel_spacing_m=range_pixel_spacing_m,
        azimuth_pixel_spacing_m=float(_text(image_info, "azimuthPixelSpacing")),
        azimuth_time_interval_s=float(_text(image_info, "azimuthTimeInterval")),
        slant_range_time_s=float(_text(image_info, "slantRangeTime")),
        range_sampling_rate_hz=range_sampling_rate_hz,
        radar_frequency_hz=float(_text(product_info, "radarFrequency")),
        sensing_start=_parse_time(_text(ads, "startTime")),
        sensing_stop=_parse_time(_text(ads, "stopTime")),
        bursts=tuple(bursts),
        orbit=OrbitMetadata(
            reference_frame="Earth Fixed",
            source=annotation_path,
            vectors=tuple(vectors),
        ),
        doppler_centroid=tuple(doppler),
        azimuth_fm_rate=tuple(fm_rates),
        azimuth_steering_rate_rad_s=float(_text(product_info, "azimuthSteeringRate"))
        * (3.141592653589793 / 180.0),
    )


def load_annotation_file(
    path: Path,
    *,
    measurement_path: str,
) -> S1Swath:
    """Load and parse an annotation file from disk."""
    return parse_annotation_xml(
        path.read_text(encoding="utf-8"),
        annotation_path=str(path),
        measurement_path=measurement_path,
    )
