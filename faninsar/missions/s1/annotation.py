"""Parse Sentinel-1 Level-1 annotation XML into typed metadata."""

from __future__ import annotations

import numbers
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.logging import setup_logger
from faninsar.missions.protocols import DopplerCentroidPolynomial

from .errors import reject_product
from .types import ElevationAngleRecord, S1Burst, S1Swath

if TYPE_CHECKING:
    from pathlib import Path

logger = setup_logger(__name__)

SPEED_OF_LIGHT_M_S = 299_792_458.0
PARENT_BURST_NUMBER_NS = "https://faninsar.org/ns/s1"
_PARENT_BURST_NUMBER_TAG = f"{{{PARENT_BURST_NUMBER_NS}}}number"


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


def _qualified_parent_burst_number(
    value: object,
    *,
    annotation_path: str = "",
) -> int:
    """Return ``value`` as a zero-based parent ordinal, or reject it."""
    where = f" in {annotation_path}" if annotation_path else ""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        reject_product(f"invalid parent burst number {value!r}{where}")
    number = int(value)
    if number < 0:
        reject_product(f"negative parent burst number {number}{where}")
    return number


def _read_parent_burst_number(
    burst_el: ET.Element,
    *,
    annotation_path: str = "",
) -> int | None:
    """Return the namespaced original-parent ordinal, or ``None`` if absent.

    Un-namespaced ``<number>`` children are ignored.  Malformed or negative
    explicit values are rejected rather than guessed from local order.
    """
    found = [child for child in burst_el if child.tag == _PARENT_BURST_NUMBER_TAG]
    if not found:
        return None
    where = f" in {annotation_path}" if annotation_path else ""
    if len(found) > 1:
        reject_product(f"multiple parent burst number extensions{where}")
    text = (found[0].text or "").strip()
    if not text:
        reject_product(f"empty parent burst number{where}")
    try:
        parsed = int(text, 10)
    except ValueError:
        reject_product(f"invalid parent burst number {text!r}{where}")
    return _qualified_parent_burst_number(parsed, annotation_path=annotation_path)


def _resolved_parent_burst_number(
    existing: int | None,
    mapped: object,
    *,
    annotation_path: str = "",
) -> int | None:
    """Combine a persisted stamp with an explicit source ordinal.

    Distinct non-``None`` values are a contradictory trusted mapping.
    """
    mapped_number = (
        None
        if mapped is None
        else _qualified_parent_burst_number(mapped, annotation_path=annotation_path)
    )
    if existing is None:
        return mapped_number
    if mapped_number is None or mapped_number == existing:
        return existing
    where = f" in {annotation_path}" if annotation_path else ""
    message = f"parent burst number {existing} contradicts {mapped_number}{where}"
    reject_product(message)
    raise AssertionError(message)


def _stamp_parent_burst_number(burst_el: ET.Element, number: object) -> None:
    """Write one namespaced original-parent ordinal onto ``burst_el``."""
    value = _qualified_parent_burst_number(number)
    ET.register_namespace("faninsar", PARENT_BURST_NUMBER_NS)
    found = [child for child in burst_el if child.tag == _PARENT_BURST_NUMBER_TAG]
    if found:
        found[0].text = str(value)
        for extra in found[1:]:
            burst_el.remove(extra)
        return
    element = ET.SubElement(burst_el, _PARENT_BURST_NUMBER_TAG)
    element.text = str(value)


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


def _geolocation_grid_points(
    root: ET.Element,
) -> dict[int, list[tuple[float, float]]]:
    """Return ``{line: [(lon, lat), ...]}`` grouped by physical TIFF row.

    ``line`` is the slot row index.  Burst *i*'s footprint is the bounding
    box of the gridline sitting at ``i*lines_per_burst`` — the swath's own
    scan line, which matches the equal-height nominal burst grid ASF's
    ``processingLevel=BURST`` catalogue returns.  Points keep document
    order within the line (the range direction).
    """
    grid = next(
        (child for child in root if _local(child.tag) == "geolocationGrid"), None
    )
    if grid is None:
        return {}
    rows: dict[int, list[tuple[float, float]]] = {}
    for gp in grid.iter():
        if _local(gp.tag) != "geolocationGridPoint":
            continue
        line = int(_text(gp, "line"))
        lat = float(_text(gp, "latitude"))
        lon = float(_text(gp, "longitude"))
        rows.setdefault(line, []).append((lon, lat))
    return rows


def _parse_burst_footprints(
    root: ET.Element,
    lines_per_burst: int,
) -> list[tuple[tuple[float, float], ...]] | None:
    """Return one polygon per burst from the geolocation grid.

    Burst *i*'s footprint is the bounding box of the geolocation gridline
    at its slot start ``i*lines_per_burst``.  Using each burst's own scan
    line — instead of every gridline up to the product's final row — keeps
    all footprints the same height and stops the last burst from absorbing
    the slice's trailing TOPS-overlap rows (~2x inflation).

    When a slot start has no sampled gridline, the nearest sampled line is
    used so a burst still yields a finite rectangle.
    """
    rows = _geolocation_grid_points(root)
    if not rows:
        return None
    lines = sorted(rows)
    max_line = lines[-1]
    n_bursts = max_line // lines_per_burst + 1

    footprints: list[tuple[tuple[float, float], ...]] = []
    for index in range(n_bursts):
        slot_start = index * lines_per_burst
        if slot_start in rows:
            pts = rows[slot_start]
        else:
            nearest = min(lines, key=lambda ln: abs(ln - slot_start))
            pts = rows[nearest]
        lats = [c[1] for c in pts]
        lons = [c[0] for c in pts]
        footprints.append(
            (
                (min(lons), min(lats)),
                (max(lons), min(lats)),
                (max(lons), max(lats)),
                (min(lons), max(lats)),
            )
        )
    return footprints


def _parse_elevation_angle_records(
    root: ET.Element,
) -> tuple[ElevationAngleRecord, ...]:
    """Return annotation ``antennaPattern`` elevation-angle records, if present."""
    container = next(
        (child for child in root if _local(child.tag) == "antennaPattern"), None
    )
    if container is None:
        return ()
    pattern_list = next(
        (child for child in container if _local(child.tag) == "antennaPatternList"),
        None,
    )
    if pattern_list is None:
        return ()
    records: list[ElevationAngleRecord] = []
    for element in pattern_list:
        if _local(element.tag) != "antennaPattern":
            continue
        azimuth = _optional_text(element, "azimuthTime")
        taus = _optional_text(element, "slantRangeTime")
        angles = _optional_text(element, "elevationAngle")
        if azimuth is None or taus is None or angles is None:
            continue
        tau_values = _floats(taus)
        angle_values = _floats(angles)
        if len(tau_values) != len(angle_values) or len(tau_values) < 2:
            continue
        records.append(
            ElevationAngleRecord(
                azimuth_time=_parse_time(azimuth),
                slant_range_time_s=tau_values,
                elevation_angle_deg=angle_values,
            )
        )
    return tuple(records)


def parse_annotation_xml(
    xml_text: str,
    *,
    annotation_path: str,
    measurement_path: str,
    _original_parent: bool = False,
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
    _original_parent : bool, optional
        Private source-context flag used by mission owners that already hold
        a complete original parent annotation (CDSE parent fetch).  When
        True, missing numbers are the enumerate ordinals.  Generic callers,
        including :func:`load_annotation_file` and SAFE directory/ZIP open,
        must leave this False: a full-looking burstList or filename is not
        qualification, so ``number`` stays ``None`` unless a namespaced
        stamp is present.

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
    saw_absolute_burst_id = False
    for index, burst_el in enumerate(
        child for child in burst_list if _local(child.tag) == "burst"
    ):
        first = _ints(_text(burst_el, "firstValidSample"))
        last = _ints(_text(burst_el, "lastValidSample"))
        if len(first) != lines_per_burst or len(last) != lines_per_burst:
            reject_product(
                f"burst {index} valid-sample length mismatch in {annotation_path}"
            )
        # Standard ESA annotation XML carries no absolute-burst id; some
        # reprocessed/derived products emit ``absoluteBurstID`` or
        # ``burstIdAbsolute`` and we adopt it when present.
        absolute_id_text = _optional_text(burst_el, "absoluteBurstID") or (
            _optional_text(burst_el, "burstIdAbsolute")
        )
        absolute_burst_id: int | None = None
        if absolute_id_text is not None:
            try:
                absolute_burst_id = int(absolute_id_text)
                saw_absolute_burst_id = True
            except ValueError:
                absolute_burst_id = None
        stamped = _read_parent_burst_number(
            burst_el,
            annotation_path=annotation_path,
        )
        if _original_parent:
            if stamped is None:
                number: int | None = index
            elif stamped != index:
                reject_product(
                    f"parent burst number {stamped} contradicts original "
                    f"burstList ordinal {index} in {annotation_path}"
                )
            else:
                number = stamped
        else:
            number = stamped
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
                measurement_path=measurement_path,
                absolute_burst_id=absolute_burst_id,
                number=number,
            )
        )
    if not bursts:
        reject_product(f"no bursts found in {annotation_path}")
    if not saw_absolute_burst_id:
        # ``sensingTime`` plus ``azimuthAnxTime`` remain the cross-mission
        # matching keys when no annotation-level absolute id exists.
        logger.debug(
            "annotation %s exposes no absoluteBurstID; using sensing_time and "
            "azimuth_anx_time_s as burst matching keys",
            annotation_path,
        )

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
                # IPF < 2.40 stores coefficients as separate c0, c1, ... elements
                coeffs = sorted(
                    (
                        item
                        for item in fm_el
                        if _local(item.tag)[0] == "c" and _local(item.tag)[1:].isdigit()
                    ),
                    key=lambda item: int(_local(item.tag)[1:]),
                )
                poly = " ".join(item.text.strip() for item in coeffs) or None
            if poly is None:
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
    ascending_node = _optional_text(image_info, "ascendingNodeTime")

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
        ascending_node_time=(
            _parse_time(ascending_node) if ascending_node is not None else None
        ),
        elevation_angle_records=_parse_elevation_angle_records(root),
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
