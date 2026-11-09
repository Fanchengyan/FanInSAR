"""Tests for shared reference-point sampling and subset policy."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.core import Parameter
from faninsar.data.query.points import Points
from faninsar.data.reference import (
    ReferencePixelMap,
    copy_points,
    decode_points_payload,
    definitions_match,
    encode_points_payload,
    map_geographic_points,
    mean_offsets,
    nearest_pixels,
    normalize_stored_points,
    reference_definition,
    resolve_ref_points_argument,
    sample_at_pixels,
    select_valid_subsets,
    stored_definition_from_record,
    subtract_offsets,
)
from faninsar.processing.geocoding.coordinates import RadarCoordinates


def test_nearest_pixels_footprint_is_half_open_without_clipping() -> None:
    """Centres in ``[-0.5, size-0.5)`` are in-footprint; edges are not clipped."""
    rows, cols, inside = nearest_pixels(
        np.array([-0.5, 0.0, 3.4999, 3.5, 4.0], dtype=np.float64),
        np.array([0.0, -0.5, 1.0, 1.0, 1.0], dtype=np.float64),
        (4, 4),
    )
    np.testing.assert_array_equal(inside, [True, True, True, False, False])
    assert rows[0] == 0
    assert cols[1] == 0
    assert rows[3] == 4


def test_even_looks_centre_maps_first_layer_pixel_to_zero() -> None:
    """Look-2 box-average centre of the first pixel sits at full-radar 0.5."""
    coords = RadarCoordinates(
        np.array([[0.5]], dtype=np.float64),
        np.array([[0.5]], dtype=np.float64),
        np.array([[True]]),
    )
    layer = coords.to_layer_frame(origin=(0, 0), looks=(2, 2))
    assert layer.az[0, 0] == pytest.approx(0.0)
    assert layer.rg[0, 0] == pytest.approx(0.0)


def test_nonzero_crop_origin_maps_native_sample_to_layer_zero() -> None:
    """A cropped origin is subtracted before nearest-pixel sampling."""
    coords = RadarCoordinates(
        np.array([[10.0]], dtype=np.float64),
        np.array([[20.0]], dtype=np.float64),
        np.array([[True]]),
    )
    layer = coords.to_layer_frame(origin=(10, 20), looks=(1, 1))
    assert layer.az[0, 0] == pytest.approx(0.0)
    assert layer.rg[0, 0] == pytest.approx(0.0)
    rows, cols, inside = nearest_pixels(
        layer.az.reshape(-1), layer.rg.reshape(-1), (4, 5)
    )
    assert inside[0]
    assert int(rows[0]) == 0
    assert int(cols[0]) == 0


def test_geographic_affine_uses_pixel_centres() -> None:
    """Inverse affine corners are shifted by -0.5 onto pixel centres."""
    transform = (1.0, 0.0, 10.0, 0.0, -1.0, 50.0)
    points = Points([[10.5, 49.5]], crs="EPSG:4326")
    mapping = map_geographic_points(
        points, transform=transform, shape=(4, 6), crs="EPSG:4326"
    )
    assert mapping.mapped[0]
    assert mapping.rows[0] == 0
    assert mapping.cols[0] == 0
    assert mapping.row_frac[0] == pytest.approx(0.0)
    assert mapping.col_frac[0] == pytest.approx(0.0)


def test_copy_and_codec_preserve_dtype() -> None:
    """Stored copies keep the constructor dtype, including float64."""
    points = Points([[100.1, 31.2], [100.3, 31.4]], crs="EPSG:4326", dtype=np.float64)
    copied = copy_points(points)
    assert copied.dtype == np.dtype(np.float64)
    payload = encode_points_payload(points)
    restored = decode_points_payload(payload)
    assert restored.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(restored.values, points.values)


def test_crs_less_points_normalize_to_wgs84() -> None:
    """CRS-less input is stored as explicit WGS84 longitude, latitude."""
    stored = normalize_stored_points(Points([[12.0, 48.0]]))
    assert stored.crs is not None
    assert stored.crs.to_epsg() == 4326


def test_codec_rejects_nonfinite_and_empty_payloads() -> None:
    """Decoded payloads must be nonempty finite (n, 2) coordinates."""
    with pytest.raises(ValueError, match="nonempty"):
        decode_points_payload(
            {"coordinates": [], "crs": "EPSG:4326", "dtype": "float32"}
        )
    with pytest.raises(ValueError, match="finite"):
        decode_points_payload(
            {
                "coordinates": [[1.0, np.inf]],
                "crs": "EPSG:4326",
                "dtype": "float32",
            }
        )
    with pytest.raises(ValueError, match="dtype"):
        decode_points_payload(
            {"coordinates": [[1.0, 2.0]], "crs": "EPSG:4326", "dtype": "not-a-dtype"}
        )


def test_common_intersection_warns_once_and_per_image_warns_on_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Policies emit one summary warning and error when a required subset is empty."""
    valid = {
        "a": np.array([True, True, False]),
        "b": np.array([True, False, False]),
    }
    with caplog.at_level("WARNING"):
        common = select_valid_subsets(valid, "common")
    assert np.array_equal(common["a"], [True, False, False])
    assert "excluded 2 of 3" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        per_image = select_valid_subsets(valid, "per_image")
    assert np.array_equal(per_image["a"], [True, True, False])
    assert "differ" in caplog.text
    empty = {"a": np.array([False, False]), "b": np.array([True, False])}
    with pytest.raises(ValueError, match="no valid reference points"):
        select_valid_subsets(empty, "per_image")
    with pytest.raises(ValueError, match="every image"):
        select_valid_subsets({"a": np.array([False]), "b": np.array([False])}, "common")


def test_duplicates_keep_weight_and_component_labels_are_ignored() -> None:
    """Two requests of one pixel both enter the mean; labels are not a mask."""
    mapping = ReferencePixelMap(
        row_frac=np.array([0.0, 0.0], dtype=np.float64),
        col_frac=np.array([0.0, 0.0], dtype=np.float64),
        rows=np.array([0, 0]),
        cols=np.array([0, 0]),
        mapped=np.array([True, True]),
    )
    values = np.array([[4.0, 9.0], [1.0, 2.0]], dtype=np.float32)
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    sampled = sample_at_pixels(values, mapping, support_mask=None)
    np.testing.assert_allclose(sampled, [4.0, 4.0])
    offsets = mean_offsets({"img": sampled}, {"img": np.array([True, True])})
    assert offsets["img"] == pytest.approx(4.0)
    referenced = subtract_offsets({"img": values}, offsets)
    np.testing.assert_allclose(referenced["img"][0, 0], 0.0)
    np.testing.assert_array_equal(values[0, 0], 4.0)
    assert labels[0, 0] == 1


def test_resolve_inherit_none_and_override() -> None:
    """Inherit copies session points; None disables; AUTO is rejected."""
    inherited = Points([[1.0, 2.0]], crs="EPSG:4326")
    points, policy = resolve_ref_points_argument(
        "inherit", inherited, inherited_policy="per_image"
    )
    assert policy == "per_image"
    np.testing.assert_array_equal(points.values, inherited.values)
    points.values[0, 0] = 0.0
    assert inherited.values[0, 0] != 0.0
    disabled, policy = resolve_ref_points_argument(None, inherited, "per_image")
    assert disabled is None
    assert policy == "per_image"
    with pytest.raises(ValueError, match="automatic"):
        resolve_ref_points_argument(Parameter.AUTO, inherited, "common")
    none, policy = resolve_ref_points_argument("inherit", None, "common")
    assert none is None
    assert policy == "common"


def test_missing_reference_metadata_matches_only_disabled_requests() -> None:
    """Old unreferenced products must not match a nonempty request."""
    points = Points([[1.0, 2.0]], crs="EPSG:4326")
    requested = reference_definition(points, "common")
    assert definitions_match(None, None)
    assert not definitions_match(None, requested)
    assert not definitions_match(stored_definition_from_record({}), requested)
    record = {
        "coordinates": requested["coordinates"],
        "policy": "common",
        "method": "arithmetic_mean",
        "crs": requested["crs"],
    }
    assert definitions_match(stored_definition_from_record(record), requested)


def test_crs_conversion_keeps_float64() -> None:
    """Mapping conversion does not truncate through Points.to_crs float32."""
    from faninsar.data.reference import coordinates_in_crs

    easting, northing = 500000.123456789, 4000000.987654321
    points = Points([[easting, northing]], crs="EPSG:32632", dtype=np.float64)
    x, y = coordinates_in_crs(points, "EPSG:4326")
    assert x.dtype == np.dtype(np.float64)
    assert np.isfinite(x[0])
    assert np.isfinite(y[0])


def test_policy_rejects_non_string_values() -> None:
    """Array-valued policy input must not compare as a set membership."""
    with pytest.raises(ValueError, match="ref_policy"):
        select_valid_subsets({"a": np.array([True])}, ["common"])
