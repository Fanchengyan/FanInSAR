"""Tests for the RadarCoordinates currency and the geocode kernel table."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from faninsar.processing.coregistration.offsets import OffsetFieldResult
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.processing.geocoding.coordinates import (
    RadarCoordinates,
    _minus_offsets_torch,
)
from faninsar.processing.geocoding.geo_resample import (
    resample_complex_at_coordinates,
    sample,
)


def _grid_coordinates(rows: int, cols: int) -> RadarCoordinates:
    row_indices, col_indices = np.indices((rows, cols), dtype=np.float64)
    return RadarCoordinates(
        row_indices,
        col_indices,
        np.ones((rows, cols), dtype=bool),
    )


def _offsets(
    shape: tuple[int, int],
    azimuth: np.ndarray,
    range_offset: np.ndarray,
    coverage: np.ndarray | None = None,
) -> OffsetFieldResult:
    return OffsetFieldResult(
        range_offset_px=np.asarray(range_offset, dtype=np.float32),
        azimuth_offset_px=np.asarray(azimuth, dtype=np.float32),
        coverage=(
            np.ones(shape, dtype=bool) if coverage is None else np.asarray(coverage)
        ),
        uncertainty_px=np.zeros(shape, dtype=np.float32),
    )


def test_coordinates_reject_mismatched_arrays() -> None:
    """Non-matching or non-2-D coordinate arrays are rejected on construction."""
    with pytest.raises(InvalidProcessingStateError):
        RadarCoordinates(
            np.zeros((2, 3)),
            np.zeros((3, 2)),
            np.ones((2, 3), dtype=bool),
        )
    with pytest.raises(InvalidProcessingStateError):
        RadarCoordinates(np.zeros(3), np.zeros(3), np.ones(3, dtype=bool))
    with pytest.raises(InvalidProcessingStateError):
        RadarCoordinates(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.ones((2, 3), dtype=np.uint8),
        )


@pytest.mark.parametrize(
    ("origin", "looks"),
    [((0, 0), (1, 1)), ((0, 0), (3, 3)), ((2, 1), (2, 4)), ((4, 0), (3, 2))],
)
def test_to_layer_frame_matches_box_average_center(
    origin: tuple[int, int],
    looks: tuple[int, int],
) -> None:
    """Layer mapping inverts the multilook box-average center convention.

    ``multilook_interferogram_torch`` averages non-overlapping blocks
    ``[i * L, (i + 1) * L)``, whose centers sit at ``i * L + (L - 1) / 2``;
    ``to_layer_frame`` must map those centers back to the exact layer index
    ``i`` for odd and even look factors alike.
    """
    rows, cols = 8, 12
    az_looks, rg_looks = looks
    base = _grid_coordinates(rows, cols)
    coords = base.to_layer_frame(origin=origin, looks=looks)
    expected_az = (base.az - origin[0] - (az_looks - 1) / 2.0) / az_looks
    expected_rg = (base.rg - origin[1] - (rg_looks - 1) / 2.0) / rg_looks
    np.testing.assert_allclose(coords.az, expected_az)
    np.testing.assert_allclose(coords.rg, expected_rg)
    assert coords.valid.all()


def test_to_layer_frame_maps_block_centers_to_layer_indices() -> None:
    """Block centers of an actual box average sample their own layer pixel."""
    rows, cols = 8, 12
    az_looks, rg_looks = 2, 3
    row_indices, col_indices = np.indices((rows, cols), dtype=np.float64)
    field = 3.0 * row_indices - 2.0 * col_indices
    box_average = field.reshape(
        rows // az_looks,
        az_looks,
        cols // rg_looks,
        rg_looks,
    ).mean(axis=(1, 3))

    center_az = (np.arange(rows // az_looks) * az_looks + (az_looks - 1) / 2.0)[
        :, None
    ] * np.ones((1, cols // rg_looks))
    center_rg = (np.arange(cols // rg_looks) * rg_looks + (rg_looks - 1) / 2.0)[
        None, :
    ] * np.ones((rows // az_looks, 1))
    full_frame = RadarCoordinates(
        center_az,
        center_rg,
        np.ones(box_average.shape, dtype=bool),
    )
    layer_frame = full_frame.to_layer_frame(origin=(0, 0), looks=(az_looks, rg_looks))

    sampled, valid = sample(box_average, layer_frame, kind="continuous")
    assert valid.all()
    np.testing.assert_allclose(sampled, box_average, atol=1e-12)


def test_to_layer_frame_identity_for_single_look() -> None:
    """``looks=(1, 1)`` with a zero origin leaves coordinates untouched."""
    coords = _grid_coordinates(3, 4)
    layer = coords.to_layer_frame()
    np.testing.assert_array_equal(layer.az, coords.az)
    np.testing.assert_array_equal(layer.rg, coords.rg)
    np.testing.assert_array_equal(layer.valid, coords.valid)


@pytest.mark.parametrize(
    ("looks", "origin"),
    [((0, 1), (0, 0)), ((1, -2), (0, 0)), ((1, 1), (-1, 0)), ((1.5, 1), (0, 0))],
)
def test_to_layer_frame_rejects_invalid_geometry(
    looks: tuple[float, float],
    origin: tuple[float, float],
) -> None:
    """Non-positive looks and negative origins are rejected."""
    coords = _grid_coordinates(2, 2)
    with pytest.raises(InvalidProcessingStateError):
        coords.to_layer_frame(origin=origin, looks=looks)


def test_minus_offsets_matches_retired_compose_semantics() -> None:
    """``minus_offsets`` reproduces the retired compose semantics exactly.

    The reference below mirrors the retired implementation: bilinear
    ``map_coordinates`` of the dense offset fields at the reference LUT
    coordinates, reference-minus-secondary subtraction, and the coverage,
    finiteness, and radar-window masking.
    """

    def reference(
        az: np.ndarray,
        rg: np.ndarray,
        valid: np.ndarray,
        offsets: OffsetFieldResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.array(
            [
                np.asarray(az, dtype=np.float64).ravel(),
                np.asarray(rg, dtype=np.float64).ravel(),
            ],
        )
        azimuth_offset = map_coordinates(
            np.asarray(offsets.azimuth_offset_px, dtype=np.float32),
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
        ).reshape(np.shape(az))
        range_offset = map_coordinates(
            np.asarray(offsets.range_offset_px, dtype=np.float32),
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
        ).reshape(np.shape(az))
        coverage = map_coordinates(
            np.asarray(offsets.coverage, dtype=np.uint8),
            coordinates,
            order=0,
            mode="constant",
            cval=0,
        ).reshape(np.shape(az))
        secondary_az = np.asarray(az) - azimuth_offset
        secondary_rg = np.asarray(rg) - range_offset
        height, width = offsets.coverage.shape
        composed_valid = (
            np.asarray(valid, dtype=bool)
            & coverage.astype(bool)
            & np.isfinite(secondary_az)
            & np.isfinite(secondary_rg)
            & (secondary_az >= 0.0)
            & (secondary_az <= height - 1.0)
            & (secondary_rg >= 0.0)
            & (secondary_rg <= width - 1.0)
        )
        return secondary_az, secondary_rg, composed_valid

    rng = np.random.default_rng(7)
    radar_shape = (16, 20)
    grid_shape = (6, 8)
    az = rng.uniform(1.0, 14.0, size=grid_shape)
    rg = rng.uniform(1.0, 18.0, size=grid_shape)
    valid = np.ones(grid_shape, dtype=bool)
    valid[0, 0] = False
    offsets = _offsets(
        radar_shape,
        rng.normal(0.0, 0.4, size=radar_shape),
        rng.normal(0.0, 0.4, size=radar_shape),
    )
    coords = RadarCoordinates(az, rg, valid)

    secondary = coords.minus_offsets(offsets, device="cpu")
    expected_az, expected_rg, expected_valid = reference(az, rg, valid, offsets)

    np.testing.assert_array_equal(secondary.az, expected_az)
    np.testing.assert_array_equal(secondary.rg, expected_rg)
    np.testing.assert_array_equal(secondary.valid, expected_valid)


def test_minus_offsets_applies_reference_minus_secondary_convention() -> None:
    """Constant offsets subtract from the reference LUT coordinates."""
    coords = RadarCoordinates(
        np.array([[2.0, 3.0]], dtype=np.float64),
        np.array([[4.0, 5.0]], dtype=np.float64),
        np.ones((1, 2), dtype=bool),
    )
    offsets = _offsets(
        (8, 10),
        np.full((8, 10), 0.5),
        np.full((8, 10), -1.25),
    )
    secondary = coords.minus_offsets(offsets, device="cpu")
    np.testing.assert_allclose(secondary.az, [[1.5, 2.5]])
    np.testing.assert_allclose(secondary.rg, [[5.25, 6.25]])
    assert secondary.valid.all()


def test_minus_offsets_masks_out_of_window_and_uncovered_samples() -> None:
    """Samples outside coverage or the radar window become invalid."""
    coords = RadarCoordinates(
        np.array([[1.0, 15.5]], dtype=np.float64),
        np.array([[5.0, 5.0]], dtype=np.float64),
        np.ones((1, 2), dtype=bool),
    )
    coverage = np.ones((8, 10), dtype=bool)
    coverage[:, :3] = False
    offsets = _offsets((8, 10), np.zeros((8, 10)), np.full((8, 10), -1.0), coverage)
    secondary = coords.minus_offsets(offsets, device="cpu")
    # First sample: column 5 with coverage column 2 -> in window, covered.
    # Second sample: az 15.5 + 1.0 = 16.5 exceeds height - 1 -> invalid.
    assert secondary.valid[0, 0]
    assert not secondary.valid[0, 1]


def test_minus_offsets_torch_matches_numpy_oracle() -> None:
    """The Torch compose path matches the SciPy bilinear oracle."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    radar_shape = (16, 20)
    az = rng.uniform(1.0, 14.0, size=(8, 10))
    rg = rng.uniform(1.0, 18.0, size=(8, 10))
    coords = RadarCoordinates(az, rg, np.ones((8, 10), dtype=bool))
    offsets = _offsets(
        radar_shape,
        rng.normal(0.0, 0.4, size=radar_shape),
        rng.normal(0.0, 0.4, size=radar_shape),
    )

    numpy_coords = coords.minus_offsets(offsets, device="cpu")
    torch_coords = _minus_offsets_torch(coords, offsets, "cpu")

    assert numpy_coords.valid.shape == torch_coords.valid.shape
    common = numpy_coords.valid & torch_coords.valid
    assert common.any()
    np.testing.assert_allclose(
        numpy_coords.az[common],
        torch_coords.az[common],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        numpy_coords.rg[common],
        torch_coords.rg[common],
        atol=1e-5,
        rtol=0.0,
    )


def test_minus_offsets_rejects_unpublished_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devices outside the published cpu/cuda set fail closed."""
    pytest.importorskip("torch")
    import torch

    from faninsar.processing.geocoding import coordinates

    monkeypatch.setattr(
        coordinates,
        "resolve_torch_device",
        lambda _device: torch.device("mps"),
    )
    coords = _grid_coordinates(2, 2)
    with pytest.raises(InvalidProcessingStateError, match="cpu/cuda"):
        coords.minus_offsets(_offsets((4, 4), np.zeros((4, 4)), np.zeros((4, 4))))


def test_sample_complex_matches_direct_backend() -> None:
    """``kind='complex'`` delegates to the Lanczos complex backend."""
    rng = np.random.default_rng(3)
    source = (rng.normal(size=(16, 24)) + 1j * rng.normal(size=(16, 24))).astype(
        np.complex64,
    )
    coords = RadarCoordinates(
        rng.uniform(2.0, 13.0, size=(5, 7)),
        rng.uniform(2.0, 21.0, size=(5, 7)),
        np.ones((5, 7), dtype=bool),
    )
    sampled, valid = sample(source, coords, kind="complex")
    expected, expected_valid = resample_complex_at_coordinates(
        source,
        coords.az,
        coords.rg,
        valid=coords.valid,
    )
    np.testing.assert_array_equal(sampled, expected)
    np.testing.assert_array_equal(valid, expected_valid)


def test_sample_continuous_is_bilinear() -> None:
    """``kind='continuous'`` interpolates a linear field exactly."""
    row_indices, col_indices = np.indices((16, 24), dtype=np.float64)
    field = 3.0 * row_indices - 2.0 * col_indices + 7.0
    coords = RadarCoordinates(
        np.full((3, 4), 4.25),
        np.array([[2.5, 6.75, 11.125, 20.0]] * 3),
        np.ones((3, 4), dtype=bool),
    )
    sampled, valid = sample(field, coords, kind="continuous")
    assert valid.all()
    assert sampled.dtype == np.float32
    expected = 3.0 * coords.az - 2.0 * coords.rg + 7.0
    np.testing.assert_allclose(sampled, expected, atol=1e-4)


def test_sample_phase_has_no_wrap_artifact() -> None:
    """Wrapped phase crosses the +/-pi seam without the bilinear artifact."""
    rows, cols = 8, 16
    phase = np.where(
        np.arange(cols)[None, :] < cols // 2,
        3.1,
        -3.1,
    ).astype(np.float32) * np.ones((rows, cols), dtype=np.float32)
    coords = RadarCoordinates(
        np.full((1, 3), 4.0),
        np.array([[7.5, 7.0, 8.0]], dtype=np.float64),
        np.ones((1, 3), dtype=bool),
    )

    phase_values, phase_valid = sample(phase, coords, kind="phase")
    continuous_values, _ = sample(phase, coords, kind="continuous")

    assert phase_valid.all()
    assert phase_values.dtype == np.float32
    # Bilinear on phi averages +3.1 and -3.1 to ~0 across the seam.
    assert float(np.abs(continuous_values[0, 0])) < 0.5
    # exp(i*phi) averaging keeps the vector near +/-pi at the same sample.
    seam = phase_values[0, 0]
    wrapped_error = np.abs(np.angle(np.exp(1j * (seam - np.pi))))
    assert float(wrapped_error) < 1e-3
    # Integer coordinates away from the seam reproduce the exact phase.
    np.testing.assert_allclose(phase_values[0, 1], 3.1, atol=1e-6)
    np.testing.assert_allclose(phase_values[0, 2], -3.1, atol=1e-6)


def test_sample_phase_masks_non_finite_phase() -> None:
    """Non-finite source phase contributes zero magnitude to the average."""
    rows, cols = 8, 16
    phase = np.full((rows, cols), -3.1, dtype=np.float32)
    phase[:, cols // 2 :] = np.nan
    coords = RadarCoordinates(
        np.array([[4.0, 4.0]], dtype=np.float64),
        np.array([[7.0, 12.0]], dtype=np.float64),
        np.ones((1, 2), dtype=bool),
    )

    values, valid = sample(phase, coords, kind="phase")

    # Interior of the NaN region: no magnitude support stays invalid.
    assert valid[0, 0]
    assert not valid[0, 1]
    assert np.isnan(values[0, 1])


def test_sample_label_never_invents_categories() -> None:
    """Nearest sampling returns source categories (0 outside the window)."""
    rng = np.random.default_rng(11)
    labels = rng.integers(1, 4, size=(12, 18)).astype(np.int16)
    row_indices, col_indices = np.indices((12, 18), dtype=np.float64)
    integer_coords = RadarCoordinates(
        row_indices,
        col_indices,
        np.ones((12, 18), dtype=bool),
    )
    sampled, valid = sample(labels, integer_coords, kind="label")
    assert sampled.dtype == np.int16
    assert valid.all()
    np.testing.assert_array_equal(sampled, labels)

    fractional = RadarCoordinates(
        rng.uniform(0.0, 11.0, size=(5, 9)),
        np.concatenate(
            [rng.uniform(0.0, 17.0, size=(5, 7)), np.full((5, 2), 99.0)],
            axis=1,
        ),
        np.ones((5, 9), dtype=bool),
    )
    fractional_sampled, fractional_valid = sample(labels, fractional, kind="label")
    assert set(np.unique(fractional_sampled).tolist()) <= {0, 1, 2, 3}
    assert not fractional_valid[:, 7:].any()
    assert (fractional_sampled[~fractional_valid] == 0).all()

    bool_field = rng.random((12, 18)) > 0.5
    bool_sampled, bool_valid = sample(bool_field, integer_coords, kind="label")
    assert bool_sampled.dtype == np.bool_
    assert bool_valid.all()
    np.testing.assert_array_equal(bool_sampled, bool_field)


def test_sample_label_preserves_large_integer_labels() -> None:
    """Nearest sampling keeps integer labels exact beyond the float32 range.

    A ``float32`` intermediate rounds label magnitudes above ``2**24``; the
    kernel must sample the integer layer in its own dtype (exact through the
    ``float64`` integer range).
    """
    labels = np.array(
        [[2**24 + 1, 2**31 - 1], [5, 0]],
        dtype=np.int64,
    )
    row_indices, col_indices = np.indices(labels.shape, dtype=np.float64)
    integer_coords = RadarCoordinates(
        row_indices,
        col_indices,
        np.ones(labels.shape, dtype=bool),
    )
    sampled, valid = sample(labels, integer_coords, kind="label")
    assert sampled.dtype == np.int64
    assert valid.all()
    np.testing.assert_array_equal(sampled, labels)

    fractional = RadarCoordinates(
        np.array([[0.2, 0.7]]),
        np.array([[0.2, 0.4]]),
        np.ones((1, 2), dtype=bool),
    )
    fractional_sampled, fractional_valid = sample(labels, fractional, kind="label")
    assert fractional_valid.all()
    assert int(fractional_sampled[0, 0]) == 2**24 + 1
    assert int(fractional_sampled[0, 1]) == 5


def test_sample_valid_follows_coordinate_mask() -> None:
    """Invalid coordinate entries stay invalid and NaN-filled."""
    field = np.ones((8, 8), dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    valid[1, 1] = False
    coords = RadarCoordinates(
        np.array([[1.5, 1.5], [1.5, 1.5]]),
        np.array([[1.5, 1.5], [1.5, 99.0]]),
        valid,
    )
    values, sampled_valid = sample(field, coords, kind="continuous")
    assert not sampled_valid[1, 1]
    assert np.isnan(values[1, 1])
    assert sampled_valid[0, 0]
    assert sampled_valid[0, 1]
    assert sampled_valid[1, 0]


@pytest.mark.parametrize(
    ("layer_factory", "kind"),
    [
        (lambda: np.ones((8, 8), dtype=np.float32), "complex"),
        (lambda: np.ones((8, 8), dtype=np.complex64), "phase"),
        (lambda: np.ones((8, 8), dtype=np.complex64), "continuous"),
        (lambda: np.ones((8, 8), dtype=np.float32), "label"),
        (lambda: np.ones((8, 8), dtype=np.int16), "continuous"),
    ],
)
def test_sample_rejects_dtype_mismatches(
    layer_factory: object,
    kind: str,
) -> None:
    """Layers whose physical type does not match the kind are rejected."""
    coords = _grid_coordinates(4, 4)
    with pytest.raises(InvalidProcessingStateError):
        sample(layer_factory(), coords, kind=kind)  # type: ignore[arg-type]


def test_sample_rejects_unknown_kind() -> None:
    """Unknown kernel names are rejected."""
    coords = _grid_coordinates(2, 2)
    with pytest.raises(InvalidProcessingStateError):
        sample(np.ones((8, 8), dtype=np.float32), coords, kind="cubic")  # type: ignore[arg-type]


def test_sample_forwards_device_to_lanczos_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sample device reaches the accelerated Lanczos backend unchanged."""
    calls: dict[str, object] = {}

    def fake_resample(
        _layer: np.ndarray,
        az: np.ndarray,
        _rg: np.ndarray,
        *,
        valid: np.ndarray | None = None,
        executor: str = "torch",
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        calls["device"] = device
        calls["executor"] = executor
        values = np.full(az.shape, 1 + 0j, dtype=np.complex64)
        if valid is None:
            mask = np.ones(az.shape, dtype=bool)
        else:
            mask = np.asarray(valid, dtype=bool)
        return values, mask

    monkeypatch.setattr(
        "faninsar.processing.geocoding.geo_resample.resample_complex_at_coordinates",
        fake_resample,
    )
    coords = _grid_coordinates(3, 3)

    complex_values, complex_valid = sample(
        np.ones((8, 8), dtype=np.complex64),
        coords,
        kind="complex",
        device="cuda",
    )
    assert complex_values.shape == coords.shape
    assert complex_valid.all()
    phase_values, phase_valid = sample(
        np.ones((8, 8), dtype=np.float32),
        coords,
        kind="phase",
        device="cuda",
    )
    assert phase_values.shape == coords.shape
    assert phase_valid.all()
    assert calls == {"device": "cuda", "executor": "torch"}
