"""Focused tests for the optional NISAR RSLC reader MVP (PROPOSAL-0035)."""

# ruff: noqa: INP001, RUF043

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from faninsar.missions.nisar import NisarSensor


class FakeComplexDataset:
    """Array-backed fake that records only requested dataset selections."""

    def __init__(self, samples: np.ndarray) -> None:
        """Store source samples and expose a reader-like shape."""
        self._samples = samples
        self.shape = samples.shape
        self.selections: list[tuple[slice, slice]] = []

    def __getitem__(self, selection: tuple[slice, slice]) -> np.ndarray:
        """Record and return only one native array selection."""
        self.selections.append(selection)
        return self._samples[selection]


def install_fake_reader(
    monkeypatch: pytest.MonkeyPatch, handle: object, calls: list[str]
) -> None:
    """Install an in-memory ``nisar.products.readers`` module for one test."""
    readers = ModuleType("nisar.products.readers")
    readers.open_product = lambda uri, **_kwargs: (calls.append(uri) or handle)  # type: ignore[attr-defined]
    products = ModuleType("nisar.products")
    nisar = ModuleType("nisar")
    monkeypatch.setitem(sys.modules, "nisar", nisar)
    monkeypatch.setitem(sys.modules, "nisar.products", products)
    monkeypatch.setitem(sys.modules, "nisar.products.readers", readers)


def test_nisar_sensor_instantiation() -> None:
    """Register and instantiate the NISAR sensor adapter."""
    assert NisarSensor().name == "nisar"


def test_nisar_open_product_reports_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail only at open time if the optional NISAR reader is absent."""
    for name in ("nisar", "nisar.products", "nisar.products.readers"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(ImportError, match="optional nisar.products.readers"):
        NisarSensor().open_product("/tmp/scene.h5")


def test_nisar_open_product_uses_optional_reader_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the optional factory only when an RSLC is explicitly opened."""
    calls: list[str] = []
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: object())
    install_fake_reader(monkeypatch, handle, calls)

    result = NisarSensor().open_product("/tmp/scene.h5")

    assert result is handle
    assert calls == ["/tmp/scene.h5"]


def test_nisar_window_read_defaults_to_b_hh_and_stays_lazy() -> None:
    """Read precisely one B/HH selection instead of materializing the dataset."""
    samples = (
        np.arange(48, dtype=np.float32).reshape(6, 8)
        + 1j * np.arange(48, dtype=np.float32).reshape(6, 8)
    ).astype(np.complex64)
    dataset = FakeComplexDataset(samples)
    calls: list[tuple[str, str]] = []
    handle = SimpleNamespace(
        getSlcDatasetAsNativeComplex=lambda frequency, polarization: (
            calls.append((frequency, polarization)) or dataset
        )
    )

    result = NisarSensor().read_slc_window(handle, (slice(1, 4), slice(2, 6)))

    np.testing.assert_array_equal(result, samples[1:4, 2:6])
    assert calls == [("B", "HH")]
    assert dataset.selections == [(slice(1, 4), slice(2, 6))]


def test_nisar_window_read_normalizes_channel_aliases() -> None:
    """Accept compatibility aliases while passing normalized identifiers."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.complex64))
    calls: list[tuple[str, str]] = []
    handle = SimpleNamespace(
        getSlcDatasetAsNativeComplex=lambda frequency, polarization: (
            calls.append((frequency, polarization)) or dataset
        )
    )

    NisarSensor().read_slc_window(
        handle,
        (slice(0, 2), slice(0, 2)),
        freq=" b ",
        pol=" hh ",
    )

    assert calls == [("B", "HH")]


def test_nisar_window_read_rejects_real_samples() -> None:
    """Reject a reader response that loses native complex sample semantics."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.float32))
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: dataset)

    with pytest.raises(ValueError, match="must retain complex"):
        NisarSensor().read_slc_window(handle, (slice(0, 2), slice(0, 2)))


def test_nisar_window_read_rejects_out_of_bounds_window() -> None:
    """Reject out-of-bounds native selections before accessing the dataset."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.complex64))
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: dataset)

    with pytest.raises(ValueError, match="selection out of bounds"):
        NisarSensor().read_slc_window(handle, (slice(0, 5), slice(0, 2)))

    assert dataset.selections == []
