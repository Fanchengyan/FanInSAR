"""Focused tests for the optional NISAR RSLC reader MVP (PROPOSAL-0035)."""

# ruff: noqa: INP001, RUF043

from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import faninsar.missions.nisar as nisar_module
from faninsar.missions.nisar import (
    NisarAdmissionPolicy,
    NisarSensor,
    admit_nisar_source,
)
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path


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


def admit_fake_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handle: object,
) -> object:
    """Attach an explicit trusted admission snapshot to a reader fake."""
    source = tmp_path / f"fake-rslc-{id(handle)}.h5"
    source.write_bytes(b"reader-fake")
    if getattr(handle, "filename", None) is None:
        handle.filename = str(source)
    install_fake_reader(monkeypatch, handle, [])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return NisarSensor().open_product(
        str(source),
        admission={
            "trusted_roots": [tmp_path],
            "expected_sha256": {str(source): digest},
        },
    )


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the optional factory only when an RSLC is explicitly opened."""
    calls: list[str] = []
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: object())
    install_fake_reader(monkeypatch, handle, calls)

    source = tmp_path / "scene.h5"
    source.write_bytes(b"reader-fake")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = NisarSensor().open_product(
        str(source),
        admission={
            "trusted_roots": [tmp_path],
            "expected_sha256": {str(source): digest},
        },
    )

    assert result is handle
    assert calls == [str(source)]


def test_nisar_window_read_defaults_to_b_hh_and_stays_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    admitted = admit_fake_handle(tmp_path, monkeypatch, handle)
    result = NisarSensor().read_slc_window(admitted, (slice(1, 4), slice(2, 6)))

    np.testing.assert_array_equal(result, samples[1:4, 2:6])
    assert calls == [("B", "HH")]
    assert dataset.selections == [(slice(1, 4), slice(2, 6))]


def test_nisar_window_read_normalizes_channel_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept compatibility aliases while passing normalized identifiers."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.complex64))
    calls: list[tuple[str, str]] = []
    handle = SimpleNamespace(
        getSlcDatasetAsNativeComplex=lambda frequency, polarization: (
            calls.append((frequency, polarization)) or dataset
        )
    )

    admitted = admit_fake_handle(tmp_path, monkeypatch, handle)
    NisarSensor().read_slc_window(
        admitted,
        (slice(0, 2), slice(0, 2)),
        freq=" b ",
        pol=" hh ",
    )

    assert calls == [("B", "HH")]


def test_nisar_window_read_rejects_real_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a reader response that loses native complex sample semantics."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.float32))
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: dataset)

    admitted = admit_fake_handle(tmp_path, monkeypatch, handle)
    with pytest.raises(ValueError, match="must retain complex"):
        NisarSensor().read_slc_window(admitted, (slice(0, 2), slice(0, 2)))


def test_nisar_window_read_rejects_out_of_bounds_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject out-of-bounds native selections before accessing the dataset."""
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.complex64))
    handle = SimpleNamespace(getSlcDatasetAsNativeComplex=lambda *_args: dataset)

    admitted = admit_fake_handle(tmp_path, monkeypatch, handle)
    with pytest.raises(ValueError, match="selection out of bounds"):
        NisarSensor().read_slc_window(admitted, (slice(0, 5), slice(0, 2)))

    assert dataset.selections == []


def test_nisar_admission_rejects_content_mutation_before_window_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted handle cannot read bytes from a changed RSLC source."""
    source = tmp_path / "scene.h5"
    source.write_bytes(b"rslc-v1")
    dataset = FakeComplexDataset(np.ones((4, 4), dtype=np.complex64))
    handle = SimpleNamespace(
        filename=str(source),
        getSlcDatasetAsNativeComplex=lambda *_args: dataset,
    )
    calls: list[str] = []
    install_fake_reader(monkeypatch, handle, calls)
    admitted = NisarSensor().open_product(
        str(source),
        admission={
            "trusted_roots": [tmp_path],
            "expected_sha256": {str(source): hashlib.sha256(b"rslc-v1").hexdigest()},
        },
    )
    source.write_bytes(b"rslc-v2")

    with pytest.raises(InvalidProcessingStateError, match="content changed"):
        NisarSensor().read_slc_window(admitted, (slice(0, 2), slice(0, 2)))
    assert dataset.selections == []


@pytest.mark.parametrize(
    ("frequency", "polarization", "match"),
    [
        ("", "HH", "non-empty string"),
        ("B", "", "non-empty string"),
        (None, "HH", "non-empty string"),
        ("B", None, "non-empty string"),
        ("A", "VV", "frequency .* unavailable"),
        ("B", "VV", "polarization .* unavailable"),
    ],
)
def test_nisar_product_rejects_invalid_or_unavailable_channel_before_read(
    frequency: str | None,
    polarization: str | None,
    match: str,
) -> None:
    """Channel admission fails before the reader can select a dataset."""
    calls: list[tuple[str, str]] = []
    handle = SimpleNamespace(
        frequencies=("B",),
        polarizations={"B": ("HH",)},
        getSlcDatasetAsNativeComplex=lambda freq, pol: (
            calls.append((freq, pol)) or object()
        ),
    )

    with pytest.raises(ValueError, match=match):
        NisarSensor().to_slc_product(
            handle,
            frequency=frequency,
            polarization=polarization,
        )

    assert calls == []


def test_nisar_product_defaults_to_explicit_b_hh_and_fails_closed() -> None:
    """Omitted channel arguments mean B/HH, not an available-channel fallback."""
    handle = SimpleNamespace(
        frequencies=("A",),
        polarizations={"A": ("VV",)},
        getSlcDatasetAsNativeComplex=lambda *_args: object(),
    )

    with pytest.raises(ValueError, match="frequency 'B' is unavailable"):
        NisarSensor().to_slc_product(handle)


def test_nisar_product_rejects_conflicting_channel_aliases() -> None:
    """Compatibility aliases cannot override an explicit channel request."""
    handle = SimpleNamespace(frequencies=("B",), polarizations={"B": ("HH",)})

    with pytest.raises(ValueError, match="different channels"):
        NisarSensor().to_slc_product(handle, frequency="A", freq="B")

