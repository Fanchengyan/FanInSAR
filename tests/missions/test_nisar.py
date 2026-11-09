"""Focused tests for the optional NISAR RSLC reader MVP (PROPOSAL-0035)."""

# ruff: noqa: INP001, RUF043

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from faninsar.missions.nisar import (
    NisarAdmissionPolicy,
    NisarSensor,
    admit_nisar_source,
)
from faninsar.processing.errors import InvalidProcessingStateError


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
    readers.open_product = lambda uri, **_kwargs: calls.append(uri) or handle  # type: ignore[attr-defined]
    products = ModuleType("nisar.products")
    nisar = ModuleType("nisar")
    monkeypatch.setitem(sys.modules, "nisar", nisar)
    monkeypatch.setitem(sys.modules, "nisar.products", products)
    monkeypatch.setitem(sys.modules, "nisar.products.readers", readers)


def _spy_source_reads(monkeypatch: pytest.MonkeyPatch, source: Path) -> list[str]:
    """Record read-binary opens of ``source`` used for full-file hashing."""
    target = source.expanduser().resolve(strict=False)
    opens: list[str] = []
    real_open = Path.open

    def wrapped(
        self: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        path = Path(self).expanduser().resolve(strict=False)
        text = str(mode)
        if path == target and "b" in text and "r" in text and "w" not in text:
            opens.append(text)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", wrapped)
    return opens


def admit_fake_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handle: object,
) -> object:
    """Attach optional source-identity metadata to a reader fake."""
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


def test_nisar_admission_does_not_reject_content_mutation_before_window_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-path source byte edits are no longer a processing reject.

    PROPOSAL-0082 withdraws full-file snapshot drift checks. Local inputs are
    treated as stable for the run; a matching inventory digest is metadata,
    not an open/read gate.
    """
    source = tmp_path / "scene.h5"
    source.write_bytes(b"rslc-v1")
    samples = np.ones((4, 4), dtype=np.complex64)
    dataset = FakeComplexDataset(samples)
    handle = SimpleNamespace(
        filename=str(source),
        getSlcDatasetAsNativeComplex=lambda *_args: dataset,
    )
    install_fake_reader(monkeypatch, handle, [])
    admitted = NisarSensor().open_product(
        str(source),
        admission={
            "trusted_roots": [tmp_path],
            "expected_sha256": {str(source): hashlib.sha256(b"rslc-v1").hexdigest()},
        },
    )
    source.write_bytes(b"rslc-v2")

    result = NisarSensor().read_slc_window(admitted, (slice(0, 2), slice(0, 2)))
    np.testing.assert_array_equal(result, samples[:2, :2])
    assert dataset.selections == [(slice(0, 2), slice(0, 2))]


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


def test_repeated_window_reads_do_not_hash_entire_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public adapter window reads never scan the whole RSLC for identity."""
    source = tmp_path / "scene.h5"
    source.write_bytes(b"rslc-payload" * 4096)
    samples = (
        np.arange(48, dtype=np.float32).reshape(6, 8)
        + 1j * np.arange(48, dtype=np.float32).reshape(6, 8)
    ).astype(np.complex64)
    dataset = FakeComplexDataset(samples)
    handle = SimpleNamespace(
        filename=str(source),
        getSlcDatasetAsNativeComplex=lambda *_args: dataset,
    )
    install_fake_reader(monkeypatch, handle, [])
    opens = _spy_source_reads(monkeypatch, source)
    digest = hashlib.sha256(b"declared-inventory").hexdigest()
    sensor = NisarSensor()
    admitted = sensor.open_product(
        str(source),
        admission={
            "trusted_roots": [tmp_path],
            "expected_sha256": {str(source): digest},
        },
    )
    first = sensor.read_slc_window(admitted, (slice(1, 4), slice(2, 6)))
    second = sensor.read_slc_window(admitted, (slice(1, 4), slice(2, 6)))

    np.testing.assert_array_equal(first, samples[1:4, 2:6])
    np.testing.assert_array_equal(second, first)
    assert dataset.selections == [
        (slice(1, 4), slice(2, 6)),
        (slice(1, 4), slice(2, 6)),
    ]
    assert opens == []
    metadata = sensor.admission_metadata(admitted)
    assert metadata is not None
    assert metadata["source_id"] == str(source.resolve())
    assert "source_digest" not in metadata
    assert metadata["policy"]["expected_sha256"][str(source)] == digest


def test_admit_nisar_source_records_inventory_without_hashing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """admit_nisar_source keeps expected SHA-256 inventory as metadata only."""
    source = tmp_path / "scene.h5"
    payload = b"rslc-v1" * 2048
    source.write_bytes(payload)
    digest = hashlib.sha256(b"not-the-file-bytes").hexdigest()
    opens = _spy_source_reads(monkeypatch, source)

    record = admit_nisar_source(
        source,
        expected_sha256={str(source): digest},
        trusted_roots=[tmp_path],
    )

    assert record["source_id"] == str(source.resolve())
    assert "source_digest" not in record
    assert "source_size_bytes" not in record
    assert record["policy"]["expected_sha256"][str(source)] == digest
    assert opens == []


def test_nisar_expected_sha256_inventory_rejects_malformed_digest() -> None:
    """Inventory metadata must stay well-formed even though it is not enforced."""
    with pytest.raises(InvalidProcessingStateError, match="64 hex digits"):
        NisarAdmissionPolicy(expected_sha256={"scene.h5": "abcd"}).as_metadata()
