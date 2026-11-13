"""Strict integrity baseline helper tests (PROPOSAL-0088 section 8)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from faninsar.io.storage._zarr_stac import write_zarr_array
from faninsar.io.storage.product import publish_product
from faninsar.processing.errors import InvalidProcessingStateError
from faninsar.stack.integrity import (
    BASELINE_SCHEMA,
    establish_integrity,
    verify_integrity,
)


def _zarr_product(
    root: Path,
    value: np.ndarray,
    *,
    overwrite: bool = False,
):
    """Publish one product whose only payload is a Zarr array asset."""

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        write_zarr_array(staging / "data.zarr", value)
        return {"data": {"href": "data.zarr"}}

    return publish_product(
        root,
        kind="integrity-test",
        properties={},
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


def _tiff_product(
    root: Path,
    value: np.ndarray,
    *,
    overwrite: bool = False,
):
    """Publish one product whose only payload is a real GeoTIFF array."""

    def write_payloads(staging: Path) -> dict[str, dict[str, str]]:
        path = staging / "raster.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=value.shape[1],
            height=value.shape[0],
            count=1,
            dtype=value.dtype,
            transform=from_origin(0.0, 0.0, 1.0, 1.0),
            crs="EPSG:4326",
        ) as dataset:
            dataset.write(value, 1)
        return {"raster": {"href": "raster.tif"}}

    return publish_product(
        root,
        kind="integrity-test",
        properties={},
        write_payloads=write_payloads,
        overwrite=overwrite,
    )


def _mutate_same_size(path: Path) -> None:
    """Flip one byte without changing the file length."""
    data = bytearray(path.read_bytes())
    assert data
    data[0] ^= 0xFF
    path.write_bytes(data)


def _baseline_payload(baseline) -> dict:
    return json.loads(baseline.baseline_path.read_text(encoding="utf-8"))


def test_strict_verifies_untouched_zarr_and_tiff_products(tmp_path: Path) -> None:
    root = tmp_path / "integrity"
    zarr_product = _zarr_product(tmp_path / "zarr-product", np.arange(9).reshape(3, 3))
    tiff_product = _tiff_product(
        tmp_path / "tiff-product", np.arange(6, dtype=np.float32).reshape(2, 3)
    )

    zarr_baseline = establish_integrity(
        root, zarr_product.path, write_id=zarr_product.write_id
    )
    tiff_baseline = establish_integrity(
        root, tiff_product.path, write_id=tiff_product.write_id
    )

    payload = _baseline_payload(zarr_baseline)
    assert payload["schema"] == BASELINE_SCHEMA
    assert payload["product"] == str(zarr_product.path.resolve())
    assert payload["write_id"] == zarr_product.write_id
    assert any(name.startswith("data.zarr/") for name in payload["files"])

    zarr_verdict = verify_integrity(root, zarr_product.path, strict=True)
    tiff_verdict = verify_integrity(root, tiff_product.path, strict=True)
    assert zarr_verdict.status == "verified"
    assert zarr_verdict.basis == "baseline"
    assert zarr_verdict.checked_files == len(zarr_baseline.files)
    assert tiff_verdict.status == "verified"
    assert tiff_verdict.checked_files == len(tiff_baseline.files) == 1


def test_absent_baseline_is_unverifiable_without_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))

    def unexpected_hash(path: Path) -> tuple[str, int]:
        raise AssertionError(f"fast verification hashed {path}")

    monkeypatch.setattr("faninsar.stack.integrity._hash_file", unexpected_hash)

    fast = verify_integrity(tmp_path / "integrity", product.path)
    strict = verify_integrity(tmp_path / "integrity", product.path, strict=True)

    assert fast.status == "integrity_unverifiable"
    assert strict.status == "integrity_unverifiable"
    assert fast.checked_files == strict.checked_files == 0


def test_fast_verify_does_not_reread_or_traverse_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.arange(4).reshape(2, 2))
    establish_integrity(root, product.path, write_id=product.write_id)

    def unexpected_hash(path: Path) -> tuple[str, int]:
        raise AssertionError(f"fast verification hashed {path}")

    def unexpected_walk(*args: object, **kwargs: object) -> object:
        raise AssertionError("fast verification traversed payloads")

    monkeypatch.setattr("faninsar.stack.integrity._hash_file", unexpected_hash)
    monkeypatch.setattr("faninsar.stack.integrity._walk_payload", unexpected_walk)

    verdict = verify_integrity(root, product.path)

    assert verdict.status == "metadata_verified"
    assert verdict.baseline_write_id == product.write_id
    assert verdict.checked_files == 0


def test_strict_detects_same_size_zarr_and_tiff_mutations(tmp_path: Path) -> None:
    root = tmp_path / "integrity"
    zarr_product = _zarr_product(tmp_path / "zarr-product", np.arange(9).reshape(3, 3))
    tiff_product = _tiff_product(
        tmp_path / "tiff-product", np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    establish_integrity(root, zarr_product.path, write_id=zarr_product.write_id)
    establish_integrity(root, tiff_product.path, write_id=tiff_product.write_id)

    zarr_file = next(
        path
        for path in sorted((zarr_product.path / "data.zarr").rglob("*"))
        if path.is_file()
    )
    _mutate_same_size(zarr_file)
    _mutate_same_size(tiff_product.path / "raster.tif")

    zarr_verdict = verify_integrity(root, zarr_product.path, strict=True)
    tiff_verdict = verify_integrity(root, tiff_product.path, strict=True)

    assert zarr_verdict.status == "mismatch"
    assert zarr_file.relative_to(zarr_product.path).as_posix() in (
        zarr_verdict.mismatched
    )
    assert tiff_verdict.status == "mismatch"
    assert tiff_verdict.mismatched == ("raster.tif",)


def test_corrupted_baseline_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))
    baseline = establish_integrity(root, product.path, write_id=product.write_id)

    baseline.baseline_path.write_text("{not json", encoding="utf-8")
    verdict = verify_integrity(root, product.path, strict=True)

    assert verdict.status == "invalid_baseline"
    assert "cannot be read" in verdict.detail
    assert verdict.checked_files == 0


def test_replacement_write_id_makes_the_baseline_stale(tmp_path: Path) -> None:
    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))
    establish_integrity(root, product.path, write_id=product.write_id)
    replacement = _zarr_product(
        tmp_path / "product", np.ones((2, 2), dtype=np.float32), overwrite=True
    )
    assert replacement.write_id != product.write_id

    verdict = verify_integrity(root, replacement.path, strict=True)

    assert verdict.status == "stale"
    assert verdict.write_id == replacement.write_id
    assert verdict.baseline_write_id == product.write_id
    assert verdict.checked_files == 0


def test_interrupted_baseline_publish_preserves_the_previous_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))
    baseline = establish_integrity(root, product.path, write_id=product.write_id)
    replacement = _zarr_product(
        tmp_path / "product", np.ones((2, 2), dtype=np.float32), overwrite=True
    )

    def interrupted_replace(source: Path, target: Path) -> None:
        raise OSError("simulated interrupted baseline publish")

    monkeypatch.setattr("faninsar.stack.integrity._replace_file", interrupted_replace)
    with pytest.raises(OSError, match="simulated"):
        establish_integrity(root, replacement.path, write_id=replacement.write_id)

    payload = _baseline_payload(baseline)
    assert payload["write_id"] == product.write_id
    assert sorted(path.name for path in root.iterdir()) == [baseline.baseline_path.name]

    verdict = verify_integrity(root, replacement.path, strict=True)
    assert verdict.status == "stale"


def test_same_write_id_cannot_rebaseline_mutated_bytes(tmp_path: Path) -> None:
    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.arange(9).reshape(3, 3))
    baseline = establish_integrity(root, product.path, write_id=product.write_id)
    original = _baseline_payload(baseline)
    zarr_file = next(
        path
        for path in sorted((product.path / "data.zarr").rglob("*"))
        if path.is_file()
    )
    _mutate_same_size(zarr_file)

    with pytest.raises(InvalidProcessingStateError, match="already exists"):
        establish_integrity(root, product.path, write_id=product.write_id)

    assert _baseline_payload(baseline) == original
    assert verify_integrity(root, product.path, strict=True).status == "mismatch"


def test_raw_payload_baseline_excludes_its_own_file(tmp_path: Path) -> None:
    zarr_root = tmp_path / "raw.zarr"
    write_zarr_array(zarr_root, np.arange(4).reshape(2, 2))
    write_id = "01HZZZZZZZZZZZZZZZZZZZZZZZ"

    baseline = establish_integrity(zarr_root, zarr_root, write_id=write_id)
    payload = _baseline_payload(baseline)

    assert baseline.baseline_path.parent == zarr_root.resolve()
    assert baseline.baseline_path.name not in payload["files"]
    assert verify_integrity(
        zarr_root, zarr_root, strict=True, write_id=write_id
    ).status == ("verified")


def test_raw_payload_requires_an_explicit_write_id(tmp_path: Path) -> None:
    payload = tmp_path / "payload.tif"
    payload.write_bytes(b"independent tiff bytes")
    write_id = "01HZZZZZZZZZZZZZZZZZZZZZZZ"
    root = tmp_path / "integrity"

    establish_integrity(root, payload, write_id=write_id)
    assert verify_integrity(root, payload, strict=True, write_id=write_id).status == (
        "verified"
    )
    with pytest.raises(InvalidProcessingStateError, match="requires"):
        verify_integrity(root, payload, strict=True)


def test_provider_digest_verifies_without_a_self_baseline(tmp_path: Path) -> None:
    payload = tmp_path / "payload.tif"
    payload.write_bytes(b"provider-verifiable tiff bytes")
    write_id = "01HZZZZZZZZZZZZZZZZZZZZZZZ"
    root = tmp_path / "integrity"
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    verdict = verify_integrity(
        root,
        payload,
        strict=True,
        write_id=write_id,
        provider_digest=f"sha256:{digest}",
    )

    assert verdict.status == "verified"
    assert verdict.basis == "provider"
    assert verdict.checked_files == 1
    assert not list(root.glob("baseline-*.json"))

    wrong = verify_integrity(
        root,
        payload,
        strict=True,
        write_id=write_id,
        provider_digest="sha256:" + "0" * 64,
    )
    assert wrong.status == "mismatch"
    assert wrong.basis == "provider"


def test_provider_digest_is_not_a_multi_file_manifest_hash(tmp_path: Path) -> None:
    product = _zarr_product(tmp_path / "product", np.arange(9).reshape(3, 3))

    with pytest.raises(ValueError, match="single-file"):
        verify_integrity(
            tmp_path / "integrity",
            product.path,
            strict=True,
            provider_digest="sha256:" + "0" * 64,
        )


def test_strict_verify_rejects_a_product_replaced_during_the_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from faninsar.stack import integrity as integrity_module

    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))
    establish_integrity(root, product.path, write_id=product.write_id)
    original_fingerprint = integrity_module._fingerprint

    def replacing_fingerprint(files):
        result = original_fingerprint(files)
        _zarr_product(
            tmp_path / "product", np.ones((2, 2), dtype=np.float32), overwrite=True
        )
        return result

    monkeypatch.setattr("faninsar.stack.integrity._fingerprint", replacing_fingerprint)

    verdict = verify_integrity(root, product.path, strict=True)

    assert verdict.status == "stale"
    assert "changed" in verdict.detail


def test_establish_rejects_a_product_replaced_during_the_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from faninsar.stack import integrity as integrity_module

    root = tmp_path / "integrity"
    product = _zarr_product(tmp_path / "product", np.zeros((2, 2), dtype=np.float32))
    original_fingerprint = integrity_module._fingerprint

    def replacing_fingerprint(files):
        result = original_fingerprint(files)
        _zarr_product(
            tmp_path / "product", np.ones((2, 2), dtype=np.float32), overwrite=True
        )
        return result

    monkeypatch.setattr("faninsar.stack.integrity._fingerprint", replacing_fingerprint)

    with pytest.raises(InvalidProcessingStateError, match="changed"):
        establish_integrity(root, product.path, write_id=product.write_id)
    assert not list(root.glob("baseline-*.json"))


def test_unsafe_asset_href_is_rejected(tmp_path: Path) -> None:
    product = tmp_path / "product"
    product.mkdir()
    (product / "item.json").write_text(
        json.dumps(
            {
                "properties": {
                    "faninsar:kind": "integrity-test",
                    "faninsar:write_id": "01HZZZZZZZZZZZZZZZZZZZZZZZ",
                },
                "assets": {"escape": {"href": "../escape.zarr"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidProcessingStateError, match="basename"):
        establish_integrity(
            tmp_path / "integrity",
            product,
            write_id="01HZZZZZZZZZZZZZZZZZZZZZZZ",
        )
