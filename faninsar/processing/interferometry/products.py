"""Interferogram product values and persistence writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.logging import setup_logger
from faninsar.processing.errors import reject_grid_mismatch, reject_invalid_state
from faninsar.processing.geometry.coordinates import (
    ArrayDescriptor,
    ArrayRepresentation,
    ProcessingGrid,
)

if TYPE_CHECKING:
    from faninsar.processing.slc.products import SLCProduct

logger = setup_logger(__name__)


class CarrierState(StrEnum):
    """TOPS carrier handling state."""

    PRESENT = "present"
    DERAMPED = "deramped"
    RESTORED = "restored"


class CoregistrationState(StrEnum):
    """Registration state relative to the stack reference."""

    NOT_REGISTERED = "not_registered"
    REGISTERED = "registered"


class FlatteningState(StrEnum):
    """Interferometric reference-phase removal state."""

    NOT_APPLIED = "not_applied"
    APPLIED = "applied"


class CalibrationState(StrEnum):
    """Radiometric calibration state."""

    RAW_DN = "raw_dn"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True)
class ComplexInterferogram:
    """Metadata-only complex interferogram (URI/descriptor contracts).

    .. deprecated::
        Prefer the canonical Network interferogram products as the
        processing ↔ timeseries seam for array-bearing products.  This type
        remains for storage/STAC descriptor contracts until PairProduct is
        fully migrated.
    """

    pair_id: str
    primary_id: str
    secondary_id: str
    grid: ProcessingGrid
    samples: ArrayDescriptor
    flattening: FlatteningState

    @classmethod
    def form(
        cls,
        primary: SLCProduct,
        secondary: SLCProduct,
        *,
        uri: str | None = None,
    ) -> ComplexInterferogram:
        """Create metadata for an interferogram after validating its inputs."""
        if (
            primary.coregistration is not CoregistrationState.REGISTERED
            or secondary.coregistration is not CoregistrationState.REGISTERED
        ):
            reject_invalid_state("interferogram inputs must both be registered")
        if primary.grid != secondary.grid:
            reject_grid_mismatch("interferogram inputs must use the same grid")
        pair_id = f"{primary.acquisition_id}_{secondary.acquisition_id}"
        return cls(
            pair_id=pair_id,
            primary_id=primary.acquisition_id,
            secondary_id=secondary.acquisition_id,
            grid=primary.grid,
            samples=ArrayDescriptor(
                uri=uri or f"memory://{pair_id}/complex",
                shape=primary.grid.shape,
                dtype="complex64",
                representation=ArrayRepresentation.COMPLEX,
            ),
            flattening=FlatteningState.NOT_APPLIED,
        )

    def __post_init__(self) -> None:
        """Validate complex representation and grid identity."""
        if self.samples.representation is not ArrayRepresentation.COMPLEX:
            reject_invalid_state("interferogram samples must remain complex")
        if self.samples.shape != self.grid.shape:
            reject_grid_mismatch("interferogram samples must match their grid")


@dataclass(frozen=True, slots=True)
class UnwrapResult:
    """Unwrapped phase and algorithm identity for one pair."""

    pair_id: str
    grid: ProcessingGrid
    phase: ArrayDescriptor
    method: str

    def __post_init__(self) -> None:
        """Validate unwrapped phase representation and grid identity."""
        if self.phase.representation is not ArrayRepresentation.PHASE:
            reject_invalid_state("unwrap output must be a phase array")
        if self.phase.shape != self.grid.shape:
            reject_grid_mismatch("unwrap phase must match its grid")
        if not self.method:
            reject_invalid_state("unwrap method must not be empty")


@dataclass(frozen=True, slots=True)
class PairProduct:
    """Interferogram and optional unwrapped result for one acquisition pair."""

    interferogram: ComplexInterferogram
    unwrap: UnwrapResult | None = None

    def __post_init__(self) -> None:
        """Validate that unwrapping belongs to the same pair and grid."""
        if self.unwrap is not None and (
            self.unwrap.pair_id != self.interferogram.pair_id
            or self.unwrap.grid != self.interferogram.grid
        ):
            reject_invalid_state("unwrap result must match its interferogram")


@dataclass(frozen=True, slots=True)
class PairProductArrays:
    """In-memory pair product layers ready for persistence."""

    pair_id: str
    complex_ifg: np.ndarray
    coherence: np.ndarray
    wrapped_phase: np.ndarray
    unwrapped_phase: np.ndarray
    connected_components: np.ndarray
    metadata: dict[str, Any]


def write_pair_zarr(
    product: PairProductArrays,
    store_path: str | Path,
) -> Path:
    """Write pair product arrays to a Zarr store.

    Parameters
    ----------
    product : PairProductArrays
        In-memory pair layers.
    store_path : str or pathlib.Path
        Destination Zarr directory.

    Returns
    -------
    pathlib.Path
        Path to the written store.

    """
    import zarr

    path = Path(store_path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    layers = {
        "complex_ifg": product.complex_ifg,
        "coherence": product.coherence,
        "wrapped_phase": product.wrapped_phase,
        "unwrapped_phase": product.unwrapped_phase,
        "connected_components": product.connected_components,
    }
    for name, array in layers.items():
        root.create_array(name, data=np.asarray(array), overwrite=True)
    root.attrs.update(
        {
            "pair_id": product.pair_id,
            "created_utc": datetime.now(UTC).isoformat(),
            **{str(k): str(v) for k, v in product.metadata.items()},
        }
    )
    logger.info("Wrote pair Zarr product: %s", path)
    return path


def write_pair_stac_item(
    product: PairProductArrays,
    zarr_path: str | Path,
    item_path: str | Path,
    stac_item_id: str | None = None,
) -> Path:
    """Write a minimal STAC item describing the pair Zarr product.

    When the pair metadata carries a mask product (``mask_href`` plus the
    ``mask_product``/``mask_provider``/``mask_retrieved``/``mask_threshold``/
    ``mask_buffer_km`` provenance keys, PROPOSAL-0039), the buffered binary
    mask is registered as a ``mask`` STAC asset next to the Zarr store and
    the provenance keys surface as ``faninsar:mask_*`` item properties.

    Parameters
    ----------
    product : PairProductArrays
        Pair product metadata and arrays.
    zarr_path : str or pathlib.Path
        Path to the Zarr store asset.
    item_path : str or pathlib.Path
        Output STAC item JSON path.
    stac_item_id : str, optional
        Look-qualified item id override for sweep outputs.

    Returns
    -------
    pathlib.Path
        Path to the written STAC item.

    """
    zarr_path = Path(zarr_path)
    item_path = Path(item_path)
    item_id = stac_item_id if stac_item_id is not None else product.pair_id
    if not item_id:
        reject_invalid_state("pair_id is required for STAC item identity")
    height, width = product.wrapped_phase.shape
    assets: dict[str, dict[str, Any]] = {
        "zarr": {
            "href": str(zarr_path),
            "type": "application/vnd+zarr",
            "roles": ["data"],
            "title": "Pair product Zarr store",
        }
    }
    mask_href = product.metadata.get("mask_href")
    if mask_href:
        assets["mask"] = {
            "href": str(mask_href),
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "roles": ["mask", "data"],
            "title": "Buffered mask (1 = water/removed)",
        }
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": None,
        "bbox": None,
        "properties": {
            "datetime": datetime.now(UTC).isoformat(),
            "faninsar:pair_id": product.pair_id,
            "faninsar:shape": [height, width],
            "faninsar:unwrap_method": product.metadata.get("unwrap_method", "snaphu"),
            **{f"faninsar:{k}": v for k, v in product.metadata.items()},
        },
        "assets": assets,
        "links": [],
        "stac_extensions": [],
    }
    item_path.parent.mkdir(parents=True, exist_ok=True)
    item_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote pair STAC item: %s", item_path)
    return item_path


__all__ = [
    "CalibrationState",
    "CarrierState",
    "ComplexInterferogram",
    "CoregistrationState",
    "FlatteningState",
    "PairProduct",
    "PairProductArrays",
    "UnwrapResult",
    "write_pair_stac_item",
    "write_pair_zarr",
]
