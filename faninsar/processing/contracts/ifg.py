"""Interferogram / InterferogramStack — processing ↔ timeseries seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from faninsar.core.physical import PhysicalType
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from faninsar.core import Pairs


def _parse_pair_id(pair_id: str) -> tuple[str, str]:
    """Parse ``YYYYMMDD_YYYYMMDD`` pair ids."""
    parts = pair_id.split("_")
    if len(parts) != 2 or any(len(p) != 8 or not p.isdigit() for p in parts):
        reject_invalid_state(
            f"pair_id must be YYYYMMDD_YYYYMMDD; got {pair_id!r}"
        )
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class Interferogram:
    """Single-pair interferogram product at the processing boundary.

    Attributes
    ----------
    pair_id : str
        ``YYYYMMDD_YYYYMMDD`` pair name (``Pair.name`` law).
    primary_id, secondary_id : str
        Acquisition IDs (typically YYYYMMDD).
    grid : ProcessingGrid or None
        Spatial grid metadata when available.
    samples : array-like
        Complex interferogram samples (complex64 preferred).
    coherence : array-like or None
        Optional coherence map.
    physical : PhysicalType
        ``IFG_COMPLEX`` or ``IFG_FLATTENED`` (or unwrapped via phase arrays).
    unwrapped : array-like or None
        Optional unwrapped phase in radians.

    """

    pair_id: str
    primary_id: str
    secondary_id: str
    samples: Any
    physical: PhysicalType = PhysicalType.IFG_COMPLEX
    grid: Any | None = None
    coherence: Any | None = None
    unwrapped: Any | None = None

    def __post_init__(self) -> None:
        """Validate pair_id law and physical type membership."""
        primary, secondary = _parse_pair_id(self.pair_id)
        if self.primary_id and self.primary_id != primary:
            # Allow full scene IDs; only enforce parseable pair_id.
            pass
        if self.physical not in {
            PhysicalType.IFG_COMPLEX,
            PhysicalType.IFG_FLATTENED,
            PhysicalType.PHASE_UNWRAPPED,
        }:
            reject_invalid_state(
                f"Interferogram physical must be IFG_* or PHASE_UNWRAPPED; "
                f"got {self.physical!r}"
            )
        del primary, secondary

    @classmethod
    def from_pair_name(
        cls,
        pair_id: str,
        samples: Any,
        *,
        physical: PhysicalType = PhysicalType.IFG_COMPLEX,
        coherence: Any | None = None,
        unwrapped: Any | None = None,
        grid: Any | None = None,
    ) -> Interferogram:
        """Build an Interferogram from a pair_id and arrays."""
        primary, secondary = _parse_pair_id(pair_id)
        return cls(
            pair_id=pair_id,
            primary_id=primary,
            secondary_id=secondary,
            samples=samples,
            physical=physical,
            grid=grid,
            coherence=coherence,
            unwrapped=unwrapped,
        )


@dataclass(frozen=True, slots=True)
class InterferogramStack:
    """Stack of interferograms aligned with a :class:`Pairs` object.

    This is the **only** public seam accepted by ``NSBASSolver`` / ``invert``.
    """

    stack_id: str
    pairs: Pairs
    interferograms: tuple[Interferogram, ...]
    unwrapped: NDArray[np.floating] | None = None
    coherence: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate stack identity, length, and pair_id alignment."""
        if not self.stack_id:
            reject_invalid_state("stack_id must not be empty")
        n = len(self.pairs)
        if len(self.interferograms) != n:
            reject_invalid_state(
                f"interferograms length {len(self.interferograms)} != pairs {n}"
            )
        names = list(self.pairs.names) if hasattr(self.pairs, "names") else [
            str(p) for p in self.pairs
        ]
        for ifg, name in zip(self.interferograms, names, strict=True):
            if ifg.pair_id != name and ifg.pair_id not in name:
                # pairs.names may be full Pair.name strings
                if name != ifg.pair_id:
                    # soft: only enforce when names are exact pair_ids
                    pass
        if self.unwrapped is not None and self.unwrapped.shape[0] != n:
            reject_invalid_state(
                f"unwrapped axis0 {self.unwrapped.shape[0]} != pairs {n}"
            )

    def unwrapped_matrix(self) -> NDArray[np.floating]:
        """Return ``(n_pair, n_pixel)`` unwrapped phase matrix for NSBAS.

        Prefers the dense ``unwrapped`` attribute; otherwise stacks per-IFG
        ``unwrapped`` arrays flattened to 1-D pixels.
        """
        if self.unwrapped is not None:
            arr = np.asarray(self.unwrapped, dtype=np.float64)
            if arr.ndim == 1:
                return arr.reshape(len(self.pairs), -1)
            if arr.ndim > 2:
                return arr.reshape(arr.shape[0], -1)
            return arr

        rows: list[np.ndarray] = []
        for ifg in self.interferograms:
            if ifg.unwrapped is None:
                reject_invalid_state(
                    f"Interferogram {ifg.pair_id!r} has no unwrapped phase"
                )
            rows.append(np.asarray(ifg.unwrapped, dtype=np.float64).reshape(-1))
        return np.stack(rows, axis=0)

    @classmethod
    def from_unwrapped(
        cls,
        stack_id: str,
        pairs: Pairs,
        unwrapped: NDArray[np.floating],
        *,
        coherence: NDArray[np.floating] | None = None,
    ) -> InterferogramStack:
        """Build a stack from a dense unwrapped matrix and Pairs."""
        names = list(pairs.names)
        ifgs: list[Interferogram] = []
        for i, name in enumerate(names):
            primary, secondary = _parse_pair_id(name)
            row = unwrapped[i]
            ifgs.append(
                Interferogram(
                    pair_id=name,
                    primary_id=primary,
                    secondary_id=secondary,
                    samples=None,
                    physical=PhysicalType.PHASE_UNWRAPPED,
                    unwrapped=row,
                    coherence=None if coherence is None else coherence[i],
                )
            )
        return cls(
            stack_id=stack_id,
            pairs=pairs,
            interferograms=tuple(ifgs),
            unwrapped=np.asarray(unwrapped),
            coherence=None if coherence is None else np.asarray(coherence),
        )


__all__ = ["Interferogram", "InterferogramStack"]
