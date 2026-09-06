"""Processing seam for stacks of network interferogram products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from faninsar.network.products import Interferogram
from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from faninsar.core.pair import Pairs


@dataclass(frozen=True, slots=True)
class InterferogramStack:
    """Stack of interferograms aligned with a :class:`~faninsar.core.pair.Pairs`.

    This is the only public processing seam accepted by the time-series
    solvers. Each member retains its exact Pair lineage through the product.
    """

    stack_id: str
    pairs: Pairs
    interferograms: tuple[Interferogram, ...]
    unwrapped: NDArray[np.floating] | None = None
    coherence: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate stack identity, length, and pair alignment."""
        if not self.stack_id:
            reject_invalid_state("stack_id must not be empty")
        n = len(self.pairs)
        if len(self.interferograms) != n:
            reject_invalid_state(
                f"interferograms length {len(self.interferograms)} != pairs {n}"
            )
        names = list(self.pairs.names)
        for ifg, name in zip(self.interferograms, names, strict=True):
            if not isinstance(ifg, Interferogram):
                reject_invalid_state("interferograms must contain Interferogram values")
            if ifg.pair_id != name:
                reject_invalid_state(
                    f"interferogram {ifg.pair_id!r} does not match pair {name!r}"
                )
        if self.unwrapped is not None and self.unwrapped.shape[0] != n:
            reject_invalid_state(
                f"unwrapped axis0 {self.unwrapped.shape[0]} != pairs {n}"
            )

    def unwrapped_matrix(self) -> NDArray[np.floating]:
        """Return an ``(n_pair, n_pixel)`` unwrapped phase matrix."""
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
        """Build a stack from a dense unwrapped matrix and ``Pairs``."""
        values = np.asarray(unwrapped)
        if values.ndim == 0 or values.shape[0] != len(pairs):
            axis_size = values.shape[0] if values.ndim else 0
            message = f"unwrapped axis0 {axis_size} != pairs {len(pairs)}"
            reject_invalid_state(message)
        coherence_values = None if coherence is None else np.asarray(coherence)
        if coherence_values is not None and coherence_values.shape[0] != len(pairs):
            reject_invalid_state(
                f"coherence axis0 {coherence_values.shape[0]} != pairs {len(pairs)}"
            )
        ifgs = tuple(
            Interferogram(
                pair=pair,
                unwrapped_phase=values[i],
                coherence=None if coherence_values is None else coherence_values[i],
            )
            for i, pair in enumerate(pairs)
        )
        return cls(
            stack_id=stack_id,
            pairs=pairs,
            interferograms=ifgs,
            unwrapped=values,
            coherence=coherence_values,
        )


__all__ = ["InterferogramStack"]
