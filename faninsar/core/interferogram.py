# ruff: noqa: EM101, TRY003
"""Scientific interferogram result values and pair lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from faninsar.core.pairs import Pair
from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class Interferogram:
    """Immutable result bundle for one existing :class:`~faninsar.core.Pair`.

    Parameters
    ----------
    pair : Pair
        Lossless pair object retained as the result lineage.
    complex_phase, wrapped_phase, coherence, unwrapped_phase : array-like, optional
        Optional scientific representations. Arrays are converted to NumPy views.
    masks : mapping[str, array-like], optional
        Named validity or quality masks.
    metadata : mapping[str, object], optional
        Representation metadata retained as a read-only mapping.

    Notes
    -----
    This additive value does not change the existing date-oriented ``Pair``
    constructor. A later migration adapter can supply physical acquisition
    lineage without requiring a second result envelope.

    """

    pair: Pair
    complex_phase: NDArray | None = None
    wrapped_phase: NDArray | None = None
    coherence: NDArray | None = None
    unwrapped_phase: NDArray | None = None
    masks: Mapping[str, NDArray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate lineage and normalize optional array representations."""
        if not isinstance(self.pair, Pair):
            logger.error("Interferogram lineage must be a Pair")
            raise TypeError("pair must be a Pair")
        for field_name in (
            "complex_phase",
            "wrapped_phase",
            "coherence",
            "unwrapped_phase",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, np.asarray(value))
        if not isinstance(self.masks, Mapping):
            logger.error("Interferogram masks must be a mapping")
            raise TypeError("masks must be a mapping")
        object.__setattr__(
            self,
            "masks",
            MappingProxyType(
                {str(name): np.asarray(mask) for name, mask in self.masks.items()}
            ),
        )
        if not isinstance(self.metadata, Mapping):
            logger.error("Interferogram metadata must be a mapping")
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def lineage(self) -> Pair:
        """Return the exact pair object that produced this result."""
        return self.pair

    @property
    def complex_interferogram(self) -> NDArray | None:
        """Return the complex representation under its scientific alias."""
        return self.complex_phase

    @property
    def complex_ifg(self) -> NDArray | None:
        """Return the complex representation under its legacy alias."""
        return self.complex_phase


__all__ = ["Interferogram"]
