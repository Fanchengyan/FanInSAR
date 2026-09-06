"""Network-facing views over committed Stack interferogram products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from faninsar.network.readers.interferogram import (
    InterferogramCollection,
)

if TYPE_CHECKING:
    from pathlib import Path

    from faninsar.stack.session import Stack


class StackInterferogramCollection(InterferogramCollection):
    """Read-only interferogram collection backed by a committed Stack.

    The Stack artifact store is the canonical producer format and stores
    validated NumPy payloads rather than georeferenced COGs.  This collection
    exposes the same pair-level Dataset surface used by :class:`Network`,
    without copying products into a second on-disk layout.
    """

    def __init__(self, stack: Stack) -> None:
        """Bind the collection to one Stack session."""
        self._stack = stack

    @property
    def root(self) -> Path:
        """Return the committed Stack IFG artifact view root."""
        looks = self._stack.config.multilook
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    @property
    def index_metadata(self) -> dict[str, Any]:
        """Return the Network index metadata for the committed Pair set."""
        pair_names = [f"{first}_{second}" for first, second in self._pair_dates()]
        return {
            "type": "NetworkInterferogramIndex",
            "version": "stack_ifg_artifact_v1",
            "pair_count": len(pair_names),
            "pairs": pair_names,
            "assets_by_pair": {
                pair: ["coherence", "wrapped_phase", "amplitude", "unw_phase"]
                for pair in pair_names
            },
            "common_grid": True,
        }

    def _pair_dates(self) -> list[tuple[str, str]]:
        """Return the Stack's immutable Pair order."""
        from faninsar.stack.session import _iter_pair_dates

        return _iter_pair_dates(self._stack.pairs)

    def pairs(self) -> Any:
        """Return the committed Stack Pair graph."""
        from faninsar.core import Pairs

        return Pairs.from_names(
            [f"{first}_{second}" for first, second in self._pair_dates()]
        )

    def _stores(self) -> list[Any]:
        """Open the exact validated artifact set for this Stack."""
        return self._stack._pair_artifact_stores(
            looks=self._stack.config.multilook,
            ifg_root=None,
        )

    def _array(self, store: Any, name: str) -> np.ndarray:
        """Read one validated named layer from a pinned artifact store."""
        if name == "unw_phase":
            generation = self._stack._unwrap_generation
            if generation is None:
                message = "Stack has no committed unwrap generation"
                raise FileNotFoundError(message)
            pair_id = f"{store.pair[0]}_{store.pair[1]}"
            return np.asarray(generation.products[pair_id]["unwrapped_phase"])
        artifact = store.read()
        try:
            if name == "coherence":
                if artifact.coherence is None:
                    message = "coherence is not present"
                    raise FileNotFoundError(message)
                return np.asarray(artifact.coherence)
            if name == "wrapped_phase":
                return np.asarray(artifact.wrapped_phase)
            if name == "amplitude":
                return np.asarray(artifact.amplitude)
            if name == "complex_ifg":
                return np.asarray(artifact.complex_ifg)
        finally:
            del artifact
        message = f"unknown Stack interferogram asset: {name}"
        raise ValueError(message)

    def open_stack(
        self,
        name: str,
        *,
        pairs: Any = None,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Open a named Stack asset across the committed Pair network."""
        del chunks
        requested = (
            self.pairs().to_names().tolist()
            if pairs is None
            else pairs.to_names().tolist()
        )
        stores = self._stores()
        try:
            by_pair = {f"{store.pair[0]}_{store.pair[1]}": store for store in stores}
            arrays = [self._array(by_pair[pair], name) for pair in requested]
        finally:
            for store in stores:
                store.close()
        return xr.DataArray(
            np.stack(arrays, axis=0),
            dims=("pair", "y", "x"),
            coords={"pair": requested},
            name=name,
        )

    def open(self, pair: str, name: str, **kwargs: Any) -> xr.DataArray:
        """Open one named Stack asset for a Pair."""
        del kwargs
        pair_name = str(pair)
        data = self.open_stack(name)
        return data.sel(pair=pair_name, drop=True)

    def exists(self, pair: str, name: str) -> bool:
        """Return whether a named asset is committed for a Pair."""
        try:
            self.open(str(pair), name)
        except (FileNotFoundError, KeyError, ValueError):
            return False
        return True
