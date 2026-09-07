"""Network-facing views over committed Stack interferogram products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from faninsar.network.readers.interferogram import (
    InterferogramCollection,
)

if TYPE_CHECKING:
    from faninsar.stack.session import Stack


class StackInterferogramCollection(InterferogramCollection):
    """Read-only interferogram collection backed by a committed Stack.

    The Stack artifact store is the canonical producer format and stores
    validated NumPy payloads rather than georeferenced COGs.  This collection
    exposes the same pair-level Dataset surface used by :class:`Network`,
    without copying products into a second on-disk layout.
    """

    def __init__(self, stack: Stack | str | Path) -> None:
        """Bind the collection to a Stack or an exported Stack root."""
        self._stack = stack if hasattr(stack, "config") else None
        self._root = Path(stack) / "interferograms" if self._stack is None else None
        self._index = self.index_metadata

    @property
    def root(self) -> Path:
        """Return the committed Stack IFG artifact view root."""
        if self._root is not None:
            return self._root
        assert self._stack is not None
        looks = self._stack.config.multilook
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    @property
    def index_metadata(self) -> dict[str, Any]:
        """Return the Network index metadata for the committed Pair set."""
        if self._stack is None:
            index_path = self.root / "interferograms_index.json"
            if not index_path.is_file():
                raise FileNotFoundError(index_path)
            return json.loads(index_path.read_text(encoding="utf-8"))
        assert self._stack is not None
        pair_names = [f"{first}_{second}" for first, second in self._pair_dates()]
        stores = self._stores()
        try:
            assets_by_pair: dict[str, list[str]] = {}
            for pair_name, store in zip(pair_names, stores, strict=True):
                assets = ["complex_ifg", "wrapped_phase", "amplitude"]
                if "coherence" in store._payloads:
                    assets.append("coherence")
                if (
                    self._stack._unwrap_generation is not None
                    or (store.root / "UNWRAP_CURRENT").is_file()
                ):
                    assets.append("unw_phase")
                assets_by_pair[pair_name] = assets
        finally:
            for store in stores:
                store.close()
        return {
            "type": "NetworkInterferogramIndex",
            "version": "stack_ifg_artifact_v1",
            "pair_count": len(pair_names),
            "pairs": pair_names,
            "assets_by_pair": assets_by_pair,
            "common_grid": True,
        }

    def _pair_dates(self) -> list[tuple[str, str]]:
        """Return the Stack's immutable Pair order."""
        assert self._stack is not None
        from faninsar.stack.session import _iter_pair_dates

        return _iter_pair_dates(self._stack.pairs)

    def pairs(self) -> Any:
        """Return the committed Stack Pair graph."""
        from faninsar.core import Pairs

        if self._stack is None:
            return Pairs.from_names(self.index_metadata.get("pairs", []))
        assert self._stack is not None
        return Pairs.from_names(
            [f"{first}_{second}" for first, second in self._pair_dates()]
        )

    def _stores(self) -> list[Any]:
        """Open the exact validated artifact set for this Stack."""
        assert self._stack is not None
        return self._stack._pair_artifact_stores(
            looks=self._stack.config.multilook,
            ifg_root=None,
        )

    def _array(self, store: Any, name: str) -> np.ndarray:
        """Read one validated named layer from a pinned artifact store."""
        if self._stack is None:
            path = self.path(str(store), name)
            try:
                return np.load(path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise FileNotFoundError(path) from error
        assert self._stack is not None
        if name == "unw_phase":
            generation = self._stack._unwrap_generation
            if generation is None:
                return np.asarray(store.read_unwrapped().unwrapped_phase)
            pair_id = f"{store.pair[0]}_{store.pair[1]}"
            return np.asarray(generation.products[pair_id]["unwrapped_phase"])
        artifact = store.read()
        try:
            if name == "coherence":
                if artifact.coherence is None:
                    message = "coherence is not present"
                    raise FileNotFoundError(message)
                value = artifact.coherence
            elif name == "wrapped_phase":
                value = artifact.wrapped_phase
            elif name == "amplitude":
                value = artifact.amplitude
            elif name == "complex_ifg":
                value = artifact.complex_ifg
            else:
                message = f"unknown Stack interferogram asset: {name}"
                raise ValueError(message)
        finally:
            del artifact
        return np.asarray(value)

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
        if self._stack is None:
            arrays = [self._array(pair, name) for pair in requested]
            return xr.DataArray(
                np.stack(arrays, axis=0),
                dims=("pair", "y", "x"),
                coords={"pair": requested},
                name=name,
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
        if self._stack is None:
            return xr.DataArray(
                self._array(str(pair), name), dims=("y", "x"), name=name
            )
        data = self.open_stack(name)
        return data.sel(pair=pair_name, drop=True)

    def exists(self, pair: str, name: str) -> bool:
        """Return whether a named asset is committed for a Pair."""
        try:
            self.open(str(pair), name)
        except (FileNotFoundError, KeyError, ValueError):
            return False
        return True

    def path(self, pair: str, name: str) -> Path:
        """Return the Stack artifact path for one pair asset."""
        if name not in {
            "unw_phase",
            "wrapped_phase",
            "amplitude",
            "coherence",
            "complex_ifg",
        }:
            message = f"unknown Stack interferogram asset: {name}"
            raise ValueError(message)
        if self._stack is not None:
            stores = self._stores()
            try:
                selected = next(
                    store
                    for store in stores
                    if f"{store.pair[0]}_{store.pair[1]}" == str(pair)
                )
                if name == "unw_phase":
                    generation = self._stack._unwrap_generation
                    if generation is None:
                        from faninsar.io.storage.artifact_transaction import (
                            open_current_generation,
                        )

                        if not (selected.root / "UNWRAP_CURRENT").is_file():
                            raise FileNotFoundError(selected.root / "UNWRAP_CURRENT")
                        opened = open_current_generation(selected.root, "unwrap")
                        try:
                            return opened.path / "unwrapped_phase.npy"
                        finally:
                            opened.lease.close()
                    pair_id = f"{selected.pair[0]}_{selected.pair[1]}"
                    return generation.generation_root / pair_id / "unwrapped_phase.npy"
                return selected.generation_root / f"{name}.npy"
            finally:
                for store in stores:
                    store.close()
        return self.root / str(pair) / f"{name}.npy"

    def item(self, pair: str) -> dict[str, Any]:
        """Return a generated item-like record for one Stack Pair."""
        if str(pair) not in self.pairs().to_names().tolist():
            raise KeyError(pair)
        return {
            "id": str(pair),
            "pair": str(pair),
            "assets": {
                name: {"href": str(self.path(str(pair), name))}
                for name in (
                    "complex_ifg",
                    "unw_phase",
                    "wrapped_phase",
                    "amplitude",
                    "coherence",
                )
                if self.exists(str(pair), name)
            },
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the committed Stack artifact collection."""
        return {
            "root": str(self.root),
            "pair_count": len(self.pairs()),
            "pairs": self.pairs().to_names().tolist(),
            "assets_by_pair": self.index_metadata.get("assets_by_pair", {}),
            "common_grid": bool(self.index_metadata.get("common_grid", True)),
        }

    def to_zarr_stack(
        self,
        name: str = "unw_phase",
        *,
        zarr_path: str | Path | None = None,
        chunks: tuple[int, int, int] = (1, 512, 512),
        overwrite: bool = False,
    ) -> Path:
        """Write Stack NPY artifacts as one xarray Zarr cube."""
        out = Path(zarr_path) if zarr_path is not None else self.root / f"{name}.zarr"
        if out.exists() and not overwrite:
            raise FileExistsError(out)
        if out.exists():
            import shutil

            shutil.rmtree(out)
        data = self.open_stack(name).chunk(
            {"pair": chunks[0], "y": chunks[1], "x": chunks[2]}
        )
        data.to_dataset(name=name).to_zarr(str(out), mode="w", consolidated=False)
        return out
