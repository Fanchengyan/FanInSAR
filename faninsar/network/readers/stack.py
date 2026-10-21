"""Network-facing view over the committed products of a Stack (PROPOSAL-0059).

The collection reads ``ifg/ml_AxR/<pair>`` and ``unwrap/ml_AxR/<pair>``
products in place; nothing is copied into a second layout.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from faninsar.network.readers.interferogram import InterferogramCollection

if TYPE_CHECKING:
    from faninsar.io.storage.ifg_store import InterferogramArtifactStore
    from faninsar.stack.session import Stack

_ASSETS = ("complex_ifg", "unw_phase", "wrapped_phase", "amplitude", "coherence")


class StackInterferogramCollection(InterferogramCollection):
    """Read-only interferogram collection backed by a Stack work directory.

    Parameters
    ----------
    stack : Stack
        The Stack whose planned Pair network and multilook select the
        products.

    """

    def __init__(self, stack: Stack) -> None:
        """Bind the collection to a Stack."""
        self._stack = stack

    @property
    def root(self) -> Path:
        """Return the IFG product directory of the configured multilook."""
        looks = self._stack.config.multilook
        return self._stack.config.work_dir / "ifg" / f"ml_{looks[0]}x{looks[1]}"

    @property
    def index_metadata(self) -> dict[str, Any]:
        """Return the Network index metadata for the committed Pair set."""
        stores = self._stores()
        assets_by_pair: dict[str, list[str]] = {}
        for store in stores:
            assets = ["complex_ifg", "wrapped_phase", "amplitude"]
            if "coherence" in store._payloads:
                assets.append("coherence")
            if store.has_unwrapped():
                assets.append("unw_phase")
            assets_by_pair[_pair_id(store)] = assets
        return {
            "type": "NetworkInterferogramIndex",
            "pair_count": len(stores),
            "pairs": [_pair_id(store) for store in stores],
            "assets_by_pair": assets_by_pair,
            "common_grid": True,
        }

    @property
    def spatial_metadata(self) -> dict[str, Any]:
        """Return the common spatial metadata of the IFG products.

        Returns
        -------
        dict
            ``domain``, ``shape``, ``grid_identity``, and for geo products
            ``crs``/``transform``. Radar collections omit geographic metadata
            instead of fabricating it.

        """
        stores = self._stores()
        if not stores:
            message = "Stack has no interferogram products"
            raise ValueError(message)
        first = stores[0]
        values: dict[str, Any] = {
            "domain": first.domain,
            "shape": list(first.shape),
            "grid_identity": first.grid_identity,
        }
        if first.crs is not None:
            values["crs"] = first.crs
        if first.transform is not None:
            values["transform"] = list(first.transform)
        return values

    def pairs(self) -> Any:
        """Return the planned Stack Pair graph."""
        from faninsar.core import Pairs
        from faninsar.stack.session import _iter_pair_dates

        return Pairs.from_names(
            [
                f"{first}_{second}"
                for first, second in _iter_pair_dates(self._stack.pairs)
            ]
        )

    def _stores(self) -> list[InterferogramArtifactStore]:
        """Open the exact planned IFG product set."""
        return self._stack._pair_artifact_stores(looks=self._stack.config.multilook)

    def _array(self, store: InterferogramArtifactStore, name: str) -> np.ndarray:
        """Read one named layer of one Pair."""
        if name == "unw_phase":
            return np.asarray(store.read_unwrapped().unwrapped_phase)
        artifact = store.read()
        if name == "coherence":
            if artifact.coherence is None:
                message = "coherence is not present"
                raise FileNotFoundError(message)
            return np.asarray(artifact.coherence)
        if name not in {"wrapped_phase", "amplitude", "complex_ifg"}:
            message = f"unknown Stack interferogram asset: {name}"
            raise ValueError(message)
        return np.asarray(getattr(artifact, name))

    def open_stack(
        self,
        name: str,
        *,
        pairs: Any = None,
        chunks: Any = None,
    ) -> xr.DataArray:
        """Open a named asset across the planned Pair network."""
        del chunks
        requested = (
            self.pairs().to_names().tolist()
            if pairs is None
            else pairs.to_names().tolist()
        )
        by_pair = {_pair_id(store): store for store in self._stores()}
        arrays = [self._array(by_pair[pair], name) for pair in requested]
        return xr.DataArray(
            np.stack(arrays, axis=0),
            dims=("pair", "y", "x"),
            coords={"pair": requested},
            name=name,
        )

    def open(self, pair: str, name: str, **kwargs: Any) -> xr.DataArray:
        """Open one named asset of one Pair."""
        del kwargs
        return self.open_stack(name).sel(pair=str(pair), drop=True)

    def exists(self, pair: str, name: str) -> bool:
        """Return whether a named asset is committed for a Pair."""
        try:
            self.open(str(pair), name)
        except (FileNotFoundError, KeyError, ValueError):
            return False
        return True

    def path(self, pair: str, name: str) -> Path:
        """Return the product path of one Pair asset."""
        if name not in _ASSETS:
            message = f"unknown Stack interferogram asset: {name}"
            raise ValueError(message)
        selected = next(
            (store for store in self._stores() if _pair_id(store) == str(pair)),
            None,
        )
        if selected is None:
            raise KeyError(pair)
        if name == "unw_phase":
            if not selected.has_unwrapped():
                raise FileNotFoundError(selected.unwrap_root)
            return selected.unwrap_root / "unwrapped_phase.zarr"
        return selected.root / f"{name}.zarr"

    def item(self, pair: str) -> dict[str, Any]:
        """Return an item-like record for one Stack Pair."""
        if str(pair) not in self.pairs().to_names().tolist():
            raise KeyError(pair)
        return {
            "id": str(pair),
            "pair": str(pair),
            "assets": {
                name: {"href": str(self.path(str(pair), name))}
                for name in _ASSETS
                if self.exists(str(pair), name)
            },
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the committed Stack products."""
        index = self.index_metadata
        return {
            "root": str(self.root),
            "pair_count": index["pair_count"],
            "pairs": index["pairs"],
            "assets_by_pair": index["assets_by_pair"],
            "common_grid": True,
        }

    def unwrap_write_ids(self) -> dict[str, str]:
        """Return the ``write_id`` of each Pair's current unwrap product."""
        return {
            _pair_id(store): store.unwrap_product().write_id for store in self._stores()
        }

    def to_zarr_stack(
        self,
        name: str = "unw_phase",
        *,
        zarr_path: str | Path,
        chunks: tuple[int, int, int] = (1, 512, 512),
        overwrite: bool = False,
    ) -> Path:
        """Export one asset across the Pair network as an xarray Zarr cube.

        Parameters
        ----------
        name : str, optional
            Asset to export.
        zarr_path : str or pathlib.Path
            Destination outside the Stack product tree.
        chunks : tuple[int, int, int], optional
            ``(pair, y, x)`` chunking.
        overwrite : bool, optional
            Replace an existing destination.

        Returns
        -------
        pathlib.Path
            The written Zarr store.

        """
        out = Path(zarr_path)
        if out.exists() and not overwrite:
            raise FileExistsError(out)
        if out.exists():
            shutil.rmtree(out)
        data = self.open_stack(name).chunk(
            {"pair": chunks[0], "y": chunks[1], "x": chunks[2]}
        )
        data.to_dataset(name=name).to_zarr(str(out), mode="w", consolidated=False)
        return out


def _pair_id(store: InterferogramArtifactStore) -> str:
    """Return the canonical Pair id of one IFG product."""
    return f"{store.pair[0]}_{store.pair[1]}"
