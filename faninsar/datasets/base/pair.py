"""PairDataset base class extracted from faninsar.datasets.base."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import rioxarray  # noqa: F401

from faninsar._core.sar.pairs import Pairs
from faninsar.logging import setup_logger
from faninsar.query import BoundingBox, GeoQuery, Points, Polygons

from .raster import RasterDataset

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from numpy.typing import NDArray
    from xarray import Dataset, DataTree

    from ._base_common import PairParser

logger = setup_logger(__name__)


class PairDataset(RasterDataset):
    """A base class for pair-like (contain two dates for one pair) datasets."""

    _pairs: Pairs
    _pair_parser: PairParser | None

    def __init__(
        self,
        *args,
        pair_parser: PairParser | None = None,
        **kwargs,
    ) -> None:
        """Initialize the dataset and attach pair metadata.

        Parameters
        ----------
        *args :
            Positional arguments forwarded to :class:`RasterDataset`.
        pair_parser : PairParser or None, optional
            Callable that parses file paths into :class:`Pairs`. When ``None``,
            :meth:`parse_pairs` is used.
        **kwargs :
            Keyword arguments forwarded to :class:`RasterDataset`.

        Returns
        -------
        None
            This method returns ``None``.

        Notes
        -----
        The provided ``pair_parser`` is stored and reused whenever the internal
        file list changes, ensuring coherence datasets can share interferogram
        parsing logic.

        See Also
        --------
        RasterDataset : Base class handling core raster operations.

        """
        self._pair_parser = pair_parser
        super().__init__(*args, **kwargs)
        self._assign_pairs_from_paths()

    @property
    def pairs(self) -> Pairs:
        """Return Pairs parsed from filenames."""
        return self._pairs

    def _assign_pairs_from_paths(self) -> None:
        """Parse interferometric pairs from current paths."""
        paths = self._files.paths.tolist()
        if len(paths) == 0:
            self._pairs = Pairs([])
            self._files.loc[:, "pair_name"] = ""
            return

        parser = (
            self._pair_parser if self._pair_parser is not None else self.parse_pairs
        )
        parsed = parser(paths)
        pairs = parsed if isinstance(parsed, Pairs) else Pairs(parsed)

        if len(pairs) != len(self._files):
            msg = (
                "Parsed interferometric pairs do not align with scanned files: "
                f"{len(pairs)} pairs for {len(self._files)} files."
            )
            raise ValueError(msg)

        self._pairs = pairs
        self._files.loc[:, "pair_name"] = pd.Series(
            pairs.to_names(), index=self._files.index
        )

    @classmethod
    def parse_pairs(cls, paths: Iterable[str | PathLike]) -> Pairs:
        """Parse pairs from filenames.

        Parameters
        ----------
        paths : list of str or PathLike
            list of file paths to parse pairs

        Returns
        -------
        pairs : Pairs object
            pairs parsed from filenames

        """
        msg = (
            "parse_pairs method must be implemented in subclass"
            " if no pair_parser is provided."
        )
        raise NotImplementedError(msg)

    @property
    def file_dim_name(self) -> str:
        """Dimension name for pair stacks."""
        return "pair"

    def _file_coords(
        self,
        indexes: NDArray,  # noqa: ARG002
        paths: list[str],  # noqa: ARG002
        files_df: pd.DataFrame,
    ) -> dict[str, tuple[str, NDArray]]:
        """Attach pair metadata to stacked coordinates."""
        if "pair_name" in files_df:
            pair_names = files_df["pair_name"].astype(str).to_numpy()
        else:
            pair_names = self.pairs.to_names()
        pairs = Pairs.from_names(pair_names)

        coords: dict[str, tuple[str, NDArray]] = {"pair": ("pair", pair_names)}
        return coords

    # New explicit per-shape query methods using pairs instead of indexes
    def get_indexes(
        self,
        pairs: Pairs | None = None,
    ) -> NDArray:
        """Return file indexes for the given pairs.

        Parameters
        ----------
        pairs : Pairs | None, optional
            Pairs to select. If None, returns indexes of all valid files.

        Returns
        -------
        indexes : np.ndarray
            Integer array of file indexes matching the given pairs.

        """
        files_df = self.files
        mask = files_df.valid.copy()
        if pairs is not None:
            pair_mask = self.pairs.where(pairs, return_type="mask")
            mask = mask & pd.Series(pair_mask, index=files_df.index)
        return files_df[mask].index.to_numpy(dtype=int)

    def points_query(
        self,
        points: Points,
        pairs: Pairs | None = None,
        verbose: bool | None = None,
    ) -> Dataset:
        """Query points for the given pairs subset (no indexes support)."""
        resolved_indexes = self.get_indexes(pairs=pairs)
        return self._compute_points_ds(points, resolved_indexes, verbose)

    def boxes_query(
        self,
        bbox: BoundingBox | list[BoundingBox],
        pairs: Pairs | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> DataTree:
        """Query bbox/boxes for the given pairs subset (no indexes support)."""
        resolved_indexes = self.get_indexes(pairs=pairs)
        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._compute_bboxes_tree_lazy(bbox, resolved_indexes, c)
        return self._compute_bboxes_tree(bbox, resolved_indexes, verbose)

    def polygons_query(
        self,
        polygons: Polygons,
        pairs: Pairs | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> DataTree:
        """Query polygons for the given pairs subset (no indexes support)."""
        resolved_indexes = self.get_indexes(pairs=pairs)
        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._compute_polygons_tree_lazy(polygons, resolved_indexes, c)
        return self._compute_polygons_tree(polygons, resolved_indexes, verbose)

    def query(
        self,
        query: GeoQuery | Points | BoundingBox | Polygons,
        pairs: Pairs | None = None,
        verbose: bool | None = None,
        chunks: dict[str, int] | int | Literal["auto", False] | None = None,
    ) -> DataTree:
        """Retrieve image values for given query using pairs subset only."""
        if isinstance(query, Points):
            query = GeoQuery(points=query)
        if isinstance(query, BoundingBox):
            query = GeoQuery(boxes=query)
        if isinstance(query, Polygons):
            query = GeoQuery(polygons=query)

        resolved_indexes = self.get_indexes(pairs=pairs)
        paths = self.files.iloc[resolved_indexes].paths.tolist()

        c = self._resolve_chunks(chunks)
        if c is not None:
            return self._sample_files_lazy(paths, query, verbose, c)
        return self._sample_files(paths, query, verbose)
