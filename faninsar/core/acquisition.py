"""A module for SAR acquisition related operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

import pandas as pd
import xarray as xr

from faninsar.logging import setup_logger
from faninsar.plotting.render import array_repr

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from faninsar.network.products import AcquisitionKey

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """Immutable physical acquisition keyed by ``AcquisitionKey``."""

    key: AcquisitionKey
    sensing_time: datetime

    def __post_init__(self) -> None:
        """Validate the physical identity and normalize time to UTC."""
        from faninsar.network.products import AcquisitionKey

        if not isinstance(self.key, AcquisitionKey):
            message = "key must be an AcquisitionKey"
            raise TypeError(message)
        if self.sensing_time.tzinfo is None:
            message = "sensing_time must be timezone-aware"
            raise ValueError(message)
        object.__setattr__(self, "sensing_time", self.sensing_time.astimezone(UTC))


class Acquisitions:
    """Immutable ordered collection of physical acquisition values.

    Both :class:`AcquisitionRecord` and the record-shaped constructor form of
    :class:`Acquisition` represent one physical observation.  Accepting both
    keeps the public singular and collection forms on the same identity
    boundary while preserving the value supplied by the caller.
    """

    def __init__(
        self, records: tuple[AcquisitionRecord | Acquisition, ...] = ()
    ) -> None:
        """Create a validated collection."""
        if any(
            not isinstance(record, (AcquisitionRecord, Acquisition))
            for record in records
        ):
            message = "records must contain physical Acquisition values"
            raise TypeError(message)
        keys = [record.key for record in records]
        if len(set(keys)) != len(keys):
            message = "acquisition keys must be unique"
            raise ValueError(message)
        self._records = tuple(records)

    def __getitem__(
        self, index: int | slice
    ) -> AcquisitionRecord | Acquisition | tuple[AcquisitionRecord | Acquisition, ...]:
        """Return one record or an immutable slice."""
        return self._records[index]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[AcquisitionRecord | Acquisition]:
        """Iterate over records."""
        return iter(self._records)


class Acquisition(pd.DatetimeIndex):
    """A class to handle SAR acquisition dates in FanInSAR.

    This class is a wrapper around :class:`pandas.DatetimeIndex` to handle
    SAR acquisition dates in FanInSAR.

    Set-style ``+``/``-``/``&``/``|``/``^`` operators combine two date sets
    and return :class:`Acquisition` views; the named methods
    :meth:`union`, :meth:`difference`, :meth:`intersection`, and
    :meth:`symmetric_difference` do the same. Element-wise arithmetic with
    scalars or :class:`pandas.Timedelta` keeps the pandas behaviour.
    """

    dims = ("dates",)
    _in_memory = True

    def __new__(cls, *args, **kwargs) -> Self:
        """Create a new instance of Acquisition."""
        from faninsar.network.products import AcquisitionKey

        if len(args) == 2 and isinstance(args[0], AcquisitionKey):
            record = super(Acquisition, cls).__new__(cls, [args[1]])
            record.key = args[0]
            record.sensing_time = args[1]
            return record
        return super(Acquisition, cls).__new__(cls, *args, **kwargs)

    @property
    def key(self) -> AcquisitionKey:
        """Return the physical identity for a record-shaped acquisition."""
        value = getattr(self, "_physical_key", None)
        if value is None:
            message = "date collections do not have a physical key"
            raise AttributeError(message)
        return value

    @key.setter
    def key(self, value: AcquisitionKey) -> None:
        if hasattr(self, "_physical_key"):
            message = "physical acquisition identity is immutable"
            raise AttributeError(message)
        self._physical_key = value

    @property
    def sensing_time(self) -> datetime:
        """Return the sensing time for a record-shaped acquisition."""
        value = getattr(self, "_sensing_time", None)
        if value is None:
            message = "date collections do not have a sensing time"
            raise AttributeError(message)
        return value

    @sensing_time.setter
    def sensing_time(self, value: datetime) -> None:
        if hasattr(self, "_sensing_time"):
            message = "physical acquisition sensing time is immutable"
            raise AttributeError(message)
        self._sensing_time = value

    def __eq__(self, other: object) -> object:
        """Compare physical records by key and date collections by values."""
        if hasattr(self, "_physical_key"):
            return (
                isinstance(other, Acquisition)
                and getattr(other, "_physical_key", None) == self._physical_key
            )
        return super().__eq__(other)

    def __ne__(self, other: object) -> object:
        """Compare physical records by the inverse of their key identity."""
        if hasattr(self, "_physical_key"):
            return not (
                isinstance(other, Acquisition)
                and getattr(other, "_physical_key", None) == self._physical_key
            )
        return super().__ne__(other)

    def __hash__(self) -> int:
        """Hash a physical record by its immutable acquisition key."""
        if hasattr(self, "_physical_key"):
            return hash(self._physical_key)
        message = "date collection acquisitions are unhashable"
        raise TypeError(message)

    def _repr_html_(self) -> str:
        """Return the HTML representation of the class."""
        return array_repr(self)

    def to_xarray(self) -> xr.Variable:
        """Convert the acquisition dates to xarray format."""
        return xr.Variable("Acquisition", self)

    @property
    def stats(self) -> dict:
        """Return the statistical attributes of the acquisition dates."""
        length = len(self)
        return {
            "start": self.min().strftime("%F") if length > 0 else None,
            "end": self.max().strftime("%F") if length > 0 else None,
            "unique": len(self.unique()) if length > 0 else 0,
            "total": length,
        }

    @property
    def data(self) -> np.ndarray:
        """Return the internal data of the index."""
        return self._data

    # ---- set-style combination -------------------------------------------

    @staticmethod
    def _ensure_acquisition(other: object) -> Acquisition:
        """Coerce a date collection operand to ``Acquisition``.

        A plain :class:`pandas.DatetimeIndex` is wrapped because
        ``DatetimeIndex`` set methods assert ``type(other) is type(self)``
        on the subclass.
        """
        if isinstance(other, Acquisition):
            return other
        if isinstance(other, pd.DatetimeIndex):
            return Acquisition(other)
        message = (
            "expected an Acquisition, DatetimeIndex, or list of dates; "
            f"got {type(other).__name__}"
        )
        raise TypeError(message)

    def union(self, other: object, sort: bool = True) -> Acquisition:  # type: ignore[override]
        """Return the set union of two date collections as ``Acquisition``."""
        result = super().union(self._ensure_acquisition(other), sort=sort)
        return Acquisition(result)

    def difference(self, other: object, sort: bool = True) -> Acquisition:  # type: ignore[override]
        """Return the set difference ``self - other`` as ``Acquisition``."""
        result = super().difference(self._ensure_acquisition(other), sort=sort)
        return Acquisition(result)

    def intersection(self, other: object, sort: bool = True) -> Acquisition:  # type: ignore[override]
        """Return the set intersection of two date collections as ``Acquisition``."""
        result = super().intersection(self._ensure_acquisition(other), sort=sort)
        return Acquisition(result)

    def symmetric_difference(
        self, other: object, result_name: object = None, sort: bool = True
    ) -> Acquisition:  # type: ignore[override]
        """Return the symmetric difference of two date collections."""
        del result_name
        result = super().symmetric_difference(
            self._ensure_acquisition(other), sort=sort
        )
        return Acquisition(result)

    def __add__(self, other: object) -> Acquisition:
        """Return the set union ``self | other`` as ``Acquisition``."""
        if isinstance(other, pd.DatetimeIndex):
            return self.union(other)
        return Acquisition(super().__add__(other))

    def __sub__(self, other: object) -> Acquisition:
        """Return the set difference ``self - other`` as ``Acquisition``."""
        if isinstance(other, pd.DatetimeIndex):
            return self.difference(other)
        return Acquisition(super().__sub__(other))

    def __and__(self, other: object) -> Acquisition:
        """Return the set intersection ``self & other`` as ``Acquisition``."""
        return self.intersection(other)

    def __or__(self, other: object) -> Acquisition:
        """Return the set union ``self | other`` as ``Acquisition``."""
        return self.union(other)

    def __xor__(self, other: object) -> Acquisition:
        """Return the symmetric difference ``self ^ other`` as ``Acquisition``."""
        return self.symmetric_difference(other)
