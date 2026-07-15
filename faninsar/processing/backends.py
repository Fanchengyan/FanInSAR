"""Backend protocols for lazy processing-array storage."""

from __future__ import annotations

import collections.abc  # noqa: TC003
from typing import Protocol, TypeVar

from . import coordinates  # noqa: TC001

ReadValueT_co = TypeVar("ReadValueT_co", covariant=True)
WriteValueT_contra = TypeVar("WriteValueT_contra", contravariant=True)


class ArrayReader(Protocol[ReadValueT_co]):
    """Read typed slices without requiring an eager storage implementation."""

    def read(
        self,
        descriptor: coordinates.ArrayDescriptor,
        selection: tuple[slice, slice],
    ) -> ReadValueT_co:
        """Read one two-dimensional selection from an array asset."""


class ArrayWriter(Protocol[WriteValueT_contra]):
    """Persist a typed array and return its immutable descriptor."""

    def write(
        self,
        uri: str,
        value: WriteValueT_contra,
        metadata: collections.abc.Mapping[str, str],
    ) -> coordinates.ArrayDescriptor:
        """Write an array value to a backend-specific URI."""
