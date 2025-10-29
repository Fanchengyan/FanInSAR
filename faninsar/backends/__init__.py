"""Backends for lazy loading and parallel computation."""

from __future__ import annotations

from .lazy_rasterio import LazyMultiFileReader, LazyRasterioReader

__all__ = [
    "LazyMultiFileReader",
    "LazyRasterioReader",
]
