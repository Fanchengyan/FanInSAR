from __future__ import annotations

from pathlib import Path


def retrieve_meta_value(path: str | Path, key: str) -> str:
    """Parse the metadata of the HyP3 Sentinel-1 product.

    Parameters
    ----------
    path : Path
        The path of metadata file to be parsed.
    key : str
        The key of the metadata to be retrieved.

    Returns
    -------
    value : str
        The value of the metadata with the given key.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if key in line:
                value = line.split(":")[1].strip()
                return value
