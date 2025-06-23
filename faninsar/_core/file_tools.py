"""Utility functions for working with files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


def load_meta(
    path: PathLike,
    key: str,
    sep: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> str | None:
    r"""Load a metadata record from a file.

    .. note::
        - The key is case-insensitive.
        - If you need to retrieve multiple metadata records, use the
            :func:`load_metas` function instead. It is more io-efficient
            to retrieve multiple records at once.

    Parameters
    ----------
    path : PathLike
        The path of metadata file to be parsed.
    key : str
        The key of the metadata to be retrieved.
    sep : str, optional
        The separator between the key and value. Default is None, which splits
        on any whitespace (including including \n \r \t \f and spaces).
    line_start : int, optional
        The line number to start and end reading the metadata file. Default is None.
    line_end : int, optional
        The line number to end reading the metadata file. Can be negative, in
        which case it is treated as the number of lines from the end of the file.
        Default is None, which reads to the end of the file.

    Returns
    -------
    value : str
        The value of the metadata with the given key.

    Raises
    ------
    ValueError
        If the line_start or line_end is not an integer, or if line_start is
        greater than line_end.

    """
    if line_start is not None:
        line_start = ensure_int(line_start, "line_start")
    if line_end is not None:
        line_end = ensure_int(line_end, "line_end")
    if line_start and line_end and line_start > line_end and line_end >= 0:
        msg = (
            "line_start must be less than or equal to line_end."
            f"line_start: {line_start}, line_end: {line_end}"
        )
        raise ValueError(msg)
    with Path(path).open(encoding="utf-8") as f:
        lines = f.readlines()[line_start:line_end]
        for line in lines:
            if key.lower() in line.lower():
                return strip_str(line.split(sep)[1])
        return None


def load_metas(
    path: PathLike,
    keys: list[str],
    sep: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> dict[str, str | None]:
    r"""Load multiple metadata records from a file at once.

    .. tip::
        - The keys are case-insensitive.

    Parameters
    ----------
    path : Path
        The path of metadata file to be parsed.
    keys : list[str]
        The keys of the metadata to be retrieved.
    sep : str, optional
        The separator between the key and value. Default is None, which splits
        on any whitespace (including including \n \r \t \f and spaces).
    line_start : int, optional
        The line number to start and end reading the metadata file. Default is None.
    line_end : int, optional
        The line number to end reading the metadata file. Can be negative, in
        which case it is treated as the number of lines from the end of the file.
        Default is None, which reads to the end of the file.

    Returns
    -------
    values : dict[str, str]
        A dictionary containing the values of the metadata with the given keys.

    """
    if line_start is not None:
        line_start = ensure_int(line_start, "line_start")
    if line_end is not None:
        line_end = ensure_int(line_end, "line_end")
    if line_start and line_end and line_start > line_end and line_end >= 0:
        msg = (
            "line_start must be less than or equal to line_end."
            f"line_start: {line_start}, line_end: {line_end}"
        )
        raise ValueError(msg)
    with Path(path).open(encoding="utf-8") as f:
        lines = f.readlines()[line_start:line_end]
        values = [None for _ in keys]
        for line in lines:
            for key in keys:
                if key.lower() not in line.lower():
                    continue
                values[keys.index(key)] = strip_str(line.split(sep)[1])
            # once all values are found, return the dictionary
            if all(values):
                return dict(zip(keys, values))
        return dict(zip(keys, values))


def ensure_int(value: str | int, name: str) -> int:
    """Ensure a value is an integer.

    Parameters
    ----------
    value : str
        The value to be converted to an integer.
    name : str
        The name of the value to be converted.

    Returns
    -------
    int
        The integer value of the input string.

    Raises
    ------
    ValueError
        If the input string cannot be converted to an integer.

    """
    try:
        return int(value)
    except ValueError as e:
        msg = f"{name} must be an integer."
        raise TypeError(msg) from e


def strip_str(string: str) -> str:
    """Strip characters from a string.

    Any leading or trailing whitespace, double or single quotes, and newline
    or carriage return characters are removed from the input string.

    Parameters
    ----------
    string : str
        The string to be stripped.

    Returns
    -------
    str
        The stripped string.

    """
    return string.strip().strip('"').strip("'").strip(r"\n").strip(r"\r").strip()
