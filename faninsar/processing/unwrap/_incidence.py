"""SBAS incidence matrix construction for temporal network stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from faninsar.processing.errors import reject_invalid_state

if TYPE_CHECKING:
    from collections.abc import Sequence

    from faninsar import Pairs


def pair_id(primary: str, secondary: str) -> str:
    """Format a pair id as ``YYYYMMDD_YYYYMMDD``."""
    p = str(primary).replace("-", "")[:8]
    s = str(secondary).replace("-", "")[:8]
    return f"{p}_{s}"


def build_incidence_matrix(
    pair_dates: Pairs | Sequence[tuple[str, str]] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the SBAS incidence matrix ``A`` of shape ``(n_pairs, n_intervals)``.

    Parameters
    ----------
    pair_dates : Pairs or sequence of (primary, secondary) or ndarray
        Temporal pair definitions.

    Returns
    -------
    A : numpy.ndarray
        Incidence matrix with ones on intervals spanned by each pair.
    pair_ids : tuple of str
        Pair identifiers aligned with the rows of ``A``.

    """
    from faninsar import Pairs

    if isinstance(pair_dates, Pairs):
        a_mat = np.asarray(pair_dates.sbas_matrix(dtype=np.float64), dtype=np.float64)
        pair_ids = tuple(str(n) for n in pair_dates.names)
        return a_mat, pair_ids

    if isinstance(pair_dates, np.ndarray):
        if pair_dates.ndim != 2 or pair_dates.shape[1] != 2:
            reject_invalid_state(
                "pair_dates ndarray must have shape (n_pairs, 2)",
            )
        pairs_seq = [(str(row[0]), str(row[1])) for row in pair_dates]
    else:
        pairs_seq = [(str(a), str(b)) for a, b in pair_dates]

    if not pairs_seq:
        reject_invalid_state("pair_dates must not be empty")

    all_dates = sorted({d.replace("-", "")[:8] for p in pairs_seq for d in p})
    date_index = {d: i for i, d in enumerate(all_dates)}
    n_intervals = len(all_dates) - 1
    if n_intervals < 1:
        reject_invalid_state("need at least two unique dates for temporal unwrap")

    a_mat = np.zeros((len(pairs_seq), n_intervals), dtype=np.float64)
    pair_ids: list[str] = []
    for row, (primary, secondary) in enumerate(pairs_seq):
        d_ref = primary.replace("-", "")[:8]
        d_sec = secondary.replace("-", "")[:8]
        i0 = date_index[d_ref]
        i1 = date_index[d_sec]
        if i1 <= i0:
            reject_invalid_state(
                f"pair ({primary}, {secondary}) is not forward in time",
            )
        a_mat[row, i0:i1] = 1.0
        pair_ids.append(pair_id(d_ref, d_sec))

    return a_mat, tuple(pair_ids)
