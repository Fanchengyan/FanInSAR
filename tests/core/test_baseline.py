"""Tests for baseline conversion across Pair networks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from faninsar import Baselines, Pairs


def test_to_pair_wise_preserves_triangle_pair_order() -> None:
    """Convert date baselines without reindexing repeated acquisition dates."""
    pairs = Pairs.from_names(
        [
            "20200101_20200113",
            "20200113_20200125",
            "20200101_20200125",
        ]
    )
    baselines = Baselines(
        pd.to_datetime(["20200101", "20200113", "20200125"]),
        np.array([10.0, 22.0, 47.0], dtype=np.float32),
    )

    result = baselines.to_pair_wise(pairs)

    assert result.index.tolist() == pairs.to_names().tolist()
    np.testing.assert_array_equal(result.to_numpy(), [12.0, 25.0, 37.0])
