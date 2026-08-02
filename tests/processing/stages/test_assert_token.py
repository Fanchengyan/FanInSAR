"""assert_token runtime checks."""

from __future__ import annotations

import pytest

from faninsar.core.physical import PhysicalType
from faninsar.processing.contracts.tokens import ArrayToken, assert_token
from faninsar.processing.errors import StageError


def test_assert_token_ok() -> None:
    token = ArrayToken(physical=PhysicalType.SLC_RAW)
    assert_token(token, PhysicalType.SLC_RAW)


def test_assert_token_mismatch() -> None:
    token = ArrayToken(physical=PhysicalType.SLC_RAW)
    with pytest.raises(StageError):
        assert_token(token, PhysicalType.IFG_COMPLEX, stage="stage_ifg")
