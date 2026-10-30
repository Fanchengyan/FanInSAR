"""assert_token runtime checks."""

from __future__ import annotations

import pytest

from faninsar.core.physical import PhysicalType
from faninsar.processing.errors import ProcessingContractError
from faninsar.processing.runtime.tokens import ArrayToken, assert_token


def test_assert_token_ok() -> None:
    token = ArrayToken(physical=PhysicalType.SLC_RAW)
    assert_token(token, PhysicalType.SLC_RAW)


def test_assert_token_mismatch() -> None:
    token = ArrayToken(physical=PhysicalType.SLC_RAW)
    with pytest.raises(ProcessingContractError):
        assert_token(token, PhysicalType.IFG_COMPLEX, stage="stage_ifg")
