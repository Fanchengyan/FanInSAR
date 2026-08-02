"""fis.run front-door smoke tests."""

from __future__ import annotations

import pytest

from faninsar import run


def test_run_requires_keys() -> None:
    with pytest.raises(ValueError, match="reference"):
        run({})
