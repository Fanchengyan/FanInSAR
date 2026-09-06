"""Scientific accuracy gates for compile kernel targets."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.runtime.compute.compile import COMPILE_TARGETS, get_compile_manager

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("name", sorted(COMPILE_TARGETS))
def test_target_registered(name: str) -> None:
    assert name in COMPILE_TARGETS
    fn = COMPILE_TARGETS[name].factory()
    assert callable(fn)


def test_compile_manager_multilook() -> None:
    mgr = get_compile_manager()
    fn = mgr.get("multilook", device="cpu")
    x = torch.ones(32, 32, dtype=torch.complex64)
    y = fn(x, looks=(2, 2))
    assert y.shape == (16, 16)


def test_form_interferogram_matches_eager() -> None:
    mgr = get_compile_manager()
    eager = COMPILE_TARGETS["form_interferogram"].factory()
    compiled = mgr.get("form_interferogram", device="cpu")
    ref = torch.randn(32, 32, dtype=torch.complex64)
    sec = torch.randn(32, 32, dtype=torch.complex64)
    a = eager(ref, sec)
    b = compiled(ref, sec)
    np.testing.assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
