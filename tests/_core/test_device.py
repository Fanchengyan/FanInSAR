"""Unit tests for the shared Torch device resolver."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from faninsar._core.device import parse_device
from faninsar.processing.resampling_torch import _resolve_torch_device
from faninsar.processing.torch_kernels import resolve_torch_device

Resolver = Callable[[object], torch.device]


def _force_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    count: int = 1,
) -> None:
    """Pin CUDA visibility for resolver tests."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        "faninsar._core.device.cuda_available",
        lambda: available,
    )


@pytest.fixture(
    params=[
        pytest.param(parse_device, id="parse_device"),
        pytest.param(resolve_torch_device, id="resolve_torch_device"),
        pytest.param(_resolve_torch_device, id="resampling_resolve"),
    ]
)
def resolver(request: pytest.FixtureRequest) -> Resolver:
    """Return each public resolver that must share parse_device identity."""
    return request.param


def test_auto_aliases_select_first_published_device(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None/auto/gpu resolve to CUDA when present, otherwise CPU."""
    _force_cuda(monkeypatch, available=False, count=0)
    for request in (None, "auto", "AUTO", "gpu", "GPU", " auto "):
        resolved = resolver(request)
        assert isinstance(resolved, torch.device)
        assert resolved.type == "cpu"

    _force_cuda(monkeypatch, available=True, count=2)
    for request in (None, "auto", "gpu"):
        resolved = resolver(request)
        assert resolved.type == "cuda"
        assert resolved.type != "mps"


def test_auto_never_selects_unpublished_backend(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto stays on the published set even when MPS is visible."""
    _force_cuda(monkeypatch, available=False, count=0)
    monkeypatch.setattr("faninsar._core.device.mps_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    resolved = resolver("auto")
    assert resolved.type == "cpu"
    assert resolved.type != "mps"


def test_explicit_cpu_stays_on_cpu(resolver: Resolver) -> None:
    """An explicit CPU request is admitted as CPU."""
    resolved = resolver("cpu")
    assert resolved.type == "cpu"
    assert resolver(torch.device("cpu")).type == "cpu"


def test_explicit_cuda_stays_on_cuda_when_present(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare cuda and cuda:N keep that identity when the ordinal exists."""
    _force_cuda(monkeypatch, available=True, count=2)
    bare = resolver("cuda")
    assert bare.type == "cuda"
    indexed = resolver("cuda:1")
    assert indexed.type == "cuda"
    assert indexed.index == 1
    constructed = resolver(torch.device("cuda:0"))
    assert constructed.type == "cuda"
    assert constructed.index == 0


def test_missing_cuda_fails_before_admission(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA is fail-closed when the backend is absent."""
    _force_cuda(monkeypatch, available=False, count=0)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolver("cuda")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolver("cuda:0")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolver(torch.device("cuda"))


def test_cuda_ordinal_out_of_range_is_not_wrapped(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cuda:256 must not wrap to cuda:0; construction is not admission."""
    _force_cuda(monkeypatch, available=True, count=1)
    assert torch.device("cuda:256").index == 0
    with pytest.raises(RuntimeError, match="256"):
        resolver("cuda:256")
    with pytest.raises(RuntimeError, match=r"range\(1\)|ordinal 1"):
        resolver("cuda:1")


def test_valid_cuda_ordinal_is_kept(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible CUDA ordinal stays on that ordinal."""
    _force_cuda(monkeypatch, available=True, count=1)
    resolved = resolver("cuda:0")
    assert resolved.type == "cuda"
    assert resolved.index == 0


def test_explicit_unavailable_mps_fails_closed(
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit MPS fails before compute when the backend cannot run."""
    monkeypatch.setattr("faninsar._core.device.mps_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        resolver("mps")


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS must be present to admit an explicit unpublished request",
)
def test_explicit_mps_stays_on_mps_when_usable(resolver: Resolver) -> None:
    """An explicit unpublished backend stays on that device when it can run."""
    resolved = resolver("mps")
    assert resolved.type == "mps"
    assert parse_device("auto").type != "mps"


def test_unsupported_device_string_fails_closed(resolver: Resolver) -> None:
    """Strings Torch cannot construct fail before any array copy."""
    with pytest.raises(RuntimeError):
        resolver("not-a-device")


def test_constructible_unpublished_backend_is_admitted(
    resolver: Resolver,
) -> None:
    """Any device Torch can construct is a valid explicit request."""
    constructed = torch.device("xla")
    resolved = resolver("xla")
    assert resolved.type == constructed.type


def test_non_device_type_is_rejected(resolver: Resolver) -> None:
    """Only strings, torch.device, and None are accepted."""
    with pytest.raises(TypeError, match=r"string or torch\.device"):
        resolver(1)


def test_wrappers_share_parse_device_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed-enum helpers must not change the parse_device identity."""
    _force_cuda(monkeypatch, available=True, count=2)
    for request in ("auto", "cpu", "cuda", "cuda:1"):
        expected = parse_device(request)
        assert resolve_torch_device(request) == expected
        assert _resolve_torch_device(request) == expected
