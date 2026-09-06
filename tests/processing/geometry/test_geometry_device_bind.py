"""Geometry dispatch honors the shared ``parse_device`` resolver."""

from __future__ import annotations

import numpy as np
import pytest

from faninsar.processing.geometry import Operation, execute_geometry, prepare_geometry
from faninsar.processing.geometry import public as geometry_public
from faninsar.processing.geometry.backend_dispatch import (
    Dispatcher,
    resolve_geometry_device,
)
from faninsar.processing.geometry.torch_backends_v2 import _resolve_device
from faninsar.processing.geometry.v2 import CandidateKey, DeviceKey

from .test_public_geometry_v2 import _model


def test_resolve_geometry_device_auto_selects_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` follows ``parse_device`` onto CUDA when a GPU is present."""
    monkeypatch.setattr("faninsar.processing.runtime.device.cuda_available", lambda: True)
    assert resolve_geometry_device("auto").type == "cuda"
    assert resolve_geometry_device("gpu").type == "cuda"
    assert _resolve_device("auto").type == "cuda"
    assert geometry_public._device_key("auto", "GPU-test") == DeviceKey.cuda("GPU-test")


def test_resolve_geometry_device_explicit_cpu_stays_on_cpu() -> None:
    """An explicit CPU request is not rewritten to another device."""
    assert resolve_geometry_device("cpu").type == "cpu"
    assert _resolve_device("cpu").type == "cpu"
    assert geometry_public._device_key("cpu", None) == DeviceKey.cpu()

    prepared = prepare_geometry(Operation.GEO2RDR, _model(), shape=(1,), device="cpu")
    assert prepared.device.split(":", 1)[0] == "cpu"
    result = execute_geometry(
        prepared,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        selector="auto",
    )
    assert result.fields
    assert prepared.dispatcher.records[-1].backend == "eager"


def test_resolve_geometry_device_missing_cuda_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA is a hard error when the shared resolver has no GPU."""
    monkeypatch.setattr("faninsar.processing.runtime.device.cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        resolve_geometry_device("cuda")
    with pytest.raises(RuntimeError, match="not available"):
        _resolve_device("cuda")
    with pytest.raises(RuntimeError, match="not available"):
        _resolve_device("cuda:1")
    with pytest.raises(RuntimeError, match="not available"):
        geometry_public._device_key("cuda", "GPU-test")
    with pytest.raises(RuntimeError, match="not available"):
        prepare_geometry(Operation.GEO2RDR, _model(), shape=(1,), device="cuda")


def test_auto_dispatch_uses_cuda_identity_not_host_only() -> None:
    """Auto may pick native/compile/eager on CUDA, never a host-only sibling."""
    cuda_key = CandidateKey(
        Operation.GEO2RDR, "native", DeviceKey.cuda("GPU-test"), shape=(1,)
    )
    cpu_key = CandidateKey(Operation.GEO2RDR, "native", DeviceKey.cpu(), shape=(1,))
    called: list[str] = []
    dispatcher = Dispatcher(lambda: called.append("eager") or object())
    dispatcher.register(
        cpu_key,
        lambda: called.append("cpu-native") or object(),
        correctness_qualified=True,
        performance_eligible=True,
    )
    dispatcher.register(
        cuda_key,
        lambda: called.append("cuda-native") or object(),
        correctness_qualified=True,
        performance_eligible=True,
    )

    dispatcher.dispatch(cuda_key, "auto")
    assert called == ["cuda-native"]
    assert dispatcher.records[-1].backend == "native"
    assert dispatcher.records[-1].candidate == cuda_key
