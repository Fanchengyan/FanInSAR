"""Tests for the explicit device capability matrix."""

from __future__ import annotations

import torch

from faninsar.processing.runtime.device_matrix import (
    DeviceKind,
    capability_matrix,
    complex_multiply,
    probe_devices,
    select_device_for_kernel,
)


def test_capability_matrix_lists_required_kernels() -> None:
    """Capability matrix includes CPU truth and accelerator rows."""
    names = {row.name for row in capability_matrix()}
    assert "complex_multiply" in names
    assert "irls_pcg" in names
    assert "fft_filter" in names


def test_select_device_falls_back_to_cpu_for_unsupported_mps_kernel() -> None:
    """Unsupported MPS kernels fall back rather than claiming parity."""
    device = select_device_for_kernel("irls_pcg", preferred=DeviceKind.MPS)
    assert device in {DeviceKind.CPU, DeviceKind.CUDA}


def test_complex_multiply_cpu_kernel() -> None:
    """Torch complex multiply works on CPU tensors."""
    left = torch.tensor([1 + 1j, 2 + 0j], dtype=torch.complex64)
    right = torch.tensor([0 + 1j, 1 + 1j], dtype=torch.complex64)
    out = complex_multiply(left, right)
    expected = left * right
    assert torch.allclose(out, expected)
    assert probe_devices()["cpu"] is True
