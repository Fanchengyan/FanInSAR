"""Tests for faninsar._core.alg module."""  # noqa: D100, INP001

from __future__ import annotations

import numpy as np
import pytest
import torch

from faninsar._core.alg import gradient_magnitude, percentile_range


class TestGradientMagnitude:
    """Test gradient_magnitude function."""

    def test_2d_numpy_array(self) -> None:
        """Test gradient magnitude for 2D numpy array."""
        # Create a simple 3x3 array
        img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        grad = gradient_magnitude(img)

        # Expected output shape: (H-1, W-1) = (2, 2)
        assert grad.shape == (2, 2)
        assert isinstance(grad, np.ndarray)

        # Verify gradient calculation
        # grad_x = [[1, 1], [1, 1]], grad_y = [[3, 3], [3, 3]]
        # grad_magnitude = sqrt(1^2 + 3^2) = sqrt(10) ≈ 3.162
        expected = np.full((2, 2), np.sqrt(1.0**2 + 3.0**2), dtype=np.float32)
        np.testing.assert_allclose(grad, expected, rtol=1e-5)

    def test_2d_torch_tensor(self) -> None:
        """Test gradient magnitude for 2D torch tensor."""
        img = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        grad = gradient_magnitude(img)

        assert grad.shape == (2, 2)
        assert isinstance(grad, torch.Tensor)

    def test_3d_channel_axis_0(self) -> None:
        """Test gradient magnitude for 3D array with channel_axis=0."""
        # Create a 3D array with shape (C, H, W) = (2, 3, 3)
        img = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            ],
            dtype=np.float32,
        )
        grad = gradient_magnitude(img, channel_axis=0)

        # Expected output shape: (C, H-1, W-1) = (2, 2, 2)
        assert grad.shape == (2, 2, 2)
        assert isinstance(grad, np.ndarray)

    def test_3d_channel_axis_minus_1(self) -> None:
        """Test gradient magnitude for 3D array with channel_axis=-1."""
        # Create a 3D array with shape (H, W, C) = (3, 3, 2)
        img = np.array(
            [
                [[1, 9], [2, 8], [3, 7]],
                [[4, 6], [5, 5], [6, 4]],
                [[7, 3], [8, 2], [9, 1]],
            ],
            dtype=np.float32,
        )
        grad = gradient_magnitude(img, channel_axis=-1)

        # Expected output shape: (H-1, W-1, C) = (2, 2, 2)
        assert grad.shape == (2, 2, 2)
        assert isinstance(grad, np.ndarray)

    def test_3d_channel_axis_2(self) -> None:
        """Test gradient magnitude for 3D array with channel_axis=2."""
        # Create a 3D array with shape (H, W, C) = (3, 3, 2)
        img = np.array(
            [
                [[1, 9], [2, 8], [3, 7]],
                [[4, 6], [5, 5], [6, 4]],
                [[7, 3], [8, 2], [9, 1]],
            ],
            dtype=np.float32,
        )
        grad = gradient_magnitude(img, channel_axis=2)

        # Expected output shape: (H-1, W-1, C) = (2, 2, 2)
        assert grad.shape == (2, 2, 2)
        assert isinstance(grad, np.ndarray)

    def test_device_cpu(self) -> None:
        """Test gradient magnitude with CPU device."""
        img = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        grad = gradient_magnitude(img, device="cpu")

        assert isinstance(grad, torch.Tensor)
        assert grad.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self) -> None:
        """Test gradient magnitude with CUDA device."""
        img = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        grad = gradient_magnitude(img, device="cuda")

        assert isinstance(grad, torch.Tensor)
        assert grad.device.type == "cuda"

    def test_dtype_float64(self) -> None:
        """Test gradient magnitude with float64 dtype."""
        img_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        grad_np = gradient_magnitude(img_np, dtype=torch.float64)

        assert isinstance(grad_np, np.ndarray)
        assert grad_np.dtype == np.float64

        img_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        grad_tensor = gradient_magnitude(img_tensor, dtype=torch.float64)

        assert isinstance(grad_tensor, torch.Tensor)
        assert grad_tensor.dtype == torch.float64

    def test_invalid_ndim(self) -> None:
        """Test error for invalid number of dimensions."""
        # 1D array should raise error
        img = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
            gradient_magnitude(img)

        # 4D array should raise error
        img = np.ones((2, 3, 4, 5))
        with pytest.raises(ValueError, match="Input image must be 2D or 3D"):
            gradient_magnitude(img)

    def test_invalid_channel_axis(self) -> None:
        """Test error for invalid channel_axis."""
        img = np.ones((3, 4, 5))
        with pytest.raises(ValueError, match="Invalid channel_axis"):
            gradient_magnitude(img, channel_axis=1)

    def test_edge_case_small_image(self) -> None:
        """Test gradient magnitude for 2x2 image."""
        img = np.array([[1, 2], [3, 4]], dtype=np.float32)
        grad = gradient_magnitude(img)

        # Expected output shape: (1, 1)
        assert grad.shape == (1, 1)

    def test_gradient_values_known_case(self) -> None:
        """Test gradient values with a known case."""
        # Create a simple gradient in x-direction only
        img = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float32)
        grad = gradient_magnitude(img)

        # grad_x = [[1, 1], [1, 1]], grad_y = [[0, 0], [0, 0]]
        # grad_magnitude = sqrt(1^2 + 0^2) = 1
        np.testing.assert_allclose(grad, 1.0, rtol=1e-5)


class TestPercentileValueRange:
    """Test percentile_range function."""

    def test_basic_numpy_array(self) -> None:
        """Test basic functionality with numpy array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 10.0
        assert isinstance(vmin, float)
        assert isinstance(vmax, float)

    def test_torch_tensor(self) -> None:
        """Test with torch tensor."""
        data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 10.0

    def test_percentile_values(self) -> None:
        """Test with different percentile values."""
        data = np.arange(0, 101)  # 0 to 100
        vmin, vmax = percentile_range(data, min_percent=10, max_percent=90, symmetric=False)

        assert vmin == 10.0
        assert vmax == 90.0

    def test_symmetric_true(self) -> None:
        """Test symmetric bounds."""
        data = np.array([-5, -3, -1, 1, 3, 8])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=True)

        # Max absolute value is 8
        assert vmin == -8.0
        assert vmax == 8.0

    def test_symmetric_false(self) -> None:
        """Test non-symmetric bounds."""
        data = np.array([-5, -3, -1, 1, 3, 8])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == -5.0
        assert vmax == 8.0

    def test_with_nan_values(self) -> None:
        """Test with NaN values - should be ignored."""
        data = np.array([1, 2, np.nan, 4, 5, np.nan])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 5.0

    def test_with_inf_values(self) -> None:
        """Test with inf values - should be ignored."""
        data = np.array([1, 2, np.inf, 4, 5, -np.inf])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 5.0

    def test_2d_array(self) -> None:
        """Test with 2D array."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 9.0

    def test_list_input(self) -> None:
        """Test with list input."""
        data = [1, 2, 3, 4, 5]
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=100, symmetric=False)

        assert vmin == 1.0
        assert vmax == 5.0

    def test_percentiles_swapped(self) -> None:
        """Test when min_percent > max_percent - should be sorted."""
        data = np.arange(0, 101)
        vmin, vmax = percentile_range(data, min_percent=90, max_percent=10, symmetric=False)

        # Should be sorted internally
        assert vmin == 10.0
        assert vmax == 90.0

    def test_invalid_min_percent_negative(self) -> None:
        """Test error for negative min_percent."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be within"):
            percentile_range(data, min_percent=-10)

    def test_invalid_min_percent_too_large(self) -> None:
        """Test error for min_percent > 100."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be within"):
            percentile_range(data, min_percent=110)

    def test_invalid_max_percent_negative(self) -> None:
        """Test error for negative max_percent."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be within"):
            percentile_range(data, max_percent=-10)

    def test_invalid_max_percent_too_large(self) -> None:
        """Test error for max_percent > 100."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be within"):
            percentile_range(data, max_percent=110)

    def test_all_non_finite_values(self) -> None:
        """Test error when all values are non-finite."""
        data = np.array([np.nan, np.inf, -np.inf])
        with pytest.raises(ValueError, match="at least one finite value"):
            percentile_range(data)

    def test_non_numeric_data(self) -> None:
        """Test error for non-numeric data."""
        data = ["a", "b", "c"]
        with pytest.raises(ValueError, match="could not convert"):
            percentile_range(data)

    def test_torch_tensor_with_gradient(self) -> None:
        """Test torch tensor with gradient tracking."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        vmin, vmax = percentile_range(data, symmetric=False)

        # Should work and detach properly
        assert vmin == 1.0
        assert vmax == 5.0

    def test_symmetric_with_percentiles(self) -> None:
        """Test symmetric bounds with custom percentiles."""
        data = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        vmin, vmax = percentile_range(
            data, min_percent=20, max_percent=80, symmetric=True
        )

        # 20th percentile: -6, 80th percentile: 6
        # Max abs is 6
        assert vmin == -6.0
        assert vmax == 6.0

    def test_edge_case_single_value(self) -> None:
        """Test with single value."""
        data = np.array([42.0])
        vmin, vmax = percentile_range(data, symmetric=False)

        assert vmin == 42.0
        assert vmax == 42.0

    def test_edge_case_all_same_values(self) -> None:
        """Test with all same values."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        vmin, vmax = percentile_range(data, symmetric=False)

        assert vmin == 5.0
        assert vmax == 5.0

    def test_percentile_boundary_values(self) -> None:
        """Test with boundary percentile values."""
        data = np.arange(0, 101)

        # 0th percentile
        vmin, vmax = percentile_range(data, min_percent=0, max_percent=0, symmetric=False)
        assert vmin == 0.0
        assert vmax == 0.0

        # 100th percentile
        vmin, vmax = percentile_range(data, min_percent=100, max_percent=100, symmetric=False)
        assert vmin == 100.0
        assert vmax == 100.0
