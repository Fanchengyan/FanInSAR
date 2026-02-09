import numpy as np
import pytest
from faninsar._core.sar.sar_tools import multi_look

class TestMultiLook:
    """Test class for multi_look function."""

    def test_2d_real(self):
        """Test multi_look with 2D real data."""
        data = np.arange(16).reshape(4, 4).astype(float)
        # [[ 0,  1,  2,  3],
        #  [ 4,  5,  6,  7],
        #  [ 8,  9, 10, 11],
        #  [12, 13, 14, 15]]
        
        # 2x2 multi-look
        res = multi_look(data, 2, 2)
        # Block (0,0): [[0, 1], [4, 5]] -> mean = 10/4 = 2.5
        # Block (0,1): [[2, 3], [6, 7]] -> mean = 18/4 = 4.5
        # Block (1,0): [[8, 9], [12, 13]] -> mean = 42/4 = 10.5
        # Block (1,1): [[10, 11], [14, 15]] -> mean = 50/4 = 12.5
        expected = np.array([[2.5, 4.5], [10.5, 12.5]])
        np.testing.assert_allclose(res, expected)

    def test_2d_complex(self):
        """Test multi_look with 2D complex data."""
        data = np.arange(16).reshape(4, 4).astype(complex)
        data += 1j * (np.arange(16).reshape(4, 4) + 1)
        
        res = multi_look(data, 2, 2)
        
        # Real part same as test_2d_real
        # Imag part: [[1, 2], [5, 6]] -> mean = 14/4 = 3.5
        expected_real = np.array([[2.5, 4.5], [10.5, 12.5]])
        expected_imag = np.array([[3.5, 5.5], [11.5, 13.5]])
        expected = expected_real + 1j * expected_imag
        
        np.testing.assert_allclose(res, expected)

    def test_non_divisible_shape(self):
        """Test multi_look with non-divisible shape."""
        data = np.ones((5, 5))
        res = multi_look(data, 2, 2)
        assert res.shape == (2, 2)
        np.testing.assert_allclose(res, np.ones((2, 2)))

    def test_nd_support(self):
        """Test multi_look with multi-dimensional arrays."""
        data = np.ones((2, 4, 4))
        data[0] *= 1
        data[1] *= 2
        
        res = multi_look(data, 2, 2)
        assert res.shape == (2, 2, 2)
        np.testing.assert_allclose(res[0], np.ones((2, 2)) * 1)
        np.testing.assert_allclose(res[1], np.ones((2, 2)) * 2)

    def test_looks_one(self):
        """Test multi_look with looks = 1."""
        data = np.random.rand(4, 4)
        res = multi_look(data, 1, 1)
        np.testing.assert_allclose(res, data)
        # Ensure it's a copy
        assert res is not data

    def test_looks_different(self):
        """Test multi_look with different azimuth and range looks."""
        data = np.ones((6, 4))
        res = multi_look(data, 3, 2)
        assert res.shape == (2, 2)
        np.testing.assert_allclose(res, np.ones((2, 2)))
