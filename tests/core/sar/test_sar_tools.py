import numpy as np
import pytest
from faninsar.core.sar_tools import Looks, coerce_looks, multi_look


class TestLooks:
    """Test the range-first Looks value."""

    def test_range_first_attributes(self):
        """Looks exposes range first and computes nominal looks."""
        looks = Looks(10, 2)
        assert looks.range == 10
        assert looks.azimuth == 2
        assert looks.nlooks == 20
        assert tuple(looks) == (10, 2)

    def test_root_import(self):
        """Looks is part of the root public surface."""
        from faninsar import Looks as RootLooks

        assert RootLooks is Looks

    @pytest.mark.parametrize(
        ("value", "error"),
        [((True, 2), TypeError), ((1.5, 2), TypeError), ((0, 2), ValueError), ((1, -1), ValueError)],
    )
    def test_rejects_invalid_factors(self, value, error):
        """Looks rejects bools, non-integers, and non-positive factors."""
        with pytest.raises(error):
            Looks(*value)

    def test_coerce_looks_accepts_tuple_and_value(self):
        """coerce_looks normalizes a range-first pair or a ready value."""
        looks = Looks(10, 2)
        assert coerce_looks(looks) is looks
        assert coerce_looks((10, 2)) == looks
        assert coerce_looks([np.int64(10), np.int64(2)]) == looks

    @pytest.mark.parametrize("value", [(1,), (1, 2, 3), ("a", "b"), 5])
    def test_coerce_looks_rejects_malformed_values(self, value):
        """coerce_looks rejects anything that is not a two-factor pair."""
        with pytest.raises(TypeError):
            coerce_looks(value)


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
        res = multi_look(data, Looks(2, 2))
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

        res = multi_look(data, Looks(2, 2))

        # Real part same as test_2d_real
        # Imag part: [[1, 2], [5, 6]] -> mean = 14/4 = 3.5
        expected_real = np.array([[2.5, 4.5], [10.5, 12.5]])
        expected_imag = np.array([[3.5, 5.5], [11.5, 13.5]])
        expected = expected_real + 1j * expected_imag

        np.testing.assert_allclose(res, expected)

    def test_non_divisible_shape(self):
        """Test multi_look with non-divisible shape."""
        data = np.ones((5, 5))
        res = multi_look(data, Looks(2, 2))
        assert res.shape == (2, 2)
        np.testing.assert_allclose(res, np.ones((2, 2)))

    def test_nd_support(self):
        """Test multi_look with multi-dimensional arrays."""
        data = np.ones((2, 4, 4))
        data[0] *= 1
        data[1] *= 2

        res = multi_look(data, Looks(2, 2))
        assert res.shape == (2, 2, 2)
        np.testing.assert_allclose(res[0], np.ones((2, 2)) * 1)
        np.testing.assert_allclose(res[1], np.ones((2, 2)) * 2)

    def test_looks_one(self):
        """Test multi_look with looks = 1."""
        data = np.random.rand(4, 4)
        res = multi_look(data, Looks(1, 1))
        np.testing.assert_allclose(res, data)
        # Ensure it's a copy
        assert res is not data

    def test_looks_different(self):
        """Test multi_look with different range and azimuth looks."""
        data = np.ones((6, 4))
        res = multi_look(data, Looks(2, 3))
        assert res.shape == (2, 2)
        np.testing.assert_allclose(res, np.ones((2, 2)))
