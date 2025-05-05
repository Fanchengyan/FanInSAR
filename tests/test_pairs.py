from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from faninsar import Pairs, PairsFactory, DaySpan


class TestPairs:
    @pytest.fixture
    def sample_pairs(self):
        dates = pd.date_range("20130101", "20231231").values
        n = len(dates)
        pair_ls = []
        for i in range(5):
            np.random.seed(i)
            pair_ls.append(dates[np.random.randint(0, n, 2)])
        return Pairs(pair_ls)

    def test_init(self, sample_pairs):
        assert isinstance(sample_pairs, Pairs)
        assert len(sample_pairs) == 5

    def test_str_repr(self, sample_pairs):
        assert str(sample_pairs) == "Pairs(5)"
        assert repr(sample_pairs) == sample_pairs._to_frame().__repr__()

    def test_eq(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        assert sample_pairs == copy_pairs

    def test_order(self, sample_pairs):
        values = sample_pairs.values
        assert sample_pairs == Pairs(values[::-1])

    def test_add_sub(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        union_pairs = sample_pairs + copy_pairs
        assert len(union_pairs) == len(sample_pairs)
        diff_pairs = sample_pairs - copy_pairs
        assert len(diff_pairs) == 0

    def test_getitem(self, sample_pairs):
        pair = sample_pairs[0]
        # assert isinstance(pair, Pair)
        sliced_pairs = sample_pairs[:3]
        assert isinstance(sliced_pairs, Pairs)
        assert len(sliced_pairs) == 3

    def test_values_property(self, sample_pairs):
        values = sample_pairs.values
        assert isinstance(values, np.ndarray)
        assert values.shape == (5, 2)

    def test_names_property(self, sample_pairs):
        names = sample_pairs.names
        assert isinstance(names, np.ndarray)
        assert names.shape == (5,)

    def test_dates_property(self, sample_pairs):
        dates = sample_pairs.dates
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(dates) == 10  # 5 pairs, 10 unique dates

    def test_days_property(self, sample_pairs):
        days = sample_pairs.days
        assert isinstance(days, DaySpan)
        assert days.shape == (5,)

    def test_primary_property(self, sample_pairs):
        primary = sample_pairs.primary
        assert isinstance(primary, pd.DatetimeIndex)
        assert len(primary) == 5

    def test_secondary_property(self, sample_pairs):
        secondary = sample_pairs.secondary
        assert isinstance(secondary, pd.DatetimeIndex)
        assert len(secondary) == 5

    def test_primary_string(self, sample_pairs):
        primary_str = sample_pairs.primary_string()
        assert isinstance(primary_str, pd.Index)
        assert len(primary_str) == 5

    def test_secondary_string(self, sample_pairs):
        secondary_str = sample_pairs.secondary_string()
        assert isinstance(secondary_str, pd.Index)
        assert len(secondary_str) == 5

    def test_edge_index_property(self, sample_pairs):
        edge_index = sample_pairs.edge_index
        assert isinstance(edge_index, np.ndarray)
        assert edge_index.shape == (5, 2)

    def test_shape_property(self, sample_pairs):
        shape = sample_pairs.shape
        assert isinstance(shape, tuple)
        assert shape == (5, 2)

    def test_from_names(self):
        names = ["20130101_20130102", "20130102_20130103"]
        pairs = Pairs.from_names(names)
        assert isinstance(pairs, Pairs)
        assert len(pairs) == 2

    def test_where(self, sample_pairs):
        names = sample_pairs.names.tolist()
        index = sample_pairs.where(names)
        assert isinstance(index, np.ndarray)
        assert len(index) == 5

    def test_intersect(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        intersection = sample_pairs.intersect(copy_pairs)
        assert isinstance(intersection, Pairs)
        assert len(intersection) == 5

    def test_union(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        union = sample_pairs.union(copy_pairs)
        assert isinstance(union, Pairs)
        assert len(union) == 5

    def test_difference(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        difference = sample_pairs.difference(copy_pairs)
        assert isinstance(difference, Pairs)
        assert len(difference) == 0

    def test_copy(self, sample_pairs):
        copy_pairs = sample_pairs.copy()
        assert isinstance(copy_pairs, Pairs)
        assert copy_pairs == sample_pairs

    def test_sort(self, sample_pairs):
        sorted_pairs, index = sample_pairs.sort(inplace=False)
        assert isinstance(sorted_pairs, Pairs)
        assert isinstance(index, np.ndarray)

    def test_to_names(self, sample_pairs):
        names = sample_pairs.to_names()
        assert isinstance(names, np.ndarray)
        assert names.shape == (5,)

    def test_to_frame(self, sample_pairs):
        frame = sample_pairs.to_frame()
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (5, 3)

    def test_to_matrix(self, sample_pairs):
        matrix = sample_pairs.to_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (5, len(sample_pairs.dates) - 1)

    def test_parse_gaps(self, sample_pairs):
        gaps = sample_pairs.parse_gaps()
        assert isinstance(gaps, np.ndarray)
