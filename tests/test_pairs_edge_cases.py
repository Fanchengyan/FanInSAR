"""Test edge cases for Pairs class, specifically 0 and 1 pairs scenarios."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from faninsar import Acquisition, DaySpan, Pair, Pairs


class TestPairsEdgeCases:
    """Test edge cases for Pairs class."""

    def test_empty_pairs_initialization(self):
        """Test initialization with empty pairs list."""
        empty_pairs = Pairs([])
        assert isinstance(empty_pairs, Pairs)
        assert len(empty_pairs) == 0
        assert empty_pairs.values.shape == (0, 2)
        assert empty_pairs.names.shape == (0,)

    def test_empty_pairs_properties(self):
        """Test properties of empty pairs."""
        empty_pairs = Pairs([])

        # Test basic properties
        assert len(empty_pairs.dates) == 0
        assert isinstance(empty_pairs.dates, Acquisition)
        assert len(empty_pairs.primary) == 0
        assert isinstance(empty_pairs.primary, Acquisition)
        assert len(empty_pairs.secondary) == 0
        assert isinstance(empty_pairs.secondary, Acquisition)
        assert len(empty_pairs.days) == 0
        assert isinstance(empty_pairs.days, DaySpan)

        # Test edge index
        assert empty_pairs.edge_index.shape == (0, 2)

        # Test shape
        assert empty_pairs.shape == (0, 2)

    def test_empty_pairs_string_methods(self):
        """Test string representation methods for empty pairs."""
        empty_pairs = Pairs([])

        # Test string representations
        assert str(empty_pairs) == "Pairs(0)"
        assert len(empty_pairs.names) == 0
        assert len(empty_pairs.to_names()) == 0

        # Test string formatting methods
        primary_str = empty_pairs.primary_string()
        secondary_str = empty_pairs.secondary_string()
        assert len(primary_str) == 0
        assert len(secondary_str) == 0

    def test_empty_pairs_operations(self):
        """Test operations on empty pairs."""
        empty_pairs1 = Pairs([])
        empty_pairs2 = Pairs([])

        # Test equality
        assert empty_pairs1 == empty_pairs2

        # Test addition (union)
        union_result = empty_pairs1 + empty_pairs2
        assert len(union_result) == 0

        # Test subtraction (difference)
        diff_result = empty_pairs1 - empty_pairs2
        assert len(diff_result) == 0

        # Test copy
        copied = empty_pairs1.copy()
        assert len(copied) == 0
        assert copied == empty_pairs1

    def test_empty_pairs_conversion_methods(self):
        """Test conversion methods for empty pairs."""
        empty_pairs = Pairs([])

        # Test to_dataframe
        df = empty_pairs.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ['primary', 'secondary', 'days']

        # Test to_numpy
        np_array = empty_pairs.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (0, 2)

        # Test to_matrix
        matrix = empty_pairs.to_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (0, 0)

    def test_empty_pairs_sort(self):
        """Test sorting empty pairs."""
        empty_pairs = Pairs([])

        # Test sort inplace
        result = empty_pairs.sort(inplace=True)
        assert result is None
        assert len(empty_pairs) == 0

        # Test sort not inplace
        sorted_pairs,_ = empty_pairs.sort(inplace=False)
        assert len(sorted_pairs) == 0

    def test_empty_pairs_gaps(self):
        """Test gap parsing for empty pairs."""
        empty_pairs = Pairs([])
        gaps = empty_pairs.parse_gaps()
        assert len(gaps) == 0

    def test_single_pair_initialization(self):
        """Test initialization with single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        assert isinstance(single_pair, Pairs)
        assert len(single_pair) == 1
        assert single_pair.values.shape == (1, 2)
        assert single_pair.names.shape == (1,)

    def test_single_pair_properties(self):
        """Test properties of single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        # Test basic properties
        assert len(single_pair.dates) == 2
        assert len(single_pair.primary) == 1
        assert len(single_pair.secondary) == 1
        assert len(single_pair.days) == 1
        assert single_pair.days[0] == 14  # 15 - 1 = 14 days

        # Test edge index
        assert single_pair.edge_index.shape == (1, 2)

        # Test shape
        assert single_pair.shape == (1, 2)

    def test_single_pair_string_methods(self):
        """Test string representation methods for single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        # Test string representations
        assert str(single_pair) == "Pairs(1)"
        assert len(single_pair.names) == 1
        assert single_pair.names[0] == "20200101_20200115"

        # Test to_names
        names = single_pair.to_names()
        assert len(names) == 1
        assert names[0] == "20200101_20200115"

        # Test with prefix
        names_with_prefix = single_pair.to_names(prefix="test")
        assert names_with_prefix[0] == "test_20200101_20200115"

    def test_single_pair_operations(self):
        """Test operations on single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair1 = Pairs([(date1, date2)])
        single_pair2 = Pairs([(date1, date2)])

        # Test equality
        assert single_pair1 == single_pair2

        # Test copy
        copied = single_pair1.copy()
        assert len(copied) == 1
        assert copied == single_pair1

    def test_single_pair_conversion_methods(self):
        """Test conversion methods for single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        # Test to_dataframe
        df = single_pair.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ['primary', 'secondary', 'days']

        # Test to_numpy
        np_array = single_pair.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (1, 2)

        # Test to_matrix
        matrix = single_pair.to_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (1, 1)  # 1 pair, 1 interval (2 dates - 1)

    def test_single_pair_indexing(self):
        """Test indexing operations on single pair."""
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        # Test integer indexing
        pair = single_pair[0]
        assert isinstance(pair, Pair)
        assert pair.name == "20200101_20200115"

        # Test slice indexing
        sliced = single_pair[0:1]
        assert isinstance(sliced, Pairs)
        assert len(sliced) == 1

    def test_from_names_empty(self):
        """Test creating Pairs from empty names list."""
        empty_pairs = Pairs.from_names([])
        assert len(empty_pairs) == 0

    def test_from_names_single(self):
        """Test creating Pairs from single name."""
        single_pairs = Pairs.from_names(["20200101_20200115"])
        assert len(single_pairs) == 1
        assert single_pairs.names[0] == "20200101_20200115"

    def test_mixed_operations(self):
        """Test operations between empty, single, and multiple pairs."""
        empty_pairs = Pairs([])
        date1 = datetime(2020, 1, 1)
        date2 = datetime(2020, 1, 15)
        single_pair = Pairs([(date1, date2)])

        # Test empty + single
        result1 = empty_pairs + single_pair
        assert len(result1) == 1

        # Test single + empty
        result2 = single_pair + empty_pairs
        assert len(result2) == 1

        # Test single - empty
        result3 = single_pair - empty_pairs
        assert len(result3) == 1

        # Test empty - single
        result4 = empty_pairs - single_pair
        assert len(result4) == 0

    def test_edge_case_date_ordering(self):
        """Test that dates are properly ordered in pairs."""
        # Test with reversed dates
        date1 = datetime(2020, 1, 15)  # Later date first
        date2 = datetime(2020, 1, 1)   # Earlier date second
        pairs = Pairs([(date1, date2)])

        # Should be automatically reordered
        assert pairs.primary[0] == pd.Timestamp(date2)  # Earlier date becomes primary
        assert pairs.secondary[0] == pd.Timestamp(date1)  # Later date becomes secondary
        assert pairs.names[0] == "20200101_20200115"  # Name should reflect correct order
