"""Test Baselines plotting functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from faninsar._core.sar.pairs import Pairs
from faninsar._core.sar.sar_tools import Baselines


class TestBaselinesPlot:
    """Test class for Baselines plotting."""

    @pytest.fixture
    def sample_pairs(self):
        """Create sample pairs for testing."""
        dates = pd.date_range("2020-01-01", periods=10, freq="12D")
        # Create pairs with varying temporal baselines
        pair_list = [
            (dates[0], dates[1]),  # 12 days
            (dates[0], dates[2]),  # 24 days
            (dates[1], dates[3]),  # 24 days
            (dates[2], dates[4]),  # 24 days
            (dates[0], dates[3]),  # 36 days
            (dates[3], dates[6]),  # 36 days
            (dates[4], dates[8]),  # 48 days
        ]
        return Pairs(pair_list)

    @pytest.fixture
    def sample_baselines(self, sample_pairs):
        """Create sample baselines for testing."""
        dates = sample_pairs.dates
        # Generate random baseline values
        np.random.seed(42)
        values = np.random.randn(len(dates)) * 100
        return Baselines(dates, values)

    def test_plot_without_cmap(self, sample_baselines, sample_pairs):
        """Test plotting without colormap (single color)."""
        fig, ax = plt.subplots()
        result = sample_baselines.plot(sample_pairs, ax=ax)

        assert result is not None
        assert result.ax == ax
        assert result.pairs_collection is not None
        assert result.colorbar is None  # No colorbar without cmap
        plt.close(fig)

    def test_plot_with_cmap(self, sample_baselines, sample_pairs):
        """Test plotting with colormap."""
        fig, ax = plt.subplots()
        result = sample_baselines.plot(sample_pairs, ax=ax, cmap="viridis")

        assert result is not None
        assert result.ax == ax
        assert result.colorbar is not None  # Colorbar should be created

        # Check if colorbar was created
        # The colorbar is attached to the figure
        assert len(fig.axes) == 2  # ax + colorbar ax
        plt.close(fig)

    def test_plot_with_custom_cmap(self, sample_baselines, sample_pairs):
        """Test plotting with custom colormap."""
        fig, ax = plt.subplots()
        result = sample_baselines.plot(
            sample_pairs,
            ax=ax,
            cmap="plasma",
        )
        # Customize colorbar after creation
        result.set_colorbar_label("Custom Label")

        assert result is not None
        assert result.colorbar is not None
        plt.close(fig)

    def test_plot_with_removed_pairs(self, sample_baselines, sample_pairs):
        """Test plotting with removed pairs."""
        # Remove some pairs
        pairs_removed = Pairs([sample_pairs[0]])

        fig, ax = plt.subplots()
        result = sample_baselines.plot(
            sample_pairs,
            pairs_removed=pairs_removed,
            ax=ax,
            cmap="coolwarm",
        )

        assert result is not None
        assert len(result.pairs_removed_lines) > 0
        plt.close(fig)

    def test_plot_empty_pairs(self, sample_baselines):
        """Test plotting with empty pairs."""
        empty_pairs = Pairs([])

        fig, ax = plt.subplots()
        result = sample_baselines.plot(empty_pairs, ax=ax, cmap="viridis")

        assert result is not None
        plt.close(fig)

    def test_plot_custom_kwargs(self, sample_baselines, sample_pairs):
        """Test plotting with customization via BaselinePlotResult."""
        fig, ax = plt.subplots()
        result = sample_baselines.plot(
            sample_pairs,
            ax=ax,
            cmap="jet",
        )
        # Customize after creation
        result.set_pairs_style(linewidths=2)
        result.set_acq_style(marker="s", markersize=8)
        result.set_xlabel("Custom X")
        result.set_ylabel("Custom Y")

        assert result is not None
        assert result.ax.get_xlabel() == "Custom X"
        assert result.ax.get_ylabel() == "Custom Y"
        plt.close(fig)

    def test_plot_without_legend(self, sample_baselines, sample_pairs):
        """Test plotting without legend."""
        fig, ax = plt.subplots()
        result = sample_baselines.plot(
            sample_pairs,
            ax=ax,
            cmap="viridis",
            legend=False,
        )

        assert result is not None
        assert result.ax.get_legend() is None
        plt.close(fig)

    def test_plot_chain_call(self, sample_baselines, sample_pairs):
        """Test plotting with chain call customization."""
        fig, ax = plt.subplots()
        result = (
            sample_baselines.plot(sample_pairs, ax=ax, cmap="plasma")
            .set_pairs_style(linewidths=2, alpha=0.8)
            .set_acq_style(markersize=10)
            .set_xlabel("Date")
            .set_ylabel("Baseline (m)")
            .set_colorbar_label("Days")
            .set_legend_labels(pairs="Valid", acquisitions="Acq")
        )

        assert result is not None
        assert result.ax.get_xlabel() == "Date"
        assert result.ax.get_ylabel() == "Baseline (m)"
        assert result.colorbar is not None
        plt.close(fig)
