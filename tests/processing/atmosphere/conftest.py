"""Shared fixtures for the atmosphere package tests."""

from __future__ import annotations

import pytest
import torch

from faninsar.processing.atmosphere.config import IonosphereEstimationConfig

F0 = 1257.5e6
BANDWIDTH = 28.0e6


@pytest.fixture
def fine_mode_config() -> IonosphereEstimationConfig:
    """ALOS-2-like fine-mode thirds split (single-frequency L-band)."""
    return IonosphereEstimationConfig(
        f0=F0,
        freq_low=F0 - BANDWIDTH / 3,
        freq_high=F0 + BANDWIDTH / 3,
    )
