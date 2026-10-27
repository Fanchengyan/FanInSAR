"""Stack coregistration defaults and all-burst selection.

NESD is opt-in: tests that exercise measure/invert/apply pass
``coreg_mode="nesd"`` and note why.  The production default is geometry.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from faninsar.missions.s1 import S1Stack
from faninsar.stack import Stack, StackConfig
from faninsar.stack import config as stack_config


def _two_date_safe_paths(tmp_path: Path) -> list[Path]:
    """Create two empty SAFE-like directories for Stack construction."""
    paths: list[Path] = []
    for day in ("20160101", "20160113"):
        path = tmp_path / f"S1A_IW_SLC__1SDV_{day}T000000_{day}T000001.SAFE"
        path.mkdir()
        paths.append(path)
    return paths


def test_stack_config_defaults_are_geometry_all_swaths_all_bursts(
    tmp_path: Path,
) -> None:
    """StackConfig defaults to geometry with every sub-swath and burst selected."""
    cfg = StackConfig(work_dir=tmp_path, activation_mode="reference")
    assert cfg.coreg_mode == "geometry"
    assert cfg.swaths is None
    assert cfg.bursts is None
    assert cfg.esd_coherence_threshold == 0.85
    assert cfg.ampcor_snr_threshold == 6.0


def test_stack_config_rejects_pair_and_network_coreg_mode(tmp_path: Path) -> None:
    """coreg_mode is geometry or nesd; pair and network are gone."""
    for mode in ("pair", "network"):
        with pytest.raises(ValueError, match="geometry.*nesd"):
            StackConfig(
                work_dir=tmp_path,
                activation_mode="reference",
                coreg_mode=mode,  # type: ignore[arg-type]
            )


def test_stack_config_removes_esd_method() -> None:
    """EsdMethod and esd_method are removed without a compatibility shim."""
    assert not hasattr(stack_config, "EsdMethod")
    assert "esd_method" not in StackConfig.__dataclass_fields__
    assert "EsdMethod" not in stack_config.__all__


def test_from_safes_keyword_defaults_are_geometry_all_swaths_all_bursts(
    tmp_path: Path,
) -> None:
    """Stack._from_safes and S1Stack.from_safes default to geometry all bursts."""
    stack_params = inspect.signature(Stack._from_safes).parameters
    assert stack_params["coreg_mode"].default == "geometry"
    assert stack_params["swaths"].default is None
    assert stack_params["bursts"].default is None
    assert "esd_method" not in stack_params
    assert stack_params["esd_coherence_threshold"].default == 0.85
    assert stack_params["ampcor_snr_threshold"].default == 6.0

    s1_params = inspect.signature(S1Stack.from_safes).parameters
    assert s1_params["coreg_mode"].default == "geometry"
    assert s1_params["swaths"].default is None
    assert s1_params["bursts"].default is None

    stack = S1Stack.from_safes(
        _two_date_safe_paths(tmp_path),
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    assert stack.config.coreg_mode == "geometry"
    assert stack.config.swaths is None
    assert stack.config.bursts is None
    assert stack.config.esd_coherence_threshold == 0.85
    assert stack.config.ampcor_snr_threshold == 6.0


def test_burst_kwargs_pass_none_bursts_through(tmp_path: Path) -> None:
    """Default burst kwargs keep None rather than rewriting to first-burst IW1."""
    stack = S1Stack.from_safes(
        _two_date_safe_paths(tmp_path),
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    kwargs = stack._burst_kwargs()
    assert kwargs["swaths"] is None
    assert kwargs["bursts"] is None


def test_coreg_resume_identity_serializes_none_swaths(tmp_path: Path) -> None:
    """All-swath selection serializes without list(None) and is not first-burst."""
    stack = S1Stack.from_safes(
        _two_date_safe_paths(tmp_path),
        work_dir=tmp_path / "out",
        activation_mode="reference",
    )
    digest_all = stack._coreg_resume_identity(
        "20160113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert len(digest_all) == 64

    stack.config.swaths = ("IW1",)
    stack.config.bursts = {"IW1": [0]}
    digest_first_burst = stack._coreg_resume_identity(
        "20160113",
        misreg_az_px=0.0,
        misreg_rg_px=0.0,
    )
    assert digest_all != digest_first_burst
