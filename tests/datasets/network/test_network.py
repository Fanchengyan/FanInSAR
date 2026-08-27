"""Public path-based Network construction tests (PROPOSAL-0037)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from faninsar import Network as RootNetwork
from faninsar.datasets import Network
from faninsar.datasets.frame import Frame
from faninsar.datasets.network import (
    LegacyNetworkLayoutError,
    NetworkConstructionError,
    NetworkPathError,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_network_is_concrete_dataset_backed_frame(tmp_path: Path) -> None:
    """The public Network class mounts the existing concrete Frame product."""
    root = tmp_path / "network"
    (root / "geometry").mkdir(parents=True)

    network = Network(root)

    assert RootNetwork is Network
    assert isinstance(network, Network)
    assert isinstance(network, Frame)
    assert network.root == root
    assert network.geometry is not None
    assert network.interferograms is None
    assert repr(network).startswith("Network(root=")


def test_network_from_path_is_explicit_path_constructor(tmp_path: Path) -> None:
    """The named constructor has the same path-only contract."""
    root = tmp_path / "network"
    root.mkdir()

    network = Network.from_path(root)

    assert network.root == root


@pytest.mark.parametrize(
    "marker",
    [("ifg",), ("ifg_index.json",), ("interferograms/ifg_index.json",)],
)
def test_network_rejects_legacy_markers_before_frame_discovery(
    tmp_path: Path, marker: tuple[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy marker paths fail before the Frame implementation is entered."""
    root = tmp_path / "legacy"
    marker_name = marker[0]
    marker_path = root / marker_name
    if marker_path.suffix:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("{}", encoding="utf-8")
    else:
        marker_path.mkdir(parents=True)

    def fail_if_discovered(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "Frame discovery must not run for legacy layouts"
        raise AssertionError(msg)

    monkeypatch.setattr(Frame, "__init__", fail_if_discovered)

    with pytest.raises(LegacyNetworkLayoutError, match="Legacy") as exc_info:
        Network(root)

    assert marker_path in exc_info.value.markers


def test_network_missing_path_has_typed_and_filesystem_error(tmp_path: Path) -> None:
    """Missing roots use a typed construction error retaining FileNotFoundError."""
    with pytest.raises(NetworkPathError) as exc_info:
        Network(tmp_path / "missing")

    assert isinstance(exc_info.value, NetworkConstructionError)
    assert isinstance(exc_info.value, FileNotFoundError)


def test_frame_legacy_fallback_remains_unchanged(tmp_path: Path) -> None:
    """The compatibility behavior belongs to Frame and remains available."""
    root = tmp_path / "frame"
    (root / "ifg").mkdir(parents=True)

    frame = Frame(root)

    assert frame.interferograms is not None
    assert frame.interferograms.root == root / "ifg"
