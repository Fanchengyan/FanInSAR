"""Network view construction and dispatch tests (PROPOSAL-0059)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from faninsar import Pairs
from faninsar.io.storage.ifg_store import write_ifg_artifact, write_unwrapped_artifact
from faninsar.missions.s1 import S1Stack
from faninsar.network import IncompleteNetworkProductError, Network, NetworkPathError

if TYPE_CHECKING:
    pass


def _stack_with_network(tmp_path: Path) -> S1Stack:
    """Build one small Stack with a complete IFG and unwrap product."""
    dates = ("20240101", "20240113")
    safe_paths = []
    for date in dates:
        path = tmp_path / f"S1A_IW_SLC__1SDV_{date}T000000.SAFE"
        path.mkdir()
        safe_paths.append(path)
    stack = S1Stack.from_safes(
        safe_paths,
        work_dir=tmp_path / "out",
        activation_mode="reference",
        pairs=Pairs.from_names(["20240101_20240113"]),
        multilook=(1, 1),
    ).prepare_scenes()
    phase = np.full((2, 3), 0.2, dtype=np.float32)
    store = write_ifg_artifact(
        stack.config.work_dir / "ifg" / "ml_1x1" / "20240101_20240113",
        pair=dates,
        looks=(1, 1),
        filter_name="none",
        filter_parameters={},
        source_manifest_digests={"primary": "a" * 64, "secondary": "b" * 64},
        complex_ifg=np.exp(1j * phase).astype(np.complex64),
        coherence=np.ones_like(phase),
    )
    stack.ifg_dirs = [store.root]
    write_unwrapped_artifact(
        store.root,
        unwrapped_phase=phase,
        connected_components=np.zeros((2, 3), dtype=np.int32),
        method="SpatialIRLS",
        method_parameters={},
        ifg_manifest_digest=store.write_id,
    )
    stack._refresh_network()
    return stack


def test_network_rejects_missing_directory(tmp_path: Path) -> None:
    """A path that is not a directory cannot construct a Network."""
    with pytest.raises(NetworkPathError):
        Network(tmp_path / "missing")


def test_network_rejects_incomplete_stack(tmp_path: Path) -> None:
    """A Stack work directory without a complete product set is refused."""
    work = tmp_path / "out"
    work.mkdir(parents=True)
    (work / "stack.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Exception):
        Network(work)


def test_from_stack_shares_the_live_product_index(tmp_path: Path) -> None:
    """A Network bound to a live Stack reflects its in-memory product index."""
    stack = _stack_with_network(tmp_path)

    network = Network._from_stack(stack)

    assert network.root == stack.config.work_dir
    assert network.network_generation_id == stack.network_generation_id
    assert network.interferograms.pairs().to_names().tolist() == ["20240101_20240113"]
    unwrapped = network.interferograms.open_stack("unw_phase")
    np.testing.assert_allclose(unwrapped.values[0], 0.2)


def test_network_public_surface_excludes_retired_aliases() -> None:
    """Retired manifest/generation/revision names are not exported."""
    import faninsar.network as public

    retired = {
        "LegacyNetworkLayoutError",
        "NetworkCurrentError",
        "NetworkGenerationError",
        "NetworkManifestError",
        "UnknownNetworkIndexTypeError",
    }
    assert retired.isdisjoint(public.__all__)
    assert all(not hasattr(public, name) for name in retired)


def test_incomplete_network_product_error_is_exported() -> None:
    """The one remaining construction error stays importable."""
    assert issubclass(IncompleteNetworkProductError, Exception)
