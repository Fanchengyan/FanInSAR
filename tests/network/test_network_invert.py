"""Tests for the small Network time-series inversion interface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from faninsar import Pairs
from faninsar.network import Network
from faninsar.processing.interferometry.contracts import InterferogramStack
from faninsar.timeseries.models import LinearModel, TimeSeriesModels

if TYPE_CHECKING:
    from pathlib import Path


class _FakeInterferograms:
    """Minimal Dataset facade used through Network's public seam."""

    def __init__(self, pairs: Pairs) -> None:
        self._pairs = pairs

    def pairs(self) -> Pairs:
        """Return the complete pair cohort."""
        return self._pairs

    def open_stack(self, asset: str, *, pairs: Pairs) -> xr.DataArray:
        """Return one deterministic phase array for each requested pair."""
        if asset == "coherence":
            raise FileNotFoundError(asset)
        return xr.DataArray(
            np.zeros((len(pairs), 1, 1), dtype=np.float32),
            dims=("pair", "y", "x"),
            coords={"pair": list(pairs.names)},
        )


def _network(root: Path) -> tuple[Network, Pairs]:
    """Build a Network shell with a valid in-memory Dataset facade."""
    pairs = Pairs.from_names(
        [
            "20240101_20240113",
            "20240113_20240125",
            "20240101_20240125",
        ]
    )
    network = Network.__new__(Network)
    network._root = root
    network.manifest = {"generation_id": "generation-test"}
    network._network_generation_id = "generation-test"
    network._interferograms = _FakeInterferograms(pairs)
    return network, pairs


def test_invert_without_model_uses_sbas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting model selects the pure SBAS public path."""
    network, _ = _network(tmp_path)
    calls: dict[str, object] = {}

    def fake_sbas(pair_phases: dict[str, np.ndarray], **kwargs: object) -> str:
        calls["pair_phases"] = pair_phases
        calls["kwargs"] = kwargs
        return "sbas-result"

    monkeypatch.setattr("faninsar.timeseries.results.invert_unwrapped_pairs", fake_sbas)

    result = network.invert()

    assert result == "sbas-result"
    assert len(calls["pair_phases"]) == 3  # type: ignore[arg-type]


def test_invert_persists_canonical_timeseries_product(tmp_path: Path) -> None:
    """Network inversion commits a result as the Network's time-series product."""
    network, _ = _network(tmp_path)

    result = network.invert()

    canonical = tmp_path / "timeseries"
    assert result.canonical_path == canonical
    assert (canonical / "item.json").is_file()
    assert (canonical / "timeseries.zarr").is_dir()
    from faninsar.io.storage.product import open_product

    assert open_product(canonical).write_id


def test_repeated_and_subset_inversion_advances_canonical_generation(
    tmp_path: Path,
) -> None:
    """Repeated Network inversions replace the committed time-series product."""
    network, _ = _network(tmp_path)
    from faninsar.io.storage.product import open_product

    network.invert()
    canonical = tmp_path / "timeseries"
    first_write_id = open_product(canonical).write_id

    network.invert()
    second_write_id = open_product(canonical).write_id
    assert second_write_id != first_write_id

    subset = Pairs.from_names(["20240101_20240113", "20240113_20240125"])
    result = network.invert(pairs=subset)
    assert result.pair_ids == tuple(subset.to_names().tolist())
    assert open_product(canonical).write_id not in {first_write_id, second_write_id}


def test_invert_preserves_facade_spatial_metadata(tmp_path: Path) -> None:
    """Inversion binds common grid metadata from the interferogram facade."""
    network, _ = _network(tmp_path)
    network.interferograms.spatial_metadata = {  # type: ignore[attr-defined]
        "domain": "radar",
        "shape": (1, 1),
        "grid_identity": "grid-test",
        "radar_coordinates": {"range_spacing_m": 2.3},
    }

    result = network.invert()

    assert result.spatial is not None
    assert result.spatial.domain == "radar"
    assert result.spatial.shape == (1, 1)
    assert result.spatial.grid_identity == "grid-test"


def test_invert_rejects_facade_spatial_shape_mismatch(tmp_path: Path) -> None:
    """Inversion rejects metadata that cannot describe result raster arrays."""
    network, _ = _network(tmp_path)
    network.interferograms.spatial_metadata = {  # type: ignore[attr-defined]
        "domain": "radar",
        "shape": (2, 2),
        "grid_identity": "grid-test",
    }

    with pytest.raises(ValueError, match="spatial shape"):
        network.invert()


def test_legacy_analyze_nsbas_without_model_keeps_legacy_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deprecated solver selector must not silently invoke public SBAS."""
    network, pairs = _network(tmp_path)
    stack = InterferogramStack.from_unwrapped(
        stack_id="test",
        pairs=pairs,
        unwrapped=np.zeros((len(pairs), 1), dtype=np.float32),
    )
    monkeypatch.setattr(network, "to_ifg_stack", lambda **_: stack)

    def fake_nsbas(_stack: object, *, model: object, **_kwargs: object) -> tuple:
        assert model is None
        return (
            np.zeros((2, 1), dtype=np.float64),
            None,
            np.zeros((3, 1), dtype=np.float64),
            None,
        )

    invert_module = import_module("faninsar.timeseries.invert")
    monkeypatch.setattr(invert_module, "invert", fake_nsbas)

    result = network.analyze_time_series(solver="nsbas")

    assert result.metadata["method"] == "nsbas"


def test_invert_model_class_is_instantiated_with_selected_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model class is configured internally from the selected pairs."""
    network, pairs = _network(tmp_path)
    calls: dict[str, object] = {}
    stack = InterferogramStack.from_unwrapped(
        stack_id="test",
        pairs=pairs,
        unwrapped=np.zeros((len(pairs), 1), dtype=np.float32),
    )

    monkeypatch.setattr(network, "to_ifg_stack", lambda **_: stack)

    def fake_nsbas(
        received_stack: object, *, model: object, **_kwargs: object
    ) -> tuple:
        calls["stack"] = received_stack
        calls["model"] = model
        return (
            np.zeros((2, 1), dtype=np.float64),
            None,
            np.zeros((3, 1), dtype=np.float64),
            None,
        )

    invert_module = import_module("faninsar.timeseries.invert")
    monkeypatch.setattr(invert_module, "invert", fake_nsbas)

    result = network.invert(model=LinearModel)

    assert result.metadata["method"] == "nsbas"
    assert isinstance(calls["model"], LinearModel)
    assert np.array_equal(calls["model"].dates, pairs.dates)  # type: ignore[union-attr]


class _KeywordDatesModel(TimeSeriesModels):
    """Model whose public constructor requires keyword-only acquisition dates."""

    def __init__(self, *, dates: object) -> None:
        super().__init__(dates)
        self._G_br = np.ones((len(self.dates), 1), dtype=np.float32)
        self._param_names = ["constant"]


def test_invert_model_class_receives_dates_by_keyword(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model classes receive selected dates through the documented keyword."""
    network, pairs = _network(tmp_path)
    stack = InterferogramStack.from_unwrapped(
        stack_id="test",
        pairs=pairs,
        unwrapped=np.zeros((len(pairs), 1), dtype=np.float32),
    )
    monkeypatch.setattr(network, "to_ifg_stack", lambda **_: stack)

    def fake_nsbas(_stack: object, *, model: object, **_kwargs: object) -> tuple:
        assert isinstance(model, _KeywordDatesModel)
        return (
            np.zeros((2, 1), dtype=np.float64),
            None,
            np.zeros((3, 1), dtype=np.float64),
            None,
        )

    invert_module = import_module("faninsar.timeseries.invert")
    monkeypatch.setattr(invert_module, "invert", fake_nsbas)

    result = network.invert(model=_KeywordDatesModel)

    assert result.metadata["method"] == "nsbas"


def test_invert_rejects_model_instance_with_mismatched_dates(tmp_path: Path) -> None:
    """A configured model must describe exactly the selected acquisitions."""
    network, pairs = _network(tmp_path)
    model = LinearModel(pairs.dates[:-1])

    with pytest.raises(ValueError, match="dates"):
        network.invert(model=model)


def test_invert_rejects_invalid_model_with_configured_instance_hint(
    tmp_path: Path,
) -> None:
    """Invalid model values fail with a useful configured-instance hint."""
    network, _ = _network(tmp_path)

    with pytest.raises(TypeError, match="configured instance"):
        network.invert(model=object)


class _NeedsConfiguration(TimeSeriesModels):
    """Model class whose constructor cannot be inferred from acquisition dates."""

    def __init__(self, dates: object, _required: object) -> None:
        super().__init__(dates)
        self._G_br = np.empty((0, 0))
        self._param_names = []


def test_invert_model_class_constructor_error_suggests_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models needing extra domain inputs must be preconfigured by callers."""
    network, pairs = _network(tmp_path)
    stack = InterferogramStack.from_unwrapped(
        stack_id="test",
        pairs=pairs,
        unwrapped=np.zeros((len(pairs), 1), dtype=np.float32),
    )
    monkeypatch.setattr(network, "to_ifg_stack", lambda **_: stack)

    with pytest.raises(TypeError, match="configured instance"):
        network.invert(model=_NeedsConfiguration)
