"""Tests for the Dask scheduling layer over Torch GPU kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from faninsar.processing.runtime.dask_gpu import (
    CUDA_PERFORMANCE_QUALIFIED,
    _schedule_carrier_multiply,
    _schedule_goldstein_filter,
    _schedule_multilook_interferogram,
    _schedule_multilook_real,
    _schedule_remove_topographic_phase,
    has_gpu_workers,
    run_carrier_multiply,
    run_carrier_multiply_at_points,
    run_esd_azimuth_shift,
    run_goldstein_filter,
    should_accelerate,
    use_gpu_resources,
    validate_cuda_worker_binding,
)
from faninsar.processing.interferometry.pair import (
    _block_reduce,
    form_interferogram,
    goldstein_filter,
)
from faninsar.processing.tops import TOPSCarrierModel, deramp


def _carrier_model() -> TOPSCarrierModel:
    return TOPSCarrierModel(
        radar_frequency_hz=5.405e9,
        slant_range_time0_s=0.00533,
        range_sampling_rate_hz=6.434e7,
        azimuth_time_interval_s=0.002055,
        doppler_centroid_hz=(100.0, -50.0, 10.0),
        doppler_t0_s=0.00534,
        fm_rate_hz_s=(-2300.0, 4.5e5, -7.9e7),
        fm_t0_s=0.00533,
        burst_sensing_time_s=0.0,
        burst_start_slant_range_time_s=0.00533,
        azimuth_steering_rate_hz_s=6500.0,
    )


def _random_slc(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)


def test_no_client_means_no_gpu_resources() -> None:
    """Without a Dask client the GPU-resource probe stays negative."""
    assert has_gpu_workers(None) is False
    assert use_gpu_resources("cpu") is False


def test_schedule_builders_are_internal_only() -> None:
    """Only exact-client run functions form the public scheduling API."""
    from faninsar.processing.runtime import dask_gpu

    for name in (
        "map_torch_blocks",
        "schedule_carrier_multiply",
        "schedule_goldstein_filter",
        "schedule_multilook_interferogram",
        "schedule_multilook_real",
        "schedule_remove_topographic_phase",
    ):
        assert name not in dask_gpu.__all__
        assert not hasattr(dask_gpu, name)


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "1"])
def test_malformed_gpu_resource_is_rejected(value: object) -> None:
    """Non-finite, boolean, and non-numeric GPU resources fail closed."""

    class MalformedClient:
        def scheduler_info(self) -> dict[str, object]:
            return {"workers": {"worker-0": {"resources": {"gpu": value}}}}

    assert has_gpu_workers(MalformedClient()) is False


class _FakeClient:
    def __init__(self, *, gpu_workers: bool) -> None:
        self.gpu_workers = gpu_workers

    def scheduler_info(self) -> dict[str, object]:
        resources = {"gpu": 1} if self.gpu_workers else {}
        return {"workers": {"worker-0": {"resources": resources}}}

    def compute(self, graph: Any) -> Any:
        self.computed_graph = graph
        return graph.compute()

    def gather(self, future: Any) -> Any:
        self.gathered_future = future
        return future

    def submit(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        self.submitted = (function, args, kwargs)
        return function(*args, **{k: v for k, v in kwargs.items() if k == "device"})

    def run(self, _function: Any) -> dict[str, str]:
        return {"worker-0": "0"}


class _BrokenClient:
    def scheduler_info(self) -> dict[str, object]:
        message = "scheduler unavailable"
        raise OSError(message)


def test_gpu_annotation_depends_only_on_cluster_resources() -> None:
    """A GPU-less or unreachable cluster never receives GPU annotations."""
    assert has_gpu_workers(_FakeClient(gpu_workers=True)) is True
    assert has_gpu_workers(_FakeClient(gpu_workers=False)) is False
    assert has_gpu_workers(_BrokenClient()) is False
    assert use_gpu_resources("auto", _FakeClient(gpu_workers=True)) is True
    assert use_gpu_resources("auto", _FakeClient(gpu_workers=False)) is False
    assert use_gpu_resources("cpu", _FakeClient(gpu_workers=True)) is False


def test_cuda_worker_binding_requires_one_visible_device() -> None:
    """GPU resource labels require an explicit one-device worker binding."""

    class BoundClient(_FakeClient):
        def run(self, _function: Any) -> dict[str, str]:
            return {"worker-0": "0"}

    class UnboundClient(_FakeClient):
        def run(self, _function: Any) -> dict[str, str]:
            return {"worker-0": "0,1"}

    assert validate_cuda_worker_binding(BoundClient(gpu_workers=True)) is True
    assert validate_cuda_worker_binding(UnboundClient(gpu_workers=True)) is False

    class NoBindingClient:
        def scheduler_info(self) -> dict[str, object]:
            return {"workers": {"worker-0": {"resources": {"gpu": 1}}}}

    assert validate_cuda_worker_binding(NoBindingClient()) is False


def test_cuda_worker_binding_rejects_capacity_above_one() -> None:
    """A worker capacity above one cannot prove one-process/one-device binding."""

    class CapacityClient(_FakeClient):
        def scheduler_info(self) -> dict[str, object]:
            return {"workers": {"worker-0": {"resources": {"gpu": 2}}}}

    assert validate_cuda_worker_binding(CapacityClient(gpu_workers=True)) is False


def test_cuda_worker_binding_rejects_shared_physical_identity() -> None:
    """Two workers cannot claim the same physical CUDA device."""

    class SharedClient:
        def scheduler_info(self) -> dict[str, object]:
            return {
                "workers": {
                    "worker-0": {"resources": {"gpu": 1}},
                    "worker-1": {"resources": {"gpu": 1}},
                }
            }

        def run(self, _function: Any) -> dict[str, dict[str, object]]:
            return {
                "worker-0": {"visible": "0", "identities": ["GPU-A"]},
                "worker-1": {"visible": "0", "identities": ["GPU-A"]},
            }

    assert validate_cuda_worker_binding(SharedClient()) is False


def test_explicit_cpu_never_uses_remote_gpu() -> None:
    """An explicit CPU request stays local while using the unified kernel."""
    assert use_gpu_resources("cpu", _FakeClient(gpu_workers=True)) is False
    assert should_accelerate("cpu", _FakeClient(gpu_workers=True)) is True


def test_cuda_dispatch_uses_stage_qualification_registry() -> None:
    """CUDA admission keeps Torch for unqualified stages as well as qualified ones."""
    client = _FakeClient(gpu_workers=True)
    assert {
        "goldstein_filter",
        "esd_azimuth_shift",
    } == CUDA_PERFORMANCE_QUALIFIED
    assert should_accelerate("auto", client, kernel="goldstein_filter") is True
    assert should_accelerate("auto", client, kernel="multilook_interferogram") is True
    assert should_accelerate("auto", client, kernel="carrier_multiply") is True
    assert should_accelerate("auto", client, kernel="remove_topographic_phase") is True


def test_local_cuda_dispatch_uses_stage_qualification_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local CUDA host runs the Torch kernel even for unqualified stages."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert should_accelerate("auto", None, kernel="multilook_interferogram") is True
    assert should_accelerate("auto", None, kernel="goldstein_filter") is True
    assert should_accelerate("auto", None, kernel="carrier_multiply") is True


@pytest.mark.parametrize("device", ["mps", "tpu", "npu"])
def test_explicit_unpublished_backend_fails_closed(device: str) -> None:
    """Unpublished backends fail closed instead of succeeding on host NumPy."""
    with pytest.raises(RuntimeError, match="published"):
        should_accelerate(device, _FakeClient(gpu_workers=True))


def test_scheduled_carrier_forwards_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker block kernels receive the caller's device string unchanged."""
    import faninsar.processing.torch_kernels as kernels

    observed_devices: list[str] = []
    original_phase = kernels.carrier_phase_at_points_torch
    original_multiply = kernels.carrier_multiply_torch

    def phase_wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return original_phase(*args, **kwargs)

    def multiply_wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return original_multiply(*args, **kwargs)

    monkeypatch.setattr(kernels, "carrier_phase_at_points_torch", phase_wrapper)
    monkeypatch.setattr(kernels, "carrier_multiply_torch", multiply_wrapper)
    graph = _schedule_carrier_multiply(
        _random_slc((16, 24), seed=70),
        _carrier_model(),
        sign=-1.0,
        device="cpu",
        chunk_rows=8,
    )
    graph.compute()
    assert observed_devices
    assert set(observed_devices) == {"cpu"}


def test_gpu_worker_graph_carries_resource_annotation() -> None:
    """A graph built for advertised GPU workers carries ``gpu: 1``."""
    graph = _schedule_carrier_multiply(
        _random_slc((8, 12), seed=73),
        _carrier_model(),
        sign=-1.0,
        device="auto",
        chunk_rows=4,
        client=_FakeClient(gpu_workers=True),
    )
    annotations = [
        layer.annotations
        for layer in graph.dask.layers.values()
        if layer.annotations is not None
    ]
    assert {"resources": {"gpu": 1}} in annotations


def test_gpu_worker_graph_forwards_fail_closed_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU-resource tasks receive explicit CUDA instead of ambiguous auto."""
    import faninsar.processing.torch_kernels as kernels

    observed_devices: list[str] = []

    def phase_wrapper(
        _model: object,
        rows: np.ndarray,
        columns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return np.zeros(np.broadcast_shapes(rows.shape, columns.shape))

    def multiply_wrapper(samples: np.ndarray, *_args: Any, **kwargs: Any) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return np.asarray(samples)

    monkeypatch.setattr(kernels, "carrier_phase_at_points_torch", phase_wrapper)
    monkeypatch.setattr(kernels, "carrier_multiply_torch", multiply_wrapper)
    graph = _schedule_carrier_multiply(
        _random_slc((8, 12), seed=79),
        _carrier_model(),
        sign=-1.0,
        device="auto",
        chunk_rows=4,
        client=_FakeClient(gpu_workers=True),
    )
    graph.compute()
    assert observed_devices
    assert set(observed_devices) == {"cuda"}


def test_gpu_worker_graph_preserves_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``cuda:N`` is forwarded unchanged when CUDA-N parsing is incomplete."""
    import faninsar.processing.torch_kernels as kernels

    observed_devices: list[str] = []

    def phase_wrapper(
        _model: object,
        rows: np.ndarray,
        columns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return np.zeros(np.broadcast_shapes(rows.shape, columns.shape))

    def multiply_wrapper(samples: np.ndarray, *_args: Any, **kwargs: Any) -> np.ndarray:
        observed_devices.append(kwargs["device"])
        return np.asarray(samples)

    monkeypatch.setattr(kernels, "carrier_phase_at_points_torch", phase_wrapper)
    monkeypatch.setattr(kernels, "carrier_multiply_torch", multiply_wrapper)
    graph = _schedule_carrier_multiply(
        _random_slc((8, 12), seed=86),
        _carrier_model(),
        sign=-1.0,
        device="cuda:1",
        chunk_rows=4,
        client=_FakeClient(gpu_workers=True),
    )
    graph.compute()
    assert observed_devices
    assert set(observed_devices) == {"cuda:1"}


def test_gpu_less_client_falls_back_without_building_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connected GPU-less client uses eager execution instead of hanging."""
    from faninsar.processing.runtime import dask_gpu

    def reject_graph(*_args: Any, **_kwargs: Any) -> None:
        message = "an unschedulable graph was built"
        raise AssertionError(message)

    monkeypatch.setattr(dask_gpu, "_schedule_carrier_multiply", reject_graph)
    samples = _random_slc((12, 20), seed=72)
    actual = run_carrier_multiply(
        samples,
        _carrier_model(),
        sign=-1.0,
        device="cpu",
        client=_FakeClient(gpu_workers=False),
    )
    np.testing.assert_allclose(actual, deramp(samples, _carrier_model()), atol=1e-6)


def test_run_uses_exact_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distributed execution computes and gathers through the selected client."""
    from faninsar.processing.runtime import dask_gpu

    class _FakeGraph:
        def compute(self) -> np.ndarray:
            return samples

    client = _FakeClient(gpu_workers=True)
    samples = _random_slc((12, 20), seed=74)
    monkeypatch.setattr(
        dask_gpu,
        "_schedule_goldstein_filter",
        lambda *_args, **_kwargs: _FakeGraph(),
    )
    result = run_goldstein_filter(
        samples,
        alpha=0.5,
        window=32,
        device="auto",
        client=client,
    )
    assert hasattr(client, "computed_graph")
    assert hasattr(client, "gathered_future")
    np.testing.assert_array_equal(result, samples)


def test_false_gpu_advertisement_fails_closed_on_selected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU worker claiming a GPU cannot silently execute the task on CPU."""
    import torch
    from dask.distributed import Client, LocalCluster

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        resources={"gpu": 1},
        dashboard_address=None,
    )
    client = Client(cluster, set_as_default=False)
    try:
        samples = _random_slc((12, 20), seed=77)
        with pytest.raises(RuntimeError, match=r"CUDA requested|binding preflight"):
            run_goldstein_filter(
                samples,
                alpha=0.5,
                window=32,
                device="auto",
                client=client,
            )
    finally:
        client.close()
        cluster.close()


def test_ambient_client_does_not_trigger_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ambient default client is ignored without explicit injection."""
    from dask.distributed import Client, LocalCluster

    from faninsar.processing.runtime import dask_gpu

    def reject_graph(*_args: Any, **_kwargs: Any) -> None:
        message = "ambient client triggered distributed graph construction"
        raise AssertionError(message)

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        resources={"gpu": 1},
        dashboard_address=None,
    )
    client = Client(cluster, set_as_default=True)
    monkeypatch.setattr(dask_gpu, "_schedule_carrier_multiply", reject_graph)
    samples = _random_slc((12, 20), seed=78)
    try:
        result = run_carrier_multiply(
            samples,
            _carrier_model(),
            sign=-1.0,
            device="auto",
        )
    finally:
        client.close()
        cluster.close()
    np.testing.assert_allclose(result, deramp(samples, _carrier_model()), atol=1e-6)


def test_esd_is_submitted_as_one_gpu_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESD uses one whole-array task on the explicitly selected GPU client."""
    import faninsar.processing.torch_kernels as kernels

    sentinel = object()
    monkeypatch.setattr(
        kernels,
        "esd_azimuth_shift_torch",
        lambda *_args, **_kwargs: sentinel,
    )
    client = _FakeClient(gpu_workers=True)
    reference = _random_slc((16, 16), seed=75)
    secondary = _random_slc((16, 16), seed=76)
    result = run_esd_azimuth_shift(
        reference,
        secondary,
        device="auto",
        client=client,
    )
    assert result is sentinel
    _, submitted_args, submitted_kwargs = client.submitted
    assert submitted_args[0] is reference
    assert submitted_args[1] is secondary
    assert submitted_kwargs["device"] == "cuda"
    assert submitted_kwargs["resources"] == {"gpu": 1}


def test_unpublished_esd_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit MPS request must not succeed on the NumPy ESD path."""
    from faninsar.processing.coreg import esd

    def reject_numpy(*_args: object) -> None:
        message = "unpublished device used NumPy ESD"
        raise AssertionError(message)

    monkeypatch.setattr(esd, "estimate_azimuth_shift_esd", reject_numpy)
    with pytest.raises(RuntimeError, match="published"):
        run_esd_azimuth_shift(
            _random_slc((16, 16), seed=84),
            _random_slc((16, 16), seed=85),
            device="mps",
            client=_FakeClient(gpu_workers=True),
        )


def test_geo_carrier_unqualified_stage_stays_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unqualified reramp stays on the admitted device instead of hopping to CPU."""
    from faninsar.processing.runtime import dask_gpu

    sentinel = np.ones((4, 5), dtype=np.complex64)
    observed: dict[str, str] = {}

    def capture(*_args: object, **kwargs: object) -> np.ndarray:
        device = kwargs.get("device")
        if isinstance(device, str):
            observed["device"] = device
        return sentinel

    monkeypatch.setattr(dask_gpu, "_carrier_multiply_at_points", capture)
    client = _FakeClient(gpu_workers=True)
    result = run_carrier_multiply_at_points(
        sentinel,
        _carrier_model(),
        np.zeros((4, 5)),
        np.zeros((4, 5)),
        centre_row=2.0,
        sign=1.0,
        device="auto",
        client=client,
    )
    np.testing.assert_array_equal(result, sentinel)
    assert not hasattr(client, "submitted")
    assert observed["device"] == "auto"


def test_invalid_chunk_rows_are_rejected() -> None:
    """Invalid chunk sizes fail before a graph can exhaust scheduler resources."""
    with pytest.raises(ValueError, match="chunk_rows must be >= 1"):
        _schedule_carrier_multiply(
            _random_slc((8, 12), seed=80),
            _carrier_model(),
            sign=-1.0,
            device="cpu",
            chunk_rows=0,
        )


def test_empty_multilook_shape_has_stable_metadata() -> None:
    """Look factors larger than the input produce empty typed graphs."""
    array = np.ones((2, 3), dtype=np.float32)
    graph = _schedule_multilook_real(array, 4, 5, device="cpu")
    result = np.asarray(graph.compute())
    assert result.shape == (0, 0)
    assert result.dtype == array.dtype


def test_schedule_carrier_multiply_matches_deramp() -> None:
    """Scheduled carrier multiply reproduces the NumPy deramp reference."""
    model = _carrier_model()
    samples = _random_slc((40, 64), seed=71)
    graph = _schedule_carrier_multiply(
        samples,
        model,
        sign=-1.0,
        device="cpu",
        chunk_rows=16,
    )
    result = np.asarray(graph.compute())
    expected = deramp(samples, model)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.shape == samples.shape


def test_schedule_multilook_interferogram_matches_numpy() -> None:
    """Chunked multilook scheduling matches the NumPy interferogram."""
    primary = _random_slc((64, 96), seed=81)
    secondary = _random_slc((64, 96), seed=82)
    graphs = _schedule_multilook_interferogram(
        primary,
        secondary,
        multilook=(2, 4),
        device="cpu",
        chunk_rows=32,
    )
    computed = tuple(np.asarray(graph.compute()) for graph in graphs)
    reference = form_interferogram(primary, secondary, multilook=(2, 4))
    assert computed[0].shape == reference.complex_ifg.shape
    np.testing.assert_allclose(computed[0], reference.complex_ifg, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(computed[1], reference.coherence, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        computed[2],
        reference.wrapped_phase,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(computed[3], reference.amplitude, rtol=1e-4, atol=1e-5)


def test_schedule_goldstein_matches_full_filter() -> None:
    """Halo-chunked Goldstein scheduling matches the full NumPy filter."""
    ifg = _random_slc((70, 60), seed=83)
    graph = _schedule_goldstein_filter(
        ifg,
        alpha=0.5,
        window=16,
        device="cpu",
        chunk_rows=24,
    )
    result = np.asarray(graph.compute())
    expected = goldstein_filter(ifg, alpha=0.5, window=16)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_schedule_remove_topographic_phase_matches_numpy() -> None:
    """Scheduled flatten multiply matches the NumPy reference."""
    ifg = _random_slc((48, 60), seed=91)
    topo = np.linspace(-1.0, 2.0, 48 * 60).reshape(48, 60)
    graph = _schedule_remove_topographic_phase(
        ifg,
        topo,
        device="cpu",
        chunk_rows=16,
    )
    result = np.asarray(graph.compute())
    expected = ifg * np.exp(-1j * topo)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_schedule_multilook_real_matches_numpy() -> None:
    """Scheduled real multilook matches the NumPy block reduce."""
    rng = np.random.default_rng(95)
    array = rng.normal(size=(60, 80)).astype(np.float32)
    graph = _schedule_multilook_real(
        array,
        3,
        5,
        device="cpu",
        chunk_rows=30,
    )
    result = np.asarray(graph.compute())
    expected = _block_reduce(array, 3, 5)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
