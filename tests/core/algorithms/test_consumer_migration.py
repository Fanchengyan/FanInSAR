"""Public toolbox integration and intentionally retained domain consumer contracts."""

from __future__ import annotations

import ast
import subprocess
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

import faninsar
import faninsar.core
from faninsar.core import algorithms

PUBLIC_FAMILIES = {
    "interpolate2d": "sampling",
    "block_reduce2d": "reduction",
    "filter2d": "filtering",
    "polyval": "fitting",
    "fit_surface2d": "fitting",
    "PolynomialSurface": "fitting",
    "least_squares": "linalg",
    "LeastSquaresResult": "linalg",
}
UPPER_LAYERS = {
    "processing",
    "runtime",
    "missions",
    "timeseries",
    "stack",
    "network",
    "io",
    "remote",
    "data",
    "plotting",
    "cli",
}


@pytest.mark.parametrize(("name", "family"), PUBLIC_FAMILIES.items())
def test_all_integrated_public_exports_resolve_to_their_owning_family(
    name: str, family: str
) -> None:
    """All six operations and both result values must be present, without skips."""
    public = getattr(algorithms, name)
    owner = import_module(f"faninsar.core.algorithms.{family}")
    assert public is getattr(owner, name)
    assert callable(public)


def test_numerical_exports_do_not_expand_root_or_parent_core_namespaces() -> None:
    """Discovery of the child does not add numerical aliases to existing facades."""
    assert set(algorithms.__all__) == set(PUBLIC_FAMILIES)
    assert set(PUBLIC_FAMILIES).isdisjoint(faninsar.__all__)
    assert set(PUBLIC_FAMILIES).isdisjoint(faninsar.core.__all__)
    for name in PUBLIC_FAMILIES:
        assert not hasattr(faninsar, name)
        assert not hasattr(faninsar.core, name)


def test_toolbox_source_imports_no_upper_layer() -> None:
    """Check every integrated source file, including type-only and lazy imports."""
    root = Path(algorithms.__file__).resolve().parent
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = "faninsar.core.algorithms" + ".".join(
            ("", *path.relative_to(root).parts[:-1])
        )
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parents = package.split(".")
                    base = parents[: len(parents) - node.level + 1]
                    module = ".".join((*base, module)).rstrip(".")
                targets = [module]
                if module == "faninsar":
                    targets.extend(f"faninsar.{alias.name}" for alias in node.names)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Name, ast.Attribute))
                and (
                    node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                )
                in {"import_module", "__import__"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                targets = [node.args[0].value]
            for target in targets:
                parts = target.split(".")
                if (
                    len(parts) > 1
                    and parts[0] == "faninsar"
                    and parts[1] in UPPER_LAYERS
                ):
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: {target}"
                    )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("faninsar.processing.alg", ("detrend_surface",)),
        (
            "faninsar.processing.atmosphere.filter",
            ("gaussian_kernel_1d", "smooth_inverse_variance"),
        ),
        (
            "faninsar.processing.interferometry.phase_filter",
            ("BoxcarFilter", "GaussianFilter"),
        ),
        (
            "faninsar.processing.runtime.torch_kernels",
            ("multilook_real_torch",),
        ),
        (
            "faninsar.timeseries.solver",
            ("censored_lstsq", "batch_lstsq", "LstsqResult"),
        ),
        ("faninsar.processing.geometry.api", ("_sample_biquintic",)),
    ],
)
def test_retained_domain_entry_points_still_import(
    module: str, names: tuple[str, ...]
) -> None:
    """New core capabilities do not remove the old owning-domain entry points."""
    imported = import_module(module)
    for name in names:
        assert callable(getattr(imported, name))


def test_selected_numpy_oracle_definitions_preserve_source_and_ast() -> None:
    """Compare only named historical definitions, never whole mixed-source files."""
    root = Path(algorithms.__file__).resolve().parents[3]
    selected = {
        "faninsar/processing/geometry/api.py": (
            "_natural_spline_six",
            "_sample_biquintic",
        ),
        "faninsar/processing/geometry/dem.py": ("_natural_spline_six",),
        "faninsar/processing/interferometry/pair.py": (
            "_block_reduce",
        ),
        "faninsar/processing/coregistration/tops/deramp.py": ("_poly_eval",),
        "faninsar/processing/geocoding/coordinates.py": ("_offset_fields_numpy",),
    }
    if not (root / ".git").exists():
        pytest.skip("Historical definition preservation requires a Git source checkout")
    for relative, names in selected.items():
        baseline = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"460fc9dd93257baed9c935e563918e528ea2ff1f:{relative}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        current = (root / relative).read_text(encoding="utf-8")
        old_definitions = {
            node.name: node
            for node in ast.parse(baseline).body
            if isinstance(node, ast.FunctionDef)
        }
        new_definitions = {
            node.name: node
            for node in ast.parse(current).body
            if isinstance(node, ast.FunctionDef)
        }
        for name in names:
            assert ast.dump(old_definitions[name]) == ast.dump(new_definitions[name])
            assert ast.get_source_segment(baseline, old_definitions[name]) == (
                ast.get_source_segment(current, new_definitions[name])
            )


def test_existing_downstream_imports_keep_their_domain_owners() -> None:
    """Check actual consumer aliases/exports rather than mocking kernel internals."""
    solver = import_module("faninsar.timeseries.solver")
    timeseries = import_module("faninsar.timeseries")
    inversion = import_module("faninsar.timeseries.invert")
    for name in ("LstsqResult", "batch_lstsq", "censored_lstsq"):
        assert name in timeseries.__all__
        assert getattr(timeseries, name) is getattr(solver, name)
        assert getattr(inversion.NSBAS, name) is getattr(solver, name)
    assert solver.LstsqResult is not algorithms.LeastSquaresResult
    assert solver.censored_lstsq is not algorithms.least_squares

    atmosphere = import_module("faninsar.processing.atmosphere")
    filtering = import_module("faninsar.processing.atmosphere.filter")
    for name in ("gaussian_kernel_1d", "smooth_inverse_variance"):
        assert name in atmosphere.__all__
        assert getattr(atmosphere, name) is getattr(filtering, name)
    accessor = import_module("faninsar.data._adapters.xarray")
    domain_algorithms = import_module("faninsar.processing.alg")
    assert accessor._detrend_surface is domain_algorithms.detrend_surface


def test_numpy_public_calls_return_tensor_values_and_mathematical_results() -> None:
    """Exercise all operations through the stable namespace using small CPU data."""
    rows, cols = np.indices((3, 4), dtype=np.float64)
    source = 1.0 + 2.0 * cols - rows
    sampled = algorithms.interpolate2d(source, np.array([0.5]), np.array([1.5]))
    reduced = algorithms.block_reduce2d(source, (1, 2))
    filtered = algorithms.filter2d(source, method="boxcar", window=(1, 1))
    evaluated = algorithms.polyval(np.array([1.0, 2.0, 3.0]), np.array([2.0]))
    surface = algorithms.fit_surface2d(source)
    result = algorithms.least_squares(
        np.ones((2, 1)), np.array([[2.0, np.nan], [np.nan, 4.0]])
    )
    assert isinstance(surface, algorithms.PolynomialSurface)
    assert isinstance(result, algorithms.LeastSquaresResult)
    for value in (
        sampled,
        reduced,
        filtered,
        evaluated,
        surface.coefficients,
        surface.evaluate(rows, cols),
        result.solution,
        result.condition_number,
    ):
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float64
        assert value.device.type == "cpu"
    torch.testing.assert_close(sampled, torch.tensor([3.5], dtype=torch.float64))
    torch.testing.assert_close(filtered, torch.from_numpy(source))
    torch.testing.assert_close(evaluated, torch.tensor([17.0], dtype=torch.float64))
    torch.testing.assert_close(surface.evaluate(rows, cols), torch.from_numpy(source))
    torch.testing.assert_close(
        result.solution, torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    )
    assert result.n_observations.tolist() == [1, 1]


def test_core_reduction_defaults_are_not_legacy_finite_mean_defaults() -> None:
    """A shared Tensor leaf does not change the NumPy oracle's omission policy."""
    pair = import_module("faninsar.processing.interferometry.pair")
    source = np.array([[1.0, np.nan, 900.0], [3.0, 5.0, 900.0]])
    original = source.copy()
    legacy = pair._block_finite_mean(source, (2, 2))
    default = algorithms.block_reduce2d(source, (2, 2))
    compatible = algorithms.block_reduce2d(
        source, (2, 2), invalid="omit", remainder="drop"
    )
    assert isinstance(legacy, np.ndarray)
    np.testing.assert_array_equal(legacy, [[3.0]])
    assert bool(torch.isnan(default).all())
    torch.testing.assert_close(compatible, torch.from_numpy(legacy))
    np.testing.assert_array_equal(source, original)


def test_core_gaussian_minimum_support_and_retained_domain_identity() -> None:
    """The strict new API cannot silently replace sigma-zero or clamping policy."""
    filtering = import_module("faninsar.processing.atmosphere.filter")
    source = torch.ones((3, 3), dtype=torch.float64)
    source[1, 1] = torch.nan
    default = algorithms.filter2d(source, method="gaussian", sigma=(0.16, 0.16))
    explicit = algorithms.filter2d(
        source, method="gaussian", sigma=(0.16, 0.16), min_radius=1
    )
    single = algorithms.filter2d(
        source, method="gaussian", sigma=(0.16, 0.16), min_radius=0
    )
    torch.testing.assert_close(default, explicit, equal_nan=True)
    assert bool(torch.isfinite(default[1, 1]))
    assert bool(torch.isnan(single[1, 1]))
    weights = torch.ones_like(source)
    weights[0, 0] = -1.0
    legacy = filtering.smooth_inverse_variance(
        source, weights, sigma_y=0.0, sigma_x=0.0
    )
    assert bool(torch.isnan(legacy[0, 0]))
    assert bool(torch.isnan(legacy[1, 1]))
    torch.testing.assert_close(legacy[2, 2], source[2, 2])
    with pytest.raises(ValueError, match=r"sigma.*positive"):
        algorithms.filter2d(source, method="gaussian", sigma=(0.0, 0.0))
    with pytest.raises(ValueError, match=r"weights.*nonnegative"):
        algorithms.filter2d(
            source, method="gaussian", sigma=(0.16, 0.16), weights=weights
        )


def test_new_singular_value_diagnostics_do_not_relabel_legacy_proxies() -> None:
    """Keep the intentionally different rank cutoff and condition meanings visible."""
    solver = import_module("faninsar.timeseries.solver")
    design = np.diag([1.0, 1e-7])
    observations = np.array([1.0, 1e-7])
    legacy = solver.censored_lstsq(design, observations, device="cpu")
    result = algorithms.least_squares(design, observations)
    assert isinstance(legacy.solution, np.ndarray)
    assert legacy.solution.shape == (2, 1)
    assert legacy.rank.tolist() == [1]
    assert result.solution.shape == (2,)
    assert result.rank.item() == 2

    design = np.diag([1.0, 0.01])
    legacy = solver.censored_lstsq(design, np.array([1.0, 0.01]), device="cpu")
    result = algorithms.least_squares(design, np.array([1.0, 0.01]))
    assert legacy.condition_number.item() == pytest.approx(0.01)
    assert result.condition_number.item() == pytest.approx(100.0)
