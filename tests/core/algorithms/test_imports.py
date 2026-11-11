"""Import closure for the toolbox and the existing packaging metadata seam."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(code: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);\n" + code,
            str(ROOT),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_plain_toolbox_import_adds_no_torch_or_family_loading():
    """Compare to the existing root import, which already loads Torch."""
    _run("""
import builtins
import importlib
import sys
import faninsar
before = set(sys.modules)
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    assert name != "torch" and not name.startswith("torch."), name
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
algorithms = importlib.import_module("faninsar.core.algorithms")
assert "interpolate2d" in algorithms.__all__
assert "LeastSquaresResult" in dir(algorithms)
new = set(sys.modules) - before
assert not any(name == "torch" or name.startswith("torch.") for name in new)
assert not any(name.startswith("faninsar.core.algorithms.") for name in new)
""")


def test_packaging_version_metadata_stays_torch_free():
    """Setuptools can still read the static version without importing the root."""
    _run("""
import importlib.abc
import sys
from setuptools.config.expand import read_attr
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoTorch())
assert read_attr("faninsar.__version__", root_dir=sys.argv[1]) == "0.1.dev0"
assert "torch" not in sys.modules
assert "faninsar" not in sys.modules
""")


def test_input_metadata_is_lazy_and_missing_torch_has_actionable_error():
    """Isolate this child from the unchanged eager legacy root dependencies."""
    _run("""
import importlib.abc
import pathlib
import sys
import types
import numpy as np
root = pathlib.Path(sys.argv[1])
for name, path in [("faninsar", root / "faninsar"),
                   ("faninsar.core", root / "faninsar" / "core")]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("test: Torch unavailable")
sys.meta_path.insert(0, NoTorch())
from faninsar.core.algorithms._validation import _inspect_array, _require_torch
info = _inspect_array(np.array([1., float("inf")], dtype=np.float32), name="source")
assert info.shape == (2,) and info.invalid is None
assert "torch" not in sys.modules
try:
    _require_torch()
except ImportError as exc:
    assert "requires Torch" in str(exc) and "install" in str(exc)
else:
    raise AssertionError("missing Torch was not reported")
""")
