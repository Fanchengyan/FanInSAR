"""Verify that wheels retain the native-v2 preparation source set."""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from os import environ
from pathlib import Path, PurePosixPath

NATIVE_SOURCE_SUFFIXES = frozenset({".cpp", ".h", ".cu", ".cuh"})
NATIVE_PACKAGE_PATH = PurePosixPath("faninsar", "processing", "geometry", "native_v2")
REQUIRED_NATIVE_V2_SOURCES = frozenset(
    {
        NATIVE_PACKAGE_PATH / "bindings.cpp",
        NATIVE_PACKAGE_PATH / "native_v2_abi.cpp",
        NATIVE_PACKAGE_PATH / "native_v2_abi.h",
        NATIVE_PACKAGE_PATH / "geo2rdr.cpp",
        NATIVE_PACKAGE_PATH / "rdr2geo.cpp",
        NATIVE_PACKAGE_PATH / "cuda" / "geo2rdr_cuda.cu",
        NATIVE_PACKAGE_PATH / "cuda" / "rdr2geo_tcn_cuda.cu",
        NATIVE_PACKAGE_PATH / "cuda" / "geometry_cuda.cuh",
    }
)


def _native_source_paths(repository_root: Path) -> set[PurePosixPath]:
    """Return the native-v2 source paths expected in a wheel.

    Parameters
    ----------
    repository_root : pathlib.Path
        Repository containing the source package.

    Returns
    -------
    set of pathlib.PurePosixPath
        Wheel-relative paths for every preparation source and header.

    """
    source_root = repository_root.joinpath(*NATIVE_PACKAGE_PATH.parts)
    return {
        PurePosixPath(NATIVE_PACKAGE_PATH, *path.relative_to(source_root).parts)
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in NATIVE_SOURCE_SUFFIXES
    }


def test_wheel_contains_exact_native_v2_preparation_source_set(
    tmp_path: Path,
) -> None:
    """Built distributions preserve native-v2 sources and installed resolution."""
    repository_root = Path(__file__).parents[2]
    distribution_directory = tmp_path / "dist"
    distribution_directory.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(distribution_directory),
        ],
        cwd=repository_root,
        check=True,
    )

    wheels = sorted(distribution_directory.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        actual = {
            PurePosixPath(name)
            for name in archive.namelist()
            if PurePosixPath(name).is_relative_to(NATIVE_PACKAGE_PATH)
            and PurePosixPath(name).suffix in NATIVE_SOURCE_SUFFIXES
        }

    sdists = sorted(distribution_directory.glob("*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0]) as archive:
        sdist_root = PurePosixPath(archive.getnames()[0]).parts[0]
        sdist_actual = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:])
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).parts[0] == sdist_root
            and PurePosixPath(member.name).suffix in NATIVE_SOURCE_SUFFIXES
            and PurePosixPath(member.name).is_relative_to(
                PurePosixPath(sdist_root, *NATIVE_PACKAGE_PATH.parts)
            )
        }

    assert set(REQUIRED_NATIVE_V2_SOURCES) == sdist_actual
    assert _native_source_paths(repository_root) == REQUIRED_NATIVE_V2_SOURCES
    assert actual == REQUIRED_NATIVE_V2_SOURCES

    wheel_venv = tmp_path / "wheel-venv"
    uv_executable = shutil.which("uv")
    assert uv_executable is not None
    subprocess.run(
        [uv_executable, "venv", "--python", sys.executable, str(wheel_venv)],
        cwd=tmp_path,
        check=True,
    )
    python_name = "python.exe" if sys.platform == "win32" else "python"
    wheel_python = (
        wheel_venv / ("Scripts" if sys.platform == "win32" else "bin") / python_name
    )
    subprocess.run(
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            str(wheel_python),
            "--no-deps",
            str(wheels[0]),
        ],
        cwd=tmp_path,
        check=True,
    )
    environment = environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    environment["FANINSAR_WHEEL_VENV"] = str(wheel_venv)
    subprocess.run(
        [
            str(wheel_python),
            "-c",
            """
from pathlib import Path
import os

from faninsar.processing.geometry.native_v2 import (
    GeometryOperation,
    NativeBackend,
    NativeBuildRequest,
    NativeBuilder,
)
import faninsar.processing.geometry.native_v2.builder as builder

assert Path(builder.__file__).resolve().is_relative_to(
    Path(os.environ["FANINSAR_WHEEL_VENV"]).resolve()
)

for backend in (NativeBackend.CPU, NativeBackend.CUDA):
    plan = NativeBuilder().plan(
        NativeBuildRequest(GeometryOperation.GEO2RDR, backend, platform="windows")
    )
    assert all(path.is_absolute() and path.exists() for path in plan.sources)
    assert all("native_v2" in Path(path).parts for path in plan.sources)
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
