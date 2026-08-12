"""Build the optional C++/OpenMP geo2rdr shared library.

The built library is loaded at runtime by
:func:`faninsar.processing.geometry.geo2rdr_backends.cpp_geo2rdr`. When the
library or a suitable compiler is unavailable, the Torch backend is used
instead, so this build step is strictly optional.

Usage::

    uv run python -m faninsar.processing.geometry.cpp.build_geo2rdr_cpp

"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _library_extension() -> str:
    """Return the platform shared-library extension."""
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _find_compiler() -> str | None:
    """Return the first available C++ compiler executable."""
    if sys.platform == "win32":
        for candidate in ("cl", "clang-cl", "g++"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None
    for candidate in ("g++", "clang++", "c++"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def build(
    source: Path | None = None,
    output: Path | None = None,
) -> Path:
    """Compile the C++ library and return its path.

    Parameters
    ----------
    source : Path or None, optional
        Path to ``geo2rdr.cpp``; defaults to the sibling file.
    output : Path or None, optional
        Output library path; defaults to ``cpp/lib`` next to the source.

    Returns
    -------
    Path
        Compiled shared library path.

    Raises
    ------
    RuntimeError
        If no compiler is available or compilation fails.

    """
    source = source or Path(__file__).resolve().parent / "geo2rdr.cpp"
    output = output or Path(__file__).resolve().parent / "lib" / (
        "libgeo2rdr" + _library_extension()
    )
    compiler = _find_compiler()
    if compiler is None:
        message = "No C++ compiler found; install g++/clang or use the Torch backend."
        raise RuntimeError(message)

    output.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        command = [
            compiler,
            "/O2",
            "/LD",
            "/EHsc",
            str(source),
            f"/Fe:{output}",
        ]
    else:
        command = [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(output),
        ]
        with_openmp = [*command, "-fopenmp"]
        try:
            subprocess.run(with_openmp, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(command, check=True, capture_output=True)
    return output


def main() -> None:
    """Build the shared library and print its path."""
    try:
        path = build()
    except RuntimeError as error:
        sys.stderr.write(str(error) + "\n")
        raise SystemExit(1) from error
    sys.stdout.write(f"Built geo2rdr C++ library: {path}\n")


if __name__ == "__main__":
    main()
