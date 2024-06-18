import csv
from pathlib import Path

import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

install_requires = [
    "geopandas",
    "pandas",
    "numpy",
    "torch",
    "matplotlib",
    "data_downloader",
    "rasterio >= 1.0.26",
    "xarray",
    "psutil",
    "rioxarray",
    "netcdf4",
    "h5netcdf",
    "tqdm",
    "rtree",
]

with open("faninsar/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue


def get_scm_files():
    """
    Returns a list of all files in the SCM directory and its subdirectories.
    """
    scm_dir = Path("faninsar/cmaps")
    scm_files = []
    for file_path in scm_dir.glob("**/*"):
        if file_path.is_file():
            scm_files.append("/".join(list(file_path.parts[1:])))
    return scm_files


setuptools.setup(
    name="FanInSAR",
    version=version,
    author="Fancy",
    author_email="fanchy14@lzu.edu.cn",
    description="A fantastic InSAR processing library, in a more pythonic way, to accelerate your InSAR processing workflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fanchengyan/FanInSAR",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    package_data={"faninsar": get_scm_files()},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
