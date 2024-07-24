<h1 align="center">
<img src="https://raw.githubusercontent.com/Fanchengyan/FanInSAR/main/docs/source/_static/logo/logo.svg" width="400">
</h1><br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg)](https://doi.org/10.5281/zenodo.11398347) [![Documentation Status](https://readthedocs.org/projects/faninsar/badge/?version=latest)](https://faninsar.readthedocs.io/en/latest/?badge=latest)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
FanInSAR is a Fancy Interferometric Synthetic Aperture Radar (InSAR) time series analysis library written in Python. It aims to provide a foundational library for the development of InSAR algorithms, facilitating efficient processing of InSAR time series data by offering a Pythonic, fast, and flexible approach. FanInSAR’s high-level API abstracts the complex processing pipeline and conceals the low-level programming details, enabling users to focus on algorithm development. For researchers and developers aiming to rapidly implement their own InSAR algorithms, FanInSAR offers a quick start for their projects.


## Highlight Features

- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. Its API is designed to be simple and intuitive, by abstracting the complex processing pipeline and concealing the low-level programming details, which allows users to focus on algorithm development. For example, loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Filtering interferometric pairs can be performed by a time slice, similar to the ``pandas`` package.
- **Fast**: The core computation in FanInSAR is implemented using ``PyTorch``, a high-performance deep learning library. This allows for efficient processing on both CPU and GPU, enabling faster execution.
- **Flexible**: FanInSAR is designed to be flexible, allowing for customization and extension. Users can easily inherit classes or customize the processing pipeline for their specific needs.

## Installation

FanInSAR is a Python package, and requires ``Python >= 3.8``. You can install the latest release of FanInSAR using ``pip`` from the PyPI:

```bash
pip install FanInSAR
```

or from GitHub:

```bash
pip install git+https://github.com/Fanchengyan/FanInSAR.git
```

## Documentation

The detailed documentation is available at: <https://faninsar.readthedocs.io/en/latest/>

> :warning: **Note**
>
>FanInSAR is under active development and is currently in the alpha stage. Its API may change in the future until it reaches a stable version.

## Citation

> Fan, C., & Liu, L. (2024). FanInSAR: A Fancy InSAR time series library, in a Pythonic, fast, and flexible way (0.0.1). Zenodo. https://doi.org/10.5281/zenodo.11398347

```bib
@software{fan_2024_11398347,
  author       = {Fan, Chengyan and
                  Liu, Lin},
  title        = {{FanInSAR: A Fancy InSAR time series library, in a
                   Pythonic, fast, and flexible way}},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.11398347},
  url          = {https://doi.org/10.5281/zenodo.11398347}
}
```
