<h1 align="center">
<img src="https://raw.githubusercontent.com/Fanchengyan/FanInSAR/main/docs/source/_static/logo/logo.svg" width="400">
</h1><br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg)](https://doi.org/10.5281/zenodo.11398347) [![Documentation Status](https://readthedocs.org/projects/faninsar/badge/?version=latest)](https://faninsar.readthedocs.io/en/latest/?badge=latest)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


**FanInSAR** is a **Fan**cy **In**terferometric **S**ynthetic **A**perture **R**adar (InSAR) scientific computing library written in Python. It provides reusable data, processing, Stack, Network, and time-series components for researchers and developers.



## Why FanInSAR?

Most existing community InSAR software adopts a workflow-oriented approach. While this lowers the entry barrier for new users, it often compromises flexibility and extensibility. For algorithm researchers, integrating new methods into these rigid workflows can be challenging, highlighting the need for a more adaptable framework for InSAR time series analysis.

**FanInSAR** is designed to bridge this gap with two clear product boundaries. A concrete **Stack** processes mission source data through interferogram generation and export. A **Network** opens an immutable, already processed generation for interferogram and time-series analysis. Lower-level scientific functions remain available from their owning packages without creating another workflow facade.

## Highlight Features

- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. For example, a series of well-known InSAR datasets are provided in the form of Python classes; loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Sampling values from an interferometric dataset is as easy as calling the ``query()`` method by passing the spatial (Points, BoundingBox, Polygons) and temporal (Pairs) queries. The warping process during sampling (such as reprojecting and resampling) is automatically handled by the library.
- **Fast**: The core computation in FanInSAR is implemented using ``PyTorch``, a high-performance deep learning library. This allows for efficient processing on both CPU and GPU, enabling faster execution.
- **Flexible**: Generic datasets, spatial and temporal queries, and samplers live under `faninsar.data`; geometry, coregistration, interferometry, geocoding, and unwrapping remain independently reusable processing modules.
- **Two clear workflows**: Use `S1Stack.from_safes(...)` or `NISARStack.from_rslc(...)` for source-to-product processing, and `Network.open(...)` for analysis of a completed product generation.

## API at a glance

```python
from faninsar import Acquisition, Network, Pair, Pairs
from faninsar.missions import NISARStack, S1Stack
from faninsar import data
```

- User-facing scientific values are available from `faninsar`.
- Concrete processing entry points are available from `faninsar.missions`.
- Reusable datasets, queries, and samplers are available from `faninsar.data`.
- Developer APIs live in focused modules such as `faninsar.processing.geometry`, `faninsar.processing.interferometry`, `faninsar.remote`, and `faninsar.io`.

## Installation

FanInSAR is a Python package and requires Python 3.11 or newer. You can install the latest release of FanInSAR using `pip` from PyPI:

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
