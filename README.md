<h1 align="center">
<img src="https://raw.githubusercontent.com/Fanchengyan/FanInSAR/main/docs/source/_static/logo/logo.svg" width="400">
</h1><br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg)](https://doi.org/10.5281/zenodo.11398347) [![Documentation Status](https://readthedocs.org/projects/faninsar/badge/?version=latest)](https://faninsar.readthedocs.io/en/latest/?badge=latest)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# FanInSAR

**FanInSAR** (**F**oundational **A**PIs and **N**umerics for **InSAR**) is a Python scientific computing library for InSAR processing and analysis. Inspired by **NumPy, SciPy, and xarray**, it provides fundamental data structures, numerical algorithms, and intuitive Python APIs, enabling researchers to directly manipulate InSAR data, explore individual algorithms, and build custom workflows rather than rely on predefined processing pipelines.

By **unifying data, algorithms, and benchmarks**, FanInSAR aims to establish a common computational foundation for **flexible, interoperable, and reproducible InSAR research**.

## Core Components

### 1. Data Commons — Standardized Data Structures and Interoperability

Data Commons aims to provide a common data model for InSAR processing and analysis, together with standardized interfaces for data access, manipulation, and exchange.

Its key capabilities include:

- **Common InSAR Data Model:** Standardized, metadata-aware data structures covering spatial and temporal information, phase conventions, acquisition geometry, reference systems, quality indicators, and processing provenance.
- **Bidirectional Data Interoperability:** Python APIs for importing, converting, and exporting supported data products across existing InSAR software packages, including ISCE, GMTSAR, and MintPy.
- **Extensible Data Adapters:** A plugin mechanism allowing developers to register custom data readers and writers without modifying the core library.
- **Cloud-Native Data Access:** Integration with xarray, STAC, Zarr, and Cloud Optimized GeoTIFF (COG) for efficient discovery, access, and analysis of large-scale InSAR datasets.
- **Metadata Validation and Provenance:** Explicit handling of physical conventions, metadata consistency, and information preservation during data conversion.

By providing common data structures and bidirectional conversion interfaces, Data Commons aims to reduce software integration costs and facilitate data exchange and algorithm comparisons across the InSAR community.

### 2. Compute Commons — Core Algorithms and Pluggable Processing Engines

Compute Commons provides the fundamental computational building blocks for InSAR processing and analysis.

Unlike workflow-oriented processing systems, FanInSAR exposes individual algorithms through Python APIs, allowing researchers to execute specific operations, inspect intermediate results, and develop customized processing methods.

Building upon its existing native implementation, Compute Commons will introduce:

- **Core InSAR Algorithms:** Reusable numerical algorithms and processing functions accessible through intuitive Python APIs.
- **Unified Algorithm Interfaces:** Standardized interfaces for common InSAR operations, such as phase filtering, phase unwrapping, geometric transformations, and time-series inversion.
- **Pluggable Processing Engines:** An extensible engine architecture allowing users to select different algorithm implementations through the same API.
- **User-Registered Engines:** A plugin registration mechanism enabling developers to integrate their own algorithms or established implementations from existing InSAR software without modifying the core library.
- **Hardware-Accelerated Computing:** PyTorch-based numerical computing across supported CPU and GPU architectures.
- **Distributed Computing:** Integration with Dask to support parallel execution and large-scale processing.

For example, researchers will be able to call different phase-unwrapping implementations through a common Python interface simply by selecting a different processing engine.

This architecture aims to simplify algorithm development, encourage reuse of existing methods, and provide a flexible computational foundation for both interactive scientific analysis and automated workflows.

### 3. Benchmark Commons — Open and Reproducible Algorithm Evaluation

Benchmark Commons aims to establish an open and extensible benchmarking suite for evaluating InSAR algorithms and software implementations.

Building upon the common data structures and pluggable processing engines, the benchmarking suite will enable researchers to evaluate different algorithms under standardized and reproducible experimental conditions.

Its main capabilities will include:

- **Open Benchmark Datasets:** Synthetic and real-world InSAR datasets, with reference results and documented experimental configurations.
- **Scientific Accuracy:** Evaluation of algorithm accuracy, robustness, and numerical consistency.
- **Computational Efficiency:** Systematic comparison of processing time, memory consumption, and scalability.
- **Automated Engine Benchmarking:** Direct evaluation of registered processing engines through common algorithm interfaces.
- **Reproducible Experiments:** Recording of input datasets, algorithm parameters, software versions, computing environments, and processing histories.
- **Community Contributions:** Support for researchers to register new algorithms and evaluate them using existing benchmark datasets and procedures.

By connecting standardized data, interchangeable algorithm implementations, and common evaluation metrics, Benchmark Commons aims to enable fair, transparent, and reproducible comparisons across different InSAR methods and software packages.

## Highlight Features

### Pythonic — InSAR Computing Made Accessible

FanInSAR provides intuitive Python APIs for InSAR data access, manipulation, and analysis.

It already supports commonly used InSAR products, including HyP3 and LiCSAR datasets, through dedicated Python classes.

Users can perform spatial and temporal queries using points, bounding boxes, polygons, and interferometric pairs. Coordinate transformations, reprojection, and resampling during data sampling are handled automatically by the library.

The goal is to make working with InSAR data as straightforward as working with multidimensional scientific datasets in xarray.

### Fast — Hardware-Accelerated Scientific Computing

FanInSAR uses PyTorch for its core numerical computations, enabling efficient execution on supported CPU and GPU devices.

The planned integration of distributed computing and cloud-native data access will further extend its capabilities for large-scale InSAR processing.

### Flexible — Algorithms as Reusable Building Blocks

FanInSAR is designed around independently accessible computational functions rather than rigid processing pipelines.

Researchers can directly manipulate intermediate results, customize individual processing steps, and combine existing algorithms to develop new methods.

Its extensible architecture will further support user-defined data adapters and processing engines, allowing researchers to integrate external algorithms without modifying the core library.

### Interoperable — Connecting Existing InSAR Software

Through standardized data structures and extensible import/export interfaces, FanInSAR aims to facilitate data exchange between different InSAR software packages.

Unified algorithm APIs and pluggable processing engines will further enable researchers to access and compare implementations from multiple software packages within a common Python environment.

### Reproducible — Transparent and Comparable InSAR Processing

Reproducibility is a central principle of FanInSAR.

By standardizing data conventions, documenting algorithm configurations, tracking processing provenance, and providing open benchmark datasets, FanInSAR aims to make InSAR results easier to reproduce, understand, and compare.

Rather than requiring different algorithms to produce identical results, the goal is to ensure that their differences can be systematically evaluated and explained.

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
