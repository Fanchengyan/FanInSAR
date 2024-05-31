<h1 align="center">
<img src="https://raw.githubusercontent.com/Fanchengyan/FanInSAR/main/docs/source/_static/logo/logo.svg" width="400">
</h1><br>

[![Documentation Status](https://readthedocs.org/projects/faninsar/badge/?version=latest)](https://faninsar.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11398347.svg)](https://doi.org/10.5281/zenodo.11398347)

FanInSAR is a fancy InSAR post-processing library written in Python. It is specifically designed to assist in the efficient processing of InSAR data, offering a Pythonic, fast, and flexible approach. If you are looking to implement your own InSAR algorithm, FanInSAR is highly recommended.

## Highlight Features

- **Pythonic**: FanInSAR is written in Python and provides a user-friendly API. The API is designed to be simple and intuitive, making it easy for InSAR users to work with. For example, loading data from ``HyP3`` or ``LiCSAR`` products is as simple as providing the corresponding home directory. Filtering interferometric pairs can be performed by a time slice, similar to the ``pandas`` package. The complex processing pipeline is abstracted away, allowing users to focus on algorithm development.
- **Fast**: The core computation in FanInSAR is implemented using ``PyTorch``, a high-performance deep learning library. This allows for efficient processing on both CPU and GPU, enabling faster execution.
- **Flexible**: FanInSAR is designed to be flexible, allowing for customization and extension. Users can easily inherit classes or customize the processing pipeline to suit their specific needs.

## Installation 

FanInSAR is a Python package, and requires ``Python >= 3.8``. You can install the latest release of FanInSAR using ``pip``:

```bash
pip install git+https://github.com/Fanchengyan/FanInSAR.git
```

## Documentation

The detailed documentation is available at: <https://faninsar.readthedocs.io/en/latest/>

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

