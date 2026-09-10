"""Time-series inversion (NSBAS / SBAS). Consumes InterferogramStack only."""

from __future__ import annotations

from faninsar.timeseries.export import (
    ExporterRegistry,
    NetcdfExporter,
    StacCogExporter,
    StacZarrExporter,
    TimeSeriesExporter,
    registry,
    write_timeseries_product,
)
from faninsar.timeseries.invert import NSBAS, SBAS, invert
from faninsar.timeseries.io import (
    TimeSeriesZarrStore,
    open_timeseries_zarr,
    write_timeseries_zarr,
)
from faninsar.timeseries.models import (
    AnnualSemiannualSinusoidal,
    AnnualSinusoidalModel,
    CubicModel,
    FreezeThawCycleModel,
    FreezeThawCycleModelWithVelocity,
    LinearModel,
    QuadraticModel,
    TimeSeriesModels,
)
from faninsar.timeseries.results import (
    SpatialMetadata,
    TimeSeries,
    invert_unwrapped_pairs,
)
from faninsar.timeseries.solver import (
    NSBASSolver,
    batch_lstsq,
    calculate_u,
    censored_lstsq,
)
from faninsar.timeseries.uncertainty import (
    G2M,
    ReferencePointsUncertainty,
    Uncertainty,
    UncertaintyPropagation,
    data2param,
    data2param_cov,
    data2param_sequence,
    get_var_patch,
    safe_remove,
)

__all__ = [
    "G2M",
    "NSBAS",
    "SBAS",
    "AnnualSemiannualSinusoidal",
    "AnnualSinusoidalModel",
    "CubicModel",
    "ExporterRegistry",
    "FreezeThawCycleModel",
    "FreezeThawCycleModelWithVelocity",
    "LinearModel",
    "NSBASSolver",
    "NetcdfExporter",
    "QuadraticModel",
    "ReferencePointsUncertainty",
    "SpatialMetadata",
    "StacCogExporter",
    "StacZarrExporter",
    "TimeSeries",
    "TimeSeriesExporter",
    "TimeSeriesModels",
    "TimeSeriesZarrStore",
    "Uncertainty",
    "UncertaintyPropagation",
    "batch_lstsq",
    "calculate_u",
    "censored_lstsq",
    "data2param",
    "data2param_cov",
    "data2param_sequence",
    "get_var_patch",
    "invert",
    "invert_unwrapped_pairs",
    "open_timeseries_zarr",
    "registry",
    "safe_remove",
    "write_timeseries_product",
    "write_timeseries_zarr",
]
