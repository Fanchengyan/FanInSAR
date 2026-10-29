"""Time-series inversion (NSBAS / SBAS). Consumes InterferogramStack only."""

from __future__ import annotations

from faninsar.timeseries.coherence_weight import (
    WeightFunc,
    coherence_to_weight,
    normalize_weight_func,
)
from faninsar.timeseries.export import (
    ExporterRegistry,
    NetcdfExporter,
    StacCogExporter,
    StacZarrExporter,
    TimeSeriesExporter,
    registry,
)
from faninsar.timeseries.invert import NSBAS, SBAS, invert
from faninsar.timeseries.io import (
    read_timeseries,
    write_timeseries_product,
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
    LstsqResult,
    NSBASResult,
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
    "LstsqResult",
    "NSBASResult",
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
    "Uncertainty",
    "UncertaintyPropagation",
    "WeightFunc",
    "batch_lstsq",
    "calculate_u",
    "censored_lstsq",
    "coherence_to_weight",
    "data2param",
    "data2param_cov",
    "data2param_sequence",
    "get_var_patch",
    "invert",
    "invert_unwrapped_pairs",
    "normalize_weight_func",
    "read_timeseries",
    "registry",
    "safe_remove",
    "write_timeseries_product",
]
