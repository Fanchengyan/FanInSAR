from .freeze_thaw_process import FreezeThawCycle
from .inversion import (
    NSBASInversion,
    NSBASMatrixFactory,
    batch_lstsq,
    calculate_u,
    censored_lstsq,
)
from .tsmodels import (
    AnnualSemiannualSinusoidal,
    AnnualSinusoidalModel,
    CubicModel,
    FreezeThawCycleModel,
    FreezeThawCycleModelWithVelocity,
    LinearModel,
    QuadraticModel,
    TimeSeriesModels,
)

available_models = [
    "LinearModel",
    "QuadraticModel",
    "CubicModel",
    "AnnualSinusoidalModel",
    "AnnualSemiannualSinusoidal",
    "FreezeThawCycleModel",
    "FreezeThawCycleModelWithVelocity",
]

MAP_MODEL = {
    "LinearModel": LinearModel,
    "QuadraticModel": QuadraticModel,
    "CubicModel": CubicModel,
    "AnnualSinusoidalModel": AnnualSinusoidalModel,
    "AnnualSemiannualSinusoidal": AnnualSemiannualSinusoidal,
    "FreezeThawCycleModel": FreezeThawCycleModel,
    "FreezeThawCycleModelWithVelocity": FreezeThawCycleModelWithVelocity,
}


def get_model(name: str) -> TimeSeriesModels:
    """Get a model object from a string name.

    Parameters
    ----------
    name : str
        Name of the model to return.

    Returns
    -------
    model : Model
        Model class.

    """
    if name in MAP_MODEL:
        return MAP_MODEL[name]
    msg = f"Model {name} not found. Available models are: {available_models}"
    raise ValueError(msg)
