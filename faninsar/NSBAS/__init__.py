from faninsar.NSBAS.inversion import (
    NSBASInversion,
    NSBASMatrixFactory,
    PhaseDeformationConverter,
    batch_lstsq,
    calculate_u,
    censored_lstsq,
)
from faninsar.NSBAS.tsmodels import (
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


def get_model(name):
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
    if name in MAP_MODEL.keys():
        return MAP_MODEL[name]
    else:
        raise ValueError(f"Model {name} not found.")
