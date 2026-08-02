from __future__ import annotations

import pytest

from faninsar.missions.nisar import NisarSensor


def test_nisar_sensor_instantiation() -> None:
    sensor = NisarSensor()
    assert sensor.name == "nisar"


def test_nisar_open_product_raises_not_implemented() -> None:
    sensor = NisarSensor()
    with pytest.raises(NotImplementedError, match="NISAR open_product requires ISCE3 fixtures"):
        sensor.open_product("s3://bucket/nisar.h5")
