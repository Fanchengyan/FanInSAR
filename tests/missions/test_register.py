"""Mission registry tests."""

from __future__ import annotations

from faninsar import list_missions, register
from faninsar.missions.base import Sensor, get_mission


def test_list_missions_includes_defaults() -> None:
    names = list_missions()
    assert "sentinel1" in names
    assert "nisar" in names
    assert "alos2" in names


def test_register_custom() -> None:
    @register(name="toy_mission_xyz")
    class Toy(Sensor):
        name = "toy_mission_xyz"

    assert "toy_mission_xyz" in list_missions()
    assert get_mission("toy_mission_xyz") is Toy
