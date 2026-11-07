"""Standard SAFE relative-orbit decoding for merge-layout track identity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from faninsar.core.orbit import OrbitMetadata, OrbitStateVector
from faninsar.missions.protocols import DopplerCentroidPolynomial
from faninsar.missions.s1.merge_layout import track_identity
from faninsar.missions.s1.track_map import (
    relative_orbit_from_absolute,
    relative_orbit_from_manifest,
    resolve_track_acquisition,
    track_key_for_product,
)
from faninsar.missions.s1.types import S1Burst, S1Product, S1Swath
from faninsar.processing.errors import InvalidProcessingStateError

if TYPE_CHECKING:
    from pathlib import Path

# S1A 2021-11-28 abs 40778 → relative 106 (ESA / ISCE cycle formula).
_STANDARD_SAFE = (
    "S1A_IW_SLC__1SDV_20211128T000000_20211128T000130_040778_04DXXX_XXXX.SAFE"
)
_S1A_ABS_ORBIT = 40778
_S1A_REL_ORBIT = 106
_ORIGIN = datetime(2021, 11, 28, tzinfo=UTC)

_MANIFEST = """\
<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
           xmlns:safe="http://www.esa.int/safe/sentinel-1.0">
  <metadataSection>
    <metadataObject ID="measurementOrbitReference">
      <metadataWrap><xmlData>
        <safe:orbitReference>
          <safe:orbitNumber type="start">40778</safe:orbitNumber>
          <safe:relativeOrbitNumber type="start">106</safe:relativeOrbitNumber>
        </safe:orbitReference>
      </xmlData></metadataWrap>
    </metadataObject>
  </metadataSection>
</xfdu:XFDU>
"""


def _ascending_orbit() -> OrbitMetadata:
    """Return ECEF vectors whose northward motion is an ascending pass."""
    return OrbitMetadata(
        reference_frame="ITRF",
        source="test",
        vectors=tuple(
            OrbitStateVector(
                time=_ORIGIN + timedelta(seconds=10.0 * step),
                position_m=(7_000_000.0, 0.0, 75_000.0 * step),
                velocity_m_s=(0.0, 0.0, 7_500.0),
            )
            for step in range(-3, 4)
        ),
    )


def _swath() -> S1Swath:
    """Return one IW1 burst carrying ascending orbit vectors."""
    burst = S1Burst(
        index=0,
        azimuth_time=_ORIGIN,
        sensing_time=_ORIGIN,
        azimuth_anx_time_s=0.0,
        byte_offset=0,
        lines=8,
        samples=8,
        first_valid_sample=(0,) * 8,
        last_valid_sample=(7,) * 8,
    )
    return S1Swath(
        swath="IW1",
        polarization="VV",
        annotation_path="memory://annotation.xml",
        measurement_path="memory://measurement.tiff",
        lines=8,
        samples=8,
        lines_per_burst=8,
        samples_per_burst=8,
        range_pixel_spacing_m=2.3,
        azimuth_pixel_spacing_m=14.0,
        azimuth_time_interval_s=1.0,
        slant_range_time_s=0.005,
        range_sampling_rate_hz=64_000_000.0,
        radar_frequency_hz=5.405e9,
        sensing_start=_ORIGIN,
        sensing_stop=_ORIGIN + timedelta(seconds=7),
        bursts=(burst,),
        orbit=_ascending_orbit(),
        doppler_centroid=(
            DopplerCentroidPolynomial(
                coefficients_hz=(100.0, -0.01),
                reference_range_m=800_000.0,
                reference_time_s=0.0,
            ),
        ),
        azimuth_fm_rate=(),
        azimuth_steering_rate_rad_s=0.0,
    )


def _product(
    path: Path,
    *,
    mission_id: str = "S1A",
    absolute_orbit: int = _S1A_ABS_ORBIT,
) -> S1Product:
    """Return a product at a raw SAFE path (no ``tracks/`` staging)."""
    return S1Product(
        product_id=path.stem,
        source_path=path,
        mission_id=mission_id,
        product_type="SLC",
        mode="IW",
        absolute_orbit=absolute_orbit,
        polarizations=("VV",),
        swaths=(_swath(),),
    )


def test_s1a_absolute_orbit_maps_to_known_relative_orbit() -> None:
    """ESA/ISCE S1A cycle: abs 40778 is relative 106, not a track by itself."""
    assert relative_orbit_from_absolute("S1A", _S1A_ABS_ORBIT) == _S1A_REL_ORBIT
    assert relative_orbit_from_absolute("S1A", 0) is None
    assert relative_orbit_from_absolute("S1C", _S1A_ABS_ORBIT) is None


def test_manifest_relative_orbit_number_is_authoritative() -> None:
    """SAFE manifest relativeOrbitNumber is the ESA relative orbit."""
    assert relative_orbit_from_manifest(_MANIFEST) == _S1A_REL_ORBIT


def test_standard_safe_filename_resolves_track_without_tracks_layout(
    tmp_path: Path,
) -> None:
    """A raw ESA product name loads the selection key; no symlink rename."""
    safe = tmp_path / _STANDARD_SAFE
    safe.mkdir()
    product = _product(safe)
    assert "tracks" not in product.source_path.parts
    assert product.absolute_orbit == _S1A_ABS_ORBIT
    assert track_key_for_product(product) == "106A"
    assert track_identity(product) == "106A"
    assert track_identity(product) != f"{product.mission_id}-{product.absolute_orbit}"


def test_manifest_relative_orbit_is_used_when_product_field_is_missing(
    tmp_path: Path,
) -> None:
    """S1Product may omit relative_orbit; the manifest still supplies it."""
    safe = tmp_path / "S1A_IW_SLC__1SDV_20211128T000000.SAFE"
    safe.mkdir()
    (safe / "manifest.safe").write_text(_MANIFEST, encoding="utf-8")
    product = _product(safe, absolute_orbit=0)
    assert product.absolute_orbit == 0
    assert track_key_for_product(product) == "106A"
    assert track_identity(product) == "106A"


def test_path_without_manifest_keeps_encoded_track(tmp_path: Path) -> None:
    """Path-SAFE burst trees may omit relativeOrbitNumber; the path still counts."""
    safe = (
        tmp_path
        / "tracks"
        / "88A"
        / "acquisitions"
        / "S1A_IW_SLC__1SDV_20211128T000000.SAFE"
    )
    safe.mkdir(parents=True)
    product = _product(safe, absolute_orbit=0)
    assert not (safe / "manifest.safe").exists()
    assert track_key_for_product(product) == "88A"


def test_path_and_manifest_conflict_is_rejected(tmp_path: Path) -> None:
    """A path-encoded track that disagrees with the manifest is invalid metadata."""
    safe = (
        tmp_path
        / "tracks"
        / "88A"
        / "acquisitions"
        / "S1A_IW_SLC__1SDV_20211128T000000.SAFE"
    )
    safe.mkdir(parents=True)
    (safe / "manifest.safe").write_text(_MANIFEST, encoding="utf-8")
    product = _product(safe, absolute_orbit=0)
    with pytest.raises(InvalidProcessingStateError, match="disagree"):
        track_key_for_product(product)


def test_path_and_manifest_agree(tmp_path: Path) -> None:
    """Matching path and manifest relative orbits publish the same track."""
    safe = tmp_path / "tracks" / "106A" / "acquisitions" / _STANDARD_SAFE
    safe.mkdir(parents=True)
    (safe / "manifest.safe").write_text(_MANIFEST, encoding="utf-8")
    product = _product(safe, absolute_orbit=0)
    assert track_key_for_product(product) == "106A"


def test_s1c_missing_metadata_is_not_an_unsupported_mission(tmp_path: Path) -> None:
    """S1C without relative-orbit metadata fails as missing, not unsupported."""
    product = _product(
        tmp_path / "S1C_IW_SLC__1SDV_20211128T000000.SAFE",
        mission_id="S1C",
        absolute_orbit=_S1A_ABS_ORBIT,
    )
    with pytest.raises(InvalidProcessingStateError, match="relative orbit") as caught:
        track_identity(product)
    assert "unsupported" not in str(caught.value).lower()


def test_unparsed_absolute_orbit_without_metadata_rejects(tmp_path: Path) -> None:
    """No path, manifest, or valid formula input cannot invent a track."""
    product = _product(
        tmp_path / "S1A_IW_SLC__1SDV_20200101T000000.SAFE",
        absolute_orbit=0,
    )
    with pytest.raises(InvalidProcessingStateError, match="relative orbit"):
        track_identity(product)


def test_resolve_track_acquisition_still_ignores_unopened_standard_name(
    tmp_path: Path,
) -> None:
    """Path-only resolution does not guess a track from an unopened SAFE."""
    safe = tmp_path / _STANDARD_SAFE
    safe.mkdir()
    info = resolve_track_acquisition([safe], open_products=False)
    assert info.relative_orbit is None
    assert info.track_key is None
