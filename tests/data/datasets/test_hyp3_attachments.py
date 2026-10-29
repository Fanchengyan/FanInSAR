"""Public-seam tests for HyP3 LOS attachment and GACOS inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from pyproj.crs import CRS
from rasterio.transform import from_bounds

from faninsar.data._adapters.hyp3 import HyP3S1
from faninsar.data.datasets.gacos import GACOS
from faninsar.data.datasets.ifg import InterferogramDataset
from faninsar.data.query import Points

if TYPE_CHECKING:
    import pytest


def _write_tif(
    path: Path,
    value: float,
    *,
    crs: CRS,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 4.0, 4.0),
) -> None:
    """Write a 4x4 GeoTIFF filled with ``value``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height = width = 4
    transform = from_bounds(*bounds, width, height)
    data = np.full((height, width), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(data, 1)


def _hyp3_product_name(primary: str, secondary: str) -> str:
    return f"S1AA_{primary}T000000_{secondary}T000000_VVP012_INT40_G_ueF_TEST"


def _write_hyp3_product(
    root: Path,
    primary: str,
    secondary: str,
    *,
    write_los: bool = True,
    extra_los: tuple[str, ...] = (),
    crs: CRS | None = None,
) -> Path:
    """Create one HyP3-like product folder and return it."""
    crs = CRS.from_epsg(32605) if crs is None else crs
    name = _hyp3_product_name(primary, secondary)
    product = root / name
    _write_tif(product / f"{name}_unw_phase.tif", 1.0, crs=crs)
    _write_tif(product / f"{name}_corr.tif", 0.8, crs=crs)
    if write_los:
        _write_tif(product / f"{name}_lv_theta.tif", 0.7, crs=crs)
    for suffix in extra_los:
        _write_tif(product / f"{name}_{suffix}.tif", 0.1, crs=crs)
    return product


def test_parse_pairs_accepts_hyp3_datetime_stamps() -> None:
    """HyP3 names with THHMMSS still parse to YYYYMMDD pair names."""
    name = _hyp3_product_name("20210101", "20210113")
    pairs = InterferogramDataset.parse_pairs([Path(f"{name}_unw_phase.tif")])
    assert pairs.to_names().tolist() == ["20210101_20210113"]


def test_hyp3_attaches_canonical_lv_theta(tmp_path: Path) -> None:
    """HyP3 initialization attaches canonical lv_theta files when present."""
    _write_hyp3_product(
        tmp_path,
        "20210101",
        "20210113",
        extra_los=("lv_theta_clipped", "lv_theta_roi"),
    )
    ds = HyP3S1(root_dir=tmp_path, verbose=False, nodata=0.0)
    assert ds.los_dataset is not None
    names = [Path(p).name for p in ds.los_dataset.files.paths]
    assert any(name.endswith("_lv_theta.tif") for name in names)
    assert not any("clipped" in name or "roi" in name for name in names)


def test_hyp3_load_los_ratio_defaults_to_look_sine(tmp_path: Path) -> None:
    """HyP3 lv_theta is look-vector elevation; default ratio is sin(θ)."""
    _write_hyp3_product(tmp_path, "20210101", "20210113")
    ds = HyP3S1(root_dir=tmp_path, verbose=False, nodata=0.0)
    assert ds.los_angle_type == "look"
    ratio = ds.load_los_ratio()
    np.testing.assert_allclose(
        ratio, np.full((4, 4), np.sin(0.7)), rtol=1e-5, atol=1e-5
    )
    wrong = ds.load_los_ratio(angle_type="incidence")
    np.testing.assert_allclose(
        wrong, np.full((4, 4), np.cos(0.7)), rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(ratio, wrong)


def test_hyp3_warns_when_lv_theta_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing lv_theta leaves los_dataset unset and logs a warning."""
    _write_hyp3_product(tmp_path, "20210101", "20210113", write_los=False)
    with caplog.at_level("WARNING"):
        ds = HyP3S1(root_dir=tmp_path, verbose=False, nodata=0.0)
    assert ds.los_dataset is None
    assert "lv_theta" in caplog.text


def test_set_gacos_dataset_inherits_interferogram_crs_res(tmp_path: Path) -> None:
    """GACOS attached through the interferogram dataset inherits crs and res."""
    hyp3_dir = tmp_path / "hyp3"
    gacos_dir = tmp_path / "gacos"
    crs_utm = CRS.from_epsg(32605)
    _write_hyp3_product(hyp3_dir, "20210101", "20210113", crs=crs_utm)
    _write_tif(gacos_dir / "20210101.ztd.tif", 2.4, crs=crs_utm)
    _write_tif(gacos_dir / "20210113.ztd.tif", 2.5, crs=crs_utm)

    ds = HyP3S1(root_dir=hyp3_dir, verbose=False, nodata=0.0)
    ds.set_gacos_dataset(root_dir=gacos_dir, nodata=np.nan, verbose=False)

    assert ds.gacos is not None
    assert isinstance(ds.gacos, GACOS)
    assert ds.gacos.crs == ds.crs
    assert ds.gacos.res == ds.res
    query = Points([(2.0, 2.0)], crs=ds.crs)
    increments = ds.gacos.increments(
        ["20210101", "20210113"],
        query=query,
        ref_points=query,
    )
    assert increments.shape == (1, 1)
    np.testing.assert_allclose(increments, [[0.0]], atol=1e-6)
