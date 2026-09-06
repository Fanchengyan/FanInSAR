"""Open Sentinel-1 Level-1 SAFE products from directories or ZIP archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

from faninsar.logging import setup_logger

from .annotation import parse_annotation_xml
from .errors import UnsupportedPolarizationError, reject_product
from .types import S1Product, S1Swath

logger = setup_logger(__name__)

_SUPPORTED_POLARIZATIONS = frozenset({"VV", "VH", "HH", "HV"})


def open_safe_product(
    path: str | Path,
    *,
    polarizations: tuple[str, ...] | None = None,
) -> S1Product:
    """Open a SAFE directory or ZIP and parse annotation metadata lazily.

    Measurement GeoTIFF pixels are not loaded. Paths to measurement assets are
    retained for later burst-window reads.

    Parameters
    ----------
    path : str or pathlib.Path
        SAFE directory or ``.zip`` archive.
    polarizations : tuple of str, optional
        Subset of polarizations to load. Defaults to all present.

    Returns
    -------
    S1Product
        Typed product handle with sub-swath and burst metadata.

    Raises
    ------
    Sentinel1ProductError
        If the product structure or required metadata is invalid.
    UnsupportedPolarizationError
        If a requested polarization is not present.

    """
    source = Path(path)
    if not source.exists():
        reject_product(f"SAFE product does not exist: {source}")

    if source.is_dir():
        return _open_safe_directory(source, polarizations=polarizations)
    if source.suffix.lower() == ".zip":
        return _open_safe_zip(source, polarizations=polarizations)
    return reject_product(f"unsupported SAFE source type: {source}")


def _open_safe_directory(
    root: Path,
    *,
    polarizations: tuple[str, ...] | None,
) -> S1Product:
    if (root / "manifest.safe").exists():
        safe_roots = [root]
    else:
        safe_roots = list(root.glob("*.SAFE"))
    if (
        not safe_roots
        and (root / "annotation").is_dir()
        and (root / "measurement").is_dir()
    ):
        safe_roots = [root]
    if len(safe_roots) != 1:
        reject_product(f"expected one SAFE root under {root}")
    safe_root = safe_roots[0]
    annotation_files = sorted(
        path
        for path in (safe_root / "annotation").glob("*.xml")
        if path.is_file() and "calibration" not in path.parts
    )
    if not annotation_files:
        reject_product(f"no annotation XML found under {safe_root}")

    swaths: list[S1Swath] = []
    for annotation in annotation_files:
        measurement = _measurement_for_annotation(safe_root, annotation.name)
        xml_text = annotation.read_text(encoding="utf-8")
        swath = parse_annotation_xml(
            xml_text,
            annotation_path=str(annotation),
            measurement_path=str(measurement),
        )
        swaths.append(swath)
    return _assemble_product(
        source_path=root,
        swaths=swaths,
        polarizations=polarizations,
    )


def _open_safe_zip(
    zip_path: Path,
    *,
    polarizations: tuple[str, ...] | None,
) -> S1Product:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        annotation_names = sorted(
            name
            for name in names
            if "/annotation/" in name
            and name.endswith(".xml")
            and "/calibration/" not in name
            and not name.endswith("/")
        )
        if not annotation_names:
            reject_product(f"no annotation XML found in {zip_path}")

        swaths: list[S1Swath] = []
        for annotation_name in annotation_names:
            measurement_name = _measurement_member_for_annotation(
                names,
                annotation_name,
            )
            xml_text = archive.read(annotation_name).decode("utf-8")
            measurement_uri = f"/vsizip/{zip_path}/{measurement_name}"
            swath = parse_annotation_xml(
                xml_text,
                annotation_path=f"/vsizip/{zip_path}/{annotation_name}",
                measurement_path=measurement_uri,
            )
            swaths.append(swath)
    return _assemble_product(
        source_path=zip_path,
        swaths=swaths,
        polarizations=polarizations,
    )


def _measurement_for_annotation(safe_root: Path, annotation_name: str) -> Path:
    stem = Path(annotation_name).stem
    candidate = safe_root / "measurement" / f"{stem}.tiff"
    if not candidate.exists():
        matches = list((safe_root / "measurement").glob(f"{stem}.*"))
        if not matches:
            reject_product(f"missing measurement for annotation {annotation_name}")
        return matches[0]
    return candidate


def _measurement_member_for_annotation(
    names: list[str],
    annotation_name: str,
) -> str:
    stem = Path(annotation_name).stem
    for name in names:
        if "/measurement/" in name and Path(name).stem == stem:
            return name
    return reject_product(f"missing measurement member for {annotation_name}")


def _assemble_product(
    *,
    source_path: Path,
    swaths: list[S1Swath],
    polarizations: tuple[str, ...] | None,
) -> S1Product:
    present = tuple(sorted({swath.polarization for swath in swaths}))
    requested = polarizations or present
    allowed = sorted(_SUPPORTED_POLARIZATIONS)
    for pol in requested:
        if pol not in _SUPPORTED_POLARIZATIONS:
            message = f"unsupported polarization {pol!r}; allowed={allowed}"
            logger.error(message)
            raise UnsupportedPolarizationError(message)
        if pol not in present:
            message = (
                f"polarization {pol!r} not present in {source_path}; "
                f"available={present}"
            )
            logger.error(message)
            raise UnsupportedPolarizationError(message)

    filtered = tuple(swath for swath in swaths if swath.polarization in requested)
    if not filtered:
        reject_product(f"no swaths remain after polarization filter for {source_path}")

    product_id = source_path.stem.replace(".SAFE", "")
    if product_id.startswith(("S1A", "S1B", "S1C")):
        mission_id = product_id[:3]
    else:
        mission_id = "S1"
    mode = "IW" if "IW" in product_id else filtered[0].swath[:2]
    absolute_orbit = 0
    for part in product_id.split("_"):
        if part.isdigit() and len(part) == 6:
            absolute_orbit = int(part)
            break

    return S1Product(
        product_id=product_id,
        source_path=source_path,
        mission_id=mission_id,
        product_type="SLC",
        mode=mode,
        absolute_orbit=absolute_orbit,
        polarizations=tuple(requested),
        swaths=filtered,
    )
