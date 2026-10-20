"""Write Stack scene products with the production one-Reference layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from faninsar.io.storage.scene_store import CoregisteredSceneStore, write_scene

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def coreg_dir(work_dir: Path, date_id: str, reference_id: str | None = None) -> Path:
    """Return the Stack scene product directory of one acquisition."""
    del reference_id
    return work_dir / "coreg" / date_id


def write_test_scene(
    work_dir: Path,
    *,
    date_id: str,
    reference_id: str,
    payload: np.ndarray,
    reference_payload: np.ndarray | None = None,
    tag: str | tuple[str, ...] = "f0_IW1_b0",
    domain: str = "radar",
    row_origin: int = 0,
    col_origin: int = 0,
    phase_screen: np.ndarray | None = None,
    phase_state: dict[str, object] | None = None,
    properties: dict[str, object] | None = None,
    inputs: dict[str, str] | None = None,
    **header: Any,
) -> Path:
    """Write one acquisition product where a Stack would.

    ``tag`` may be a tuple to write several units sharing one payload.

    A secondary first ensures the Reference product exists (writing
    ``reference_payload``, default ones, when it is missing) and binds its
    ``write_id``, exactly as coregistration does.

    Returns
    -------
    pathlib.Path
        The acquisition's scene product directory.

    """
    import numpy as np

    header.setdefault(
        "grid_shape",
        (row_origin + payload.shape[0], col_origin + payload.shape[1]),
    )

    tags = (tag,) if isinstance(tag, str) else tuple(tag)

    def unit(data: np.ndarray, **extra: object) -> list[dict[str, object]]:
        return [
            {
                "tag": name,
                "payload": np.asarray(data, dtype=np.complex64),
                "row_origin": row_origin,
                "col_origin": col_origin,
                "phase_state": phase_state,
                **extra,
            }
            for name in tags
        ]

    reference_root = coreg_dir(work_dir, reference_id)
    if date_id == reference_id:
        write_scene(
            reference_root,
            date_id=reference_id,
            reference_id=reference_id,
            domain=domain,
            units=unit(payload),
            overwrite=True,
            **header,
        )
        return reference_root
    if (reference_root / "item.json").is_file():
        reference = CoregisteredSceneStore.open(reference_root)
    else:
        reference = write_scene(
            reference_root,
            date_id=reference_id,
            reference_id=reference_id,
            domain=domain,
            units=unit(
                np.ones(payload.shape, dtype=np.complex64)
                if reference_payload is None
                else reference_payload
            ),
            **header,
        )
    root = coreg_dir(work_dir, date_id)
    write_scene(
        root,
        date_id=date_id,
        reference_id=reference_id,
        domain=domain,
        units=unit(payload, phase_screen=phase_screen),
        inputs={**(inputs or {}), "reference": reference.write_id},
        properties=properties,
        overwrite=True,
        **header,
    )
    return root


def open_test_scene(root: Path) -> CoregisteredSceneStore:
    """Open the scene product in one acquisition directory."""
    return CoregisteredSceneStore.open(root)
