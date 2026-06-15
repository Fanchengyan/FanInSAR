"""Hugging Face Hub publishing helpers for frame products (M5).

Uploads a standardised ``frame/`` directory tree to a Hugging Face Hub
dataset repository. ``huggingface_hub`` is an optional dependency; the
``dry_run`` mode exercises the file-walking/upload-plan logic without any
network access, which is what the unit tests use.

Real upload is only attempted when ``dry_run=False`` and
``huggingface_hub`` is importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from faninsar.logging import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = setup_logger(__name__)


def iter_frame_assets(frame_root: str | Path) -> Iterator[Path]:
    """Yield every regular file under *frame_root* (depth-first, deterministic).

    Hidden files (starting with ``.``) are skipped.
    """
    frame_root = Path(frame_root)
    for p in sorted(frame_root.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def build_upload_plan(frame_root: str | Path) -> list[dict[str, str]]:
    """Return the list of ``{path_in_repo, local_path}`` entries to upload."""
    frame_root = Path(frame_root).resolve()
    plan: list[dict[str, str]] = []
    for p in iter_frame_assets(frame_root):
        rel = p.relative_to(frame_root).as_posix()
        plan.append({"path_in_repo": rel, "local_path": str(p)})
    return plan


def publish_to_huggingface(
    frame_root: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Upload a frame product to a Hugging Face Hub dataset repository.

    Parameters
    ----------
    frame_root : str or Path
        Local ``frame/`` directory to upload.
    repo_id : str
        HF dataset repo id, e.g. ``"username/dataset-name"``.
    token : str, optional
        HF token. If *None*, uses the locally configured token
        (``huggingface_hub.login``).
    private : bool
        Create the repo as private.
    dry_run : bool
        If *True* (default), no network calls are made; returns the upload
        plan only. Used by tests.

    Returns
    -------
    dict
        Result with keys:

        - ``repo_id``: the target repo id.
        - ``dry_run``: whether the upload was simulated.
        - ``file_count``: number of files in the plan.
        - ``files``: the upload plan (list of dicts).
        - ``uploaded``: bool — whether files were actually uploaded.

    Raises
    ------
    ImportError
        If ``dry_run=False`` and ``huggingface_hub`` is not installed.
    FileNotFoundError
        If *frame_root* does not exist.

    """
    frame_root = Path(frame_root)
    if not frame_root.is_dir():
        msg = f"frame_root does not exist: {frame_root}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    plan = build_upload_plan(frame_root)
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "dry_run": dry_run,
        "file_count": len(plan),
        "files": plan,
        "uploaded": False,
    }

    if dry_run:
        logger.info("Dry run: %d files would be uploaded to %s", len(plan), repo_id)
        return result

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        msg = (
            "huggingface_hub is required for publish_to_huggingface(dry_run=False). "
            "Install it with: pip install huggingface_hub"
            " (or the 'cloud' extra)."
        )
        raise ImportError(msg) from e

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    for entry in plan:
        api.upload_file(
            path_or_fileobj=entry["local_path"],
            path_in_repo=entry["path_in_repo"],
            repo_id=repo_id,
            repo_type="dataset",
        )
    result["uploaded"] = True
    logger.info("Uploaded %d files to %s", len(plan), repo_id)
    return result


__all__ = ["build_upload_plan", "iter_frame_assets", "publish_to_huggingface"]
