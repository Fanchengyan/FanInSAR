"""Progress tracker for ISCE2 workflow execution.

This module provides progress tracking and checkpoint/resume functionality.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class ProgressTracker:
    """Workflow progress tracker.

    Records workflow execution progress and supports checkpoint/resume.

    Parameters
    ----------
    work_dir : Path
        Working directory.
    workflow_name : str
        Workflow name (e.g., "interferogram_stack").

    Examples
    --------
    >>> tracker = ProgressTracker(Path("/work"), "interferogram_stack")
    >>> tracker.mark_completed("run_01_unpack_topo_reference")
    >>> tracker.is_completed("run_01_unpack_topo_reference")
    True
    >>> tracker.get_progress(total_runs=16)
    0.0625  # 1/16 completed

    """

    def __init__(self, work_dir: Path, workflow_name: str) -> None:
        """Initialize the progress tracker."""
        self.work_dir = work_dir
        self.workflow_name = workflow_name
        self.state_file = work_dir / f".{workflow_name}_progress.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load progress state.

        Returns
        -------
        dict
            Progress state dictionary.

        """
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "workflow": self.workflow_name,
            "started_at": datetime.now(UTC).isoformat(),
            "completed_runs": [],
            "failed_runs": [],
        }

    def _save_state(self) -> None:
        """Save progress state."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def mark_completed(self, run_name: str) -> None:
        """Mark run file as completed.

        Parameters
        ----------
        run_name : str
            Run file name (e.g., "run_01_unpack_topo_reference").

        """
        if run_name not in self.state["completed_runs"]:
            self.state["completed_runs"].append(run_name)
            self.state["last_updated"] = datetime.now(UTC).isoformat()
            self._save_state()
            logger.info("Marked %s as completed", run_name)

    def mark_failed(self, run_name: str, error: str) -> None:
        """Mark run file as failed.

        Parameters
        ----------
        run_name : str
            Run file name.
        error : str
            Error message.

        """
        self.state["failed_runs"].append(
            {
                "run": run_name,
                "error": error,
                "time": datetime.now(UTC).isoformat(),
            }
        )
        self._save_state()
        logger.error("Marked %s as failed: %s", run_name, error)

    def is_completed(self, run_name: str) -> bool:
        """Check if run file is completed.

        Parameters
        ----------
        run_name : str
            Run file name.

        Returns
        -------
        bool
            True if completed, False otherwise.

        """
        return run_name in self.state["completed_runs"]

    def get_progress(self, total_runs: int) -> float:
        """Get progress percentage.

        Parameters
        ----------
        total_runs : int
            Total number of runs in the workflow.

        Returns
        -------
        float
            Progress percentage (0.0 to 1.0).

        """
        if total_runs == 0:
            return 0.0
        return len(self.state["completed_runs"]) / total_runs

    def get_pending_runs(self, all_runs: list[str]) -> list[str]:
        """Get list of pending runs.

        Parameters
        ----------
        all_runs : list[str]
            List of all run names.

        Returns
        -------
        list[str]
            List of pending run names.

        """
        return [run for run in all_runs if not self.is_completed(run)]

    def reset(self) -> None:
        """Reset progress."""
        self.state = {
            "workflow": self.workflow_name,
            "started_at": datetime.now(UTC).isoformat(),
            "completed_runs": [],
            "failed_runs": [],
        }
        self._save_state()
        logger.info("Progress reset")

    def get_summary(self) -> dict:
        """Get progress summary.

        Returns
        -------
        dict
            Summary dictionary with keys:
            - workflow: workflow name
            - started_at: start time
            - completed_count: number of completed runs
            - failed_count: number of failed runs
            - last_updated: last update time

        """
        return {
            "workflow": self.state["workflow"],
            "started_at": self.state["started_at"],
            "completed_count": len(self.state["completed_runs"]),
            "failed_count": len(self.state["failed_runs"]),
            "last_updated": self.state.get("last_updated", self.state["started_at"]),
        }
