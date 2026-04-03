from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MDTrajectoryAppState:
    """Mutable UI state for the mdtrajectory application."""

    trajectory_file: Path | None = None
    topology_file: Path | None = None
    energy_file: Path | None = None
    backend: str = "auto"

    start: int | None = None
    stop: int | None = None
    stride: int = 1

    temp_target_k: float = 300.0
    temp_tol_k: float = 1.0
    window: int = 3

    suggested_cutoff_fs: float | None = None
    selected_cutoff_fs: float | None = None

    output_dir: Path | None = None
    use_cutoff_for_export: bool = True
    use_post_cutoff_stride: bool = False
    post_cutoff_stride: int = 1
