from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.base import SAXShellConfig


@dataclass(slots=True)
class ExtractionConfig(SAXShellConfig):
    """Configuration for cluster extraction."""

    trajectory_path: Path = Path("data/raw/trajectory.xyz")
    topology_path: Path | None = None
    solute_name: str = "solute"
    solvent_name: str = "solvent"
    frame_stride: int = 1
