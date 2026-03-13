from __future__ import annotations

from ..core.base import SAXShellResult, SAXShellWorkflow
from .config import ExtractionConfig


class ClusterExtractor(SAXShellWorkflow):
    """Skeleton workflow for extracting clusters from a trajectory."""

    workflow_name = "extract-clusters"

    def __init__(self, config: ExtractionConfig) -> None:
        super().__init__(config=config)
        self.config: ExtractionConfig = config

    def run(self) -> SAXShellResult:
        message = (
            "Initialized cluster extraction with "
            f"trajectory='{self.config.trajectory_path}', "
            f"solute='{self.config.solute_name}', "
            f"solvent='{self.config.solvent_name}', "
            f"frame_stride={self.config.frame_stride}"
        )
        return SAXShellResult(
            workflow_name=self.workflow_name,
            message=message,
        )
