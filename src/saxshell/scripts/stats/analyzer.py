from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.base import SAXShellConfig, SAXShellResult, SAXShellWorkflow


@dataclass(slots=True)
class ClusterStatsConfig(SAXShellConfig):
    """Configuration for cluster statistics."""

    cluster_input_path: Path = Path("data/interim/clusters")
    histogram_bins: int = 50


class ClusterStatsAnalyzer(SAXShellWorkflow):
    """Skeleton workflow for cluster statistics analysis."""

    workflow_name = "cluster-stats"

    def __init__(self, config: ClusterStatsConfig) -> None:
        super().__init__(config=config)
        self.config: ClusterStatsConfig = config

    def run(self) -> SAXShellResult:
        message = (
            "Initialized cluster statistics with "
            f"cluster_input_path='{self.config.cluster_input_path}', "
            f"histogram_bins={self.config.histogram_bins}"
        )
        return SAXShellResult(
            workflow_name=self.workflow_name,
            message=message,
        )
