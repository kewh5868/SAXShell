from __future__ import annotations

"""Compatibility wrapper for the new bondanalysis UI.

The old distribution-analysis window mixed bond analysis, motif analysis, and
displacement analysis in one Tk interface. The new application only exposes
bond-pair and angle-distribution analysis in the primary window, while the
legacy displacement workflow is treated as deprecated until it is refreshed.
"""

from pathlib import Path

from .ui.main_window import BondAnalysisMainWindow, launch_bondanalysis_ui, main


class DistributionAnalysisWindow(BondAnalysisMainWindow):
    """Backwards-compatible name for the new bondanalysis main window."""

    def __init__(
        self,
        parent=None,
        proj_dir: str | Path | None = None,
        cluster_dir: str | Path | None = None,
    ) -> None:
        del parent
        del proj_dir
        super().__init__(initial_clusters_dir=cluster_dir)


__all__ = [
    "BondAnalysisMainWindow",
    "DistributionAnalysisWindow",
    "launch_bondanalysis_ui",
    "main",
]
