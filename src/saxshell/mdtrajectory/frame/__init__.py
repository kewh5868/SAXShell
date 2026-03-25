"""Frame extraction and cutoff-selection tools for SAXSShell."""

from .base import FrameRecord
from .manager import (
    CP2KFrameExtractionWorkflow,
    FrameSelectionPreview,
    TrajectoryManager,
)

__all__ = [
    "CP2KFrameExtractionWorkflow",
    "FrameSelectionPreview",
    "FrameRecord",
    "TrajectoryManager",
]

try:
    from .cp2k_ener import CP2KEnergyData
except ModuleNotFoundError:
    CP2KEnergyData = None
else:
    __all__.append("CP2KEnergyData")

try:
    from .cutoff_analysis import CP2KEnergyAnalyzer, SteadyStateResult
except ModuleNotFoundError:
    CP2KEnergyAnalyzer = None
    SteadyStateResult = None
else:
    __all__.extend(
        [
            "CP2KEnergyAnalyzer",
            "SteadyStateResult",
        ]
    )

try:
    from .cutoff_plot import CP2KEnergyCutoffSelector
except ModuleNotFoundError:
    CP2KEnergyCutoffSelector = None
else:
    __all__.append("CP2KEnergyCutoffSelector")
