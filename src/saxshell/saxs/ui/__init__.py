from __future__ import annotations

__all__ = [
    "DistributionSetupWindow",
    "ExperimentalDataHeaderDialog",
    "PriorHistogramWindow",
    "SAXSProgressDialog",
    "SAXSMainWindow",
    "launch_saxs_ui",
]


def __getattr__(name: str):
    if name == "DistributionSetupWindow":
        from .distribution_window import DistributionSetupWindow

        return DistributionSetupWindow
    if name == "ExperimentalDataHeaderDialog":
        from .experimental_data_loader import ExperimentalDataHeaderDialog

        return ExperimentalDataHeaderDialog
    if name == "PriorHistogramWindow":
        from .prior_histogram_window import PriorHistogramWindow

        return PriorHistogramWindow
    if name == "SAXSProgressDialog":
        from .progress_dialog import SAXSProgressDialog

        return SAXSProgressDialog
    if name in {"SAXSMainWindow", "launch_saxs_ui"}:
        from .main_window import SAXSMainWindow, launch_saxs_ui

        return {
            "SAXSMainWindow": SAXSMainWindow,
            "launch_saxs_ui": launch_saxs_ui,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
