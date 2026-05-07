from __future__ import annotations

__all__ = [
    "ElectronDensityMappingMainWindow",
    "launch_electron_density_mapping_ui",
]


def __getattr__(name: str):
    if name in {
        "ElectronDensityMappingMainWindow",
        "launch_electron_density_mapping_ui",
    }:
        from .main_window import (
            ElectronDensityMappingMainWindow,
            launch_electron_density_mapping_ui,
        )

        return {
            "ElectronDensityMappingMainWindow": (
                ElectronDensityMappingMainWindow
            ),
            "launch_electron_density_mapping_ui": (
                launch_electron_density_mapping_ui
            ),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
