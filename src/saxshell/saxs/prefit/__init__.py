from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cluster_geometry import (
        DEFAULT_ANISOTROPY_THRESHOLD,
        ClusterGeometryMetadataRow,
        ClusterGeometryMetadataTable,
        apply_default_component_mapping,
        cluster_identifier,
        compute_cluster_geometry_metadata,
        copy_cluster_geometry_rows,
        load_cluster_geometry_metadata,
        save_cluster_geometry_metadata,
    )
    from .workflow import (
        PrefitComponent,
        PrefitEvaluation,
        PrefitFitResult,
        PrefitParameterEntry,
        PrefitSavedState,
        PrefitScaleRecommendation,
        SAXSPrefitWorkflow,
        normalize_prefit_parameter_expression,
        raise_for_negative_prefit_r_squared,
        resolve_prefit_parameter_entries,
    )

__all__ = [
    "DEFAULT_ANISOTROPY_THRESHOLD",
    "ClusterGeometryMetadataRow",
    "ClusterGeometryMetadataTable",
    "PrefitComponent",
    "PrefitEvaluation",
    "PrefitFitResult",
    "PrefitParameterEntry",
    "PrefitScaleRecommendation",
    "PrefitSavedState",
    "SAXSPrefitWorkflow",
    "apply_default_component_mapping",
    "cluster_identifier",
    "compute_cluster_geometry_metadata",
    "copy_cluster_geometry_rows",
    "load_cluster_geometry_metadata",
    "normalize_prefit_parameter_expression",
    "raise_for_negative_prefit_r_squared",
    "resolve_prefit_parameter_entries",
    "save_cluster_geometry_metadata",
]

_LAZY_EXPORTS = {
    "DEFAULT_ANISOTROPY_THRESHOLD": ("saxshell.saxs.prefit.cluster_geometry"),
    "ClusterGeometryMetadataRow": ("saxshell.saxs.prefit.cluster_geometry"),
    "ClusterGeometryMetadataTable": ("saxshell.saxs.prefit.cluster_geometry"),
    "PrefitComponent": "saxshell.saxs.prefit.workflow",
    "PrefitEvaluation": "saxshell.saxs.prefit.workflow",
    "PrefitFitResult": "saxshell.saxs.prefit.workflow",
    "PrefitParameterEntry": "saxshell.saxs.prefit.workflow",
    "PrefitScaleRecommendation": "saxshell.saxs.prefit.workflow",
    "PrefitSavedState": "saxshell.saxs.prefit.workflow",
    "SAXSPrefitWorkflow": "saxshell.saxs.prefit.workflow",
    "apply_default_component_mapping": (
        "saxshell.saxs.prefit.cluster_geometry"
    ),
    "cluster_identifier": "saxshell.saxs.prefit.cluster_geometry",
    "compute_cluster_geometry_metadata": (
        "saxshell.saxs.prefit.cluster_geometry"
    ),
    "copy_cluster_geometry_rows": "saxshell.saxs.prefit.cluster_geometry",
    "load_cluster_geometry_metadata": (
        "saxshell.saxs.prefit.cluster_geometry"
    ),
    "normalize_prefit_parameter_expression": ("saxshell.saxs.prefit.workflow"),
    "raise_for_negative_prefit_r_squared": "saxshell.saxs.prefit.workflow",
    "resolve_prefit_parameter_entries": "saxshell.saxs.prefit.workflow",
    "save_cluster_geometry_metadata": (
        "saxshell.saxs.prefit.cluster_geometry"
    ),
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
