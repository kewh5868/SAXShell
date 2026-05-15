"""Fullrmc setup scaffolding and launch helpers."""

from typing import TYPE_CHECKING

from .packmol_docker import (
    DEFAULT_PACKMOL_CONTAINER_ROOT,
    PackmolDockerClient,
    PackmolDockerContainerRecord,
    PackmolDockerDirectoryEntry,
    PackmolDockerLink,
    PackmolDockerSyncResult,
    PackmolDockerValidationResult,
    container_project_root_is_valid,
    load_packmol_docker_link_metadata,
    normalize_container_directory,
    save_packmol_docker_link_metadata,
)
from .packmol_planning import (
    PackmolPlanningEntry,
    PackmolPlanningMetadata,
    PackmolPlanningSettings,
    build_packmol_plan,
    load_packmol_planning_metadata,
    save_packmol_planning_metadata,
)
from .packmol_setup import (
    PackmolSetupEntry,
    PackmolSetupMetadata,
    PackmolSetupSettings,
    build_packmol_setup,
    load_packmol_setup_metadata,
    save_packmol_setup_metadata,
)
from .project_loader import (
    RMCDreamProjectSource,
    RMCDreamRunRecord,
    discover_valid_dream_runs,
    load_rmc_project_source,
)
from .project_model import (
    ClusterSourceValidationResult,
    RMCSetupPaths,
    build_rmcsetup_paths,
    collect_cluster_count_rows,
    ensure_rmcsetup_structure,
    expected_cluster_inventory_rows,
    validate_cluster_source,
)
from .representatives import (
    DistributionSelectionEntry,
    DistributionSelectionMetadata,
    RepresentativePreviewCluster,
    RepresentativePreviewSeries,
    RepresentativeSelectionEntry,
    RepresentativeSelectionIssue,
    RepresentativeSelectionMetadata,
    RepresentativeSelectionSettings,
    build_distribution_selection,
    build_representative_preview_clusters,
    load_distribution_selection_metadata,
    load_representative_selection_metadata,
    parse_angle_triplet_text,
    parse_bond_pair_text,
    save_distribution_selection_metadata,
    save_representative_selection_metadata,
    select_distribution_representatives,
    select_first_file_representatives,
)
from .solution_properties import (
    SolutionProperties,
    SolutionPropertiesMetadata,
    SolutionPropertiesResult,
    SolutionPropertiesSettings,
    calculate_solution_properties,
    load_solution_properties_metadata,
    save_solution_properties_metadata,
)
from .solution_property_presets import (
    SolutionPropertiesPreset,
    load_solution_property_presets,
    ordered_solution_property_preset_names,
    save_custom_solution_property_preset,
    solution_property_presets_path,
)
from .solvent_handling import (
    SolventHandlingEntry,
    SolventHandlingMetadata,
    SolventHandlingSettings,
    build_representative_solvent_outputs,
    list_solvent_reference_presets,
    load_solvent_handling_metadata,
    save_solvent_handling_metadata,
)
from .solvent_shell_builder import (
    DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
    SolventShellAnalysisResult,
    SolventShellBuildResult,
    SolventShellResidueMismatchSummary,
    SolventShellResidueSummary,
    analyze_solvent_shell,
    build_solvent_shell_output,
    default_director_atom_name,
    reference_atom_choices,
)

if TYPE_CHECKING:
    from .constraint_generation import (
        ConstraintGenerationEntry,
        ConstraintGenerationMetadata,
        ConstraintGenerationSettings,
        build_constraint_generation,
        load_constraint_generation_metadata,
        save_constraint_generation_metadata,
    )
    from .ui.main_window import RMCSetupMainWindow, launch_rmcsetup_ui
    from .ui.representative_preview_window import RepresentativePreviewWindow
    from .ui.solvent_shell_builder_window import (
        SolventShellBuilderMainWindow,
        launch_solvent_shell_builder_ui,
    )

__all__ = [
    "ClusterSourceValidationResult",
    "ConstraintGenerationEntry",
    "ConstraintGenerationMetadata",
    "ConstraintGenerationSettings",
    "DEFAULT_REFERENCE_MATCH_TOLERANCE_A",
    "DEFAULT_PACKMOL_CONTAINER_ROOT",
    "container_project_root_is_valid",
    "RMCDreamProjectSource",
    "RMCDreamRunRecord",
    "RMCSetupPaths",
    "build_rmcsetup_paths",
    "collect_cluster_count_rows",
    "DistributionSelectionEntry",
    "DistributionSelectionMetadata",
    "PackmolDockerClient",
    "PackmolDockerContainerRecord",
    "PackmolDockerDirectoryEntry",
    "PackmolDockerLink",
    "PackmolDockerSyncResult",
    "PackmolDockerValidationResult",
    "PackmolPlanningEntry",
    "PackmolPlanningMetadata",
    "PackmolPlanningSettings",
    "PackmolSetupEntry",
    "PackmolSetupMetadata",
    "PackmolSetupSettings",
    "RMCSetupMainWindow",
    "RepresentativePreviewCluster",
    "RepresentativePreviewSeries",
    "RepresentativePreviewWindow",
    "RepresentativeSelectionEntry",
    "RepresentativeSelectionIssue",
    "SolventShellBuildResult",
    "build_solvent_shell_output",
    "default_director_atom_name",
    "reference_atom_choices",
    "RepresentativeSelectionMetadata",
    "RepresentativeSelectionSettings",
    "SolutionProperties",
    "SolutionPropertiesMetadata",
    "SolutionPropertiesPreset",
    "SolutionPropertiesResult",
    "SolutionPropertiesSettings",
    "SolventShellAnalysisResult",
    "SolventShellBuilderMainWindow",
    "SolventShellResidueMismatchSummary",
    "SolventShellResidueSummary",
    "SolventHandlingEntry",
    "SolventHandlingMetadata",
    "SolventHandlingSettings",
    "analyze_solvent_shell",
    "build_constraint_generation",
    "build_distribution_selection",
    "build_packmol_setup",
    "build_packmol_plan",
    "build_representative_preview_clusters",
    "build_representative_solvent_outputs",
    "calculate_solution_properties",
    "discover_valid_dream_runs",
    "ensure_rmcsetup_structure",
    "expected_cluster_inventory_rows",
    "launch_solvent_shell_builder_ui",
    "launch_rmcsetup_ui",
    "load_constraint_generation_metadata",
    "load_packmol_docker_link_metadata",
    "load_distribution_selection_metadata",
    "load_packmol_setup_metadata",
    "load_packmol_planning_metadata",
    "load_rmc_project_source",
    "load_representative_selection_metadata",
    "load_solution_properties_metadata",
    "load_solution_property_presets",
    "load_solvent_handling_metadata",
    "list_solvent_reference_presets",
    "ordered_solution_property_preset_names",
    "normalize_container_directory",
    "parse_angle_triplet_text",
    "parse_bond_pair_text",
    "save_constraint_generation_metadata",
    "save_packmol_docker_link_metadata",
    "save_distribution_selection_metadata",
    "save_packmol_setup_metadata",
    "save_packmol_planning_metadata",
    "save_representative_selection_metadata",
    "save_custom_solution_property_preset",
    "save_solution_properties_metadata",
    "save_solvent_handling_metadata",
    "select_distribution_representatives",
    "select_first_file_representatives",
    "solution_property_presets_path",
    "validate_cluster_source",
]


def __getattr__(name: str):
    if name in {
        "ConstraintGenerationEntry",
        "ConstraintGenerationMetadata",
        "ConstraintGenerationSettings",
        "build_constraint_generation",
        "load_constraint_generation_metadata",
        "save_constraint_generation_metadata",
    }:
        from .constraint_generation import (
            ConstraintGenerationEntry,
            ConstraintGenerationMetadata,
            ConstraintGenerationSettings,
            build_constraint_generation,
            load_constraint_generation_metadata,
            save_constraint_generation_metadata,
        )

        exports = {
            "ConstraintGenerationEntry": ConstraintGenerationEntry,
            "ConstraintGenerationMetadata": ConstraintGenerationMetadata,
            "ConstraintGenerationSettings": ConstraintGenerationSettings,
            "build_constraint_generation": build_constraint_generation,
            "load_constraint_generation_metadata": (
                load_constraint_generation_metadata
            ),
            "save_constraint_generation_metadata": (
                save_constraint_generation_metadata
            ),
        }
        return exports[name]
    if name in {"RMCSetupMainWindow", "launch_rmcsetup_ui"}:
        from .ui.main_window import RMCSetupMainWindow, launch_rmcsetup_ui

        exports = {
            "RMCSetupMainWindow": RMCSetupMainWindow,
            "launch_rmcsetup_ui": launch_rmcsetup_ui,
        }
        return exports[name]
    if name in {
        "SolventShellBuilderMainWindow",
        "launch_solvent_shell_builder_ui",
    }:
        from .ui.solvent_shell_builder_window import (
            SolventShellBuilderMainWindow,
            launch_solvent_shell_builder_ui,
        )

        exports = {
            "SolventShellBuilderMainWindow": SolventShellBuilderMainWindow,
            "launch_solvent_shell_builder_ui": (
                launch_solvent_shell_builder_ui
            ),
        }
        return exports[name]
    if name == "RepresentativePreviewWindow":
        from .ui.representative_preview_window import (
            RepresentativePreviewWindow,
        )

        return RepresentativePreviewWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
