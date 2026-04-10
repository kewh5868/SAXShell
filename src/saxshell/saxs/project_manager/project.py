from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.saxs.contrast.debye import (
    ContrastDebyeBuildResult,
    build_contrast_component_profiles,
)
from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_NEAT,
    ContrastGeometryDensityResult,
    ContrastGeometryDensitySettings,
    ContrastSolventDensitySettings,
    compute_contrast_geometry_and_electron_density,
)
from saxshell.saxs.contrast.representatives import (
    ContrastRepresentativeSelectionResult,
    analyze_contrast_representatives,
)
from saxshell.saxs.contrast.settings import (
    COMPONENT_BUILD_MODE_CONTRAST,
    COMPONENT_BUILD_MODE_NO_CONTRAST,
    component_build_mode_label,
    normalize_component_build_mode,
)
from saxshell.saxs.debye import (
    ClusterBin,
    DebyeProfileBuilder,
    compute_debye_intensity,
    discover_cluster_bins,
    scan_structure_element_counts,
    scan_structure_elements,
)

from .prior_plot import export_prior_plot_data

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class ProjectPaths:
    project_dir: Path
    project_file: Path
    saved_distributions_dir: Path
    experimental_data_dir: Path
    scattering_components_dir: Path
    predicted_scattering_components_dir: Path
    exported_results_dir: Path
    exported_plots_dir: Path
    exported_data_dir: Path
    plots_dir: Path
    prefit_dir: Path
    cluster_geometry_metadata_file: Path
    predicted_cluster_geometry_metadata_file: Path
    dream_dir: Path
    dream_runtime_dir: Path
    reports_dir: Path


def build_project_paths(project_dir: str | Path) -> ProjectPaths:
    project_dir = Path(project_dir).expanduser().resolve()
    return ProjectPaths(
        project_dir=project_dir,
        project_file=project_dir / "saxs_project.json",
        saved_distributions_dir=project_dir / "saved_distributions",
        experimental_data_dir=project_dir / "experimental_data",
        scattering_components_dir=project_dir / "scattering_components",
        predicted_scattering_components_dir=(
            project_dir / "scattering_components_predicted_structures"
        ),
        exported_results_dir=project_dir / "exported_results",
        exported_plots_dir=project_dir / "exported_results" / "plots",
        exported_data_dir=project_dir / "exported_results" / "data",
        plots_dir=project_dir / "plots",
        prefit_dir=project_dir / "prefit",
        cluster_geometry_metadata_file=(
            project_dir / "prefit" / "cluster_geometry_metadata.json"
        ),
        predicted_cluster_geometry_metadata_file=(
            project_dir
            / "prefit"
            / "cluster_geometry_metadata_predicted_structures.json"
        ),
        dream_dir=project_dir / "dream",
        dream_runtime_dir=project_dir / "dream" / "runtime_scripts",
        reports_dir=project_dir / "reports",
    )


def load_built_component_q_range(
    project_dir: str | Path,
    *,
    include_predicted_structures: bool = False,
    component_dir: str | Path | None = None,
) -> tuple[float, float] | None:
    resolved_component_dir = (
        Path(component_dir).expanduser().resolve()
        if component_dir is not None
        else (
            build_project_paths(
                project_dir
            ).predicted_scattering_components_dir
            if include_predicted_structures
            else build_project_paths(project_dir).scattering_components_dir
        )
    )
    component_files = sorted(resolved_component_dir.glob("*.txt"))
    if not component_files:
        return None

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for component_file in component_files:
        raw_data = np.loadtxt(component_file, comments="#")
        if raw_data.size == 0:
            continue
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        q_values = np.asarray(raw_data[:, 0], dtype=float)
        if q_values.size == 0:
            continue
        lower_bounds.append(float(np.min(q_values)))
        upper_bounds.append(float(np.max(q_values)))
    if not lower_bounds or not upper_bounds:
        return None

    supported_min = max(lower_bounds)
    supported_max = min(upper_bounds)
    if supported_min > supported_max:
        raise ValueError(
            "The built SAXS component files do not share an overlapping q-range."
        )
    return (float(supported_min), float(supported_max))


@dataclass(slots=True)
class ExperimentalDataSummary:
    path: Path
    q_values: np.ndarray
    intensities: np.ndarray
    errors: np.ndarray | None = None
    header_rows: int = 0
    column_names: list[str] = field(default_factory=list)
    q_column: int = 0
    intensity_column: int = 1
    error_column: int | None = None


@dataclass(slots=True)
class ProjectComponentEntry:
    structure: str
    motif: str
    file_count: int
    representative: str | None
    profile_file: str
    source_dir: str


@dataclass(slots=True)
class ProjectBuildResult:
    q_values: np.ndarray
    component_entries: list[ProjectComponentEntry]
    cluster_rows: list[dict[str, object]] = field(default_factory=list)
    staged_experimental_data_path: Path | None = None
    md_prior_weights_path: Path | None = None
    model_map_path: Path | None = None
    prior_plot_data_path: Path | None = None
    used_predicted_structure_weights: bool = False
    predicted_dataset_file: Path | None = None
    predicted_component_count: int = 0


@dataclass(slots=True)
class ClusterImportResult:
    available_elements: list[str]
    cluster_rows: list[dict[str, object]]
    total_files: int


@dataclass(slots=True)
class _ClusterInventory:
    cluster_bins: list[ClusterBin]
    available_elements: list[str]
    cluster_rows: list[dict[str, object]]
    total_files: int


@dataclass(slots=True, frozen=True)
class ProjectArtifactPaths:
    root_dir: Path
    plots_dir: Path
    component_dir: Path
    component_map_file: Path
    contrast_dir: Path
    prior_weights_file: Path
    prior_plot_data_file: Path
    prefit_dir: Path
    cluster_geometry_metadata_file: Path
    predicted_cluster_geometry_metadata_file: Path
    dream_dir: Path
    dream_runtime_dir: Path
    distribution_id: str | None = None
    distribution_metadata_file: Path | None = None
    uses_distribution_storage: bool = False
    includes_predicted_structures: bool = False


@dataclass(slots=True, frozen=True)
class PredictedStructuresProjectState:
    dataset_file: Path | None
    prediction_count: int
    component_artifacts_ready: bool
    prior_artifacts_ready: bool


@dataclass(slots=True, frozen=True)
class SavedDistributionRecord:
    distribution_id: str
    label: str
    distribution_dir: Path
    metadata_path: Path
    created_at: str | None = None
    updated_at: str | None = None
    template_name: str | None = None
    component_build_mode: str = COMPONENT_BUILD_MODE_NO_CONTRAST
    use_predicted_structure_weights: bool = False
    exclude_elements: tuple[str, ...] = ()
    clusters_dir: str | None = None
    q_min: float | None = None
    q_max: float | None = None
    use_experimental_grid: bool = True
    q_points: int | None = None
    component_artifacts_ready: bool = False
    prior_artifacts_ready: bool = False


@dataclass(slots=True)
class DreamBestFitSelection:
    run_name: str
    run_relative_path: str
    bestfit_method: str = "map"
    posterior_filter_mode: str = "all_post_burnin"
    posterior_top_percent: float = 10.0
    posterior_top_n: int = 500
    credible_interval_low: float = 16.0
    credible_interval_high: float = 84.0
    label: str | None = None
    template_name: str | None = None
    model_name: str | None = None
    selection_source: str | None = None
    selected_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "DreamBestFitSelection":
        return cls(
            run_name=str(payload.get("run_name", "")).strip(),
            run_relative_path=str(
                payload.get("run_relative_path", "")
            ).strip(),
            bestfit_method=str(payload.get("bestfit_method", "map")).strip()
            or "map",
            posterior_filter_mode=str(
                payload.get("posterior_filter_mode", "all_post_burnin")
            ).strip()
            or "all_post_burnin",
            posterior_top_percent=float(
                payload.get("posterior_top_percent", 10.0)
            ),
            posterior_top_n=int(payload.get("posterior_top_n", 500)),
            credible_interval_low=float(
                payload.get("credible_interval_low", 16.0)
            ),
            credible_interval_high=float(
                payload.get("credible_interval_high", 84.0)
            ),
            label=_optional_str(payload.get("label")),
            template_name=_optional_str(payload.get("template_name")),
            model_name=_optional_str(payload.get("model_name")),
            selection_source=_optional_str(payload.get("selection_source")),
            selected_at=_optional_str(payload.get("selected_at")),
        )

    def resolved_run_dir(self, project_dir: str | Path) -> Path:
        return (
            Path(project_dir).expanduser().resolve() / self.run_relative_path
        ).resolve()


@dataclass(slots=True)
class PowerPointExportSettings:
    font_family: str = "Arial"
    component_color_map: str = "viridis"
    prior_histogram_color_map: str = "viridis"
    solvent_sort_histogram_color_map: str = "summer"
    text_color: str = "#1f2933"
    experimental_trace_color: str = "#111827"
    model_trace_color: str = "#86d549"
    residual_trace_color: str = "#375a8c"
    solvent_trace_color: str = "#20a386"
    structure_factor_color: str = "#e5e419"
    table_header_fill: str = "#E5E7EB"
    table_even_row_fill: str = "#FFFFFF"
    table_odd_row_fill: str = "#F3F4F6"
    table_rule_color: str = "#4B5563"
    include_prior_histograms: bool = True
    include_initial_traces: bool = True
    include_prefit_model: bool = True
    include_prefit_parameters: bool = True
    include_geometry_table: bool = True
    include_estimator_metrics: bool = True
    include_dream_settings: bool = True
    include_dream_prior_table: bool = True
    include_dream_output_model: bool = True
    include_posterior_comparisons: bool = True
    include_output_summary: bool = True
    include_directory_summary: bool = True
    generate_manifest: bool = True
    export_figure_assets: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> "PowerPointExportSettings":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            font_family=_normalized_nonempty_text(
                payload.get("font_family"),
                default="Arial",
            ),
            component_color_map=_normalized_nonempty_text(
                payload.get("component_color_map"),
                default="viridis",
            ),
            prior_histogram_color_map=_normalized_nonempty_text(
                payload.get("prior_histogram_color_map"),
                default="viridis",
            ),
            solvent_sort_histogram_color_map=_normalized_nonempty_text(
                payload.get("solvent_sort_histogram_color_map"),
                default="summer",
            ),
            text_color=_normalized_hex_color(
                payload.get("text_color"),
                default="#1f2933",
            ),
            experimental_trace_color=_normalized_hex_color(
                payload.get("experimental_trace_color"),
                default="#111827",
            ),
            model_trace_color=_normalized_hex_color(
                payload.get("model_trace_color"),
                default="#86d549",
            ),
            residual_trace_color=_normalized_hex_color(
                payload.get("residual_trace_color"),
                default="#375a8c",
            ),
            solvent_trace_color=_normalized_hex_color(
                payload.get("solvent_trace_color"),
                default="#20a386",
            ),
            structure_factor_color=_normalized_hex_color(
                payload.get("structure_factor_color"),
                default="#e5e419",
            ),
            table_header_fill=_normalized_hex_color(
                payload.get("table_header_fill"),
                default="#E5E7EB",
            ),
            table_even_row_fill=_normalized_hex_color(
                payload.get("table_even_row_fill"),
                default="#FFFFFF",
            ),
            table_odd_row_fill=_normalized_hex_color(
                payload.get("table_odd_row_fill"),
                default="#F3F4F6",
            ),
            table_rule_color=_normalized_hex_color(
                payload.get("table_rule_color"),
                default="#4B5563",
            ),
            include_prior_histograms=bool(
                payload.get("include_prior_histograms", True)
            ),
            include_initial_traces=bool(
                payload.get("include_initial_traces", True)
            ),
            include_prefit_model=bool(
                payload.get("include_prefit_model", True)
            ),
            include_prefit_parameters=bool(
                payload.get("include_prefit_parameters", True)
            ),
            include_geometry_table=bool(
                payload.get("include_geometry_table", True)
            ),
            include_estimator_metrics=bool(
                payload.get("include_estimator_metrics", True)
            ),
            include_dream_settings=bool(
                payload.get("include_dream_settings", True)
            ),
            include_dream_prior_table=bool(
                payload.get("include_dream_prior_table", True)
            ),
            include_dream_output_model=bool(
                payload.get("include_dream_output_model", True)
            ),
            include_posterior_comparisons=bool(
                payload.get("include_posterior_comparisons", True)
            ),
            include_output_summary=bool(
                payload.get("include_output_summary", True)
            ),
            include_directory_summary=bool(
                payload.get("include_directory_summary", True)
            ),
            generate_manifest=bool(payload.get("generate_manifest", True)),
            export_figure_assets=bool(
                payload.get("export_figure_assets", True)
            ),
        )


@dataclass(slots=True, frozen=True)
class RegisteredFolderSnapshot:
    version: int = 1
    signature: str = ""
    file_count: int = 0
    directory_count: int = 0
    total_size_bytes: int = 0
    latest_mtime_ns: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: object,
    ) -> "RegisteredFolderSnapshot | None":
        if not isinstance(payload, dict):
            return None
        signature = _optional_str(payload.get("signature"))
        if signature is None:
            return None
        return cls(
            version=max(int(payload.get("version", 1)), 1),
            signature=signature,
            file_count=max(int(payload.get("file_count", 0)), 0),
            directory_count=max(int(payload.get("directory_count", 0)), 0),
            total_size_bytes=max(int(payload.get("total_size_bytes", 0)), 0),
            latest_mtime_ns=max(int(payload.get("latest_mtime_ns", 0)), 0),
        )


@dataclass(slots=True, frozen=True)
class RegisteredFileSnapshot:
    version: int = 1
    signature: str = ""
    size_bytes: int = 0
    mtime_ns: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: object,
    ) -> "RegisteredFileSnapshot | None":
        if not isinstance(payload, dict):
            return None
        signature = _optional_str(payload.get("signature"))
        if signature is None:
            return None
        return cls(
            version=max(int(payload.get("version", 1)), 1),
            signature=signature,
            size_bytes=max(int(payload.get("size_bytes", 0)), 0),
            mtime_ns=max(int(payload.get("mtime_ns", 0)), 0),
        )


@dataclass(slots=True)
class ProjectSettings:
    project_name: str
    project_dir: str
    model_only_mode: bool = False
    use_predicted_structure_weights: bool = False
    frames_dir: str | None = None
    pdb_frames_dir: str | None = None
    clusters_dir: str | None = None
    frames_dir_snapshot: RegisteredFolderSnapshot | None = None
    pdb_frames_dir_snapshot: RegisteredFolderSnapshot | None = None
    clusters_dir_snapshot: RegisteredFolderSnapshot | None = None
    trajectory_file: str | None = None
    topology_file: str | None = None
    energy_file: str | None = None
    trajectory_file_snapshot: RegisteredFileSnapshot | None = None
    topology_file_snapshot: RegisteredFileSnapshot | None = None
    energy_file_snapshot: RegisteredFileSnapshot | None = None
    experimental_data_path: str | None = None
    copied_experimental_data_file: str | None = None
    solvent_data_path: str | None = None
    copied_solvent_data_file: str | None = None
    experimental_header_rows: int = 0
    experimental_q_column: int | None = None
    experimental_intensity_column: int | None = None
    experimental_error_column: int | None = None
    solvent_header_rows: int = 0
    solvent_q_column: int | None = None
    solvent_intensity_column: int | None = None
    solvent_error_column: int | None = None
    q_min: float | None = None
    q_max: float | None = None
    use_experimental_grid: bool = True
    q_points: int | None = None
    available_elements: list[str] = field(default_factory=list)
    cluster_inventory_rows: list[dict[str, object]] = field(
        default_factory=list
    )
    include_elements: list[str] = field(default_factory=list)
    exclude_elements: list[str] = field(default_factory=list)
    component_trace_colors: dict[str, str] = field(default_factory=dict)
    component_trace_color_scheme: str = "default"
    experimental_trace_visible: bool = True
    experimental_trace_color: str = "#000000"
    solvent_trace_visible: bool = True
    solvent_trace_color: str = "#008000"
    runtime_bundle_opener: str | None = None
    template_reset_template: str | None = None
    template_reset_parameter_entries: list[dict[str, object]] = field(
        default_factory=list
    )
    best_prefit_template: str | None = None
    best_prefit_parameter_entries: list[dict[str, object]] = field(
        default_factory=list
    )
    prefit_sequence_history_enabled: bool = False
    dream_favorite_selection: DreamBestFitSelection | None = None
    dream_favorite_history: list[DreamBestFitSelection] = field(
        default_factory=list
    )
    selected_model_template: str | None = None
    component_build_mode: str = COMPONENT_BUILD_MODE_NO_CONTRAST
    autosave_prefits: bool = False
    powerpoint_export_settings: PowerPointExportSettings = field(
        default_factory=PowerPointExportSettings
    )
    prior_histogram_x_axis_order: list[list[str]] = field(default_factory=list)

    @property
    def resolved_project_dir(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()

    @property
    def resolved_frames_dir(self) -> Path | None:
        if self.frames_dir is None or not self.frames_dir.strip():
            return None
        return Path(self.frames_dir).expanduser().resolve()

    @property
    def resolved_pdb_frames_dir(self) -> Path | None:
        if self.pdb_frames_dir is None or not self.pdb_frames_dir.strip():
            return None
        return Path(self.pdb_frames_dir).expanduser().resolve()

    @property
    def resolved_clusters_dir(self) -> Path | None:
        if self.clusters_dir is None or not self.clusters_dir.strip():
            return None
        return Path(self.clusters_dir).expanduser().resolve()

    @property
    def resolved_trajectory_file(self) -> Path | None:
        if self.trajectory_file is None or not self.trajectory_file.strip():
            return None
        return Path(self.trajectory_file).expanduser().resolve()

    @property
    def resolved_topology_file(self) -> Path | None:
        if self.topology_file is None or not self.topology_file.strip():
            return None
        return Path(self.topology_file).expanduser().resolve()

    @property
    def resolved_energy_file(self) -> Path | None:
        if self.energy_file is None or not self.energy_file.strip():
            return None
        return Path(self.energy_file).expanduser().resolve()

    @property
    def resolved_experimental_data_path(self) -> Path | None:
        if (
            self.copied_experimental_data_file is not None
            and self.copied_experimental_data_file.strip()
        ):
            return (
                Path(self.copied_experimental_data_file).expanduser().resolve()
            )
        if (
            self.experimental_data_path is None
            or not self.experimental_data_path
        ):
            return None
        return Path(self.experimental_data_path).expanduser().resolve()

    @property
    def resolved_solvent_data_path(self) -> Path | None:
        if (
            self.copied_solvent_data_file is not None
            and self.copied_solvent_data_file.strip()
        ):
            return Path(self.copied_solvent_data_file).expanduser().resolve()
        if self.solvent_data_path is None or not self.solvent_data_path:
            return None
        return Path(self.solvent_data_path).expanduser().resolve()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["include_elements"] = list(self.include_elements)
        payload["exclude_elements"] = list(self.exclude_elements)
        payload["cluster_inventory_rows"] = [
            dict(row) for row in self.cluster_inventory_rows
        ]
        payload["component_trace_colors"] = dict(self.component_trace_colors)
        payload["component_trace_color_scheme"] = (
            str(self.component_trace_color_scheme).strip() or "default"
        )
        payload["experimental_trace_visible"] = bool(
            self.experimental_trace_visible
        )
        payload["experimental_trace_color"] = str(
            self.experimental_trace_color
        )
        payload["solvent_trace_visible"] = bool(self.solvent_trace_visible)
        payload["solvent_trace_color"] = str(self.solvent_trace_color)
        payload["runtime_bundle_opener"] = _optional_str(
            self.runtime_bundle_opener
        )
        payload["template_reset_parameter_entries"] = [
            dict(entry) for entry in self.template_reset_parameter_entries
        ]
        payload["best_prefit_parameter_entries"] = [
            dict(entry) for entry in self.best_prefit_parameter_entries
        ]
        payload["dream_favorite_selection"] = (
            None
            if self.dream_favorite_selection is None
            else self.dream_favorite_selection.to_dict()
        )
        payload["dream_favorite_history"] = [
            entry.to_dict() for entry in self.dream_favorite_history
        ]
        payload["frames_dir_snapshot"] = (
            None
            if self.frames_dir_snapshot is None
            else self.frames_dir_snapshot.to_dict()
        )
        payload["pdb_frames_dir_snapshot"] = (
            None
            if self.pdb_frames_dir_snapshot is None
            else self.pdb_frames_dir_snapshot.to_dict()
        )
        payload["clusters_dir_snapshot"] = (
            None
            if self.clusters_dir_snapshot is None
            else self.clusters_dir_snapshot.to_dict()
        )
        payload["trajectory_file_snapshot"] = (
            None
            if self.trajectory_file_snapshot is None
            else self.trajectory_file_snapshot.to_dict()
        )
        payload["topology_file_snapshot"] = (
            None
            if self.topology_file_snapshot is None
            else self.topology_file_snapshot.to_dict()
        )
        payload["energy_file_snapshot"] = (
            None
            if self.energy_file_snapshot is None
            else self.energy_file_snapshot.to_dict()
        )
        payload["powerpoint_export_settings"] = (
            self.powerpoint_export_settings.to_dict()
        )
        payload["component_build_mode"] = normalize_component_build_mode(
            payload.get("component_build_mode")
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProjectSettings":
        return cls(
            project_name=str(payload.get("project_name", "SAXS Project")),
            project_dir=str(payload.get("project_dir", "")),
            model_only_mode=bool(payload.get("model_only_mode", False)),
            use_predicted_structure_weights=bool(
                payload.get("use_predicted_structure_weights", False)
            ),
            frames_dir=_optional_str(payload.get("frames_dir")),
            pdb_frames_dir=_optional_str(payload.get("pdb_frames_dir")),
            clusters_dir=_optional_str(payload.get("clusters_dir")),
            frames_dir_snapshot=RegisteredFolderSnapshot.from_dict(
                payload.get("frames_dir_snapshot")
            ),
            pdb_frames_dir_snapshot=RegisteredFolderSnapshot.from_dict(
                payload.get("pdb_frames_dir_snapshot")
            ),
            clusters_dir_snapshot=RegisteredFolderSnapshot.from_dict(
                payload.get("clusters_dir_snapshot")
            ),
            trajectory_file=_optional_str(payload.get("trajectory_file")),
            topology_file=_optional_str(payload.get("topology_file")),
            energy_file=_optional_str(payload.get("energy_file")),
            trajectory_file_snapshot=RegisteredFileSnapshot.from_dict(
                payload.get("trajectory_file_snapshot")
            ),
            topology_file_snapshot=RegisteredFileSnapshot.from_dict(
                payload.get("topology_file_snapshot")
            ),
            energy_file_snapshot=RegisteredFileSnapshot.from_dict(
                payload.get("energy_file_snapshot")
            ),
            experimental_data_path=_optional_str(
                payload.get("experimental_data_path")
            ),
            copied_experimental_data_file=_optional_str(
                payload.get("copied_experimental_data_file")
            ),
            solvent_data_path=_optional_str(payload.get("solvent_data_path")),
            copied_solvent_data_file=_optional_str(
                payload.get("copied_solvent_data_file")
            ),
            experimental_header_rows=_optional_int(
                payload.get("experimental_header_rows")
            )
            or 0,
            experimental_q_column=_optional_int(
                payload.get("experimental_q_column")
            ),
            experimental_intensity_column=_optional_int(
                payload.get("experimental_intensity_column")
            ),
            experimental_error_column=_optional_int(
                payload.get("experimental_error_column")
            ),
            solvent_header_rows=_optional_int(
                payload.get("solvent_header_rows")
            )
            or 0,
            solvent_q_column=_optional_int(payload.get("solvent_q_column")),
            solvent_intensity_column=_optional_int(
                payload.get("solvent_intensity_column")
            ),
            solvent_error_column=_optional_int(
                payload.get("solvent_error_column")
            ),
            q_min=_optional_float(payload.get("q_min")),
            q_max=_optional_float(payload.get("q_max")),
            use_experimental_grid=bool(
                payload.get(
                    "use_experimental_grid",
                    payload.get("q_points") in (None, ""),
                )
            ),
            q_points=_optional_int(payload.get("q_points")),
            available_elements=_normalized_elements(
                payload.get("available_elements", [])
            ),
            cluster_inventory_rows=_normalized_cluster_inventory_rows(
                payload.get("cluster_inventory_rows", [])
            ),
            include_elements=_normalized_elements(
                payload.get("include_elements", [])
            ),
            exclude_elements=_normalized_elements(
                payload.get("exclude_elements", [])
            ),
            component_trace_colors=_normalized_text_map(
                payload.get("component_trace_colors", {})
            ),
            component_trace_color_scheme=str(
                payload.get("component_trace_color_scheme", "default")
            ).strip()
            or "default",
            experimental_trace_visible=bool(
                payload.get("experimental_trace_visible", True)
            ),
            experimental_trace_color=str(
                payload.get("experimental_trace_color", "#000000")
            ).strip()
            or "#000000",
            solvent_trace_visible=bool(
                payload.get("solvent_trace_visible", True)
            ),
            solvent_trace_color=str(
                payload.get("solvent_trace_color", "#008000")
            ).strip()
            or "#008000",
            runtime_bundle_opener=_optional_str(
                payload.get("runtime_bundle_opener")
            ),
            template_reset_template=_optional_str(
                payload.get("template_reset_template")
            ),
            template_reset_parameter_entries=_normalized_parameter_payloads(
                payload.get("template_reset_parameter_entries", [])
            ),
            best_prefit_template=_optional_str(
                payload.get("best_prefit_template")
            ),
            best_prefit_parameter_entries=_normalized_parameter_payloads(
                payload.get("best_prefit_parameter_entries", [])
            ),
            prefit_sequence_history_enabled=bool(
                payload.get("prefit_sequence_history_enabled", False)
            ),
            dream_favorite_selection=_optional_dream_bestfit_selection(
                payload.get("dream_favorite_selection")
            ),
            dream_favorite_history=_normalized_dream_bestfit_history(
                payload.get("dream_favorite_history", [])
            ),
            selected_model_template=_optional_str(
                payload.get("selected_model_template")
            ),
            component_build_mode=normalize_component_build_mode(
                payload.get("component_build_mode")
            ),
            autosave_prefits=bool(payload.get("autosave_prefits", False)),
            powerpoint_export_settings=PowerPointExportSettings.from_dict(
                payload.get("powerpoint_export_settings", {})
            ),
            prior_histogram_x_axis_order=_normalized_prior_x_axis_order(
                payload.get("prior_histogram_x_axis_order", [])
            ),
        )


def _normalized_prior_x_axis_order(raw: object) -> list[list[str]]:
    if not isinstance(raw, list):
        return []
    result: list[list[str]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            result.append([str(entry[0]), str(entry[1])])
    return result


def _distribution_id_for_settings(
    settings: ProjectSettings,
    *,
    include_template: bool,
    include_build_mode: bool = True,
) -> str:
    payload = _distribution_signature_payload(
        settings,
        include_template=include_template,
        include_build_mode=include_build_mode,
    )
    signature = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha1(signature).hexdigest()[:10]
    mode = (
        "predicted-structures"
        if settings.use_predicted_structure_weights
        else "observed-only"
    )
    excluded = (
        "exclude-" + "-".join(sorted(set(settings.exclude_elements)))
        if settings.exclude_elements
        else "exclude-none"
    )
    return f"{mode}__{excluded}__{digest}"


def distribution_id_for_settings(settings: ProjectSettings) -> str:
    return _distribution_id_for_settings(
        settings,
        include_template=True,
    )


def distribution_label_for_settings(settings: ProjectSettings) -> str:
    mode = (
        "Observed + Predicted Structures"
        if settings.use_predicted_structure_weights
        else "Observed Only"
    )
    build_mode = component_build_mode_label(settings.component_build_mode)
    template_name = (
        str(settings.selected_model_template or "").strip() or "Unspecified"
    )
    excluded = ", ".join(sorted(set(settings.exclude_elements))) or "None"
    if settings.q_min is None or settings.q_max is None:
        q_range = "default"
    else:
        q_range = f"{float(settings.q_min):.6g} to {float(settings.q_max):.6g}"
    grid = (
        "experimental grid"
        if settings.use_experimental_grid
        else f"resample {int(settings.q_points or 0)}"
    )
    return (
        f"{mode} | Build: {build_mode} | Template: {template_name} | "
        f"Excluded: {excluded} | q-range: {q_range} | Grid: {grid}"
    )


def _distribution_id_candidates_for_settings(
    settings: ProjectSettings,
) -> tuple[str, ...]:
    candidates = [
        _distribution_id_for_settings(
            settings,
            include_template=True,
            include_build_mode=True,
        )
    ]
    if (
        normalize_component_build_mode(settings.component_build_mode)
        == COMPONENT_BUILD_MODE_NO_CONTRAST
    ):
        for include_template in (True, False):
            candidate = _distribution_id_for_settings(
                settings,
                include_template=include_template,
                include_build_mode=False,
            )
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def project_artifact_paths(
    settings: ProjectSettings,
    *,
    storage_mode: str = "auto",
    allow_legacy_fallback: bool = True,
) -> ProjectArtifactPaths:
    paths = build_project_paths(settings.project_dir)
    normalized_mode = str(storage_mode).strip().lower() or "auto"
    use_distribution_storage = normalized_mode == "distribution" or (
        normalized_mode == "auto"
        and _project_has_saved_distributions(paths.project_dir)
    )
    root_dir = paths.project_dir
    distribution_id: str | None = None
    distribution_metadata_file: Path | None = None
    uses_distribution_storage = False
    if use_distribution_storage:
        distribution_id_candidates = _distribution_id_candidates_for_settings(
            settings
        )
        desired_distribution_id = distribution_id_candidates[0]
        distribution_id = desired_distribution_id
        root_dir = paths.saved_distributions_dir / desired_distribution_id
        if allow_legacy_fallback and not root_dir.exists():
            for legacy_distribution_id in distribution_id_candidates[1:]:
                legacy_root_dir = (
                    paths.saved_distributions_dir / legacy_distribution_id
                )
                if legacy_root_dir.exists():
                    distribution_id = legacy_distribution_id
                    root_dir = legacy_root_dir
                    break
        distribution_metadata_file = root_dir / "distribution.json"
        uses_distribution_storage = True
    includes_predicted_structures = bool(
        settings.use_predicted_structure_weights
    )
    if includes_predicted_structures:
        component_dir = root_dir / "scattering_components_predicted_structures"
        component_map_file = root_dir / "md_saxs_map_predicted_structures.json"
        prior_weights_file = (
            root_dir / "md_prior_weights_predicted_structures.json"
        )
        prior_plot_data_file = (
            root_dir
            / "plots"
            / "prior_histogram_data_predicted_structures.json"
        )
    else:
        component_dir = root_dir / "scattering_components"
        component_map_file = root_dir / "md_saxs_map.json"
        prior_weights_file = root_dir / "md_prior_weights.json"
        prior_plot_data_file = root_dir / "plots" / "prior_histogram_data.json"
    prefit_dir = root_dir / "prefit"
    dream_dir = root_dir / "dream"
    return ProjectArtifactPaths(
        root_dir=root_dir,
        plots_dir=root_dir / "plots",
        component_dir=component_dir,
        component_map_file=component_map_file,
        contrast_dir=root_dir / "contrast",
        prior_weights_file=prior_weights_file,
        prior_plot_data_file=prior_plot_data_file,
        prefit_dir=prefit_dir,
        cluster_geometry_metadata_file=(
            prefit_dir / "cluster_geometry_metadata.json"
        ),
        predicted_cluster_geometry_metadata_file=(
            prefit_dir / "cluster_geometry_metadata_predicted_structures.json"
        ),
        dream_dir=dream_dir,
        dream_runtime_dir=dream_dir / "runtime_scripts",
        distribution_id=distribution_id,
        distribution_metadata_file=distribution_metadata_file,
        uses_distribution_storage=uses_distribution_storage,
        includes_predicted_structures=includes_predicted_structures,
    )


def _distribution_signature_payload(
    settings: ProjectSettings,
    *,
    include_template: bool = True,
    include_build_mode: bool = True,
) -> dict[str, object]:
    experimental_source = _distribution_experimental_data_path(settings)
    q_min, q_max = _distribution_signature_q_range(settings)
    payload = {
        "clusters_dir": (
            None
            if settings.resolved_clusters_dir is None
            else str(settings.resolved_clusters_dir)
        ),
        "use_predicted_structure_weights": bool(
            settings.use_predicted_structure_weights
        ),
        "exclude_elements": sorted(set(settings.exclude_elements)),
        "q_min": q_min,
        "q_max": q_max,
        "use_experimental_grid": bool(settings.use_experimental_grid),
        "q_points": (
            None
            if settings.use_experimental_grid
            else _optional_int(settings.q_points)
        ),
        "model_only_mode": bool(settings.model_only_mode),
        "experimental_data_path": (
            None
            if settings.use_experimental_grid is False
            else (
                None
                if experimental_source is None
                else str(experimental_source)
            )
        ),
    }
    if include_build_mode:
        payload["component_build_mode"] = normalize_component_build_mode(
            settings.component_build_mode
        )
    if include_template:
        payload["selected_model_template"] = _optional_str(
            settings.selected_model_template
        )
    return payload


def _distribution_experimental_data_path(
    settings: ProjectSettings,
) -> str | None:
    for candidate in (
        settings.experimental_data_path,
        settings.copied_experimental_data_file,
    ):
        text = str(candidate or "").strip()
        if not text:
            continue
        try:
            return str(Path(text).expanduser().resolve())
        except Exception:
            return text
    return None


def _distribution_signature_q_range(
    settings: ProjectSettings,
) -> tuple[float | None, float | None]:
    q_min = _optional_float(settings.q_min)
    q_max = _optional_float(settings.q_max)
    if (
        settings.model_only_mode
        or not settings.use_experimental_grid
        or q_min is None
        or q_max is None
    ):
        return q_min, q_max
    default_range = _distribution_default_experimental_q_range(settings)
    if default_range is None:
        return q_min, q_max
    default_q_min, default_q_max = default_range
    tolerance = _distribution_q_range_tolerance(
        default_q_min,
        default_q_max,
    )
    if (
        abs(q_min - default_q_min) <= tolerance
        and abs(q_max - default_q_max) <= tolerance
    ):
        return None, None
    return q_min, q_max


def _distribution_default_experimental_q_range(
    settings: ProjectSettings,
) -> tuple[float, float] | None:
    experimental_source = _distribution_experimental_data_path(settings)
    if experimental_source is None:
        return None
    try:
        summary = load_experimental_data_file(
            experimental_source,
            skiprows=settings.experimental_header_rows,
            q_column=settings.experimental_q_column,
            intensity_column=settings.experimental_intensity_column,
            error_column=settings.experimental_error_column,
        )
    except Exception:
        return None
    q_values = np.asarray(summary.q_values, dtype=float)
    if q_values.size == 0:
        return None
    return float(np.min(q_values)), float(np.max(q_values))


def _distribution_q_range_tolerance(
    lower: float,
    upper: float,
) -> float:
    return max(
        1e-12,
        1e-9 * max(abs(lower), abs(upper), 1.0),
    )


def effective_q_range_for_settings(
    settings: ProjectSettings,
    source_q_values: np.ndarray,
) -> tuple[float, float]:
    effective_q = _requested_q_values_on_source_grid(
        settings,
        source_q_values,
    )
    return float(np.min(effective_q)), float(np.max(effective_q))


def _project_has_saved_distributions(project_dir: str | Path) -> bool:
    saved_dir = build_project_paths(project_dir).saved_distributions_dir
    if not saved_dir.is_dir():
        return False
    return any(path.is_dir() for path in saved_dir.iterdir())


def _distribution_metadata_from_payload(
    distribution_dir: Path,
    metadata_path: Path,
    payload: dict[str, object],
) -> SavedDistributionRecord | None:
    distribution_id = str(payload.get("distribution_id", "")).strip()
    label = str(payload.get("label", "")).strip()
    if not distribution_id or not label:
        return None
    exclude_elements = tuple(
        _normalized_elements(payload.get("exclude_elements", []))
    )
    return SavedDistributionRecord(
        distribution_id=distribution_id,
        label=label,
        distribution_dir=distribution_dir,
        metadata_path=metadata_path,
        created_at=_optional_str(payload.get("created_at")),
        updated_at=_optional_str(payload.get("updated_at")),
        template_name=_optional_str(payload.get("template_name")),
        component_build_mode=normalize_component_build_mode(
            payload.get("component_build_mode")
        ),
        use_predicted_structure_weights=bool(
            payload.get("use_predicted_structure_weights", False)
        ),
        exclude_elements=exclude_elements,
        clusters_dir=_optional_str(payload.get("clusters_dir")),
        q_min=_optional_float(payload.get("q_min")),
        q_max=_optional_float(payload.get("q_max")),
        use_experimental_grid=bool(payload.get("use_experimental_grid", True)),
        q_points=_optional_int(payload.get("q_points")),
        component_artifacts_ready=bool(
            payload.get("component_artifacts_ready", False)
        ),
        prior_artifacts_ready=bool(
            payload.get("prior_artifacts_ready", False)
        ),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalized_nonempty_text(value: object, *, default: str) -> str:
    text = _optional_str(value)
    return text if text is not None else default


def _normalized_hex_color(value: object, *, default: str) -> str:
    text = _optional_str(value)
    if text is None:
        return default
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.upper()
    return default


def _normalized_elements(values: object) -> list[str]:
    if isinstance(values, str):
        raw_values = [
            token.strip()
            for token in values.replace(",", " ").split()
            if token.strip()
        ]
    else:
        raw_values = [
            str(token).strip() for token in values if str(token).strip()
        ]
    return [token[:1].upper() + token[1:].lower() for token in raw_values]


def _normalized_text_map(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in values.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if normalized_key and normalized_value:
            normalized[normalized_key] = normalized_value
    return normalized


def _normalized_parameter_payloads(values: object) -> list[dict[str, object]]:
    if not isinstance(values, list):
        return []
    payloads: list[dict[str, object]] = []
    for value in values:
        if isinstance(value, dict):
            payloads.append(dict(value))
    return payloads


def _normalized_cluster_inventory_rows(
    values: object,
) -> list[dict[str, object]]:
    if not isinstance(values, list):
        return []
    rows: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        structure = str(value.get("structure", "")).strip()
        motif = str(value.get("motif", "no_motif")).strip() or "no_motif"
        if not structure:
            continue
        count = int(_optional_int(value.get("count")) or 0)
        row: dict[str, object] = {
            "structure": structure,
            "motif": motif,
            "count": count,
        }
        for key in (
            "source_kind",
            "source_dir",
            "source_file",
            "source_file_name",
            "representative",
            "profile_file",
        ):
            normalized_value = _optional_str(value.get(key))
            if normalized_value is not None:
                row[key] = normalized_value
        for key in (
            "weight",
            "atom_fraction_percent",
            "structure_fraction_percent",
        ):
            number = _optional_float(value.get(key))
            if number is not None:
                row[key] = number
        rows.append(row)
    return rows


def _optional_dream_bestfit_selection(
    value: object,
) -> DreamBestFitSelection | None:
    if not isinstance(value, dict):
        return None
    run_name = str(value.get("run_name", "")).strip()
    run_relative_path = str(value.get("run_relative_path", "")).strip()
    if not run_name or not run_relative_path:
        return None
    return DreamBestFitSelection.from_dict(dict(value))


def _normalized_dream_bestfit_history(
    values: object,
) -> list[DreamBestFitSelection]:
    if not isinstance(values, list):
        return []
    history: list[DreamBestFitSelection] = []
    for value in values:
        entry = _optional_dream_bestfit_selection(value)
        if entry is not None:
            history.append(entry)
    return history


def _registered_folder_snapshot(
    folder_path: str | Path | None,
) -> RegisteredFolderSnapshot | None:
    if folder_path is None:
        return None
    try:
        resolved_dir = Path(folder_path).expanduser().resolve()
    except Exception:
        return None
    if not resolved_dir.is_dir():
        return None

    digest = hashlib.sha1()
    file_count = 0
    directory_count = 0
    total_size_bytes = 0
    latest_mtime_ns = 0

    try:
        root_stat = resolved_dir.stat()
        root_mtime_ns = int(
            getattr(
                root_stat,
                "st_mtime_ns",
                int(float(root_stat.st_mtime) * 1_000_000_000),
            )
        )
        latest_mtime_ns = max(latest_mtime_ns, root_mtime_ns)
        digest.update(
            f"root\0{resolved_dir.name}\0{root_mtime_ns}\n".encode("utf-8")
        )
        for path in sorted(resolved_dir.rglob("*")):
            try:
                path_stat = path.stat()
            except OSError:
                return None
            relative_path = path.relative_to(resolved_dir).as_posix()
            is_directory = path.is_dir()
            entry_type = "d" if is_directory else "f"
            entry_size = 0 if is_directory else int(path_stat.st_size)
            entry_mtime_ns = int(
                getattr(
                    path_stat,
                    "st_mtime_ns",
                    int(float(path_stat.st_mtime) * 1_000_000_000),
                )
            )
            digest.update(
                (
                    f"{entry_type}\0{relative_path}\0{entry_size}\0"
                    f"{entry_mtime_ns}\n"
                ).encode("utf-8")
            )
            latest_mtime_ns = max(latest_mtime_ns, entry_mtime_ns)
            if is_directory:
                directory_count += 1
            else:
                file_count += 1
                total_size_bytes += entry_size
    except OSError:
        return None

    return RegisteredFolderSnapshot(
        signature=digest.hexdigest(),
        file_count=file_count,
        directory_count=directory_count,
        total_size_bytes=total_size_bytes,
        latest_mtime_ns=latest_mtime_ns,
    )


def _registered_file_snapshot(
    file_path: str | Path | None,
) -> RegisteredFileSnapshot | None:
    if file_path is None:
        return None
    try:
        resolved_file = Path(file_path).expanduser().resolve()
    except Exception:
        return None
    if not resolved_file.is_file():
        return None

    try:
        file_stat = resolved_file.stat()
    except OSError:
        return None

    size_bytes = int(file_stat.st_size)
    mtime_ns = int(
        getattr(
            file_stat,
            "st_mtime_ns",
            int(float(file_stat.st_mtime) * 1_000_000_000),
        )
    )
    digest = hashlib.sha1()
    digest.update(f"size\0{size_bytes}\0mtime\0{mtime_ns}\n".encode("utf-8"))
    try:
        with resolved_file.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None

    return RegisteredFileSnapshot(
        signature=digest.hexdigest(),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
    )


def _refresh_registered_folder_snapshots(
    settings: ProjectSettings,
) -> None:
    settings.frames_dir_snapshot = _registered_folder_snapshot(
        settings.frames_dir
    )
    settings.pdb_frames_dir_snapshot = _registered_folder_snapshot(
        settings.pdb_frames_dir
    )
    settings.clusters_dir_snapshot = _registered_folder_snapshot(
        settings.clusters_dir
    )


def _refresh_registered_file_snapshots(
    settings: ProjectSettings,
) -> None:
    settings.trajectory_file_snapshot = _registered_file_snapshot(
        settings.trajectory_file
    )
    settings.topology_file_snapshot = _registered_file_snapshot(
        settings.topology_file
    )
    settings.energy_file_snapshot = _registered_file_snapshot(
        settings.energy_file
    )


def _registered_folder_validation_messages(
    *,
    label: str,
    path_text: str | None,
    saved_snapshot: RegisteredFolderSnapshot | None,
) -> list[str]:
    if path_text is None or not str(path_text).strip():
        return []
    try:
        resolved_dir = Path(path_text).expanduser().resolve()
    except Exception:
        return [f"Registered {label} folder path is invalid: {path_text}"]
    if not resolved_dir.exists():
        return [f"Registered {label} folder is missing: {resolved_dir}"]
    if not resolved_dir.is_dir():
        return [
            f"Registered {label} folder is no longer a directory: {resolved_dir}"
        ]
    if saved_snapshot is None:
        return []

    current_snapshot = _registered_folder_snapshot(resolved_dir)
    if current_snapshot is None:
        return [
            f"Registered {label} folder could not be scanned: {resolved_dir}"
        ]
    if current_snapshot.signature == saved_snapshot.signature:
        return []

    details: list[str] = []
    if current_snapshot.file_count != saved_snapshot.file_count:
        details.append(
            f"files {saved_snapshot.file_count} -> {current_snapshot.file_count}"
        )
    if current_snapshot.directory_count != saved_snapshot.directory_count:
        details.append(
            "subfolders "
            f"{saved_snapshot.directory_count} -> {current_snapshot.directory_count}"
        )
    if current_snapshot.total_size_bytes != saved_snapshot.total_size_bytes:
        details.append(
            "bytes "
            f"{saved_snapshot.total_size_bytes} -> {current_snapshot.total_size_bytes}"
        )
    if not details:
        details.append("file metadata changed in place")
    return [
        "Registered "
        f"{label} folder contents changed since the project was last saved: "
        f"{resolved_dir} ({'; '.join(details)})"
    ]


def _registered_file_validation_messages(
    *,
    label: str,
    path_text: str | None,
    saved_snapshot: RegisteredFileSnapshot | None,
) -> list[str]:
    if path_text is None or not str(path_text).strip():
        return []
    try:
        resolved_file = Path(path_text).expanduser().resolve()
    except Exception:
        return [f"Registered {label} file path is invalid: {path_text}"]
    if not resolved_file.exists():
        return [f"Registered {label} file is missing: {resolved_file}"]
    if not resolved_file.is_file():
        return [
            f"Registered {label} path is no longer a file: {resolved_file}"
        ]
    if saved_snapshot is None:
        return []

    current_snapshot = _registered_file_snapshot(resolved_file)
    if current_snapshot is None:
        return [f"Registered {label} file could not be read: {resolved_file}"]
    if current_snapshot.signature == saved_snapshot.signature:
        return []

    details: list[str] = []
    if current_snapshot.size_bytes != saved_snapshot.size_bytes:
        details.append(
            f"bytes {saved_snapshot.size_bytes} -> {current_snapshot.size_bytes}"
        )
    if current_snapshot.mtime_ns != saved_snapshot.mtime_ns:
        details.append("modified timestamp changed")
    if not details:
        details.append("file contents changed in place")
    return [
        "Registered "
        f"{label} file changed since the project was last saved: "
        f"{resolved_file} ({'; '.join(details)})"
    ]


class SAXSProjectManager:
    """Persist project settings and build SAXS component files inside
    one project directory."""

    def __init__(self) -> None:
        self._cluster_inventory_cache: dict[Path, _ClusterInventory] = {}

    def create_project(
        self,
        project_dir: str | Path,
        *,
        project_name: str | None = None,
    ) -> ProjectSettings:
        paths = build_project_paths(project_dir)
        self.ensure_project_dirs(paths)
        settings = ProjectSettings(
            project_name=project_name or paths.project_dir.name,
            project_dir=str(paths.project_dir),
        )
        self.save_project(
            settings,
            refresh_registered_paths=False,
        )
        return settings

    def load_project(
        self,
        project_dir_or_file: str | Path,
    ) -> ProjectSettings:
        candidate = Path(project_dir_or_file).expanduser().resolve()
        project_file = (
            candidate
            if candidate.name.endswith(".json")
            else build_project_paths(candidate).project_file
        )
        payload = json.loads(project_file.read_text(encoding="utf-8"))
        settings = ProjectSettings.from_dict(payload)
        self.ensure_project_dirs(build_project_paths(settings.project_dir))
        return settings

    def registered_folder_warnings(
        self,
        settings: ProjectSettings,
    ) -> tuple[str, ...]:
        messages: list[str] = []
        messages.extend(
            _registered_folder_validation_messages(
                label="frames",
                path_text=settings.frames_dir,
                saved_snapshot=settings.frames_dir_snapshot,
            )
        )
        messages.extend(
            _registered_folder_validation_messages(
                label="PDB structure",
                path_text=settings.pdb_frames_dir,
                saved_snapshot=settings.pdb_frames_dir_snapshot,
            )
        )
        messages.extend(
            _registered_folder_validation_messages(
                label="clusters",
                path_text=settings.clusters_dir,
                saved_snapshot=settings.clusters_dir_snapshot,
            )
        )
        messages.extend(
            _registered_file_validation_messages(
                label="trajectory",
                path_text=settings.trajectory_file,
                saved_snapshot=settings.trajectory_file_snapshot,
            )
        )
        messages.extend(
            _registered_file_validation_messages(
                label="topology",
                path_text=settings.topology_file,
                saved_snapshot=settings.topology_file_snapshot,
            )
        )
        messages.extend(
            _registered_file_validation_messages(
                label="energy",
                path_text=settings.energy_file,
                saved_snapshot=settings.energy_file_snapshot,
            )
        )
        return tuple(messages)

    def save_project(
        self,
        settings: ProjectSettings,
        *,
        refresh_registered_paths: bool = True,
    ) -> Path:
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        if refresh_registered_paths:
            _refresh_registered_folder_snapshots(settings)
            _refresh_registered_file_snapshots(settings)
        paths.project_file.write_text(
            json.dumps(settings.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return paths.project_file

    def ensure_project_dirs(self, paths: ProjectPaths) -> None:
        for directory in (
            paths.project_dir,
            paths.saved_distributions_dir,
            paths.experimental_data_dir,
            paths.scattering_components_dir,
            paths.predicted_scattering_components_dir,
            paths.exported_results_dir,
            paths.exported_plots_dir,
            paths.exported_data_dir,
            paths.plots_dir,
            paths.prefit_dir,
            paths.dream_dir,
            paths.dream_runtime_dir,
            paths.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_artifact_dirs(
        self,
        artifact_paths: ProjectArtifactPaths,
    ) -> None:
        for directory in (
            artifact_paths.root_dir,
            artifact_paths.plots_dir,
            artifact_paths.component_dir,
            artifact_paths.contrast_dir,
            artifact_paths.prefit_dir,
            artifact_paths.dream_dir,
            artifact_paths.dream_runtime_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def stage_experimental_data(
        self,
        settings: ProjectSettings,
    ) -> Path:
        source = self._resolve_experimental_source(settings)
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        destination = paths.experimental_data_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        settings.copied_experimental_data_file = str(destination)
        return destination

    def stage_solvent_data(
        self,
        settings: ProjectSettings,
    ) -> Path | None:
        source = self._resolve_solvent_source(settings)
        if source is None:
            settings.copied_solvent_data_file = None
            return None
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        destination = paths.experimental_data_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        settings.copied_solvent_data_file = str(destination)
        return destination

    def load_experimental_data(
        self,
        settings: ProjectSettings,
    ) -> ExperimentalDataSummary:
        path = self._resolve_experimental_source(settings)
        return load_experimental_data_file(
            path,
            skiprows=settings.experimental_header_rows,
            q_column=settings.experimental_q_column,
            intensity_column=settings.experimental_intensity_column,
            error_column=settings.experimental_error_column,
        )

    def load_solvent_data(
        self,
        settings: ProjectSettings,
    ) -> ExperimentalDataSummary | None:
        path = self._resolve_solvent_source(settings)
        if path is None:
            return None
        return load_experimental_data_file(
            path,
            skiprows=settings.solvent_header_rows,
            q_column=settings.solvent_q_column,
            intensity_column=settings.solvent_intensity_column,
            error_column=settings.solvent_error_column,
        )

    def scan_cluster_inventory(
        self,
        clusters_dir: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ClusterImportResult:
        inventory = self._collect_cluster_inventory(
            clusters_dir,
            progress_callback=progress_callback,
            force_refresh=True,
        )
        return ClusterImportResult(
            available_elements=inventory.available_elements,
            cluster_rows=inventory.cluster_rows,
            total_files=inventory.total_files,
        )

    def inspect_predicted_structures(
        self,
        project_dir: str | Path,
    ) -> PredictedStructuresProjectState:
        paths = build_project_paths(project_dir)
        dataset_file = self._latest_predicted_structures_dataset_file(
            paths.project_dir
        )
        prediction_count = 0
        if dataset_file is not None:
            try:
                payload = json.loads(dataset_file.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            prediction_count = len(payload.get("predictions", []))
        component_artifacts_ready = bool(
            paths.predicted_scattering_components_dir.is_dir()
            and any(paths.predicted_scattering_components_dir.glob("*.txt"))
            and (
                paths.project_dir / "md_saxs_map_predicted_structures.json"
            ).is_file()
        )
        prior_artifacts_ready = bool(
            (
                paths.project_dir
                / "md_prior_weights_predicted_structures.json"
            ).is_file()
        )
        return PredictedStructuresProjectState(
            dataset_file=dataset_file,
            prediction_count=int(prediction_count),
            component_artifacts_ready=component_artifacts_ready,
            prior_artifacts_ready=prior_artifacts_ready,
        )

    def list_saved_distributions(
        self,
        project_dir: str | Path,
    ) -> list[SavedDistributionRecord]:
        saved_dir = build_project_paths(project_dir).saved_distributions_dir
        if not saved_dir.is_dir():
            return []
        records: list[SavedDistributionRecord] = []
        for distribution_dir in saved_dir.iterdir():
            if not distribution_dir.is_dir():
                continue
            metadata_path = distribution_dir / "distribution.json"
            if not metadata_path.is_file():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            record = _distribution_metadata_from_payload(
                distribution_dir,
                metadata_path,
                payload,
            )
            if record is not None:
                records.append(record)
        records.sort(
            key=lambda record: (
                record.updated_at or "",
                record.created_at or "",
                record.label.lower(),
            ),
            reverse=True,
        )
        return records

    def load_saved_distribution(
        self,
        project_dir: str | Path,
        distribution_id: str,
    ) -> SavedDistributionRecord:
        distribution_dir = (
            build_project_paths(project_dir).saved_distributions_dir
            / str(distribution_id).strip()
        )
        metadata_path = distribution_dir / "distribution.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "No saved distribution metadata was found for "
                f"{distribution_id}."
            )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = _distribution_metadata_from_payload(
            distribution_dir,
            metadata_path,
            payload,
        )
        if record is None:
            raise ValueError(
                f"Saved distribution metadata in {metadata_path} is invalid."
            )
        return record

    def settings_for_saved_distribution(
        self,
        project_dir: str | Path,
        distribution_id: str,
        *,
        base_settings: ProjectSettings | None = None,
    ) -> ProjectSettings:
        record = self.load_saved_distribution(project_dir, distribution_id)
        working_settings = ProjectSettings.from_dict(
            (
                base_settings.to_dict()
                if base_settings is not None
                else self.load_project(project_dir).to_dict()
            )
        )
        working_settings.use_predicted_structure_weights = bool(
            record.use_predicted_structure_weights
        )
        working_settings.exclude_elements = list(record.exclude_elements)
        working_settings.clusters_dir = record.clusters_dir
        working_settings.q_min = record.q_min
        working_settings.q_max = record.q_max
        working_settings.use_experimental_grid = bool(
            record.use_experimental_grid
        )
        working_settings.q_points = record.q_points
        working_settings.component_build_mode = record.component_build_mode
        if record.template_name is not None:
            working_settings.selected_model_template = record.template_name
        return working_settings

    def clone_distribution_for_template(
        self,
        project_dir: str | Path,
        source_distribution_id: str,
        template_name: str,
        *,
        base_settings: ProjectSettings | None = None,
    ) -> SavedDistributionRecord:
        normalized_template = _optional_str(template_name)
        if normalized_template is None:
            raise ValueError("Select a template before changing it.")
        source_record = self.load_saved_distribution(
            project_dir,
            source_distribution_id,
        )
        target_settings = self.settings_for_saved_distribution(
            project_dir,
            source_distribution_id,
            base_settings=base_settings,
        )
        target_settings.selected_model_template = normalized_template
        target_artifact_paths = project_artifact_paths(
            target_settings,
            storage_mode="distribution",
            allow_legacy_fallback=False,
        )
        if target_artifact_paths.distribution_id is None:
            raise ValueError(
                "The template-scoped distribution path could not be resolved."
            )
        if target_artifact_paths.distribution_metadata_file is None:
            raise ValueError(
                "The template-scoped distribution metadata path is unavailable."
            )
        if target_artifact_paths.distribution_metadata_file.is_file():
            return self.load_saved_distribution(
                project_dir,
                target_artifact_paths.distribution_id,
            )

        self.ensure_project_dirs(build_project_paths(project_dir))
        self.ensure_artifact_dirs(target_artifact_paths)
        self._copy_distribution_artifacts(
            source_root=source_record.distribution_dir,
            target_artifact_paths=target_artifact_paths,
            include_predicted_structures=bool(
                source_record.use_predicted_structure_weights
            ),
        )
        self._write_distribution_metadata(
            target_settings,
            artifact_paths=target_artifact_paths,
        )
        return self.load_saved_distribution(
            project_dir,
            target_artifact_paths.distribution_id,
        )

    def predicted_structure_cluster_bins(
        self,
        project_dir: str | Path,
        *,
        included_components: set[tuple[str, str]] | None = None,
    ) -> list[ClusterBin]:
        loaded_dataset = self._load_latest_predicted_structures_dataset(
            project_dir
        )
        requested_components = (
            {
                (
                    str(structure).strip(),
                    str(motif).strip() or "no_motif",
                )
                for structure, motif in included_components
            }
            if included_components is not None
            else None
        )
        cluster_bins: list[ClusterBin] = []
        discovered_components: set[tuple[str, str]] = set()
        for prediction in loaded_dataset.result.predictions:
            structure = str(prediction.label).strip()
            motif = _predicted_structure_motif(int(prediction.rank))
            component_key = (structure, motif)
            if (
                requested_components is not None
                and component_key not in requested_components
            ):
                continue
            source_path = self._predicted_structure_source_path(
                loaded_dataset,
                prediction,
            )
            if source_path is None or not source_path.is_file():
                raise FileNotFoundError(
                    "Predicted Structures mode is enabled, but the XYZ file "
                    f"for predicted structure {structure}/{motif} could not "
                    "be found in this project. Re-run Cluster Dynamics ML "
                    "or rebuild the prediction bundle before computing "
                    "cluster geometry metadata."
                )
            cluster_bins.append(
                ClusterBin(
                    structure=structure,
                    motif=motif,
                    source_dir=source_path.parent,
                    files=(source_path,),
                    representative=source_path.name,
                )
            )
            discovered_components.add(component_key)
        if requested_components is not None:
            missing_components = sorted(
                requested_components - discovered_components,
                key=lambda item: (
                    _natural_sort_key(item[0]),
                    _natural_sort_key(item[1]),
                ),
            )
            if missing_components:
                missing_text = ", ".join(
                    f"{structure}/{motif}"
                    for structure, motif in missing_components
                )
                raise FileNotFoundError(
                    "Predicted Structures mode is enabled, but the "
                    "geometry source files for these predicted structures "
                    f"could not be resolved: {missing_text}."
                )
        return cluster_bins

    def contrast_cluster_inventory(
        self,
        settings: ProjectSettings,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[_ClusterInventory, Path | None, int]:
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory before building contrast models."
            )
        observed_inventory = self._collect_cluster_inventory(
            clusters_dir,
            progress_callback=progress_callback,
        )
        if not settings.use_predicted_structure_weights:
            return observed_inventory, None, 0
        (
            loaded_dataset,
            observed_motif_weights,
            predicted_payloads,
            predicted_available_elements,
        ) = self._predicted_structure_weight_payload(
            settings,
            cluster_inventory=observed_inventory,
        )
        predicted_cluster_bins = (
            self._predicted_structure_cluster_bins_from_payloads(
                predicted_payloads
            )
        )
        merged_cluster_bins = sorted(
            [
                *observed_inventory.cluster_bins,
                *predicted_cluster_bins,
            ],
            key=lambda cluster_bin: (
                _natural_sort_key(cluster_bin.structure),
                _natural_sort_key(cluster_bin.motif),
            ),
        )
        merged_available_elements = sorted(
            {
                *observed_inventory.available_elements,
                *predicted_available_elements,
            },
            key=_natural_sort_key,
        )
        cluster_rows = self._combined_cluster_rows(
            cluster_inventory=observed_inventory,
            observed_motif_weights=observed_motif_weights,
            predicted_payloads=predicted_payloads,
        )
        return (
            _ClusterInventory(
                cluster_bins=merged_cluster_bins,
                available_elements=merged_available_elements,
                cluster_rows=cluster_rows,
                total_files=sum(
                    len(cluster_bin.files)
                    for cluster_bin in merged_cluster_bins
                ),
            ),
            loaded_dataset.dataset_file,
            len(predicted_payloads),
        )

    def _predicted_structure_cluster_bins_from_payloads(
        self,
        predicted_payloads: list[dict[str, object]],
    ) -> list[ClusterBin]:
        cluster_bins: list[ClusterBin] = []
        for payload in predicted_payloads:
            prediction = payload.get("prediction")
            structure = str(getattr(prediction, "label", "")).strip()
            motif = (
                str(payload.get("motif") or "no_motif").strip() or "no_motif"
            )
            if not structure:
                continue
            source_path_value = payload.get("source_path")
            source_path = (
                None
                if source_path_value is None
                else Path(source_path_value).expanduser().resolve()
            )
            if source_path is None or not source_path.is_file():
                raise FileNotFoundError(
                    "Predicted Structures mode is enabled, but the XYZ/PDB file "
                    f"for predicted structure {structure}/{motif} could not be found."
                )
            cluster_bins.append(
                ClusterBin(
                    structure=structure,
                    motif=motif,
                    source_dir=source_path.parent,
                    files=(source_path,),
                    representative=source_path.name,
                )
            )
        return cluster_bins

    def build_scattering_project(
        self,
        settings: ProjectSettings,
    ) -> ProjectBuildResult:
        component_result = self.build_scattering_components(settings)
        prior_result = self.generate_prior_weights(settings)
        return ProjectBuildResult(
            q_values=component_result.q_values,
            component_entries=component_result.component_entries,
            cluster_rows=component_result.cluster_rows,
            staged_experimental_data_path=(
                component_result.staged_experimental_data_path
            ),
            md_prior_weights_path=prior_result.md_prior_weights_path,
            model_map_path=component_result.model_map_path,
            prior_plot_data_path=prior_result.prior_plot_data_path,
        )

    def build_scattering_components(
        self,
        settings: ProjectSettings,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectBuildResult:
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        staged_data_path: Path | None = None
        experimental_data: ExperimentalDataSummary | None = None
        if not settings.model_only_mode:
            staged_data_path = self.stage_experimental_data(settings)
            self.stage_solvent_data(settings)
            experimental_data = self.load_experimental_data(settings)
        q_values = self._build_q_grid(settings, experimental_data)
        self._store_effective_q_range(
            settings,
            q_values=q_values,
            experimental_data=experimental_data,
        )
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
        )
        self.ensure_artifact_dirs(artifact_paths)
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory before building models."
            )
        predicted_dataset_file: Path | None = None
        predicted_component_count = 0
        if (
            normalize_component_build_mode(settings.component_build_mode)
            == COMPONENT_BUILD_MODE_CONTRAST
        ):
            (
                cluster_inventory,
                predicted_dataset_file,
                predicted_component_count,
            ) = self.contrast_cluster_inventory(
                settings,
                progress_callback=progress_callback,
            )
            settings.available_elements = cluster_inventory.available_elements
            settings.cluster_inventory_rows = cluster_inventory.cluster_rows
            cluster_rows = list(cluster_inventory.cluster_rows)
            component_entries = self._build_contrast_scattering_components(
                settings,
                artifact_paths=artifact_paths,
                q_values=q_values,
                cluster_inventory=cluster_inventory,
                progress_callback=progress_callback,
            )
        else:
            cluster_inventory = self._collect_cluster_inventory(
                clusters_dir,
                progress_callback=progress_callback,
            )
            settings.available_elements = cluster_inventory.available_elements
            settings.cluster_inventory_rows = cluster_inventory.cluster_rows
            cluster_rows = list(cluster_inventory.cluster_rows)
            component_entries = self._component_entries_from_cluster_bins(
                cluster_inventory.cluster_bins
            )
            reused_observed_components = (
                settings.use_predicted_structure_weights
                and self._reuse_observed_component_artifacts(
                    settings,
                    artifact_paths=artifact_paths,
                    component_entries=component_entries,
                    progress_callback=progress_callback,
                )
            )
            if not reused_observed_components:
                builder = DebyeProfileBuilder(
                    q_values=q_values,
                    exclude_elements=settings.exclude_elements or None,
                    output_dir=artifact_paths.component_dir,
                )
                averaged_components = builder.build_profiles(
                    cluster_bins=cluster_inventory.cluster_bins,
                    progress_callback=progress_callback,
                    progress_total=max(cluster_inventory.total_files, 1),
                )
                component_entries = [
                    ProjectComponentEntry(
                        structure=component.structure,
                        motif=component.motif,
                        file_count=component.file_count,
                        representative=component.representative,
                        profile_file=component.output_path.name,
                        source_dir=str(component.source_dir),
                    )
                    for component in averaged_components
                ]
        if (
            normalize_component_build_mode(settings.component_build_mode)
            != COMPONENT_BUILD_MODE_CONTRAST
            and settings.use_predicted_structure_weights
        ):
            (
                component_entries,
                cluster_rows,
                predicted_available_elements,
                predicted_dataset_file,
                predicted_component_count,
            ) = self._augment_components_with_predicted_structures(
                settings,
                artifact_paths=artifact_paths,
                q_values=q_values,
                component_entries=component_entries,
                cluster_inventory=cluster_inventory,
            )
            settings.available_elements = sorted(
                {
                    *settings.available_elements,
                    *predicted_available_elements,
                },
                key=_natural_sort_key,
            )
            settings.cluster_inventory_rows = cluster_rows

        model_map_path = artifact_paths.component_map_file
        self._write_component_map(model_map_path, component_entries)
        self._write_distribution_metadata(
            settings,
            artifact_paths=artifact_paths,
        )
        self.save_project(
            settings,
            refresh_registered_paths=False,
        )

        return ProjectBuildResult(
            q_values=q_values,
            component_entries=component_entries,
            cluster_rows=cluster_rows,
            staged_experimental_data_path=staged_data_path,
            model_map_path=model_map_path,
            used_predicted_structure_weights=(
                settings.use_predicted_structure_weights
            ),
            predicted_dataset_file=predicted_dataset_file,
            predicted_component_count=predicted_component_count,
        )

    def _default_contrast_density_settings(
        self,
        *,
        exclude_elements: list[str] | tuple[str, ...] | None = None,
    ) -> ContrastGeometryDensitySettings:
        return ContrastGeometryDensitySettings(
            solvent=ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula="H2O",
                solvent_density_g_per_ml=1.0,
            ),
            exclude_elements=tuple(
                _normalized_elements(exclude_elements or [])
            ),
        )

    def build_contrast_scattering_components_from_results(
        self,
        settings: ProjectSettings,
        *,
        representative_result: ContrastRepresentativeSelectionResult,
        density_result: ContrastGeometryDensityResult,
        progress_callback: ProgressCallback | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> ProjectBuildResult:
        staged_data_path: Path | None = None
        experimental_data: ExperimentalDataSummary | None = None
        if not settings.model_only_mode:
            staged_data_path = self.stage_experimental_data(settings)
            self.stage_solvent_data(settings)
            experimental_data = self.load_experimental_data(settings)
        q_values = self._build_q_grid(settings, experimental_data)
        self._store_effective_q_range(
            settings,
            q_values=q_values,
            experimental_data=experimental_data,
        )
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
        )
        self.ensure_artifact_dirs(artifact_paths)
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory before building models."
            )
        if (
            normalize_component_build_mode(settings.component_build_mode)
            != COMPONENT_BUILD_MODE_CONTRAST
        ):
            raise ValueError(
                "build_contrast_scattering_components_from_results requires "
                "Contrast Mode settings."
            )
        (
            cluster_inventory,
            predicted_dataset_file,
            predicted_component_count,
        ) = self.contrast_cluster_inventory(settings)
        settings.available_elements = cluster_inventory.available_elements
        settings.cluster_inventory_rows = cluster_inventory.cluster_rows
        component_entries = (
            self._build_contrast_scattering_components_from_results(
                artifact_paths=artifact_paths,
                q_values=q_values,
                representative_result=representative_result,
                density_result=density_result,
                exclude_elements=settings.exclude_elements or None,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        )
        model_map_path = artifact_paths.component_map_file
        self._write_component_map(model_map_path, component_entries)
        self._write_distribution_metadata(
            settings,
            artifact_paths=artifact_paths,
        )
        self.save_project(
            settings,
            refresh_registered_paths=False,
        )
        return ProjectBuildResult(
            q_values=q_values,
            component_entries=component_entries,
            cluster_rows=list(cluster_inventory.cluster_rows),
            staged_experimental_data_path=staged_data_path,
            model_map_path=model_map_path,
            used_predicted_structure_weights=(
                settings.use_predicted_structure_weights
            ),
            predicted_dataset_file=predicted_dataset_file,
            predicted_component_count=predicted_component_count,
        )

    def _build_contrast_scattering_components(
        self,
        settings: ProjectSettings,
        *,
        artifact_paths: ProjectArtifactPaths,
        q_values: np.ndarray,
        cluster_inventory: _ClusterInventory | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ProjectComponentEntry]:
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Contrast Mode requires a clusters directory before SAXS "
                "components can be built."
            )
        if cluster_inventory is None:
            cluster_inventory, _dataset_file, _predicted_count = (
                self.contrast_cluster_inventory(
                    settings,
                    progress_callback=progress_callback,
                )
            )
        representative_result = analyze_contrast_representatives(
            settings.project_dir,
            clusters_dir,
            cluster_bins=tuple(cluster_inventory.cluster_bins),
            progress_callback=progress_callback,
        )
        density_result = compute_contrast_geometry_and_electron_density(
            representative_result,
            self._default_contrast_density_settings(
                exclude_elements=settings.exclude_elements,
            ),
            progress_callback=progress_callback,
        )
        return self._build_contrast_scattering_components_from_results(
            artifact_paths=artifact_paths,
            q_values=q_values,
            representative_result=representative_result,
            density_result=density_result,
            exclude_elements=settings.exclude_elements or None,
            progress_callback=progress_callback,
        )

    def _build_contrast_scattering_components_from_results(
        self,
        *,
        artifact_paths: ProjectArtifactPaths,
        q_values: np.ndarray,
        representative_result: ContrastRepresentativeSelectionResult,
        density_result: ContrastGeometryDensityResult,
        exclude_elements: list[str] | tuple[str, ...] | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> list[ProjectComponentEntry]:
        contrast_metadata_dir = artifact_paths.contrast_dir / "debye"
        contrast_build_result = build_contrast_component_profiles(
            representative_result,
            density_result,
            q_values=q_values,
            output_dir=artifact_paths.component_dir,
            metadata_dir=contrast_metadata_dir,
            component_map_path=artifact_paths.component_map_file,
            exclude_elements=exclude_elements,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        self._persist_contrast_distribution_artifacts(
            artifact_paths=artifact_paths,
            representative_result=representative_result,
            density_result=density_result,
            contrast_build_result=contrast_build_result,
        )
        return [
            ProjectComponentEntry(
                structure=trace_result.structure,
                motif=trace_result.motif,
                file_count=trace_result.file_count,
                representative=trace_result.representative_file.name,
                profile_file=trace_result.profile_file,
                source_dir=str(trace_result.representative_file.parent),
            )
            for trace_result in contrast_build_result.trace_results
        ]

    def _persist_contrast_distribution_artifacts(
        self,
        *,
        artifact_paths: ProjectArtifactPaths,
        representative_result: ContrastRepresentativeSelectionResult,
        density_result: ContrastGeometryDensityResult,
        contrast_build_result: ContrastDebyeBuildResult,
    ) -> None:
        contrast_dir = artifact_paths.contrast_dir
        contrast_dir.mkdir(parents=True, exist_ok=True)
        representative_root = representative_result.output_dir
        if representative_root.is_dir():
            for child in representative_root.iterdir():
                destination = contrast_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, destination, dirs_exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, destination)
        debye_root = contrast_build_result.metadata_dir
        if debye_root.is_dir():
            destination = contrast_dir / "debye"
            if debye_root.resolve() != destination.resolve():
                shutil.copytree(debye_root, destination, dirs_exist_ok=True)
        for density_path in (
            density_result.summary_json_path,
            density_result.summary_table_path,
            density_result.summary_text_path,
        ):
            if density_path.is_file():
                destination = contrast_dir / density_path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(density_path, destination)

    def generate_prior_weights(
        self,
        settings: ProjectSettings,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectBuildResult:
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        staged_data_path: Path | None = None
        experimental_data: ExperimentalDataSummary | None = None
        if not settings.model_only_mode:
            staged_data_path = self.stage_experimental_data(settings)
            self.stage_solvent_data(settings)
            experimental_data = self.load_experimental_data(settings)
        q_values = self._build_q_grid(settings, experimental_data)
        self._store_effective_q_range(
            settings,
            q_values=q_values,
            experimental_data=experimental_data,
        )
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
        )
        self.ensure_artifact_dirs(artifact_paths)
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory before generating prior weights."
            )
        cluster_inventory = self._collect_cluster_inventory(
            clusters_dir,
            progress_callback=progress_callback,
        )
        settings.available_elements = cluster_inventory.available_elements
        settings.cluster_inventory_rows = cluster_inventory.cluster_rows
        component_entries = self._component_entries_for_prior_weights(
            settings,
            artifact_paths=artifact_paths,
            cluster_bins=cluster_inventory.cluster_bins,
        )
        md_prior_weights_path = artifact_paths.prior_weights_file
        prior_plot_data_path = artifact_paths.prior_plot_data_file
        cluster_rows = list(cluster_inventory.cluster_rows)
        predicted_dataset_file: Path | None = None
        predicted_component_count = 0
        if progress_callback is not None:
            progress_callback(
                0,
                max(len(component_entries), 1),
                "Generating prior weights...",
            )
        for index, entry in enumerate(component_entries, start=1):
            if progress_callback is not None:
                progress_callback(
                    index,
                    max(len(component_entries), 1),
                    (
                        "Generating prior weights: "
                        f"{entry.structure}/{entry.motif}"
                    ),
                )
        if settings.use_predicted_structure_weights:
            (
                cluster_rows,
                predicted_available_elements,
                predicted_dataset_file,
                predicted_component_count,
            ) = self._write_md_prior_weights_with_predicted_structures(
                md_prior_weights_path=md_prior_weights_path,
                clusters_dir=clusters_dir,
                component_entries=component_entries,
                cluster_bins=cluster_inventory.cluster_bins,
                available_elements=cluster_inventory.available_elements,
                q_values=q_values,
                settings=settings,
            )
            settings.available_elements = sorted(
                {
                    *cluster_inventory.available_elements,
                    *predicted_available_elements,
                },
                key=_natural_sort_key,
            )
            settings.cluster_inventory_rows = cluster_rows
        else:
            self._write_md_prior_weights(
                md_prior_weights_path=md_prior_weights_path,
                clusters_dir=clusters_dir,
                component_entries=component_entries,
                cluster_bins=cluster_inventory.cluster_bins,
                available_elements=cluster_inventory.available_elements,
                q_values=q_values,
            )
        export_prior_plot_data(md_prior_weights_path, prior_plot_data_path)
        self._write_distribution_metadata(
            settings,
            artifact_paths=artifact_paths,
        )
        self.save_project(
            settings,
            refresh_registered_paths=False,
        )
        if progress_callback is not None:
            progress_callback(
                max(len(component_entries), 1),
                max(len(component_entries), 1),
                "Prior-weight generation complete.",
            )
        return ProjectBuildResult(
            q_values=q_values,
            component_entries=component_entries,
            cluster_rows=cluster_rows,
            staged_experimental_data_path=staged_data_path,
            md_prior_weights_path=md_prior_weights_path,
            prior_plot_data_path=prior_plot_data_path,
            used_predicted_structure_weights=(
                settings.use_predicted_structure_weights
            ),
            predicted_dataset_file=predicted_dataset_file,
            predicted_component_count=predicted_component_count,
        )

    def _write_distribution_metadata(
        self,
        settings: ProjectSettings,
        *,
        artifact_paths: ProjectArtifactPaths,
    ) -> None:
        metadata_path = artifact_paths.distribution_metadata_file
        if metadata_path is None:
            return
        now = datetime.now().isoformat(timespec="seconds")
        created_at = now
        if metadata_path.is_file():
            try:
                existing = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing = {}
            created_at = str(existing.get("created_at", now)).strip() or now
        component_ready = bool(
            artifact_paths.component_dir.is_dir()
            and any(artifact_paths.component_dir.glob("*.txt"))
            and artifact_paths.component_map_file.is_file()
        )
        prior_ready = bool(artifact_paths.prior_weights_file.is_file())
        payload = {
            "schema_version": 1,
            "distribution_id": artifact_paths.distribution_id,
            "label": distribution_label_for_settings(settings),
            "created_at": created_at,
            "updated_at": now,
            "template_name": _optional_str(settings.selected_model_template),
            "component_build_mode": normalize_component_build_mode(
                settings.component_build_mode
            ),
            "use_predicted_structure_weights": bool(
                settings.use_predicted_structure_weights
            ),
            "exclude_elements": sorted(set(settings.exclude_elements)),
            "clusters_dir": (
                None
                if settings.resolved_clusters_dir is None
                else str(settings.resolved_clusters_dir)
            ),
            "q_min": _optional_float(settings.q_min),
            "q_max": _optional_float(settings.q_max),
            "use_experimental_grid": bool(settings.use_experimental_grid),
            "q_points": (
                None
                if settings.use_experimental_grid
                else _optional_int(settings.q_points)
            ),
            "component_artifacts_ready": component_ready,
            "prior_artifacts_ready": prior_ready,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _copy_distribution_artifacts(
        self,
        *,
        source_root: Path,
        target_artifact_paths: ProjectArtifactPaths,
        include_predicted_structures: bool,
    ) -> None:
        if include_predicted_structures:
            source_component_dir = (
                source_root / "scattering_components_predicted_structures"
            )
            source_component_map = (
                source_root / "md_saxs_map_predicted_structures.json"
            )
            source_prior_weights = (
                source_root / "md_prior_weights_predicted_structures.json"
            )
        else:
            source_component_dir = source_root / "scattering_components"
            source_component_map = source_root / "md_saxs_map.json"
            source_prior_weights = source_root / "md_prior_weights.json"
        source_plots_dir = source_root / "plots"
        source_contrast_dir = source_root / "contrast"

        if source_component_dir.is_dir():
            shutil.copytree(
                source_component_dir,
                target_artifact_paths.component_dir,
                dirs_exist_ok=True,
            )
        if source_component_map.is_file():
            target_artifact_paths.component_map_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                source_component_map,
                target_artifact_paths.component_map_file,
            )
        if source_prior_weights.is_file():
            target_artifact_paths.prior_weights_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                source_prior_weights,
                target_artifact_paths.prior_weights_file,
            )
        if source_plots_dir.is_dir():
            shutil.copytree(
                source_plots_dir,
                target_artifact_paths.plots_dir,
                dirs_exist_ok=True,
            )
        if source_contrast_dir.is_dir():
            shutil.copytree(
                source_contrast_dir,
                target_artifact_paths.contrast_dir,
                dirs_exist_ok=True,
            )

    def _resolve_experimental_source(self, settings: ProjectSettings) -> Path:
        source = settings.resolved_experimental_data_path
        if source is None:
            raise ValueError(
                "Select an experimental data file or folder before continuing."
            )
        if source.is_dir():
            candidates = sorted(
                [
                    path
                    for pattern in ("exp_*", "*.dat", "*.txt", "*.iq")
                    for path in source.glob(pattern)
                    if path.is_file()
                ]
            )
            if not candidates:
                raise ValueError(
                    "The selected experimental data folder does not contain a "
                    "supported text data file."
                )
            return candidates[0]
        if not source.is_file():
            raise ValueError(
                f"Experimental data path does not exist: {source}"
            )
        return source

    def _resolve_solvent_source(
        self,
        settings: ProjectSettings,
    ) -> Path | None:
        source = settings.resolved_solvent_data_path
        if source is None:
            return None
        if source.is_dir():
            candidates = sorted(
                [
                    path
                    for pattern in ("solv_*", "*.dat", "*.txt", "*.iq")
                    for path in source.glob(pattern)
                    if path.is_file()
                ]
            )
            if not candidates:
                raise ValueError(
                    "The selected solvent data folder does not contain a "
                    "supported text data file."
                )
            return candidates[0]
        if not source.is_file():
            raise ValueError(f"Solvent data path does not exist: {source}")
        return source

    def _build_q_grid(
        self,
        settings: ProjectSettings,
        experimental_data: ExperimentalDataSummary | None,
    ) -> np.ndarray:
        if experimental_data is None:
            return self._build_model_only_q_grid(settings)

        filtered_q = _requested_q_values_on_source_grid(
            settings,
            experimental_data.q_values,
        )
        if (
            not settings.use_experimental_grid
            and settings.q_points is not None
            and settings.q_points > 1
        ):
            return np.linspace(
                float(filtered_q.min()),
                float(filtered_q.max()),
                settings.q_points,
            )
        return filtered_q

    def _store_effective_q_range(
        self,
        settings: ProjectSettings,
        *,
        q_values: np.ndarray,
        experimental_data: ExperimentalDataSummary | None,
    ) -> None:
        q_values = np.asarray(q_values, dtype=float)
        if q_values.size == 0:
            return
        effective_min = float(np.min(q_values))
        effective_max = float(np.max(q_values))
        if experimental_data is not None and settings.use_experimental_grid:
            experimental_q = np.asarray(
                experimental_data.q_values,
                dtype=float,
            )
            if experimental_q.size != 0:
                default_min = float(np.min(experimental_q))
                default_max = float(np.max(experimental_q))
                tolerance = _distribution_q_range_tolerance(
                    default_min,
                    default_max,
                )
                if (
                    abs(effective_min - default_min) <= tolerance
                    and abs(effective_max - default_max) <= tolerance
                ):
                    settings.q_min = None
                    settings.q_max = None
                    return
        settings.q_min = effective_min
        settings.q_max = effective_max

    def _build_model_only_q_grid(
        self,
        settings: ProjectSettings,
    ) -> np.ndarray:
        artifact_paths = project_artifact_paths(settings)
        supported_range = load_built_component_q_range(
            settings.project_dir,
            include_predicted_structures=(
                settings.use_predicted_structure_weights
            ),
            component_dir=artifact_paths.component_dir,
        )
        if (
            supported_range is None
            and settings.use_predicted_structure_weights
        ):
            supported_range = load_built_component_q_range(
                settings.project_dir,
                component_dir=project_artifact_paths(
                    ProjectSettings.from_dict(
                        {
                            **settings.to_dict(),
                            "use_predicted_structure_weights": False,
                        }
                    )
                ).component_dir,
            )
        q_min = (
            float(settings.q_min)
            if settings.q_min is not None
            else (
                float(supported_range[0])
                if supported_range is not None
                else None
            )
        )
        q_max = (
            float(settings.q_max)
            if settings.q_max is not None
            else (
                float(supported_range[1])
                if supported_range is not None
                else None
            )
        )
        if q_min is None or q_max is None:
            raise ValueError(
                "Model Only Mode requires q min and q max before SAXS "
                "components or prior weights can be generated."
            )
        if q_min > q_max:
            raise ValueError("q min must be less than or equal to q max.")
        q_points = (
            int(settings.q_points)
            if settings.q_points is not None and settings.q_points > 1
            else 500
        )
        return np.linspace(q_min, q_max, q_points)

    def _component_entries_from_clusters(
        self,
        clusters_dir: Path,
    ) -> list[ProjectComponentEntry]:
        return self._component_entries_from_cluster_bins(
            discover_cluster_bins(clusters_dir)
        )

    def _component_entries_for_prior_weights(
        self,
        settings: ProjectSettings,
        *,
        artifact_paths: ProjectArtifactPaths,
        cluster_bins: list[ClusterBin],
    ) -> list[ProjectComponentEntry]:
        if (
            normalize_component_build_mode(settings.component_build_mode)
            != COMPONENT_BUILD_MODE_CONTRAST
        ):
            return self._component_entries_from_cluster_bins(cluster_bins)
        return self._contrast_component_entries_from_distribution_artifacts(
            artifact_paths=artifact_paths,
            cluster_bins=cluster_bins,
        )

    def _component_entries_from_cluster_bins(
        self,
        cluster_bins: list[ClusterBin],
    ) -> list[ProjectComponentEntry]:
        return [
            ProjectComponentEntry(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                file_count=len(cluster_bin.files),
                representative=cluster_bin.representative,
                profile_file=self._component_profile_filename(
                    cluster_bin.structure,
                    cluster_bin.motif,
                ),
                source_dir=str(cluster_bin.source_dir),
            )
            for cluster_bin in cluster_bins
        ]

    def _contrast_component_entries_from_distribution_artifacts(
        self,
        *,
        artifact_paths: ProjectArtifactPaths,
        cluster_bins: list[ClusterBin],
    ) -> list[ProjectComponentEntry]:
        summary_path = (
            artifact_paths.contrast_dir / "debye" / "component_summary.json"
        )
        if not summary_path.is_file():
            raise ValueError(
                "Contrast Mode prior weights require existing contrast SAXS "
                "component artifacts. Build contrast SAXS components before "
                "generating prior weights."
            )
        try:
            summary_payload = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ValueError(
                "Could not read the saved contrast component summary for the "
                "active computed distribution."
            ) from exc
        raw_trace_results = summary_payload.get("trace_results")
        if not isinstance(raw_trace_results, list) or not raw_trace_results:
            raise ValueError(
                "The saved contrast component summary does not contain any "
                "representative trace records."
            )

        trace_by_key: dict[tuple[str, str], dict[str, object]] = {}
        for raw_trace in raw_trace_results:
            if not isinstance(raw_trace, dict):
                continue
            structure = _optional_str(raw_trace.get("structure"))
            if structure is None:
                continue
            motif = _normalized_nonempty_text(
                raw_trace.get("motif"),
                default="no_motif",
            )
            trace_by_key[(structure, motif)] = raw_trace

        missing_traces = [
            f"{cluster_bin.structure}/{cluster_bin.motif}"
            for cluster_bin in cluster_bins
            if (cluster_bin.structure, cluster_bin.motif) not in trace_by_key
        ]
        if missing_traces:
            raise ValueError(
                "The saved contrast component summary is missing representative "
                "traces for: "
                + ", ".join(sorted(missing_traces))
                + ". Rebuild contrast SAXS components for this computed "
                "distribution before generating prior weights."
            )

        component_entries: list[ProjectComponentEntry] = []
        missing_profiles: list[str] = []
        for cluster_bin in cluster_bins:
            trace_payload = trace_by_key[
                (cluster_bin.structure, cluster_bin.motif)
            ]
            profile_file = _optional_str(trace_payload.get("profile_file"))
            if profile_file is None:
                profile_file = self._component_profile_filename(
                    cluster_bin.structure,
                    cluster_bin.motif,
                )
            if not (artifact_paths.component_dir / profile_file).is_file():
                missing_profiles.append(profile_file)

            representative_file = self._contrast_representative_file_for_trace(
                artifact_paths=artifact_paths,
                trace_payload=trace_payload,
            )
            source_dir = (
                representative_file.parent
                if representative_file is not None
                else cluster_bin.source_dir
            )
            component_entries.append(
                ProjectComponentEntry(
                    structure=cluster_bin.structure,
                    motif=cluster_bin.motif,
                    file_count=len(cluster_bin.files),
                    representative=(
                        None
                        if representative_file is None
                        else representative_file.name
                    ),
                    profile_file=profile_file,
                    source_dir=str(source_dir.resolve()),
                )
            )

        if missing_profiles:
            raise ValueError(
                "The saved contrast SAXS component directory is missing trace "
                "files for: "
                + ", ".join(sorted(set(missing_profiles)))
                + ". Rebuild contrast SAXS components for this computed "
                "distribution before generating prior weights."
            )
        return component_entries

    def _contrast_representative_file_for_trace(
        self,
        *,
        artifact_paths: ProjectArtifactPaths,
        trace_payload: dict[str, object],
    ) -> Path | None:
        representative_text = _optional_str(
            trace_payload.get("representative_file")
        )
        if representative_text is None:
            return None
        representative_name = Path(representative_text).name
        if not representative_name:
            return None
        distribution_snapshot_path = (
            artifact_paths.contrast_dir
            / "representative_structures"
            / representative_name
        ).resolve()
        if distribution_snapshot_path.is_file():
            return distribution_snapshot_path
        return distribution_snapshot_path

    def _collect_cluster_inventory(
        self,
        clusters_dir: str | Path,
        *,
        progress_callback: ProgressCallback | None = None,
        force_refresh: bool = False,
    ) -> _ClusterInventory:
        resolved_dir = Path(clusters_dir).expanduser().resolve()
        if not force_refresh:
            cached = self._cluster_inventory_cache.get(resolved_dir)
            if cached is not None:
                return cached

        cluster_bins = discover_cluster_bins(resolved_dir)
        total_files = sum(
            len(cluster_bin.files) for cluster_bin in cluster_bins
        )
        if progress_callback is not None:
            progress_callback(
                0,
                max(total_files, 1),
                "Importing cluster files...",
            )

        unique_elements: set[str] = set()
        processed_files = 0
        last_progress_report = -1
        for cluster_bin in cluster_bins:
            for file_path in cluster_bin.files:
                unique_elements.update(scan_structure_elements(file_path))
                processed_files += 1
                if progress_callback is not None and _should_emit_progress(
                    processed_files,
                    total_files,
                    last_progress_report,
                ):
                    last_progress_report = processed_files
                    progress_callback(
                        processed_files,
                        max(total_files, 1),
                        (
                            "Importing cluster files: "
                            f"{cluster_bin.structure}/{cluster_bin.motif}"
                        ),
                    )

        total_atom_weight = sum(
            len(cluster_bin.files)
            * _structure_atom_weight(cluster_bin.structure)
            for cluster_bin in cluster_bins
        )
        cluster_rows: list[dict[str, object]] = []
        for cluster_bin in cluster_bins:
            count = len(cluster_bin.files)
            structure_weight = count / total_files if total_files else 0.0
            atom_fraction = (
                count
                * _structure_atom_weight(cluster_bin.structure)
                / total_atom_weight
                * 100.0
                if total_atom_weight
                else 0.0
            )
            cluster_rows.append(
                {
                    "structure": cluster_bin.structure,
                    "motif": cluster_bin.motif,
                    "count": count,
                    "weight": structure_weight,
                    "atom_fraction_percent": atom_fraction,
                    "structure_fraction_percent": structure_weight * 100.0,
                }
            )

        inventory = _ClusterInventory(
            cluster_bins=cluster_bins,
            available_elements=sorted(unique_elements, key=_natural_sort_key),
            cluster_rows=cluster_rows,
            total_files=total_files,
        )
        self._cluster_inventory_cache[resolved_dir] = inventory
        if progress_callback is not None:
            progress_callback(
                max(total_files, 1),
                max(total_files, 1),
                "Cluster import complete.",
            )
        return inventory

    @staticmethod
    def _component_profile_filename(structure: str, motif: str) -> str:
        safe_name = f"{structure}_{motif}".replace("/", "_")
        return f"{safe_name}.txt"

    def _write_md_prior_weights(
        self,
        *,
        md_prior_weights_path: Path,
        clusters_dir: Path,
        component_entries: list[ProjectComponentEntry],
        cluster_bins: list[ClusterBin],
        available_elements: list[str],
        q_values: np.ndarray,
    ) -> None:
        total_files = sum(entry.file_count for entry in component_entries)
        payload: dict[str, object] = {
            "origin": clusters_dir.name,
            "total_files": total_files,
            "available_elements": list(available_elements),
            "q_range": {
                "qmin": float(q_values.min()),
                "qmax": float(q_values.max()),
                "points": int(q_values.size),
            },
            "structures": {},
        }
        structures_payload: dict[str, dict[str, object]] = {}
        cluster_bin_lookup = {
            (cluster_bin.structure, cluster_bin.motif): cluster_bin
            for cluster_bin in cluster_bins
        }
        for entry in component_entries:
            cluster_bin = cluster_bin_lookup.get(
                (entry.structure, entry.motif)
            )
            source_file = (
                str(
                    (
                        Path(entry.source_dir).expanduser().resolve()
                        / entry.representative
                    ).resolve()
                )
                if entry.representative
                else ""
            )
            structures_payload.setdefault(entry.structure, {})
            structures_payload[entry.structure][entry.motif] = {
                "count": entry.file_count,
                "weight": (
                    round(entry.file_count / total_files, 6)
                    if total_files
                    else 0.0
                ),
                "representative": entry.representative or "",
                "profile_file": entry.profile_file,
                "source_kind": "cluster_dir",
                "source_dir": entry.source_dir,
                "source_file": source_file,
                "source_file_name": entry.representative or "",
                "secondary_atom_distributions": self._build_secondary_atom_distributions(
                    cluster_bin
                ),
            }
        payload["structures"] = structures_payload
        md_prior_weights_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_secondary_atom_distributions(
        self,
        cluster_bin: ClusterBin | None,
    ) -> dict[str, dict[str, int]]:
        if cluster_bin is None:
            return {}
        structure_elements = set(
            re.findall(r"[A-Z][a-z]*", cluster_bin.structure)
        )
        file_counts = [
            scan_structure_element_counts(file_path)
            for file_path in cluster_bin.files
        ]
        candidate_elements = sorted(
            {
                element
                for counts in file_counts
                for element in counts
                if element not in structure_elements
            },
            key=_natural_sort_key,
        )
        distributions: dict[str, dict[str, int]] = {
            element: {} for element in candidate_elements
        }
        for element_counts in file_counts:
            for element in candidate_elements:
                count = int(element_counts.get(element, 0))
                key = str(int(count))
                distributions[element][key] = (
                    distributions[element].get(key, 0) + 1
                )
        return {
            element: {
                count_key: int(distributions[element][count_key])
                for count_key in sorted(
                    distributions[element],
                    key=lambda value: int(value),
                )
            }
            for element in sorted(distributions, key=_natural_sort_key)
        }

    def _write_component_map(
        self,
        model_map_path: Path,
        component_entries: list[ProjectComponentEntry],
    ) -> None:
        model_map_path.write_text(
            json.dumps(
                {"saxs_map": self._component_map_payload(component_entries)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _component_map_payload(
        component_entries: list[ProjectComponentEntry],
    ) -> dict[str, dict[str, str]]:
        payload: dict[str, dict[str, str]] = {}
        for entry in component_entries:
            payload.setdefault(entry.structure, {})
            payload[entry.structure][entry.motif] = entry.profile_file
        return payload

    def _observed_only_distribution_artifact_paths(
        self,
        settings: ProjectSettings,
    ) -> ProjectArtifactPaths | None:
        if not settings.use_predicted_structure_weights:
            return None
        observed_settings = replace(
            settings,
            use_predicted_structure_weights=False,
        )
        return project_artifact_paths(
            observed_settings,
            storage_mode="distribution",
        )

    def _reuse_observed_component_artifacts(
        self,
        settings: ProjectSettings,
        *,
        artifact_paths: ProjectArtifactPaths,
        component_entries: list[ProjectComponentEntry],
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        observed_artifact_paths = (
            self._observed_only_distribution_artifact_paths(settings)
        )
        if observed_artifact_paths is None:
            return False
        if not observed_artifact_paths.component_map_file.is_file():
            return False
        try:
            component_map_payload = json.loads(
                observed_artifact_paths.component_map_file.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return False
        expected_map = self._component_map_payload(component_entries)
        if component_map_payload.get("saxs_map") != expected_map:
            return False
        missing_files = [
            entry.profile_file
            for entry in component_entries
            if not (
                observed_artifact_paths.component_dir / entry.profile_file
            ).is_file()
        ]
        if missing_files:
            return False
        if progress_callback is not None:
            progress_callback(
                0,
                max(len(component_entries), 1),
                (
                    "Reusing observed SAXS components from the matching "
                    "Observed Only distribution..."
                ),
            )
        for index, entry in enumerate(component_entries, start=1):
            source_path = (
                observed_artifact_paths.component_dir / entry.profile_file
            )
            destination_path = (
                artifact_paths.component_dir / entry.profile_file
            )
            if source_path.resolve() != destination_path.resolve():
                shutil.copy2(source_path, destination_path)
            if progress_callback is not None:
                progress_callback(
                    index,
                    max(len(component_entries), 1),
                    (
                        "Reusing observed SAXS components: "
                        f"{entry.structure}/{entry.motif}"
                    ),
                )
        return True

    def _augment_components_with_predicted_structures(
        self,
        settings: ProjectSettings,
        *,
        artifact_paths: ProjectArtifactPaths,
        q_values: np.ndarray,
        component_entries: list[ProjectComponentEntry],
        cluster_inventory: _ClusterInventory,
    ) -> tuple[
        list[ProjectComponentEntry],
        list[dict[str, object]],
        list[str],
        Path,
        int,
    ]:
        (
            loaded_dataset,
            observed_motif_weights,
            predicted_payloads,
            predicted_available_elements,
        ) = self._predicted_structure_weight_payload(
            settings,
            cluster_inventory=cluster_inventory,
        )
        combined_entries = list(component_entries)
        for payload in predicted_payloads:
            prediction = payload["prediction"]
            motif = str(payload["motif"])
            source_path = payload["source_path"]
            source_dir = (
                source_path.parent
                if source_path is not None
                else build_project_paths(
                    settings.project_dir
                ).predicted_scattering_components_dir.parent
            )
            profile_file = self._component_profile_filename(
                prediction.label,
                motif,
            )
            profile_path = artifact_paths.component_dir / profile_file
            trace = np.asarray(
                compute_debye_intensity(
                    prediction.generated_coordinates,
                    list(prediction.generated_elements),
                    q_values,
                ),
                dtype=float,
            )
            self._write_predicted_component_file(profile_path, q_values, trace)
            combined_entries.append(
                ProjectComponentEntry(
                    structure=prediction.label,
                    motif=motif,
                    file_count=1,
                    representative=(
                        None if source_path is None else source_path.name
                    ),
                    profile_file=profile_file,
                    source_dir=str(source_dir),
                )
            )
            payload["profile_file"] = profile_file
        cluster_rows = self._combined_cluster_rows(
            cluster_inventory=cluster_inventory,
            observed_motif_weights=observed_motif_weights,
            predicted_payloads=predicted_payloads,
        )
        return (
            combined_entries,
            cluster_rows,
            predicted_available_elements,
            loaded_dataset.dataset_file,
            len(predicted_payloads),
        )

    def _predicted_structure_weight_payload(
        self,
        settings: ProjectSettings,
        *,
        cluster_inventory: _ClusterInventory,
    ) -> tuple[
        object,
        dict[tuple[str, str], float],
        list[dict[str, object]],
        list[str],
    ]:
        loaded_dataset = self._load_latest_predicted_structures_dataset(
            settings.project_dir
        )
        from saxshell.clusterdynamicsml.workflow import (
            _resolved_population_weights,
        )

        result = loaded_dataset.result
        observed_weights, predicted_weights = _resolved_population_weights(
            result.training_observations,
            result.predictions,
            frame_timestep_fs=float(
                result.dynamics_result.preview.frame_timestep_fs
            ),
        )
        combined_total = float(
            np.sum(observed_weights) + np.sum(predicted_weights)
        )
        if combined_total <= 0.0:
            raise ValueError(
                "The latest Cluster Dynamics ML result did not produce any "
                "positive observed or predicted structure weights."
            )
        observed_label_weights: dict[str, float] = {}
        for observation, weight in zip(
            result.training_observations,
            observed_weights,
            strict=False,
        ):
            observed_label_weights[observation.label] = (
                observed_label_weights.get(observation.label, 0.0)
                + max(float(weight), 0.0) / combined_total
            )
        observed_total_weight = max(float(np.sum(observed_weights)), 0.0)
        observed_total_weight /= combined_total
        inventory_label_counts: dict[str, int] = {}
        for cluster_bin in cluster_inventory.cluster_bins:
            inventory_label_counts[cluster_bin.structure] = (
                inventory_label_counts.get(cluster_bin.structure, 0)
                + len(cluster_bin.files)
            )
        if set(observed_label_weights) != set(inventory_label_counts):
            total_inventory = max(sum(inventory_label_counts.values()), 1)
            observed_label_weights = {
                label: observed_total_weight * count / total_inventory
                for label, count in inventory_label_counts.items()
            }
        observed_motif_weights: dict[tuple[str, str], float] = {}
        for cluster_bin in cluster_inventory.cluster_bins:
            label_total = max(
                inventory_label_counts.get(cluster_bin.structure, 0), 1
            )
            observed_motif_weights[
                (cluster_bin.structure, cluster_bin.motif)
            ] = (
                observed_label_weights.get(cluster_bin.structure, 0.0)
                * len(cluster_bin.files)
                / label_total
            )

        predicted_payloads: list[dict[str, object]] = []
        predicted_available_elements = sorted(
            {
                str(element)
                for prediction in result.predictions
                for element in prediction.generated_elements
            },
            key=_natural_sort_key,
        )
        for prediction, weight in zip(
            result.predictions,
            predicted_weights,
            strict=False,
        ):
            normalized_weight = max(float(weight), 0.0) / combined_total
            if normalized_weight <= 0.0:
                continue
            motif = _predicted_structure_motif(prediction.rank)
            predicted_payloads.append(
                {
                    "prediction": prediction,
                    "motif": motif,
                    "weight": normalized_weight,
                    "source_path": self._predicted_structure_source_path(
                        loaded_dataset,
                        prediction,
                    ),
                }
            )
        if not predicted_payloads:
            raise ValueError(
                "Predicted Structures mode is enabled, but the latest "
                "Cluster Dynamics ML result does not contain any non-zero "
                "predicted structure weights to include."
            )
        return (
            loaded_dataset,
            observed_motif_weights,
            predicted_payloads,
            predicted_available_elements,
        )

    def _combined_cluster_rows(
        self,
        *,
        cluster_inventory: _ClusterInventory,
        observed_motif_weights: dict[tuple[str, str], float],
        predicted_payloads: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for cluster_bin in cluster_inventory.cluster_bins:
            source_file = (
                str(
                    (
                        cluster_bin.source_dir / cluster_bin.representative
                    ).resolve()
                )
                if cluster_bin.representative
                else ""
            )
            rows.append(
                {
                    "structure": cluster_bin.structure,
                    "motif": cluster_bin.motif,
                    "count": int(len(cluster_bin.files)),
                    "weight": float(
                        observed_motif_weights.get(
                            (cluster_bin.structure, cluster_bin.motif),
                            0.0,
                        )
                    ),
                    "source_kind": "cluster_dir",
                    "source_dir": str(cluster_bin.source_dir),
                    "source_file": source_file,
                    "source_file_name": cluster_bin.representative or "",
                    "representative": cluster_bin.representative or "",
                }
            )
        for payload in predicted_payloads:
            prediction = payload["prediction"]
            source_path = payload["source_path"]
            rows.append(
                {
                    "structure": prediction.label,
                    "motif": str(payload["motif"]),
                    "count": 1,
                    "weight": float(payload["weight"]),
                    "source_kind": "predicted_structure",
                    "source_dir": (
                        "" if source_path is None else str(source_path.parent)
                    ),
                    "source_file": (
                        "" if source_path is None else str(source_path)
                    ),
                    "source_file_name": (
                        "" if source_path is None else source_path.name
                    ),
                    "representative": (
                        "" if source_path is None else source_path.name
                    ),
                }
            )
        atom_weight_total = sum(
            float(row["weight"])
            * _structure_atom_weight(str(row["structure"]))
            for row in rows
        )
        for row in rows:
            structure_weight = max(float(row["weight"]), 0.0)
            row["structure_fraction_percent"] = structure_weight * 100.0
            row["atom_fraction_percent"] = (
                structure_weight
                * _structure_atom_weight(str(row["structure"]))
                / atom_weight_total
                * 100.0
                if atom_weight_total > 0.0
                else 0.0
            )
        rows.sort(
            key=lambda row: (
                _natural_sort_key(str(row["structure"])),
                _natural_sort_key(str(row["motif"])),
            )
        )
        return rows

    def _load_latest_predicted_structures_dataset(
        self,
        project_dir: str | Path,
    ):
        dataset_file = self._latest_predicted_structures_dataset_file(
            project_dir
        )
        if dataset_file is None:
            raise FileNotFoundError(
                "Use Predicted Structure Weights is enabled, but no "
                "Cluster Dynamics ML prediction bundle was found in this "
                "project. Open Tools > Cluster Dynamics > Open Cluster "
                "Dynamics (ML), run a prediction, then rebuild the SAXS "
                "components or prior weights."
            )
        from saxshell.clusterdynamicsml import load_cluster_dynamicsai_dataset

        loaded = load_cluster_dynamicsai_dataset(dataset_file)
        if not loaded.result.predictions:
            raise ValueError(
                "The latest Cluster Dynamics ML result bundle does not "
                "contain any predicted structures."
            )
        return loaded

    def _latest_predicted_structures_dataset_file(
        self,
        project_dir: str | Path,
    ) -> Path | None:
        saved_results_dir = (
            build_project_paths(project_dir).exported_data_dir
            / "clusterdynamicsml"
            / "saved_results"
        )
        if not saved_results_dir.is_dir():
            return None
        candidates = [
            path
            for path in saved_results_dir.rglob("*_clusterdynamicsml.json")
            if path.is_file()
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda path: (
                path.stat().st_mtime,
                path.name.lower(),
            ),
            reverse=True,
        )
        return candidates[0]

    def _predicted_structure_source_path(
        self,
        loaded_dataset,
        prediction,
    ) -> Path | None:
        comparison = getattr(loaded_dataset.result, "saxs_comparison", None)
        if comparison is not None:
            for entry in comparison.component_weights:
                if (
                    str(entry.source) == "predicted"
                    and str(entry.label) == str(prediction.label)
                    and entry.structure_path is not None
                    and Path(entry.structure_path).is_file()
                ):
                    return Path(entry.structure_path).expanduser().resolve()
        bundle_dir = loaded_dataset.dataset_file.parent
        candidate = (
            bundle_dir
            / f"{loaded_dataset.dataset_file.stem}_predicted_structures"
            / _predicted_structure_filename(
                prediction.target_node_count,
                prediction.rank,
                prediction.label,
            )
        )
        if candidate.is_file():
            return candidate.resolve()
        return None

    def _write_predicted_component_file(
        self,
        output_path: Path,
        q_values: np.ndarray,
        intensity: np.ndarray,
    ) -> None:
        header = (
            "# Number of files: 1\n"
            "# Columns: q, S(q)_avg, S(q)_std, S(q)_se\n"
        )
        data = np.column_stack(
            [
                np.asarray(q_values, dtype=float),
                np.asarray(intensity, dtype=float),
                np.zeros_like(q_values, dtype=float),
                np.zeros_like(q_values, dtype=float),
            ]
        )
        np.savetxt(
            output_path,
            data,
            comments="",
            header=header,
            fmt=["%.8f", "%.8f", "%.8f", "%.8f"],
        )

    def _write_md_prior_weights_with_predicted_structures(
        self,
        *,
        md_prior_weights_path: Path,
        clusters_dir: Path,
        component_entries: list[ProjectComponentEntry],
        cluster_bins: list[ClusterBin],
        available_elements: list[str],
        q_values: np.ndarray,
        settings: ProjectSettings,
    ) -> tuple[list[dict[str, object]], list[str], Path, int]:
        cluster_inventory = _ClusterInventory(
            cluster_bins=cluster_bins,
            available_elements=list(available_elements),
            cluster_rows=[],
            total_files=sum(
                len(cluster_bin.files) for cluster_bin in cluster_bins
            ),
        )
        (
            loaded_dataset,
            observed_motif_weights,
            predicted_payloads,
            predicted_available_elements,
        ) = self._predicted_structure_weight_payload(
            settings,
            cluster_inventory=cluster_inventory,
        )
        total_files = sum(entry.file_count for entry in component_entries)
        payload: dict[str, object] = {
            "origin": f"{clusters_dir.name}_predicted_structures",
            "total_files": total_files,
            "available_elements": sorted(
                {
                    *available_elements,
                    *predicted_available_elements,
                },
                key=_natural_sort_key,
            ),
            "q_range": {
                "qmin": float(q_values.min()),
                "qmax": float(q_values.max()),
                "points": int(q_values.size),
            },
            "value_kind": "normalized_weight",
            "includes_predicted_structures": True,
            "prediction_dataset_file": str(loaded_dataset.dataset_file),
            "structures": {},
        }
        structures_payload: dict[str, dict[str, object]] = {}
        cluster_bin_lookup = {
            (cluster_bin.structure, cluster_bin.motif): cluster_bin
            for cluster_bin in cluster_bins
        }
        for entry in component_entries:
            cluster_bin = cluster_bin_lookup.get(
                (entry.structure, entry.motif)
            )
            source_file = (
                str(
                    (
                        Path(entry.source_dir).expanduser().resolve()
                        / entry.representative
                    ).resolve()
                )
                if entry.representative
                else ""
            )
            normalized_weight = float(
                observed_motif_weights.get((entry.structure, entry.motif), 0.0)
            )
            observed_only_weight = (
                round(entry.file_count / total_files, 6)
                if total_files
                else 0.0
            )
            structures_payload.setdefault(entry.structure, {})
            structures_payload[entry.structure][entry.motif] = {
                "count": entry.file_count,
                "weight": normalized_weight,
                "normalized_weight": normalized_weight,
                "observed_only_weight": observed_only_weight,
                "representative": entry.representative or "",
                "profile_file": entry.profile_file,
                "source_kind": "cluster_dir",
                "source_dir": entry.source_dir,
                "source_file": source_file,
                "source_file_name": entry.representative or "",
                "secondary_atom_distributions": self._build_secondary_atom_distributions(
                    cluster_bin
                ),
            }
        for payload_entry in predicted_payloads:
            prediction = payload_entry["prediction"]
            source_path = payload_entry["source_path"]
            motif = str(payload_entry["motif"])
            structures_payload.setdefault(prediction.label, {})
            structures_payload[prediction.label][motif] = {
                "count": 1,
                "weight": float(payload_entry["weight"]),
                "normalized_weight": float(payload_entry["weight"]),
                "observed_only_weight": 0.0,
                "representative": (
                    "" if source_path is None else source_path.name
                ),
                "profile_file": self._component_profile_filename(
                    prediction.label,
                    motif,
                ),
                "source_kind": "predicted_structure",
                "source_dir": (
                    "" if source_path is None else str(source_path.parent)
                ),
                "source_file": (
                    "" if source_path is None else str(source_path)
                ),
                "source_file_name": (
                    "" if source_path is None else source_path.name
                ),
                "secondary_atom_distributions": (
                    _predicted_secondary_atom_distributions(prediction)
                ),
            }
        payload["structures"] = structures_payload
        md_prior_weights_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return (
            self._combined_cluster_rows(
                cluster_inventory=cluster_inventory,
                observed_motif_weights=observed_motif_weights,
                predicted_payloads=predicted_payloads,
            ),
            predicted_available_elements,
            loaded_dataset.dataset_file,
            len(predicted_payloads),
        )


def _nearest_cropped_q_values(
    q_values: np.ndarray,
    q_min: float,
    q_max: float,
) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=float)
    if q_values.size == 0:
        raise ValueError(
            "The experimental data does not contain any q-values."
        )
    start_index = int(np.argmin(np.abs(q_values - q_min)))
    end_index = int(np.argmin(np.abs(q_values - q_max)))
    lo_index, hi_index = sorted((start_index, end_index))
    cropped_q = q_values[lo_index : hi_index + 1]
    if cropped_q.size == 0:
        raise ValueError(
            "The requested q-range does not overlap the experimental data."
        )
    return cropped_q


def _requested_q_values_on_source_grid(
    settings: ProjectSettings,
    source_q_values: np.ndarray,
) -> np.ndarray:
    q_values = np.asarray(source_q_values, dtype=float)
    if q_values.size == 0:
        raise ValueError(
            "The experimental data does not contain any q-values."
        )
    q_min = (
        float(settings.q_min)
        if settings.q_min is not None
        else float(np.min(q_values))
    )
    q_max = (
        float(settings.q_max)
        if settings.q_max is not None
        else float(np.max(q_values))
    )
    if q_min > q_max:
        raise ValueError("q min must be less than or equal to q max.")
    if settings.use_experimental_grid:
        return _nearest_cropped_q_values(q_values, q_min, q_max)
    mask = (q_values >= q_min) & (q_values <= q_max)
    filtered_q = np.asarray(q_values[mask], dtype=float)
    if filtered_q.size == 0:
        raise ValueError(
            "The requested q-range does not overlap the experimental data."
        )
    return filtered_q


def _predicted_structure_motif(rank: int) -> str:
    return f"predicted_rank{int(rank):02d}"


def _predicted_structure_filename(
    target_node_count: int,
    rank: int,
    label: str,
) -> str:
    return f"{int(target_node_count):02d}_rank{int(rank):02d}_{label}.xyz"


def _predicted_secondary_atom_distributions(
    prediction,
) -> dict[str, dict[str, int]]:
    structure_elements = set(re.findall(r"[A-Z][a-z]*", str(prediction.label)))
    element_counts: dict[str, int] = {}
    for element in prediction.generated_elements:
        normalized = str(element).strip()
        if not normalized:
            continue
        element_counts[normalized] = element_counts.get(normalized, 0) + 1
    return {
        element: {str(int(count)): 1}
        for element, count in sorted(
            element_counts.items(),
            key=lambda item: _natural_sort_key(item[0]),
        )
        if element not in structure_elements
    }


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _structure_atom_weight(structure: str) -> int:
    return max(sum(int(token) for token in re.findall(r"(\d+)", structure)), 1)


def _should_emit_progress(
    processed: int,
    total: int,
    last_reported: int,
) -> bool:
    if processed <= 0:
        return True
    if total <= 0 or processed >= total:
        return True
    if total <= 200:
        return processed != last_reported
    interval = max(total // 200, 25)
    return processed - last_reported >= interval


def load_experimental_data_file(
    path: str | Path,
    *,
    skiprows: int = 0,
    q_column: int | None = None,
    intensity_column: int | None = None,
    error_column: int | None = None,
) -> ExperimentalDataSummary:
    file_path = Path(path).expanduser().resolve()
    effective_skiprows = max(skiprows, 0)
    parse_error: Exception | None = None
    try:
        data = np.loadtxt(
            file_path,
            comments="#",
            skiprows=effective_skiprows,
        )
        column_names = _read_experimental_column_names(
            file_path,
            effective_skiprows,
        )
    except Exception as exc:
        parse_error = exc
        guessed_header_rows = _guess_experimental_header_rows(file_path)
        if guessed_header_rows <= effective_skiprows:
            raise ValueError(
                f"Unable to parse experimental data file {file_path} with "
                f"skiprows={skiprows}: {exc}"
            ) from exc
        effective_skiprows = guessed_header_rows
        column_names = _read_experimental_column_names(
            file_path,
            effective_skiprows,
        )
        try:
            data = np.loadtxt(
                file_path,
                comments="#",
                skiprows=effective_skiprows,
            )
        except Exception as header_exc:
            raise ValueError(
                f"Unable to parse experimental data file {file_path} with "
                f"skiprows={skiprows}: {parse_error}\n"
                f"Header-based parse with skiprows={effective_skiprows} also "
                f"failed: {header_exc}"
            ) from header_exc
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(
            "Experimental data must contain at least two columns: q and I(q)."
        )
    resolved_q, resolved_i, inferred_error = _resolve_experimental_columns(
        column_names,
        data.shape[1],
        q_column=q_column,
        intensity_column=intensity_column,
        error_column=error_column,
    )
    q_values = np.asarray(data[:, resolved_q], dtype=float)
    intensities = np.asarray(data[:, resolved_i], dtype=float)
    errors = (
        np.asarray(data[:, inferred_error], dtype=float)
        if inferred_error is not None
        else None
    )
    return ExperimentalDataSummary(
        path=file_path,
        q_values=q_values,
        intensities=intensities,
        errors=errors,
        header_rows=effective_skiprows,
        column_names=column_names,
        q_column=resolved_q,
        intensity_column=resolved_i,
        error_column=inferred_error,
    )


def guess_experimental_header_rows(path: str | Path) -> int:
    return _guess_experimental_header_rows(Path(path).expanduser().resolve())


def read_experimental_column_names(
    path: str | Path,
    *,
    skiprows: int,
) -> list[str]:
    return _read_experimental_column_names(
        Path(path).expanduser().resolve(),
        max(skiprows, 0),
    )


def infer_experimental_columns(
    column_names: list[str],
) -> tuple[int | None, int | None, int | None]:
    q_column = _find_matching_column(
        column_names,
        patterns=("q", "q_", "qvalue", "scatteringvector"),
    )
    intensity_column = _find_matching_column(
        column_names,
        patterns=("intensity", "i_", "iq", "int"),
        exclude={q_column} if q_column is not None else set(),
    )
    error_column = _find_matching_column(
        column_names,
        patterns=("error", "err", "sigma", "uncert", "e_"),
        exclude={
            index
            for index in (q_column, intensity_column)
            if index is not None
        },
    )
    return q_column, intensity_column, error_column


def _resolve_experimental_columns(
    column_names: list[str],
    n_columns: int,
    *,
    q_column: int | None,
    intensity_column: int | None,
    error_column: int | None,
) -> tuple[int, int, int | None]:
    inferred_q, inferred_i, inferred_e = infer_experimental_columns(
        column_names
    )
    resolved_q = q_column if q_column is not None else inferred_q
    resolved_i = (
        intensity_column if intensity_column is not None else inferred_i
    )

    if resolved_q is None and n_columns >= 1:
        resolved_q = 0
    if resolved_i is None and n_columns >= 2:
        resolved_i = 1

    if resolved_q is None or resolved_i is None:
        raise ValueError(
            "Unable to determine which columns contain q and intensity. "
            "Please select the q and intensity columns manually."
        )
    if resolved_q == resolved_i:
        raise ValueError("q and intensity must use different columns.")
    for index, label in (
        (resolved_q, "q"),
        (resolved_i, "intensity"),
    ):
        if index < 0 or index >= n_columns:
            raise ValueError(
                f"The selected {label} column index {index} is out of range."
            )

    if error_column is not None:
        resolved_error = error_column
    elif inferred_e is not None:
        resolved_error = inferred_e
    elif n_columns >= 3:
        candidate = 2
        resolved_error = (
            candidate if candidate not in {resolved_q, resolved_i} else None
        )
    else:
        resolved_error = None

    if resolved_error is not None:
        if resolved_error in {resolved_q, resolved_i}:
            raise ValueError(
                "The error column must be different from q and intensity."
            )
        if resolved_error < 0 or resolved_error >= n_columns:
            raise ValueError(
                f"The selected error column index {resolved_error} is out of range."
            )
    return resolved_q, resolved_i, resolved_error


def _guess_experimental_header_rows(file_path: Path) -> int:
    header_rows = 0
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                header_rows += 1
                continue
            tokens = _split_experimental_line(stripped)
            if len(tokens) >= 2 and _tokens_look_numeric(tokens):
                return header_rows
            header_rows += 1
    return 0


def _read_experimental_column_names(
    file_path: Path,
    header_rows: int,
) -> list[str]:
    lines = file_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if header_rows > 0 and header_rows <= len(lines):
        header_tokens = _split_experimental_line(
            lines[header_rows - 1].lstrip("#").strip()
        )
        if header_tokens and not _tokens_look_numeric(header_tokens):
            return _normalize_column_names(header_tokens)
    first_data_tokens = _first_data_tokens(lines, header_rows)
    if first_data_tokens is None:
        return []
    return [f"Column {index + 1}" for index in range(len(first_data_tokens))]


def _first_data_tokens(
    lines: list[str],
    start_index: int,
) -> list[str] | None:
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = _split_experimental_line(stripped)
        if len(tokens) >= 2 and _tokens_look_numeric(tokens):
            return tokens
    return None


def _split_experimental_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if "\t" in stripped:
        return [
            token.strip() for token in stripped.split("\t") if token.strip()
        ]
    if "," in stripped:
        return [
            token.strip() for token in stripped.split(",") if token.strip()
        ]
    return [token.strip() for token in stripped.split() if token.strip()]


def _tokens_look_numeric(tokens: list[str]) -> bool:
    try:
        float(tokens[0])
        float(tokens[1])
    except (IndexError, TypeError, ValueError):
        return False
    return True


def _normalize_column_names(column_names: list[str]) -> list[str]:
    names: list[str] = []
    for index, name in enumerate(column_names):
        clean_name = str(name).strip()
        names.append(clean_name or f"Column {index + 1}")
    return names


def _find_matching_column(
    column_names: list[str],
    *,
    patterns: tuple[str, ...],
    exclude: set[int] | None = None,
) -> int | None:
    excluded = exclude or set()
    normalized_names = [
        re.sub(r"[^a-z0-9]+", "", column_name.lower())
        for column_name in column_names
    ]
    for pattern in patterns:
        normalized_pattern = re.sub(r"[^a-z0-9]+", "", pattern.lower())
        for index, name in enumerate(normalized_names):
            if index in excluded:
                continue
            if (
                name.startswith(normalized_pattern)
                or normalized_pattern in name
            ):
                return index
    return None
