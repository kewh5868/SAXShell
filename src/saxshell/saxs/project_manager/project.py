from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from saxshell.saxs.debye import (
    ClusterBin,
    DebyeProfileBuilder,
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
    experimental_data_dir: Path
    scattering_components_dir: Path
    plots_dir: Path
    prefit_dir: Path
    dream_dir: Path
    dream_runtime_dir: Path
    reports_dir: Path


def build_project_paths(project_dir: str | Path) -> ProjectPaths:
    project_dir = Path(project_dir).expanduser().resolve()
    return ProjectPaths(
        project_dir=project_dir,
        project_file=project_dir / "saxs_project.json",
        experimental_data_dir=project_dir / "experimental_data",
        scattering_components_dir=project_dir / "scattering_components",
        plots_dir=project_dir / "plots",
        prefit_dir=project_dir / "prefit",
        dream_dir=project_dir / "dream",
        dream_runtime_dir=project_dir / "dream" / "runtime_scripts",
        reports_dir=project_dir / "reports",
    )


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


@dataclass(slots=True)
class ProjectSettings:
    project_name: str
    project_dir: str
    clusters_dir: str | None = None
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
    include_elements: list[str] = field(default_factory=list)
    exclude_elements: list[str] = field(default_factory=list)
    component_trace_colors: dict[str, str] = field(default_factory=dict)
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
    selected_model_template: str | None = None
    autosave_prefits: bool = False

    @property
    def resolved_project_dir(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()

    @property
    def resolved_clusters_dir(self) -> Path | None:
        if self.clusters_dir is None or not self.clusters_dir.strip():
            return None
        return Path(self.clusters_dir).expanduser().resolve()

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
        payload["component_trace_colors"] = dict(self.component_trace_colors)
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
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProjectSettings":
        return cls(
            project_name=str(payload.get("project_name", "SAXS Project")),
            project_dir=str(payload.get("project_dir", "")),
            clusters_dir=_optional_str(payload.get("clusters_dir")),
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
            include_elements=_normalized_elements(
                payload.get("include_elements", [])
            ),
            exclude_elements=_normalized_elements(
                payload.get("exclude_elements", [])
            ),
            component_trace_colors=_normalized_text_map(
                payload.get("component_trace_colors", {})
            ),
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
            selected_model_template=_optional_str(
                payload.get("selected_model_template")
            ),
            autosave_prefits=bool(payload.get("autosave_prefits", False)),
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
        self.save_project(settings)
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

    def save_project(self, settings: ProjectSettings) -> Path:
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        paths.project_file.write_text(
            json.dumps(settings.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return paths.project_file

    def ensure_project_dirs(self, paths: ProjectPaths) -> None:
        for directory in (
            paths.project_dir,
            paths.experimental_data_dir,
            paths.scattering_components_dir,
            paths.plots_dir,
            paths.prefit_dir,
            paths.dream_dir,
            paths.dream_runtime_dir,
            paths.reports_dir,
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
        staged_data_path = self.stage_experimental_data(settings)
        self.stage_solvent_data(settings)
        experimental_data = self.load_experimental_data(settings)
        q_values = self._build_q_grid(settings, experimental_data)
        builder = DebyeProfileBuilder(
            q_values=q_values,
            exclude_elements=settings.exclude_elements or None,
            output_dir=paths.scattering_components_dir,
        )
        clusters_dir = settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory before building models."
            )
        cluster_inventory = self._collect_cluster_inventory(
            clusters_dir,
            progress_callback=progress_callback,
        )
        settings.available_elements = cluster_inventory.available_elements
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

        model_map_path = paths.project_dir / "md_saxs_map.json"
        self._write_component_map(model_map_path, component_entries)
        self.save_project(settings)

        return ProjectBuildResult(
            q_values=q_values,
            component_entries=component_entries,
            cluster_rows=cluster_inventory.cluster_rows,
            staged_experimental_data_path=staged_data_path,
            model_map_path=model_map_path,
        )

    def generate_prior_weights(
        self,
        settings: ProjectSettings,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ProjectBuildResult:
        paths = build_project_paths(settings.project_dir)
        self.ensure_project_dirs(paths)
        staged_data_path = self.stage_experimental_data(settings)
        self.stage_solvent_data(settings)
        experimental_data = self.load_experimental_data(settings)
        q_values = self._build_q_grid(settings, experimental_data)
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
        component_entries = self._component_entries_from_cluster_bins(
            cluster_inventory.cluster_bins
        )
        md_prior_weights_path = paths.project_dir / "md_prior_weights.json"
        prior_plot_data_path = paths.plots_dir / "prior_histogram_data.json"
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
        self._write_md_prior_weights(
            md_prior_weights_path=md_prior_weights_path,
            clusters_dir=clusters_dir,
            component_entries=component_entries,
            cluster_bins=cluster_inventory.cluster_bins,
            available_elements=cluster_inventory.available_elements,
            q_values=q_values,
        )
        export_prior_plot_data(md_prior_weights_path, prior_plot_data_path)
        self.save_project(settings)
        if progress_callback is not None:
            progress_callback(
                max(len(component_entries), 1),
                max(len(component_entries), 1),
                "Prior-weight generation complete.",
            )
        return ProjectBuildResult(
            q_values=q_values,
            component_entries=component_entries,
            cluster_rows=cluster_inventory.cluster_rows,
            staged_experimental_data_path=staged_data_path,
            md_prior_weights_path=md_prior_weights_path,
            prior_plot_data_path=prior_plot_data_path,
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
        experimental_data: ExperimentalDataSummary,
    ) -> np.ndarray:
        q_values = experimental_data.q_values
        q_min = (
            settings.q_min
            if settings.q_min is not None
            else float(q_values.min())
        )
        q_max = (
            settings.q_max
            if settings.q_max is not None
            else float(q_values.max())
        )
        if q_min > q_max:
            raise ValueError("q min must be less than or equal to q max.")

        if settings.use_experimental_grid:
            filtered_q = _nearest_cropped_q_values(q_values, q_min, q_max)
        else:
            mask = (q_values >= q_min) & (q_values <= q_max)
            filtered_q = q_values[mask]
            if filtered_q.size == 0:
                raise ValueError(
                    "The requested q-range does not overlap the experimental data."
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

    def _component_entries_from_clusters(
        self,
        clusters_dir: Path,
    ) -> list[ProjectComponentEntry]:
        return self._component_entries_from_cluster_bins(
            discover_cluster_bins(clusters_dir)
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
        payload: dict[str, dict[str, str]] = {}
        for entry in component_entries:
            payload.setdefault(entry.structure, {})
            payload[entry.structure][entry.motif] = entry.profile_file
        model_map_path.write_text(
            json.dumps({"saxs_map": payload}, indent=2) + "\n",
            encoding="utf-8",
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
