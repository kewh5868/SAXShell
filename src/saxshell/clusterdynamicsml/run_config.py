from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from saxshell.cluster import (
    SEARCH_MODE_KDTREE,
    PairCutoffDefinitions,
    PDBShellReferenceDefinition,
    normalize_pair_cutoffs,
    normalize_search_mode,
)
from saxshell.cluster.workflow import (
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
)
from saxshell.clusterdynamics.run_config import (
    coerce_atom_type_definitions,
    coerce_box_dimensions,
    coerce_int_tuple,
    coerce_pair_cutoff_definitions,
    optional_float,
    optional_positive_float,
    optional_text,
    path_text_for_run_config,
    resolve_run_config_path,
    serialize_atom_type_definitions,
    serialize_pair_cutoff_definitions,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.structure import (
    AtomTypeDefinitions,
    normalize_atom_type_definitions,
)

from .dataset import (
    SavedClusterDynamicsMLDataset,
    save_cluster_dynamicsai_dataset,
)
from .workflow import ClusterDynamicsMLResult, ClusterDynamicsMLWorkflow

DEFAULT_RUN_FILE_NAME = "cluster_dynamics_ml_cli_run.json"
RUN_CONFIG_VERSION = 1
ClusterDynamicsMLRunLogCallback = Callable[[str], None]


@dataclass(slots=True)
class ClusterDynamicsMLRunConfig:
    frames_dir: str
    output_file: str | None
    clusters_dir: str | None = None
    project_dir: str | None = None
    experimental_data_file: str | None = None
    energy_file: str | None = None
    atom_type_definitions: AtomTypeDefinitions = field(
        default_factory=example_atom_type_definitions
    )
    pair_cutoff_definitions: PairCutoffDefinitions = field(
        default_factory=example_pair_cutoff_definitions
    )
    box_dimensions: tuple[float, float, float] | None = None
    use_pbc: bool = False
    default_cutoff: float | None = None
    shell_levels: tuple[int, ...] = ()
    shared_shells: bool = False
    include_shell_atoms_in_stoichiometry: bool = False
    search_mode: str = SEARCH_MODE_KDTREE
    shell_reference_definitions: tuple[PDBShellReferenceDefinition, ...] = ()
    folder_start_time_fs: float | None = None
    first_frame_time_fs: float = 0.0
    frame_timestep_fs: float = 0.5
    frames_per_colormap_timestep: int = 1
    analysis_start_fs: float | None = None
    analysis_stop_fs: float | None = None
    target_node_counts: tuple[int, ...] = (4, 5)
    candidates_per_size: int = 3
    prediction_population_share_threshold: float = 0.02
    q_min: float | None = 0.02
    q_max: float | None = 1.20
    q_points: int = 250
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": RUN_CONFIG_VERSION,
            "created_at": self.created_at,
            "frames_dir": self.frames_dir,
            "output_file": self.output_file,
            "clusters_dir": self.clusters_dir,
            "project_dir": self.project_dir,
            "experimental_data_file": self.experimental_data_file,
            "energy_file": self.energy_file,
            "atom_type_definitions": serialize_atom_type_definitions(
                self.atom_type_definitions
            ),
            "pair_cutoff_definitions": serialize_pair_cutoff_definitions(
                self.pair_cutoff_definitions
            ),
            "box_dimensions": self.box_dimensions,
            "use_pbc": bool(self.use_pbc),
            "default_cutoff": self.default_cutoff,
            "shell_levels": [int(level) for level in self.shell_levels],
            "shared_shells": bool(self.shared_shells),
            "include_shell_atoms_in_stoichiometry": bool(
                self.include_shell_atoms_in_stoichiometry
            ),
            "search_mode": normalize_search_mode(self.search_mode),
            "shell_reference_definitions": [
                serialize_shell_reference_definition(definition)
                for definition in self.shell_reference_definitions
            ],
            "folder_start_time_fs": self.folder_start_time_fs,
            "first_frame_time_fs": float(self.first_frame_time_fs),
            "frame_timestep_fs": float(self.frame_timestep_fs),
            "frames_per_colormap_timestep": int(
                self.frames_per_colormap_timestep
            ),
            "analysis_start_fs": self.analysis_start_fs,
            "analysis_stop_fs": self.analysis_stop_fs,
            "target_node_counts": [
                int(value) for value in self.target_node_counts
            ],
            "candidates_per_size": int(self.candidates_per_size),
            "prediction_population_share_threshold": float(
                self.prediction_population_share_threshold
            ),
            "q_min": self.q_min,
            "q_max": self.q_max,
            "q_points": int(self.q_points),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ClusterDynamicsMLRunConfig":
        frames_dir = str(payload.get("frames_dir", "")).strip()
        if not frames_dir:
            raise ValueError(
                "Cluster dynamics ML run file is missing frames_dir."
            )
        return cls(
            frames_dir=frames_dir,
            output_file=optional_text(payload.get("output_file")),
            clusters_dir=optional_text(payload.get("clusters_dir")),
            project_dir=optional_text(payload.get("project_dir")),
            experimental_data_file=optional_text(
                payload.get("experimental_data_file")
            ),
            energy_file=optional_text(payload.get("energy_file")),
            atom_type_definitions=coerce_atom_type_definitions(
                payload.get("atom_type_definitions")
            ),
            pair_cutoff_definitions=coerce_pair_cutoff_definitions(
                payload.get("pair_cutoff_definitions")
            ),
            box_dimensions=coerce_box_dimensions(
                payload.get("box_dimensions")
            ),
            use_pbc=bool(payload.get("use_pbc", False)),
            default_cutoff=optional_positive_float(
                payload.get("default_cutoff")
            ),
            shell_levels=coerce_int_tuple(payload.get("shell_levels")),
            shared_shells=bool(payload.get("shared_shells", False)),
            include_shell_atoms_in_stoichiometry=bool(
                payload.get("include_shell_atoms_in_stoichiometry", False)
            ),
            search_mode=normalize_search_mode(
                str(payload.get("search_mode", SEARCH_MODE_KDTREE))
            ),
            shell_reference_definitions=coerce_shell_reference_definitions(
                payload.get("shell_reference_definitions")
            ),
            folder_start_time_fs=optional_float(
                payload.get("folder_start_time_fs")
            ),
            first_frame_time_fs=float(payload.get("first_frame_time_fs", 0.0)),
            frame_timestep_fs=float(payload.get("frame_timestep_fs", 0.5)),
            frames_per_colormap_timestep=max(
                int(payload.get("frames_per_colormap_timestep", 1)),
                1,
            ),
            analysis_start_fs=optional_float(payload.get("analysis_start_fs")),
            analysis_stop_fs=optional_float(payload.get("analysis_stop_fs")),
            target_node_counts=coerce_int_tuple(
                payload.get("target_node_counts")
            )
            or (4, 5),
            candidates_per_size=max(
                int(payload.get("candidates_per_size", 3)), 1
            ),
            prediction_population_share_threshold=max(
                float(
                    payload.get("prediction_population_share_threshold", 0.02)
                ),
                0.0,
            ),
            q_min=optional_float(payload.get("q_min")),
            q_max=optional_float(payload.get("q_max")),
            q_points=max(int(payload.get("q_points", 250)), 10),
            created_at=str(payload.get("created_at", "")).strip()
            or datetime.now().isoformat(timespec="seconds"),
        )


@dataclass(slots=True, frozen=True)
class ClusterDynamicsMLRunExecutionSummary:
    project_dir: Path
    run_file_path: Path | None
    frames_dir: Path
    output_file: Path
    result: ClusterDynamicsMLResult
    saved_dataset: SavedClusterDynamicsMLDataset
    project_file: Path | None

    @property
    def written_count(self) -> int:
        return len(self.saved_dataset.written_files)


def default_clusterdynamicsml_run_file_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / DEFAULT_RUN_FILE_NAME


def save_clusterdynamicsml_run_config(
    output_path: str | Path,
    config: ClusterDynamicsMLRunConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_clusterdynamicsml_run_config(
    run_file_path: str | Path,
) -> ClusterDynamicsMLRunConfig:
    path = Path(run_file_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "Cluster dynamics ML run file must contain a JSON object: "
            f"{path}"
        )
    return ClusterDynamicsMLRunConfig.from_dict(payload)


def build_clusterdynamicsml_run_config(
    *,
    project_dir: str | Path,
    frames_dir: str | Path,
    output_file: str | Path | None,
    clusters_dir: str | Path | None = None,
    experimental_data_file: str | Path | None = None,
    energy_file: str | Path | None = None,
    atom_type_definitions: AtomTypeDefinitions,
    pair_cutoff_definitions: PairCutoffDefinitions,
    box_dimensions: tuple[float, float, float] | None = None,
    use_pbc: bool = False,
    default_cutoff: float | None = None,
    shell_levels: tuple[int, ...] = (),
    shared_shells: bool = False,
    include_shell_atoms_in_stoichiometry: bool = False,
    search_mode: str = SEARCH_MODE_KDTREE,
    shell_reference_definitions: tuple[PDBShellReferenceDefinition, ...] = (),
    folder_start_time_fs: float | None = None,
    first_frame_time_fs: float = 0.0,
    frame_timestep_fs: float = 0.5,
    frames_per_colormap_timestep: int = 1,
    analysis_start_fs: float | None = None,
    analysis_stop_fs: float | None = None,
    target_node_counts: tuple[int, ...] = (4, 5),
    candidates_per_size: int = 3,
    prediction_population_share_threshold: float = 0.02,
    q_min: float | None = 0.02,
    q_max: float | None = 1.20,
    q_points: int = 250,
) -> ClusterDynamicsMLRunConfig:
    return ClusterDynamicsMLRunConfig(
        frames_dir=path_text_for_run_config(
            frames_dir,
            project_dir=project_dir,
        )
        or "",
        output_file=path_text_for_run_config(
            output_file,
            project_dir=project_dir,
        ),
        clusters_dir=path_text_for_run_config(
            clusters_dir,
            project_dir=project_dir,
        ),
        project_dir=path_text_for_run_config(
            project_dir,
            project_dir=project_dir,
        ),
        experimental_data_file=path_text_for_run_config(
            experimental_data_file,
            project_dir=project_dir,
        ),
        energy_file=path_text_for_run_config(
            energy_file,
            project_dir=project_dir,
        ),
        atom_type_definitions=normalize_atom_type_definitions(
            atom_type_definitions
        ),
        pair_cutoff_definitions=normalize_pair_cutoffs(
            pair_cutoff_definitions
        ),
        box_dimensions=box_dimensions,
        use_pbc=bool(use_pbc),
        default_cutoff=default_cutoff,
        shell_levels=tuple(sorted({int(level) for level in shell_levels})),
        shared_shells=bool(shared_shells),
        include_shell_atoms_in_stoichiometry=bool(
            include_shell_atoms_in_stoichiometry
        ),
        search_mode=normalize_search_mode(search_mode),
        shell_reference_definitions=tuple(shell_reference_definitions),
        folder_start_time_fs=folder_start_time_fs,
        first_frame_time_fs=float(first_frame_time_fs),
        frame_timestep_fs=float(frame_timestep_fs),
        frames_per_colormap_timestep=max(
            int(frames_per_colormap_timestep),
            1,
        ),
        analysis_start_fs=analysis_start_fs,
        analysis_stop_fs=analysis_stop_fs,
        target_node_counts=tuple(
            sorted({int(value) for value in target_node_counts})
        )
        or (4, 5),
        candidates_per_size=max(int(candidates_per_size), 1),
        prediction_population_share_threshold=max(
            float(prediction_population_share_threshold),
            0.0,
        ),
        q_min=q_min,
        q_max=q_max,
        q_points=max(int(q_points), 10),
    )


def suggest_clusterdynamicsml_output_file(
    *,
    project_dir: str | Path,
    frames_dir: str | Path | None,
) -> Path:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    dataset_dir = (
        build_project_paths(resolved_project_dir).exported_data_dir
        / "clusterdynamicsml"
    )
    folder_label = "cluster_dynamics_ml"
    if frames_dir is not None:
        frames_path = Path(frames_dir).expanduser()
        if frames_path.name:
            folder_label = frames_path.name
    return dataset_dir / f"{folder_label}_clusterdynamicsml.json"


def workflow_from_clusterdynamicsml_run_config(
    *,
    project_dir: str | Path,
    config: ClusterDynamicsMLRunConfig,
) -> ClusterDynamicsMLWorkflow:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster dynamics ML run file is missing frames_dir.")
    return ClusterDynamicsMLWorkflow(
        frames_dir,
        atom_type_definitions=config.atom_type_definitions,
        pair_cutoff_definitions=config.pair_cutoff_definitions,
        clusters_dir=resolve_run_config_path(
            config.clusters_dir,
            project_dir=resolved_project_dir,
        ),
        project_dir=resolve_run_config_path(
            config.project_dir,
            project_dir=resolved_project_dir,
        )
        or resolved_project_dir,
        experimental_data_file=resolve_run_config_path(
            config.experimental_data_file,
            project_dir=resolved_project_dir,
        ),
        box_dimensions=config.box_dimensions,
        use_pbc=config.use_pbc,
        default_cutoff=config.default_cutoff,
        shell_levels=config.shell_levels,
        shared_shells=config.shared_shells,
        include_shell_atoms_in_stoichiometry=(
            config.include_shell_atoms_in_stoichiometry
        ),
        search_mode=config.search_mode,
        pdb_shell_reference_definitions=config.shell_reference_definitions,
        folder_start_time_fs=config.folder_start_time_fs,
        first_frame_time_fs=config.first_frame_time_fs,
        frame_timestep_fs=config.frame_timestep_fs,
        frames_per_colormap_timestep=config.frames_per_colormap_timestep,
        analysis_start_fs=config.analysis_start_fs,
        analysis_stop_fs=config.analysis_stop_fs,
        energy_file=resolve_run_config_path(
            config.energy_file,
            project_dir=resolved_project_dir,
        ),
        target_node_counts=config.target_node_counts,
        candidates_per_size=config.candidates_per_size,
        prediction_population_share_threshold=(
            config.prediction_population_share_threshold
        ),
        q_min=config.q_min,
        q_max=config.q_max,
        q_points=config.q_points,
    )


def preview_clusterdynamicsml_run_config(
    *,
    project_dir: str | Path,
    config: ClusterDynamicsMLRunConfig,
) -> dict[str, object]:
    workflow = workflow_from_clusterdynamicsml_run_config(
        project_dir=project_dir,
        config=config,
    )
    preview = workflow.preview_selection()
    return {
        "dynamics": preview.dynamics_preview.to_dict(),
        "clusters_dir": (
            None if preview.clusters_dir is None else str(preview.clusters_dir)
        ),
        "project_dir": (
            None if preview.project_dir is None else str(preview.project_dir)
        ),
        "experimental_data_path": (
            None
            if preview.experimental_data_path is None
            else str(preview.experimental_data_path)
        ),
        "structure_label_count": int(preview.structure_label_count),
        "total_structure_files": int(preview.total_structure_files),
        "observed_node_counts": list(preview.observed_node_counts),
        "target_node_counts": list(preview.target_node_counts),
        "warnings": list(preview.warnings),
    }


def run_clusterdynamicsml_run_config(
    project_dir: str | Path,
    config: ClusterDynamicsMLRunConfig,
    *,
    run_file_path: str | Path | None = None,
    log_callback: ClusterDynamicsMLRunLogCallback | None = None,
) -> ClusterDynamicsMLRunExecutionSummary:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster dynamics ML run file is missing frames_dir.")
    output_file = resolve_run_config_path(
        config.output_file,
        project_dir=resolved_project_dir,
    )
    if output_file is None:
        output_file = suggest_clusterdynamicsml_output_file(
            project_dir=resolved_project_dir,
            frames_dir=frames_dir,
        )
    workflow = workflow_from_clusterdynamicsml_run_config(
        project_dir=resolved_project_dir,
        config=config,
    )
    _emit_log(log_callback, f"Frames folder: {frames_dir}")
    _emit_log(log_callback, f"Output dataset: {output_file}")
    result = workflow.analyze(progress_callback=log_callback)
    saved = save_cluster_dynamicsai_dataset(
        result,
        output_file,
        analysis_settings=config.to_dict(),
    )
    project_file = _register_project_references(
        resolved_project_dir,
        frames_dir=frames_dir,
        clusters_dir=resolve_run_config_path(
            config.clusters_dir,
            project_dir=resolved_project_dir,
        ),
        energy_file=resolve_run_config_path(
            config.energy_file,
            project_dir=resolved_project_dir,
        ),
        experimental_data_file=resolve_run_config_path(
            config.experimental_data_file,
            project_dir=resolved_project_dir,
        ),
    )
    _emit_log(
        log_callback,
        f"Saved cluster-dynamics ML dataset: {saved.dataset_file}",
    )
    return ClusterDynamicsMLRunExecutionSummary(
        project_dir=resolved_project_dir,
        run_file_path=(
            None if run_file_path is None else Path(run_file_path).resolve()
        ),
        frames_dir=frames_dir,
        output_file=saved.dataset_file,
        result=result,
        saved_dataset=saved,
        project_file=project_file,
    )


def serialize_shell_reference_definition(
    definition: PDBShellReferenceDefinition,
) -> dict[str, str | None]:
    return {
        "shell_element": definition.shell_element,
        "shell_residue": definition.shell_residue,
        "reference_name": definition.reference_name,
        "backbone_atom1_name": definition.backbone_atom1_name,
        "backbone_atom2_name": definition.backbone_atom2_name,
    }


def coerce_shell_reference_definitions(
    value: object,
) -> tuple[PDBShellReferenceDefinition, ...]:
    if not isinstance(value, list):
        return ()
    definitions: list[PDBShellReferenceDefinition] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        shell_element = str(entry.get("shell_element", "")).strip().title()
        reference_name = str(entry.get("reference_name", "")).strip()
        if not shell_element or not reference_name:
            continue
        definitions.append(
            PDBShellReferenceDefinition(
                shell_element=shell_element,
                shell_residue=optional_text(entry.get("shell_residue")),
                reference_name=reference_name,
                backbone_atom1_name=optional_text(
                    entry.get("backbone_atom1_name")
                ),
                backbone_atom2_name=optional_text(
                    entry.get("backbone_atom2_name")
                ),
            )
        )
    return tuple(definitions)


def _register_project_references(
    project_dir: Path,
    *,
    frames_dir: Path,
    clusters_dir: Path | None,
    energy_file: Path | None,
    experimental_data_file: Path | None,
) -> Path | None:
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        return None
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.frames_dir = str(Path(frames_dir).expanduser().resolve())
    settings.clusters_dir = (
        None
        if clusters_dir is None
        else str(Path(clusters_dir).expanduser().resolve())
    )
    settings.energy_file = (
        None
        if energy_file is None
        else str(Path(energy_file).expanduser().resolve())
    )
    if experimental_data_file is not None:
        settings.experimental_data_path = str(
            Path(experimental_data_file).expanduser().resolve()
        )
    return manager.save_project(settings)


def _emit_log(
    callback: ClusterDynamicsMLRunLogCallback | None,
    message: str,
) -> None:
    if callback is not None:
        callback(str(message).strip())


__all__ = [
    "DEFAULT_RUN_FILE_NAME",
    "ClusterDynamicsMLRunConfig",
    "ClusterDynamicsMLRunExecutionSummary",
    "build_clusterdynamicsml_run_config",
    "coerce_shell_reference_definitions",
    "default_clusterdynamicsml_run_file_path",
    "load_clusterdynamicsml_run_config",
    "preview_clusterdynamicsml_run_config",
    "run_clusterdynamicsml_run_config",
    "save_clusterdynamicsml_run_config",
    "serialize_shell_reference_definition",
    "suggest_clusterdynamicsml_output_file",
    "workflow_from_clusterdynamicsml_run_config",
]
