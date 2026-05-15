from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from saxshell.cluster import (
    SEARCH_MODE_KDTREE,
    PairCutoffDefinitions,
    normalize_pair_cutoffs,
    normalize_search_mode,
)
from saxshell.cluster.workflow import (
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.structure import (
    AtomTypeDefinitions,
    normalize_atom_type_definitions,
)

from .dataset import SavedClusterDynamicsDataset, save_cluster_dynamics_dataset
from .workflow import ClusterDynamicsResult, ClusterDynamicsWorkflow

DEFAULT_RUN_FILE_NAME = "cluster_dynamics_cli_run.json"
RUN_CONFIG_VERSION = 1
ClusterDynamicsRunLogCallback = Callable[[str], None]
ClusterDynamicsRunProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class ClusterDynamicsRunConfig:
    frames_dir: str
    output_file: str | None
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
    folder_start_time_fs: float | None = None
    first_frame_time_fs: float = 0.0
    frame_timestep_fs: float = 0.5
    frames_per_colormap_timestep: int = 1
    analysis_start_fs: float | None = None
    analysis_stop_fs: float | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": RUN_CONFIG_VERSION,
            "created_at": self.created_at,
            "frames_dir": self.frames_dir,
            "output_file": self.output_file,
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
            "folder_start_time_fs": self.folder_start_time_fs,
            "first_frame_time_fs": float(self.first_frame_time_fs),
            "frame_timestep_fs": float(self.frame_timestep_fs),
            "frames_per_colormap_timestep": int(
                self.frames_per_colormap_timestep
            ),
            "analysis_start_fs": self.analysis_start_fs,
            "analysis_stop_fs": self.analysis_stop_fs,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ClusterDynamicsRunConfig":
        frames_dir = str(payload.get("frames_dir", "")).strip()
        if not frames_dir:
            raise ValueError(
                "Cluster dynamics run file is missing frames_dir."
            )
        return cls(
            frames_dir=frames_dir,
            output_file=optional_text(payload.get("output_file")),
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
            created_at=str(payload.get("created_at", "")).strip()
            or datetime.now().isoformat(timespec="seconds"),
        )


@dataclass(slots=True, frozen=True)
class ClusterDynamicsRunExecutionSummary:
    project_dir: Path
    run_file_path: Path | None
    frames_dir: Path
    output_file: Path
    result: ClusterDynamicsResult
    saved_dataset: SavedClusterDynamicsDataset
    project_file: Path | None

    @property
    def written_count(self) -> int:
        return len(self.saved_dataset.written_files)


def default_clusterdynamics_run_file_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / DEFAULT_RUN_FILE_NAME


def save_clusterdynamics_run_config(
    output_path: str | Path,
    config: ClusterDynamicsRunConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_clusterdynamics_run_config(
    run_file_path: str | Path,
) -> ClusterDynamicsRunConfig:
    path = Path(run_file_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Cluster dynamics run file must contain a JSON object: {path}"
        )
    return ClusterDynamicsRunConfig.from_dict(payload)


def path_text_for_run_config(
    path: str | Path | None,
    *,
    project_dir: str | Path,
) -> str | None:
    if path is None:
        return None
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_project_dir).as_posix()
    except ValueError:
        return str(resolved_path)


def resolve_run_config_path(
    path_text: str | None,
    *,
    project_dir: str | Path,
) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(project_dir).expanduser().resolve() / path
    return path.resolve()


def build_clusterdynamics_run_config(
    *,
    project_dir: str | Path,
    frames_dir: str | Path,
    output_file: str | Path | None,
    energy_file: str | Path | None,
    atom_type_definitions: AtomTypeDefinitions,
    pair_cutoff_definitions: PairCutoffDefinitions,
    box_dimensions: tuple[float, float, float] | None = None,
    use_pbc: bool = False,
    default_cutoff: float | None = None,
    shell_levels: tuple[int, ...] = (),
    shared_shells: bool = False,
    include_shell_atoms_in_stoichiometry: bool = False,
    search_mode: str = SEARCH_MODE_KDTREE,
    folder_start_time_fs: float | None = None,
    first_frame_time_fs: float = 0.0,
    frame_timestep_fs: float = 0.5,
    frames_per_colormap_timestep: int = 1,
    analysis_start_fs: float | None = None,
    analysis_stop_fs: float | None = None,
) -> ClusterDynamicsRunConfig:
    return ClusterDynamicsRunConfig(
        frames_dir=path_text_for_run_config(
            frames_dir,
            project_dir=project_dir,
        )
        or "",
        output_file=path_text_for_run_config(
            output_file,
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
        folder_start_time_fs=folder_start_time_fs,
        first_frame_time_fs=float(first_frame_time_fs),
        frame_timestep_fs=float(frame_timestep_fs),
        frames_per_colormap_timestep=max(
            int(frames_per_colormap_timestep),
            1,
        ),
        analysis_start_fs=analysis_start_fs,
        analysis_stop_fs=analysis_stop_fs,
    )


def suggest_clusterdynamics_output_file(
    *,
    project_dir: str | Path,
    frames_dir: str | Path | None,
) -> Path:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    dataset_dir = (
        build_project_paths(resolved_project_dir).exported_data_dir
        / "clusterdynamics"
    )
    folder_label = "cluster_dynamics"
    if frames_dir is not None:
        frames_path = Path(frames_dir).expanduser()
        if frames_path.name:
            folder_label = frames_path.name
    return dataset_dir / f"{folder_label}_cluster_dynamics.json"


def workflow_from_clusterdynamics_run_config(
    *,
    project_dir: str | Path,
    config: ClusterDynamicsRunConfig,
) -> ClusterDynamicsWorkflow:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster dynamics run file is missing frames_dir.")
    energy_file = resolve_run_config_path(
        config.energy_file,
        project_dir=resolved_project_dir,
    )
    return ClusterDynamicsWorkflow(
        frames_dir,
        atom_type_definitions=config.atom_type_definitions,
        pair_cutoff_definitions=config.pair_cutoff_definitions,
        box_dimensions=config.box_dimensions,
        use_pbc=config.use_pbc,
        default_cutoff=config.default_cutoff,
        shell_levels=config.shell_levels,
        shared_shells=config.shared_shells,
        include_shell_atoms_in_stoichiometry=(
            config.include_shell_atoms_in_stoichiometry
        ),
        search_mode=config.search_mode,
        folder_start_time_fs=config.folder_start_time_fs,
        first_frame_time_fs=config.first_frame_time_fs,
        frame_timestep_fs=config.frame_timestep_fs,
        frames_per_colormap_timestep=config.frames_per_colormap_timestep,
        analysis_start_fs=config.analysis_start_fs,
        analysis_stop_fs=config.analysis_stop_fs,
        energy_file=energy_file,
    )


def preview_clusterdynamics_run_config(
    *,
    project_dir: str | Path,
    config: ClusterDynamicsRunConfig,
) -> dict[str, object]:
    workflow = workflow_from_clusterdynamics_run_config(
        project_dir=project_dir,
        config=config,
    )
    return workflow.preview_selection().to_dict()


def run_clusterdynamics_run_config(
    project_dir: str | Path,
    config: ClusterDynamicsRunConfig,
    *,
    run_file_path: str | Path | None = None,
    log_callback: ClusterDynamicsRunLogCallback | None = None,
    progress_callback: ClusterDynamicsRunProgressCallback | None = None,
) -> ClusterDynamicsRunExecutionSummary:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster dynamics run file is missing frames_dir.")
    output_file = resolve_run_config_path(
        config.output_file,
        project_dir=resolved_project_dir,
    )
    if output_file is None:
        output_file = suggest_clusterdynamics_output_file(
            project_dir=resolved_project_dir,
            frames_dir=frames_dir,
        )
    workflow = workflow_from_clusterdynamics_run_config(
        project_dir=resolved_project_dir,
        config=config,
    )
    _emit_log(log_callback, f"Frames folder: {frames_dir}")
    _emit_log(log_callback, f"Output dataset: {output_file}")
    result = workflow.analyze(progress_callback=progress_callback)
    saved = save_cluster_dynamics_dataset(
        result,
        output_file,
        analysis_settings=config.to_dict(),
    )
    project_file = _register_project_inputs(
        resolved_project_dir,
        frames_dir=frames_dir,
        energy_file=resolve_run_config_path(
            config.energy_file,
            project_dir=resolved_project_dir,
        ),
    )
    _emit_log(
        log_callback,
        f"Saved cluster-dynamics dataset: {saved.dataset_file}",
    )
    return ClusterDynamicsRunExecutionSummary(
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


def serialize_atom_type_definitions(
    definitions: AtomTypeDefinitions,
) -> dict[str, list[dict[str, str | None]]]:
    normalized = normalize_atom_type_definitions(definitions)
    return {
        atom_type: [
            {"element": element, "residue": residue}
            for element, residue in entries
        ]
        for atom_type, entries in normalized.items()
    }


def serialize_pair_cutoff_definitions(
    definitions: PairCutoffDefinitions,
) -> list[dict[str, object]]:
    normalized = normalize_pair_cutoffs(definitions)
    payload: list[dict[str, object]] = []
    for atom1, atom2 in sorted(normalized):
        payload.append(
            {
                "atom1": atom1,
                "atom2": atom2,
                "shell_cutoffs": {
                    str(level): float(cutoff)
                    for level, cutoff in sorted(
                        normalized[(atom1, atom2)].items()
                    )
                },
            }
        )
    return payload


def coerce_atom_type_definitions(value: object) -> AtomTypeDefinitions:
    if not isinstance(value, dict):
        return example_atom_type_definitions()
    definitions: AtomTypeDefinitions = {}
    for atom_type, entries in value.items():
        if not isinstance(entries, list):
            continue
        parsed: list[tuple[str, str | None]] = []
        for entry in entries:
            if isinstance(entry, dict):
                element_value = entry.get("element")
                residue_value = entry.get("residue")
            elif isinstance(entry, (list, tuple)):
                element_value = entry[0] if len(entry) >= 1 else None
                residue_value = entry[1] if len(entry) >= 2 else None
            else:
                element_value = entry
                residue_value = None
            element = str(element_value or "").strip().title()
            residue_text = str(residue_value or "").strip()
            if element:
                parsed.append((element, residue_text or None))
        if parsed:
            definitions[str(atom_type).strip()] = parsed
    return normalize_atom_type_definitions(definitions)


def coerce_pair_cutoff_definitions(value: object) -> PairCutoffDefinitions:
    if not isinstance(value, list):
        return example_pair_cutoff_definitions()
    definitions: PairCutoffDefinitions = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        atom1 = str(entry.get("atom1", "")).strip().title()
        atom2 = str(entry.get("atom2", "")).strip().title()
        cutoffs = entry.get("shell_cutoffs")
        if not atom1 or not atom2 or not isinstance(cutoffs, dict):
            continue
        parsed: dict[int, float] = {}
        for level, cutoff in cutoffs.items():
            parsed[int(level)] = float(cutoff)
        if parsed:
            definitions[(atom1, atom2)] = parsed
    return normalize_pair_cutoffs(definitions)


def coerce_box_dimensions(
    value: object,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("box_dimensions must be a list of three numbers.")
    box = tuple(float(component) for component in value)
    if len(box) != 3:
        raise ValueError("box_dimensions must contain exactly three numbers.")
    return box


def coerce_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(sorted({int(entry) for entry in value}))


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def optional_positive_float(value: object) -> float | None:
    result = optional_float(value)
    if result is None:
        return None
    return result if result > 0.0 else None


def _register_project_inputs(
    project_dir: Path,
    *,
    frames_dir: Path,
    energy_file: Path | None,
) -> Path | None:
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        return None
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.frames_dir = str(Path(frames_dir).expanduser().resolve())
    settings.energy_file = (
        None
        if energy_file is None
        else str(Path(energy_file).expanduser().resolve())
    )
    return manager.save_project(settings)


def _emit_log(
    callback: ClusterDynamicsRunLogCallback | None,
    message: str,
) -> None:
    if callback is not None:
        callback(str(message).strip())


__all__ = [
    "DEFAULT_RUN_FILE_NAME",
    "ClusterDynamicsRunConfig",
    "ClusterDynamicsRunExecutionSummary",
    "build_clusterdynamics_run_config",
    "coerce_atom_type_definitions",
    "coerce_box_dimensions",
    "coerce_int_tuple",
    "coerce_pair_cutoff_definitions",
    "default_clusterdynamics_run_file_path",
    "load_clusterdynamics_run_config",
    "optional_float",
    "optional_positive_float",
    "optional_text",
    "path_text_for_run_config",
    "preview_clusterdynamics_run_config",
    "resolve_run_config_path",
    "run_clusterdynamics_run_config",
    "save_clusterdynamics_run_config",
    "serialize_atom_type_definitions",
    "serialize_pair_cutoff_definitions",
    "suggest_clusterdynamics_output_file",
    "workflow_from_clusterdynamics_run_config",
]
