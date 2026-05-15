from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from saxshell.structure import (
    AtomTypeDefinitions,
    normalize_atom_type_definitions,
)

from .clusternetwork import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_KDTREE,
    PairCutoffDefinitions,
    normalize_pair_cutoffs,
    normalize_save_state_frequency,
    normalize_search_mode,
)
from .workflow import (
    ClusterExportResult,
    ClusterWorkflow,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
    suggest_output_dir,
)

DEFAULT_RUN_FILE_NAME = "cluster_extraction_cli_run.json"
RUN_CONFIG_VERSION = 1
ClusterRunLogCallback = Callable[[str], None]


@dataclass(slots=True)
class ClusterRunConfig:
    frames_dir: str
    output_dir: str | None
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
    include_shell_levels: tuple[int, ...] = (0,)
    shared_shells: bool = False
    smart_solvation_shells: bool = True
    include_shell_atoms_in_stoichiometry: bool = False
    search_mode: str = SEARCH_MODE_KDTREE
    save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": RUN_CONFIG_VERSION,
            "created_at": self.created_at,
            "frames_dir": self.frames_dir,
            "output_dir": self.output_dir,
            "atom_type_definitions": _serialize_atom_type_definitions(
                self.atom_type_definitions
            ),
            "pair_cutoff_definitions": _serialize_pair_cutoff_definitions(
                self.pair_cutoff_definitions
            ),
            "box_dimensions": self.box_dimensions,
            "use_pbc": bool(self.use_pbc),
            "default_cutoff": self.default_cutoff,
            "shell_levels": [int(level) for level in self.shell_levels],
            "include_shell_levels": [
                int(level) for level in self.include_shell_levels
            ],
            "shared_shells": bool(self.shared_shells),
            "smart_solvation_shells": bool(self.smart_solvation_shells),
            "include_shell_atoms_in_stoichiometry": bool(
                self.include_shell_atoms_in_stoichiometry
            ),
            "search_mode": normalize_search_mode(self.search_mode),
            "save_state_frequency": normalize_save_state_frequency(
                self.save_state_frequency
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ClusterRunConfig":
        frames_dir = str(payload.get("frames_dir", "")).strip()
        if not frames_dir:
            raise ValueError("Cluster run file is missing frames_dir.")
        return cls(
            frames_dir=frames_dir,
            output_dir=_optional_text(payload.get("output_dir")),
            atom_type_definitions=_coerce_atom_type_definitions(
                payload.get("atom_type_definitions")
            ),
            pair_cutoff_definitions=_coerce_pair_cutoff_definitions(
                payload.get("pair_cutoff_definitions")
            ),
            box_dimensions=_coerce_box_dimensions(
                payload.get("box_dimensions")
            ),
            use_pbc=bool(payload.get("use_pbc", False)),
            default_cutoff=_optional_float(payload.get("default_cutoff")),
            shell_levels=_coerce_int_tuple(payload.get("shell_levels")),
            include_shell_levels=(
                _coerce_int_tuple(payload.get("include_shell_levels")) or (0,)
            ),
            shared_shells=bool(payload.get("shared_shells", False)),
            smart_solvation_shells=bool(
                payload.get("smart_solvation_shells", True)
            ),
            include_shell_atoms_in_stoichiometry=bool(
                payload.get("include_shell_atoms_in_stoichiometry", False)
            ),
            search_mode=normalize_search_mode(
                str(payload.get("search_mode", SEARCH_MODE_KDTREE))
            ),
            save_state_frequency=normalize_save_state_frequency(
                _optional_int(
                    payload.get("save_state_frequency"),
                    DEFAULT_SAVE_STATE_FREQUENCY,
                )
            ),
            created_at=str(payload.get("created_at", "")).strip()
            or datetime.now().isoformat(timespec="seconds"),
        )


@dataclass(slots=True, frozen=True)
class ClusterRunExecutionSummary:
    project_dir: Path
    run_file_path: Path | None
    frames_dir: Path
    output_dir: Path
    result: ClusterExportResult
    project_file: Path

    @property
    def written_count(self) -> int:
        return len(self.result.written_files)


def default_cluster_run_file_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / DEFAULT_RUN_FILE_NAME


def save_cluster_run_config(
    output_path: str | Path,
    config: ClusterRunConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_cluster_run_config(run_file_path: str | Path) -> ClusterRunConfig:
    path = Path(run_file_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Cluster run file must contain a JSON object: {path}"
        )
    return ClusterRunConfig.from_dict(payload)


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


def build_cluster_run_config(
    *,
    project_dir: str | Path,
    frames_dir: str | Path,
    output_dir: str | Path | None,
    atom_type_definitions: AtomTypeDefinitions,
    pair_cutoff_definitions: PairCutoffDefinitions,
    box_dimensions: tuple[float, float, float] | None = None,
    use_pbc: bool = False,
    default_cutoff: float | None = None,
    shell_levels: tuple[int, ...] = (),
    include_shell_levels: tuple[int, ...] = (0,),
    shared_shells: bool = False,
    smart_solvation_shells: bool = True,
    include_shell_atoms_in_stoichiometry: bool = False,
    search_mode: str = SEARCH_MODE_KDTREE,
    save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY,
) -> ClusterRunConfig:
    return ClusterRunConfig(
        frames_dir=path_text_for_run_config(
            frames_dir,
            project_dir=project_dir,
        )
        or "",
        output_dir=path_text_for_run_config(
            output_dir,
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
        include_shell_levels=tuple(
            sorted({int(level) for level in include_shell_levels})
        )
        or (0,),
        shared_shells=bool(shared_shells),
        smart_solvation_shells=bool(smart_solvation_shells),
        include_shell_atoms_in_stoichiometry=bool(
            include_shell_atoms_in_stoichiometry
        ),
        search_mode=normalize_search_mode(search_mode),
        save_state_frequency=normalize_save_state_frequency(
            save_state_frequency
        ),
    )


def suggest_run_config_output_dir(
    *,
    frames_dir: str | Path,
) -> Path:
    return suggest_output_dir(frames_dir)


def workflow_from_cluster_run_config(
    *,
    project_dir: str | Path,
    config: ClusterRunConfig,
) -> ClusterWorkflow:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster run file is missing frames_dir.")
    return ClusterWorkflow(
        frames_dir=frames_dir,
        atom_type_definitions=config.atom_type_definitions,
        pair_cutoff_definitions=config.pair_cutoff_definitions,
        box_dimensions=config.box_dimensions,
        use_pbc=config.use_pbc,
        default_cutoff=config.default_cutoff,
        shell_levels=config.shell_levels,
        include_shell_levels=config.include_shell_levels,
        shared_shells=config.shared_shells,
        smart_solvation_shells=config.smart_solvation_shells,
        include_shell_atoms_in_stoichiometry=(
            config.include_shell_atoms_in_stoichiometry
        ),
        search_mode=config.search_mode,
        save_state_frequency=config.save_state_frequency,
    )


def preview_cluster_run_config(
    *,
    project_dir: str | Path,
    config: ClusterRunConfig,
) -> dict[str, object]:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    output_dir = resolve_run_config_path(
        config.output_dir,
        project_dir=resolved_project_dir,
    )
    workflow = workflow_from_cluster_run_config(
        project_dir=resolved_project_dir,
        config=config,
    )
    return workflow.preview_selection(output_dir=output_dir).to_dict()


def run_cluster_run_config(
    project_dir: str | Path,
    config: ClusterRunConfig,
    *,
    run_file_path: str | Path | None = None,
    log_callback: ClusterRunLogCallback | None = None,
) -> ClusterRunExecutionSummary:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    frames_dir = resolve_run_config_path(
        config.frames_dir,
        project_dir=resolved_project_dir,
    )
    if frames_dir is None:
        raise ValueError("Cluster run file is missing frames_dir.")
    output_dir = resolve_run_config_path(
        config.output_dir,
        project_dir=resolved_project_dir,
    )
    workflow = workflow_from_cluster_run_config(
        project_dir=resolved_project_dir,
        config=config,
    )
    _emit_log(log_callback, f"Frames folder: {frames_dir}")
    _emit_log(
        log_callback,
        "Output folder: "
        + str(
            output_dir
            if output_dir is not None
            else suggest_output_dir(frames_dir)
        ),
    )
    result = workflow.export_clusters(output_dir=output_dir)
    project_file = _register_project_clusters_dir(
        resolved_project_dir,
        result.output_dir,
    )
    _emit_log(log_callback, f"Project clusters folder: {result.output_dir}")
    return ClusterRunExecutionSummary(
        project_dir=resolved_project_dir,
        run_file_path=(
            None if run_file_path is None else Path(run_file_path).resolve()
        ),
        frames_dir=frames_dir,
        output_dir=result.output_dir,
        result=result,
        project_file=project_file,
    )


def _register_project_clusters_dir(
    project_dir: Path, clusters_dir: Path
) -> Path:
    from saxshell.saxs.project_manager import SAXSProjectManager

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.clusters_dir = str(Path(clusters_dir).expanduser().resolve())
    return manager.save_project(settings)


def _serialize_atom_type_definitions(
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


def _serialize_pair_cutoff_definitions(
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


def _coerce_atom_type_definitions(value: object) -> AtomTypeDefinitions:
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


def _coerce_pair_cutoff_definitions(value: object) -> PairCutoffDefinitions:
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


def _coerce_box_dimensions(
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


def _coerce_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(sorted({int(entry) for entry in value}))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    result = float(text)
    return result if result > 0.0 else None


def _optional_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    return int(text)


def _emit_log(callback: ClusterRunLogCallback | None, message: str) -> None:
    if callback is not None:
        callback(str(message).strip())


__all__ = [
    "DEFAULT_RUN_FILE_NAME",
    "ClusterRunConfig",
    "ClusterRunExecutionSummary",
    "build_cluster_run_config",
    "default_cluster_run_file_path",
    "load_cluster_run_config",
    "path_text_for_run_config",
    "preview_cluster_run_config",
    "resolve_run_config_path",
    "run_cluster_run_config",
    "save_cluster_run_config",
    "suggest_run_config_output_dir",
    "workflow_from_cluster_run_config",
]
