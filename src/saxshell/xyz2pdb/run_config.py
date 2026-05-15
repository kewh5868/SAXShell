from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    ReferenceBondToleranceInput,
    XYZToPDBMappingWorkflow,
)

DEFAULT_RUN_FILE_NAME = "xyz2pdb_cli_run.json"
RUN_CONFIG_VERSION = 1
XYZToPDBRunLogCallback = Callable[[str], None]
XYZToPDBRunProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class XYZToPDBRunConfig:
    input_path: str
    output_dir: str | None
    reference_library_dir: str | None = None
    molecule_inputs: tuple[MoleculeMappingInput, ...] = ()
    free_atom_inputs: tuple[FreeAtomMappingInput, ...] = ()
    hydrogen_mode: str = "leave_unassigned"
    selected_solution_index: int = 0
    assertion_mode: bool = False
    pbc_params: dict[str, float | str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": RUN_CONFIG_VERSION,
            "created_at": self.created_at,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "reference_library_dir": self.reference_library_dir,
            "molecule_inputs": [
                _molecule_input_to_dict(item) for item in self.molecule_inputs
            ],
            "free_atom_inputs": [
                _free_atom_input_to_dict(item)
                for item in self.free_atom_inputs
            ],
            "hydrogen_mode": self.hydrogen_mode,
            "selected_solution_index": int(self.selected_solution_index),
            "assertion_mode": bool(self.assertion_mode),
            "pbc_params": dict(self.pbc_params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "XYZToPDBRunConfig":
        input_path = str(payload.get("input_path", "")).strip()
        if not input_path:
            raise ValueError("XYZ-to-PDB run file is missing input_path.")
        return cls(
            input_path=input_path,
            output_dir=_optional_text(payload.get("output_dir")),
            reference_library_dir=_optional_text(
                payload.get("reference_library_dir")
            ),
            molecule_inputs=tuple(
                _molecule_input_from_dict(item)
                for item in _dict_items(payload.get("molecule_inputs"))
            ),
            free_atom_inputs=tuple(
                _free_atom_input_from_dict(item)
                for item in _dict_items(payload.get("free_atom_inputs"))
            ),
            hydrogen_mode=str(
                payload.get("hydrogen_mode", "leave_unassigned")
            ).strip()
            or "leave_unassigned",
            selected_solution_index=max(
                int(payload.get("selected_solution_index", 0)),
                0,
            ),
            assertion_mode=bool(payload.get("assertion_mode", False)),
            pbc_params=_pbc_params_from_dict(payload.get("pbc_params")),
            created_at=str(payload.get("created_at", "")).strip()
            or datetime.now().isoformat(timespec="seconds"),
        )


@dataclass(slots=True, frozen=True)
class XYZToPDBRunExecutionSummary:
    project_dir: Path
    run_file_path: Path | None
    output_dir: Path
    written_files: tuple[Path, ...]
    project_file: Path

    @property
    def written_count(self) -> int:
        return len(self.written_files)


def default_xyz2pdb_run_file_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / DEFAULT_RUN_FILE_NAME


def save_xyz2pdb_run_config(
    output_path: str | Path,
    config: XYZToPDBRunConfig,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_xyz2pdb_run_config(run_file_path: str | Path) -> XYZToPDBRunConfig:
    path = Path(run_file_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"XYZ-to-PDB run file must be a JSON object: {path}")
    return XYZToPDBRunConfig.from_dict(payload)


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


def build_xyz2pdb_run_config(
    *,
    project_dir: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None,
    reference_library_dir: str | Path | None = None,
    molecule_inputs: tuple[MoleculeMappingInput, ...] = (),
    free_atom_inputs: tuple[FreeAtomMappingInput, ...] = (),
    hydrogen_mode: str = "leave_unassigned",
    selected_solution_index: int = 0,
    assertion_mode: bool = False,
    pbc_params: dict[str, float | str] | None = None,
) -> XYZToPDBRunConfig:
    return XYZToPDBRunConfig(
        input_path=path_text_for_run_config(
            input_path,
            project_dir=project_dir,
        )
        or "",
        output_dir=path_text_for_run_config(
            output_dir,
            project_dir=project_dir,
        ),
        reference_library_dir=path_text_for_run_config(
            reference_library_dir,
            project_dir=project_dir,
        ),
        molecule_inputs=tuple(molecule_inputs),
        free_atom_inputs=tuple(free_atom_inputs),
        hydrogen_mode=hydrogen_mode,
        selected_solution_index=max(int(selected_solution_index), 0),
        assertion_mode=bool(assertion_mode),
        pbc_params=dict(pbc_params or {}),
    )


def run_xyz2pdb_run_config(
    project_dir: str | Path,
    config: XYZToPDBRunConfig,
    *,
    run_file_path: str | Path | None = None,
    log_callback: XYZToPDBRunLogCallback | None = None,
    progress_callback: XYZToPDBRunProgressCallback | None = None,
) -> XYZToPDBRunExecutionSummary:
    from saxshell.saxs.project_manager import SAXSProjectManager

    resolved_project_dir = Path(project_dir).expanduser().resolve()
    input_path = resolve_run_config_path(
        config.input_path,
        project_dir=resolved_project_dir,
    )
    if input_path is None:
        raise ValueError("XYZ-to-PDB run file is missing input_path.")
    output_dir = resolve_run_config_path(
        config.output_dir,
        project_dir=resolved_project_dir,
    )
    reference_library_dir = resolve_run_config_path(
        config.reference_library_dir,
        project_dir=resolved_project_dir,
    )
    _emit_log(log_callback, f"Starting XYZ-to-PDB conversion: {input_path}")
    workflow = XYZToPDBMappingWorkflow(
        input_path,
        reference_library_dir=reference_library_dir,
        output_dir=output_dir,
    )
    result = workflow.export_with_mapping(
        molecule_inputs=config.molecule_inputs,
        free_atom_inputs=config.free_atom_inputs,
        hydrogen_mode=config.hydrogen_mode,
        pbc_params=config.pbc_params,
        selected_solution_index=config.selected_solution_index,
        output_dir=output_dir,
        assert_molecule_shapes=config.assertion_mode,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    manager = SAXSProjectManager()
    settings = manager.load_project(resolved_project_dir)
    settings.pdb_frames_dir = str(result.output_dir.expanduser().resolve())
    project_file = manager.save_project(settings)
    _emit_log(
        log_callback,
        "Registered converted PDB frames with project: "
        f"{settings.pdb_frames_dir}",
    )
    return XYZToPDBRunExecutionSummary(
        project_dir=resolved_project_dir,
        run_file_path=(
            None if run_file_path is None else Path(run_file_path).resolve()
        ),
        output_dir=result.output_dir,
        written_files=result.written_files,
        project_file=project_file,
    )


def _molecule_input_to_dict(item: MoleculeMappingInput) -> dict[str, object]:
    return {
        "reference_name": item.reference_name,
        "residue_name": item.residue_name,
        "bond_tolerances": [
            {
                "atom1_name": bond.atom1_name,
                "atom2_name": bond.atom2_name,
                "tolerance": float(bond.tolerance),
            }
            for bond in item.bond_tolerances
        ],
        "tight_pass_scale": float(item.tight_pass_scale),
        "relaxed_pass_scale": float(item.relaxed_pass_scale),
        "max_assignment_distance": item.max_assignment_distance,
        "max_missing_hydrogens": int(item.max_missing_hydrogens),
    }


def _molecule_input_from_dict(
    payload: dict[str, object]
) -> MoleculeMappingInput:
    return MoleculeMappingInput(
        reference_name=str(payload.get("reference_name", "")).strip(),
        residue_name=str(payload.get("residue_name", "")).strip(),
        bond_tolerances=tuple(
            ReferenceBondToleranceInput(
                atom1_name=str(bond.get("atom1_name", "")).strip(),
                atom2_name=str(bond.get("atom2_name", "")).strip(),
                tolerance=float(bond.get("tolerance", 0.0)),
            )
            for bond in _dict_items(payload.get("bond_tolerances"))
        ),
        tight_pass_scale=float(payload.get("tight_pass_scale", 0.85)),
        relaxed_pass_scale=float(payload.get("relaxed_pass_scale", 1.35)),
        max_assignment_distance=(
            None
            if payload.get("max_assignment_distance") is None
            else float(payload.get("max_assignment_distance"))
        ),
        max_missing_hydrogens=max(
            int(payload.get("max_missing_hydrogens", 0)),
            0,
        ),
    )


def _free_atom_input_to_dict(item: FreeAtomMappingInput) -> dict[str, object]:
    return {
        "element": item.element,
        "residue_name": item.residue_name,
    }


def _free_atom_input_from_dict(
    payload: dict[str, object]
) -> FreeAtomMappingInput:
    return FreeAtomMappingInput(
        element=str(payload.get("element", "")).strip(),
        residue_name=str(payload.get("residue_name", "")).strip(),
    )


def _pbc_params_from_dict(value: object) -> dict[str, float | str]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float | str] = {}
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        if value.get(key) is not None:
            parsed[key] = float(value[key])
    if value.get("space_group") is not None:
        parsed["space_group"] = str(value["space_group"])
    return parsed


def _dict_items(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _emit_log(
    callback: XYZToPDBRunLogCallback | None,
    message: str,
) -> None:
    if callback is not None:
        callback(str(message).strip())


__all__ = [
    "DEFAULT_RUN_FILE_NAME",
    "XYZToPDBRunConfig",
    "XYZToPDBRunExecutionSummary",
    "build_xyz2pdb_run_config",
    "default_xyz2pdb_run_file_path",
    "load_xyz2pdb_run_config",
    "path_text_for_run_config",
    "resolve_run_config_path",
    "run_xyz2pdb_run_config",
    "save_xyz2pdb_run_config",
]
