from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.fullrmc.packmol_planning import PackmolPlanningMetadata
from saxshell.fullrmc.representatives import (
    RepresentativeSelectionEntry,
    RepresentativeSelectionMetadata,
)
from saxshell.fullrmc.solvent_handling import SolventHandlingMetadata
from saxshell.saxs.debye import load_structure_file
from saxshell.structure import PDBAtom, PDBStructure

if False:  # pragma: no cover
    from .project_loader import RMCDreamProjectSource


@dataclass(slots=True)
class PackmolSetupSettings:
    tolerance_angstrom: float = 2.0
    output_filename: str = "packmol_combined.inp"
    packed_output_filename: str = "packed_combined.pdb"
    use_completed_representatives: bool = True
    include_free_solvent: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolSetupSettings":
        source = dict(payload or {})
        return cls(
            tolerance_angstrom=max(
                0.1,
                float(source.get("tolerance_angstrom", 2.0)),
            ),
            output_filename=_safe_filename(
                str(source.get("output_filename", "packmol_combined.inp"))
            ),
            packed_output_filename=_safe_filename(
                str(
                    source.get(
                        "packed_output_filename",
                        "packed_combined.pdb",
                    )
                )
            ),
            use_completed_representatives=bool(
                source.get("use_completed_representatives", True)
            ),
            include_free_solvent=bool(
                source.get("include_free_solvent", True)
            ),
        )


@dataclass(slots=True)
class PackmolSetupEntry:
    structure: str
    motif: str
    param: str
    planned_count: int
    selected_weight: float
    planned_count_weight: float
    planned_atom_weight: float
    residue_name: str
    source_pdb: str
    packmol_pdb: str
    atom_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "PackmolSetupEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            planned_count=int(payload.get("planned_count", 0)),
            selected_weight=float(payload.get("selected_weight", 0.0)),
            planned_count_weight=float(
                payload.get("planned_count_weight", 0.0)
            ),
            planned_atom_weight=float(payload.get("planned_atom_weight", 0.0)),
            residue_name=str(payload.get("residue_name", "")).strip(),
            source_pdb=str(payload.get("source_pdb", "")).strip(),
            packmol_pdb=str(payload.get("packmol_pdb", "")).strip(),
            atom_count=int(payload.get("atom_count", 0)),
        )


@dataclass(slots=True)
class PackmolSetupMetadata:
    settings: PackmolSetupSettings
    updated_at: str
    planning_mode: str
    representative_selection_mode: str
    box_side_length_a: float
    packmol_input_path: str
    packed_output_filename: str
    solvent_pdb_path: str | None
    free_solvent_molecules: int
    audit_report_path: str
    entries: list[PackmolSetupEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.settings.to_dict(),
            "updated_at": self.updated_at,
            "planning_mode": self.planning_mode,
            "representative_selection_mode": (
                self.representative_selection_mode
            ),
            "box_side_length_a": self.box_side_length_a,
            "packmol_input_path": self.packmol_input_path,
            "packed_output_filename": self.packed_output_filename,
            "solvent_pdb_path": self.solvent_pdb_path,
            "free_solvent_molecules": self.free_solvent_molecules,
            "audit_report_path": self.audit_report_path,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolSetupMetadata | None":
        if not payload:
            return None
        return cls(
            settings=PackmolSetupSettings.from_dict(
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else None
            ),
            updated_at=str(payload.get("updated_at", "")).strip(),
            planning_mode=str(payload.get("planning_mode", "")).strip(),
            representative_selection_mode=str(
                payload.get("representative_selection_mode", "")
            ).strip(),
            box_side_length_a=float(payload.get("box_side_length_a", 0.0)),
            packmol_input_path=str(
                payload.get("packmol_input_path", "")
            ).strip(),
            packed_output_filename=str(
                payload.get("packed_output_filename", "")
            ).strip(),
            solvent_pdb_path=_optional_text(payload.get("solvent_pdb_path")),
            free_solvent_molecules=int(
                payload.get("free_solvent_molecules", 0)
            ),
            audit_report_path=str(
                payload.get("audit_report_path", "")
            ).strip(),
            entries=[
                PackmolSetupEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
        )

    def summary_text(self) -> str:
        lines = [
            f"Planning mode: {self.planning_mode}",
            f"Representative mode: {self.representative_selection_mode}",
            f"Saved at: {self.updated_at}",
            f"Box side: {self.box_side_length_a:.3f} A",
            f"Packmol input: {Path(self.packmol_input_path).name}",
            f"Representative PDBs copied: {len(self.entries)}",
            f"Free solvent molecules: {self.free_solvent_molecules}",
        ]
        if self.entries:
            first = self.entries[0]
            lines.extend(
                [
                    "",
                    "Example Packmol structure:",
                    f"  {first.structure}/{first.motif}",
                    f"  residue: {first.residue_name}",
                    f"  count: {first.planned_count}",
                    f"  file: {Path(first.packmol_pdb).name}",
                ]
            )
        if self.solvent_pdb_path:
            lines.append(f"Solvent input: {Path(self.solvent_pdb_path).name}")
        return "\n".join(lines)


def build_packmol_setup(
    project_source: "RMCDreamProjectSource",
    settings: PackmolSetupSettings | None = None,
    *,
    plan_metadata: PackmolPlanningMetadata | None = None,
    representative_metadata: RepresentativeSelectionMetadata | None = None,
    solvent_metadata: SolventHandlingMetadata | None = None,
) -> PackmolSetupMetadata:
    active_settings = settings or PackmolSetupSettings()
    active_plan = plan_metadata or project_source.packmol_planning
    if active_plan is None or not active_plan.entries:
        raise ValueError(
            "Compute Packmol planning counts before building Packmol setup."
        )
    active_representatives = (
        representative_metadata or project_source.representative_selection
    )
    if (
        active_representatives is None
        or not active_representatives.representative_entries
    ):
        raise ValueError(
            "Compute representative clusters before building Packmol setup."
        )

    active_solvent = solvent_metadata or project_source.solvent_handling
    if active_settings.include_free_solvent:
        if active_solvent is None:
            raise ValueError(
                "Build representative solvent outputs before generating Packmol inputs."
            )
        if not active_solvent.reference_path:
            raise ValueError(
                "No solvent reference PDB is available for Packmol input generation."
            )

    representative_lookup = {
        (entry.structure, entry.motif, entry.param): entry
        for entry in active_representatives.representative_entries
    }
    solvent_lookup = {}
    if active_solvent is not None:
        solvent_lookup = {
            (entry.structure, entry.motif, entry.param): entry
            for entry in active_solvent.entries
        }

    entries: list[PackmolSetupEntry] = []
    box_side_length_a = active_plan.settings.box_side_length_a
    for index, plan_entry in enumerate(active_plan.entries):
        if plan_entry.planned_count <= 0:
            continue
        key = (plan_entry.structure, plan_entry.motif, plan_entry.param)
        representative_entry = representative_lookup.get(key)
        if representative_entry is None:
            raise ValueError(
                "Packmol planning referenced a cluster bin without a representative: "
                f"{plan_entry.structure}/{plan_entry.motif}"
            )
        source_structure, source_pdb_path = _resolve_structure_for_packmol(
            representative_entry,
            solvent_lookup.get(key),
            use_completed=active_settings.use_completed_representatives,
        )
        residue_name = _packmol_residue_code(index)
        packmol_filename = (
            f"{index + 1:03d}_"
            f"{_safe_name(plan_entry.structure)}_"
            f"{_safe_name(plan_entry.motif)}_"
            f"{residue_name}.pdb"
        )
        packmol_path = (
            project_source.rmcsetup_paths.packmol_inputs_dir / packmol_filename
        )
        prepared_structure = _prepare_packmol_structure(
            source_structure,
            residue_name=residue_name,
        )
        prepared_structure.write_pdb_file(packmol_path)
        entries.append(
            PackmolSetupEntry(
                structure=plan_entry.structure,
                motif=plan_entry.motif,
                param=plan_entry.param,
                planned_count=plan_entry.planned_count,
                selected_weight=plan_entry.selected_weight,
                planned_count_weight=plan_entry.planned_count_weight,
                planned_atom_weight=plan_entry.planned_atom_weight,
                residue_name=residue_name,
                source_pdb=str(source_pdb_path),
                packmol_pdb=str(packmol_path),
                atom_count=len(prepared_structure.atoms),
            )
        )

    if not entries:
        raise ValueError(
            "The current Packmol plan did not produce any cluster entries with positive counts."
        )

    solvent_pdb_path: str | None = None
    free_solvent_molecules = 0
    if active_settings.include_free_solvent and active_solvent is not None:
        source_solvent = (
            Path(active_solvent.reference_path).expanduser().resolve()
        )
        free_solvent_molecules = int(
            round(
                float(
                    active_plan.target_box_composition.get(
                        "solvent_molecules",
                        0,
                    )
                )
            )
        )
        solvent_copy_name = (
            f"{_safe_name(active_solvent.reference_name)}_single.pdb"
        )
        destination = (
            project_source.rmcsetup_paths.packmol_inputs_dir
            / solvent_copy_name
        )
        shutil.copy2(source_solvent, destination)
        solvent_pdb_path = str(destination)

    input_path = _write_packmol_input(
        project_source.rmcsetup_paths.packmol_inputs_dir,
        entries,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_molecules=free_solvent_molecules,
        box_side_length_a=box_side_length_a,
        settings=active_settings,
    )
    audit_path = _write_packmol_audit_report(
        project_source,
        active_plan,
        entries,
        input_path=input_path,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_molecules=free_solvent_molecules,
    )
    metadata = PackmolSetupMetadata(
        settings=active_settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        planning_mode=active_plan.settings.planning_mode,
        representative_selection_mode=active_representatives.selection_mode,
        box_side_length_a=box_side_length_a,
        packmol_input_path=str(input_path),
        packed_output_filename=active_settings.packed_output_filename,
        solvent_pdb_path=solvent_pdb_path,
        free_solvent_molecules=free_solvent_molecules,
        audit_report_path=str(audit_path),
        entries=entries,
    )
    save_packmol_setup_metadata(
        project_source.rmcsetup_paths.packmol_setup_path,
        metadata,
    )
    return metadata


def save_packmol_setup_metadata(
    output_path: str | Path,
    metadata: PackmolSetupMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_packmol_setup_metadata(
    metadata_path: str | Path,
) -> PackmolSetupMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return PackmolSetupMetadata.from_dict(payload)


def _resolve_structure_for_packmol(
    representative_entry: RepresentativeSelectionEntry,
    solvent_entry: object | None,
    *,
    use_completed: bool,
) -> tuple[PDBStructure, Path]:
    candidate_path: Path | None = None
    if solvent_entry is not None and use_completed:
        candidate_path = Path(
            getattr(solvent_entry, "completed_pdb", "")
        ).expanduser()
    if (
        candidate_path is None
        or not str(candidate_path)
        or not candidate_path.is_file()
    ) and solvent_entry is not None:
        candidate_path = Path(
            getattr(solvent_entry, "no_solvent_pdb", "")
        ).expanduser()
    if (
        candidate_path is None
        or not str(candidate_path)
        or not candidate_path.is_file()
    ):
        source_path = (
            Path(representative_entry.source_file).expanduser().resolve()
        )
        return (
            _load_structure_as_pdb(
                source_path,
                structure_label=representative_entry.structure,
            ),
            source_path,
        )
    resolved = candidate_path.resolve()
    return PDBStructure.from_file(resolved), resolved


def _prepare_packmol_structure(
    structure: PDBStructure,
    *,
    residue_name: str,
) -> PDBStructure:
    copied_atoms = [atom.copy() for atom in structure.atoms]
    for index, atom in enumerate(copied_atoms, start=1):
        atom.atom_id = index
        atom.residue_number = 1
        atom.residue_name = residue_name
    prepared = PDBStructure(
        atoms=copied_atoms, source_name=structure.source_name
    )
    prepared.rename_atom_names_by_element(reindex_serial=True)
    return prepared


def _write_packmol_input(
    output_dir: Path,
    entries: list[PackmolSetupEntry],
    *,
    solvent_pdb_path: str | None,
    free_solvent_molecules: int,
    box_side_length_a: float,
    settings: PackmolSetupSettings,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / settings.output_filename
    with input_path.open("w", encoding="utf-8") as handle:
        handle.write(f"tolerance {settings.tolerance_angstrom:.3f}\n")
        handle.write("filetype pdb\n")
        handle.write(f"output {settings.packed_output_filename}\n\n")
        for entry in entries:
            handle.write(f"structure {Path(entry.packmol_pdb).name}\n")
            handle.write(f"  number {entry.planned_count}\n")
            handle.write(
                "  inside box 0.0 0.0 0.0 "
                f"{box_side_length_a:.3f} {box_side_length_a:.3f} {box_side_length_a:.3f}\n"
            )
            handle.write("end structure\n\n")
        if solvent_pdb_path and free_solvent_molecules > 0:
            handle.write(f"structure {Path(solvent_pdb_path).name}\n")
            handle.write(f"  number {free_solvent_molecules}\n")
            handle.write(
                "  inside box 0.0 0.0 0.0 "
                f"{box_side_length_a:.3f} {box_side_length_a:.3f} {box_side_length_a:.3f}\n"
            )
            handle.write("end structure\n")
    return input_path


def _write_packmol_audit_report(
    project_source: "RMCDreamProjectSource",
    plan_metadata: PackmolPlanningMetadata,
    entries: list[PackmolSetupEntry],
    *,
    input_path: Path,
    solvent_pdb_path: str | None,
    free_solvent_molecules: int,
) -> Path:
    lines = [
        "# Packmol Build Audit",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project: {project_source.settings.project_dir}",
        "",
        "## Packmol Input",
        f"- Input file: {input_path}",
        f"- Referenced packed output: {input_path.parent / input_path.stem}.pdb",
        (
            "- Solvent input: "
            f"{solvent_pdb_path if solvent_pdb_path is not None else '(none)'}"
        ),
        f"- Free solvent molecules: {free_solvent_molecules}",
        "",
        "## Planned Clusters",
        f"- Planning mode: {plan_metadata.settings.planning_mode}",
        f"- Box side: {plan_metadata.settings.box_side_length_a:.3f} A",
        f"- Cluster entries: {len(entries)}",
        f"- Total cluster count: {sum(entry.planned_count for entry in entries)}",
        "",
        "## Structure Table",
        "| Structure | Motif | Param | Count | Residue | File |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            "| "
            f"{entry.structure} | {entry.motif} | {entry.param} | "
            f"{entry.planned_count} | {entry.residue_name} | "
            f"{Path(entry.packmol_pdb).name} |"
        )
    lines.extend(
        [
            "",
            "## Related Reports",
            f"- Count table: {project_source.rmcsetup_paths.cluster_counts_csv_path}",
            f"- Count-normalized weights: {project_source.rmcsetup_paths.planned_count_weights_csv_path}",
            f"- Atom-normalized weights: {project_source.rmcsetup_paths.planned_atom_weights_csv_path}",
            f"- Planning report: {project_source.rmcsetup_paths.packmol_plan_report_path}",
            "",
            "## Notes",
            "- Cluster PDBs were rewritten with unique residue names for Packmol use.",
            "- Free solvent counts currently follow the target solvent molecules from the solution-properties box composition.",
            "- Coordinated-solvent completion assumptions come from the saved solvent-handling step when available.",
        ]
    )
    audit_path = project_source.rmcsetup_paths.packmol_audit_report_path
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_path


def _packmol_residue_code(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 0:
        index = 0
    prefix = "C"
    middle = alphabet[(index // 26) % 26]
    suffix = alphabet[index % 26]
    return f"{prefix}{middle}{suffix}"


def _load_structure_as_pdb(
    source_file: str | Path,
    *,
    structure_label: str,
) -> PDBStructure:
    path = Path(source_file).expanduser().resolve()
    if path.suffix.lower() == ".pdb":
        source_structure = PDBStructure.from_file(path)
        copied_atoms = [atom.copy() for atom in source_structure.atoms]
        for index, atom in enumerate(copied_atoms, start=1):
            atom.atom_id = index
        return PDBStructure(atoms=copied_atoms, source_name=path.stem)

    positions, elements = load_structure_file(path)
    residue_name = _normalized_residue_name(structure_label)
    counters: dict[str, int] = {}
    atoms: list[PDBAtom] = []
    for index, (coordinates, element) in enumerate(
        zip(positions, elements, strict=True),
        start=1,
    ):
        counters[element] = counters.get(element, 0) + 1
        atoms.append(
            PDBAtom(
                atom_id=index,
                atom_name=f"{element}{counters[element]}",
                residue_name=residue_name,
                residue_number=1,
                coordinates=np.asarray(coordinates, dtype=float),
                element=str(element),
            )
        )
    return PDBStructure(atoms=atoms, source_name=path.stem)


def _normalized_residue_name(text: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    if not collapsed:
        collapsed = "CLU"
    return collapsed[:3]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_name(text: str) -> str:
    collapsed = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip())
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return collapsed or "item"


def _safe_filename(text: str) -> str:
    name = Path(text.strip() or "item").name
    return name or "item"


__all__ = [
    "PackmolSetupEntry",
    "PackmolSetupMetadata",
    "PackmolSetupSettings",
    "build_packmol_setup",
    "load_packmol_setup_metadata",
    "save_packmol_setup_metadata",
]
