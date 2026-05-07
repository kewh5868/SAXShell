from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from saxshell.fullrmc.solvent_handling import (
    _best_solvent_orientation,
    _next_residue_number,
    _normalize_vector,
    _normalized_residue_name,
    _rotate_points,
    _rotation_between_vectors,
    _weighted_center,
)
from saxshell.saxs.debye import load_structure_file
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import (
    AnchorPairDefinition,
    MoleculeDefinition,
    ReferenceLibraryEntry,
    XYZToPDBConfiguration,
    XYZToPDBWorkflow,
    default_reference_library_dir,
    list_reference_library,
)
from saxshell.xyz2pdb.workflow import _normalized_atom_name

_DEFAULT_REFERENCE_MATCH_TOLERANCE_A = 0.25
DEFAULT_REFERENCE_MATCH_TOLERANCE_A = _DEFAULT_REFERENCE_MATCH_TOLERANCE_A


@dataclass(slots=True)
class SolventShellResidueSummary:
    residue_name: str
    molecule_count: int
    residue_numbers: tuple[int, ...]
    atom_count: int
    element_counts: dict[str, int]

    @property
    def residue_numbers_text(self) -> str:
        if not self.residue_numbers:
            return "n/a"
        return ", ".join(str(number) for number in self.residue_numbers)

    @property
    def element_counts_text(self) -> str:
        if not self.element_counts:
            return "none"
        return ", ".join(
            f"{element}:{count}"
            for element, count in sorted(self.element_counts.items())
        )


@dataclass(slots=True)
class SolventShellResidueMismatchSummary:
    residue_name: str
    residue_number: int
    observed_atom_count: int
    common_atom_count: int
    reference_atom_count: int
    missing_atom_names: tuple[str, ...]
    extra_atom_names: tuple[str, ...]
    distance_pair_count: int
    distribution_rmsd_a: float
    max_distance_delta_a: float
    mismatch_reason: str
    source_atom_ids: tuple[int, ...] = ()

    @property
    def residue_label(self) -> str:
        return f"{self.residue_name} {self.residue_number}"

    @property
    def missing_atom_names_text(self) -> str:
        if not self.missing_atom_names:
            return "none"
        return ", ".join(self.missing_atom_names)

    @property
    def extra_atom_names_text(self) -> str:
        if not self.extra_atom_names:
            return "none"
        return ", ".join(self.extra_atom_names)

    @property
    def matched_atom_ratio_text(self) -> str:
        return f"{self.common_atom_count}/{self.reference_atom_count}"

    @property
    def source_atom_ids_text(self) -> str:
        if not self.source_atom_ids:
            return "n/a"
        return ", ".join(str(value) for value in self.source_atom_ids)


@dataclass(slots=True)
class SolventShellAnalysisResult:
    input_path: Path
    input_format: str
    reference_name: str
    reference_path: Path
    reference_residue_name: str
    reference_atom_count: int
    detected_solvent_molecules: int
    matched_atom_count: int
    unmatched_atom_count: int
    total_atoms: int
    match_tolerance_a: float
    solute_element_counts: dict[str, int]
    complete_solvent_source_atom_ids: tuple[int, ...] = ()
    complete_solvent_source_atom_groups: tuple[tuple[int, ...], ...] = ()
    partial_solvent_source_atom_ids: tuple[int, ...] = ()
    matched_residue_summaries: tuple[SolventShellResidueSummary, ...] = ()
    residue_mismatch_summaries: tuple[
        SolventShellResidueMismatchSummary, ...
    ] = ()
    notes: tuple[str, ...] = ()

    @property
    def has_solvent_molecules(self) -> bool:
        return (
            self.complete_solvent_molecule_count > 0
            or self.partial_solvent_molecule_count > 0
        )

    @property
    def complete_solvent_molecule_count(self) -> int:
        return int(self.detected_solvent_molecules)

    @property
    def partial_solvent_molecule_count(self) -> int:
        return int(len(self.residue_mismatch_summaries))

    @property
    def partial_solvent_status_supported(self) -> bool:
        return True

    @property
    def complete_solvent_status_text(self) -> str:
        return "yes" if self.complete_solvent_molecule_count > 0 else "no"

    @property
    def partial_solvent_status_text(self) -> str:
        return "yes" if self.partial_solvent_molecule_count > 0 else "no"

    @property
    def no_solvent_status_text(self) -> str:
        return "yes" if not self.has_solvent_molecules else "no"

    @property
    def cluster_solvent_status_text(self) -> str:
        if self.complete_solvent_molecule_count > 0:
            if self.partial_solvent_molecule_count > 0:
                return "Complete and partial solvent molecules detected."
            return "Complete solvent molecules detected."
        if self.partial_solvent_molecule_count > 0:
            return "Partial solvent molecules detected."
        return "No solvent molecules detected."

    @property
    def solvent_presence_text(self) -> str:
        return "yes" if self.has_solvent_molecules else "no"

    @property
    def solute_elements_text(self) -> str:
        if not self.solute_element_counts:
            return "none"
        return ", ".join(
            f"{element}:{count}"
            for element, count in sorted(self.solute_element_counts.items())
        )

    def status_statistics_text(self) -> str:
        lines = [
            f"No solvent molecules: {self.no_solvent_status_text}",
            f"Partial solvent molecules: {self.partial_solvent_status_text}",
            f"Complete solvent molecules: {self.complete_solvent_status_text}",
            f"Complete solvent count: {self.complete_solvent_molecule_count}",
        ]
        if self.input_format == "pdb":
            lines.append(
                "Partial solvent residue count: "
                f"{self.partial_solvent_molecule_count}"
            )
        else:
            lines.append(
                "Partial solvent candidate count: "
                f"{self.partial_solvent_molecule_count}"
            )
        lines.extend(
            [
                f"Recognized solute elements: {self.solute_elements_text}",
                f"Matched atoms: {self.matched_atom_count}/{self.total_atoms}",
                f"Unmatched atoms: {self.unmatched_atom_count}",
            ]
        )
        return "\n".join(lines)

    def summary_text(self) -> str:
        lines = [
            f"Input file: {self.input_path}",
            f"Input format: {self.input_format.upper()}",
            f"Reference molecule: {self.reference_name}",
            f"Reference residue: {self.reference_residue_name}",
            f"Reference atom count: {self.reference_atom_count}",
            f"Reference match tolerance: {self.match_tolerance_a:.3g} A",
            f"Total atoms: {self.total_atoms}",
            f"Cluster solvent status: {self.cluster_solvent_status_text}",
            f"No solvent molecules: {self.no_solvent_status_text}",
            f"Partial solvent molecules: {self.partial_solvent_status_text}",
            f"Complete solvent molecules: {self.complete_solvent_status_text}",
            f"Complete solvent count: {self.complete_solvent_molecule_count}",
            f"Solvent molecules detected: {self.detected_solvent_molecules}",
            f"Solvent present: {self.solvent_presence_text}",
            f"Recognized solute elements: {self.solute_elements_text}",
            f"Matched atoms: {self.matched_atom_count}",
            f"Unmatched atoms: {self.unmatched_atom_count}",
        ]
        if self.input_format == "pdb":
            lines.append(
                "Partial solvent residue count: "
                f"{self.partial_solvent_molecule_count}"
            )
            lines.append(
                "Matched residue types: "
                f"{len(self.matched_residue_summaries)}"
            )
            lines.append(
                "Residue mismatches preserved: "
                f"{len(self.residue_mismatch_summaries)}"
            )
        else:
            lines.append(
                "Partial solvent candidate count: "
                f"{self.partial_solvent_molecule_count}"
            )
            lines.append("Matched residue types: n/a for XYZ inputs")
            lines.append(
                "Partial solvent candidates inferred: "
                f"{len(self.residue_mismatch_summaries)}"
            )
        if self.matched_residue_summaries:
            lines.extend(
                [
                    "",
                    "PDB residue matches:",
                ]
            )
            for summary in self.matched_residue_summaries:
                lines.append(
                    f"  {summary.residue_name}: {summary.molecule_count} "
                    f"molecule(s) in residue(s) {summary.residue_numbers_text}"
                )
        if self.residue_mismatch_summaries:
            lines.extend(
                [
                    "",
                    (
                        "PDB residue mismatches:"
                        if self.input_format == "pdb"
                        else "XYZ partial solvent candidates:"
                    ),
                ]
            )
            for summary in self.residue_mismatch_summaries:
                detail_parts = [
                    f"matched {summary.matched_atom_ratio_text} reference atom(s)",
                ]
                if summary.missing_atom_names:
                    detail_parts.append(
                        f"missing {summary.missing_atom_names_text}"
                    )
                if summary.extra_atom_names:
                    detail_parts.append(
                        f"extra {summary.extra_atom_names_text}"
                    )
                if summary.source_atom_ids:
                    detail_parts.append(
                        f"source atom ids {summary.source_atom_ids_text}"
                    )
                lines.append(
                    f"  {summary.residue_label}: {summary.mismatch_reason}; "
                    + ", ".join(detail_parts)
                )
        if self.notes:
            lines.extend(["", "Notes:"])
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass(slots=True)
class SolventShellBuildResult:
    input_path: Path
    output_path: Path
    input_format: str
    reference_name: str
    reference_residue_name: str
    director_atom_name: str
    build_mode: str
    solvent_molecules_added: int
    solvent_atoms_added: int
    partial_candidates_completed: int
    replaced_source_atom_count: int
    minimum_solvent_atom_separation_a: float
    solute_distance_cutoffs_a: dict[str, float]
    coordinating_center_elements: tuple[str, ...] = ()
    target_average_coordination_numbers: dict[str, float] | None = None
    achieved_average_coordination_numbers: dict[str, float] | None = None

    def summary_text(self) -> str:
        cutoff_text = (
            ", ".join(
                f"{element}:{distance:.3g}"
                for element, distance in sorted(
                    self.solute_distance_cutoffs_a.items()
                )
            )
            if self.solute_distance_cutoffs_a
            else "none"
        )
        center_text = (
            ", ".join(self.coordinating_center_elements)
            if self.coordinating_center_elements
            else "none"
        )
        target_coordination_text = (
            ", ".join(
                f"{element}:{value:.3g}"
                for element, value in sorted(
                    (self.target_average_coordination_numbers or {}).items()
                )
            )
            if self.target_average_coordination_numbers
            else "none"
        )
        achieved_coordination_text = (
            ", ".join(
                f"{element}:{value:.3g}"
                for element, value in sorted(
                    (self.achieved_average_coordination_numbers or {}).items()
                )
            )
            if self.achieved_average_coordination_numbers
            else "none"
        )
        return "\n".join(
            [
                f"Output file: {self.output_path}",
                f"Build mode: {self.build_mode}",
                f"Input format: {self.input_format.upper()}",
                f"Reference molecule: {self.reference_name}",
                f"Reference residue: {self.reference_residue_name}",
                f"Director atom: {self.director_atom_name}",
                (
                    "Minimum solvent atom separation: "
                    f"{self.minimum_solvent_atom_separation_a:.3g} A"
                ),
                f"Solute distance cutoffs: {cutoff_text}",
                f"Coordinating center elements: {center_text}",
                f"Target average coordination: {target_coordination_text}",
                f"Achieved average coordination: {achieved_coordination_text}",
                f"Solvent molecules added: {self.solvent_molecules_added}",
                f"Solvent atoms added: {self.solvent_atoms_added}",
                (
                    "Partial solvent candidates completed: "
                    f"{self.partial_candidates_completed}"
                ),
                (
                    "Source atoms replaced during completion: "
                    f"{self.replaced_source_atom_count}"
                ),
            ]
        )


@dataclass(slots=True)
class _MatchingAtom:
    atom_id: int
    element: str
    coordinates: np.ndarray


@dataclass(slots=True)
class _MatchingFrame:
    filepath: Path
    atoms: list[_MatchingAtom]


def analyze_solvent_shell(
    input_path: str | Path,
    reference_name: str,
    *,
    reference_library_dir: str | Path | None = None,
    reference_match_tolerance_a: float = _DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
) -> SolventShellAnalysisResult:
    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(
            f"Input structure file was not found: {resolved_input}"
        )

    resolved_library_dir = (
        default_reference_library_dir()
        if reference_library_dir is None
        else Path(reference_library_dir).expanduser().resolve()
    )
    reference_entry = _resolve_reference_entry(
        reference_name,
        library_dir=resolved_library_dir,
    )
    reference_path = reference_entry.path.expanduser().resolve()
    reference_structure = PDBStructure.from_file(reference_path)
    reference_atoms = tuple(atom.copy() for atom in reference_structure.atoms)
    if not reference_atoms:
        raise ValueError(
            f"Reference molecule has no atoms: {reference_entry.name}"
        )

    if resolved_input.suffix.lower() == ".pdb":
        return _analyze_pdb_input(
            resolved_input,
            reference_entry=reference_entry,
            reference_atoms=reference_atoms,
            reference_library_dir=resolved_library_dir,
            reference_match_tolerance_a=reference_match_tolerance_a,
        )
    if resolved_input.suffix.lower() == ".xyz":
        return _analyze_xyz_input(
            resolved_input,
            reference_entry=reference_entry,
            reference_atoms=reference_atoms,
            reference_library_dir=resolved_library_dir,
            reference_match_tolerance_a=reference_match_tolerance_a,
        )
    raise ValueError(
        "Solvent Shell Builder supports only PDB and XYZ input files."
    )


def build_solvent_shell_output(
    input_path: str | Path,
    reference_name: str,
    *,
    output_path: str | Path,
    director_atom_name: str,
    minimum_solvent_atom_separation_a: float,
    solute_distance_cutoffs_a: dict[str, float],
    coordinating_center_elements: Sequence[str] | None = None,
    target_average_coordination_numbers: dict[str, float] | None = None,
    reference_library_dir: str | Path | None = None,
    reference_match_tolerance_a: float = _DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
    analysis_result: SolventShellAnalysisResult | None = None,
) -> SolventShellBuildResult:
    resolved_input = Path(input_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    if analysis_result is None:
        analysis_result = analyze_solvent_shell(
            resolved_input,
            reference_name,
            reference_library_dir=reference_library_dir,
            reference_match_tolerance_a=reference_match_tolerance_a,
        )

    resolved_library_dir = (
        default_reference_library_dir()
        if reference_library_dir is None
        else Path(reference_library_dir).expanduser().resolve()
    )
    reference_entry = _resolve_reference_entry(
        reference_name,
        library_dir=resolved_library_dir,
    )
    reference_path = reference_entry.path.expanduser().resolve()
    reference_structure = PDBStructure.from_file(reference_path)
    if not reference_structure.atoms:
        raise ValueError(
            f"Reference molecule has no atoms: {reference_entry.name}"
        )
    director_atom_index = _resolve_reference_director_atom_index(
        reference_structure,
        director_atom_name=director_atom_name,
    )

    input_structure = _load_input_structure_as_pdb(
        resolved_input,
        structure_label=resolved_input.stem,
    )
    atoms_to_replace = {
        int(atom_id)
        for atom_id in analysis_result.partial_solvent_source_atom_ids
    }
    build_mode = "partial_solvent_completion"
    partial_anchor_positions = _partial_candidate_anchor_positions(
        input_structure=input_structure,
        analysis_result=analysis_result,
        reference_structure=reference_structure,
        director_atom_index=director_atom_index,
    )
    partial_candidate_count = len(partial_anchor_positions)
    if not partial_anchor_positions:
        if analysis_result.complete_solvent_molecule_count > 0:
            raise ValueError(
                "The analyzed structure already contains complete solvent "
                "molecules and does not expose partial solvent candidates to rebuild."
            )
        build_mode = "no_solvent_shell_build"

    remaining_atoms = [
        atom.copy()
        for atom in input_structure.atoms
        if int(atom.atom_id) not in atoms_to_replace
    ]
    selected_coordination_elements = tuple(
        sorted(
            {
                str(element)
                for element in (coordinating_center_elements or ())
                if str(element).strip()
            }
        )
    )
    target_coordination_by_element = {
        str(element): max(float(value), 0.0)
        for element, value in (
            target_average_coordination_numbers or {}
        ).items()
        if str(element).strip() and max(float(value), 0.0) > 0.0
    }
    placed_partial_atoms = _place_anchor_positions(
        reference_structure=reference_structure,
        director_atom_index=director_atom_index,
        anchor_positions=partial_anchor_positions,
        solute_atoms=remaining_atoms,
        occupied_atoms=remaining_atoms,
        starting_atom_id=len(remaining_atoms) + 1,
        starting_residue_number=_next_residue_number(remaining_atoms),
        minimum_atom_separation_a=max(
            float(minimum_solvent_atom_separation_a),
            0.0,
        ),
        require_clearance=False,
    )
    coordination_center_atoms = _coordination_center_atoms(
        input_structure=input_structure,
        analysis_result=analysis_result,
        coordinating_center_elements=selected_coordination_elements,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
    )
    anchor_positions_for_counting = [
        position.copy() for position in partial_anchor_positions
    ]
    additional_placed_atoms = _build_coordination_target_solvent_atoms(
        reference_structure=reference_structure,
        director_atom_index=director_atom_index,
        solute_atoms=remaining_atoms,
        occupied_atoms=remaining_atoms + placed_partial_atoms,
        center_atoms=coordination_center_atoms,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
        target_average_coordination_numbers=target_coordination_by_element,
        existing_anchor_positions=anchor_positions_for_counting,
        starting_atom_id=len(remaining_atoms) + len(placed_partial_atoms) + 1,
        starting_residue_number=(
            _next_residue_number(remaining_atoms + placed_partial_atoms)
        ),
        minimum_atom_separation_a=max(
            float(minimum_solvent_atom_separation_a),
            0.0,
        ),
    )
    if (
        not partial_anchor_positions
        and not additional_placed_atoms
        and not coordination_center_atoms
    ):
        raise ValueError(
            "No solvent anchor positions could be determined. Select at least "
            "one coordinating center element, provide its director-distance "
            "cutoff, and set a target average coordination number."
        )
    if (
        not partial_anchor_positions
        and not additional_placed_atoms
        and coordination_center_atoms
    ):
        raise ValueError(
            "No solvent molecules could be placed from the selected "
            "coordination targets while respecting the current cutoff and "
            "minimum-separation settings."
        )
    placed_atoms = placed_partial_atoms + additional_placed_atoms
    completed_atoms = remaining_atoms + placed_atoms
    for atom_id, atom in enumerate(completed_atoms, start=1):
        atom.atom_id = atom_id
    output_structure = PDBStructure(
        atoms=completed_atoms,
        source_name=resolved_output.stem,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    output_structure.write_pdb_file(resolved_output)
    placed_molecule_count = (
        int(len(placed_atoms) / max(len(reference_structure.atoms), 1))
        if reference_structure.atoms
        else 0
    )
    achieved_coordination_by_element = _average_coordination_by_element(
        center_atoms=coordination_center_atoms,
        anchor_positions=anchor_positions_for_counting,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
    )
    return SolventShellBuildResult(
        input_path=resolved_input,
        output_path=resolved_output,
        input_format=analysis_result.input_format,
        reference_name=reference_entry.name,
        reference_residue_name=reference_entry.residue_name,
        director_atom_name=str(
            reference_structure.atoms[director_atom_index].atom_name
        ),
        build_mode=build_mode,
        solvent_molecules_added=placed_molecule_count,
        solvent_atoms_added=max(len(placed_atoms) - len(atoms_to_replace), 0),
        partial_candidates_completed=partial_candidate_count,
        replaced_source_atom_count=len(atoms_to_replace),
        minimum_solvent_atom_separation_a=float(
            minimum_solvent_atom_separation_a
        ),
        solute_distance_cutoffs_a=dict(
            sorted(
                (
                    str(element),
                    float(distance),
                )
                for element, distance in solute_distance_cutoffs_a.items()
                if float(distance) > 0.0
            )
        ),
        coordinating_center_elements=selected_coordination_elements,
        target_average_coordination_numbers=dict(
            sorted(target_coordination_by_element.items())
        ),
        achieved_average_coordination_numbers=achieved_coordination_by_element,
    )


def reference_atom_choices(
    reference_name: str,
    *,
    reference_library_dir: str | Path | None = None,
) -> tuple[str, ...]:
    resolved_library_dir = (
        default_reference_library_dir()
        if reference_library_dir is None
        else Path(reference_library_dir).expanduser().resolve()
    )
    reference_entry = _resolve_reference_entry(
        reference_name,
        library_dir=resolved_library_dir,
    )
    structure = PDBStructure.from_file(
        reference_entry.path.expanduser().resolve()
    )
    return tuple(str(atom.atom_name) for atom in structure.atoms)


def default_director_atom_name(
    reference_name: str,
    *,
    reference_library_dir: str | Path | None = None,
) -> str | None:
    resolved_library_dir = (
        default_reference_library_dir()
        if reference_library_dir is None
        else Path(reference_library_dir).expanduser().resolve()
    )
    reference_entry = _resolve_reference_entry(
        reference_name,
        library_dir=resolved_library_dir,
    )
    structure = PDBStructure.from_file(
        reference_entry.path.expanduser().resolve()
    )
    if not structure.atoms:
        return None
    director_index = _default_director_atom_index(structure)
    if director_index is None:
        return None
    return str(structure.atoms[director_index].atom_name)


def _load_input_structure_as_pdb(
    input_path: Path,
    *,
    structure_label: str,
) -> PDBStructure:
    if input_path.suffix.lower() == ".pdb":
        source_structure = PDBStructure.from_file(input_path)
        copied_atoms = [atom.copy() for atom in source_structure.atoms]
        for index, atom in enumerate(copied_atoms, start=1):
            atom.atom_id = index
        return PDBStructure(
            atoms=copied_atoms,
            source_name=input_path.stem,
        )

    positions, elements = load_structure_file(input_path)
    residue_name = _normalized_residue_name(structure_label)
    counters: dict[str, int] = {}
    atoms: list[object] = []
    for index, (coordinates, element) in enumerate(
        zip(positions, elements, strict=True),
        start=1,
    ):
        counters[str(element)] = counters.get(str(element), 0) + 1
        atoms.append(
            PDBAtom(
                atom_id=index,
                atom_name=f"{element}{counters[str(element)]}",
                residue_name=residue_name,
                residue_number=1,
                coordinates=np.asarray(coordinates, dtype=float),
                element=str(element),
            )
        )
    return PDBStructure(atoms=atoms, source_name=input_path.stem)


def _resolve_reference_director_atom_index(
    reference_structure: PDBStructure,
    *,
    director_atom_name: str,
) -> int:
    normalized_name = _normalized_atom_name(
        director_atom_name,
        fallback="DIR1",
    )
    for index, atom in enumerate(reference_structure.atoms):
        if (
            _normalized_atom_name(
                str(atom.atom_name),
                fallback=f"{atom.element}{index + 1}",
            )
            == normalized_name
        ):
            return index
    raise ValueError(
        f"Director atom {director_atom_name!r} was not found in the selected solvent reference."
    )


def _default_director_atom_index(
    reference_structure: PDBStructure,
) -> int | None:
    if not reference_structure.atoms:
        return None
    oxygen_indices = [
        index
        for index, atom in enumerate(reference_structure.atoms)
        if str(atom.element).upper() == "O"
    ]
    if len(oxygen_indices) == 1:
        return oxygen_indices[0]
    available_elements = {
        str(atom.element).upper() for atom in reference_structure.atoms
    }
    return _select_partial_anchor_index(
        reference_structure.atoms,
        available_elements=available_elements,
    )


def _partial_candidate_anchor_positions(
    *,
    input_structure: PDBStructure,
    analysis_result: SolventShellAnalysisResult,
    reference_structure: PDBStructure,
    director_atom_index: int,
) -> list[np.ndarray]:
    if not analysis_result.residue_mismatch_summaries:
        return []
    atoms_by_id = {int(atom.atom_id): atom for atom in input_structure.atoms}
    director_atom = reference_structure.atoms[director_atom_index]
    director_name = _normalized_atom_name(
        str(director_atom.atom_name),
        fallback=f"{director_atom.element}{director_atom_index + 1}",
    )
    director_element = str(director_atom.element).upper()
    positions: list[np.ndarray] = []
    for summary in analysis_result.residue_mismatch_summaries:
        candidate_atoms = [
            atoms_by_id[int(atom_id)]
            for atom_id in summary.source_atom_ids
            if int(atom_id) in atoms_by_id
        ]
        if not candidate_atoms:
            continue
        director_matches = [
            atom
            for atom in candidate_atoms
            if _normalized_atom_name(
                str(atom.atom_name),
                fallback=f"{atom.element}{int(atom.atom_id)}",
            )
            == director_name
        ]
        if director_matches:
            positions.append(director_matches[0].coordinates.copy())
            continue
        element_matches = [
            atom
            for atom in candidate_atoms
            if str(atom.element).upper() == director_element
        ]
        if element_matches:
            positions.append(element_matches[0].coordinates.copy())
            continue
        positions.append(
            np.mean(
                np.asarray(
                    [atom.coordinates for atom in candidate_atoms],
                    dtype=float,
                ),
                axis=0,
            )
        )
    return positions


def _no_solvent_anchor_positions(
    *,
    input_structure: PDBStructure,
    analysis_result: SolventShellAnalysisResult,
    solute_distance_cutoffs_a: dict[str, float],
) -> list[np.ndarray]:
    cutoff_by_element = {
        str(element): max(float(distance), 0.0)
        for element, distance in solute_distance_cutoffs_a.items()
        if max(float(distance), 0.0) > 0.0
    }
    if not cutoff_by_element:
        return []
    solvent_like_ids = {
        int(atom_id)
        for atom_id in analysis_result.complete_solvent_source_atom_ids
    }.union(
        int(atom_id)
        for atom_id in analysis_result.partial_solvent_source_atom_ids
    )
    solute_atoms = [
        atom.copy()
        for atom in input_structure.atoms
        if int(atom.atom_id) not in solvent_like_ids
        and str(atom.element) in cutoff_by_element
    ]
    if not solute_atoms:
        return []
    cluster_center = _weighted_center(solute_atoms)
    positions: list[np.ndarray] = []
    for atom in solute_atoms:
        outward_vector = _normalize_vector(
            atom.coordinates - cluster_center,
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        cutoff_distance = cutoff_by_element.get(str(atom.element))
        if cutoff_distance is None or cutoff_distance <= 0.0:
            continue
        positions.append(
            np.asarray(atom.coordinates, dtype=float)
            + outward_vector * float(cutoff_distance)
        )
    return positions


def _place_anchor_positions(
    *,
    reference_structure: PDBStructure,
    director_atom_index: int,
    anchor_positions: Sequence[np.ndarray],
    solute_atoms: list[PDBAtom],
    occupied_atoms: list[PDBAtom],
    starting_atom_id: int,
    starting_residue_number: int,
    minimum_atom_separation_a: float,
    require_clearance: bool,
) -> list[PDBAtom]:
    placed_atoms: list[PDBAtom] = []
    next_atom_id = int(starting_atom_id)
    next_residue_number = int(starting_residue_number)
    current_occupied_atoms = [atom.copy() for atom in occupied_atoms]
    for anchor_position in anchor_positions:
        trial_atoms, clearance_met, _min_distance = (
            _trial_place_solvent_molecule(
                reference_structure=reference_structure,
                director_atom_index=director_atom_index,
                anchor_position=np.asarray(
                    anchor_position, dtype=float
                ).copy(),
                solute_atoms=solute_atoms,
                occupied_atoms=current_occupied_atoms,
                starting_atom_id=next_atom_id,
                residue_number=next_residue_number,
                minimum_atom_separation_a=minimum_atom_separation_a,
            )
        )
        if require_clearance and not clearance_met:
            continue
        placed_atoms.extend(trial_atoms)
        current_occupied_atoms.extend(atom.copy() for atom in trial_atoms)
        next_atom_id += len(trial_atoms)
        next_residue_number += 1
    return placed_atoms


def _trial_place_solvent_molecule(
    *,
    reference_structure: PDBStructure,
    director_atom_index: int,
    anchor_position: np.ndarray,
    solute_atoms: list[PDBAtom],
    occupied_atoms: list[PDBAtom],
    starting_atom_id: int,
    residue_number: int,
    minimum_atom_separation_a: float,
) -> tuple[list[PDBAtom], bool, float]:
    reference_atoms = [atom.copy() for atom in reference_structure.atoms]
    reference_anchor = reference_atoms[director_atom_index]
    reference_anchor_coord = reference_anchor.coordinates.copy()
    reference_offsets = np.asarray(
        [
            atom.coordinates - reference_anchor_coord
            for atom in reference_atoms
        ],
        dtype=float,
    )
    reference_body_atoms = [
        atom
        for index, atom in enumerate(reference_atoms)
        if index != director_atom_index
    ]
    reference_body_center = _weighted_center(reference_body_atoms)
    reference_body_vector = reference_body_center - reference_anchor_coord
    if np.linalg.norm(reference_body_vector) <= 1e-8 and reference_body_atoms:
        reference_body_vector = (
            reference_body_atoms[0].coordinates - reference_anchor_coord
        )
    if np.linalg.norm(reference_body_vector) <= 1e-8:
        reference_body_vector = np.array([1.0, 0.0, 0.0], dtype=float)

    solute_center = _weighted_center(
        solute_atoms if solute_atoms else occupied_atoms
    )
    outward_vector = _normalize_vector(
        np.asarray(anchor_position, dtype=float) - solute_center,
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    alignment = _rotation_between_vectors(
        reference_body_vector,
        outward_vector,
    )
    aligned_offsets = _rotate_points(reference_offsets, alignment)
    occupied_coords = [atom.coordinates.copy() for atom in occupied_atoms]
    candidate_coords, min_distance, clearance_met = _best_solvent_orientation(
        aligned_offsets,
        anchor_position=np.asarray(anchor_position, dtype=float),
        outward_vector=outward_vector,
        occupied_coords=occupied_coords,
        excluded_index=director_atom_index,
        minimum_atom_separation_a=minimum_atom_separation_a,
    )
    placed_atoms: list[PDBAtom] = []
    next_atom_id = int(starting_atom_id)
    for reference_atom, coordinate in zip(
        reference_atoms,
        np.asarray(candidate_coords, dtype=float),
        strict=True,
    ):
        atom = reference_atom.copy()
        atom.atom_id = next_atom_id
        atom.residue_number = int(residue_number)
        atom.coordinates = np.asarray(coordinate, dtype=float).copy()
        placed_atoms.append(atom)
        next_atom_id += 1
    return placed_atoms, bool(clearance_met), float(min_distance)


def _coordination_center_atoms(
    *,
    input_structure: PDBStructure,
    analysis_result: SolventShellAnalysisResult,
    coordinating_center_elements: Sequence[str],
    solute_distance_cutoffs_a: dict[str, float],
) -> list[PDBAtom]:
    selected_elements = {
        str(element)
        for element in coordinating_center_elements
        if str(element).strip()
        and float(solute_distance_cutoffs_a.get(str(element), 0.0)) > 0.0
    }
    if not selected_elements:
        return []
    solvent_like_ids = {
        int(atom_id)
        for atom_id in analysis_result.complete_solvent_source_atom_ids
    }.union(
        int(atom_id)
        for atom_id in analysis_result.partial_solvent_source_atom_ids
    )
    return [
        atom.copy()
        for atom in input_structure.atoms
        if int(atom.atom_id) not in solvent_like_ids
        and str(atom.element) in selected_elements
    ]


def _build_coordination_target_solvent_atoms(
    *,
    reference_structure: PDBStructure,
    director_atom_index: int,
    solute_atoms: list[PDBAtom],
    occupied_atoms: list[PDBAtom],
    center_atoms: list[PDBAtom],
    solute_distance_cutoffs_a: dict[str, float],
    target_average_coordination_numbers: dict[str, float],
    existing_anchor_positions: list[np.ndarray],
    starting_atom_id: int,
    starting_residue_number: int,
    minimum_atom_separation_a: float,
) -> list[PDBAtom]:
    if not center_atoms or not target_average_coordination_numbers:
        return []
    current_counts = _coordination_count_by_center_atom(
        center_atoms=center_atoms,
        anchor_positions=existing_anchor_positions,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
    )
    placed_atoms: list[PDBAtom] = []
    occupied_atom_copies = [atom.copy() for atom in occupied_atoms]
    next_atom_id = int(starting_atom_id)
    next_residue_number = int(starting_residue_number)
    used_candidate_keys: set[tuple[float, float, float]] = set()
    while _coordination_targets_unmet(
        center_atoms=center_atoms,
        current_counts=current_counts,
        target_average_coordination_numbers=target_average_coordination_numbers,
    ):
        candidate_positions = _coordination_candidate_positions(
            center_atoms=center_atoms,
            solute_atoms=solute_atoms,
            existing_anchor_positions=existing_anchor_positions,
            solute_distance_cutoffs_a=solute_distance_cutoffs_a,
        )
        if not candidate_positions:
            break
        best_choice: (
            tuple[
                float,
                int,
                list[PDBAtom],
                np.ndarray,
                tuple[int, ...],
                float,
            ]
            | None
        ) = None
        for candidate_position in candidate_positions:
            candidate_key = tuple(
                float(value)
                for value in np.round(candidate_position, decimals=4)
            )
            if candidate_key in used_candidate_keys:
                continue
            coordinated_center_ids = _coordinated_center_atom_ids(
                center_atoms=center_atoms,
                anchor_position=candidate_position,
                solute_distance_cutoffs_a=solute_distance_cutoffs_a,
            )
            if not coordinated_center_ids:
                continue
            benefit_score = _coordination_candidate_benefit(
                center_atoms=center_atoms,
                current_counts=current_counts,
                coordinated_center_ids=coordinated_center_ids,
                target_average_coordination_numbers=target_average_coordination_numbers,
            )
            if benefit_score <= 1e-8:
                continue
            (
                refined_position,
                trial_atoms,
                clearance_met,
                min_distance,
            ) = _refine_anchor_position(
                candidate_position=candidate_position,
                center_atoms=center_atoms,
                coordinated_center_ids=coordinated_center_ids,
                solute_atoms=solute_atoms,
                occupied_atoms=occupied_atom_copies,
                reference_structure=reference_structure,
                director_atom_index=director_atom_index,
                solute_distance_cutoffs_a=solute_distance_cutoffs_a,
                minimum_atom_separation_a=minimum_atom_separation_a,
            )
            refined_center_ids = _coordinated_center_atom_ids(
                center_atoms=center_atoms,
                anchor_position=refined_position,
                solute_distance_cutoffs_a=solute_distance_cutoffs_a,
            )
            if not refined_center_ids:
                continue
            benefit_score = _coordination_candidate_benefit(
                center_atoms=center_atoms,
                current_counts=current_counts,
                coordinated_center_ids=refined_center_ids,
                target_average_coordination_numbers=target_average_coordination_numbers,
            )
            if benefit_score <= 1e-8:
                continue
            if not clearance_met:
                continue
            candidate_rank = (
                benefit_score,
                len(refined_center_ids),
                min_distance,
            )
            if best_choice is None or candidate_rank > (
                best_choice[0],
                best_choice[1],
                best_choice[5],
            ):
                best_choice = (
                    benefit_score,
                    len(refined_center_ids),
                    trial_atoms,
                    refined_position.copy(),
                    refined_center_ids,
                    min_distance,
                )
        if best_choice is None:
            break
        (
            _benefit,
            _coordination_count,
            accepted_atoms,
            accepted_position,
            center_ids,
            _accepted_min_distance,
        ) = best_choice
        placed_atoms.extend(accepted_atoms)
        occupied_atom_copies.extend(atom.copy() for atom in accepted_atoms)
        existing_anchor_positions.append(accepted_position.copy())
        for center_id in center_ids:
            current_counts[int(center_id)] = (
                current_counts.get(int(center_id), 0) + 1
            )
        used_candidate_keys.add(
            tuple(float(value) for value in np.round(accepted_position, 4))
        )
        next_atom_id += len(accepted_atoms)
        next_residue_number += 1
    return placed_atoms


def _coordination_candidate_positions(
    *,
    center_atoms: list[PDBAtom],
    solute_atoms: list[PDBAtom],
    existing_anchor_positions: Sequence[np.ndarray],
    solute_distance_cutoffs_a: dict[str, float],
) -> list[np.ndarray]:
    if not center_atoms:
        return []
    cluster_center = _weighted_center(
        solute_atoms if solute_atoms else center_atoms
    )
    candidates: list[np.ndarray] = []
    seen_keys: set[tuple[float, float, float]] = set()
    for center_atom in center_atoms:
        cutoff_distance = max(
            float(
                solute_distance_cutoffs_a.get(str(center_atom.element), 0.0)
            ),
            0.0,
        )
        if cutoff_distance <= 0.0:
            continue
        for candidate_position in _octahedral_candidate_positions_for_center(
            center_atom=center_atom,
            cutoff_distance=cutoff_distance,
            solute_atoms=solute_atoms,
            existing_anchor_positions=existing_anchor_positions,
            cluster_center=cluster_center,
        ):
            candidate_key = tuple(
                float(value)
                for value in np.round(candidate_position, decimals=4)
            )
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            candidates.append(candidate_position)
        for direction in _single_center_directions(
            center_coordinates=center_atom.coordinates,
            cluster_center=cluster_center,
        ):
            candidate_position = (
                np.asarray(center_atom.coordinates, dtype=float)
                + direction * cutoff_distance
            )
            candidate_key = tuple(
                float(value)
                for value in np.round(candidate_position, decimals=4)
            )
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            candidates.append(candidate_position)
    for index, center_atom in enumerate(center_atoms):
        cutoff_a = max(
            float(
                solute_distance_cutoffs_a.get(str(center_atom.element), 0.0)
            ),
            0.0,
        )
        if cutoff_a <= 0.0:
            continue
        for other_atom in center_atoms[index + 1 :]:
            cutoff_b = max(
                float(
                    solute_distance_cutoffs_a.get(str(other_atom.element), 0.0)
                ),
                0.0,
            )
            if cutoff_b <= 0.0:
                continue
            for candidate_position in _pair_center_intersection_positions(
                center_a=center_atom.coordinates,
                cutoff_a=cutoff_a,
                center_b=other_atom.coordinates,
                cutoff_b=cutoff_b,
                cluster_center=cluster_center,
            ):
                candidate_key = tuple(
                    float(value)
                    for value in np.round(candidate_position, decimals=4)
                )
                if candidate_key in seen_keys:
                    continue
                seen_keys.add(candidate_key)
                candidates.append(candidate_position)
    return candidates


def _refine_anchor_position(
    *,
    candidate_position: np.ndarray,
    center_atoms: list[PDBAtom],
    coordinated_center_ids: Sequence[int],
    solute_atoms: list[PDBAtom],
    occupied_atoms: list[PDBAtom],
    reference_structure: PDBStructure,
    director_atom_index: int,
    solute_distance_cutoffs_a: dict[str, float],
    minimum_atom_separation_a: float,
) -> tuple[np.ndarray, list[PDBAtom], bool, float]:
    center_by_id = {int(atom.atom_id): atom for atom in center_atoms}
    coordinated_centers = [
        center_by_id[int(center_id)]
        for center_id in coordinated_center_ids
        if int(center_id) in center_by_id
    ]
    best_position = _project_anchor_position_to_coordination_shells(
        anchor_position=np.asarray(candidate_position, dtype=float).copy(),
        coordinated_centers=coordinated_centers,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
    )
    best_atoms, best_clearance, best_min_distance = (
        _trial_place_solvent_molecule(
            reference_structure=reference_structure,
            director_atom_index=director_atom_index,
            anchor_position=best_position,
            solute_atoms=solute_atoms,
            occupied_atoms=occupied_atoms,
            starting_atom_id=1,
            residue_number=1,
            minimum_atom_separation_a=minimum_atom_separation_a,
        )
    )
    best_score = _anchor_refinement_score(
        anchor_position=best_position,
        coordinated_centers=coordinated_centers,
        clearance_met=best_clearance,
        min_distance=best_min_distance,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
        minimum_atom_separation_a=minimum_atom_separation_a,
    )
    radial_direction = _normalize_vector(
        best_position - _weighted_center(coordinated_centers),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    tangent_a = _orthogonal_unit_vector(radial_direction)
    tangent_b = _normalize_vector(
        np.cross(radial_direction, tangent_a),
        fallback=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    search_directions = (
        radial_direction,
        -radial_direction,
        tangent_a,
        -tangent_a,
        tangent_b,
        -tangent_b,
        _normalize_vector(
            radial_direction + tangent_a,
            fallback=radial_direction,
        ),
        _normalize_vector(
            radial_direction - tangent_a,
            fallback=radial_direction,
        ),
        _normalize_vector(
            radial_direction + tangent_b,
            fallback=radial_direction,
        ),
        _normalize_vector(
            radial_direction - tangent_b,
            fallback=radial_direction,
        ),
    )
    for step_size in (0.35, 0.18, 0.08):
        improved = True
        while improved:
            improved = False
            for direction in search_directions:
                trial_position = (
                    _project_anchor_position_to_coordination_shells(
                        anchor_position=best_position
                        + np.asarray(direction, dtype=float)
                        * float(step_size),
                        coordinated_centers=coordinated_centers,
                        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
                    )
                )
                (
                    trial_atoms,
                    trial_clearance,
                    trial_min_distance,
                ) = _trial_place_solvent_molecule(
                    reference_structure=reference_structure,
                    director_atom_index=director_atom_index,
                    anchor_position=trial_position,
                    solute_atoms=solute_atoms,
                    occupied_atoms=occupied_atoms,
                    starting_atom_id=1,
                    residue_number=1,
                    minimum_atom_separation_a=minimum_atom_separation_a,
                )
                trial_score = _anchor_refinement_score(
                    anchor_position=trial_position,
                    coordinated_centers=coordinated_centers,
                    clearance_met=trial_clearance,
                    min_distance=trial_min_distance,
                    solute_distance_cutoffs_a=solute_distance_cutoffs_a,
                    minimum_atom_separation_a=minimum_atom_separation_a,
                )
                if trial_score + 1e-6 >= best_score:
                    continue
                best_position = trial_position
                best_atoms = trial_atoms
                best_clearance = trial_clearance
                best_min_distance = trial_min_distance
                best_score = trial_score
                improved = True
                break
    return best_position, best_atoms, best_clearance, best_min_distance


def _project_anchor_position_to_coordination_shells(
    *,
    anchor_position: np.ndarray,
    coordinated_centers: Sequence[PDBAtom],
    solute_distance_cutoffs_a: dict[str, float],
    iterations: int = 8,
) -> np.ndarray:
    projected = np.asarray(anchor_position, dtype=float).copy()
    if not coordinated_centers:
        return projected
    for _ in range(max(int(iterations), 1)):
        correction = np.zeros(3, dtype=float)
        contributing_centers = 0
        for center_atom in coordinated_centers:
            cutoff_distance = max(
                float(
                    solute_distance_cutoffs_a.get(
                        str(center_atom.element), 0.0
                    )
                ),
                0.0,
            )
            if cutoff_distance <= 0.0:
                continue
            displacement = projected - np.asarray(
                center_atom.coordinates, dtype=float
            )
            observed_distance = float(np.linalg.norm(displacement))
            if observed_distance <= 1e-8:
                displacement = np.array([1.0, 0.0, 0.0], dtype=float)
                observed_distance = 1.0
            correction += (
                (cutoff_distance - observed_distance)
                * displacement
                / observed_distance
            )
            contributing_centers += 1
        if contributing_centers == 0:
            break
        projected += correction / float(contributing_centers)
        if float(np.linalg.norm(correction)) <= 1e-6:
            break
    return projected


def _anchor_refinement_score(
    *,
    anchor_position: np.ndarray,
    coordinated_centers: Sequence[PDBAtom],
    clearance_met: bool,
    min_distance: float,
    solute_distance_cutoffs_a: dict[str, float],
    minimum_atom_separation_a: float,
) -> float:
    anchor = np.asarray(anchor_position, dtype=float)
    score = 0.0
    for center_atom in coordinated_centers:
        cutoff_distance = max(
            float(
                solute_distance_cutoffs_a.get(str(center_atom.element), 0.0)
            ),
            0.0,
        )
        if cutoff_distance <= 0.0:
            continue
        observed_distance = float(
            np.linalg.norm(
                anchor - np.asarray(center_atom.coordinates, dtype=float)
            )
        )
        normalized_error = (observed_distance - cutoff_distance) / max(
            cutoff_distance,
            1e-6,
        )
        score += normalized_error * normalized_error
        if observed_distance > cutoff_distance + 0.2:
            score += 2.0 * (observed_distance - cutoff_distance)
    clearance_gap = max(
        float(minimum_atom_separation_a) - float(min_distance), 0.0
    )
    score += 20.0 * clearance_gap * clearance_gap
    if not clearance_met:
        score += 5.0
    score -= 0.05 * max(float(min_distance), 0.0)
    return score


def _octahedral_candidate_positions_for_center(
    *,
    center_atom: PDBAtom,
    cutoff_distance: float,
    solute_atoms: list[PDBAtom],
    existing_anchor_positions: Sequence[np.ndarray],
    cluster_center: np.ndarray,
) -> tuple[np.ndarray, ...]:
    neighbor_vectors = _existing_first_shell_neighbor_vectors(
        center_atom=center_atom,
        solute_atoms=solute_atoms,
        existing_anchor_positions=existing_anchor_positions,
        cutoff_distance=cutoff_distance,
    )
    octahedral_directions = _octahedral_direction_frame(
        existing_neighbor_vectors=neighbor_vectors,
        preferred_direction=_normalize_vector(
            np.asarray(center_atom.coordinates, dtype=float)
            - np.asarray(cluster_center, dtype=float),
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        ),
    )
    occupied_indices = _occupied_octahedral_direction_indices(
        existing_neighbor_vectors=neighbor_vectors,
        octahedral_directions=octahedral_directions,
    )
    return tuple(
        np.asarray(center_atom.coordinates, dtype=float)
        + np.asarray(direction, dtype=float) * float(cutoff_distance)
        for index, direction in enumerate(octahedral_directions)
        if index not in occupied_indices
    )


def _existing_first_shell_neighbor_vectors(
    *,
    center_atom: PDBAtom,
    solute_atoms: list[PDBAtom],
    existing_anchor_positions: Sequence[np.ndarray],
    cutoff_distance: float,
) -> tuple[np.ndarray, ...]:
    coordination_radius = float(cutoff_distance) + 0.35
    center_coordinates = np.asarray(center_atom.coordinates, dtype=float)
    neighbor_vectors: list[np.ndarray] = []
    for atom in solute_atoms:
        if int(atom.atom_id) == int(center_atom.atom_id):
            continue
        displacement = (
            np.asarray(atom.coordinates, dtype=float) - center_coordinates
        )
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-8 or distance > coordination_radius:
            continue
        neighbor_vectors.append(
            _normalize_vector(
                displacement,
                fallback=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        )
    for anchor_position in existing_anchor_positions:
        displacement = (
            np.asarray(anchor_position, dtype=float) - center_coordinates
        )
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-8 or distance > coordination_radius:
            continue
        neighbor_vectors.append(
            _normalize_vector(
                displacement,
                fallback=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        )
    unique_vectors: list[np.ndarray] = []
    for vector in neighbor_vectors:
        if any(
            float(np.dot(vector, other)) > 0.95 for other in unique_vectors
        ):
            continue
        unique_vectors.append(vector)
    return tuple(unique_vectors)


def _octahedral_direction_frame(
    *,
    existing_neighbor_vectors: Sequence[np.ndarray],
    preferred_direction: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if existing_neighbor_vectors:
        primary_axis = _normalize_vector(
            np.asarray(existing_neighbor_vectors[0], dtype=float),
            fallback=preferred_direction,
        )
    else:
        primary_axis = _normalize_vector(
            preferred_direction,
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
    secondary_seed: np.ndarray | None = None
    for vector in existing_neighbor_vectors[1:]:
        projected = np.asarray(vector, dtype=float) - primary_axis * float(
            np.dot(np.asarray(vector, dtype=float), primary_axis)
        )
        if float(np.linalg.norm(projected)) > 0.25:
            secondary_seed = projected
            break
    if secondary_seed is None:
        preferred_projected = np.asarray(
            preferred_direction, dtype=float
        ) - primary_axis * float(
            np.dot(np.asarray(preferred_direction, dtype=float), primary_axis)
        )
        if float(np.linalg.norm(preferred_projected)) > 0.25:
            secondary_seed = preferred_projected
    if secondary_seed is None:
        secondary_axis = _orthogonal_unit_vector(primary_axis)
    else:
        secondary_axis = _normalize_vector(
            secondary_seed,
            fallback=_orthogonal_unit_vector(primary_axis),
        )
    tertiary_axis = _normalize_vector(
        np.cross(primary_axis, secondary_axis),
        fallback=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    secondary_axis = _normalize_vector(
        np.cross(tertiary_axis, primary_axis),
        fallback=secondary_axis,
    )
    return (
        primary_axis,
        -primary_axis,
        secondary_axis,
        -secondary_axis,
        tertiary_axis,
        -tertiary_axis,
    )


def _occupied_octahedral_direction_indices(
    *,
    existing_neighbor_vectors: Sequence[np.ndarray],
    octahedral_directions: Sequence[np.ndarray],
) -> set[int]:
    occupied_indices: set[int] = set()
    for vector in existing_neighbor_vectors:
        normalized_vector = _normalize_vector(
            np.asarray(vector, dtype=float),
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        best_index = max(
            range(len(octahedral_directions)),
            key=lambda index: float(
                np.dot(
                    normalized_vector,
                    np.asarray(octahedral_directions[index], dtype=float),
                )
            ),
        )
        if (
            float(
                np.dot(
                    normalized_vector,
                    np.asarray(octahedral_directions[best_index], dtype=float),
                )
            )
            < 0.55
        ):
            continue
        occupied_indices.add(int(best_index))
    return occupied_indices


def _single_center_directions(
    *,
    center_coordinates: np.ndarray,
    cluster_center: np.ndarray,
) -> tuple[np.ndarray, ...]:
    preferred = _normalize_vector(
        np.asarray(center_coordinates, dtype=float)
        - np.asarray(cluster_center, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    basis1 = _orthogonal_unit_vector(preferred)
    basis2 = _normalize_vector(
        np.cross(preferred, basis1),
        fallback=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    direction_vectors = [
        preferred,
        preferred + 0.6 * basis1,
        preferred - 0.6 * basis1,
        preferred + 0.6 * basis2,
        preferred - 0.6 * basis2,
        preferred + 0.45 * basis1 + 0.45 * basis2,
        preferred - 0.45 * basis1 + 0.45 * basis2,
        preferred + 0.45 * basis1 - 0.45 * basis2,
        preferred - 0.45 * basis1 - 0.45 * basis2,
    ]
    return tuple(
        _normalize_vector(
            np.asarray(direction, dtype=float),
            fallback=preferred,
        )
        for direction in direction_vectors
    )


def _pair_center_intersection_positions(
    *,
    center_a: np.ndarray,
    cutoff_a: float,
    center_b: np.ndarray,
    cutoff_b: float,
    cluster_center: np.ndarray,
) -> tuple[np.ndarray, ...]:
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    axis_vector = center_b - center_a
    axis_distance = float(np.linalg.norm(axis_vector))
    if axis_distance <= 1e-8:
        return ()
    if axis_distance > float(cutoff_a + cutoff_b) + 1e-6:
        return ()
    if axis_distance < abs(float(cutoff_a - cutoff_b)) - 1e-6:
        return ()
    axis_unit = axis_vector / axis_distance
    offset_along_axis = (
        axis_distance * axis_distance
        - float(cutoff_b) * float(cutoff_b)
        + float(cutoff_a) * float(cutoff_a)
    ) / (2.0 * axis_distance)
    circle_center = center_a + axis_unit * offset_along_axis
    circle_radius_sq = float(cutoff_a) * float(cutoff_a) - (
        offset_along_axis * offset_along_axis
    )
    if circle_radius_sq < -1e-6:
        return ()
    circle_radius = float(np.sqrt(max(circle_radius_sq, 0.0)))
    if circle_radius <= 1e-8:
        return (circle_center,)
    outward_hint = np.asarray(circle_center, dtype=float) - np.asarray(
        cluster_center,
        dtype=float,
    )
    plane_projection = outward_hint - axis_unit * float(
        np.dot(outward_hint, axis_unit)
    )
    basis1 = _normalize_vector(
        plane_projection,
        fallback=_orthogonal_unit_vector(axis_unit),
    )
    basis2 = _normalize_vector(
        np.cross(axis_unit, basis1),
        fallback=np.array([0.0, 1.0, 0.0], dtype=float),
    )
    return (
        circle_center + circle_radius * basis1,
        circle_center - circle_radius * basis1,
        circle_center + circle_radius * basis2,
        circle_center - circle_radius * basis2,
    )


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    normalized = _normalize_vector(
        np.asarray(vector, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    if abs(float(normalized[0])) < 0.9:
        return _normalize_vector(
            np.cross(normalized, np.array([1.0, 0.0, 0.0], dtype=float)),
            fallback=np.array([0.0, 1.0, 0.0], dtype=float),
        )
    return _normalize_vector(
        np.cross(normalized, np.array([0.0, 1.0, 0.0], dtype=float)),
        fallback=np.array([0.0, 0.0, 1.0], dtype=float),
    )


def _coordination_count_by_center_atom(
    *,
    center_atoms: list[PDBAtom],
    anchor_positions: Sequence[np.ndarray],
    solute_distance_cutoffs_a: dict[str, float],
) -> dict[int, int]:
    counts = {int(atom.atom_id): 0 for atom in center_atoms}
    if not center_atoms or not anchor_positions:
        return counts
    center_by_id = {int(atom.atom_id): atom for atom in center_atoms}
    for anchor_position in anchor_positions:
        anchor = np.asarray(anchor_position, dtype=float)
        for atom_id, atom in center_by_id.items():
            cutoff_distance = max(
                float(solute_distance_cutoffs_a.get(str(atom.element), 0.0)),
                0.0,
            )
            if cutoff_distance <= 0.0:
                continue
            if (
                float(
                    np.linalg.norm(
                        anchor - np.asarray(atom.coordinates, dtype=float)
                    )
                )
                <= cutoff_distance + 1e-6
            ):
                counts[int(atom_id)] = counts.get(int(atom_id), 0) + 1
    return counts


def _coordination_targets_unmet(
    *,
    center_atoms: list[PDBAtom],
    current_counts: dict[int, int],
    target_average_coordination_numbers: dict[str, float],
) -> bool:
    achieved = _average_coordination_by_element_from_counts(
        center_atoms=center_atoms,
        current_counts=current_counts,
    )
    for element, target_value in target_average_coordination_numbers.items():
        if achieved.get(str(element), 0.0) + 1e-6 < float(target_value):
            return True
    return False


def _coordinated_center_atom_ids(
    *,
    center_atoms: list[PDBAtom],
    anchor_position: np.ndarray,
    solute_distance_cutoffs_a: dict[str, float],
) -> tuple[int, ...]:
    coordinated_ids: list[int] = []
    anchor = np.asarray(anchor_position, dtype=float)
    for atom in center_atoms:
        cutoff_distance = max(
            float(solute_distance_cutoffs_a.get(str(atom.element), 0.0)),
            0.0,
        )
        if cutoff_distance <= 0.0:
            continue
        if (
            float(
                np.linalg.norm(
                    anchor - np.asarray(atom.coordinates, dtype=float)
                )
            )
            <= cutoff_distance + 1e-6
        ):
            coordinated_ids.append(int(atom.atom_id))
    return tuple(sorted(coordinated_ids))


def _coordination_candidate_benefit(
    *,
    center_atoms: list[PDBAtom],
    current_counts: dict[int, int],
    coordinated_center_ids: Sequence[int],
    target_average_coordination_numbers: dict[str, float],
) -> float:
    center_by_id = {int(atom.atom_id): atom for atom in center_atoms}
    benefit = 0.0
    for center_id in coordinated_center_ids:
        atom = center_by_id.get(int(center_id))
        if atom is None:
            continue
        target_value = float(
            target_average_coordination_numbers.get(str(atom.element), 0.0)
        )
        current_value = float(current_counts.get(int(center_id), 0))
        benefit += max(target_value - current_value, 0.0)
    return benefit


def _average_coordination_by_element(
    *,
    center_atoms: list[PDBAtom],
    anchor_positions: Sequence[np.ndarray],
    solute_distance_cutoffs_a: dict[str, float],
) -> dict[str, float]:
    current_counts = _coordination_count_by_center_atom(
        center_atoms=center_atoms,
        anchor_positions=anchor_positions,
        solute_distance_cutoffs_a=solute_distance_cutoffs_a,
    )
    return _average_coordination_by_element_from_counts(
        center_atoms=center_atoms,
        current_counts=current_counts,
    )


def _average_coordination_by_element_from_counts(
    *,
    center_atoms: list[PDBAtom],
    current_counts: dict[int, int],
) -> dict[str, float]:
    centers_by_element: dict[str, list[int]] = defaultdict(list)
    for atom in center_atoms:
        centers_by_element[str(atom.element)].append(int(atom.atom_id))
    averages: dict[str, float] = {}
    for element, atom_ids in sorted(centers_by_element.items()):
        if not atom_ids:
            continue
        averages[str(element)] = float(
            sum(current_counts.get(int(atom_id), 0) for atom_id in atom_ids)
        ) / float(len(atom_ids))
    return averages


def _director_anchor_positions_from_atoms(
    *,
    placed_atoms: list[PDBAtom],
    reference_structure: PDBStructure,
    director_atom_index: int,
) -> list[np.ndarray]:
    if not placed_atoms:
        return []
    director_atom_name = str(
        reference_structure.atoms[director_atom_index].atom_name
    )
    director_name = _normalized_atom_name(
        director_atom_name,
        fallback=f"{reference_structure.atoms[director_atom_index].element}{director_atom_index + 1}",
    )
    positions: list[np.ndarray] = []
    residue_groups: dict[tuple[str, int], list[PDBAtom]] = defaultdict(list)
    for atom in placed_atoms:
        residue_groups[
            (str(atom.residue_name), int(atom.residue_number))
        ].append(atom)
    for residue_atoms in residue_groups.values():
        director_atom = next(
            (
                atom
                for atom in residue_atoms
                if _normalized_atom_name(
                    str(atom.atom_name),
                    fallback=f"{atom.element}{int(atom.atom_id)}",
                )
                == director_name
            ),
            None,
        )
        if director_atom is None:
            continue
        positions.append(
            np.asarray(director_atom.coordinates, dtype=float).copy()
        )
    return positions


def _analyze_pdb_input(
    input_path: Path,
    *,
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_library_dir: Path,
    reference_match_tolerance_a: float,
) -> SolventShellAnalysisResult:
    structure = PDBStructure.from_file(input_path)
    total_atoms = len(structure.atoms)
    reference_atom_count = len(reference_atoms)
    if reference_atom_count == 1:
        return _analyze_single_atom_pdb_input(
            input_path,
            structure=structure,
            reference_entry=reference_entry,
            reference_atoms=reference_atoms,
            reference_match_tolerance_a=reference_match_tolerance_a,
        )

    workflow = XYZToPDBWorkflow(
        input_path,
        reference_library_dir=reference_library_dir,
    )
    configuration = _single_reference_configuration(
        workflow,
        reference_entry=reference_entry,
        reference_atoms=reference_atoms,
        reference_match_tolerance_a=reference_match_tolerance_a,
    )
    molecule_name = configuration.molecules[0].name
    residue_groups: dict[tuple[str, int], list[object]] = defaultdict(list)
    for atom in structure.atoms:
        residue_groups[(atom.residue_name, atom.residue_number)].append(atom)

    residue_numbers_by_name: dict[str, list[int]] = defaultdict(list)
    molecule_counts_by_name: Counter[str] = Counter()
    residue_mismatches: list[SolventShellResidueMismatchSummary] = []
    complete_solvent_atom_ids: set[int] = set()
    matched_groups: list[tuple[str, int, list[object]]] = []
    matched_atom_count = 0
    matched_molecule_count = 0
    for (residue_name, residue_number), residue_atoms in sorted(
        residue_groups.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        frame = _matching_frame_from_pdb_atoms(
            input_path,
            residue_atoms=residue_atoms,
        )
        converted_residues = workflow._convert_first_frame(
            frame,
            configuration,
        )
        matched_molecules = [
            residue
            for residue in converted_residues
            if residue.molecule_name == molecule_name
        ]
        unmatched_atom_total = sum(
            len(residue.atoms)
            for residue in converted_residues
            if residue.molecule_name != molecule_name
        )
        matched_atom_total = sum(
            len(residue.atoms) for residue in matched_molecules
        )
        if (
            not matched_molecules
            or unmatched_atom_total > 0
            or matched_atom_total != len(residue_atoms)
        ):
            mismatch_summary = _build_pdb_residue_mismatch_summary(
                residue_name=residue_name,
                residue_number=int(residue_number),
                residue_atoms=residue_atoms,
                reference_entry=reference_entry,
                reference_atoms=reference_atoms,
                reference_match_tolerance_a=reference_match_tolerance_a,
            )
            if mismatch_summary is not None:
                residue_mismatches.append(mismatch_summary)
            continue
        matched_here = len(matched_molecules)
        residue_numbers_by_name[residue_name].append(residue_number)
        molecule_counts_by_name[residue_name] += matched_here
        complete_solvent_atom_ids.update(
            int(atom.atom_id) for atom in residue_atoms
        )
        matched_groups.append(
            (str(residue_name), int(residue_number), list(residue_atoms))
        )
        matched_molecule_count += matched_here
        matched_atom_count += matched_atom_total

    partial_solvent_atom_ids = {
        atom_id
        for summary in residue_mismatches
        for atom_id in summary.source_atom_ids
    }
    solute_element_counts = dict(
        sorted(
            Counter(
                atom.element
                for atom in structure.atoms
                if int(atom.atom_id) not in complete_solvent_atom_ids
                and int(atom.atom_id) not in partial_solvent_atom_ids
            ).items()
        )
    )

    reference_element_counts = dict(
        sorted(Counter(atom.element for atom in reference_atoms).items())
    )
    residue_summaries = tuple(
        SolventShellResidueSummary(
            residue_name=residue_name,
            molecule_count=int(molecule_counts_by_name[residue_name]),
            residue_numbers=tuple(
                sorted(residue_numbers_by_name[residue_name])
            ),
            atom_count=reference_atom_count,
            element_counts=reference_element_counts,
        )
        for residue_name in sorted(residue_numbers_by_name, key=str.casefold)
    )
    return SolventShellAnalysisResult(
        input_path=input_path,
        input_format="pdb",
        reference_name=reference_entry.name,
        reference_path=reference_entry.path.expanduser().resolve(),
        reference_residue_name=reference_entry.residue_name,
        reference_atom_count=reference_atom_count,
        detected_solvent_molecules=matched_molecule_count,
        matched_atom_count=matched_atom_count,
        unmatched_atom_count=max(total_atoms - matched_atom_count, 0),
        total_atoms=total_atoms,
        match_tolerance_a=reference_match_tolerance_a,
        solute_element_counts=solute_element_counts,
        complete_solvent_source_atom_ids=tuple(
            sorted(complete_solvent_atom_ids)
        ),
        complete_solvent_source_atom_groups=tuple(
            tuple(sorted(int(atom.atom_id) for atom in residue_atoms))
            for _residue_name, _residue_number, residue_atoms in matched_groups
        ),
        partial_solvent_source_atom_ids=tuple(
            sorted(partial_solvent_atom_ids)
        ),
        matched_residue_summaries=residue_summaries,
        residue_mismatch_summaries=tuple(residue_mismatches),
        notes=_build_pdb_analysis_notes(
            residue_mismatch_count=len(residue_mismatches)
        ),
    )


def _analyze_single_atom_pdb_input(
    input_path: Path,
    *,
    structure: PDBStructure,
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_match_tolerance_a: float,
) -> SolventShellAnalysisResult:
    reference_atom = reference_atoms[0]
    residue_numbers_by_name: dict[str, list[int]] = defaultdict(list)
    molecule_counts_by_name: Counter[str] = Counter()
    for atom in structure.atoms:
        if atom.element != reference_atom.element:
            continue
        residue_numbers_by_name[atom.residue_name].append(atom.residue_number)
        molecule_counts_by_name[atom.residue_name] += 1
    residue_summaries = tuple(
        SolventShellResidueSummary(
            residue_name=residue_name,
            molecule_count=int(molecule_counts_by_name[residue_name]),
            residue_numbers=tuple(
                sorted(residue_numbers_by_name[residue_name])
            ),
            atom_count=1,
            element_counts={reference_atom.element: 1},
        )
        for residue_name in sorted(residue_numbers_by_name, key=str.casefold)
    )
    matched_atom_count = sum(
        summary.molecule_count for summary in residue_summaries
    )
    matched_atom_ids = {
        int(atom.atom_id)
        for atom in structure.atoms
        if atom.element == reference_atom.element
    }
    solute_element_counts = dict(
        sorted(
            Counter(
                atom.element
                for atom in structure.atoms
                if int(atom.atom_id) not in matched_atom_ids
            ).items()
        )
    )
    return SolventShellAnalysisResult(
        input_path=input_path,
        input_format="pdb",
        reference_name=reference_entry.name,
        reference_path=reference_entry.path.expanduser().resolve(),
        reference_residue_name=reference_entry.residue_name,
        reference_atom_count=1,
        detected_solvent_molecules=matched_atom_count,
        matched_atom_count=matched_atom_count,
        unmatched_atom_count=max(len(structure.atoms) - matched_atom_count, 0),
        total_atoms=len(structure.atoms),
        match_tolerance_a=reference_match_tolerance_a,
        solute_element_counts=solute_element_counts,
        complete_solvent_source_atom_ids=tuple(sorted(matched_atom_ids)),
        complete_solvent_source_atom_groups=tuple(
            (int(atom.atom_id),)
            for atom in structure.atoms
            if atom.element == reference_atom.element
        ),
        matched_residue_summaries=residue_summaries,
        notes=(
            "Single-atom references are matched by element within each PDB "
            "residue.",
        ),
    )


def _analyze_xyz_input(
    input_path: Path,
    *,
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_library_dir: Path,
    reference_match_tolerance_a: float,
) -> SolventShellAnalysisResult:
    workflow = XYZToPDBWorkflow(
        input_path,
        reference_library_dir=reference_library_dir,
    )
    if len(reference_atoms) == 1:
        return _analyze_single_atom_xyz_input(
            input_path,
            workflow=workflow,
            reference_entry=reference_entry,
            reference_atoms=reference_atoms,
            reference_match_tolerance_a=reference_match_tolerance_a,
        )

    frame = workflow.read_xyz_frame(input_path)
    configuration = _single_reference_configuration(
        workflow,
        reference_entry=reference_entry,
        reference_atoms=reference_atoms,
        reference_match_tolerance_a=reference_match_tolerance_a,
    )
    converted_residues = workflow._convert_first_frame(frame, configuration)
    molecule_name = configuration.molecules[0].name
    matched_residues = [
        residue
        for residue in converted_residues
        if residue.molecule_name == molecule_name
    ]
    matched_source_indices = {
        int(source_index)
        for residue in matched_residues
        for source_index in residue.source_atom_indices
    }
    matched_source_atom_ids = {
        int(source_index) + 1 for source_index in matched_source_indices
    }
    unmatched_atom_records = [
        (index, frame.atoms[index])
        for index in range(len(frame.atoms))
        if index not in matched_source_indices
    ]
    partial_candidates = _build_xyz_partial_candidate_summaries(
        unmatched_atom_records=unmatched_atom_records,
        reference_entry=reference_entry,
        reference_atoms=reference_atoms,
        reference_match_tolerance_a=reference_match_tolerance_a,
    )
    partial_candidate_atom_ids = {
        atom_id
        for summary in partial_candidates
        for atom_id in summary.source_atom_ids
    }
    matched_atom_count = sum(
        len(residue.atoms) for residue in matched_residues
    )
    total_atoms = len(frame.atoms)
    solute_element_counts = dict(
        sorted(
            Counter(
                atom.element
                for atom in frame.atoms
                if int(atom.atom_id) not in matched_source_atom_ids
                and int(atom.atom_id) not in partial_candidate_atom_ids
            ).items()
        )
    )
    return SolventShellAnalysisResult(
        input_path=input_path,
        input_format="xyz",
        reference_name=reference_entry.name,
        reference_path=reference_entry.path.expanduser().resolve(),
        reference_residue_name=reference_entry.residue_name,
        reference_atom_count=len(reference_atoms),
        detected_solvent_molecules=len(matched_residues),
        matched_atom_count=matched_atom_count,
        unmatched_atom_count=max(total_atoms - matched_atom_count, 0),
        total_atoms=total_atoms,
        match_tolerance_a=reference_match_tolerance_a,
        solute_element_counts=solute_element_counts,
        complete_solvent_source_atom_ids=tuple(
            sorted(matched_source_atom_ids)
        ),
        complete_solvent_source_atom_groups=tuple(
            tuple(
                sorted(
                    int(source_index) + 1
                    for source_index in residue.source_atom_indices
                )
            )
            for residue in matched_residues
        ),
        partial_solvent_source_atom_ids=tuple(
            sorted(partial_candidate_atom_ids)
        ),
        residue_mismatch_summaries=partial_candidates,
        notes=(
            _build_xyz_analysis_note(
                partial_candidate_count=len(partial_candidates)
            ),
        ),
    )


def _analyze_single_atom_xyz_input(
    input_path: Path,
    *,
    workflow: XYZToPDBWorkflow,
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_match_tolerance_a: float,
) -> SolventShellAnalysisResult:
    frame = workflow.read_xyz_frame(input_path)
    reference_atom = reference_atoms[0]
    matched_atom_count = sum(
        1 for atom in frame.atoms if atom.element == reference_atom.element
    )
    total_atoms = len(frame.atoms)
    solute_element_counts = dict(
        sorted(
            Counter(
                atom.element
                for atom in frame.atoms
                if atom.element != reference_atom.element
            ).items()
        )
    )
    return SolventShellAnalysisResult(
        input_path=input_path,
        input_format="xyz",
        reference_name=reference_entry.name,
        reference_path=reference_entry.path.expanduser().resolve(),
        reference_residue_name=reference_entry.residue_name,
        reference_atom_count=1,
        detected_solvent_molecules=matched_atom_count,
        matched_atom_count=matched_atom_count,
        unmatched_atom_count=max(total_atoms - matched_atom_count, 0),
        total_atoms=total_atoms,
        match_tolerance_a=reference_match_tolerance_a,
        solute_element_counts=solute_element_counts,
        complete_solvent_source_atom_ids=tuple(
            int(atom.atom_id)
            for atom in frame.atoms
            if atom.element == reference_atom.element
        ),
        complete_solvent_source_atom_groups=tuple(
            (int(atom.atom_id),)
            for atom in frame.atoms
            if atom.element == reference_atom.element
        ),
        notes=(
            "Single-atom references are matched by element only for XYZ "
            "inputs.",
        ),
    )


def _single_reference_configuration(
    workflow: XYZToPDBWorkflow,
    *,
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_match_tolerance_a: float,
) -> XYZToPDBConfiguration:
    backbone_pairs = (
        reference_entry.backbone_pairs
        or _fallback_backbone_pairs(reference_atoms)
    )
    if not backbone_pairs:
        raise ValueError(
            "The selected solvent reference does not provide enough atoms "
            "to define a matching backbone pair."
        )
    anchors = tuple(
        AnchorPairDefinition(
            atom1_name=atom1_name,
            atom2_name=atom2_name,
            tolerance=reference_match_tolerance_a,
        )
        for atom1_name, atom2_name in backbone_pairs
    )
    resolved_anchor_indices = tuple(
        workflow._resolve_anchor_pair_indices(
            reference_atoms,
            anchor.atom1_name,
            anchor.atom2_name,
            molecule_name=reference_entry.name,
        )
        + (anchor.tolerance,)
        for anchor in anchors
    )
    molecule = MoleculeDefinition(
        name=reference_entry.name,
        reference_name=reference_entry.name,
        reference_path=reference_entry.path.expanduser().resolve(),
        residue_name=reference_entry.residue_name,
        reference_atoms=reference_atoms,
        anchors=anchors,
        resolved_anchor_indices=resolved_anchor_indices,
        preferred_anchor_indices=tuple(
            (index1, index2)
            for index1, index2, _tolerance in resolved_anchor_indices
        ),
        max_assignment_distance=None,
    )
    return XYZToPDBConfiguration(
        molecules=(molecule,),
        free_atoms={},
        exclude_hydrogen=False,
        pbc_params={},
    )


def _resolve_reference_entry(
    reference_name: str,
    *,
    library_dir: Path,
) -> ReferenceLibraryEntry:
    presets = list_reference_library(library_dir)
    lowered_name = str(reference_name).strip().casefold()
    for preset in presets:
        if preset.name.casefold() == lowered_name:
            return preset
    reference_path = Path(str(reference_name)).expanduser()
    if reference_path.is_file():
        structure = PDBStructure.from_file(reference_path)
        residue_name = (
            structure.atoms[0].residue_name if structure.atoms else "UNK"
        )
        return ReferenceLibraryEntry(
            name=reference_path.stem,
            path=reference_path.resolve(),
            residue_name=residue_name,
            atom_count=len(structure.atoms),
            atom_names=tuple(atom.atom_name for atom in structure.atoms),
            backbone_pairs=(),
        )
    raise ValueError(
        f"Reference molecule {reference_name!r} was not found in {library_dir}."
    )


def _fallback_backbone_pairs(
    reference_atoms: tuple[object, ...],
) -> tuple[tuple[str, str], ...]:
    non_hydrogen_atoms = [
        atom for atom in reference_atoms if atom.element.upper() != "H"
    ]
    candidate_atoms = (
        non_hydrogen_atoms if len(non_hydrogen_atoms) >= 2 else reference_atoms
    )
    if len(candidate_atoms) < 2:
        return ()
    return (
        (
            str(candidate_atoms[0].atom_name),
            str(candidate_atoms[1].atom_name),
        ),
    )


_ANCHOR_ELEMENT_PRIORITY = (
    "O",
    "N",
    "S",
    "P",
    "F",
    "CL",
    "BR",
    "I",
    "C",
)


def _matching_frame_from_pdb_atoms(
    input_path: Path,
    *,
    residue_atoms: list[object],
) -> _MatchingFrame:
    return _MatchingFrame(
        filepath=input_path,
        atoms=[
            _MatchingAtom(
                atom_id=index,
                element=str(atom.element),
                coordinates=np.asarray(atom.coordinates, dtype=float).copy(),
            )
            for index, atom in enumerate(residue_atoms, start=1)
        ],
    )


def _build_xyz_partial_candidate_summaries(
    *,
    unmatched_atom_records: Sequence[tuple[int, object]],
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_match_tolerance_a: float,
) -> tuple[SolventShellResidueMismatchSummary, ...]:
    if not unmatched_atom_records or not reference_atoms:
        return ()

    reference_element_counts = Counter(
        str(atom.element).upper() for atom in reference_atoms
    )
    candidate_records = [
        (index, atom)
        for index, atom in unmatched_atom_records
        if str(atom.element).upper() in reference_element_counts
    ]
    if not candidate_records:
        return ()

    available_elements = {
        str(atom.element).upper() for _index, atom in candidate_records
    }
    anchor_index = _select_partial_anchor_index(
        reference_atoms,
        available_elements=available_elements,
    )
    if anchor_index is None:
        return ()

    reference_names = [
        _normalized_atom_name(
            str(atom.atom_name),
            fallback=f"{atom.element}{index + 1}",
        )
        for index, atom in enumerate(reference_atoms)
    ]
    reference_anchor_atom = reference_atoms[anchor_index]
    anchor_element = str(reference_anchor_atom.element).upper()
    anchor_reference_indices = [anchor_index] + [
        index
        for index, atom in enumerate(reference_atoms)
        if index != anchor_index
        and str(atom.element).upper() == anchor_element
    ]
    anchor_records = [
        (source_index, atom)
        for source_index, atom in candidate_records
        if str(atom.element).upper() == anchor_element
    ]
    if not anchor_records:
        return ()

    anchor_capacity = max(len(anchor_reference_indices), 1)
    summaries: list[SolventShellResidueMismatchSummary] = []
    used_source_indices: set[int] = set()
    sorted_anchor_records = sorted(
        anchor_records,
        key=lambda item: (int(item[1].atom_id), int(item[0])),
    )
    for candidate_index, chunk_start in enumerate(
        range(0, len(sorted_anchor_records), anchor_capacity),
        start=1,
    ):
        anchor_chunk = [
            item
            for item in sorted_anchor_records[
                chunk_start : chunk_start + anchor_capacity
            ]
            if int(item[0]) not in used_source_indices
        ]
        if not anchor_chunk:
            continue
        assigned_source_by_ref_index: dict[int, tuple[int, object]] = {}
        assigned_ref_indices: set[int] = set()
        for ref_index, (source_index, source_atom) in zip(
            anchor_reference_indices,
            anchor_chunk,
        ):
            used_source_indices.add(int(source_index))
            assigned_source_by_ref_index[int(ref_index)] = (
                int(source_index),
                source_atom,
            )
            assigned_ref_indices.add(int(ref_index))

        primary_source_index, primary_source_atom = (
            assigned_source_by_ref_index[anchor_index]
        )
        primary_anchor_coordinates = np.asarray(
            primary_source_atom.coordinates,
            dtype=float,
        )
        for ref_index, reference_atom in sorted(
            enumerate(reference_atoms),
            key=lambda item: (
                float(
                    np.linalg.norm(
                        np.asarray(item[1].coordinates, dtype=float)
                        - np.asarray(
                            reference_anchor_atom.coordinates, dtype=float
                        )
                    )
                ),
                item[0],
            ),
        ):
            if ref_index in assigned_ref_indices:
                continue
            reference_element = str(reference_atom.element).upper()
            candidate_matches = [
                (source_index, atom)
                for source_index, atom in candidate_records
                if int(source_index) not in used_source_indices
                and str(atom.element).upper() == reference_element
            ]
            if not candidate_matches:
                continue
            expected_distance = float(
                np.linalg.norm(
                    np.asarray(reference_atom.coordinates, dtype=float)
                    - np.asarray(
                        reference_anchor_atom.coordinates, dtype=float
                    )
                )
            )
            best_source_index: int | None = None
            best_source_atom: object | None = None
            best_delta: float | None = None
            for source_index, source_atom in candidate_matches:
                observed_distance = float(
                    np.linalg.norm(
                        np.asarray(source_atom.coordinates, dtype=float)
                        - primary_anchor_coordinates
                    )
                )
                distance_delta = abs(observed_distance - expected_distance)
                if best_delta is None or distance_delta < best_delta:
                    best_source_index = int(source_index)
                    best_source_atom = source_atom
                    best_delta = distance_delta
            if (
                best_source_index is None
                or best_source_atom is None
                or best_delta is None
                or best_delta
                > _xyz_partial_assignment_tolerance(
                    element=reference_element,
                    reference_match_tolerance_a=reference_match_tolerance_a,
                )
            ):
                continue
            used_source_indices.add(best_source_index)
            assigned_source_by_ref_index[int(ref_index)] = (
                best_source_index,
                best_source_atom,
            )
            assigned_ref_indices.add(int(ref_index))

        summaries.append(
            _build_xyz_partial_candidate_summary(
                residue_name=reference_entry.residue_name,
                residue_number=candidate_index,
                assigned_source_by_ref_index=assigned_source_by_ref_index,
                reference_atoms=reference_atoms,
                reference_names=reference_names,
            )
        )
    return tuple(summaries)


def _build_xyz_partial_candidate_summary(
    *,
    residue_name: str,
    residue_number: int,
    assigned_source_by_ref_index: dict[int, tuple[int, object]],
    reference_atoms: tuple[object, ...],
    reference_names: Sequence[str],
) -> SolventShellResidueMismatchSummary:
    ordered_assignments = [
        (ref_index, source_index, source_atom)
        for ref_index, (source_index, source_atom) in sorted(
            assigned_source_by_ref_index.items()
        )
    ]
    missing_atom_names = tuple(
        reference_names[index]
        for index in range(len(reference_atoms))
        if index not in assigned_source_by_ref_index
    )
    common_reference_coordinates = [
        np.asarray(reference_atoms[ref_index].coordinates, dtype=float)
        for ref_index, _source_index, _source_atom in ordered_assignments
    ]
    common_source_coordinates = [
        np.asarray(source_atom.coordinates, dtype=float)
        for _ref_index, _source_index, source_atom in ordered_assignments
    ]
    distance_pair_count = 0
    distribution_rmsd_a = 0.0
    max_distance_delta_a = 0.0
    if len(common_reference_coordinates) >= 2:
        reference_distances = _pairwise_distance_vector(
            common_reference_coordinates
        )
        source_distances = _pairwise_distance_vector(common_source_coordinates)
        if len(reference_distances) == len(source_distances):
            distance_deltas = source_distances - reference_distances
            distance_pair_count = int(len(reference_distances))
            if distance_pair_count > 0:
                distribution_rmsd_a = float(
                    np.sqrt(np.mean(np.square(distance_deltas)))
                )
                max_distance_delta_a = float(np.max(np.abs(distance_deltas)))
    mismatch_reason = (
        "partial XYZ solvent candidate inferred from unmatched anchor atoms"
        if len(ordered_assignments) == 1
        else "partial XYZ solvent candidate inferred from unmatched reference-element atoms"
    )
    return SolventShellResidueMismatchSummary(
        residue_name=residue_name,
        residue_number=residue_number,
        observed_atom_count=len(ordered_assignments),
        common_atom_count=len(ordered_assignments),
        reference_atom_count=len(reference_atoms),
        missing_atom_names=missing_atom_names,
        extra_atom_names=(),
        distance_pair_count=distance_pair_count,
        distribution_rmsd_a=distribution_rmsd_a,
        max_distance_delta_a=max_distance_delta_a,
        mismatch_reason=mismatch_reason,
        source_atom_ids=tuple(
            int(source_atom.atom_id)
            for _ref_index, _source_index, source_atom in ordered_assignments
        ),
    )


def _select_partial_anchor_index(
    reference_atoms: Sequence[object],
    *,
    available_elements: set[str] | None = None,
) -> int | None:
    if not reference_atoms:
        return None
    non_hydrogen_indices = [
        index
        for index, atom in enumerate(reference_atoms)
        if str(atom.element).upper() != "H"
        and (
            available_elements is None
            or str(atom.element).upper() in available_elements
        )
    ]
    if not non_hydrogen_indices:
        fallback_indices = [
            index
            for index, atom in enumerate(reference_atoms)
            if available_elements is None
            or str(atom.element).upper() in available_elements
        ]
        return fallback_indices[0] if fallback_indices else None

    non_hydrogen_counts = Counter(
        str(reference_atoms[index].element).upper()
        for index in non_hydrogen_indices
    )
    for element in _ANCHOR_ELEMENT_PRIORITY:
        matches = [
            index
            for index in non_hydrogen_indices
            if str(reference_atoms[index].element).upper() == element
            and non_hydrogen_counts[element] == 1
        ]
        if matches:
            return matches[0]
    for element in _ANCHOR_ELEMENT_PRIORITY:
        matches = [
            index
            for index in non_hydrogen_indices
            if str(reference_atoms[index].element).upper() == element
        ]
        if matches:
            return matches[0]
    return non_hydrogen_indices[0]


def _xyz_partial_assignment_tolerance(
    *,
    element: str,
    reference_match_tolerance_a: float,
) -> float:
    base_tolerance = max(float(reference_match_tolerance_a) * 4.0, 0.75)
    if str(element).upper() == "H":
        return base_tolerance
    return max(base_tolerance, 1.25)


def _build_pdb_residue_mismatch_summary(
    *,
    residue_name: str,
    residue_number: int,
    residue_atoms: list[object],
    reference_entry: ReferenceLibraryEntry,
    reference_atoms: tuple[object, ...],
    reference_match_tolerance_a: float,
) -> SolventShellResidueMismatchSummary | None:
    reference_names = [
        _normalized_atom_name(
            str(atom.atom_name),
            fallback=f"{atom.element}{index + 1}",
        )
        for index, atom in enumerate(reference_atoms)
    ]
    residue_names = [
        _normalized_atom_name(
            str(atom.atom_name),
            fallback=f"{atom.element}{index + 1}",
        )
        for index, atom in enumerate(residue_atoms)
    ]
    reference_counts = Counter(reference_names)
    residue_counts = Counter(residue_names)

    remaining_missing = reference_counts - residue_counts
    missing_names: list[str] = []
    for atom_name in reference_names:
        if remaining_missing[atom_name] <= 0:
            continue
        missing_names.append(atom_name)
        remaining_missing[atom_name] -= 1

    remaining_extra = residue_counts - reference_counts
    extra_names: list[str] = []
    for atom_name in residue_names:
        if remaining_extra[atom_name] <= 0:
            continue
        extra_names.append(atom_name)
        remaining_extra[atom_name] -= 1

    missing_atom_names = tuple(missing_names)
    extra_atom_names = tuple(extra_names)
    if not missing_atom_names and not extra_atom_names:
        return None

    residue_atoms_by_name: dict[str, list[object]] = defaultdict(list)
    for index, atom in enumerate(residue_atoms):
        residue_atoms_by_name[residue_names[index]].append(atom)

    consumed_counts: Counter[str] = Counter()
    common_reference_coordinates: list[np.ndarray] = []
    common_residue_coordinates: list[np.ndarray] = []
    common_atom_count = 0
    for index, reference_atom in enumerate(reference_atoms):
        atom_name = reference_names[index]
        atom_matches = residue_atoms_by_name.get(atom_name, [])
        occurrence = consumed_counts[atom_name]
        if occurrence >= len(atom_matches):
            continue
        common_atom_count += 1
        common_reference_coordinates.append(
            np.asarray(reference_atom.coordinates, dtype=float)
        )
        common_residue_coordinates.append(
            np.asarray(atom_matches[occurrence].coordinates, dtype=float)
        )
        consumed_counts[atom_name] += 1
    if common_atom_count == 0:
        return None

    distance_pair_count = 0
    distribution_rmsd_a = 0.0
    max_distance_delta_a = 0.0
    if common_atom_count >= 2:
        reference_distances = _pairwise_distance_vector(
            common_reference_coordinates
        )
        residue_distances = _pairwise_distance_vector(
            common_residue_coordinates
        )
        if len(reference_distances) == len(residue_distances):
            distance_deltas = residue_distances - reference_distances
            distance_pair_count = int(len(reference_distances))
            if distance_pair_count > 0:
                distribution_rmsd_a = float(
                    np.sqrt(np.mean(np.square(distance_deltas)))
                )
                max_distance_delta_a = float(np.max(np.abs(distance_deltas)))

    same_reference_residue = (
        residue_name.upper().strip()
        == reference_entry.residue_name.upper().strip()
    )
    geometry_consistent = common_atom_count < 2 or max_distance_delta_a <= max(
        float(reference_match_tolerance_a), 0.35
    )
    if not same_reference_residue and (
        common_atom_count < 2 or not geometry_consistent
    ):
        return None

    reason_parts: list[str] = []
    if missing_atom_names:
        reason_parts.append("missing reference atoms")
    if extra_atom_names:
        reason_parts.append("contains extra non-reference atoms")
    if not reason_parts:
        reason_parts.append("did not resolve to a complete solvent residue")
    return SolventShellResidueMismatchSummary(
        residue_name=residue_name,
        residue_number=residue_number,
        observed_atom_count=len(residue_atoms),
        common_atom_count=common_atom_count,
        reference_atom_count=len(reference_atoms),
        missing_atom_names=missing_atom_names,
        extra_atom_names=extra_atom_names,
        distance_pair_count=distance_pair_count,
        distribution_rmsd_a=distribution_rmsd_a,
        max_distance_delta_a=max_distance_delta_a,
        mismatch_reason=", ".join(reason_parts),
        source_atom_ids=tuple(int(atom.atom_id) for atom in residue_atoms),
    )


def _pairwise_distance_vector(
    coordinates: Sequence[np.ndarray],
) -> np.ndarray:
    coordinate_array = np.asarray(coordinates, dtype=float)
    if len(coordinate_array) < 2:
        return np.zeros(0, dtype=float)
    distances: list[float] = []
    for index in range(len(coordinate_array) - 1):
        deltas = coordinate_array[index + 1 :] - coordinate_array[index]
        distances.extend(
            float(value) for value in np.linalg.norm(deltas, axis=1)
        )
    return np.asarray(distances, dtype=float)


def _build_pdb_analysis_notes(
    *,
    residue_mismatch_count: int,
) -> tuple[str, ...]:
    notes = [
        "PDB matching is constrained to existing residue groups so "
        "reported residue names reflect the source file.",
    ]
    if residue_mismatch_count > 0:
        notes.append(
            "Incomplete residue groups that retained identifiable solvent "
            "atom names are preserved as mismatches with missing-atom details."
        )
    return tuple(notes)


def _build_xyz_analysis_note(
    *,
    partial_candidate_count: int,
) -> str:
    if partial_candidate_count > 0:
        return (
            "XYZ matching uses geometric complete-molecule fits first, then "
            "infers partial solvent candidates heuristically from unmatched "
            "reference-element atoms so those anchors can guide solvent rebuilds."
        )
    return (
        "XYZ matching uses geometric complete-molecule fits first, then "
        "checks unmatched atoms for solvent-like partial candidates that "
        "could guide solvent rebuilds."
    )


__all__ = [
    "DEFAULT_REFERENCE_MATCH_TOLERANCE_A",
    "SolventShellAnalysisResult",
    "SolventShellBuildResult",
    "SolventShellResidueMismatchSummary",
    "SolventShellResidueSummary",
    "analyze_solvent_shell",
    "build_solvent_shell_output",
    "default_director_atom_name",
    "reference_atom_choices",
]
