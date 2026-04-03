from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saxshell.structure import PDBAtom, PDBStructure

from .workflow import (
    AnchorPairDefinition,
    ConvertedResidue,
    MoleculeDefinition,
    ReferenceLibraryEntry,
    XYZToPDBAssertionResidueSummary,
    XYZToPDBAssertionResult,
    XYZToPDBExportResult,
    XYZToPDBInspectionResult,
    XYZToPDBOperationCancelled,
    XYZToPDBPreviewResult,
    XYZToPDBReferenceUpdateCandidate,
    XYZToPDBWorkflow,
    _build_cryst1_line,
    _element_order_mismatch_message,
    _FrameSearchCache,
    _normalized_atom_name,
    _normalized_element_symbol,
    _normalized_residue_name,
    _raise_if_cancelled,
    _resolve_reference_backbone_pairs,
    default_reference_library_dir,
    list_reference_library,
    resolve_reference_path,
    rigid_alignment_from_points,
    rotation_matrix_from_to,
    suggest_output_dir,
)

_REFERENCE_BOND_THRESHOLD_SCALE = 1.18
_REFERENCE_BOND_TOLERANCE = 0.18
_TIGHT_PASS_SCALE = 0.85
_RELAXED_PASS_SCALE = 1.35
_HIGH_RMSD_WARNING = 0.45
_MULTI_SOLUTION_LIMIT = 64
_DEFAULT_MAX_MISSING_HYDROGENS = 0
_MISSING_HYDROGEN_MODES = {
    "leave_unassigned",
    "assign_orphaned",
    "restore_missing",
}
_COVALENT_RADII = {
    "Ag": 1.45,
    "Al": 1.21,
    "As": 1.19,
    "Au": 1.36,
    "B": 0.84,
    "Ba": 2.15,
    "Be": 0.96,
    "Bi": 1.48,
    "Br": 1.20,
    "C": 0.76,
    "Ca": 1.76,
    "Cd": 1.44,
    "Cl": 1.02,
    "Co": 1.26,
    "Cr": 1.39,
    "Cs": 2.44,
    "Cu": 1.32,
    "F": 0.57,
    "Fe": 1.32,
    "Ga": 1.22,
    "Ge": 1.20,
    "H": 0.31,
    "Hg": 1.32,
    "I": 1.39,
    "In": 1.42,
    "K": 2.03,
    "Li": 1.28,
    "Mg": 1.41,
    "Mn": 1.39,
    "Mo": 1.54,
    "N": 0.71,
    "Na": 1.66,
    "Ni": 1.24,
    "O": 0.66,
    "P": 1.07,
    "Pb": 1.46,
    "Pd": 1.39,
    "Pt": 1.36,
    "S": 1.05,
    "Sb": 1.39,
    "Se": 1.20,
    "Si": 1.11,
    "Sn": 1.39,
    "Sr": 1.95,
    "Te": 1.38,
    "Ti": 1.60,
    "Tl": 1.45,
    "V": 1.53,
    "Zn": 1.22,
    "Zr": 1.75,
}


@dataclass(slots=True)
class XYZToPDBSampleAnalysis:
    """Summary of the sample XYZ frame chosen for mapping."""

    inspection: XYZToPDBInspectionResult
    sample_file: Path
    sample_comment: str
    total_atoms: int
    element_counts: dict[str, int]

    @property
    def first_output_file(self) -> Path:
        return suggest_output_dir(self.inspection.input_path) / (
            f"{self.sample_file.stem}.pdb"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "inspection": self.inspection.to_dict(),
            "sample_file": str(self.sample_file),
            "sample_comment": self.sample_comment,
            "total_atoms": int(self.total_atoms),
            "element_counts": dict(self.element_counts),
        }


@dataclass(slots=True)
class FreeAtomMappingInput:
    """User-entered free-atom assignment."""

    element: str
    residue_name: str


@dataclass(slots=True)
class ReferenceBondToleranceInput:
    """User-editable bond tolerance percentage for one direct bond."""

    atom1_name: str
    atom2_name: str
    tolerance: float


@dataclass(slots=True)
class MoleculeMappingInput:
    """User-entered reference-molecule mapping."""

    reference_name: str
    residue_name: str
    bond_tolerances: tuple[ReferenceBondToleranceInput, ...] = ()
    tight_pass_scale: float = _TIGHT_PASS_SCALE
    relaxed_pass_scale: float = _RELAXED_PASS_SCALE
    max_assignment_distance: float | None = None
    max_missing_hydrogens: int = _DEFAULT_MAX_MISSING_HYDROGENS


@dataclass(slots=True)
class ResolvedReferenceBond:
    """One resolved direct bond inside a reference molecule."""

    atom1_index: int
    atom2_index: int
    atom1_name: str
    atom2_name: str
    reference_length: float
    tolerance: float


@dataclass(slots=True)
class _ResolvedMoleculeVariant:
    """One searchable molecule variant, optionally missing hydrogens."""

    molecule_definition: MoleculeDefinition
    variant_reference_atoms: tuple[PDBAtom, ...]
    variant_bonds: tuple[ResolvedReferenceBond, ...]
    kept_full_indices: tuple[int, ...]
    missing_full_indices: tuple[int, ...]


@dataclass(slots=True)
class ResolvedMoleculeMapping:
    """Fully resolved mapping input built from one reference
    molecule."""

    reference_name: str
    reference_path: Path
    residue_name: str
    reference_atoms: tuple[PDBAtom, ...]
    bonds: tuple[ResolvedReferenceBond, ...]
    preferred_backbone_pairs: tuple[tuple[str, str], ...]
    element_counts: dict[str, int]
    tight_pass_scale: float
    relaxed_pass_scale: float
    max_assignment_distance: float | None
    max_missing_hydrogens: int
    variants: tuple[_ResolvedMoleculeVariant, ...]

    @property
    def label(self) -> str:
        return self.residue_name

    @property
    def full_variants(self) -> tuple[_ResolvedMoleculeVariant, ...]:
        return tuple(
            variant
            for variant in self.variants
            if not variant.missing_full_indices
        )

    @property
    def deprotonated_variants(self) -> tuple[_ResolvedMoleculeVariant, ...]:
        return tuple(
            variant
            for variant in self.variants
            if variant.missing_full_indices
        )


@dataclass(slots=True)
class XYZToPDBMappingPlan:
    """Resolved UI mapping inputs ready for estimation and matching."""

    molecules: tuple[ResolvedMoleculeMapping, ...]
    free_atoms: dict[str, FreeAtomMappingInput]
    hydrogen_mode: str
    pbc_params: dict[str, float | str]


@dataclass(slots=True)
class XYZToPDBEstimateSolution:
    """One integer solution for the estimated molecule/free-atom
    counts."""

    molecule_counts: tuple[int, ...]
    free_atom_counts: dict[str, int]
    unassigned_counts: dict[str, int]
    assigned_atoms: int
    total_atoms: int

    @property
    def is_complete(self) -> bool:
        return sum(self.unassigned_counts.values()) == 0

    @property
    def unassigned_total(self) -> int:
        return int(sum(self.unassigned_counts.values()))

    def molecule_count_by_residue(
        self,
        plan: XYZToPDBMappingPlan,
    ) -> dict[str, int]:
        return {
            molecule.residue_name: int(self.molecule_counts[index])
            for index, molecule in enumerate(plan.molecules)
            if self.molecule_counts[index] > 0
        }

    def to_dict(self, plan: XYZToPDBMappingPlan) -> dict[str, object]:
        return {
            "molecule_counts": self.molecule_count_by_residue(plan),
            "free_atom_counts": dict(self.free_atom_counts),
            "unassigned_counts": dict(self.unassigned_counts),
            "assigned_atoms": int(self.assigned_atoms),
            "total_atoms": int(self.total_atoms),
            "is_complete": bool(self.is_complete),
        }


@dataclass(slots=True)
class XYZToPDBEstimateResult:
    """Mapping-estimate summary for the sample frame."""

    analysis: XYZToPDBSampleAnalysis
    plan: XYZToPDBMappingPlan
    solutions: tuple[XYZToPDBEstimateSolution, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis": self.analysis.to_dict(),
            "solutions": [
                solution.to_dict(self.plan) for solution in self.solutions
            ],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class XYZToPDBMappingTestResult:
    """Result of testing the PDB mapping on the sample frame."""

    analysis: XYZToPDBSampleAnalysis
    plan: XYZToPDBMappingPlan
    solution: XYZToPDBEstimateSolution | None
    output_dir: Path
    residues: tuple[ConvertedResidue, ...]
    molecule_counts: dict[str, int]
    residue_counts: dict[str, int]
    free_atom_counts: dict[str, int]
    unassigned_counts: dict[str, int]
    total_atoms: int
    warnings: tuple[str, ...]
    console_messages: tuple[str, ...]

    @property
    def first_output_file(self) -> Path:
        return self.output_dir / f"{self.analysis.sample_file.stem}.pdb"

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis": self.analysis.to_dict(),
            "output_dir": str(self.output_dir),
            "first_output_file": str(self.first_output_file),
            "molecule_counts": dict(self.molecule_counts),
            "residue_counts": dict(self.residue_counts),
            "free_atom_counts": dict(self.free_atom_counts),
            "unassigned_counts": dict(self.unassigned_counts),
            "total_atoms": int(self.total_atoms),
            "warnings": list(self.warnings),
            "console_messages": list(self.console_messages),
        }


@dataclass(slots=True)
class _VariantMatch:
    """Internal match payload before atoms are written."""

    variant: _ResolvedMoleculeVariant
    assignment: tuple[int, ...]
    pass_name: str
    fit_rmsd: float
    mean_bond_deviation: float
    transformed_full_coordinates: np.ndarray


@dataclass(slots=True)
class _MoleculeAssertionEntry:
    """One exported molecule checked during assertion mode."""

    residue_name: str
    molecule_file: Path
    common_atom_count: int
    distance_pair_count: int
    distribution_rmsd: float
    max_distance_delta: float
    missing_atom_names: tuple[str, ...] = ()


class XYZToPDBMappingWorkflow(XYZToPDBWorkflow):
    """Native UI workflow that estimates and tests mappings without a
    JSON file."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        reference_library_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            input_path,
            config_file=None,
            reference_library_dir=reference_library_dir,
            output_dir=output_dir,
        )

    def analyze_input(self) -> XYZToPDBSampleAnalysis:
        inspection = self.inspect()
        if not inspection.xyz_files:
            raise ValueError("No XYZ files were found for conversion.")
        sample_file = inspection.xyz_files[0]
        frame = self.read_xyz_frame(sample_file)
        counts = Counter(atom.element for atom in frame.atoms)
        return XYZToPDBSampleAnalysis(
            inspection=inspection,
            sample_file=sample_file,
            sample_comment=frame.comment,
            total_atoms=len(frame.atoms),
            element_counts=dict(sorted(counts.items())),
        )

    def default_bond_tolerances(
        self,
        reference_name: str,
    ) -> tuple[ReferenceBondToleranceInput, ...]:
        reference_path = resolve_reference_path(
            reference_name,
            library_dir=self.reference_library_dir,
        )
        atoms = tuple(PDBStructure.from_file(reference_path).atoms)
        bonds = _infer_direct_reference_bonds(atoms)
        return tuple(
            ReferenceBondToleranceInput(
                atom1_name=bond.atom1_name,
                atom2_name=bond.atom2_name,
                tolerance=_tolerance_percent_from_absolute(
                    bond.reference_length,
                    bond.tolerance,
                ),
            )
            for bond in bonds
        )

    def reference_summary(
        self,
        reference_name: str,
    ) -> tuple[ReferenceLibraryEntry, tuple[ReferenceBondToleranceInput, ...]]:
        entries = {
            entry.name: entry
            for entry in list_reference_library(self.reference_library_dir)
        }
        if reference_name not in entries:
            raise FileNotFoundError(
                f"Reference molecule {reference_name!r} is not available."
            )
        return entries[reference_name], self.default_bond_tolerances(
            reference_name
        )

    def estimate_mapping(
        self,
        *,
        molecule_inputs: Sequence[MoleculeMappingInput],
        free_atom_inputs: Sequence[FreeAtomMappingInput],
        hydrogen_mode: str = "leave_unassigned",
        pbc_params: dict[str, float | str] | None = None,
        max_solutions: int = _MULTI_SOLUTION_LIMIT,
    ) -> XYZToPDBEstimateResult:
        analysis = self.analyze_input()
        plan = self._resolve_mapping_plan(
            molecule_inputs=molecule_inputs,
            free_atom_inputs=free_atom_inputs,
            hydrogen_mode=hydrogen_mode,
            pbc_params=pbc_params,
        )
        solutions, truncated = _enumerate_estimate_solutions(
            total_counts=analysis.element_counts,
            plan=plan,
            max_solutions=max_solutions,
        )
        warnings: list[str] = []
        if len(solutions) > 1:
            warnings.append(
                "Multiple stoichiometric mapping solutions were found. "
                "Select the solution you want to test or export."
            )
        if truncated:
            warnings.append(
                "The estimator found more solutions than were shown. Refine "
                "the molecule/free-atom list to reduce ambiguity."
            )
        if not solutions:
            warnings.append(
                "No molecule-count solution could be formed from the selected "
                "reference molecules. The test/export steps will treat all "
                "atoms as free or unassigned."
            )
        return XYZToPDBEstimateResult(
            analysis=analysis,
            plan=plan,
            solutions=tuple(solutions),
            warnings=tuple(warnings),
        )

    def test_mapping(
        self,
        *,
        molecule_inputs: Sequence[MoleculeMappingInput],
        free_atom_inputs: Sequence[FreeAtomMappingInput],
        hydrogen_mode: str = "leave_unassigned",
        pbc_params: dict[str, float | str] | None = None,
        selected_solution_index: int = 0,
        output_dir: str | Path | None = None,
    ) -> XYZToPDBMappingTestResult:
        estimate = self.estimate_mapping(
            molecule_inputs=molecule_inputs,
            free_atom_inputs=free_atom_inputs,
            hydrogen_mode=hydrogen_mode,
            pbc_params=pbc_params,
        )
        solution = _select_solution(estimate, selected_solution_index)
        frame = self.read_xyz_frame(estimate.analysis.sample_file)
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else (
                self.output_dir
                if self.output_dir is not None
                else suggest_output_dir(self.input_path)
            )
        )
        (
            residues,
            molecule_counts,
            residue_counts,
            free_atom_counts,
            unassigned_counts,
            warnings,
            console_messages,
            total_atoms,
        ) = self._map_frame(
            frame,
            estimate.plan,
            solution,
        )
        merged_warnings = list(estimate.warnings)
        merged_warnings.extend(warnings)
        return XYZToPDBMappingTestResult(
            analysis=estimate.analysis,
            plan=estimate.plan,
            solution=solution,
            output_dir=resolved_output_dir,
            residues=tuple(residues),
            molecule_counts=molecule_counts,
            residue_counts=residue_counts,
            free_atom_counts=free_atom_counts,
            unassigned_counts=unassigned_counts,
            total_atoms=total_atoms,
            warnings=tuple(merged_warnings),
            console_messages=tuple(console_messages),
        )

    def export_with_mapping(
        self,
        *,
        molecule_inputs: Sequence[MoleculeMappingInput],
        free_atom_inputs: Sequence[FreeAtomMappingInput],
        hydrogen_mode: str = "leave_unassigned",
        pbc_params: dict[str, float | str] | None = None,
        selected_solution_index: int = 0,
        output_dir: str | Path | None = None,
        assert_molecule_shapes: bool = False,
        estimate_result: XYZToPDBEstimateResult | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> XYZToPDBExportResult:
        _raise_if_cancelled(cancel_callback)
        estimate = (
            estimate_result
            if estimate_result is not None
            else self.estimate_mapping(
                molecule_inputs=molecule_inputs,
                free_atom_inputs=free_atom_inputs,
                hydrogen_mode=hydrogen_mode,
                pbc_params=pbc_params,
            )
        )
        solution = _select_solution(estimate, selected_solution_index)
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else (
                self.output_dir
                if self.output_dir is not None
                else suggest_output_dir(self.input_path)
            )
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        header_lines = _header_lines_from_pbc(estimate.plan.pbc_params)
        written_files: list[Path] = []
        first_test_result: XYZToPDBMappingTestResult | None = None
        total_files = estimate.analysis.inspection.total_files
        validation_steps = total_files if total_files > 1 else 0
        mapping_step = validation_steps + 1
        total_steps = max(
            validation_steps
            + 1
            + total_files
            + (1 if assert_molecule_shapes else 0),
            1,
        )
        template_targets = _template_mapping_targets(
            estimate.plan,
            solution,
        )
        assertion_entries: list[_MoleculeAssertionEntry] = []
        assertion_result: XYZToPDBAssertionResult | None = None
        assertion_dir = (
            resolved_output_dir / "assertion_molecules"
            if assert_molecule_shapes
            else None
        )
        if assertion_dir is not None:
            assertion_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(
                0,
                total_steps,
                "Estimating first-frame mapping...",
            )
        _raise_if_cancelled(cancel_callback)
        if log_callback is not None:
            if estimate_result is None:
                log_callback(
                    "Estimated the first-frame mapping from the current "
                    "reference and free-atom definitions."
                )
            else:
                log_callback(
                    "Reusing the current mapping estimate for conversion."
                )
        self._validate_xyz_element_order(
            estimate.analysis.inspection.xyz_files,
            progress_callback=progress_callback,
            progress_step=1,
            total_steps=total_steps,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
        template_residues: tuple[ConvertedResidue, ...] | None = None
        template_molecule_counts: dict[str, int] | None = None
        template_residue_counts: dict[str, int] | None = None
        template_free_atom_counts: dict[str, int] | None = None
        template_unassigned_counts: dict[str, int] | None = None
        template_total_atoms: int | None = None
        for index, xyz_file in enumerate(
            estimate.analysis.inspection.xyz_files,
            start=1,
        ):
            _raise_if_cancelled(cancel_callback)
            frame = self.read_xyz_frame(xyz_file)
            if index == 1:
                if progress_callback is not None:
                    progress_callback(
                        mapping_step,
                        total_steps,
                        f"Mapping template from {xyz_file.name}...",
                    )

                def on_template_status(
                    current_counts: dict[str, int],
                ) -> None:
                    if progress_callback is None or not template_targets:
                        return
                    progress_callback(
                        mapping_step,
                        total_steps,
                        f"Template mapping progress in {xyz_file.name}: "
                        + _format_template_mapping_status(
                            template_targets,
                            current_counts,
                        ),
                    )

                (
                    residues,
                    molecule_counts,
                    residue_counts,
                    free_atom_counts,
                    unassigned_counts,
                    warnings,
                    console_messages,
                    total_atoms,
                ) = self._map_frame(
                    frame,
                    estimate.plan,
                    solution,
                    mapping_status_callback=on_template_status,
                    log_callback=log_callback,
                    cancel_callback=cancel_callback,
                )
                template_residues = tuple(residues)
                template_molecule_counts = dict(molecule_counts)
                template_residue_counts = dict(residue_counts)
                template_free_atom_counts = dict(free_atom_counts)
                template_unassigned_counts = dict(unassigned_counts)
                template_total_atoms = int(total_atoms)
                if log_callback is not None:
                    if template_targets:
                        log_callback(
                            "Template mapping counts: "
                            + _format_template_mapping_status(
                                template_targets,
                                template_molecule_counts,
                            )
                        )
                    log_callback(
                        "First-frame mapping succeeded. Reusing its atom-index "
                        "template for the remaining frames."
                    )
                first_test_result = XYZToPDBMappingTestResult(
                    analysis=estimate.analysis,
                    plan=estimate.plan,
                    solution=solution,
                    output_dir=resolved_output_dir,
                    residues=template_residues,
                    molecule_counts=template_molecule_counts,
                    residue_counts=template_residue_counts,
                    free_atom_counts=template_free_atom_counts,
                    unassigned_counts=template_unassigned_counts,
                    total_atoms=template_total_atoms,
                    warnings=tuple(list(estimate.warnings) + warnings),
                    console_messages=tuple(console_messages),
                )
            else:
                assert template_residues is not None
                assert template_molecule_counts is not None
                assert template_residue_counts is not None
                assert template_free_atom_counts is not None
                assert template_unassigned_counts is not None
                assert template_total_atoms is not None
                residues = _apply_template_mapping(frame, template_residues)
                molecule_counts = dict(template_molecule_counts)
                residue_counts = dict(template_residue_counts)
                free_atom_counts = dict(template_free_atom_counts)
                unassigned_counts = dict(template_unassigned_counts)
                warnings = []
                console_messages = []
                total_atoms = int(template_total_atoms)
                if log_callback is not None:
                    log_callback(
                        f"Reused the first-frame atom-order mapping for {xyz_file.name}."
                    )

            atoms = [
                atom.copy() for residue in residues for atom in residue.atoms
            ]
            _raise_if_cancelled(cancel_callback)
            structure = PDBStructure(atoms=atoms, source_name=xyz_file.stem)
            written_path = structure.write_pdb_file(
                resolved_output_dir / f"{xyz_file.stem}.pdb",
                header_lines=header_lines,
            )
            written_files.append(written_path)
            if assertion_dir is not None:
                _raise_if_cancelled(cancel_callback)
                frame_assertion_entries = _write_assertion_molecules_for_frame(
                    frame_stem=xyz_file.stem,
                    residues=residues,
                    plan=estimate.plan,
                    assertion_dir=assertion_dir,
                )
                assertion_entries.extend(frame_assertion_entries)
                if log_callback is not None:
                    log_callback(
                        f"[{index}/{total_files}] Assertion mode wrote "
                        f"{len(frame_assertion_entries)} molecule file(s) for {xyz_file.name}."
                    )
            if progress_callback is not None:
                progress_callback(
                    mapping_step + index,
                    total_steps,
                    f"[{index}/{total_files}] Wrote {xyz_file.name}",
                )
            if log_callback is not None:
                log_callback(
                    f"[{index}/{total_files}] {xyz_file.name}: "
                    f"{sum(molecule_counts.values())} molecule residues, "
                    f"{sum(free_atom_counts.values())} free atoms, "
                    f"{sum(unassigned_counts.values())} unassigned atoms."
                )
        if assertion_dir is not None:
            _raise_if_cancelled(cancel_callback)
            if progress_callback is not None:
                progress_callback(
                    total_steps,
                    total_steps,
                    "Analyzing molecule distance distributions...",
                )
            assertion_result = _build_assertion_report(
                plan=estimate.plan,
                assertion_dir=assertion_dir,
                entries=assertion_entries,
            )
            if log_callback is not None:
                log_callback(
                    "Assertion mode "
                    + (
                        "passed."
                        if assertion_result.passed
                        else "reported warnings."
                    )
                )
                log_callback(
                    f"Assertion report written to {assertion_result.report_file}"
                )
                for summary in assertion_result.residue_summaries:
                    log_callback(
                        f"Assertion {summary.residue_name}: "
                        f"{summary.molecule_count} molecule(s), median RMSD "
                        f"{summary.median_distribution_rmsd:.3f} A, max RMSD "
                        f"{summary.max_distribution_rmsd:.3f} A, outliers "
                        f"{summary.outlier_count}."
                    )
                for warning in assertion_result.warnings:
                    log_callback(f"Assertion warning: {warning}")
        if first_test_result is None:
            first_test_result = self.test_mapping(
                molecule_inputs=molecule_inputs,
                free_atom_inputs=free_atom_inputs,
                hydrogen_mode=hydrogen_mode,
                pbc_params=pbc_params,
                selected_solution_index=selected_solution_index,
                output_dir=resolved_output_dir,
            )
        return XYZToPDBExportResult(
            output_dir=resolved_output_dir,
            written_files=tuple(written_files),
            preview=first_test_result,
            progress_total_steps=total_steps,
            assertion_result=assertion_result,
        )

    def _resolve_mapping_plan(
        self,
        *,
        molecule_inputs: Sequence[MoleculeMappingInput],
        free_atom_inputs: Sequence[FreeAtomMappingInput],
        hydrogen_mode: str,
        pbc_params: dict[str, float | str] | None,
    ) -> XYZToPDBMappingPlan:
        if hydrogen_mode not in _MISSING_HYDROGEN_MODES:
            raise ValueError(
                "Hydrogen handling must be one of: "
                + ", ".join(sorted(_MISSING_HYDROGEN_MODES))
            )
        free_atoms: dict[str, FreeAtomMappingInput] = {}
        for item in free_atom_inputs:
            element = _normalized_element_symbol(item.element)
            residue_name = _validated_residue_name(item.residue_name)
            if element in free_atoms:
                raise ValueError(
                    f"Free atom {element} was listed more than once."
                )
            free_atoms[element] = FreeAtomMappingInput(
                element=element,
                residue_name=residue_name,
            )

        resolved_molecules: list[ResolvedMoleculeMapping] = []
        used_residues: dict[str, str] = {}
        for element, definition in free_atoms.items():
            used_residues[definition.residue_name] = f"free atom {element}"
        for item in molecule_inputs:
            residue_name = _validated_residue_name(item.residue_name)
            reference_path = resolve_reference_path(
                item.reference_name,
                library_dir=self.reference_library_dir,
            )
            reference_name = Path(reference_path).stem
            if residue_name in used_residues:
                raise ValueError(
                    f"Residue {residue_name} is already used by "
                    f"{used_residues[residue_name]}."
                )
            used_residues[residue_name] = f"reference {reference_name}"
            structure = PDBStructure.from_file(reference_path)
            atoms = tuple(atom.copy() for atom in structure.atoms)
            if not atoms:
                raise ValueError(
                    f"Reference {reference_name!r} does not contain any atoms."
                )
            bonds = _resolve_bond_inputs(atoms, item.bond_tolerances)
            preferred_backbone_pairs = _resolve_reference_backbone_pairs(
                reference_path,
                atoms=atoms,
            )
            variants = _build_variants(
                reference_name=reference_name,
                reference_path=reference_path,
                residue_name=residue_name,
                atoms=atoms,
                bonds=bonds,
                preferred_backbone_pairs=preferred_backbone_pairs,
                max_assignment_distance=item.max_assignment_distance,
                max_missing_hydrogens=max(int(item.max_missing_hydrogens), 0),
            )
            resolved_molecules.append(
                ResolvedMoleculeMapping(
                    reference_name=reference_name,
                    reference_path=reference_path,
                    residue_name=residue_name,
                    reference_atoms=atoms,
                    bonds=bonds,
                    preferred_backbone_pairs=preferred_backbone_pairs,
                    element_counts=dict(
                        sorted(Counter(atom.element for atom in atoms).items())
                    ),
                    tight_pass_scale=max(float(item.tight_pass_scale), 0.01),
                    relaxed_pass_scale=max(
                        float(item.relaxed_pass_scale),
                        max(float(item.tight_pass_scale), 0.01),
                    ),
                    max_assignment_distance=(
                        None
                        if item.max_assignment_distance is None
                        else float(item.max_assignment_distance)
                    ),
                    max_missing_hydrogens=max(
                        int(item.max_missing_hydrogens),
                        0,
                    ),
                    variants=variants,
                )
            )

        return XYZToPDBMappingPlan(
            molecules=tuple(resolved_molecules),
            free_atoms=free_atoms,
            hydrogen_mode=hydrogen_mode,
            pbc_params=_normalized_pbc_params(pbc_params),
        )

    def _map_frame(
        self,
        frame,
        plan: XYZToPDBMappingPlan,
        solution: XYZToPDBEstimateSolution | None,
        mapping_status_callback: (
            Callable[[dict[str, int]], None] | None
        ) = None,
        log_callback: Callable[[str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> tuple[
        list[ConvertedResidue],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        list[str],
        list[str],
        int,
    ]:
        residues: list[ConvertedResidue] = []
        used = [False] * len(frame.atoms)
        residue_number = 1
        atom_serial = 1
        molecule_counts: Counter[str] = Counter()
        residue_counts: Counter[str] = Counter()
        free_atom_counts: Counter[str] = Counter()
        unassigned_counts: Counter[str] = Counter()
        warnings: list[str] = []
        console_messages: list[str] = []
        terminated_molecule_searches: set[str] = set()
        frame_search_cache = self._build_frame_search_cache(frame)

        def append_console(message: str) -> None:
            console_messages.append(message)
            if log_callback is not None:
                log_callback(message)

        def append_warning(message: str) -> None:
            warnings.append(message)
            if log_callback is not None:
                log_callback(f"warning: {message}")

        expected_counts = (
            list(solution.molecule_counts)
            if solution is not None
            else [0] * len(plan.molecules)
        )
        relaxed_tolerance_hits: Counter[str] = Counter()
        deprotonated_hits: Counter[str] = Counter()
        high_rmsd_hits: Counter[str] = Counter()

        for molecule_index, molecule in enumerate(plan.molecules):
            _raise_if_cancelled(cancel_callback)
            target_count = (
                expected_counts[molecule_index] if solution is not None else 0
            )
            while molecule_counts[molecule.residue_name] < target_count:
                _raise_if_cancelled(cancel_callback)
                match, match_classification = self._find_staged_match(
                    frame,
                    used,
                    molecule,
                    frame_search_cache=frame_search_cache,
                    log_callback=lambda message, residue_name=molecule.residue_name: append_console(
                        f"{residue_name}: {message}"
                    ),
                    cancel_callback=cancel_callback,
                )
                if match is None:
                    break
                (
                    residue,
                    used_indices,
                    materialize_messages,
                    materialize_warnings,
                    atom_serial,
                ) = _materialize_match(
                    match=match,
                    frame=frame,
                    residue_number=residue_number,
                    atom_serial=atom_serial,
                    hydrogen_mode=plan.hydrogen_mode,
                    used=used,
                    match_classification=match_classification,
                )
                for source_index in used_indices:
                    used[source_index] = True
                residues.append(residue)
                residue_number += 1
                molecule_counts[molecule.residue_name] += 1
                residue_counts[molecule.residue_name] += 1
                if target_count > 0:
                    append_console(
                        f"{molecule.residue_name}: matched "
                        f"{molecule_counts[molecule.residue_name]}/{target_count} "
                        "molecule(s)."
                    )
                if mapping_status_callback is not None:
                    mapping_status_callback(
                        dict(sorted(molecule_counts.items()))
                    )
                for message in materialize_messages:
                    append_console(message)
                for message in materialize_warnings:
                    append_warning(message)
                if match_classification == "tolerance":
                    relaxed_tolerance_hits[molecule.residue_name] += 1
                elif match_classification == "deprotonation":
                    deprotonated_hits[molecule.residue_name] += 1
                if match.fit_rmsd > _HIGH_RMSD_WARNING:
                    high_rmsd_hits[molecule.residue_name] += 1
            if (
                molecule_counts[molecule.residue_name] < target_count
                and target_count > 0
            ):
                terminated_molecule_searches.add(molecule.residue_name)
                append_warning(
                    f"{molecule.residue_name}: no additional molecules were "
                    "found after the tight/relaxed search passes; keeping "
                    f"{molecule_counts[molecule.residue_name]}/{target_count} "
                    "matched molecule(s) and terminating that search."
                )

        for molecule_index, molecule in enumerate(plan.molecules):
            expected = (
                expected_counts[molecule_index] if solution is not None else 0
            )
            found = molecule_counts[molecule.residue_name]
            if (
                expected
                and found < expected
                and molecule.residue_name not in terminated_molecule_searches
            ):
                append_warning(
                    f"{molecule.residue_name}: expected {expected} molecule(s) "
                    f"from the estimate, but only matched {found}."
                )
        if mapping_status_callback is not None:
            mapping_status_callback(dict(sorted(molecule_counts.items())))
        for residue_name, count in sorted(relaxed_tolerance_hits.items()):
            append_warning(
                f"{residue_name}: {count} molecule(s) required the relaxed "
                "full-hydrogen pass, which points to a bond-tolerance or "
                "geometry issue rather than deprotonation."
            )
        for residue_name, count in sorted(deprotonated_hits.items()):
            append_warning(
                f"{residue_name}: {count} molecule(s) only matched after "
                "omitting reference hydrogen(s), which is consistent with "
                "deprotonation."
            )
        for residue_name, count in sorted(high_rmsd_hits.items()):
            append_warning(
                f"{residue_name}: {count} molecule(s) showed elevated fit RMSD. "
                "The reference structure may be too far from the simulated geometry."
            )

        free_atom_serials: Counter[str] = Counter()
        for atom_index, xyz_atom in enumerate(frame.atoms):
            _raise_if_cancelled(cancel_callback)
            if used[atom_index]:
                continue
            assignment = plan.free_atoms.get(xyz_atom.element)
            free_atom_serials[xyz_atom.element] += 1
            generated_atom_name = _normalized_atom_name(
                f"{xyz_atom.element}{free_atom_serials[xyz_atom.element]}",
                fallback=xyz_atom.element,
            )
            if assignment is None:
                residue_name = "UNK"
                molecule_name = f"UNASSIGNED_{xyz_atom.element}"
                unassigned_counts[xyz_atom.element] += 1
            else:
                residue_name = assignment.residue_name
                molecule_name = residue_name
                free_atom_counts[xyz_atom.element] += 1
                residue_counts[residue_name] += 1
            residues.append(
                ConvertedResidue(
                    residue_number=residue_number,
                    residue_name=residue_name,
                    molecule_name=molecule_name,
                    atoms=(
                        PDBAtom(
                            atom_id=atom_serial,
                            atom_name=generated_atom_name,
                            residue_name=residue_name,
                            residue_number=residue_number,
                            coordinates=xyz_atom.coordinates.copy(),
                            element=xyz_atom.element,
                        ),
                    ),
                    source_atom_indices=(atom_index,),
                )
            )
            residue_number += 1
            atom_serial += 1

        total_atoms = sum(len(residue.atoms) for residue in residues)
        return (
            residues,
            dict(sorted(molecule_counts.items())),
            dict(sorted(residue_counts.items())),
            dict(sorted(free_atom_counts.items())),
            dict(sorted(unassigned_counts.items())),
            warnings,
            console_messages,
            total_atoms,
        )

    def _find_staged_match(
        self,
        frame,
        used: Sequence[bool],
        molecule: ResolvedMoleculeMapping,
        *,
        frame_search_cache: _FrameSearchCache | None = None,
        log_callback: Callable[[str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> tuple[_VariantMatch | None, str | None]:
        _raise_if_cancelled(cancel_callback)
        match = self._find_variant_match(
            frame,
            used,
            molecule,
            pass_name="tight",
            variants=molecule.full_variants,
            frame_search_cache=frame_search_cache,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
        if match is not None:
            return match, None
        if log_callback is not None:
            log_callback(
                "tight full-hydrogen pass found no acceptable fit; "
                "trying relaxed full-hydrogen pass."
            )
        match = self._find_variant_match(
            frame,
            used,
            molecule,
            pass_name="relaxed",
            variants=molecule.full_variants,
            frame_search_cache=frame_search_cache,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
        if match is not None:
            return match, "tolerance"
        if molecule.max_missing_hydrogens <= 0:
            if log_callback is not None:
                log_callback(
                    "no acceptable molecule fit was found after tight and "
                    "relaxed full-hydrogen passes."
                )
            return None, None
        if log_callback is not None:
            log_callback(
                "full-hydrogen passes found no acceptable fit; trying "
                "deprotonated tight pass."
            )
        match = self._find_variant_match(
            frame,
            used,
            molecule,
            pass_name="tight",
            variants=molecule.deprotonated_variants,
            frame_search_cache=frame_search_cache,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
        if match is not None:
            return match, "deprotonation"
        if log_callback is not None:
            log_callback(
                "tight deprotonated pass found no acceptable fit; trying "
                "relaxed deprotonated pass."
            )
        match = self._find_variant_match(
            frame,
            used,
            molecule,
            pass_name="relaxed",
            variants=molecule.deprotonated_variants,
            frame_search_cache=frame_search_cache,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )
        if match is not None:
            return match, "deprotonation"
        if log_callback is not None:
            log_callback(
                "no acceptable molecule fit was found after all tight and "
                "relaxed search passes."
            )
        return None, None

    def _find_variant_match(
        self,
        frame,
        used: Sequence[bool],
        molecule: ResolvedMoleculeMapping,
        *,
        pass_name: str,
        variants: Sequence[_ResolvedMoleculeVariant] | None = None,
        frame_search_cache: _FrameSearchCache | None = None,
        log_callback: Callable[[str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> _VariantMatch | None:
        _raise_if_cancelled(cancel_callback)
        scale = (
            molecule.tight_pass_scale
            if pass_name == "tight"
            else molecule.relaxed_pass_scale
        )
        best_match: _VariantMatch | None = None
        best_score: tuple[float, float, int] | None = None
        candidate_variants = (
            molecule.variants if variants is None else variants
        )
        for variant in candidate_variants:
            _raise_if_cancelled(cancel_callback)
            variant_definition = _scaled_variant_definition(variant, scale)
            if log_callback is not None:
                log_callback(
                    f"{pass_name} pass: scanning "
                    f"{len(variant_definition.preferred_anchor_indices) or len(variant_definition.resolved_anchor_indices)} "
                    "backbone pair(s)."
                )
            if variant_definition.resolved_anchor_indices:
                assignment = self._find_best_match(
                    frame,
                    used,
                    variant_definition,
                    frame_search_cache=frame_search_cache,
                    search_status_callback=lambda message, reference_name=variant_definition.reference_name: (
                        log_callback(
                            f"{pass_name} pass [{reference_name}]: {message}"
                        )
                        if log_callback is not None
                        else None
                    ),
                    cancel_callback=cancel_callback,
                )
            else:
                assignment = _find_anchorless_assignment(
                    frame,
                    used,
                    variant_definition,
                )
            if assignment is None:
                continue
            transformed_full_coordinates = _transform_full_reference(
                full_reference_atoms=molecule.reference_atoms,
                kept_full_indices=variant.kept_full_indices,
                assignment=assignment,
                frame=frame,
            )
            fit_rmsd = _fit_rmsd(
                transformed_full_coordinates=transformed_full_coordinates,
                assignment=assignment,
                kept_full_indices=variant.kept_full_indices,
                frame=frame,
            )
            mean_bond_deviation = _mean_bond_deviation(
                bonds=variant.variant_bonds,
                assignment=assignment,
                frame=frame,
            )
            score = (
                len(variant.missing_full_indices),
                float(fit_rmsd),
                float(mean_bond_deviation),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_match = _VariantMatch(
                    variant=variant,
                    assignment=assignment,
                    pass_name=pass_name,
                    fit_rmsd=float(fit_rmsd),
                    mean_bond_deviation=float(mean_bond_deviation),
                    transformed_full_coordinates=transformed_full_coordinates,
                )
        if best_match is None and log_callback is not None:
            log_callback(
                f"{pass_name} pass: no acceptable molecule fit was found."
            )
        return best_match


def _validated_residue_name(value: str | None) -> str:
    residue_name = _normalized_residue_name(value, fallback="UNK")
    if not re.fullmatch(r"[A-Z]{3}", residue_name):
        raise ValueError(
            "Residue identifiers must be exactly three capital letters."
        )
    return residue_name


def _normalized_pbc_params(
    pbc_params: dict[str, float | str] | None,
) -> dict[str, float | str]:
    if pbc_params is None:
        return {}
    parsed: dict[str, float | str] = {}
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        if key in pbc_params and pbc_params[key] is not None:
            parsed[key] = float(pbc_params[key])
    if "space_group" in pbc_params and pbc_params["space_group"] is not None:
        parsed["space_group"] = str(pbc_params["space_group"])
    return parsed


def _header_lines_from_pbc(
    pbc_params: dict[str, float | str],
) -> list[str] | None:
    cryst1_line = _build_cryst1_line(pbc_params)
    if cryst1_line is None:
        return None
    return [cryst1_line]


def _find_anchorless_assignment(
    frame,
    used: Sequence[bool],
    molecule_definition: MoleculeDefinition,
) -> tuple[int, ...] | None:
    if len(molecule_definition.reference_atoms) != 1:
        return None
    reference_atom = molecule_definition.reference_atoms[0]
    for source_index, source_atom in enumerate(frame.atoms):
        if used[source_index]:
            continue
        if source_atom.element != reference_atom.element:
            continue
        return (source_index,)
    return None


def _enumerate_estimate_solutions(
    *,
    total_counts: dict[str, int],
    plan: XYZToPDBMappingPlan,
    max_solutions: int,
) -> tuple[list[XYZToPDBEstimateSolution], bool]:
    order = sorted(
        range(len(plan.molecules)),
        key=lambda index: (
            -sum(plan.molecules[index].element_counts.values()),
            plan.molecules[index].residue_name,
        ),
    )
    reordered_molecules = [plan.molecules[index] for index in order]
    remaining = Counter(
        {key: int(value) for key, value in total_counts.items()}
    )
    solutions: list[XYZToPDBEstimateSolution] = []
    seen: set[tuple[int, ...]] = set()
    truncated = False

    def recurse(
        position: int,
        current_counts: list[int],
        remaining_counts: Counter[str],
    ) -> None:
        nonlocal truncated
        if len(solutions) >= max_solutions:
            truncated = True
            return
        if position >= len(reordered_molecules):
            restored_counts = [0] * len(plan.molecules)
            for reorder_index, molecule_index in enumerate(order):
                restored_counts[molecule_index] = current_counts[reorder_index]
            key = tuple(restored_counts)
            if key in seen:
                return
            seen.add(key)
            free_atom_counts = {
                element: int(remaining_counts.get(element, 0))
                for element in sorted(plan.free_atoms)
                if remaining_counts.get(element, 0) > 0
            }
            unassigned_counts = {
                element: int(count)
                for element, count in sorted(remaining_counts.items())
                if count > 0 and element not in plan.free_atoms
            }
            assigned_atoms = int(
                sum(
                    restored_counts[index]
                    * sum(plan.molecules[index].element_counts.values())
                    for index in range(len(plan.molecules))
                )
                + sum(free_atom_counts.values())
            )
            solutions.append(
                XYZToPDBEstimateSolution(
                    molecule_counts=tuple(restored_counts),
                    free_atom_counts=free_atom_counts,
                    unassigned_counts=unassigned_counts,
                    assigned_atoms=assigned_atoms,
                    total_atoms=int(sum(total_counts.values())),
                )
            )
            return

        molecule = reordered_molecules[position]
        molecule_counts = molecule.element_counts
        max_count = min(
            (
                remaining_counts[element] // count
                for element, count in molecule_counts.items()
                if count > 0
            ),
            default=0,
        )
        for candidate_count in range(int(max_count), -1, -1):
            next_remaining = remaining_counts.copy()
            valid = True
            for element, count in molecule_counts.items():
                next_remaining[element] -= int(candidate_count) * int(count)
                if next_remaining[element] < 0:
                    valid = False
                    break
            if not valid:
                continue
            recurse(
                position + 1,
                current_counts + [int(candidate_count)],
                next_remaining,
            )
            if truncated and len(solutions) >= max_solutions:
                return

    recurse(0, [], remaining)
    solutions.sort(
        key=lambda solution: (
            solution.unassigned_total,
            -solution.assigned_atoms,
            tuple(-count for count in solution.molecule_counts),
        )
    )
    if solutions:
        best_unassigned_total = solutions[0].unassigned_total
        solutions = [
            solution
            for solution in solutions
            if solution.unassigned_total == best_unassigned_total
        ]
    return solutions, truncated


def _select_solution(
    estimate: XYZToPDBEstimateResult,
    selected_solution_index: int,
) -> XYZToPDBEstimateSolution | None:
    if not estimate.solutions:
        return None
    if selected_solution_index < 0 or selected_solution_index >= len(
        estimate.solutions
    ):
        raise ValueError(
            f"Estimate solution index {selected_solution_index} is out of range."
        )
    return estimate.solutions[selected_solution_index]


def _template_mapping_targets(
    plan: XYZToPDBMappingPlan,
    solution: XYZToPDBEstimateSolution | None,
) -> tuple[tuple[str, int], ...]:
    if solution is None:
        return ()
    targets: list[tuple[str, int]] = []
    for index, molecule in enumerate(plan.molecules):
        expected_count = int(solution.molecule_counts[index])
        if expected_count <= 0:
            continue
        targets.append((molecule.residue_name, expected_count))
    return tuple(targets)


def _format_template_mapping_status(
    targets: Sequence[tuple[str, int]],
    current_counts: dict[str, int],
) -> str:
    return ", ".join(
        f"{residue_name} {int(current_counts.get(residue_name, 0))}/{expected_count}"
        for residue_name, expected_count in targets
    )


def _absolute_tolerance_from_percent(
    reference_length: float,
    tolerance_percent: float,
) -> float:
    return (
        max(float(reference_length), 0.0)
        * max(float(tolerance_percent), 0.0)
        / 100.0
    )


def _tolerance_percent_from_absolute(
    reference_length: float,
    absolute_tolerance: float,
) -> float:
    reference_length = float(reference_length)
    if reference_length <= 0.0:
        return 0.0
    return max(float(absolute_tolerance), 0.0) * 100.0 / reference_length


def _infer_direct_reference_bonds(
    atoms: Sequence[PDBAtom],
    *,
    default_tolerance: float = _REFERENCE_BOND_TOLERANCE,
) -> tuple[ResolvedReferenceBond, ...]:
    candidates: list[ResolvedReferenceBond] = []
    coordinates = np.array([atom.coordinates for atom in atoms], dtype=float)
    for atom1_index, atom1 in enumerate(atoms):
        for atom2_index in range(atom1_index + 1, len(atoms)):
            atom2 = atoms[atom2_index]
            distance = float(
                np.linalg.norm(
                    coordinates[atom2_index] - coordinates[atom1_index]
                )
            )
            if distance <= 0.0:
                continue
            radius1 = _COVALENT_RADII.get(atom1.element, 0.85)
            radius2 = _COVALENT_RADII.get(atom2.element, 0.85)
            threshold = _REFERENCE_BOND_THRESHOLD_SCALE * (radius1 + radius2)
            if distance > threshold:
                continue
            candidates.append(
                ResolvedReferenceBond(
                    atom1_index=atom1_index,
                    atom2_index=atom2_index,
                    atom1_name=_normalized_atom_name(
                        atom1.atom_name,
                        fallback=f"{atom1.element}{atom1_index + 1}",
                    ),
                    atom2_name=_normalized_atom_name(
                        atom2.atom_name,
                        fallback=f"{atom2.element}{atom2_index + 1}",
                    ),
                    reference_length=distance,
                    tolerance=float(default_tolerance),
                )
            )

    hydrogen_best: dict[int, ResolvedReferenceBond] = {}
    retained: list[ResolvedReferenceBond] = []
    for bond in candidates:
        if atoms[bond.atom1_index].element == "H":
            current = hydrogen_best.get(bond.atom1_index)
            if (
                current is None
                or bond.reference_length < current.reference_length
            ):
                hydrogen_best[bond.atom1_index] = bond
            continue
        if atoms[bond.atom2_index].element == "H":
            current = hydrogen_best.get(bond.atom2_index)
            if (
                current is None
                or bond.reference_length < current.reference_length
            ):
                hydrogen_best[bond.atom2_index] = bond
            continue
        retained.append(bond)
    retained.extend(hydrogen_best.values())
    retained.sort(
        key=lambda bond: (
            bond.atom1_index,
            bond.atom2_index,
            bond.atom1_name,
            bond.atom2_name,
        )
    )
    return tuple(retained)


def _resolve_bond_inputs(
    atoms: Sequence[PDBAtom],
    bond_inputs: Sequence[ReferenceBondToleranceInput],
) -> tuple[ResolvedReferenceBond, ...]:
    atom_names = [
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        )
        for index, atom in enumerate(atoms)
    ]
    atom_lookup = {name: index for index, name in enumerate(atom_names)}
    inferred = _infer_direct_reference_bonds(atoms)
    inferred_lookup = {
        tuple(sorted((bond.atom1_name, bond.atom2_name))): bond
        for bond in inferred
    }
    if not bond_inputs:
        return inferred
    resolved: list[ResolvedReferenceBond] = []
    for item in bond_inputs:
        atom1_name = _normalized_atom_name(item.atom1_name, fallback="A1")
        atom2_name = _normalized_atom_name(item.atom2_name, fallback="A2")
        if atom1_name not in atom_lookup or atom2_name not in atom_lookup:
            raise ValueError(
                f"Bond ({atom1_name}, {atom2_name}) was not found in the reference molecule."
            )
        lookup_key = tuple(sorted((atom1_name, atom2_name)))
        inferred_bond = inferred_lookup.get(lookup_key)
        atom1_index = atom_lookup[atom1_name]
        atom2_index = atom_lookup[atom2_name]
        reference_length = (
            inferred_bond.reference_length
            if inferred_bond is not None
            else float(
                np.linalg.norm(
                    np.asarray(atoms[atom2_index].coordinates, dtype=float)
                    - np.asarray(atoms[atom1_index].coordinates, dtype=float)
                )
            )
        )
        resolved.append(
            ResolvedReferenceBond(
                atom1_index=atom1_index,
                atom2_index=atom2_index,
                atom1_name=atom1_name,
                atom2_name=atom2_name,
                reference_length=reference_length,
                tolerance=_absolute_tolerance_from_percent(
                    reference_length,
                    float(item.tolerance),
                ),
            )
        )
    resolved.sort(key=lambda bond: (bond.atom1_index, bond.atom2_index))
    return tuple(resolved)


def _build_variants(
    *,
    reference_name: str,
    reference_path: Path,
    residue_name: str,
    atoms: Sequence[PDBAtom],
    bonds: Sequence[ResolvedReferenceBond],
    preferred_backbone_pairs: Sequence[tuple[str, str]],
    max_assignment_distance: float | None,
    max_missing_hydrogens: int,
) -> tuple[_ResolvedMoleculeVariant, ...]:
    full_variant = _build_variant(
        reference_name=reference_name,
        reference_path=reference_path,
        residue_name=residue_name,
        full_atoms=atoms,
        full_bonds=bonds,
        preferred_backbone_pairs=preferred_backbone_pairs,
        kept_full_indices=tuple(range(len(atoms))),
        max_assignment_distance=max_assignment_distance,
    )
    variants = [full_variant]
    hydrogen_indices = [
        index for index, atom in enumerate(atoms) if atom.element == "H"
    ]
    for missing_count in range(
        1,
        min(len(hydrogen_indices), max_missing_hydrogens) + 1,
    ):
        for missing_indices in itertools.combinations(
            hydrogen_indices,
            missing_count,
        ):
            kept_full_indices = tuple(
                index
                for index in range(len(atoms))
                if index not in set(missing_indices)
            )
            variants.append(
                _build_variant(
                    reference_name=reference_name,
                    reference_path=reference_path,
                    residue_name=residue_name,
                    full_atoms=atoms,
                    full_bonds=bonds,
                    preferred_backbone_pairs=preferred_backbone_pairs,
                    kept_full_indices=kept_full_indices,
                    max_assignment_distance=max_assignment_distance,
                )
            )
    variants.sort(
        key=lambda variant: (
            len(variant.missing_full_indices),
            len(variant.variant_reference_atoms),
        )
    )
    return tuple(variants)


def _build_variant(
    *,
    reference_name: str,
    reference_path: Path,
    residue_name: str,
    full_atoms: Sequence[PDBAtom],
    full_bonds: Sequence[ResolvedReferenceBond],
    preferred_backbone_pairs: Sequence[tuple[str, str]],
    kept_full_indices: tuple[int, ...],
    max_assignment_distance: float | None,
) -> _ResolvedMoleculeVariant:
    full_to_variant = {
        full_index: variant_index
        for variant_index, full_index in enumerate(kept_full_indices)
    }
    variant_atoms = tuple(
        full_atoms[index].copy() for index in kept_full_indices
    )
    variant_bonds = tuple(
        ResolvedReferenceBond(
            atom1_index=full_to_variant[bond.atom1_index],
            atom2_index=full_to_variant[bond.atom2_index],
            atom1_name=bond.atom1_name,
            atom2_name=bond.atom2_name,
            reference_length=bond.reference_length,
            tolerance=bond.tolerance,
        )
        for bond in full_bonds
        if bond.atom1_index in full_to_variant
        and bond.atom2_index in full_to_variant
    )
    preferred_anchors = _resolve_variant_preferred_anchors(
        variant_atoms,
        variant_bonds,
        preferred_backbone_pairs=preferred_backbone_pairs,
    )
    anchor_items: list[tuple[str, str, int, int, float]] = []
    seen_anchor_pairs: set[tuple[int, int]] = set()
    for (
        atom1_name,
        atom2_name,
        atom1_index,
        atom2_index,
        tolerance,
    ) in preferred_anchors:
        pair_key = tuple(sorted((atom1_index, atom2_index)))
        if pair_key in seen_anchor_pairs:
            continue
        seen_anchor_pairs.add(pair_key)
        anchor_items.append(
            (
                atom1_name,
                atom2_name,
                atom1_index,
                atom2_index,
                tolerance,
            )
        )
    for bond in variant_bonds:
        pair_key = tuple(sorted((bond.atom1_index, bond.atom2_index)))
        if pair_key in seen_anchor_pairs:
            continue
        seen_anchor_pairs.add(pair_key)
        anchor_items.append(
            (
                bond.atom1_name,
                bond.atom2_name,
                bond.atom1_index,
                bond.atom2_index,
                bond.tolerance,
            )
        )
    anchors = tuple(
        AnchorPairDefinition(
            atom1_name=atom1_name,
            atom2_name=atom2_name,
            tolerance=tolerance,
        )
        for atom1_name, atom2_name, _atom1_index, _atom2_index, tolerance in anchor_items
    )
    resolved_anchor_indices = tuple(
        (atom1_index, atom2_index, tolerance)
        for _atom1_name, _atom2_name, atom1_index, atom2_index, tolerance in anchor_items
    )
    preferred_anchor_indices = tuple(
        (atom1_index, atom2_index)
        for _atom1_name, _atom2_name, atom1_index, atom2_index, _tolerance in preferred_anchors
    )
    molecule_definition = MoleculeDefinition(
        name=residue_name,
        reference_name=reference_name,
        reference_path=reference_path,
        residue_name=residue_name,
        reference_atoms=variant_atoms,
        anchors=anchors,
        resolved_anchor_indices=resolved_anchor_indices,
        preferred_anchor_indices=preferred_anchor_indices,
        max_assignment_distance=max_assignment_distance,
    )
    missing_full_indices = tuple(
        index
        for index in range(len(full_atoms))
        if index not in set(kept_full_indices)
    )
    return _ResolvedMoleculeVariant(
        molecule_definition=molecule_definition,
        variant_reference_atoms=variant_atoms,
        variant_bonds=variant_bonds,
        kept_full_indices=kept_full_indices,
        missing_full_indices=missing_full_indices,
    )


def _scaled_variant_definition(
    variant: _ResolvedMoleculeVariant,
    scale: float,
) -> MoleculeDefinition:
    anchors = tuple(
        AnchorPairDefinition(
            atom1_name=anchor.atom1_name,
            atom2_name=anchor.atom2_name,
            tolerance=anchor.tolerance * scale,
        )
        for anchor in variant.molecule_definition.anchors
    )
    resolved_anchor_indices = tuple(
        (index1, index2, tolerance * scale)
        for index1, index2, tolerance in variant.molecule_definition.resolved_anchor_indices
    )
    return MoleculeDefinition(
        name=variant.molecule_definition.name,
        reference_name=variant.molecule_definition.reference_name,
        reference_path=variant.molecule_definition.reference_path,
        residue_name=variant.molecule_definition.residue_name,
        reference_atoms=variant.molecule_definition.reference_atoms,
        anchors=anchors,
        resolved_anchor_indices=resolved_anchor_indices,
        preferred_anchor_indices=tuple(
            variant.molecule_definition.preferred_anchor_indices
        ),
        max_assignment_distance=variant.molecule_definition.max_assignment_distance,
    )


def _resolve_variant_preferred_anchors(
    atoms: Sequence[PDBAtom],
    bonds: Sequence[ResolvedReferenceBond],
    *,
    preferred_backbone_pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str, int, int, float], ...]:
    atom_lookup = {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): index
        for index, atom in enumerate(atoms)
    }
    bond_lookup = {
        tuple(sorted((bond.atom1_name, bond.atom2_name))): bond
        for bond in bonds
    }
    resolved_pairs: list[tuple[str, str, int, int, float]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for atom1_name, atom2_name in preferred_backbone_pairs:
        normalized_atom1 = _normalized_atom_name(atom1_name, fallback="A1")
        normalized_atom2 = _normalized_atom_name(atom2_name, fallback="A2")
        if (
            normalized_atom1 not in atom_lookup
            or normalized_atom2 not in atom_lookup
        ):
            continue
        atom1_index = atom_lookup[normalized_atom1]
        atom2_index = atom_lookup[normalized_atom2]
        pair_key = tuple(sorted((atom1_index, atom2_index)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        bond = bond_lookup.get(
            tuple(sorted((normalized_atom1, normalized_atom2)))
        )
        if bond is not None:
            tolerance = float(bond.tolerance)
        else:
            reference_length = float(
                np.linalg.norm(
                    np.asarray(atoms[atom2_index].coordinates, dtype=float)
                    - np.asarray(atoms[atom1_index].coordinates, dtype=float)
                )
            )
            tolerance = _preferred_backbone_tolerance(
                reference_length=reference_length,
                bonds=bonds,
            )
        resolved_pairs.append(
            (
                normalized_atom1,
                normalized_atom2,
                atom1_index,
                atom2_index,
                tolerance,
            )
        )

    return tuple(resolved_pairs)


def _preferred_backbone_tolerance(
    *,
    reference_length: float,
    bonds: Sequence[ResolvedReferenceBond],
) -> float:
    tolerance_percents = [
        _tolerance_percent_from_absolute(
            bond.reference_length,
            bond.tolerance,
        )
        for bond in bonds
        if bond.reference_length > 0.0
    ]
    tolerance_percent = (
        float(np.mean(tolerance_percents))
        if tolerance_percents
        else _tolerance_percent_from_absolute(1.5, _REFERENCE_BOND_TOLERANCE)
    )
    return _absolute_tolerance_from_percent(
        reference_length,
        tolerance_percent,
    )


def _transform_full_reference(
    *,
    full_reference_atoms: Sequence[PDBAtom],
    kept_full_indices: Sequence[int],
    assignment: Sequence[int],
    frame,
) -> np.ndarray:
    full_reference_coordinates = np.array(
        [atom.coordinates for atom in full_reference_atoms],
        dtype=float,
    )
    kept_reference_coordinates = np.array(
        [full_reference_coordinates[index] for index in kept_full_indices],
        dtype=float,
    )
    target_coordinates = np.array(
        [frame.atoms[source_index].coordinates for source_index in assignment],
        dtype=float,
    )
    if len(kept_full_indices) >= 3:
        rotation, source_centroid, target_centroid = (
            rigid_alignment_from_points(
                kept_reference_coordinates,
                target_coordinates,
            )
        )
        return (
            (full_reference_coordinates - source_centroid) @ rotation
        ) + target_centroid
    if len(kept_full_indices) == 2:
        source_vector = (
            kept_reference_coordinates[1] - kept_reference_coordinates[0]
        )
        target_vector = target_coordinates[1] - target_coordinates[0]
        rotation = rotation_matrix_from_to(source_vector, target_vector)
        return (
            (full_reference_coordinates - kept_reference_coordinates[0])
            @ rotation.T
        ) + target_coordinates[0]
    if len(kept_full_indices) == 1:
        shift = target_coordinates[0] - kept_reference_coordinates[0]
        return full_reference_coordinates + shift
    return full_reference_coordinates.copy()


def _fit_rmsd(
    *,
    transformed_full_coordinates: np.ndarray,
    assignment: Sequence[int],
    kept_full_indices: Sequence[int],
    frame,
) -> float:
    if not assignment:
        return 0.0
    squared_distances = []
    for full_index, source_index in zip(kept_full_indices, assignment):
        delta = (
            transformed_full_coordinates[full_index]
            - frame.atoms[source_index].coordinates
        )
        squared_distances.append(float(np.dot(delta, delta)))
    if not squared_distances:
        return 0.0
    return float(np.sqrt(np.mean(squared_distances)))


def _mean_bond_deviation(
    *,
    bonds: Sequence[ResolvedReferenceBond],
    assignment: Sequence[int],
    frame,
) -> float:
    if not bonds:
        return 0.0
    deviations: list[float] = []
    for bond in bonds:
        source_index1 = assignment[bond.atom1_index]
        source_index2 = assignment[bond.atom2_index]
        distance = float(
            np.linalg.norm(
                frame.atoms[source_index2].coordinates
                - frame.atoms[source_index1].coordinates
            )
        )
        deviations.append(abs(distance - bond.reference_length))
    return float(np.mean(deviations)) if deviations else 0.0


def _materialize_match(
    *,
    match: _VariantMatch,
    frame,
    residue_number: int,
    atom_serial: int,
    hydrogen_mode: str,
    used: Sequence[bool],
    match_classification: str | None,
) -> tuple[ConvertedResidue, set[int], list[str], list[str], int]:
    molecule = match.variant.molecule_definition
    kept_full_indices = match.variant.kept_full_indices
    missing_full_indices = match.variant.missing_full_indices
    assignment_lookup = {
        full_index: source_index
        for full_index, source_index in zip(
            kept_full_indices, match.assignment
        )
    }
    used_indices = set(assignment_lookup.values())
    classification_note = ""
    if match_classification == "tolerance":
        classification_note = (
            " Full-hydrogen matching only succeeded after widening the bond "
            "tolerances."
        )
    elif match_classification == "deprotonation":
        classification_note = (
            " Matched after omitting "
            f"{len(missing_full_indices)} reference hydrogen(s), which is "
            "consistent with deprotonation."
        )
    messages: list[str] = [
        f"{molecule.residue_name}: matched in the {match.pass_name} pass "
        f"(RMSD {match.fit_rmsd:.3f} A, bond delta "
        f"{match.mean_bond_deviation:.3f} A).{classification_note}"
    ]
    warnings: list[str] = []

    orphan_hydrogens = [
        index
        for index, atom in enumerate(frame.atoms)
        if atom.element == "H"
        and not used[index]
        and index not in used_indices
    ]
    residue_atoms: list[PDBAtom] = []
    residue_source_indices: list[int] = []
    full_reference_lookup = {
        full_index: atom
        for full_index, atom in enumerate(
            _expand_full_reference_atoms(
                match.variant,
                match.transformed_full_coordinates,
            )
        )
    }
    for full_index in range(len(full_reference_lookup)):
        reference_atom = full_reference_lookup[full_index]
        if full_index in assignment_lookup:
            source_index = assignment_lookup[full_index]
            xyz_atom = frame.atoms[source_index]
            coordinates = xyz_atom.coordinates.copy()
            used_indices.add(source_index)
            residue_source_indices.append(int(source_index))
        elif reference_atom.element == "H":
            predicted = match.transformed_full_coordinates[full_index].copy()
            if hydrogen_mode == "assign_orphaned" and orphan_hydrogens:
                source_index = _nearest_source_index(
                    orphan_hydrogens,
                    frame,
                    predicted,
                )
                orphan_hydrogens.remove(source_index)
                used_indices.add(source_index)
                coordinates = frame.atoms[source_index].coordinates.copy()
                residue_source_indices.append(int(source_index))
                messages.append(
                    f"{molecule.residue_name}: reassigned an orphaned hydrogen "
                    "to a deprotonated site."
                )
            elif hydrogen_mode == "restore_missing" and orphan_hydrogens:
                source_index = _nearest_source_index(
                    orphan_hydrogens,
                    frame,
                    predicted,
                )
                orphan_hydrogens.remove(source_index)
                used_indices.add(source_index)
                coordinates = predicted
                residue_source_indices.append(int(source_index))
                messages.append(
                    f"{molecule.residue_name}: restored a missing hydrogen at "
                    "the reference-aligned position."
                )
            else:
                warnings.append(
                    f"{molecule.residue_name}: a hydrogen could not be placed "
                    "for one matched molecule."
                )
                continue
        else:
            continue

        residue_atoms.append(
            PDBAtom(
                atom_id=atom_serial,
                atom_name=_normalized_atom_name(
                    reference_atom.atom_name,
                    fallback=f"{reference_atom.element}{atom_serial}",
                ),
                residue_name=molecule.residue_name,
                residue_number=residue_number,
                coordinates=coordinates,
                element=reference_atom.element,
            )
        )
        atom_serial += 1

    residue = ConvertedResidue(
        residue_number=residue_number,
        residue_name=molecule.residue_name,
        molecule_name=molecule.residue_name,
        atoms=tuple(residue_atoms),
        source_atom_indices=tuple(residue_source_indices),
    )
    return residue, used_indices, messages, warnings, atom_serial


def _expand_full_reference_atoms(
    variant: _ResolvedMoleculeVariant,
    transformed_full_coordinates: np.ndarray,
) -> tuple[PDBAtom, ...]:
    full_atoms: list[PDBAtom] = []
    kept_lookup = {
        full_index: atom
        for full_index, atom in zip(
            variant.kept_full_indices,
            variant.variant_reference_atoms,
        )
    }
    for full_index in range(len(transformed_full_coordinates)):
        atom = kept_lookup.get(full_index)
        if atom is None:
            atom = PDBAtom(
                atom_id=full_index + 1,
                atom_name=f"H{full_index + 1}",
                residue_name=variant.molecule_definition.residue_name,
                residue_number=1,
                coordinates=transformed_full_coordinates[full_index].copy(),
                element="H",
            )
        copied = atom.copy()
        copied.coordinates = transformed_full_coordinates[full_index].copy()
        full_atoms.append(copied)
    return tuple(full_atoms)


def _nearest_source_index(
    candidate_indices: Sequence[int],
    frame,
    target_coordinates: np.ndarray,
) -> int:
    return min(
        candidate_indices,
        key=lambda index: float(
            np.linalg.norm(frame.atoms[index].coordinates - target_coordinates)
        ),
    )


def _apply_template_mapping(
    frame,
    template_residues: Sequence[ConvertedResidue],
) -> list[ConvertedResidue]:
    residues: list[ConvertedResidue] = []
    for residue in template_residues:
        copied_atoms: list[PDBAtom] = []
        for template_atom, source_index in zip(
            residue.atoms,
            residue.source_atom_indices,
        ):
            if source_index >= len(frame.atoms):
                raise ValueError(
                    f"Frame {frame.filepath.name} does not contain atom index {source_index}."
                )
            xyz_atom = frame.atoms[source_index]
            if xyz_atom.element != template_atom.element:
                raise ValueError(
                    _element_order_mismatch_message(
                        frame_name=frame.filepath.name,
                        atom_index=source_index + 1,
                        expected_element=template_atom.element,
                        found_element=xyz_atom.element,
                    )
                )
            copied_atom = template_atom.copy()
            copied_atom.coordinates = xyz_atom.coordinates.copy()
            copied_atoms.append(copied_atom)
        residues.append(
            ConvertedResidue(
                residue_number=residue.residue_number,
                residue_name=residue.residue_name,
                molecule_name=residue.molecule_name,
                atoms=tuple(copied_atoms),
                source_atom_indices=tuple(residue.source_atom_indices),
            )
        )
    return residues


def _write_assertion_molecules_for_frame(
    *,
    frame_stem: str,
    residues: Sequence[ConvertedResidue],
    plan: XYZToPDBMappingPlan,
    assertion_dir: Path,
) -> list[_MoleculeAssertionEntry]:
    reference_by_residue = {
        molecule.residue_name: molecule for molecule in plan.molecules
    }
    entries: list[_MoleculeAssertionEntry] = []
    for residue in residues:
        reference = reference_by_residue.get(residue.residue_name)
        if reference is None:
            continue
        residue_dir = assertion_dir / residue.residue_name
        residue_dir.mkdir(parents=True, exist_ok=True)
        molecule_file = residue_dir / (
            f"{frame_stem}__{residue.residue_name}_{int(residue.residue_number):04d}.pdb"
        )
        PDBStructure(
            atoms=[atom.copy() for atom in residue.atoms],
            source_name=molecule_file.stem,
        ).write_pdb_file(molecule_file)
        (
            common_atom_count,
            distance_pair_count,
            distribution_rmsd,
            max_distance_delta,
            missing_atom_names,
        ) = _molecule_distribution_metrics(
            residue=residue,
            reference=reference,
        )
        entries.append(
            _MoleculeAssertionEntry(
                residue_name=residue.residue_name,
                molecule_file=molecule_file,
                common_atom_count=common_atom_count,
                distance_pair_count=distance_pair_count,
                distribution_rmsd=distribution_rmsd,
                max_distance_delta=max_distance_delta,
                missing_atom_names=missing_atom_names,
            )
        )
    return entries


def _molecule_distribution_metrics(
    *,
    residue: ConvertedResidue,
    reference: ResolvedMoleculeMapping,
) -> tuple[int, int, float, float, tuple[str, ...]]:
    reference_lookup = {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): np.asarray(atom.coordinates, dtype=float)
        for index, atom in enumerate(reference.reference_atoms)
    }
    residue_lookup = {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): np.asarray(atom.coordinates, dtype=float)
        for index, atom in enumerate(residue.atoms)
    }
    common_atom_names = tuple(
        atom_name
        for atom_name in reference_lookup
        if atom_name in residue_lookup
    )
    missing_atom_names = tuple(
        atom_name
        for atom_name in reference_lookup
        if atom_name not in residue_lookup
    )
    if len(common_atom_names) < 2:
        return (
            len(common_atom_names),
            0,
            0.0,
            0.0,
            missing_atom_names,
        )

    reference_distances = _pairwise_distance_vector(
        [reference_lookup[name] for name in common_atom_names]
    )
    residue_distances = _pairwise_distance_vector(
        [residue_lookup[name] for name in common_atom_names]
    )
    distance_deltas = residue_distances - reference_distances
    return (
        len(common_atom_names),
        int(len(reference_distances)),
        float(np.sqrt(np.mean(np.square(distance_deltas)))),
        float(np.max(np.abs(distance_deltas))),
        missing_atom_names,
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


def _build_assertion_report(
    *,
    plan: XYZToPDBMappingPlan,
    assertion_dir: Path,
    entries: Sequence[_MoleculeAssertionEntry],
) -> XYZToPDBAssertionResult:
    entries_by_residue: dict[str, list[_MoleculeAssertionEntry]] = defaultdict(
        list
    )
    for entry in entries:
        entries_by_residue[entry.residue_name].append(entry)

    residue_summaries: list[XYZToPDBAssertionResidueSummary] = []
    reference_update_candidates: list[XYZToPDBReferenceUpdateCandidate] = []
    warnings: list[str] = []
    for molecule in plan.molecules:
        residue_entries = entries_by_residue.get(molecule.residue_name, [])
        summary, residue_warnings = _assertion_summary_for_residue(
            residue_name=molecule.residue_name,
            entries=residue_entries,
        )
        residue_summaries.append(summary)
        warnings.extend(residue_warnings)
        if summary.passed and residue_entries:
            candidate = _build_reference_update_candidate(
                molecule=molecule,
                summary=summary,
                residue_entries=residue_entries,
                assertion_dir=assertion_dir,
            )
            if candidate is not None:
                reference_update_candidates.append(candidate)

    report_lines = [
        "xyz2pdb assertion mode report",
        f"Molecule folder: {assertion_dir}",
        f"Total molecules checked: {len(entries)}",
    ]
    for summary in residue_summaries:
        status = "PASS" if summary.passed else "WARN"
        report_lines.append(
            f"{status} {summary.residue_name}: "
            f"{summary.molecule_count} molecule(s), median RMSD "
            f"{summary.median_distribution_rmsd:.3f} A, max RMSD "
            f"{summary.max_distribution_rmsd:.3f} A, median max-delta "
            f"{summary.median_max_distance_delta:.3f} A, max max-delta "
            f"{summary.max_max_distance_delta:.3f} A, outliers "
            f"{summary.outlier_count}."
        )
    if reference_update_candidates:
        report_lines.append("Reference update candidates:")
        report_lines.extend(
            (
                f"- {candidate.reference_name} ({candidate.residue_name}) -> "
                f"{candidate.average_structure_file}"
            )
            for candidate in reference_update_candidates
        )
    if warnings:
        report_lines.append("Warnings:")
        report_lines.extend(f"- {warning}" for warning in warnings)
    report_file = assertion_dir / "assertion_report.txt"
    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return XYZToPDBAssertionResult(
        molecule_dir=assertion_dir,
        report_file=report_file,
        total_molecules=len(entries),
        passed=not warnings,
        residue_summaries=tuple(residue_summaries),
        reference_update_candidates=tuple(reference_update_candidates),
        warnings=tuple(warnings),
    )


def _assertion_summary_for_residue(
    *,
    residue_name: str,
    entries: Sequence[_MoleculeAssertionEntry],
) -> tuple[XYZToPDBAssertionResidueSummary, list[str]]:
    if not entries:
        return (
            XYZToPDBAssertionResidueSummary(
                residue_name=residue_name,
                molecule_count=0,
                common_atom_count=0,
                distance_pair_count=0,
                median_distribution_rmsd=0.0,
                max_distribution_rmsd=0.0,
                median_max_distance_delta=0.0,
                max_max_distance_delta=0.0,
                outlier_count=0,
                passed=True,
            ),
            [],
        )

    rms_values = np.asarray(
        [entry.distribution_rmsd for entry in entries],
        dtype=float,
    )
    max_values = np.asarray(
        [entry.max_distance_delta for entry in entries],
        dtype=float,
    )
    median_rms = float(np.median(rms_values))
    median_max = float(np.median(max_values))
    max_rms = float(np.max(rms_values))
    max_max = float(np.max(max_values))
    outlier_rms_threshold = max(
        0.25,
        median_rms + (4.0 * _median_absolute_deviation(rms_values)),
    )
    outlier_max_threshold = max(
        0.60,
        median_max + (4.0 * _median_absolute_deviation(max_values)),
    )
    outlier_entries = [
        entry
        for entry in entries
        if entry.distribution_rmsd > outlier_rms_threshold
        or entry.max_distance_delta > outlier_max_threshold
    ]
    warnings: list[str] = []
    if median_rms > 0.20 or max_max > 0.60:
        warnings.append(
            f"{residue_name}: molecule distance distributions vary noticeably "
            f"from the reference (median RMSD {median_rms:.3f} A, max "
            f"distance delta {max_max:.3f} A)."
        )
    if outlier_entries:
        warnings.append(
            f"{residue_name}: {len(outlier_entries)} molecule(s) look skewed "
            "relative to the rest of the exported set."
        )
    missing_entries = [entry for entry in entries if entry.missing_atom_names]
    if missing_entries:
        warnings.append(
            f"{residue_name}: {len(missing_entries)} molecule(s) were missing "
            "reference atoms during assertion checks."
        )
    summary = XYZToPDBAssertionResidueSummary(
        residue_name=residue_name,
        molecule_count=len(entries),
        common_atom_count=max(entry.common_atom_count for entry in entries),
        distance_pair_count=max(
            entry.distance_pair_count for entry in entries
        ),
        median_distribution_rmsd=median_rms,
        max_distribution_rmsd=max_rms,
        median_max_distance_delta=median_max,
        max_max_distance_delta=max_max,
        outlier_count=len(outlier_entries),
        passed=not warnings,
    )
    if outlier_entries:
        warnings.append(
            f"{residue_name}: inspect files such as "
            + ", ".join(
                str(entry.molecule_file.name) for entry in outlier_entries[:3]
            )
            + "."
        )
    return summary, warnings


def _build_reference_update_candidate(
    *,
    molecule: ResolvedMoleculeMapping,
    summary: XYZToPDBAssertionResidueSummary,
    residue_entries: Sequence[_MoleculeAssertionEntry],
    assertion_dir: Path,
) -> XYZToPDBReferenceUpdateCandidate | None:
    if not residue_entries:
        return None
    candidate_dir = assertion_dir / "reference_update_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    average_structure_file = candidate_dir / (
        f"{molecule.reference_name}__assertion_average.pdb"
    )
    averaged_atoms = _average_assertion_residue_atoms(
        reference_atoms=molecule.reference_atoms,
        molecule_files=[entry.molecule_file for entry in residue_entries],
        residue_name=(
            molecule.reference_atoms[0].residue_name
            if molecule.reference_atoms
            else molecule.residue_name
        ),
    )
    PDBStructure(
        atoms=[atom.copy() for atom in averaged_atoms],
        source_name=average_structure_file.stem,
    ).write_pdb_file(average_structure_file)
    return XYZToPDBReferenceUpdateCandidate(
        residue_name=molecule.residue_name,
        reference_name=molecule.reference_name,
        reference_path=molecule.reference_path,
        reference_residue_name=(
            molecule.reference_atoms[0].residue_name
            if molecule.reference_atoms
            else molecule.residue_name
        ),
        average_structure_file=average_structure_file,
        molecule_count=summary.molecule_count,
        median_distribution_rmsd=summary.median_distribution_rmsd,
        max_distribution_rmsd=summary.max_distribution_rmsd,
        backbone_pairs=tuple(molecule.preferred_backbone_pairs),
    )


def _average_assertion_residue_atoms(
    *,
    reference_atoms: Sequence[PDBAtom],
    molecule_files: Sequence[Path],
    residue_name: str,
) -> tuple[PDBAtom, ...]:
    if not reference_atoms:
        return ()
    reference_lookup = {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): atom
        for index, atom in enumerate(reference_atoms)
    }
    aligned_coordinates: dict[str, list[np.ndarray]] = defaultdict(list)
    for molecule_file in molecule_files:
        structure = PDBStructure.from_file(molecule_file)
        aligned_lookup = _aligned_residue_coordinate_lookup(
            residue_atoms=structure.atoms,
            reference_atoms=reference_atoms,
        )
        for atom_name, coordinates in aligned_lookup.items():
            aligned_coordinates[atom_name].append(coordinates)

    averaged_atoms: list[PDBAtom] = []
    for index, reference_atom in enumerate(reference_atoms, start=1):
        atom_name = _normalized_atom_name(
            reference_atom.atom_name,
            fallback=f"{reference_atom.element}{index}",
        )
        coordinate_set = aligned_coordinates.get(atom_name)
        averaged = (
            np.mean(np.asarray(coordinate_set, dtype=float), axis=0)
            if coordinate_set
            else np.asarray(reference_atom.coordinates, dtype=float)
        )
        atom = reference_lookup[atom_name].copy()
        atom.atom_id = index
        atom.residue_number = 1
        atom.residue_name = residue_name
        atom.coordinates = np.asarray(averaged, dtype=float)
        averaged_atoms.append(atom)
    return tuple(averaged_atoms)


def _aligned_residue_coordinate_lookup(
    *,
    residue_atoms: Sequence[PDBAtom],
    reference_atoms: Sequence[PDBAtom],
) -> dict[str, np.ndarray]:
    residue_lookup = {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): atom
        for index, atom in enumerate(residue_atoms)
    }
    reference_names = [
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        )
        for index, atom in enumerate(reference_atoms)
    ]
    common_names = [
        atom_name
        for atom_name in reference_names
        if atom_name in residue_lookup
    ]
    if not common_names:
        return {}

    residue_coordinates = np.asarray(
        [atom.coordinates for atom in residue_atoms],
        dtype=float,
    )
    common_residue_coordinates = np.asarray(
        [residue_lookup[atom_name].coordinates for atom_name in common_names],
        dtype=float,
    )
    common_reference_coordinates = np.asarray(
        [
            reference_atoms[reference_names.index(atom_name)].coordinates
            for atom_name in common_names
        ],
        dtype=float,
    )
    transformed_coordinates = _align_coordinates_to_reference(
        coordinates=residue_coordinates,
        source_points=common_residue_coordinates,
        target_points=common_reference_coordinates,
    )
    return {
        _normalized_atom_name(
            atom.atom_name,
            fallback=f"{atom.element}{index + 1}",
        ): transformed_coordinates[index]
        for index, atom in enumerate(residue_atoms)
    }


def _align_coordinates_to_reference(
    *,
    coordinates: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    if len(source_points) >= 3:
        rotation, source_centroid, target_centroid = (
            rigid_alignment_from_points(
                source_points,
                target_points,
            )
        )
        return ((coordinates - source_centroid) @ rotation) + target_centroid
    if len(source_points) == 2:
        rotation = rotation_matrix_from_to(
            source_points[1] - source_points[0],
            target_points[1] - target_points[0],
        )
        return ((coordinates - source_points[0]) @ rotation.T) + target_points[
            0
        ]
    if len(source_points) == 1:
        return coordinates + (target_points[0] - source_points[0])
    return coordinates.copy()


def _median_absolute_deviation(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def reference_bond_tolerances(
    reference_name: str,
    *,
    library_dir: str | Path | None = None,
) -> tuple[ReferenceBondToleranceInput, ...]:
    reference_path = resolve_reference_path(
        reference_name,
        library_dir=library_dir,
    )
    atoms = tuple(PDBStructure.from_file(reference_path).atoms)
    return tuple(
        ReferenceBondToleranceInput(
            atom1_name=bond.atom1_name,
            atom2_name=bond.atom2_name,
            tolerance=_tolerance_percent_from_absolute(
                bond.reference_length,
                bond.tolerance,
            ),
        )
        for bond in _infer_direct_reference_bonds(atoms)
    )


__all__ = [
    "FreeAtomMappingInput",
    "MoleculeMappingInput",
    "ReferenceBondToleranceInput",
    "ResolvedMoleculeMapping",
    "ResolvedReferenceBond",
    "XYZToPDBEstimateResult",
    "XYZToPDBEstimateSolution",
    "XYZToPDBMappingPlan",
    "XYZToPDBMappingTestResult",
    "XYZToPDBMappingWorkflow",
    "XYZToPDBSampleAnalysis",
    "default_reference_library_dir",
    "list_reference_library",
    "reference_bond_tolerances",
]
