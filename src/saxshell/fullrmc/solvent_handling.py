from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from saxshell.fullrmc.representatives import (
    RepresentativeSelectionEntry,
    RepresentativeSelectionMetadata,
    representative_source_solvent_mode_to_variant,
    representative_structure_variant_path,
)
from saxshell.saxs.debye import load_structure_file
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import (
    default_reference_library_dir,
    list_reference_library,
    resolve_reference_path,
)

if TYPE_CHECKING:  # pragma: no cover
    from .project_loader import RMCDreamProjectSource
    from .solvent_shell_builder import SolventShellAnalysisResult

_ANCHOR_ELEMENT_PRIORITY = (
    "O",
    "N",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "C",
)
_SOLVENT_ROTATION_ANGLES_DEG = tuple(range(0, 360, 15))
_ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.90447,
    "Pb": 207.2,
}
_DEFAULT_REFERENCE_MATCH_TOLERANCE_A = 0.25
_REPRESENTATIVE_STRUCTURE_MODE_LABELS = {
    "source": "Selected representative source files",
    "no_solvent": "No solvent",
    "partial_solvent": "Partial solvent",
    "full_solvent": "Full solvent",
}


@dataclass(slots=True)
class SoluteAtomBuildSetting:
    coordination_center: bool = False
    target_coordination_number: float = 0.0
    director_distance_cutoff_a: float = 2.5

    def to_dict(self) -> dict[str, object]:
        return {
            "coordination_center": bool(self.coordination_center),
            "target_coordination_number": float(
                self.target_coordination_number
            ),
            "director_distance_cutoff_a": float(
                self.director_distance_cutoff_a
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "SoluteAtomBuildSetting":
        source = dict(payload or {})
        return cls(
            coordination_center=bool(source.get("coordination_center", False)),
            target_coordination_number=max(
                0.0,
                _float_value(source.get("target_coordination_number"), 0.0),
            ),
            director_distance_cutoff_a=max(
                0.0,
                _float_value(source.get("director_distance_cutoff_a"), 2.5),
            ),
        )


@dataclass(slots=True)
class SolventHandlingSettings:
    coordinated_solvent_mode: str = "automatic_detection"
    reference_source: str = "preset"
    preset_name: str = "dmf"
    custom_reference_path: str | None = None
    reference_match_tolerance_a: float = _DEFAULT_REFERENCE_MATCH_TOLERANCE_A
    director_atom_name: str | None = None
    minimum_solvent_atom_separation_a: float = 1.2
    solute_atom_settings: dict[str, SoluteAtomBuildSetting] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["solute_atom_settings"] = {
            str(element): setting.to_dict()
            for element, setting in sorted(self.solute_atom_settings.items())
        }
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "SolventHandlingSettings":
        source = dict(payload or {})
        return cls(
            coordinated_solvent_mode=str(
                source.get(
                    "coordinated_solvent_mode",
                    "automatic_detection",
                )
            ).strip()
            or "automatic_detection",
            reference_source=str(
                source.get("reference_source", "preset")
            ).strip()
            or "preset",
            preset_name=str(source.get("preset_name", "dmf")).strip() or "dmf",
            custom_reference_path=_optional_text(
                source.get("custom_reference_path")
            ),
            reference_match_tolerance_a=max(
                0.0,
                _float_value(
                    source.get("reference_match_tolerance_a"),
                    _DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
                ),
            ),
            director_atom_name=_optional_text(
                source.get("director_atom_name")
            ),
            minimum_solvent_atom_separation_a=max(
                0.0,
                _float_value(
                    source.get("minimum_solvent_atom_separation_a"),
                    1.2,
                ),
            ),
            solute_atom_settings={
                str(element): SoluteAtomBuildSetting.from_dict(dict(entry))
                for element, entry in (
                    dict(source.get("solute_atom_settings", {})).items()
                    if isinstance(source.get("solute_atom_settings"), dict)
                    else []
                )
                if isinstance(entry, dict)
            },
        )


@dataclass(slots=True)
class SolventHandlingEntry:
    structure: str
    motif: str
    param: str
    source_file: str
    no_solvent_pdb: str
    completed_pdb: str
    atom_count_no_solvent: int
    atom_count_completed: int
    solvent_atoms_added: int
    solvent_molecules_added: int
    solvent_mode: str
    completion_strategy: str
    heuristic_note: str
    detected_source_status: str = "unknown"
    detected_complete_solvent_count: int = 0
    detected_partial_solvent_count: int = 0
    source_input_format: str = ""
    matched_atom_count: int = 0
    unmatched_atom_count: int = 0
    solute_element_counts: dict[str, int] = field(default_factory=dict)
    analysis_summary: str = ""
    build_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["solute_element_counts"] = dict(
            sorted(self.solute_element_counts.items())
        )
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "SolventHandlingEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            source_file=str(payload.get("source_file", "")).strip(),
            no_solvent_pdb=str(payload.get("no_solvent_pdb", "")).strip(),
            completed_pdb=str(payload.get("completed_pdb", "")).strip(),
            atom_count_no_solvent=int(payload.get("atom_count_no_solvent", 0)),
            atom_count_completed=int(payload.get("atom_count_completed", 0)),
            solvent_atoms_added=int(payload.get("solvent_atoms_added", 0)),
            solvent_molecules_added=int(
                payload.get("solvent_molecules_added", 0)
            ),
            solvent_mode=str(payload.get("solvent_mode", "")).strip(),
            completion_strategy=str(
                payload.get("completion_strategy", "")
            ).strip(),
            heuristic_note=str(payload.get("heuristic_note", "")).strip(),
            detected_source_status=str(
                payload.get("detected_source_status", "unknown")
            ).strip()
            or "unknown",
            detected_complete_solvent_count=int(
                payload.get("detected_complete_solvent_count", 0)
            ),
            detected_partial_solvent_count=int(
                payload.get("detected_partial_solvent_count", 0)
            ),
            source_input_format=str(
                payload.get("source_input_format", "")
            ).strip(),
            matched_atom_count=int(payload.get("matched_atom_count", 0)),
            unmatched_atom_count=int(payload.get("unmatched_atom_count", 0)),
            solute_element_counts={
                str(element): int(count)
                for element, count in (
                    dict(payload.get("solute_element_counts", {})).items()
                    if isinstance(payload.get("solute_element_counts"), dict)
                    else []
                )
            },
            analysis_summary=str(payload.get("analysis_summary", "")).strip(),
            build_summary=str(payload.get("build_summary", "")).strip(),
        )

    @property
    def representative_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    @property
    def detected_source_status_text(self) -> str:
        return _solvent_state_text(self.detected_source_status)


@dataclass(slots=True)
class SolventHandlingMetadata:
    settings: SolventHandlingSettings
    reference_path: str
    reference_name: str
    reference_residue_name: str
    updated_at: str
    representative_selection_mode: str
    detected_distribution_status: str
    detected_distribution_note: str
    aggregate_solute_element_counts: dict[str, int]
    entries: list[SolventHandlingEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.settings.to_dict(),
            "reference_path": self.reference_path,
            "reference_name": self.reference_name,
            "reference_residue_name": self.reference_residue_name,
            "updated_at": self.updated_at,
            "representative_selection_mode": (
                self.representative_selection_mode
            ),
            "detected_distribution_status": self.detected_distribution_status,
            "detected_distribution_note": self.detected_distribution_note,
            "aggregate_solute_element_counts": dict(
                sorted(self.aggregate_solute_element_counts.items())
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "SolventHandlingMetadata | None":
        if not payload:
            return None
        return cls(
            settings=SolventHandlingSettings.from_dict(
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else None
            ),
            reference_path=str(payload.get("reference_path", "")).strip(),
            reference_name=str(payload.get("reference_name", "")).strip(),
            reference_residue_name=str(
                payload.get("reference_residue_name", "")
            ).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            representative_selection_mode=str(
                payload.get("representative_selection_mode", "")
            ).strip(),
            detected_distribution_status=str(
                payload.get("detected_distribution_status", "unknown")
            ).strip()
            or "unknown",
            detected_distribution_note=str(
                payload.get("detected_distribution_note", "")
            ).strip(),
            aggregate_solute_element_counts={
                str(element): int(count)
                for element, count in (
                    dict(
                        payload.get("aggregate_solute_element_counts", {})
                    ).items()
                    if isinstance(
                        payload.get("aggregate_solute_element_counts"),
                        dict,
                    )
                    else []
                )
            },
            entries=[
                SolventHandlingEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
        )

    def summary_text(self) -> str:
        active_mode = resolved_representative_structure_mode(
            representative_metadata=None,
            solvent_metadata=self,
        )
        lines = [
            f"Reference source: {self.settings.reference_source}",
            f"Reference molecule: {self.reference_name}",
            f"Reference residue: {self.reference_residue_name}",
            (
                "Active representative structure set: "
                f"{representative_structure_mode_label(active_mode)}"
            ),
            (
                "Reference match tolerance: "
                f"{self.settings.reference_match_tolerance_a:.3g} A"
            ),
            (
                "Director atom: "
                f"{self.settings.director_atom_name or 'auto-selected'}"
            ),
            (
                "Minimum solvent atom separation: "
                f"{self.settings.minimum_solvent_atom_separation_a:.3g} A"
            ),
            (
                "Detected representative distribution state: "
                f"{_solvent_state_text(self.detected_distribution_status)}"
            ),
            f"Saved at: {self.updated_at}",
            f"Representative entries exported: {len(self.entries)}",
        ]
        if self.detected_distribution_note:
            lines.append(self.detected_distribution_note)
        if self.aggregate_solute_element_counts:
            lines.append(
                "Recognized solute elements: "
                + ", ".join(
                    f"{element}:{count}"
                    for element, count in sorted(
                        self.aggregate_solute_element_counts.items()
                    )
                )
            )
        if self.entries:
            first = self.entries[0]
            lines.extend(
                [
                    "",
                    "Example exported representative:",
                    f"  {first.representative_label}",
                    (
                        "  detected source status: "
                        f"{first.detected_source_status_text}"
                    ),
                    f"  no-solvent PDB: {Path(first.no_solvent_pdb).name}",
                    f"  decorated PDB: {Path(first.completed_pdb).name}",
                    f"  solvent atoms added: {first.solvent_atoms_added}",
                    f"  strategy: {first.completion_strategy}",
                ]
            )
        return "\n".join(lines)


@dataclass(slots=True)
class RepresentativeSolventAnalysisEntry:
    structure: str
    motif: str
    param: str
    source_file: str
    source_status: str
    analysis_result: "SolventShellAnalysisResult"

    @property
    def representative_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    @property
    def source_status_text(self) -> str:
        return _solvent_state_text(self.source_status)


@dataclass(slots=True)
class RepresentativeSolventDistributionAnalysis:
    reference_name: str
    reference_path: str
    reference_residue_name: str
    representative_selection_mode: str
    match_tolerance_a: float
    distribution_status: str
    distribution_note: str
    aggregate_solute_element_counts: dict[str, int]
    entries: list[RepresentativeSolventAnalysisEntry]

    @property
    def build_required(self) -> bool:
        return self.distribution_status != "complete_solvent"

    def summary_text(self) -> str:
        lines = [
            f"Reference molecule: {self.reference_name}",
            f"Reference residue: {self.reference_residue_name}",
            f"Reference match tolerance: {self.match_tolerance_a:.3g} A",
            (
                "Detected representative distribution state: "
                f"{_solvent_state_text(self.distribution_status)}"
            ),
            f"Representative entries analyzed: {len(self.entries)}",
        ]
        if self.distribution_note:
            lines.append(self.distribution_note)
        if self.aggregate_solute_element_counts:
            lines.append(
                "Recognized solute elements: "
                + ", ".join(
                    f"{element}:{count}"
                    for element, count in sorted(
                        self.aggregate_solute_element_counts.items()
                    )
                )
            )
        if self.entries:
            lines.extend(["", "Detected representative states:"])
            for entry in self.entries:
                lines.append(
                    f"  {entry.representative_label}: "
                    f"{entry.source_status_text}"
                )
        return "\n".join(lines)


@dataclass(slots=True)
class GeneratedPDBResidueSummary:
    residue_name: str
    residue_number: int
    atom_count: int
    element_counts: dict[str, int]

    @property
    def element_counts_text(self) -> str:
        if not self.element_counts:
            return "none"
        return ", ".join(
            f"{element}:{count}"
            for element, count in sorted(self.element_counts.items())
        )


@dataclass(slots=True)
class GeneratedPDBInspection:
    structure: str
    motif: str
    param: str
    file_role: str
    file_path: str
    reference_residue_name: str
    atom_count: int
    element_counts: dict[str, int]
    residue_summaries: tuple[GeneratedPDBResidueSummary, ...]
    solvent_residue_numbers: tuple[int, ...]
    atom_residue_assignments: tuple[str, ...]
    exists: bool = True
    load_error: str | None = None

    @property
    def representative_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    @property
    def variant_label(self) -> str:
        return (
            "No solvent" if self.file_role == "no_solvent" else "With solvent"
        )

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name

    @property
    def solvent_molecule_count(self) -> int:
        return len(self.solvent_residue_numbers)

    @property
    def element_counts_text(self) -> str:
        if not self.element_counts:
            return "none"
        return ", ".join(
            f"{element}:{count}"
            for element, count in sorted(self.element_counts.items())
        )

    @property
    def molecule_residue_text(self) -> str:
        if not self.residue_summaries:
            return "none"
        return "; ".join(
            f"{summary.residue_name} {summary.residue_number}"
            for summary in self.residue_summaries
        )

    def details_text(self) -> str:
        lines = [
            f"Representative: {self.representative_label}",
            f"Variant: {self.variant_label}",
            f"File: {self.file_name}",
            f"Path: {self.file_path}",
        ]
        if self.load_error is not None:
            lines.extend(
                [
                    "",
                    "Unable to inspect this generated PDB file.",
                    self.load_error,
                ]
            )
            return "\n".join(lines)

        reference_residue = self.reference_residue_name or "n/a"
        lines.extend(
            [
                f"Atom count: {self.atom_count}",
                f"Element counts: {self.element_counts_text}",
                (
                    "Solvent molecules matching reference residue "
                    f"{reference_residue}: {self.solvent_molecule_count}"
                ),
                "Molecule residues:",
            ]
        )
        if self.residue_summaries:
            for summary in self.residue_summaries:
                lines.append(
                    "  "
                    f"{summary.residue_name} {summary.residue_number}: "
                    f"{summary.atom_count} atoms "
                    f"({summary.element_counts_text})"
                )
        else:
            lines.append("  none")
        lines.append("")
        lines.append("Atom -> residue mapping:")
        if self.atom_residue_assignments:
            lines.extend(
                f"  {assignment}"
                for assignment in self.atom_residue_assignments
            )
        else:
            lines.append("  none")
        return "\n".join(lines)


def build_generated_pdb_inspections(
    metadata: SolventHandlingMetadata | None,
) -> list[GeneratedPDBInspection]:
    if metadata is None:
        return []
    inspections: list[GeneratedPDBInspection] = []
    for entry in metadata.entries:
        inspections.append(
            _build_generated_pdb_inspection(
                entry,
                file_role="no_solvent",
                file_path=entry.no_solvent_pdb,
                reference_residue_name=metadata.reference_residue_name,
            )
        )
        inspections.append(
            _build_generated_pdb_inspection(
                entry,
                file_role="completed",
                file_path=entry.completed_pdb,
                reference_residue_name=metadata.reference_residue_name,
            )
        )
    inspections.sort(
        key=lambda inspection: (
            _natural_sort_key(inspection.structure),
            _natural_sort_key(inspection.motif),
            0 if inspection.file_role == "no_solvent" else 1,
            inspection.file_name.lower(),
        )
    )
    return inspections


def list_solvent_reference_presets() -> list[object]:
    return list_reference_library(default_reference_library_dir())


def representative_structure_mode_label(mode: str) -> str:
    normalized = str(mode).strip() or "source"
    return _REPRESENTATIVE_STRUCTURE_MODE_LABELS.get(
        normalized,
        normalized.replace("_", " "),
    )


def representative_structure_entry_key(
    entry: RepresentativeSelectionEntry | SolventHandlingEntry,
) -> tuple[str, str, str]:
    return (
        str(getattr(entry, "structure", "")).strip(),
        str(getattr(entry, "motif", "no_motif")).strip() or "no_motif",
        str(getattr(entry, "param", "")).strip(),
    )


def solvent_entry_lookup_for_representatives(
    representative_metadata: RepresentativeSelectionMetadata | None,
    solvent_metadata: SolventHandlingMetadata | None,
) -> dict[tuple[str, str, str], SolventHandlingEntry]:
    if representative_metadata is None or solvent_metadata is None:
        return {}
    lookup = {
        representative_structure_entry_key(entry): entry
        for entry in solvent_metadata.entries
    }
    expected_keys = {
        representative_structure_entry_key(entry)
        for entry in representative_metadata.representative_entries
    }
    if not expected_keys or not expected_keys.issubset(lookup):
        return {}
    return {key: lookup[key] for key in expected_keys}


def _uniform_source_variant_mode(
    representative_entries: list[object],
) -> str | None:
    if not representative_entries:
        return None
    source_paths = [
        Path(entry.source_file).expanduser().resolve()
        for entry in representative_entries
        if str(entry.source_file).strip()
    ]
    if len(source_paths) != len(representative_entries) or not all(
        path.is_file() for path in source_paths
    ):
        return None
    variants = {
        _representative_entry_source_variant(entry)
        for entry in representative_entries
    }
    variants.discard(None)
    if len(variants) == 1:
        return next(iter(variants))
    return None


def _uniform_available_source_variants(
    representative_entries: list[object],
) -> list[str]:
    if not representative_entries:
        return []
    mode_order = ("no_solvent", "partial_solvent", "full_solvent")
    available_by_mode: list[str] = []
    for mode in mode_order:
        if all(
            _representative_entry_variant_path(entry, mode) is not None
            for entry in representative_entries
        ):
            available_by_mode.append(mode)
    return available_by_mode


def _representative_entry_variant_path(
    entry: object,
    mode: str,
) -> Path | None:
    source_file = getattr(entry, "source_file", None)
    if not str(source_file or "").strip():
        return None
    return representative_structure_variant_path(source_file, mode)


def _representative_entry_source_variant(entry: object) -> str | None:
    try:
        source_solvent_mode = getattr(entry, "source_solvent_mode")
    except AttributeError:
        source_solvent_mode = None
    variant = representative_source_solvent_mode_to_variant(
        source_solvent_mode
    )
    if variant is not None:
        return variant
    try:
        detected_source_status = str(
            getattr(entry, "detected_source_status")
        ).strip()
    except AttributeError:
        detected_source_status = ""
    if detected_source_status == "complete_solvent":
        return "full_solvent"
    if detected_source_status in {"partial_solvent", "no_solvent"}:
        return detected_source_status
    return None


def available_representative_structure_modes(
    representative_metadata: RepresentativeSelectionMetadata | None,
    solvent_metadata: SolventHandlingMetadata | None,
) -> list[str]:
    representative_entries = (
        list(representative_metadata.representative_entries)
        if representative_metadata is not None
        else []
    )
    if not representative_entries:
        representative_entries = (
            list(solvent_metadata.entries)
            if solvent_metadata is not None
            else []
        )
    if not representative_entries:
        return []
    source_variant = _uniform_source_variant_mode(representative_entries)

    if representative_metadata is None or solvent_metadata is None:
        if solvent_metadata is None:
            available = _uniform_available_source_variants(
                representative_entries
            )
            if source_variant is not None and source_variant not in available:
                available.append(source_variant)
            return available or ["source"]
        ordered_entries = list(solvent_metadata.entries)
    else:
        lookup = solvent_entry_lookup_for_representatives(
            representative_metadata,
            solvent_metadata,
        )
        if not lookup:
            return (
                [source_variant] if source_variant is not None else ["source"]
            )
        ordered_entries = list(lookup.values())

    available: list[str] = []
    if all(
        Path(entry.no_solvent_pdb).expanduser().is_file()
        for entry in ordered_entries
    ):
        available.append("no_solvent")
    if (
        solvent_metadata.detected_distribution_status == "partial_solvent"
        and all(
            Path(entry.source_file).expanduser().is_file()
            for entry in representative_entries
        )
    ):
        available.append("partial_solvent")
    if all(
        Path(entry.completed_pdb).expanduser().is_file()
        for entry in ordered_entries
    ):
        available.append("full_solvent")
    if source_variant is not None and source_variant not in available:
        available.append(source_variant)
    return available or ["source"]


def resolved_representative_structure_mode(
    representative_metadata: RepresentativeSelectionMetadata | None,
    solvent_metadata: SolventHandlingMetadata | None,
    *,
    preferred_mode: str | None = None,
) -> str:
    available = available_representative_structure_modes(
        representative_metadata,
        solvent_metadata,
    )
    if not available:
        return "source"
    requested = str(
        preferred_mode
        if preferred_mode is not None
        else (
            solvent_metadata.settings.coordinated_solvent_mode
            if solvent_metadata is not None
            else "source"
        )
    ).strip()
    if requested in available:
        return requested
    for candidate in (
        "full_solvent",
        "partial_solvent",
        "no_solvent",
        "source",
    ):
        if candidate in available:
            return candidate
    return available[0]


def representative_structure_mode_is_ready(
    representative_metadata: RepresentativeSelectionMetadata | None,
    solvent_metadata: SolventHandlingMetadata | None,
) -> bool:
    return (
        resolved_representative_structure_mode(
            representative_metadata,
            solvent_metadata,
        )
        == "full_solvent"
    )


def representative_structure_path_for_mode(
    representative_entry: RepresentativeSelectionEntry,
    solvent_entry: SolventHandlingEntry | None,
    mode: str,
) -> Path:
    normalized = str(mode).strip() or "source"
    if normalized == "no_solvent" and solvent_entry is not None:
        return Path(solvent_entry.no_solvent_pdb).expanduser().resolve()
    if normalized == "full_solvent" and solvent_entry is not None:
        return Path(solvent_entry.completed_pdb).expanduser().resolve()
    mirrored_variant_path = _representative_entry_variant_path(
        representative_entry,
        normalized,
    )
    if mirrored_variant_path is not None:
        return mirrored_variant_path
    if normalized == representative_source_solvent_mode_to_variant(
        representative_entry.source_solvent_mode
    ):
        return Path(representative_entry.source_file).expanduser().resolve()
    return Path(representative_entry.source_file).expanduser().resolve()


def analyze_representative_solvent_distribution(
    project_source: "RMCDreamProjectSource",
    settings: SolventHandlingSettings,
    *,
    representative_metadata: RepresentativeSelectionMetadata | None = None,
) -> RepresentativeSolventDistributionAnalysis:
    metadata = (
        representative_metadata or project_source.representative_selection
    )
    if metadata is None or not metadata.representative_entries:
        raise ValueError(
            "Save representative structures before analyzing representative solvent states."
        )

    from .solvent_shell_builder import analyze_solvent_shell

    reference_identifier = _reference_identifier(settings)
    reference_path = _resolve_reference_path(settings)
    reference_structure = PDBStructure.from_file(reference_path)
    if not reference_structure.atoms:
        raise ValueError(
            f"The solvent reference PDB has no atoms: {reference_path}"
        )

    entries: list[RepresentativeSolventAnalysisEntry] = []
    aggregate_solute_counts: Counter[str] = Counter()
    for representative_entry in metadata.representative_entries:
        analysis_result = analyze_solvent_shell(
            representative_entry.source_file,
            reference_identifier,
            reference_match_tolerance_a=settings.reference_match_tolerance_a,
        )
        source_status = _classify_source_solvent_status(analysis_result)
        entries.append(
            RepresentativeSolventAnalysisEntry(
                structure=representative_entry.structure,
                motif=representative_entry.motif,
                param=representative_entry.param,
                source_file=representative_entry.source_file,
                source_status=source_status,
                analysis_result=analysis_result,
            )
        )
        aggregate_solute_counts.update(analysis_result.solute_element_counts)
    distribution_status, distribution_note = _resolve_distribution_status(
        entries
    )
    return RepresentativeSolventDistributionAnalysis(
        reference_name=reference_path.stem,
        reference_path=str(reference_path),
        reference_residue_name=reference_structure.atoms[0].residue_name
        or "SOL",
        representative_selection_mode=metadata.selection_mode,
        match_tolerance_a=settings.reference_match_tolerance_a,
        distribution_status=distribution_status,
        distribution_note=distribution_note,
        aggregate_solute_element_counts=dict(
            sorted(aggregate_solute_counts.items())
        ),
        entries=entries,
    )


def build_representative_solvent_outputs(
    project_source: "RMCDreamProjectSource",
    settings: SolventHandlingSettings,
    *,
    representative_metadata: RepresentativeSelectionMetadata | None = None,
    distribution_analysis: (
        RepresentativeSolventDistributionAnalysis | None
    ) = None,
) -> SolventHandlingMetadata:
    metadata = (
        representative_metadata or project_source.representative_selection
    )
    if metadata is None or not metadata.representative_entries:
        raise ValueError(
            "Save representative structures before building "
            "solvent-decorated representative PDBs."
        )
    analysis = (
        distribution_analysis
        or analyze_representative_solvent_distribution(
            project_source,
            settings,
            representative_metadata=metadata,
        )
    )
    reference_path = _resolve_reference_path(settings)
    reference_structure = PDBStructure.from_file(reference_path)
    if not reference_structure.atoms:
        raise ValueError(
            f"The solvent reference PDB has no atoms: {reference_path}"
        )
    reference_identifier = _reference_identifier(settings)
    director_atom_name = (
        settings.director_atom_name
        or _default_director_atom_name_for_settings(settings)
    )
    if not director_atom_name:
        raise ValueError(
            "Select a solvent director atom before building representative solvent outputs."
        )

    entries: list[SolventHandlingEntry] = []
    for representative_entry, analysis_entry in zip(
        metadata.representative_entries,
        analysis.entries,
        strict=True,
    ):
        cluster_structure = _load_cluster_structure_as_pdb(
            representative_entry.source_file,
            structure_label=representative_entry.structure,
        )
        no_solvent_structure = _strip_detected_solvent_atoms(
            cluster_structure,
            analysis_entry.analysis_result,
        )
        no_solvent_path = _representative_pdb_output_path(
            project_source.rmcsetup_paths.pdb_no_solvent_dir,
            representative_entry,
        )
        no_solvent_structure.write_pdb_file(no_solvent_path)

        completed_path = _representative_pdb_output_path(
            project_source.rmcsetup_paths.pdb_with_solvent_dir,
            representative_entry,
        )
        completed_structure: PDBStructure
        solvent_atoms_added = 0
        solvent_molecules_added = 0
        completion_strategy = ""
        build_summary = ""
        if analysis.distribution_status == "complete_solvent":
            completed_structure = _decorated_source_structure_as_pdb(
                representative_entry.source_file,
                structure_label=representative_entry.structure,
                analysis_result=analysis_entry.analysis_result,
                reference_residue_name=analysis.reference_residue_name,
            )
            completed_structure.write_pdb_file(completed_path)
            solvent_molecules_added = int(
                analysis_entry.analysis_result.complete_solvent_molecule_count
            )
            solvent_atoms_added = max(
                len(completed_structure.atoms)
                - len(no_solvent_structure.atoms),
                0,
            )
            completion_strategy = "preserved_detected_complete_solvent"
            build_summary = (
                "Completed representative PDB was passed through because "
                "all representative structures already contained complete "
                "solvent molecules."
            )
        else:
            from .solvent_shell_builder import build_solvent_shell_output

            build_input_path = no_solvent_path
            build_analysis_result = None
            solute_distance_cutoffs = _solute_distance_cutoffs_for_analysis(
                settings,
                analysis_entry.analysis_result,
            )
            coordinating_center_elements = (
                _coordinating_center_elements_for_analysis(
                    settings,
                    analysis_entry.analysis_result,
                )
            )
            target_coordination_numbers = (
                _target_coordination_numbers_for_analysis(
                    settings,
                    analysis_entry.analysis_result,
                )
            )
            if analysis.distribution_status == "partial_solvent":
                build_input_path = Path(representative_entry.source_file)
                build_analysis_result = analysis_entry.analysis_result
            elif (
                not coordinating_center_elements
                or not target_coordination_numbers
            ):
                completed_structure = PDBStructure(
                    atoms=[atom.copy() for atom in no_solvent_structure.atoms],
                    source_name=no_solvent_structure.source_name,
                )
                completed_structure.write_pdb_file(completed_path)
                completion_strategy = (
                    "preserved_without_matching_coordination_settings"
                )
                build_summary = (
                    "No matching coordination-center settings were "
                    "selected for this representative structure, so the "
                    "stripped no-solvent PDB was preserved without adding "
                    "solvent molecules."
                )
                entries.append(
                    SolventHandlingEntry(
                        structure=representative_entry.structure,
                        motif=representative_entry.motif,
                        param=representative_entry.param,
                        source_file=representative_entry.source_file,
                        no_solvent_pdb=str(no_solvent_path),
                        completed_pdb=str(completed_path),
                        atom_count_no_solvent=len(no_solvent_structure.atoms),
                        atom_count_completed=len(completed_structure.atoms),
                        solvent_atoms_added=0,
                        solvent_molecules_added=0,
                        solvent_mode=analysis.distribution_status,
                        completion_strategy=completion_strategy,
                        heuristic_note=analysis_entry.analysis_result.cluster_solvent_status_text,
                        detected_source_status=analysis_entry.source_status,
                        detected_complete_solvent_count=int(
                            analysis_entry.analysis_result.complete_solvent_molecule_count
                        ),
                        detected_partial_solvent_count=int(
                            analysis_entry.analysis_result.partial_solvent_molecule_count
                        ),
                        source_input_format=analysis_entry.analysis_result.input_format,
                        matched_atom_count=int(
                            analysis_entry.analysis_result.matched_atom_count
                        ),
                        unmatched_atom_count=int(
                            analysis_entry.analysis_result.unmatched_atom_count
                        ),
                        solute_element_counts=dict(
                            sorted(
                                analysis_entry.analysis_result.solute_element_counts.items()
                            )
                        ),
                        analysis_summary=analysis_entry.analysis_result.summary_text(),
                        build_summary=build_summary,
                    )
                )
                continue
            build_result = build_solvent_shell_output(
                build_input_path,
                reference_identifier,
                output_path=completed_path,
                director_atom_name=director_atom_name,
                minimum_solvent_atom_separation_a=(
                    settings.minimum_solvent_atom_separation_a
                ),
                solute_distance_cutoffs_a=solute_distance_cutoffs,
                coordinating_center_elements=coordinating_center_elements,
                target_average_coordination_numbers=target_coordination_numbers,
                reference_match_tolerance_a=(
                    settings.reference_match_tolerance_a
                ),
                analysis_result=build_analysis_result,
            )
            completed_structure = PDBStructure.from_file(completed_path)
            solvent_atoms_added = int(build_result.solvent_atoms_added)
            solvent_molecules_added = int(build_result.solvent_molecules_added)
            completion_strategy = str(build_result.build_mode)
            if (
                analysis.distribution_status == "no_solvent"
                and analysis.distribution_note
            ):
                completion_strategy = "rebuilt_from_no_solvent_distribution"
            build_summary = build_result.summary_text()

        entries.append(
            SolventHandlingEntry(
                structure=representative_entry.structure,
                motif=representative_entry.motif,
                param=representative_entry.param,
                source_file=representative_entry.source_file,
                no_solvent_pdb=str(no_solvent_path),
                completed_pdb=str(completed_path),
                atom_count_no_solvent=len(no_solvent_structure.atoms),
                atom_count_completed=len(completed_structure.atoms),
                solvent_atoms_added=solvent_atoms_added,
                solvent_molecules_added=solvent_molecules_added,
                solvent_mode=analysis.distribution_status,
                completion_strategy=completion_strategy,
                heuristic_note=analysis_entry.analysis_result.cluster_solvent_status_text,
                detected_source_status=analysis_entry.source_status,
                detected_complete_solvent_count=int(
                    analysis_entry.analysis_result.complete_solvent_molecule_count
                ),
                detected_partial_solvent_count=int(
                    analysis_entry.analysis_result.partial_solvent_molecule_count
                ),
                source_input_format=analysis_entry.analysis_result.input_format,
                matched_atom_count=int(
                    analysis_entry.analysis_result.matched_atom_count
                ),
                unmatched_atom_count=int(
                    analysis_entry.analysis_result.unmatched_atom_count
                ),
                solute_element_counts=dict(
                    sorted(
                        analysis_entry.analysis_result.solute_element_counts.items()
                    )
                ),
                analysis_summary=analysis_entry.analysis_result.summary_text(),
                build_summary=build_summary,
            )
        )

    saved_settings = SolventHandlingSettings.from_dict(settings.to_dict())
    saved_settings.coordinated_solvent_mode = "full_solvent"
    solvent_metadata = SolventHandlingMetadata(
        settings=saved_settings,
        reference_path=str(reference_path),
        reference_name=analysis.reference_name,
        reference_residue_name=analysis.reference_residue_name,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_selection_mode=metadata.selection_mode,
        detected_distribution_status=analysis.distribution_status,
        detected_distribution_note=analysis.distribution_note,
        aggregate_solute_element_counts=dict(
            sorted(analysis.aggregate_solute_element_counts.items())
        ),
        entries=entries,
    )
    save_solvent_handling_metadata(
        project_source.rmcsetup_paths.solvent_handling_path,
        solvent_metadata,
    )
    return solvent_metadata


def _representative_source_kind_lookup(
    metadata: RepresentativeSelectionMetadata,
) -> dict[tuple[str, str, str], str | None]:
    return {
        (entry.structure, entry.motif, entry.param): getattr(
            entry,
            "source_kind",
            None,
        )
        for entry in metadata.distribution_selection.entries
    }


def _reference_identifier(settings: SolventHandlingSettings) -> str:
    if (
        settings.reference_source == "custom"
        and settings.custom_reference_path
    ):
        return str(Path(settings.custom_reference_path).expanduser().resolve())
    return settings.preset_name


def _default_director_atom_name_for_settings(
    settings: SolventHandlingSettings,
) -> str | None:
    from .solvent_shell_builder import default_director_atom_name

    return default_director_atom_name(_reference_identifier(settings))


def _classify_source_solvent_status(
    analysis_result: "SolventShellAnalysisResult",
) -> str:
    if analysis_result.complete_solvent_molecule_count > 0:
        if analysis_result.partial_solvent_molecule_count > 0:
            return "mixed_complete_and_partial"
        return "complete_solvent"
    if analysis_result.partial_solvent_molecule_count > 0:
        return "partial_solvent"
    return "no_solvent"


def _resolve_distribution_status(
    entries: list[RepresentativeSolventAnalysisEntry],
) -> tuple[str, str]:
    statuses = {
        entry.source_status
        for entry in entries
        if str(entry.source_status).strip()
    }
    if not statuses:
        return (
            "unknown",
            "No representative solvent states were available.",
        )
    if statuses == {"complete_solvent"}:
        return (
            "complete_solvent",
            "Every representative structure already contains complete solvent molecules, so the existing solvent-decorated structures can be passed through.",
        )
    if statuses == {"partial_solvent"}:
        return (
            "partial_solvent",
            "Every representative structure contains partial solvent molecules, so the saved anchors will be used to rebuild complete solvent molecules.",
        )
    if statuses == {"no_solvent"}:
        return (
            "no_solvent",
            "No representative structure contains coordinated solvent molecules, so solvent shells will be built from the stripped solute structures.",
        )
    return (
        "no_solvent",
        "Representative solvent detections were inconsistent across the saved structures. Following the conservative workflow rule, the current cluster distribution is treated as having no coordinated solvent.",
    )


def _solvent_state_text(status: str) -> str:
    mapping = {
        "complete_solvent": "Complete solvent molecules detected",
        "partial_solvent": "Partial solvent molecules detected",
        "no_solvent": "No solvent molecules detected",
        "mixed_complete_and_partial": (
            "Complete and partial solvent molecules detected"
        ),
        "unknown": "Unknown solvent state",
    }
    return mapping.get(str(status).strip(), str(status).replace("_", " "))


def _strip_detected_solvent_atoms(
    structure: PDBStructure,
    analysis_result: "SolventShellAnalysisResult",
) -> PDBStructure:
    stripped_atom_ids = {
        int(atom_id)
        for atom_id in analysis_result.complete_solvent_source_atom_ids
    }
    stripped_atom_ids.update(
        int(atom_id)
        for atom_id in analysis_result.partial_solvent_source_atom_ids
    )
    stripped_atoms = [
        atom.copy()
        for atom in structure.atoms
        if int(atom.atom_id) not in stripped_atom_ids
    ]
    for index, atom in enumerate(stripped_atoms, start=1):
        atom.atom_id = index
    return PDBStructure(
        atoms=stripped_atoms,
        source_name=structure.source_name,
    )


def _decorated_source_structure_as_pdb(
    source_file: str | Path,
    *,
    structure_label: str,
    analysis_result: "SolventShellAnalysisResult",
    reference_residue_name: str,
) -> PDBStructure:
    structure = _load_cluster_structure_as_pdb(
        source_file,
        structure_label=structure_label,
    )
    path = Path(source_file).expanduser().resolve()
    if path.suffix.lower() == ".pdb":
        return structure

    group_lookup = {
        int(atom_id): group_index
        for group_index, atom_group in enumerate(
            analysis_result.complete_solvent_source_atom_groups,
            start=1,
        )
        for atom_id in atom_group
    }
    solute_residue_name = _normalized_residue_name(structure_label)
    next_residue_number = 2
    solvent_residue_numbers: dict[int, int] = {}
    for atom in structure.atoms:
        group_index = group_lookup.get(int(atom.atom_id))
        if group_index is None:
            atom.residue_name = solute_residue_name
            atom.residue_number = 1
            continue
        residue_number = solvent_residue_numbers.get(group_index)
        if residue_number is None:
            residue_number = next_residue_number
            solvent_residue_numbers[group_index] = residue_number
            next_residue_number += 1
        atom.residue_name = reference_residue_name
        atom.residue_number = residue_number
    return structure


def _solute_distance_cutoffs_for_analysis(
    settings: SolventHandlingSettings,
    analysis_result: "SolventShellAnalysisResult",
) -> dict[str, float]:
    return {
        str(element): float(setting.director_distance_cutoff_a)
        for element, setting in settings.solute_atom_settings.items()
        if element in analysis_result.solute_element_counts
        and float(setting.director_distance_cutoff_a) > 0.0
    }


def _coordinating_center_elements_for_analysis(
    settings: SolventHandlingSettings,
    analysis_result: "SolventShellAnalysisResult",
) -> tuple[str, ...]:
    return tuple(
        sorted(
            element
            for element, setting in settings.solute_atom_settings.items()
            if element in analysis_result.solute_element_counts
            and setting.coordination_center
            and float(setting.target_coordination_number) > 0.0
        )
    )


def _target_coordination_numbers_for_analysis(
    settings: SolventHandlingSettings,
    analysis_result: "SolventShellAnalysisResult",
) -> dict[str, float]:
    return {
        str(element): float(setting.target_coordination_number)
        for element, setting in settings.solute_atom_settings.items()
        if element in analysis_result.solute_element_counts
        and setting.coordination_center
        and float(setting.target_coordination_number) > 0.0
    }


def save_solvent_handling_metadata(
    output_path: str | Path,
    metadata: SolventHandlingMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_solvent_handling_metadata(
    metadata_path: str | Path,
) -> SolventHandlingMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return SolventHandlingMetadata.from_dict(payload)


def _resolve_reference_path(settings: SolventHandlingSettings) -> Path:
    if settings.reference_source == "custom":
        if not settings.custom_reference_path:
            raise ValueError("Choose a custom solvent reference PDB file.")
        reference_path = (
            Path(settings.custom_reference_path).expanduser().resolve()
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Custom solvent reference PDB was not found: {reference_path}"
            )
        return reference_path
    return resolve_reference_path(
        settings.preset_name,
        library_dir=default_reference_library_dir(),
    ).resolve()


def _load_cluster_structure_as_pdb(
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
        return PDBStructure(
            atoms=copied_atoms,
            source_name=path.stem,
        )

    positions, elements = load_structure_file(path)
    counters: dict[str, int] = {}
    atoms: list[PDBAtom] = []
    residue_name = _normalized_residue_name(structure_label)
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


def _translated_reference_atoms(
    reference_structure: PDBStructure,
    *,
    anchor_atoms: list[PDBAtom],
    starting_atom_id: int,
    residue_number: int,
) -> list[PDBAtom]:
    reference_atoms = [atom.copy() for atom in reference_structure.atoms]
    if not reference_atoms:
        return []
    cluster_coords = np.array(
        [atom.coordinates for atom in anchor_atoms],
        dtype=float,
    )
    if cluster_coords.size <= 0:
        translation = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        cluster_min = np.min(cluster_coords, axis=0)
        cluster_max = np.max(cluster_coords, axis=0)
        cluster_center = np.mean(cluster_coords, axis=0)
        reference_coords = np.array(
            [atom.coordinates for atom in reference_atoms],
            dtype=float,
        )
        reference_center = np.mean(reference_coords, axis=0)
        reference_span = np.max(reference_coords, axis=0) - np.min(
            reference_coords,
            axis=0,
        )
        shift_x = float(
            cluster_max[0] - cluster_min[0] + reference_span[0] + 2.0
        )
        translation = (
            cluster_center + np.array([shift_x, 0.0, 0.0]) - reference_center
        )

    translated: list[PDBAtom] = []
    for index, atom in enumerate(reference_atoms, start=starting_atom_id):
        atom.atom_id = index
        atom.residue_number = residue_number
        atom.coordinates = atom.coordinates + translation
        translated.append(atom)
    return translated


def _build_partial_coordinated_solvent_structure(
    cluster_structure: PDBStructure,
    reference_structure: PDBStructure,
    *,
    minimum_atom_separation_a: float,
) -> tuple[PDBStructure, int, int, str]:
    anchor_index = _select_reference_anchor_atom(reference_structure)
    if anchor_index is None:
        completed_structure = PDBStructure(
            atoms=[atom.copy() for atom in cluster_structure.atoms],
            source_name=cluster_structure.source_name,
        )
        added_atoms = _translated_reference_atoms(
            reference_structure,
            anchor_atoms=completed_structure.atoms,
            starting_atom_id=len(completed_structure.atoms) + 1,
            residue_number=_next_residue_number(completed_structure.atoms),
        )
        completed_structure.add_atoms(added_atoms)
        return (
            completed_structure,
            len(added_atoms),
            1 if added_atoms else 0,
            "appended_reference_solvent",
        )

    anchor_atoms = _partial_solvent_anchor_atoms(
        cluster_structure.atoms,
        reference_structure=reference_structure,
        reference_anchor_index=anchor_index,
    )
    if not anchor_atoms:
        completed_structure = PDBStructure(
            atoms=[atom.copy() for atom in cluster_structure.atoms],
            source_name=cluster_structure.source_name,
        )
        added_atoms = _translated_reference_atoms(
            reference_structure,
            anchor_atoms=completed_structure.atoms,
            starting_atom_id=len(completed_structure.atoms) + 1,
            residue_number=_next_residue_number(completed_structure.atoms),
        )
        completed_structure.add_atoms(added_atoms)
        return (
            completed_structure,
            len(added_atoms),
            1 if added_atoms else 0,
            "appended_reference_solvent_fallback",
        )

    anchor_atom_ids = {atom.atom_id for atom in anchor_atoms}
    remaining_atoms = [
        atom.copy()
        for atom in cluster_structure.atoms
        if atom.atom_id not in anchor_atom_ids
    ]
    placed_atoms, clearances_met = _place_solvent_molecules_on_anchors(
        reference_structure,
        reference_anchor_index=anchor_index,
        anchor_positions=[
            atom.coordinates.copy()
            for atom in sorted(anchor_atoms, key=lambda atom: atom.atom_id)
        ],
        cluster_atoms=remaining_atoms,
        starting_atom_id=len(remaining_atoms) + 1,
        starting_residue_number=_next_residue_number(remaining_atoms),
        minimum_atom_separation_a=minimum_atom_separation_a,
    )
    completed_atoms = remaining_atoms + placed_atoms
    _reindex_atoms(completed_atoms)
    return (
        PDBStructure(
            atoms=completed_atoms,
            source_name=cluster_structure.source_name,
        ),
        max(0, len(completed_atoms) - len(cluster_structure.atoms)),
        len(anchor_atoms),
        (
            "anchored_solvent_completion"
            if clearances_met
            else "anchored_solvent_completion_best_effort"
        ),
    )


def _select_reference_anchor_atom(
    reference_structure: PDBStructure,
) -> int | None:
    if not reference_structure.atoms:
        return None

    non_hydrogen_indices = [
        index
        for index, atom in enumerate(reference_structure.atoms)
        if atom.element.upper() != "H"
    ]
    if not non_hydrogen_indices:
        return 0

    non_hydrogen_counts = Counter(
        reference_structure.atoms[index].element.upper()
        for index in non_hydrogen_indices
    )
    for element in _ANCHOR_ELEMENT_PRIORITY:
        matches = [
            index
            for index in non_hydrogen_indices
            if reference_structure.atoms[index].element.upper() == element
        ]
        if len(matches) == 1:
            return matches[0]
    for element in _ANCHOR_ELEMENT_PRIORITY:
        matches = [
            index
            for index in non_hydrogen_indices
            if (
                reference_structure.atoms[index].element.upper() == element
                and non_hydrogen_counts[
                    reference_structure.atoms[index].element.upper()
                ]
                == 1
            )
        ]
        if matches:
            return matches[0]
    return non_hydrogen_indices[0]


def _partial_solvent_anchor_atoms(
    cluster_atoms: list[PDBAtom],
    *,
    reference_structure: PDBStructure,
    reference_anchor_index: int,
) -> list[PDBAtom]:
    if not cluster_atoms:
        return []
    reference_anchor = reference_structure.atoms[reference_anchor_index]
    reference_residue = reference_anchor.residue_name.upper().strip()
    anchor_element = reference_anchor.element.upper()
    return [
        atom
        for atom in cluster_atoms
        if atom.element.upper() == anchor_element
        and atom.residue_name.upper().strip() != reference_residue
    ]


def _place_solvent_molecules_on_anchors(
    reference_structure: PDBStructure,
    *,
    reference_anchor_index: int,
    anchor_positions: list[np.ndarray],
    cluster_atoms: list[PDBAtom],
    starting_atom_id: int,
    starting_residue_number: int,
    minimum_atom_separation_a: float,
) -> tuple[list[PDBAtom], bool]:
    reference_atoms = [atom.copy() for atom in reference_structure.atoms]
    reference_anchor = reference_atoms[reference_anchor_index]
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
        if index != reference_anchor_index
    ]
    reference_body_center = _weighted_center(reference_body_atoms)
    reference_body_vector = reference_body_center - reference_anchor_coord
    if np.linalg.norm(reference_body_vector) <= 1e-8 and reference_body_atoms:
        reference_body_vector = (
            reference_body_atoms[0].coordinates - reference_anchor_coord
        )
    if np.linalg.norm(reference_body_vector) <= 1e-8:
        reference_body_vector = np.array([1.0, 0.0, 0.0], dtype=float)

    solute_center = _weighted_center(cluster_atoms)
    ordered_positions = sorted(
        anchor_positions,
        key=lambda position: float(np.linalg.norm(position - solute_center)),
        reverse=True,
    )
    cluster_coords = [atom.coordinates.copy() for atom in cluster_atoms]
    placements: list[dict[str, object]] = []
    next_residue_number = starting_residue_number
    for anchor_position in ordered_positions:
        outward_vector = _normalize_vector(
            anchor_position - solute_center,
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        alignment = _rotation_between_vectors(
            reference_body_vector,
            outward_vector,
        )
        aligned_offsets = _rotate_points(reference_offsets, alignment)
        candidate_coords, min_distance, clearance_met = (
            _best_solvent_orientation(
                aligned_offsets,
                anchor_position=anchor_position,
                outward_vector=outward_vector,
                occupied_coords=cluster_coords
                + [
                    coordinate
                    for placement in placements
                    for coordinate in placement["coords"]
                ],
                excluded_index=reference_anchor_index,
                minimum_atom_separation_a=minimum_atom_separation_a,
            )
        )
        placements.append(
            {
                "aligned_offsets": aligned_offsets,
                "anchor_position": anchor_position,
                "outward_vector": outward_vector,
                "coords": candidate_coords,
                "minimum_distance": min_distance,
                "clearance_met": clearance_met,
                "residue_number": next_residue_number,
            }
        )
        next_residue_number += 1

    for _ in range(2):
        for index, placement in enumerate(placements):
            occupied_coords = list(cluster_coords)
            for other_index, other in enumerate(placements):
                if other_index == index:
                    continue
                occupied_coords.extend(other["coords"])
            candidate_coords, min_distance, clearance_met = (
                _best_solvent_orientation(
                    placement["aligned_offsets"],
                    anchor_position=placement["anchor_position"],
                    outward_vector=placement["outward_vector"],
                    occupied_coords=occupied_coords,
                    excluded_index=reference_anchor_index,
                    minimum_atom_separation_a=minimum_atom_separation_a,
                )
            )
            if clearance_met and not placement["clearance_met"]:
                placement["coords"] = candidate_coords
                placement["minimum_distance"] = min_distance
                placement["clearance_met"] = clearance_met
                continue
            if min_distance > float(placement["minimum_distance"]) + 1e-6:
                placement["coords"] = candidate_coords
                placement["minimum_distance"] = min_distance
                placement["clearance_met"] = clearance_met

    all_clearances_met = all(
        bool(placement["clearance_met"]) for placement in placements
    )
    placed_atoms: list[PDBAtom] = []
    next_atom_id = starting_atom_id
    for placement in placements:
        coordinates = np.asarray(placement["coords"], dtype=float)
        residue_number = int(placement["residue_number"])
        for reference_atom, coordinate in zip(
            reference_atoms,
            coordinates,
            strict=True,
        ):
            atom = reference_atom.copy()
            atom.atom_id = next_atom_id
            atom.residue_number = residue_number
            atom.coordinates = coordinate.copy()
            placed_atoms.append(atom)
            next_atom_id += 1
    return placed_atoms, all_clearances_met


def _best_solvent_orientation(
    aligned_offsets: np.ndarray,
    *,
    anchor_position: np.ndarray,
    outward_vector: np.ndarray,
    occupied_coords: list[np.ndarray],
    excluded_index: int,
    minimum_atom_separation_a: float,
) -> tuple[np.ndarray, float, bool]:
    best_coords: np.ndarray | None = None
    best_min_distance = float("-inf")
    best_clearance_met = False
    for angle_deg in _SOLVENT_ROTATION_ANGLES_DEG:
        rotation = _axis_angle_rotation(
            outward_vector,
            np.deg2rad(float(angle_deg)),
        )
        candidate_coords = (
            _rotate_points(aligned_offsets, rotation) + anchor_position
        )
        min_distance = _minimum_distance_to_occupied_atoms(
            candidate_coords,
            occupied_coords=occupied_coords,
            excluded_index=excluded_index,
        )
        clearance_met = min_distance >= minimum_atom_separation_a - 1e-6
        if best_coords is None:
            best_coords = candidate_coords
            best_min_distance = min_distance
            best_clearance_met = clearance_met
            continue
        if clearance_met and not best_clearance_met:
            best_coords = candidate_coords
            best_min_distance = min_distance
            best_clearance_met = True
            continue
        if clearance_met == best_clearance_met and min_distance > (
            best_min_distance + 1e-6
        ):
            best_coords = candidate_coords
            best_min_distance = min_distance
            best_clearance_met = clearance_met
    if best_coords is None:
        best_coords = aligned_offsets + anchor_position
        best_min_distance = float("inf")
    return best_coords, best_min_distance, best_clearance_met


def _minimum_distance_to_occupied_atoms(
    candidate_coords: np.ndarray,
    *,
    occupied_coords: list[np.ndarray],
    excluded_index: int,
) -> float:
    if not occupied_coords:
        return float("inf")
    candidate_subset = np.asarray(candidate_coords, dtype=float)
    if 0 <= excluded_index < len(candidate_subset):
        candidate_subset = np.delete(candidate_subset, excluded_index, axis=0)
    if candidate_subset.size == 0:
        return float("inf")
    occupied = np.asarray(occupied_coords, dtype=float)
    distances = np.linalg.norm(
        candidate_subset[:, np.newaxis, :] - occupied[np.newaxis, :, :],
        axis=2,
    )
    return float(np.min(distances))


def _weighted_center(atoms: list[PDBAtom]) -> np.ndarray:
    if not atoms:
        return np.zeros(3, dtype=float)
    coordinates = np.asarray([atom.coordinates for atom in atoms], dtype=float)
    weights = np.asarray(
        [_ATOMIC_MASSES.get(atom.element, 12.0) for atom in atoms],
        dtype=float,
    )
    weight_total = float(np.sum(weights))
    if weight_total <= 0.0:
        return np.mean(coordinates, axis=0)
    return np.sum(coordinates * weights[:, np.newaxis], axis=0) / weight_total


def _normalize_vector(
    vector: np.ndarray,
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return np.asarray(vector, dtype=float) / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-8:
        return np.asarray(fallback, dtype=float) / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _rotation_between_vectors(
    source_vector: np.ndarray,
    target_vector: np.ndarray,
) -> np.ndarray:
    source = _normalize_vector(
        np.asarray(source_vector, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    target = _normalize_vector(
        np.asarray(target_vector, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return np.eye(3, dtype=float)
    if dot <= -1.0 + 1e-8:
        axis = _orthogonal_vector(source)
        return _axis_angle_rotation(axis, np.pi)
    axis = np.cross(source, target)
    angle = float(np.arccos(dot))
    return _axis_angle_rotation(axis, angle)


def _axis_angle_rotation(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis_normalized = _normalize_vector(
        np.asarray(axis, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    x_value, y_value, z_value = axis_normalized
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x_value * x_value * one_minus_cosine,
                x_value * y_value * one_minus_cosine - z_value * sine,
                x_value * z_value * one_minus_cosine + y_value * sine,
            ],
            [
                y_value * x_value * one_minus_cosine + z_value * sine,
                cosine + y_value * y_value * one_minus_cosine,
                y_value * z_value * one_minus_cosine - x_value * sine,
            ],
            [
                z_value * x_value * one_minus_cosine - y_value * sine,
                z_value * y_value * one_minus_cosine + x_value * sine,
                cosine + z_value * z_value * one_minus_cosine,
            ],
        ],
        dtype=float,
    )


def _orthogonal_vector(vector: np.ndarray) -> np.ndarray:
    normalized = _normalize_vector(
        np.asarray(vector, dtype=float),
        fallback=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    if abs(float(normalized[0])) < 0.9:
        return np.cross(normalized, np.array([1.0, 0.0, 0.0], dtype=float))
    return np.cross(normalized, np.array([0.0, 1.0, 0.0], dtype=float))


def _rotate_points(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.asarray(rotation, dtype=float) @ np.asarray(points).T).T


def _reindex_atoms(atoms: list[PDBAtom]) -> None:
    for atom_id, atom in enumerate(atoms, start=1):
        atom.atom_id = atom_id


def _next_residue_number(atoms: list[PDBAtom]) -> int:
    if not atoms:
        return 1
    return max(atom.residue_number for atom in atoms) + 1


def _representative_pdb_name(representative_entry: object) -> str:
    structure = _safe_name(
        getattr(representative_entry, "structure", "cluster")
    )
    motif = _safe_name(getattr(representative_entry, "motif", "no_motif"))
    source_name = _safe_name(
        Path(getattr(representative_entry, "source_file_name", "cluster")).stem
    )
    return f"{structure}_{motif}_{source_name}.pdb"


def _representative_pdb_output_path(
    root_dir: str | Path,
    representative_entry: object,
) -> Path:
    structure = _safe_name(
        getattr(representative_entry, "structure", "cluster")
    )
    return (
        Path(root_dir).expanduser().resolve()
        / structure
        / _representative_pdb_name(representative_entry)
    )


def _safe_name(text: str) -> str:
    collapsed = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip())
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return collapsed or "item"


def _normalized_residue_name(text: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    if not collapsed:
        collapsed = "CLU"
    return collapsed[:3]


def _solvent_heuristic_note(
    cluster_structure: PDBStructure,
    reference_structure: PDBStructure,
) -> str:
    cluster_elements = {
        atom.element.upper() for atom in cluster_structure.atoms
    }
    reference_elements = {
        atom.element.upper() for atom in reference_structure.atoms
    }
    has_organics = bool(cluster_elements.intersection({"C", "H", "N", "S"}))
    has_partial_oxygen = "O" in cluster_elements and not has_organics
    if has_organics and cluster_elements.intersection(reference_elements):
        return (
            "Source representative already contains organic solvent-like "
            "elements; coordinated solvent may already be complete."
        )
    if has_partial_oxygen:
        return (
            "Source representative contains oxygen without broader organic "
            "solvent signatures; partial coordinated solvent is plausible."
        )
    return (
        "No strong coordinated-solvent heuristic was detected from the "
        "representative element set."
    )


def _build_generated_pdb_inspection(
    entry: SolventHandlingEntry,
    *,
    file_role: str,
    file_path: str | Path,
    reference_residue_name: str,
) -> GeneratedPDBInspection:
    resolved_path = Path(file_path).expanduser().resolve()
    inspection_base = {
        "structure": entry.structure,
        "motif": entry.motif,
        "param": entry.param,
        "file_role": file_role,
        "file_path": str(resolved_path),
        "reference_residue_name": reference_residue_name,
    }
    if not resolved_path.is_file():
        return GeneratedPDBInspection(
            atom_count=0,
            element_counts={},
            residue_summaries=(),
            solvent_residue_numbers=(),
            atom_residue_assignments=(),
            exists=False,
            load_error=f"PDB file not found: {resolved_path}",
            **inspection_base,
        )
    try:
        structure = PDBStructure.from_file(resolved_path)
    except Exception as exc:
        return GeneratedPDBInspection(
            atom_count=0,
            element_counts={},
            residue_summaries=(),
            solvent_residue_numbers=(),
            atom_residue_assignments=(),
            exists=True,
            load_error=str(exc),
            **inspection_base,
        )

    residue_groups: dict[tuple[str, int], list[PDBAtom]] = {}
    for atom in structure.atoms:
        key = (atom.residue_name, int(atom.residue_number))
        residue_groups.setdefault(key, []).append(atom)

    residue_summaries = tuple(
        GeneratedPDBResidueSummary(
            residue_name=residue_name,
            residue_number=residue_number,
            atom_count=len(group_atoms),
            element_counts=dict(
                sorted(Counter(atom.element for atom in group_atoms).items())
            ),
        )
        for (residue_name, residue_number), group_atoms in sorted(
            residue_groups.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    )
    reference_name_upper = reference_residue_name.upper().strip()
    solvent_residue_numbers = tuple(
        summary.residue_number
        for summary in residue_summaries
        if summary.residue_name.upper() == reference_name_upper
    )
    atom_residue_assignments = tuple(
        (
            f"{atom.atom_id}: {atom.atom_name} ({atom.element}) -> "
            f"{atom.residue_name} {atom.residue_number}"
        )
        for atom in sorted(structure.atoms, key=lambda item: item.atom_id)
    )
    return GeneratedPDBInspection(
        atom_count=len(structure.atoms),
        element_counts=dict(
            sorted(Counter(atom.element for atom in structure.atoms).items())
        ),
        residue_summaries=residue_summaries,
        solvent_residue_numbers=solvent_residue_numbers,
        atom_residue_assignments=atom_residue_assignments,
        **inspection_base,
    )


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "GeneratedPDBInspection",
    "GeneratedPDBResidueSummary",
    "RepresentativeSolventAnalysisEntry",
    "RepresentativeSolventDistributionAnalysis",
    "SoluteAtomBuildSetting",
    "SolventHandlingEntry",
    "SolventHandlingMetadata",
    "SolventHandlingSettings",
    "analyze_representative_solvent_distribution",
    "build_generated_pdb_inspections",
    "build_representative_solvent_outputs",
    "list_solvent_reference_presets",
    "load_solvent_handling_metadata",
    "save_solvent_handling_metadata",
]
