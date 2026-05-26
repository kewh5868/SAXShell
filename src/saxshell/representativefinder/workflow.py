from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from saxshell.bondanalysis.bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
)
from saxshell.saxs.contrast.descriptors import (
    ParsedContrastStructure,
    describe_parsed_contrast_structure,
    estimate_pair_contact_distance_medians,
)
from saxshell.saxs.debye import load_structure_file, scan_structure_elements
from saxshell.saxs.stoichiometry import parse_stoich_label
from saxshell.structure_distributions import (
    StructureDistributionStore,
    application_structure_distribution_store_dir,
)

_STRUCTURE_SUFFIXES = {".pdb", ".xyz"}
_DEFAULT_QUANTILES = tuple(np.linspace(0.0, 1.0, 11).tolist())
RepresentativeFinderProgressCallback = Callable[[int, int, str], None]
RepresentativeFinderLogCallback = Callable[[str], None]
RepresentativeFinderCancelCallback = Callable[[], bool]


class RepresentativeFinderOperationCancelled(RuntimeError):
    """Raised when representative-structure analysis is canceled."""


def _emit_progress(
    callback: RepresentativeFinderProgressCallback | None,
    processed: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(
        max(int(processed), 0),
        max(int(total), 1),
        str(message).strip(),
    )


def _emit_log(
    callback: RepresentativeFinderLogCallback | None,
    message: str,
) -> None:
    if callback is None:
        return
    text = str(message).strip()
    if text:
        callback(text)


def _raise_if_cancelled(
    cancel_callback: RepresentativeFinderCancelCallback | None,
) -> None:
    if cancel_callback is not None and cancel_callback():
        raise RepresentativeFinderOperationCancelled(
            "Representative-structure analysis canceled."
        )


@dataclass(slots=True, frozen=True)
class RepresentativeFinderSettings:
    selection_algorithm: str = "target_distribution_quantile_distance"
    bond_weight: float = 1.0
    angle_weight: float = 1.0
    solvent_weight: float = 1.0
    generate_predicted_optimized_representative: bool = False
    parallel_workers: int = 0
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    bond_pairs: tuple[BondPairDefinition, ...] = ()
    angle_triplets: tuple[AngleTripletDefinition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parallel_workers",
            max(int(self.parallel_workers), 0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_algorithm": self.selection_algorithm,
            "bond_weight": self.bond_weight,
            "angle_weight": self.angle_weight,
            "solvent_weight": self.solvent_weight,
            "generate_predicted_optimized_representative": bool(
                self.generate_predicted_optimized_representative
            ),
            "parallel_workers": int(self.parallel_workers),
            "quantiles": list(self.quantiles),
            "bond_pairs": [
                definition.to_dict() for definition in self.bond_pairs
            ],
            "angle_triplets": [
                definition.to_dict() for definition in self.angle_triplets
            ],
        }


@dataclass(slots=True, frozen=True)
class RepresentativeFinderFolderCandidate:
    file_path: Path
    relative_label: str
    motif_label: str


@dataclass(slots=True, frozen=True)
class RepresentativeFinderFolderInspection:
    input_dir: Path
    structure_label: str
    candidate_count: int
    direct_file_count: int
    motif_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]

    def summary_text(self) -> str:
        lines = [
            f"Input folder: {self.input_dir}",
            f"Structure label: {self.structure_label}",
            f"Candidate files: {self.candidate_count}",
            f"Direct files: {self.direct_file_count}",
            "Motif folders: "
            + (", ".join(self.motif_labels) if self.motif_labels else "none"),
        ]
        if self.candidate_labels:
            lines.extend(["", "Candidates"])
            lines.extend(f"  {label}" for label in self.candidate_labels[:12])
            if len(self.candidate_labels) > 12:
                remaining = len(self.candidate_labels) - 12
                lines.append(f"  ... and {remaining} more")
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class RepresentativeFinderInputInspection:
    input_dir: Path
    input_is_stoichiometry_folder: bool
    stoichiometry_folders: tuple[RepresentativeFinderFolderInspection, ...]

    @property
    def stoichiometry_count(self) -> int:
        return len(self.stoichiometry_folders)

    @property
    def total_candidate_count(self) -> int:
        return sum(
            inspection.candidate_count
            for inspection in self.stoichiometry_folders
        )

    def summary_text(self) -> str:
        mode_label = (
            "single stoichiometry folder"
            if self.input_is_stoichiometry_folder
            else "stoichiometry collection folder"
        )
        lines = [
            f"Input folder: {self.input_dir}",
            f"Discovery mode: {mode_label}",
            f"Discovered stoichiometries: {self.stoichiometry_count}",
            f"Total candidate files: {self.total_candidate_count}",
        ]
        if self.stoichiometry_folders:
            lines.extend(["", "Stoichiometries"])
            for inspection in self.stoichiometry_folders[:12]:
                motif_text = (
                    ", ".join(inspection.motif_labels)
                    if inspection.motif_labels
                    else "none"
                )
                lines.append(
                    "  "
                    f"{inspection.structure_label}: "
                    f"{inspection.candidate_count} candidate(s), "
                    f"motifs={motif_text}"
                )
            if len(self.stoichiometry_folders) > 12:
                remaining = len(self.stoichiometry_folders) - 12
                lines.append(f"  ... and {remaining} more")
        return "\n".join(lines)


@dataclass(slots=True)
class RepresentativeFinderCandidate:
    file_path: Path
    relative_label: str
    motif_label: str
    atom_count: int
    element_counts: dict[str, int]
    bond_values: dict[BondPairDefinition, list[float]]
    angle_values: dict[AngleTripletDefinition, list[float]]
    solvent_metrics: dict[str, float]
    solvent_atom_count: int
    direct_solvent_atom_count: int
    outer_solvent_atom_count: int
    mean_direct_solvent_coordination: float
    descriptor_notes: tuple[str, ...] = ()
    score_total: float | None = None
    score_bond: float | None = None
    score_angle: float | None = None
    score_solvent: float | None = None

    @property
    def file_name(self) -> str:
        return self.file_path.name

    def score_sort_key(self) -> tuple[float, float, float, str]:
        return (
            float(
                self.score_total if self.score_total is not None else np.inf
            ),
            float(
                self.score_solvent if self.score_solvent is not None else 0.0
            ),
            float(self.score_bond if self.score_bond is not None else 0.0),
            str(self.file_path),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "file_path": str(self.file_path),
            "relative_label": self.relative_label,
            "motif_label": self.motif_label,
            "atom_count": self.atom_count,
            "element_counts": dict(sorted(self.element_counts.items())),
            "bond_values": _definition_value_map_to_list(self.bond_values),
            "angle_values": _definition_value_map_to_list(self.angle_values),
            "solvent_metrics": dict(sorted(self.solvent_metrics.items())),
            "solvent_atom_count": int(self.solvent_atom_count),
            "direct_solvent_atom_count": int(self.direct_solvent_atom_count),
            "outer_solvent_atom_count": int(self.outer_solvent_atom_count),
            "mean_direct_solvent_coordination": float(
                self.mean_direct_solvent_coordination
            ),
            "descriptor_notes": list(self.descriptor_notes),
            "score_total": _optional_float(self.score_total),
            "score_bond": _optional_float(self.score_bond),
            "score_angle": _optional_float(self.score_angle),
            "score_solvent": _optional_float(self.score_solvent),
        }


@dataclass(slots=True, frozen=True)
class RepresentativeFinderPlotSeries:
    category: str
    display_label: str
    xlabel: str
    distribution_values: np.ndarray
    candidate_values: tuple[float, ...]


@dataclass(slots=True)
class RepresentativeFinderResult:
    input_dir: Path
    output_dir: Path
    structure_label: str
    expected_core_counts: dict[str, int]
    settings: RepresentativeFinderSettings
    generated_at: str
    candidates: tuple[RepresentativeFinderCandidate, ...]
    selected_candidate: RepresentativeFinderCandidate
    representative_output_path: Path
    skipped_files: tuple[str, ...]
    target_bond_values: dict[BondPairDefinition, np.ndarray]
    target_angle_values: dict[AngleTripletDefinition, np.ndarray]
    target_solvent_metrics: dict[str, float]
    summary_json_path: Path
    score_table_path: Path
    summary_text_path: Path
    predicted_candidate: RepresentativeFinderCandidate | None = None
    predicted_output_path: Path | None = None
    solvent_completed_predicted_candidate: (
        RepresentativeFinderCandidate | None
    ) = None
    solvent_completed_predicted_output_path: Path | None = None
    predicted_generation_notes: tuple[str, ...] = ()

    def summary_text(self) -> str:
        lines = [
            "Representative structure selection complete",
            f"Generated at: {self.generated_at}",
            f"Input folder: {self.input_dir}",
            f"Output folder: {self.output_dir}",
            f"Stoichiometry label: {self.structure_label}",
            f"Candidate files analyzed: {len(self.candidates)}",
            f"Skipped files: {len(self.skipped_files)}",
            f"Selection algorithm: {self.settings.selection_algorithm}",
            (
                "Weights: "
                f"bond={self.settings.bond_weight:.3g}, "
                f"angle={self.settings.angle_weight:.3g}, "
                f"solvent={self.settings.solvent_weight:.3g}"
            ),
            "",
            "Selected representative",
            f"  File: {self.selected_candidate.file_name}",
            f"  Source: {self.selected_candidate.relative_label}",
            (
                "  Scores: "
                f"total={_format_score(self.selected_candidate.score_total)}, "
                f"bond={_format_score(self.selected_candidate.score_bond)}, "
                f"angle={_format_score(self.selected_candidate.score_angle)}, "
                f"solvent={_format_score(self.selected_candidate.score_solvent)}"
            ),
            (
                "  Solvent shell: "
                f"total={self.selected_candidate.solvent_atom_count}, "
                f"direct={self.selected_candidate.direct_solvent_atom_count}, "
                f"outer={self.selected_candidate.outer_solvent_atom_count}"
            ),
            f"  Copied output: {self.representative_output_path}",
        ]
        if self.selected_candidate.descriptor_notes:
            lines.extend(["", "Representative notes"])
            lines.extend(
                f"  {note}"
                for note in self.selected_candidate.descriptor_notes
            )
        if self.predicted_output_path is not None:
            predicted_candidate = self.predicted_candidate
            lines.extend(
                [
                    "",
                    "Predicted optimized representative",
                    "  File: "
                    + (
                        predicted_candidate.file_name
                        if predicted_candidate is not None
                        else self.predicted_output_path.name
                    ),
                    "  Output: " + str(self.predicted_output_path),
                ]
            )
            if predicted_candidate is not None:
                lines.append(
                    "  Scores: "
                    f"total={_format_score(predicted_candidate.score_total)}, "
                    f"bond={_format_score(predicted_candidate.score_bond)}, "
                    f"angle={_format_score(predicted_candidate.score_angle)}, "
                    f"solvent={_format_score(predicted_candidate.score_solvent)}"
                )
        if self.solvent_completed_predicted_output_path is not None:
            completed_candidate = self.solvent_completed_predicted_candidate
            lines.extend(
                [
                    "",
                    "Solvent-completed predicted representative",
                    "  File: "
                    + (
                        completed_candidate.file_name
                        if completed_candidate is not None
                        else self.solvent_completed_predicted_output_path.name
                    ),
                    "  Output: "
                    + str(self.solvent_completed_predicted_output_path),
                ]
            )
            if completed_candidate is not None:
                lines.append(
                    "  Scores: "
                    f"total={_format_score(completed_candidate.score_total)}, "
                    f"bond={_format_score(completed_candidate.score_bond)}, "
                    f"angle={_format_score(completed_candidate.score_angle)}, "
                    f"solvent={_format_score(completed_candidate.score_solvent)}"
                )
        if self.predicted_generation_notes:
            lines.extend(["", "Predicted representative notes"])
            lines.extend(
                f"  {note}" for note in self.predicted_generation_notes
            )
        if self.skipped_files:
            lines.extend(["", "Skipped files"])
            lines.extend(f"  {line}" for line in self.skipped_files[:12])
            if len(self.skipped_files) > 12:
                lines.append(f"  ... and {len(self.skipped_files) - 12} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 3,
            "generated_at": self.generated_at,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "structure_label": self.structure_label,
            "expected_core_counts": dict(
                sorted(self.expected_core_counts.items())
            ),
            "settings": self.settings.to_dict(),
            "representative_output_path": str(self.representative_output_path),
            "predicted_output_path": (
                None
                if self.predicted_output_path is None
                else str(self.predicted_output_path)
            ),
            "solvent_completed_predicted_output_path": (
                None
                if self.solvent_completed_predicted_output_path is None
                else str(self.solvent_completed_predicted_output_path)
            ),
            "summary_json_path": str(self.summary_json_path),
            "score_table_path": str(self.score_table_path),
            "summary_text_path": str(self.summary_text_path),
            "target_solvent_metrics": dict(
                sorted(self.target_solvent_metrics.items())
            ),
            "target_bond_values": _definition_value_map_to_list(
                self.target_bond_values
            ),
            "target_angle_values": _definition_value_map_to_list(
                self.target_angle_values
            ),
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "selected_candidate": self.selected_candidate.to_dict(),
            "predicted_candidate": (
                None
                if self.predicted_candidate is None
                else self.predicted_candidate.to_dict()
            ),
            "solvent_completed_predicted_candidate": (
                None
                if self.solvent_completed_predicted_candidate is None
                else self.solvent_completed_predicted_candidate.to_dict()
            ),
            "predicted_generation_notes": list(
                self.predicted_generation_notes
            ),
            "skipped_files": list(self.skipped_files),
        }

    def plot_series_for_candidate(
        self,
        candidate: RepresentativeFinderCandidate | None = None,
    ) -> tuple[RepresentativeFinderPlotSeries, ...]:
        active_candidate = candidate or self.selected_candidate
        bond_series = tuple(
            RepresentativeFinderPlotSeries(
                category="bond",
                display_label=definition.display_label,
                xlabel=f"{definition.display_label} distance (Angstrom)",
                distribution_values=np.asarray(
                    self.target_bond_values.get(
                        definition,
                        np.array([], dtype=float),
                    ),
                    dtype=float,
                ),
                candidate_values=_line_values(
                    active_candidate.bond_values.get(definition, [])
                ),
            )
            for definition in self.settings.bond_pairs
        )
        angle_series = tuple(
            RepresentativeFinderPlotSeries(
                category="angle",
                display_label=definition.display_label,
                xlabel=f"{definition.display_label} angle (deg)",
                distribution_values=np.asarray(
                    self.target_angle_values.get(
                        definition,
                        np.array([], dtype=float),
                    ),
                    dtype=float,
                ),
                candidate_values=_line_values(
                    active_candidate.angle_values.get(definition, [])
                ),
            )
            for definition in self.settings.angle_triplets
        )
        return bond_series + angle_series


def representativefinder_settings_from_dict(
    payload: object,
) -> RepresentativeFinderSettings:
    source = dict(payload) if isinstance(payload, dict) else {}
    quantile_values = source.get("quantiles") or _DEFAULT_QUANTILES
    quantiles = tuple(float(value) for value in quantile_values)
    return RepresentativeFinderSettings(
        selection_algorithm=str(
            source.get(
                "selection_algorithm",
                "target_distribution_quantile_distance",
            )
        ).strip()
        or "target_distribution_quantile_distance",
        bond_weight=_float_from_payload(source.get("bond_weight"), 1.0),
        angle_weight=_float_from_payload(source.get("angle_weight"), 1.0),
        solvent_weight=_float_from_payload(source.get("solvent_weight"), 1.0),
        generate_predicted_optimized_representative=bool(
            source.get("generate_predicted_optimized_representative", False)
        ),
        parallel_workers=_int_from_payload(source.get("parallel_workers"), 0),
        quantiles=quantiles or _DEFAULT_QUANTILES,
        bond_pairs=tuple(
            _bond_pair_definition_from_dict(entry)
            for entry in source.get("bond_pairs", [])
            if isinstance(entry, dict)
        ),
        angle_triplets=tuple(
            _angle_triplet_definition_from_dict(entry)
            for entry in source.get("angle_triplets", [])
            if isinstance(entry, dict)
        ),
    )


def load_representativefinder_result(
    result_json_path: str | Path,
) -> RepresentativeFinderResult:
    path = Path(result_json_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Representative finder result must be a JSON object: {path}"
        )
    return representativefinder_result_from_dict(payload, source_path=path)


def representativefinder_result_from_dict(
    payload: dict[str, object],
    *,
    source_path: str | Path | None = None,
) -> RepresentativeFinderResult:
    source = dict(payload)
    source_result_path = (
        None
        if source_path is None
        else Path(source_path).expanduser().resolve()
    )
    settings = representativefinder_settings_from_dict(source.get("settings"))
    candidates = tuple(
        _candidate_from_dict(candidate_payload, settings=settings)
        for candidate_payload in (source.get("candidates", []) or [])
        if isinstance(candidate_payload, dict)
    )
    selected_payload = source.get("selected_candidate")
    selected_candidate = (
        _matching_candidate_from_payload(candidates, selected_payload)
        if isinstance(selected_payload, dict)
        else None
    )
    if selected_candidate is None and isinstance(selected_payload, dict):
        selected_candidate = _candidate_from_dict(
            selected_payload,
            settings=settings,
        )
    if selected_candidate is None and candidates:
        selected_candidate = candidates[0]
    if selected_candidate is None:
        raise ValueError(
            "Representative finder result has no selected candidate."
        )
    if not candidates:
        candidates = (selected_candidate,)

    predicted_payload = source.get("predicted_candidate")
    predicted_candidate = (
        _candidate_from_dict(predicted_payload, settings=settings)
        if isinstance(predicted_payload, dict)
        else None
    )
    solvent_completed_payload = source.get(
        "solvent_completed_predicted_candidate"
    )
    solvent_completed_predicted_candidate = (
        _candidate_from_dict(solvent_completed_payload, settings=settings)
        if isinstance(solvent_completed_payload, dict)
        else None
    )

    summary_json_path = _path_from_payload(
        source.get("summary_json_path"),
        fallback=source_result_path,
    )
    output_dir = _path_from_payload(
        source.get("output_dir"),
        fallback=(
            summary_json_path.parent
            if summary_json_path is not None
            else Path.cwd()
        ),
    )
    return RepresentativeFinderResult(
        input_dir=_path_from_payload(
            source.get("input_dir"), fallback=Path.cwd()
        ),
        output_dir=output_dir,
        structure_label=str(source.get("structure_label", "")).strip()
        or output_dir.name,
        expected_core_counts={
            str(element): int(count)
            for element, count in dict(
                source.get("expected_core_counts", {}) or {}
            ).items()
        },
        settings=settings,
        generated_at=str(source.get("generated_at", "")).strip(),
        candidates=candidates,
        selected_candidate=selected_candidate,
        representative_output_path=_path_from_payload(
            source.get("representative_output_path"),
            fallback=selected_candidate.file_path,
        ),
        skipped_files=tuple(
            str(item) for item in (source.get("skipped_files", []) or [])
        ),
        target_bond_values=_definition_value_map_from_list(
            source.get("target_bond_values"),
            category="bond",
            array_values=True,
        ),
        target_angle_values=_definition_value_map_from_list(
            source.get("target_angle_values"),
            category="angle",
            array_values=True,
        ),
        target_solvent_metrics=_float_mapping_from_payload(
            source.get("target_solvent_metrics")
        ),
        summary_json_path=(
            summary_json_path or output_dir / "representative_selection.json"
        ),
        score_table_path=_path_from_payload(
            source.get("score_table_path"),
            fallback=output_dir / "candidate_scores.tsv",
        ),
        summary_text_path=_path_from_payload(
            source.get("summary_text_path"),
            fallback=output_dir / "selection_summary.txt",
        ),
        predicted_candidate=predicted_candidate,
        predicted_output_path=_optional_path_from_payload(
            source.get("predicted_output_path")
        ),
        solvent_completed_predicted_candidate=(
            solvent_completed_predicted_candidate
        ),
        solvent_completed_predicted_output_path=_optional_path_from_payload(
            source.get("solvent_completed_predicted_output_path")
        ),
        predicted_generation_notes=tuple(
            str(note)
            for note in (source.get("predicted_generation_notes", []) or [])
        ),
    )


def _candidate_from_dict(
    payload: dict[str, object],
    *,
    settings: RepresentativeFinderSettings,
) -> RepresentativeFinderCandidate:
    source = dict(payload)
    return RepresentativeFinderCandidate(
        file_path=_path_from_payload(
            source.get("file_path"), fallback=Path.cwd()
        ),
        relative_label=str(source.get("relative_label", "")).strip(),
        motif_label=str(source.get("motif_label", "no_motif")).strip()
        or "no_motif",
        atom_count=_int_from_payload(source.get("atom_count"), 0),
        element_counts={
            str(element): int(count)
            for element, count in dict(
                source.get("element_counts", {}) or {}
            ).items()
        },
        bond_values=_definition_value_map_from_list(
            source.get("bond_values"),
            category="bond",
            array_values=False,
        ),
        angle_values=_definition_value_map_from_list(
            source.get("angle_values"),
            category="angle",
            array_values=False,
        ),
        solvent_metrics=_float_mapping_from_payload(
            source.get("solvent_metrics")
        ),
        solvent_atom_count=_int_from_payload(
            source.get("solvent_atom_count"), 0
        ),
        direct_solvent_atom_count=_int_from_payload(
            source.get("direct_solvent_atom_count"),
            0,
        ),
        outer_solvent_atom_count=_int_from_payload(
            source.get("outer_solvent_atom_count"),
            0,
        ),
        mean_direct_solvent_coordination=_float_from_payload(
            source.get("mean_direct_solvent_coordination"),
            0.0,
        ),
        descriptor_notes=tuple(
            str(note) for note in (source.get("descriptor_notes", []) or [])
        ),
        score_total=_optional_float_from_payload(source.get("score_total")),
        score_bond=_optional_float_from_payload(source.get("score_bond")),
        score_angle=_optional_float_from_payload(source.get("score_angle")),
        score_solvent=_optional_float_from_payload(
            source.get("score_solvent")
        ),
    )


def _matching_candidate_from_payload(
    candidates: tuple[RepresentativeFinderCandidate, ...],
    payload: object,
) -> RepresentativeFinderCandidate | None:
    if not isinstance(payload, dict):
        return None
    target_path = str(payload.get("file_path", "")).strip()
    target_relative_label = str(payload.get("relative_label", "")).strip()
    for candidate in candidates:
        if target_path and str(candidate.file_path) == target_path:
            return candidate
        if (
            target_relative_label
            and candidate.relative_label == target_relative_label
        ):
            return candidate
    return None


def _definition_value_map_to_list(
    values_by_definition,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for definition, values in values_by_definition.items():
        rows.append(
            {
                "definition": definition.to_dict(),
                "values": _float_list(values),
            }
        )
    return rows


def _definition_value_map_from_list(
    payload: object,
    *,
    category: str,
    array_values: bool,
):
    if not isinstance(payload, list):
        return {}
    values_by_definition = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        definition_payload = entry.get("definition")
        if not isinstance(definition_payload, dict):
            continue
        definition = (
            _bond_pair_definition_from_dict(definition_payload)
            if category == "bond"
            else _angle_triplet_definition_from_dict(definition_payload)
        )
        values = _float_list(entry.get("values", []))
        values_by_definition[definition] = (
            np.asarray(values, dtype=float) if array_values else values
        )
    return values_by_definition


def _bond_pair_definition_from_dict(
    payload: dict[str, object]
) -> BondPairDefinition:
    return BondPairDefinition(
        str(payload["atom1"]),
        str(payload["atom2"]),
        float(payload["cutoff_angstrom"]),
    )


def _angle_triplet_definition_from_dict(
    payload: dict[str, object],
) -> AngleTripletDefinition:
    return AngleTripletDefinition(
        str(payload["vertex"]),
        str(payload["arm1"]),
        str(payload["arm2"]),
        float(payload["cutoff1_angstrom"]),
        float(payload["cutoff2_angstrom"]),
    )


def _float_mapping_from_payload(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in payload.items()
        if value is not None
    }


def _float_list(values: object) -> list[float]:
    if values is None:
        return []
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return []
    return [float(value) for value in array.reshape(-1).tolist()]


def _path_from_payload(value: object, *, fallback: Path | None) -> Path:
    text = str(value or "").strip()
    if not text:
        if fallback is None:
            return Path.cwd()
        return Path(fallback).expanduser().resolve()
    return Path(text).expanduser().resolve()


def _optional_path_from_payload(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return None
    return Path(text).expanduser().resolve()


def _optional_float_from_payload(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "n/a", "nan"}:
        return None
    return float(text)


def _float_from_payload(value: object, default: float) -> float:
    parsed = _optional_float_from_payload(value)
    return float(default) if parsed is None else parsed


def _int_from_payload(value: object, default: int) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    return int(text)


@dataclass(slots=True)
class _MeasuredCandidateStructure:
    candidate: RepresentativeFinderCandidate
    coordinates: np.ndarray
    elements: tuple[str, ...]
    parsed_structure: ParsedContrastStructure | None


@dataclass(slots=True, frozen=True)
class _SolventDescriptorMeasurement:
    candidate: RepresentativeFinderCandidate
    solvent_metrics: dict[str, float]
    solvent_atom_count: int
    direct_solvent_atom_count: int
    outer_solvent_atom_count: int
    mean_direct_solvent_coordination: float
    descriptor_notes: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _CandidateScoreMeasurement:
    candidate: RepresentativeFinderCandidate
    score_total: float
    score_bond: float
    score_angle: float
    score_solvent: float


def inspect_representative_structure_folder(
    input_dir: str | Path,
) -> RepresentativeFinderFolderInspection:
    resolved_input_dir = Path(input_dir).expanduser().resolve()
    candidates = _discover_candidate_files(resolved_input_dir)
    motif_labels = tuple(
        sorted(
            {
                candidate.motif_label
                for candidate in candidates
                if candidate.motif_label != "no_motif"
            },
            key=_natural_sort_key,
        )
    )
    direct_file_count = sum(
        1 for candidate in candidates if candidate.motif_label == "no_motif"
    )
    return RepresentativeFinderFolderInspection(
        input_dir=resolved_input_dir,
        structure_label=resolved_input_dir.name,
        candidate_count=len(candidates),
        direct_file_count=direct_file_count,
        motif_labels=motif_labels,
        candidate_labels=tuple(
            candidate.relative_label for candidate in candidates
        ),
    )


def inspect_representative_structure_input(
    input_dir: str | Path,
) -> RepresentativeFinderInputInspection:
    resolved_input_dir = Path(input_dir).expanduser().resolve()
    if not resolved_input_dir.is_dir():
        raise ValueError(
            f"Input directory does not exist: {resolved_input_dir}"
        )
    direct_inspection = inspect_representative_structure_folder(
        resolved_input_dir
    )
    if direct_inspection.candidate_count > 0:
        return RepresentativeFinderInputInspection(
            input_dir=resolved_input_dir,
            input_is_stoichiometry_folder=True,
            stoichiometry_folders=(direct_inspection,),
        )

    stoichiometry_folders = tuple(
        inspection
        for inspection in (
            inspect_representative_structure_folder(child)
            for child in sorted(
                [
                    child
                    for child in resolved_input_dir.iterdir()
                    if child.is_dir()
                ],
                key=lambda path: _natural_sort_key(path.name),
            )
        )
        if inspection.candidate_count > 0
    )
    if not stoichiometry_folders:
        raise ValueError(
            "No representative-structure candidate folders were found in "
            f"{resolved_input_dir}. Choose a stoichiometry folder directly, or "
            "choose a parent folder whose immediate subfolders are "
            "stoichiometries containing .xyz/.pdb cluster files."
        )
    return RepresentativeFinderInputInspection(
        input_dir=resolved_input_dir,
        input_is_stoichiometry_folder=False,
        stoichiometry_folders=stoichiometry_folders,
    )


def analyze_representative_structure_folder(
    input_dir: str | Path,
    *,
    settings: RepresentativeFinderSettings,
    output_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    progress_callback: RepresentativeFinderProgressCallback | None = None,
    log_callback: RepresentativeFinderLogCallback | None = None,
    cancel_callback: RepresentativeFinderCancelCallback | None = None,
) -> RepresentativeFinderResult:
    resolved_input_dir = Path(input_dir).expanduser().resolve()
    if not resolved_input_dir.is_dir():
        raise ValueError(
            f"Representative input directory does not exist: {resolved_input_dir}"
        )
    candidates_to_measure = _discover_candidate_files(resolved_input_dir)
    if not candidates_to_measure:
        raise ValueError(
            "No candidate .xyz or .pdb files were found in the selected "
            f"folder: {resolved_input_dir}"
        )
    candidate_count = len(candidates_to_measure)
    solvent_phase_enabled = settings.solvent_weight > 0.0
    predicted_phase_enabled = bool(
        settings.generate_predicted_optimized_representative
    )
    predicted_solvent_phase_enabled = (
        predicted_phase_enabled and project_dir is not None
    )
    total_work = estimate_representativefinder_total_work(
        candidate_count,
        solvent_phase_enabled=solvent_phase_enabled,
        predicted_phase_enabled=predicted_phase_enabled,
        predicted_solvent_phase_enabled=predicted_solvent_phase_enabled,
    )

    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else suggest_representativefinder_output_dir(resolved_input_dir)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_project_dir = (
        None
        if project_dir is None
        else Path(project_dir).expanduser().resolve()
    )
    distribution_store = StructureDistributionStore(
        application_structure_distribution_store_dir(
            project_dir=resolved_project_dir,
            output_dir=resolved_output_dir,
            application="representativefinder",
        )
    )
    analyzer = BondAnalyzer(
        bond_pairs=settings.bond_pairs,
        angle_triplets=settings.angle_triplets,
    )
    expected_core_counts = parse_stoich_label(resolved_input_dir.name)
    _emit_log(
        log_callback,
        f"Scanning {candidate_count} candidate structure file(s).",
    )
    _emit_progress(
        progress_callback,
        0,
        total_work,
        "Preparing representative-structure analysis...",
    )
    _raise_if_cancelled(cancel_callback)

    parallel_workers = _effective_parallel_workers(
        settings.parallel_workers,
        candidate_count,
    )
    processed_work = 0

    single_atom_structures = _inspect_single_atom_candidate_entries(
        candidates_to_measure,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        processed_work=processed_work,
        total_work=total_work,
    )
    if single_atom_structures is None:
        measured_structures, skipped_files, processed_work = (
            _measure_candidate_entries(
                candidates_to_measure,
                analyzer=analyzer,
                distribution_store=distribution_store,
                include_parsed_structure=settings.solvent_weight > 0.0,
                parallel_workers=parallel_workers,
                progress_callback=progress_callback,
                log_callback=log_callback,
                cancel_callback=cancel_callback,
                processed_work=processed_work,
                total_work=total_work,
            )
        )
    else:
        measured_structures = single_atom_structures
        skipped_files = []
        processed_work += len(single_atom_structures)
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "Inspected single-atom candidate structure files.",
        )
    distribution_store.flush()
    measured_candidates = [
        measured.candidate for measured in measured_structures
    ]
    parsed_structures = [
        parsed_structure
        for measured in measured_structures
        for parsed_structure in (measured.parsed_structure,)
        if parsed_structure is not None
    ]
    parsed_cache = {
        measured.candidate.file_path: measured.parsed_structure
        for measured in measured_structures
        if measured.parsed_structure is not None
    }

    if not measured_candidates:
        raise ValueError(
            "No valid candidate structures could be measured in the selected "
            "folder."
        )

    if _single_atom_shortcut_applies(measured_candidates):
        shortcut_note = (
            "Single-atom candidate structures were detected; bond, angle, "
            "and solvent-distribution scoring was skipped."
        )
        for candidate in measured_candidates:
            candidate.score_bond = 0.0
            candidate.score_angle = 0.0
            candidate.score_solvent = 0.0
            candidate.score_total = 0.0
            candidate.descriptor_notes = (shortcut_note,)

        processed_work = max(processed_work, total_work - 2)
        _emit_log(
            log_callback,
            "Detected a uniform single-atom candidate set; skipping full "
            "representative-distribution analysis.",
        )
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "Single-atom structures detected; selecting representative directly.",
        )
        ranked_candidates = tuple(
            sorted(
                measured_candidates,
                key=RepresentativeFinderCandidate.score_sort_key,
            )
        )
        selected_candidate = ranked_candidates[0]

        predicted_candidate = None
        predicted_output_path = None
        solvent_completed_predicted_candidate = None
        solvent_completed_predicted_output_path = None
        predicted_generation_notes: tuple[str, ...] = ()
        if settings.generate_predicted_optimized_representative:
            _raise_if_cancelled(cancel_callback)
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                "Generating predicted optimized representative...",
            )
            (
                predicted_candidate,
                predicted_output_path,
                solvent_completed_predicted_candidate,
                solvent_completed_predicted_output_path,
                predicted_generation_notes,
            ) = _build_optional_predicted_representatives(
                input_dir=resolved_input_dir,
                output_dir=resolved_output_dir,
                project_dir=resolved_project_dir,
                settings=settings,
                analyzer=analyzer,
                expected_core_counts=expected_core_counts,
                measured_candidates=measured_candidates,
                measured_structures=measured_structures,
                selected_candidate=selected_candidate,
                target_bond_features={},
                target_angle_features={},
                target_solvent_metrics={},
                pair_contact_distance_medians=None,
            )
            processed_work += 1
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                (
                    "Predicted optimized representative ready."
                    if predicted_output_path is not None
                    else (
                        "Predicted optimized representative unavailable for "
                        "this single-atom stoichiometry."
                    )
                ),
            )
            if resolved_project_dir is not None:
                _raise_if_cancelled(cancel_callback)
                processed_work += 1
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    (
                        "Predicted solvent-shell completion ready."
                        if solvent_completed_predicted_output_path is not None
                        else "Skipping solvent-completed predicted representative."
                    ),
                )

        _raise_if_cancelled(cancel_callback)
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "Writing representative outputs...",
        )
        representative_output_path = _copy_representative_file(
            input_dir=resolved_input_dir,
            output_dir=resolved_output_dir,
            selected_candidate=selected_candidate,
        )
        result = _build_representativefinder_result(
            input_dir=resolved_input_dir,
            output_dir=resolved_output_dir,
            expected_core_counts=expected_core_counts,
            settings=settings,
            candidates=ranked_candidates,
            selected_candidate=selected_candidate,
            representative_output_path=representative_output_path,
            skipped_files=tuple(skipped_files),
            target_bond_values={},
            target_angle_values={},
            target_solvent_metrics={},
            predicted_candidate=predicted_candidate,
            predicted_output_path=predicted_output_path,
            solvent_completed_predicted_candidate=(
                solvent_completed_predicted_candidate
            ),
            solvent_completed_predicted_output_path=(
                solvent_completed_predicted_output_path
            ),
            predicted_generation_notes=predicted_generation_notes,
        )
        _raise_if_cancelled(cancel_callback)
        _write_outputs(result)
        processed_work += 1
        _emit_progress(
            progress_callback,
            total_work,
            total_work,
            "Representative-structure selection complete.",
        )
        _emit_log(
            log_callback,
            f"Selected {selected_candidate.file_name} as the representative "
            "single-atom structure.",
        )
        return result

    if (
        not settings.bond_pairs
        and not settings.angle_triplets
        and settings.solvent_weight <= 0.0
    ):
        raise ValueError(
            "Provide at least one bond pair, one angle triplet, or a positive "
            "solvent weight before running representative selection."
        )

    _raise_if_cancelled(cancel_callback)
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Aggregating bond and angle distributions...",
    )
    target_bond_values, target_angle_values = _aggregate_candidate_values(
        measured_candidates,
        settings.bond_pairs,
        settings.angle_triplets,
    )
    processed_work += 1

    target_solvent_metrics: dict[str, float] = {}
    pair_contact_distance_medians: dict[tuple[str, str], float] | None = None
    if settings.solvent_weight > 0.0 and parsed_structures:
        _raise_if_cancelled(cancel_callback)
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "Building solvent contact-distance targets...",
        )
        pair_contact_distance_medians = estimate_pair_contact_distance_medians(
            tuple(parsed_structures)
        )
        processed_work += 1
        for candidate in measured_candidates:
            _raise_if_cancelled(cancel_callback)
            parsed_structure = parsed_cache.get(candidate.file_path)
            if parsed_structure is None:
                processed_work += 1
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    f"No solvent descriptor for {candidate.file_name}",
                )
                continue
        processed_work = _apply_solvent_descriptors(
            [
                (candidate, parsed_cache[candidate.file_path])
                for candidate in measured_candidates
                if candidate.file_path in parsed_cache
            ],
            expected_core_counts=expected_core_counts,
            pair_contact_distance_medians=pair_contact_distance_medians,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            processed_work=processed_work,
            total_work=total_work,
        )
        target_solvent_metrics = _median_summary(
            [
                candidate.solvent_metrics
                for candidate in measured_candidates
                if candidate.solvent_metrics
            ]
        )
    elif settings.solvent_weight > 0.0:
        _raise_if_cancelled(cancel_callback)
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "No solvent descriptors available; continuing with bond/angle scoring.",
        )
        processed_work += 1 + candidate_count
        _emit_log(
            log_callback,
            "No solvent descriptors could be computed; solvent weighting was "
            "ignored for this run.",
        )

    if settings.selection_algorithm == "target_distribution_moment_distance":
        target_bond_features = {
            definition: _moment_feature_vector(values)
            for definition, values in target_bond_values.items()
        }
        target_angle_features = {
            definition: _moment_feature_vector(values)
            for definition, values in target_angle_values.items()
        }
    else:
        target_bond_features = {
            definition: _quantile_feature_vector(
                values,
                quantiles=settings.quantiles,
            )
            for definition, values in target_bond_values.items()
        }
        target_angle_features = {
            definition: _quantile_feature_vector(
                values,
                quantiles=settings.quantiles,
            )
            for definition, values in target_angle_values.items()
        }

    processed_work = _score_measured_candidates(
        measured_candidates,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
        target_solvent_metrics=target_solvent_metrics,
        parallel_workers=parallel_workers,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        processed_work=processed_work,
        total_work=total_work,
    )

    _raise_if_cancelled(cancel_callback)
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Ranking candidate scores...",
    )
    ranked_candidates = tuple(
        sorted(
            measured_candidates,
            key=RepresentativeFinderCandidate.score_sort_key,
        )
    )
    selected_candidate = ranked_candidates[0]
    processed_work += 1

    predicted_candidate = None
    predicted_output_path = None
    solvent_completed_predicted_candidate = None
    solvent_completed_predicted_output_path = None
    predicted_generation_notes: tuple[str, ...] = ()
    if settings.generate_predicted_optimized_representative:
        _raise_if_cancelled(cancel_callback)
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            "Generating predicted optimized representative...",
        )
        if pair_contact_distance_medians is None and measured_structures:
            pair_contact_distance_medians = estimate_pair_contact_distance_medians(
                tuple(
                    (
                        measured.parsed_structure
                        if measured.parsed_structure is not None
                        else ParsedContrastStructure(
                            file_path=measured.candidate.file_path,
                            coordinates=np.asarray(
                                measured.coordinates, dtype=float
                            ),
                            elements=measured.elements,
                            element_counts=dict(
                                sorted(
                                    measured.candidate.element_counts.items()
                                )
                            ),
                        )
                    )
                    for measured in measured_structures
                )
            )
        (
            predicted_candidate,
            predicted_output_path,
            solvent_completed_predicted_candidate,
            solvent_completed_predicted_output_path,
            predicted_generation_notes,
        ) = _build_optional_predicted_representatives(
            input_dir=resolved_input_dir,
            output_dir=resolved_output_dir,
            project_dir=resolved_project_dir,
            settings=settings,
            analyzer=analyzer,
            expected_core_counts=expected_core_counts,
            measured_candidates=measured_candidates,
            measured_structures=measured_structures,
            selected_candidate=selected_candidate,
            target_bond_features=target_bond_features,
            target_angle_features=target_angle_features,
            target_solvent_metrics=target_solvent_metrics,
            pair_contact_distance_medians=pair_contact_distance_medians,
        )
        processed_work += 1
        _emit_progress(
            progress_callback,
            processed_work,
            total_work,
            (
                "Predicted optimized representative ready."
                if predicted_output_path is not None
                else "Predicted optimized representative unavailable."
            ),
        )
        if resolved_project_dir is not None:
            _raise_if_cancelled(cancel_callback)
            processed_work += 1
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                (
                    "Predicted solvent-shell completion ready."
                    if solvent_completed_predicted_output_path is not None
                    else "Skipping solvent-completed predicted representative."
                ),
            )

    _raise_if_cancelled(cancel_callback)
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Writing representative outputs...",
    )
    representative_output_path = _copy_representative_file(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        selected_candidate=selected_candidate,
    )

    result = _build_representativefinder_result(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        expected_core_counts=expected_core_counts,
        settings=settings,
        candidates=ranked_candidates,
        selected_candidate=selected_candidate,
        representative_output_path=representative_output_path,
        skipped_files=tuple(skipped_files),
        target_bond_values=target_bond_values,
        target_angle_values=target_angle_values,
        target_solvent_metrics=target_solvent_metrics,
        predicted_candidate=predicted_candidate,
        predicted_output_path=predicted_output_path,
        solvent_completed_predicted_candidate=(
            solvent_completed_predicted_candidate
        ),
        solvent_completed_predicted_output_path=(
            solvent_completed_predicted_output_path
        ),
        predicted_generation_notes=predicted_generation_notes,
    )
    _raise_if_cancelled(cancel_callback)
    _write_outputs(result)
    processed_work += 1
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Representative-structure selection complete.",
    )
    _emit_log(
        log_callback,
        f"Selected {selected_candidate.file_name} as the representative structure.",
    )
    return result


def estimate_representativefinder_total_work(
    candidate_count: int,
    *,
    solvent_phase_enabled: bool,
    predicted_phase_enabled: bool = False,
    predicted_solvent_phase_enabled: bool = False,
) -> int:
    total_work = int(candidate_count) + 1 + int(candidate_count) + 2
    if solvent_phase_enabled:
        total_work += 1 + int(candidate_count)
    if predicted_phase_enabled:
        total_work += 1
    if predicted_solvent_phase_enabled:
        total_work += 1
    return max(total_work, 1)


def suggest_representativefinder_output_dir(
    input_dir: str | Path,
    *,
    project_dir: str | Path | None = None,
    batch: bool = False,
) -> Path:
    source_dir = Path(input_dir).expanduser().resolve()
    if project_dir is not None:
        root = (
            Path(project_dir).expanduser().resolve() / "representative_finder"
        )
    else:
        root = source_dir.parent
    folder_name = _safe_folder_name(source_dir.name)
    prefix = "representativefinder_batch" if batch else "representativefinder"
    return _next_available_output_dir(
        root,
        f"{prefix}_{folder_name}",
    )


def suggest_representativefinder_target_output_dir(
    output_root_dir: str | Path,
    structure_label: str,
) -> Path:
    root_dir = Path(output_root_dir).expanduser().resolve()
    return _next_available_output_dir(
        root_dir,
        f"representativefinder_{_safe_folder_name(structure_label)}",
    )


def _discover_candidate_files(
    input_dir: Path,
) -> tuple[RepresentativeFinderFolderCandidate, ...]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    direct_files = _structure_files_in_dir(input_dir)
    candidates: list[RepresentativeFinderFolderCandidate] = [
        RepresentativeFinderFolderCandidate(
            file_path=file_path,
            relative_label=file_path.name,
            motif_label="no_motif",
        )
        for file_path in direct_files
    ]
    motif_dirs = sorted(
        [
            child
            for child in input_dir.iterdir()
            if child.is_dir() and child.name.startswith("motif_")
        ],
        key=lambda path: _natural_sort_key(path.name),
    )
    for motif_dir in motif_dirs:
        for file_path in _structure_files_in_dir(motif_dir):
            candidates.append(
                RepresentativeFinderFolderCandidate(
                    file_path=file_path,
                    relative_label=str(file_path.relative_to(input_dir)),
                    motif_label=motif_dir.name,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda entry: _natural_sort_key(entry.relative_label),
        )
    )


def _structure_files_in_dir(directory: Path) -> list[Path]:
    return sorted(
        [
            file_path
            for file_path in directory.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in _STRUCTURE_SUFFIXES
        ],
        key=lambda path: _natural_sort_key(path.name),
    )


def _effective_parallel_workers(
    configured_workers: int,
    item_count: int,
) -> int:
    if int(item_count) <= 1:
        return 1
    requested = int(configured_workers)
    if requested <= 0:
        env_value = os.environ.get("SAXSHELL_REPRESENTATIVEFINDER_WORKERS", "")
        if env_value.strip():
            try:
                requested = max(int(env_value), 1)
            except ValueError:
                return 1
        else:
            return 1
    return max(1, min(int(item_count), requested, 32))


def _inspect_single_atom_candidate_entries(
    entries: tuple[RepresentativeFinderFolderCandidate, ...],
    *,
    progress_callback: RepresentativeFinderProgressCallback | None,
    cancel_callback: RepresentativeFinderCancelCallback | None,
    processed_work: int,
    total_work: int,
) -> list[_MeasuredCandidateStructure] | None:
    if not entries:
        return None
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Inspecting candidate atom counts...",
    )
    scanned_elements_by_index: dict[int, tuple[str, ...]] = {}
    element_signatures: set[tuple[tuple[str, int], ...]] = set()
    for index, entry in enumerate(entries):
        _raise_if_cancelled(cancel_callback)
        try:
            elements = tuple(
                str(element).strip()
                for element in scan_structure_elements(entry.file_path)
                if str(element).strip()
            )
        except Exception:
            return None
        if len(elements) != 1:
            return None
        element_counts = Counter(elements)
        element_signatures.add(tuple(sorted(element_counts.items())))
        if len(element_signatures) > 1:
            return None
        scanned_elements_by_index[index] = elements

    measured_structures: list[_MeasuredCandidateStructure] = []
    for index, entry in enumerate(entries):
        _raise_if_cancelled(cancel_callback)
        try:
            coordinates, loaded_elements = load_structure_file(entry.file_path)
        except Exception:
            return None
        elements = tuple(str(element).strip() for element in loaded_elements)
        if len(elements) != 1:
            return None
        if elements != scanned_elements_by_index[index]:
            return None
        coordinates_array = np.asarray(coordinates, dtype=float)
        element_counts = Counter(elements)
        candidate = RepresentativeFinderCandidate(
            file_path=entry.file_path,
            relative_label=entry.relative_label,
            motif_label=entry.motif_label,
            atom_count=1,
            element_counts=dict(sorted(element_counts.items())),
            bond_values={},
            angle_values={},
            solvent_metrics={},
            solvent_atom_count=0,
            direct_solvent_atom_count=0,
            outer_solvent_atom_count=0,
            mean_direct_solvent_coordination=0.0,
        )
        measured_structures.append(
            _MeasuredCandidateStructure(
                candidate=candidate,
                coordinates=coordinates_array,
                elements=elements,
                parsed_structure=None,
            )
        )
    return measured_structures


def _measure_candidate_entries(
    entries: tuple[RepresentativeFinderFolderCandidate, ...],
    *,
    analyzer: BondAnalyzer,
    distribution_store: StructureDistributionStore,
    include_parsed_structure: bool,
    parallel_workers: int,
    progress_callback: RepresentativeFinderProgressCallback | None,
    log_callback: RepresentativeFinderLogCallback | None,
    cancel_callback: RepresentativeFinderCancelCallback | None,
    processed_work: int,
    total_work: int,
) -> tuple[list[_MeasuredCandidateStructure], list[str], int]:
    measured_by_index: dict[int, _MeasuredCandidateStructure] = {}
    skipped_by_index: dict[int, str] = {}
    worker_count = _effective_parallel_workers(parallel_workers, len(entries))
    if worker_count <= 1:
        for index, entry in enumerate(entries):
            _raise_if_cancelled(cancel_callback)
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Measuring {entry.file_path.name}",
            )
            try:
                measured = _measure_candidate_structure_file(
                    entry.file_path,
                    relative_label=entry.relative_label,
                    motif_label=entry.motif_label,
                    analyzer=analyzer,
                    distribution_store=distribution_store,
                    include_parsed_structure=include_parsed_structure,
                )
            except Exception as exc:
                skipped_by_index[index] = f"{entry.relative_label}: {exc}"
                _emit_log(
                    log_callback,
                    "Skipped unreadable structure "
                    f"{entry.relative_label}: {exc}",
                )
                processed_work += 1
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    f"Skipped {entry.file_path.name}",
                )
                continue
            measured_by_index[index] = measured
            processed_work += 1
            _emit_log(
                log_callback,
                "Measured "
                f"{entry.relative_label} ({len(measured.elements)} atoms).",
            )
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Measured {entry.file_path.name}",
            )
        return (
            [measured_by_index[index] for index in sorted(measured_by_index)],
            [skipped_by_index[index] for index in sorted(skipped_by_index)],
            processed_work,
        )

    _emit_log(
        log_callback,
        "Measuring "
        f"{len(entries)} candidate structure file(s) with {worker_count} "
        "worker threads.",
    )
    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Measuring "
        f"{len(entries)} candidate structure file(s) with {worker_count} "
        "worker thread(s)...",
    )
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="representativefinder-measure",
    ) as executor:
        futures = {}
        try:
            for index, entry in enumerate(entries):
                _raise_if_cancelled(cancel_callback)
                futures[
                    executor.submit(
                        _measure_candidate_structure_file,
                        entry.file_path,
                        relative_label=entry.relative_label,
                        motif_label=entry.motif_label,
                        analyzer=analyzer,
                        distribution_store=distribution_store,
                        include_parsed_structure=include_parsed_structure,
                    )
                ] = (index, entry)
            for future in as_completed(futures):
                _raise_if_cancelled(cancel_callback)
                index, entry = futures[future]
                try:
                    measured = future.result()
                except Exception as exc:
                    skipped_by_index[index] = f"{entry.relative_label}: {exc}"
                    _emit_log(
                        log_callback,
                        "Skipped unreadable structure "
                        f"{entry.relative_label}: {exc}",
                    )
                    processed_work += 1
                    _emit_progress(
                        progress_callback,
                        processed_work,
                        total_work,
                        f"Skipped {entry.file_path.name}",
                    )
                    continue
                measured_by_index[index] = measured
                processed_work += 1
                _emit_log(
                    log_callback,
                    "Measured "
                    f"{entry.relative_label} ({len(measured.elements)} atoms).",
                )
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    f"Measured {entry.file_path.name}",
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return (
        [measured_by_index[index] for index in sorted(measured_by_index)],
        [skipped_by_index[index] for index in sorted(skipped_by_index)],
        processed_work,
    )


def _apply_solvent_descriptors(
    candidate_rows: list[
        tuple[RepresentativeFinderCandidate, ParsedContrastStructure]
    ],
    *,
    expected_core_counts: dict[str, int],
    pair_contact_distance_medians: dict[tuple[str, str], float],
    parallel_workers: int,
    progress_callback: RepresentativeFinderProgressCallback | None,
    cancel_callback: RepresentativeFinderCancelCallback | None,
    processed_work: int,
    total_work: int,
) -> int:
    worker_count = _effective_parallel_workers(
        parallel_workers,
        len(candidate_rows),
    )
    if worker_count <= 1:
        for candidate, parsed_structure in candidate_rows:
            _raise_if_cancelled(cancel_callback)
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Building solvent descriptor for {candidate.file_name}",
            )
            measurement = _measure_solvent_descriptor(
                candidate,
                parsed_structure=parsed_structure,
                expected_core_counts=expected_core_counts,
                pair_contact_distance_medians=pair_contact_distance_medians,
            )
            _assign_solvent_descriptor_measurement(measurement)
            processed_work += 1
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Built solvent descriptor for {candidate.file_name}",
            )
        return processed_work

    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Building solvent descriptors for "
        f"{len(candidate_rows)} candidate structure(s) with {worker_count} "
        "worker thread(s)...",
    )
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="representativefinder-solvent",
    ) as executor:
        futures = {}
        try:
            for candidate, parsed_structure in candidate_rows:
                _raise_if_cancelled(cancel_callback)
                futures[
                    executor.submit(
                        _measure_solvent_descriptor,
                        candidate,
                        parsed_structure=parsed_structure,
                        expected_core_counts=expected_core_counts,
                        pair_contact_distance_medians=(
                            pair_contact_distance_medians
                        ),
                    )
                ] = candidate
            for future in as_completed(futures):
                _raise_if_cancelled(cancel_callback)
                candidate = futures[future]
                measurement = future.result()
                _assign_solvent_descriptor_measurement(measurement)
                processed_work += 1
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    f"Built solvent descriptor for {candidate.file_name}",
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return processed_work


def _measure_solvent_descriptor(
    candidate: RepresentativeFinderCandidate,
    *,
    parsed_structure: ParsedContrastStructure,
    expected_core_counts: dict[str, int],
    pair_contact_distance_medians: dict[tuple[str, str], float],
) -> _SolventDescriptorMeasurement:
    descriptor = describe_parsed_contrast_structure(
        parsed_structure,
        expected_core_counts=expected_core_counts,
        pair_contact_distance_medians=pair_contact_distance_medians,
        include_geometry_metrics=False,
    )
    return _SolventDescriptorMeasurement(
        candidate=candidate,
        solvent_metrics=descriptor.solvent_metrics(),
        solvent_atom_count=descriptor.solvent_atom_count,
        direct_solvent_atom_count=descriptor.direct_solvent_atom_count,
        outer_solvent_atom_count=descriptor.outer_solvent_atom_count,
        mean_direct_solvent_coordination=float(
            descriptor.mean_direct_solvent_coordination
        ),
        descriptor_notes=tuple(descriptor.notes),
    )


def _assign_solvent_descriptor_measurement(
    measurement: _SolventDescriptorMeasurement,
) -> None:
    candidate = measurement.candidate
    candidate.solvent_metrics = measurement.solvent_metrics
    candidate.solvent_atom_count = measurement.solvent_atom_count
    candidate.direct_solvent_atom_count = measurement.direct_solvent_atom_count
    candidate.outer_solvent_atom_count = measurement.outer_solvent_atom_count
    candidate.mean_direct_solvent_coordination = (
        measurement.mean_direct_solvent_coordination
    )
    candidate.descriptor_notes = measurement.descriptor_notes


def _score_measured_candidates(
    candidates: list[RepresentativeFinderCandidate],
    *,
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
    parallel_workers: int,
    progress_callback: RepresentativeFinderProgressCallback | None,
    cancel_callback: RepresentativeFinderCancelCallback | None,
    processed_work: int,
    total_work: int,
) -> int:
    worker_count = _effective_parallel_workers(
        parallel_workers, len(candidates)
    )
    if worker_count <= 1:
        for candidate in candidates:
            _raise_if_cancelled(cancel_callback)
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Scoring {candidate.file_name}",
            )
            measurement = _measure_candidate_score(
                candidate,
                settings=settings,
                target_bond_features=target_bond_features,
                target_angle_features=target_angle_features,
                target_solvent_metrics=target_solvent_metrics,
            )
            _assign_candidate_score_measurement(measurement)
            processed_work += 1
            _emit_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Scored {candidate.file_name}",
            )
        return processed_work

    _emit_progress(
        progress_callback,
        processed_work,
        total_work,
        "Scoring "
        f"{len(candidates)} candidate structure(s) with {worker_count} "
        "worker thread(s)...",
    )
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="representativefinder-score",
    ) as executor:
        futures = {}
        try:
            for candidate in candidates:
                _raise_if_cancelled(cancel_callback)
                futures[
                    executor.submit(
                        _measure_candidate_score,
                        candidate,
                        settings=settings,
                        target_bond_features=target_bond_features,
                        target_angle_features=target_angle_features,
                        target_solvent_metrics=target_solvent_metrics,
                    )
                ] = candidate
            for future in as_completed(futures):
                _raise_if_cancelled(cancel_callback)
                candidate = futures[future]
                measurement = future.result()
                _assign_candidate_score_measurement(measurement)
                processed_work += 1
                _emit_progress(
                    progress_callback,
                    processed_work,
                    total_work,
                    f"Scored {candidate.file_name}",
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return processed_work


def _measure_candidate_score(
    candidate: RepresentativeFinderCandidate,
    *,
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
) -> _CandidateScoreMeasurement:
    if settings.selection_algorithm == "target_distribution_moment_distance":
        bond_score = _category_moment_distance_from_features(
            target_bond_features,
            candidate.bond_values,
        )
        angle_score = _category_moment_distance_from_features(
            target_angle_features,
            candidate.angle_values,
        )
    else:
        bond_score = _category_distance_from_features(
            target_bond_features,
            candidate.bond_values,
            quantiles=settings.quantiles,
        )
        angle_score = _category_distance_from_features(
            target_angle_features,
            candidate.angle_values,
            quantiles=settings.quantiles,
        )
    solvent_score = _score_feature_map(
        candidate.solvent_metrics,
        target_solvent_metrics,
        default_scale=1.0,
    )
    total_score = float(
        settings.bond_weight * bond_score
        + settings.angle_weight * angle_score
        + settings.solvent_weight * solvent_score
    )
    return _CandidateScoreMeasurement(
        candidate=candidate,
        score_total=total_score,
        score_bond=float(bond_score),
        score_angle=float(angle_score),
        score_solvent=float(solvent_score),
    )


def _assign_candidate_score_measurement(
    measurement: _CandidateScoreMeasurement,
) -> None:
    candidate = measurement.candidate
    candidate.score_bond = measurement.score_bond
    candidate.score_angle = measurement.score_angle
    candidate.score_solvent = measurement.score_solvent
    candidate.score_total = measurement.score_total


def _aggregate_candidate_values(
    candidates: list[RepresentativeFinderCandidate],
    bond_pairs: tuple[BondPairDefinition, ...],
    angle_triplets: tuple[AngleTripletDefinition, ...],
) -> tuple[
    dict[BondPairDefinition, np.ndarray],
    dict[AngleTripletDefinition, np.ndarray],
]:
    target_bonds: dict[BondPairDefinition, np.ndarray] = {}
    target_angles: dict[AngleTripletDefinition, np.ndarray] = {}
    for definition in bond_pairs:
        merged = [
            value
            for candidate in candidates
            for value in candidate.bond_values.get(definition, [])
        ]
        target_bonds[definition] = np.asarray(merged, dtype=float)
    for definition in angle_triplets:
        merged = [
            value
            for candidate in candidates
            for value in candidate.angle_values.get(definition, [])
        ]
        target_angles[definition] = np.asarray(merged, dtype=float)
    return target_bonds, target_angles


def _median_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    values_by_key: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            values_by_key[str(key)].append(float(value))
    return {
        key: float(np.median(np.asarray(values, dtype=float)))
        for key, values in sorted(values_by_key.items())
        if values
    }


def _score_feature_map(
    candidate_values: dict[str, float],
    target_values: dict[str, float],
    *,
    default_scale: float,
    missing_penalty: float = 1.0,
) -> float:
    if not target_values:
        return 0.0
    deltas = [
        _relative_difference(
            candidate_values.get(key),
            expected_value,
            scale_floor=default_scale,
            missing_penalty=missing_penalty,
        )
        for key, expected_value in sorted(target_values.items())
    ]
    if not deltas:
        return 0.0
    return float(np.mean(np.asarray(deltas, dtype=float)))


def _relative_difference(
    observed: float | None,
    expected: float,
    *,
    scale_floor: float = 1.0,
    missing_penalty: float = 1.0,
) -> float:
    if observed is None:
        return float(missing_penalty)
    scale = max(abs(float(expected)), float(scale_floor))
    return abs(float(observed) - float(expected)) / scale


def _category_distance(
    target_values: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
    *,
    quantiles: tuple[float, ...],
) -> float:
    target_features = {
        definition: _quantile_feature_vector(target, quantiles=quantiles)
        for definition, target in target_values.items()
    }
    return _category_distance_from_features(
        target_features,
        candidate_values,
        quantiles=quantiles,
    )


def _category_distance_from_features(
    target_features: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
    *,
    quantiles: tuple[float, ...],
) -> float:
    distances: list[float] = []
    for definition, target_vector in target_features.items():
        candidate_vector = _quantile_feature_vector(
            candidate_values.get(definition, []),
            quantiles=quantiles,
        )
        distances.append(
            float(np.linalg.norm(candidate_vector - target_vector))
        )
    if not distances:
        return 0.0
    return float(np.mean(np.asarray(distances, dtype=float)))


def _category_moment_distance(
    target_values: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
) -> float:
    target_features = {
        definition: _moment_feature_vector(target)
        for definition, target in target_values.items()
    }
    return _category_moment_distance_from_features(
        target_features,
        candidate_values,
    )


def _category_moment_distance_from_features(
    target_features: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
) -> float:
    distances: list[float] = []
    for definition, target_vector in target_features.items():
        candidate_vector = _moment_feature_vector(
            candidate_values.get(definition, [])
        )
        distances.append(
            float(np.linalg.norm(candidate_vector - target_vector))
        )
    if not distances:
        return 0.0
    return float(np.mean(np.asarray(distances, dtype=float)))


def _quantile_feature_vector(
    values: list[float] | np.ndarray,
    *,
    quantiles: tuple[float, ...],
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return np.zeros(len(quantiles), dtype=float)
    return np.asarray(np.quantile(array, quantiles), dtype=float)


def _moment_feature_vector(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return np.zeros(3, dtype=float)
    return np.asarray(
        [
            float(np.mean(array)),
            float(np.std(array)),
            float(np.median(array)),
        ],
        dtype=float,
    )


def _line_values(values: list[float] | np.ndarray) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return ()
    return tuple(float(value) for value in np.unique(array))


def _single_atom_shortcut_applies(
    candidates: list[RepresentativeFinderCandidate],
) -> bool:
    if not candidates:
        return False
    if any(int(candidate.atom_count) != 1 for candidate in candidates):
        return False
    element_signatures = {
        tuple(sorted(candidate.element_counts.items()))
        for candidate in candidates
    }
    return len(element_signatures) == 1


def _build_representativefinder_result(
    *,
    input_dir: Path,
    output_dir: Path,
    expected_core_counts: dict[str, int],
    settings: RepresentativeFinderSettings,
    candidates: tuple[RepresentativeFinderCandidate, ...],
    selected_candidate: RepresentativeFinderCandidate,
    representative_output_path: Path,
    skipped_files: tuple[str, ...],
    target_bond_values: dict[BondPairDefinition, np.ndarray],
    target_angle_values: dict[AngleTripletDefinition, np.ndarray],
    target_solvent_metrics: dict[str, float],
    predicted_candidate: RepresentativeFinderCandidate | None = None,
    predicted_output_path: Path | None = None,
    solvent_completed_predicted_candidate: (
        RepresentativeFinderCandidate | None
    ) = None,
    solvent_completed_predicted_output_path: Path | None = None,
    predicted_generation_notes: tuple[str, ...] = (),
) -> RepresentativeFinderResult:
    return RepresentativeFinderResult(
        input_dir=input_dir,
        output_dir=output_dir,
        structure_label=input_dir.name,
        expected_core_counts=dict(sorted(expected_core_counts.items())),
        settings=settings,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        candidates=candidates,
        selected_candidate=selected_candidate,
        representative_output_path=representative_output_path,
        skipped_files=skipped_files,
        target_bond_values=target_bond_values,
        target_angle_values=target_angle_values,
        target_solvent_metrics=target_solvent_metrics,
        summary_json_path=output_dir / "representative_selection.json",
        score_table_path=output_dir / "candidate_scores.tsv",
        summary_text_path=output_dir / "selection_summary.txt",
        predicted_candidate=predicted_candidate,
        predicted_output_path=predicted_output_path,
        solvent_completed_predicted_candidate=(
            solvent_completed_predicted_candidate
        ),
        solvent_completed_predicted_output_path=(
            solvent_completed_predicted_output_path
        ),
        predicted_generation_notes=predicted_generation_notes,
    )


def _copy_representative_file(
    *,
    input_dir: Path,
    output_dir: Path,
    selected_candidate: RepresentativeFinderCandidate,
) -> Path:
    relative_label = re.sub(
        r"[^0-9A-Za-z._-]+",
        "_",
        selected_candidate.relative_label,
    ).strip("_")
    destination_name = (
        f"{_safe_folder_name(input_dir.name)}__representative__"
        f"{relative_label or selected_candidate.file_name}"
    )
    destination = output_dir / destination_name
    shutil.copy2(selected_candidate.file_path, destination)
    return destination


def _write_outputs(result: RepresentativeFinderResult) -> None:
    result.summary_json_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    with result.score_table_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "file_name",
                "relative_label",
                "motif_label",
                "score_total",
                "score_bond",
                "score_angle",
                "score_solvent",
                "atom_count",
                "solvent_atom_count",
                "direct_solvent_atom_count",
                "outer_solvent_atom_count",
                "mean_direct_solvent_coordination",
            ]
        )
        for rank, candidate in enumerate(result.candidates, start=1):
            writer.writerow(
                [
                    rank,
                    candidate.file_name,
                    candidate.relative_label,
                    candidate.motif_label,
                    _format_score(candidate.score_total),
                    _format_score(candidate.score_bond),
                    _format_score(candidate.score_angle),
                    _format_score(candidate.score_solvent),
                    candidate.atom_count,
                    candidate.solvent_atom_count,
                    candidate.direct_solvent_atom_count,
                    candidate.outer_solvent_atom_count,
                    f"{candidate.mean_direct_solvent_coordination:.8f}",
                ]
            )
    result.summary_text_path.write_text(
        result.summary_text() + "\n",
        encoding="utf-8",
    )


def _build_optional_predicted_representatives(
    *,
    input_dir: Path,
    output_dir: Path,
    project_dir: Path | None,
    settings: RepresentativeFinderSettings,
    analyzer: BondAnalyzer,
    expected_core_counts: dict[str, int],
    measured_candidates: list[RepresentativeFinderCandidate],
    measured_structures: list[_MeasuredCandidateStructure],
    selected_candidate: RepresentativeFinderCandidate,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
    pair_contact_distance_medians: dict[tuple[str, str], float] | None,
) -> tuple[
    RepresentativeFinderCandidate | None,
    Path | None,
    RepresentativeFinderCandidate | None,
    Path | None,
    tuple[str, ...],
]:
    notes: list[str] = []
    core_counts = {
        str(element).strip(): int(count)
        for element, count in expected_core_counts.items()
        if str(element).strip() and int(count) > 0
    }
    if not core_counts:
        notes.append(
            "Predicted optimized representative skipped because the folder name "
            "did not provide a parseable stoichiometric core."
        )
        return None, None, None, None, tuple(notes)
    if sum(core_counts.values()) <= 1:
        return _build_single_atom_predicted_representatives(
            input_dir=input_dir,
            output_dir=output_dir,
            settings=settings,
            analyzer=analyzer,
            measured_structures=measured_structures,
            selected_candidate=selected_candidate,
            target_bond_features=target_bond_features,
            target_angle_features=target_angle_features,
            target_solvent_metrics=target_solvent_metrics,
        )

    atom_type_definitions = _infer_predicted_atom_type_definitions(
        core_counts=core_counts,
        measured_candidates=measured_candidates,
        settings=settings,
    )
    pair_cutoff_definitions = _infer_predicted_pair_cutoff_definitions(
        settings=settings,
        core_counts=core_counts,
        pair_contact_distance_medians=pair_contact_distance_medians,
    )
    if not atom_type_definitions.get("node"):
        notes.append(
            "Predicted optimized representative skipped because no core node "
            "elements could be inferred from the current stoichiometry."
        )
        return None, None, None, None, tuple(notes)
    if not pair_cutoff_definitions:
        notes.append(
            "Predicted optimized representative skipped because no geometry "
            "cutoffs were available from the current bond and angle settings."
        )
        return None, None, None, None, tuple(notes)

    try:
        from saxshell.clusterdynamicsml.workflow import (
            ClusterDynamicsMLTrainingObservation,
            ClusterDynamicsMLWorkflow,
        )
    except Exception as exc:
        notes.append(
            "Predicted optimized representative skipped because the Cluster "
            f"Dynamics ML scaffold builder was unavailable: {exc}"
        )
        return None, None, None, None, tuple(notes)

    try:
        workflow = ClusterDynamicsMLWorkflow(
            frames_dir=selected_candidate.file_path.parent,
            atom_type_definitions=atom_type_definitions,
            pair_cutoff_definitions=pair_cutoff_definitions,
            clusters_dir=input_dir,
            project_dir=project_dir,
        )
        training_observations, source_observation = (
            _build_predicted_training_observations(
                observation_cls=ClusterDynamicsMLTrainingObservation,
                core_counts=core_counts,
                node_elements=workflow._atom_type_elements("node"),
                measured_structures=measured_structures,
                selected_candidate=selected_candidate,
            )
        )
        geometry_statistics = workflow._collect_training_geometry_statistics(
            training_observations
        )
        predicted_max_radius = _predicted_target_max_radius(
            measured_structures,
            core_elements=set(core_counts),
        )
        generated_elements, generated_coordinates = (
            workflow._generate_predicted_structure(
                source_observation,
                target_counts=core_counts,
                predicted_max_radius=predicted_max_radius,
                geometry_statistics=geometry_statistics,
            )
        )
    except Exception as exc:
        notes.append(
            "Predicted optimized representative generation failed while "
            f"building the synthetic scaffold: {exc}"
        )
        return None, None, None, None, tuple(notes)

    generated_array = np.asarray(generated_coordinates, dtype=float)
    if not generated_elements or generated_array.size <= 0:
        notes.append(
            "Predicted optimized representative generation produced an empty "
            "structure."
        )
        return None, None, None, None, tuple(notes)

    refined_coordinates = _refine_predicted_coordinates(
        elements=tuple(str(element) for element in generated_elements),
        coordinates=generated_array,
        analyzer=analyzer,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
    )

    predicted_output_path = output_dir / (
        f"{_safe_folder_name(input_dir.name)}"
        "__predicted_optimized_representative.xyz"
    )
    _write_xyz_structure_file(
        predicted_output_path,
        tuple(str(element) for element in generated_elements),
        refined_coordinates,
        comment=(
            "Predicted optimized representative generated from aggregate "
            "geometry targets"
        ),
    )
    predicted_measured = _measure_candidate_structure_file(
        predicted_output_path,
        relative_label=predicted_output_path.name,
        motif_label="predicted_optimized",
        analyzer=analyzer,
        include_parsed_structure=False,
    )
    predicted_candidate = predicted_measured.candidate
    predicted_candidate.descriptor_notes = (
        "Synthetic predicted structure generated from aggregate bond and "
        "angle targets.",
    )
    _score_candidate_against_targets(
        predicted_candidate,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
        target_solvent_metrics=target_solvent_metrics,
        include_solvent_component=False,
    )
    notes.append(
        "Predicted optimized representative generated with the Cluster "
        "Dynamics ML geometry scaffold and locally refined against the "
        "current geometric scoring target."
    )

    solvent_completed_predicted_candidate = None
    solvent_completed_predicted_output_path = None
    if project_dir is not None:
        (
            solvent_completed_predicted_candidate,
            solvent_completed_predicted_output_path,
            solvent_completion_notes,
        ) = _build_solvent_completed_predicted_representative(
            project_dir=project_dir,
            output_dir=output_dir,
            predicted_output_path=predicted_output_path,
            analyzer=analyzer,
            expected_core_counts=core_counts,
            settings=settings,
            target_bond_features=target_bond_features,
            target_angle_features=target_angle_features,
            target_solvent_metrics=target_solvent_metrics,
            pair_contact_distance_medians=pair_contact_distance_medians,
        )
        notes.extend(solvent_completion_notes)

    return (
        predicted_candidate,
        predicted_output_path,
        solvent_completed_predicted_candidate,
        solvent_completed_predicted_output_path,
        tuple(notes),
    )


def _build_single_atom_predicted_representatives(
    *,
    input_dir: Path,
    output_dir: Path,
    settings: RepresentativeFinderSettings,
    analyzer: BondAnalyzer,
    measured_structures: list[_MeasuredCandidateStructure],
    selected_candidate: RepresentativeFinderCandidate,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
) -> tuple[
    RepresentativeFinderCandidate | None,
    Path | None,
    RepresentativeFinderCandidate | None,
    Path | None,
    tuple[str, ...],
]:
    if not measured_structures:
        return (
            None,
            None,
            None,
            None,
            (
                "Predicted optimized representative skipped because no measured "
                "single-atom structure was available.",
            ),
        )
    source_structure = measured_structures[0]
    predicted_output_path = output_dir / (
        f"{_safe_folder_name(input_dir.name)}"
        "__predicted_optimized_representative.xyz"
    )
    _write_xyz_structure_file(
        predicted_output_path,
        source_structure.elements,
        source_structure.coordinates,
        comment=(
            "Predicted optimized representative copied from the single-atom "
            "source structure"
        ),
    )
    predicted_measured = _measure_candidate_structure_file(
        predicted_output_path,
        relative_label=predicted_output_path.name,
        motif_label="predicted_optimized",
        analyzer=analyzer,
        include_parsed_structure=False,
    )
    predicted_candidate = predicted_measured.candidate
    predicted_candidate.descriptor_notes = (
        "Single-atom stoichiometry: predicted optimized representative is "
        "identical to the observed representative.",
    )
    _score_candidate_against_targets(
        predicted_candidate,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
        target_solvent_metrics=target_solvent_metrics,
        include_solvent_component=False,
    )
    predicted_candidate.score_total = 0.0
    predicted_candidate.score_bond = 0.0
    predicted_candidate.score_angle = 0.0
    predicted_candidate.score_solvent = 0.0
    return (
        predicted_candidate,
        predicted_output_path,
        None,
        None,
        (
            "Single-atom stoichiometry: the predicted optimized representative "
            "is the same single-atom structure as the observed representative.",
        ),
    )


def _build_solvent_completed_predicted_representative(
    *,
    project_dir: Path,
    output_dir: Path,
    predicted_output_path: Path,
    analyzer: BondAnalyzer,
    expected_core_counts: dict[str, int],
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
    pair_contact_distance_medians: dict[tuple[str, str], float] | None,
) -> tuple[
    RepresentativeFinderCandidate | None,
    Path | None,
    tuple[str, ...],
]:
    notes: list[str] = []
    try:
        from saxshell.fullrmc.project_model import ensure_rmcsetup_structure
        from saxshell.fullrmc.solvent_handling import (
            load_solvent_handling_metadata,
        )
        from saxshell.fullrmc.solvent_shell_builder import (
            analyze_solvent_shell,
            build_solvent_shell_output,
            default_director_atom_name,
        )
    except Exception as exc:
        return (
            None,
            None,
            ("Predicted solvent-shell completion was unavailable: " f"{exc}",),
        )

    try:
        rmcsetup_paths = ensure_rmcsetup_structure(project_dir)
        solvent_metadata = load_solvent_handling_metadata(
            rmcsetup_paths.solvent_handling_path
        )
    except Exception as exc:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion could not read the project "
                f"solvent settings: {exc}",
            ),
        )
    if solvent_metadata is None:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion was skipped because the "
                "project does not yet have solvent-handling settings.",
            ),
        )

    reference_identifier = _solvent_reference_identifier_from_metadata(
        solvent_metadata
    )
    settings_payload = solvent_metadata.settings
    director_atom_name = (
        settings_payload.director_atom_name
        or default_director_atom_name(reference_identifier)
    )
    if not director_atom_name:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion was skipped because no "
                "director atom could be resolved for the project solvent reference.",
            ),
        )
    try:
        analysis_result = analyze_solvent_shell(
            predicted_output_path,
            reference_identifier,
            reference_match_tolerance_a=(
                settings_payload.reference_match_tolerance_a
            ),
        )
    except Exception as exc:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion was skipped because the "
                f"predicted structure could not be analyzed for solvent anchors: {exc}",
            ),
        )

    solute_distance_cutoffs = {
        str(element): float(setting.director_distance_cutoff_a)
        for element, setting in settings_payload.solute_atom_settings.items()
        if element in analysis_result.solute_element_counts
        and float(setting.director_distance_cutoff_a) > 0.0
    }
    coordinating_center_elements = tuple(
        sorted(
            element
            for element, setting in settings_payload.solute_atom_settings.items()
            if element in analysis_result.solute_element_counts
            and setting.coordination_center
            and float(setting.target_coordination_number) > 0.0
        )
    )
    target_coordination_numbers = {
        str(element): float(setting.target_coordination_number)
        for element, setting in settings_payload.solute_atom_settings.items()
        if element in analysis_result.solute_element_counts
        and setting.coordination_center
        and float(setting.target_coordination_number) > 0.0
    }
    if not solute_distance_cutoffs and not target_coordination_numbers:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion was skipped because the "
                "project solvent settings do not define any active shell-building "
                "cutoffs or coordination targets for this stoichiometry.",
            ),
        )

    solvent_completed_output_path = output_dir / (
        f"{_safe_folder_name(predicted_output_path.stem)}"
        "__solvent_completed.pdb"
    )
    try:
        build_result = build_solvent_shell_output(
            predicted_output_path,
            reference_identifier,
            output_path=solvent_completed_output_path,
            director_atom_name=director_atom_name,
            minimum_solvent_atom_separation_a=(
                settings_payload.minimum_solvent_atom_separation_a
            ),
            solute_distance_cutoffs_a=solute_distance_cutoffs,
            coordinating_center_elements=coordinating_center_elements,
            target_average_coordination_numbers=target_coordination_numbers,
            reference_match_tolerance_a=(
                settings_payload.reference_match_tolerance_a
            ),
            analysis_result=analysis_result,
        )
    except Exception as exc:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion failed while building the "
                f"solvent shell: {exc}",
            ),
        )
    if int(build_result.solvent_molecules_added) <= 0:
        return (
            None,
            None,
            (
                "Predicted solvent-shell completion finished without placing any "
                "solvent molecules, so only the no-solvent predicted structure was kept.",
            ),
        )

    measured_completed = _measure_candidate_structure_file(
        solvent_completed_output_path,
        relative_label=solvent_completed_output_path.name,
        motif_label="predicted_optimized_solvent_completed",
        analyzer=analyzer,
        include_parsed_structure=True,
    )
    completed_candidate = measured_completed.candidate
    if (
        measured_completed.parsed_structure is not None
        and pair_contact_distance_medians is not None
    ):
        _apply_solvent_descriptor(
            completed_candidate,
            parsed_structure=measured_completed.parsed_structure,
            expected_core_counts=expected_core_counts,
            pair_contact_distance_medians=pair_contact_distance_medians,
        )
    completed_candidate.descriptor_notes = (
        "Synthetic predicted structure with a solvent shell built from the "
        "current project solvent settings.",
    )
    _score_candidate_against_targets(
        completed_candidate,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
        target_solvent_metrics=target_solvent_metrics,
        include_solvent_component=bool(target_solvent_metrics),
    )
    notes.append(
        "Built a solvent-completed predicted representative from the project "
        f"solvent settings ({build_result.solvent_molecules_added} solvent "
        "molecule(s) added)."
    )
    return completed_candidate, solvent_completed_output_path, tuple(notes)


def _infer_predicted_atom_type_definitions(
    *,
    core_counts: dict[str, int],
    measured_candidates: list[RepresentativeFinderCandidate],
    settings: RepresentativeFinderSettings,
):
    core_elements = {
        str(element).strip() for element in core_counts if str(element).strip()
    }
    observed_elements = {
        str(element).strip()
        for candidate in measured_candidates
        for element in candidate.element_counts
        if str(element).strip()
    }
    node_elements = [
        definition.vertex
        for definition in settings.angle_triplets
        if definition.vertex in core_elements
    ]
    if not node_elements:
        node_elements = [
            definition.atom1
            for definition in settings.bond_pairs
            if definition.atom1 in core_elements
        ]
    if not node_elements and core_counts:
        minimum_count = min(int(count) for count in core_counts.values())
        node_elements = [
            str(element)
            for element, count in sorted(core_counts.items())
            if int(count) == minimum_count
        ]
    normalized_node_elements = tuple(dict.fromkeys(node_elements))
    linker_elements = tuple(
        element
        for element in sorted(core_elements)
        if element not in normalized_node_elements
    )
    if not normalized_node_elements:
        normalized_node_elements = tuple(sorted(core_elements))
        linker_elements = ()
    shell_elements = tuple(
        sorted(
            element
            for element in observed_elements
            if element not in core_elements
        )
    )
    definitions = {
        "node": [(element, None) for element in normalized_node_elements],
    }
    if linker_elements:
        definitions["linker"] = [
            (element, None) for element in linker_elements
        ]
    if shell_elements:
        definitions["shell"] = [(element, None) for element in shell_elements]
    return definitions


def _infer_predicted_pair_cutoff_definitions(
    *,
    settings: RepresentativeFinderSettings,
    core_counts: dict[str, int],
    pair_contact_distance_medians: dict[tuple[str, str], float] | None,
):
    pair_cutoffs: defaultdict[tuple[str, str], dict[int, float]] = defaultdict(
        dict
    )

    def add_cutoff(element_a: str, element_b: str, cutoff: float) -> None:
        normalized_pair = tuple(
            sorted((str(element_a).strip(), str(element_b).strip()))
        )
        if not normalized_pair[0] or not normalized_pair[1]:
            return
        previous = pair_cutoffs[normalized_pair].get(0, 0.0)
        pair_cutoffs[normalized_pair][0] = max(previous, float(cutoff))

    for definition in settings.bond_pairs:
        add_cutoff(
            definition.atom1,
            definition.atom2,
            max(float(definition.cutoff_angstrom), 0.1),
        )
    for definition in settings.angle_triplets:
        add_cutoff(
            definition.vertex,
            definition.arm1,
            max(float(definition.cutoff1_angstrom), 0.1),
        )
        add_cutoff(
            definition.vertex,
            definition.arm2,
            max(float(definition.cutoff2_angstrom), 0.1),
        )
    if pair_cutoffs or not pair_contact_distance_medians:
        return {
            pair: dict(levels) for pair, levels in sorted(pair_cutoffs.items())
        }

    core_elements = set(core_counts)
    for pair, median_distance in sorted(pair_contact_distance_medians.items()):
        if pair[0] not in core_elements or pair[1] not in core_elements:
            continue
        add_cutoff(
            pair[0],
            pair[1],
            max(float(median_distance) * 1.15, float(median_distance) + 0.05),
        )
    return {
        pair: dict(levels) for pair, levels in sorted(pair_cutoffs.items())
    }


def _build_predicted_training_observations(
    *,
    observation_cls,
    core_counts: dict[str, int],
    node_elements: set[str],
    measured_structures: list[_MeasuredCandidateStructure],
    selected_candidate: RepresentativeFinderCandidate,
):
    observations = []
    source_observation = None
    cluster_size = int(sum(core_counts.values()))
    node_count = (
        int(sum(core_counts.get(element, 0) for element in node_elements))
        or cluster_size
    )
    selected_path = selected_candidate.file_path.resolve()
    for measured in sorted(
        measured_structures,
        key=lambda row: str(row.candidate.file_path),
    ):
        representative_path = measured.candidate.file_path.resolve()
        observation = observation_cls(
            label=measured.candidate.relative_label,
            node_count=node_count,
            cluster_size=cluster_size,
            element_counts=dict(sorted(core_counts.items())),
            file_count=1,
            representative_path=representative_path,
            structure_dir=representative_path,
            motifs=(
                (measured.candidate.motif_label,)
                if measured.candidate.motif_label != "no_motif"
                else ()
            ),
            mean_atom_count=float(measured.candidate.atom_count),
            mean_radius_of_gyration=0.0,
            mean_max_radius=_max_radius_from_coordinates(
                _filtered_coordinates_for_elements(
                    measured.coordinates,
                    measured.elements,
                    set(core_counts),
                )
            ),
            mean_semiaxis_a=0.0,
            mean_semiaxis_b=0.0,
            mean_semiaxis_c=0.0,
            total_observations=1,
            occupied_frames=1,
            mean_count_per_frame=1.0,
            occupancy_fraction=1.0,
            association_events=0,
            dissociation_events=0,
            association_rate_per_ps=0.0,
            dissociation_rate_per_ps=0.0,
            completed_lifetime_count=0,
            window_truncated_lifetime_count=0,
            mean_lifetime_fs=None,
            std_lifetime_fs=None,
        )
        observations.append(observation)
        if representative_path == selected_path:
            source_observation = observation
    if not observations:
        raise ValueError(
            "No measured structures were available for prediction."
        )
    return observations, source_observation or observations[0]


def _predicted_target_max_radius(
    measured_structures: list[_MeasuredCandidateStructure],
    *,
    core_elements: set[str],
) -> float:
    radii = [
        _max_radius_from_coordinates(
            _filtered_coordinates_for_elements(
                measured.coordinates,
                measured.elements,
                core_elements,
            )
        )
        for measured in measured_structures
    ]
    finite_radii = [
        float(radius)
        for radius in radii
        if np.isfinite(float(radius)) and float(radius) > 0.0
    ]
    if not finite_radii:
        return 1.0
    return float(np.median(np.asarray(finite_radii, dtype=float)))


def _measure_candidate_structure_file(
    file_path: Path,
    *,
    relative_label: str,
    motif_label: str,
    analyzer: BondAnalyzer,
    include_parsed_structure: bool,
    distribution_store: StructureDistributionStore | None = None,
) -> _MeasuredCandidateStructure:
    coordinates, elements = load_structure_file(file_path)
    coordinates_array = np.asarray(coordinates, dtype=float)
    normalized_elements = tuple(str(element).strip() for element in elements)
    active_store = distribution_store or StructureDistributionStore(
        application_structure_distribution_store_dir(
            output_dir=file_path.parent,
            application="representativefinder",
        )
    )
    measurement = active_store.measure_structure_data(
        file_path,
        coordinates_array,
        normalized_elements,
        bond_pairs=analyzer.bond_pairs,
        angle_triplets=analyzer.angle_triplets,
        relative_label=relative_label,
        motif_label=motif_label,
        autosave=distribution_store is None,
    )
    bond_values = measurement.bond_values
    angle_values = measurement.angle_values
    element_counts = Counter(normalized_elements)
    parsed_structure = None
    if include_parsed_structure:
        parsed_structure = ParsedContrastStructure(
            file_path=file_path,
            coordinates=coordinates_array,
            elements=normalized_elements,
            element_counts=dict(sorted(element_counts.items())),
        )
    candidate = RepresentativeFinderCandidate(
        file_path=file_path,
        relative_label=relative_label,
        motif_label=motif_label,
        atom_count=len(normalized_elements),
        element_counts=dict(sorted(element_counts.items())),
        bond_values=bond_values,
        angle_values=angle_values,
        solvent_metrics={},
        solvent_atom_count=0,
        direct_solvent_atom_count=0,
        outer_solvent_atom_count=0,
        mean_direct_solvent_coordination=0.0,
    )
    return _MeasuredCandidateStructure(
        candidate=candidate,
        coordinates=coordinates_array,
        elements=normalized_elements,
        parsed_structure=parsed_structure,
    )


def _apply_solvent_descriptor(
    candidate: RepresentativeFinderCandidate,
    *,
    parsed_structure: ParsedContrastStructure,
    expected_core_counts: dict[str, int],
    pair_contact_distance_medians: dict[tuple[str, str], float],
) -> None:
    descriptor = describe_parsed_contrast_structure(
        parsed_structure,
        expected_core_counts=expected_core_counts,
        pair_contact_distance_medians=pair_contact_distance_medians,
        include_geometry_metrics=False,
    )
    candidate.solvent_metrics = descriptor.solvent_metrics()
    candidate.solvent_atom_count = descriptor.solvent_atom_count
    candidate.direct_solvent_atom_count = descriptor.direct_solvent_atom_count
    candidate.outer_solvent_atom_count = descriptor.outer_solvent_atom_count
    candidate.mean_direct_solvent_coordination = float(
        descriptor.mean_direct_solvent_coordination
    )
    candidate.descriptor_notes = tuple(descriptor.notes)


def _score_candidate_against_targets(
    candidate: RepresentativeFinderCandidate,
    *,
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
    target_solvent_metrics: dict[str, float],
    include_solvent_component: bool,
) -> None:
    if settings.selection_algorithm == "target_distribution_moment_distance":
        bond_score = _category_moment_distance_from_features(
            target_bond_features,
            candidate.bond_values,
        )
        angle_score = _category_moment_distance_from_features(
            target_angle_features,
            candidate.angle_values,
        )
    else:
        bond_score = _category_distance_from_features(
            target_bond_features,
            candidate.bond_values,
            quantiles=settings.quantiles,
        )
        angle_score = _category_distance_from_features(
            target_angle_features,
            candidate.angle_values,
            quantiles=settings.quantiles,
        )
    solvent_score: float | None
    total_score = float(
        settings.bond_weight * bond_score + settings.angle_weight * angle_score
    )
    if include_solvent_component:
        solvent_score = _score_feature_map(
            candidate.solvent_metrics,
            target_solvent_metrics,
            default_scale=1.0,
        )
        total_score += settings.solvent_weight * float(solvent_score)
    else:
        solvent_score = 0.0 if settings.solvent_weight <= 0.0 else None
    candidate.score_bond = float(bond_score)
    candidate.score_angle = float(angle_score)
    candidate.score_solvent = solvent_score
    candidate.score_total = total_score


def _refine_predicted_coordinates(
    *,
    elements: tuple[str, ...],
    coordinates: np.ndarray,
    analyzer: BondAnalyzer,
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
) -> np.ndarray:
    coordinate_array = np.asarray(coordinates, dtype=float)
    if (
        coordinate_array.ndim != 2
        or coordinate_array.shape[0] <= 1
        or (not target_bond_features and not target_angle_features)
    ):
        return coordinate_array
    seed_basis = "|".join(elements) + f":{coordinate_array.shape[0]}"
    rng = np.random.default_rng(_stable_seed_from_text(seed_basis))
    best_coordinates = coordinate_array.copy()
    best_score = _geometry_only_score_from_coordinates(
        elements=elements,
        coordinates=best_coordinates,
        analyzer=analyzer,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
    )
    for step in range(48):
        perturbation_scale = max(0.03, 0.22 * (0.95**step))
        trial = best_coordinates.copy()
        trial[1:] += rng.normal(
            0.0,
            perturbation_scale,
            size=trial[1:].shape,
        )
        trial -= np.mean(trial, axis=0, keepdims=True)
        trial_score = _geometry_only_score_from_coordinates(
            elements=elements,
            coordinates=trial,
            analyzer=analyzer,
            settings=settings,
            target_bond_features=target_bond_features,
            target_angle_features=target_angle_features,
        )
        if trial_score < best_score:
            best_coordinates = trial
            best_score = trial_score
    return best_coordinates


def _geometry_only_score_from_coordinates(
    *,
    elements: tuple[str, ...],
    coordinates: np.ndarray,
    analyzer: BondAnalyzer,
    settings: RepresentativeFinderSettings,
    target_bond_features: dict[object, np.ndarray],
    target_angle_features: dict[object, np.ndarray],
) -> float:
    bond_values, angle_values = analyzer.measure_structure_data(
        np.asarray(coordinates, dtype=float),
        elements,
    )
    temp_candidate = RepresentativeFinderCandidate(
        file_path=Path("."),
        relative_label="predicted_trial",
        motif_label="predicted_trial",
        atom_count=len(elements),
        element_counts=dict(sorted(Counter(elements).items())),
        bond_values=bond_values,
        angle_values=angle_values,
        solvent_metrics={},
        solvent_atom_count=0,
        direct_solvent_atom_count=0,
        outer_solvent_atom_count=0,
        mean_direct_solvent_coordination=0.0,
    )
    _score_candidate_against_targets(
        temp_candidate,
        settings=settings,
        target_bond_features=target_bond_features,
        target_angle_features=target_angle_features,
        target_solvent_metrics={},
        include_solvent_component=False,
    )
    return float(temp_candidate.score_total or 0.0)


def _filtered_coordinates_for_elements(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    allowed_elements: set[str],
) -> np.ndarray:
    if not allowed_elements:
        return np.asarray(coordinates, dtype=float)
    indices = [
        index
        for index, element in enumerate(elements)
        if element in allowed_elements
    ]
    if not indices:
        return np.asarray(coordinates, dtype=float)
    return np.asarray(coordinates, dtype=float)[indices]


def _max_radius_from_coordinates(coordinates: np.ndarray) -> float:
    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.size <= 0:
        return 0.0
    centered = coordinate_array - np.mean(
        coordinate_array,
        axis=0,
        keepdims=True,
    )
    radial = np.linalg.norm(centered, axis=1)
    if radial.size <= 0:
        return 0.0
    return float(np.max(radial))


def _write_xyz_structure_file(
    output_path: Path,
    elements: tuple[str, ...],
    coordinates: np.ndarray,
    *,
    comment: str,
) -> Path:
    coordinate_array = np.asarray(coordinates, dtype=float)
    lines = [str(len(elements)), str(comment).strip() or output_path.stem]
    for element, (x_coord, y_coord, z_coord) in zip(
        elements,
        coordinate_array,
        strict=False,
    ):
        lines.append(f"{element} {x_coord:.6f} {y_coord:.6f} {z_coord:.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _stable_seed_from_text(text: str) -> int:
    seed = 0
    for character in str(text):
        seed = (seed * 131 + ord(character)) % (2**32)
    return int(seed)


def persist_representativefinder_result_to_project(
    project_dir: str | Path,
    result: RepresentativeFinderResult,
) -> Path:
    from saxshell.fullrmc.project_model import ensure_rmcsetup_structure
    from saxshell.fullrmc.representatives import (
        DistributionSelectionEntry,
        DistributionSelectionMetadata,
        RepresentativeSelectionEntry,
        RepresentativeSelectionMetadata,
        RepresentativeSelectionSettings,
        load_representative_selection_metadata,
        save_representative_selection_metadata,
    )
    from saxshell.fullrmc.solvent_handling import (
        load_solvent_handling_metadata,
        save_solvent_handling_metadata,
    )
    from saxshell.saxs.project_manager import DreamBestFitSelection

    resolved_project_dir = Path(project_dir).expanduser().resolve()
    rmcsetup_paths = ensure_rmcsetup_structure(resolved_project_dir)
    solvent_metadata = load_solvent_handling_metadata(
        rmcsetup_paths.solvent_handling_path
    )
    source_solvent_mode = _classify_project_representative_source_solvent_mode(
        result,
        solvent_metadata=solvent_metadata,
        rmcsetup_paths=rmcsetup_paths,
    )
    shared_output_path, mirrored_output_paths = (
        _copy_project_representative_file(
            rmcsetup_paths,
            result,
            source_solvent_mode=source_solvent_mode,
        )
    )
    project_cached_results_path = _write_project_cached_result(
        rmcsetup_paths,
        result,
    )
    updated_key = (
        result.structure_label,
        "no_motif",
        result.structure_label,
    )

    now = datetime.now().isoformat(timespec="seconds")
    existing_metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    _remove_stale_project_representative_artifacts(
        rmcsetup_paths,
        updated_key,
        existing_metadata=existing_metadata,
        solvent_metadata=solvent_metadata,
        preserved_paths=tuple(mirrored_output_paths),
    )
    selection = (
        existing_metadata.selection
        if existing_metadata is not None
        else DreamBestFitSelection(
            run_name="Representative Structure Finder",
            run_relative_path=str(
                rmcsetup_paths.representative_clusters_dir.relative_to(
                    resolved_project_dir
                )
            ),
            label="Representative Structure Finder",
            selection_source="representativefinder",
            selected_at=now,
        )
    )
    selection.run_relative_path = str(
        rmcsetup_paths.representative_clusters_dir.relative_to(
            resolved_project_dir
        )
    )
    selection.label = selection.label or "Representative Structure Finder"
    selection.selection_source = (
        selection.selection_source or "representativefinder"
    )
    selection.selected_at = now

    updated_entry = RepresentativeSelectionEntry(
        structure=result.structure_label,
        motif="no_motif",
        param=result.structure_label,
        selected_weight=0.0,
        cluster_count=max(len(result.candidates), 1),
        source_dir=str(shared_output_path.parent),
        source_file=str(shared_output_path),
        source_file_name=shared_output_path.name,
        atom_count=int(result.selected_candidate.atom_count),
        element_counts=dict(
            sorted(result.selected_candidate.element_counts.items())
        ),
        source_solvent_mode=source_solvent_mode,
        analysis_source=(
            "representativefinder:"
            f"{result.selected_candidate.relative_label}"
        ),
        score_total=_optional_float(result.selected_candidate.score_total),
        score_bond=_optional_float(result.selected_candidate.score_bond),
        score_angle=_optional_float(result.selected_candidate.score_angle),
        cached_results_path=str(result.summary_json_path),
        project_cached_results_path=str(project_cached_results_path),
    )

    merged_entries = _merge_project_representative_entries(
        existing_metadata,
        updated_entry,
    )
    _reweight_project_representative_entries(merged_entries)
    distribution_entries = [
        DistributionSelectionEntry(
            param=entry.param,
            structure=entry.structure,
            motif=entry.motif,
            selected_weight=float(entry.selected_weight),
            vary=True,
            cluster_count=int(entry.cluster_count),
            source_dir=entry.source_dir,
            source_file=entry.source_file,
            source_file_name=entry.source_file_name,
            source_kind="representative_structure",
            is_active=True,
        )
        for entry in merged_entries
    ]
    distribution_selection = DistributionSelectionMetadata(
        selection_mode="representative_finder",
        selection=selection,
        run_dir=str(rmcsetup_paths.representative_clusters_dir),
        updated_at=now,
        entries=distribution_entries,
    )
    metadata = RepresentativeSelectionMetadata(
        selection_mode="representative_finder",
        selection=selection,
        distribution_selection=distribution_selection,
        settings=RepresentativeSelectionSettings(
            selection_mode="representative_finder",
            selection_algorithm=result.settings.selection_algorithm,
            minimum_cluster_count_for_analysis=1,
            bond_weight=float(result.settings.bond_weight),
            angle_weight=float(result.settings.angle_weight),
            quantiles=tuple(
                float(value) for value in result.settings.quantiles
            ),
            bond_pairs=tuple(result.settings.bond_pairs),
            angle_triplets=tuple(result.settings.angle_triplets),
        ),
        updated_at=now,
        representative_entries=merged_entries,
        missing_bins=[],
        invalid_bins=[],
    )
    save_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path,
        metadata,
    )
    if solvent_metadata is not None and solvent_metadata.entries:
        filtered_entries = [
            entry
            for entry in solvent_metadata.entries
            if not _entry_matches_project_representative_key(
                entry,
                updated_key,
            )
        ]
        if len(filtered_entries) != len(solvent_metadata.entries):
            solvent_metadata.entries = filtered_entries
            solvent_metadata.updated_at = now
            save_solvent_handling_metadata(
                rmcsetup_paths.solvent_handling_path,
                solvent_metadata,
            )
    return shared_output_path


def _write_project_cached_result(
    rmcsetup_paths,
    result: RepresentativeFinderResult,
) -> Path:
    cache_dir = (
        rmcsetup_paths.representative_clusters_dir
        / "analysis_cache"
        / _safe_folder_name(result.structure_label)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_result_path = cache_dir / "representative_selection.json"
    cached_score_table_path = cache_dir / "candidate_scores.tsv"
    cached_summary_text_path = cache_dir / "selection_summary.txt"

    payload = result.to_dict()
    payload["project_cached_at"] = datetime.now().isoformat(timespec="seconds")
    payload["summary_json_path"] = str(cached_result_path)
    payload["score_table_path"] = str(cached_score_table_path)
    payload["summary_text_path"] = str(cached_summary_text_path)
    cached_result_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if result.score_table_path.is_file():
        shutil.copy2(result.score_table_path, cached_score_table_path)
    else:
        _write_candidate_score_table(result, cached_score_table_path)
    if result.summary_text_path.is_file():
        shutil.copy2(result.summary_text_path, cached_summary_text_path)
    else:
        cached_summary_text_path.write_text(
            result.summary_text() + "\n",
            encoding="utf-8",
        )
    return cached_result_path.resolve()


def _write_candidate_score_table(
    result: RepresentativeFinderResult,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "file_name",
                "relative_label",
                "motif_label",
                "score_total",
                "score_bond",
                "score_angle",
                "score_solvent",
                "atom_count",
                "solvent_atom_count",
                "direct_solvent_atom_count",
                "outer_solvent_atom_count",
                "mean_direct_solvent_coordination",
            ]
        )
        for rank, candidate in enumerate(result.candidates, start=1):
            writer.writerow(
                [
                    rank,
                    candidate.file_name,
                    candidate.relative_label,
                    candidate.motif_label,
                    _format_score(candidate.score_total),
                    _format_score(candidate.score_bond),
                    _format_score(candidate.score_angle),
                    _format_score(candidate.score_solvent),
                    candidate.atom_count,
                    candidate.solvent_atom_count,
                    candidate.direct_solvent_atom_count,
                    candidate.outer_solvent_atom_count,
                    f"{candidate.mean_direct_solvent_coordination:.8f}",
                ]
            )


def _copy_project_representative_file(
    rmcsetup_paths,
    result: RepresentativeFinderResult,
    *,
    source_solvent_mode: str,
) -> tuple[Path, tuple[Path, ...]]:
    relative_label = re.sub(
        r"[^0-9A-Za-z._-]+",
        "_",
        result.selected_candidate.relative_label,
    ).strip("_")
    destination_name = (
        f"{_safe_folder_name(result.structure_label)}__representative__"
        f"{relative_label or result.selected_candidate.file_name}"
    )
    destination_paths: list[Path] = []
    for destination_root in _project_representative_destination_roots(
        rmcsetup_paths,
        result,
        source_solvent_mode=source_solvent_mode,
    ):
        destination_dir = destination_root / _safe_folder_name(
            result.structure_label
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / destination_name
        shutil.copy2(result.selected_candidate.file_path, destination)
        destination_paths.append(destination.resolve())
    primary_destination = next(
        (
            path
            for path in destination_paths
            if path.parent.parent.resolve()
            == _project_representative_destination_root(
                rmcsetup_paths,
                source_solvent_mode=source_solvent_mode,
            ).resolve()
        ),
        destination_paths[0],
    )
    return primary_destination, tuple(destination_paths)


def _project_representative_destination_roots(
    rmcsetup_paths,
    result: RepresentativeFinderResult,
    *,
    source_solvent_mode: str,
) -> tuple[Path, ...]:
    primary_root = _project_representative_destination_root(
        rmcsetup_paths,
        source_solvent_mode=source_solvent_mode,
    )
    if int(result.selected_candidate.atom_count) != 1:
        return (primary_root,)
    return (
        rmcsetup_paths.pdb_no_solvent_dir,
        rmcsetup_paths.representative_partial_solvent_dir,
        rmcsetup_paths.pdb_with_solvent_dir,
    )


def _project_representative_destination_root(
    rmcsetup_paths,
    *,
    source_solvent_mode: str,
) -> Path:
    normalized = str(source_solvent_mode).strip().lower()
    if normalized == "nosolv":
        return rmcsetup_paths.pdb_no_solvent_dir
    if normalized == "fullsolv":
        return rmcsetup_paths.pdb_with_solvent_dir
    return rmcsetup_paths.representative_partial_solvent_dir


def _classify_project_representative_source_solvent_mode(
    result: RepresentativeFinderResult,
    *,
    solvent_metadata,
    rmcsetup_paths,
) -> str:
    candidate = result.selected_candidate
    if int(candidate.solvent_atom_count) <= 0:
        return "nosolv"
    inferred_from_path = _infer_project_representative_source_mode_from_path(
        candidate.file_path,
        rmcsetup_paths=rmcsetup_paths,
    )
    if inferred_from_path is not None:
        return inferred_from_path
    if solvent_metadata is None:
        return "partialsolv"
    try:
        from saxshell.fullrmc.solvent_shell_builder import (
            analyze_solvent_shell,
        )
    except Exception:
        return "partialsolv"
    try:
        analysis_result = analyze_solvent_shell(
            candidate.file_path,
            _solvent_reference_identifier_from_metadata(solvent_metadata),
            reference_match_tolerance_a=(
                solvent_metadata.settings.reference_match_tolerance_a
            ),
        )
    except Exception:
        return "partialsolv"
    if analysis_result.complete_solvent_molecule_count > 0:
        if analysis_result.partial_solvent_molecule_count > 0:
            return "partialsolv"
        return "fullsolv"
    if analysis_result.partial_solvent_molecule_count > 0:
        return "partialsolv"
    return "nosolv"


def _infer_project_representative_source_mode_from_path(
    file_path: Path,
    *,
    rmcsetup_paths,
) -> str | None:
    resolved_path = Path(file_path).expanduser().resolve()
    for mode, directory in (
        ("nosolv", rmcsetup_paths.pdb_no_solvent_dir),
        ("partialsolv", rmcsetup_paths.representative_partial_solvent_dir),
        ("fullsolv", rmcsetup_paths.pdb_with_solvent_dir),
    ):
        try:
            resolved_path.relative_to(directory.resolve())
        except ValueError:
            continue
        return mode
    return None


def _remove_stale_project_representative_artifacts(
    rmcsetup_paths,
    target_key: tuple[str, str, str],
    *,
    existing_metadata,
    solvent_metadata,
    preserved_paths: tuple[Path, ...] = (),
) -> None:
    preserved = {Path(path).expanduser().resolve() for path in preserved_paths}
    candidate_paths: set[Path] = set()
    if existing_metadata is not None:
        candidate_paths.update(
            _tracked_representative_source_paths_for_key(
                existing_metadata,
                target_key,
            )
        )
    if solvent_metadata is not None:
        candidate_paths.update(
            _tracked_solvent_output_paths_for_key(
                solvent_metadata,
                target_key,
            )
        )
    allowed_roots = (
        rmcsetup_paths.representative_clusters_dir.resolve(),
        rmcsetup_paths.representative_partial_solvent_dir.resolve(),
        rmcsetup_paths.pdb_no_solvent_dir.resolve(),
        rmcsetup_paths.pdb_with_solvent_dir.resolve(),
    )
    for path in candidate_paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in preserved or not resolved.is_file():
            continue
        if not any(
            _path_is_within_dir(resolved, root) for root in allowed_roots
        ):
            continue
        resolved.unlink()


def _tracked_representative_source_paths_for_key(
    metadata,
    target_key: tuple[str, str, str],
) -> set[Path]:
    return {
        Path(entry.source_file).expanduser().resolve()
        for entry in metadata.representative_entries
        if _entry_matches_project_representative_key(entry, target_key)
        and str(entry.source_file).strip()
    }


def _tracked_solvent_output_paths_for_key(
    solvent_metadata,
    target_key: tuple[str, str, str],
) -> set[Path]:
    tracked_paths: set[Path] = set()
    for entry in solvent_metadata.entries:
        if not _entry_matches_project_representative_key(entry, target_key):
            continue
        for candidate in (entry.no_solvent_pdb, entry.completed_pdb):
            text = str(candidate).strip()
            if text:
                tracked_paths.add(Path(text).expanduser().resolve())
    return tracked_paths


def _entry_matches_project_representative_key(
    entry,
    target_key: tuple[str, str, str],
) -> bool:
    return (
        str(entry.structure).strip(),
        str(entry.motif).strip() or "no_motif",
        str(entry.param).strip(),
    ) == target_key


def _path_is_within_dir(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _solvent_reference_identifier_from_metadata(solvent_metadata) -> str:
    settings = solvent_metadata.settings
    if (
        str(settings.reference_source).strip().lower() == "custom"
        and settings.custom_reference_path
    ):
        return str(Path(settings.custom_reference_path).expanduser().resolve())
    return str(settings.preset_name).strip() or "dmf"


def _merge_project_representative_entries(
    existing_metadata,
    updated_entry,
):
    existing_entries = (
        []
        if existing_metadata is None
        else list(existing_metadata.representative_entries)
    )
    updated_key = (
        updated_entry.structure,
        updated_entry.motif,
        updated_entry.param,
    )
    merged_entries = [
        entry
        for entry in existing_entries
        if (
            str(entry.structure).strip(),
            str(entry.motif).strip() or "no_motif",
            str(entry.param).strip(),
        )
        != updated_key
    ]
    merged_entries.append(updated_entry)
    merged_entries.sort(
        key=lambda entry: (
            _natural_sort_key(entry.structure),
            _natural_sort_key(entry.motif),
            _natural_sort_key(entry.param),
        )
    )
    return merged_entries


def _reweight_project_representative_entries(entries) -> None:
    total_cluster_count = max(
        sum(max(int(entry.cluster_count), 0) for entry in entries),
        1,
    )
    for entry in entries:
        entry.selected_weight = (
            max(int(entry.cluster_count), 0) / total_cluster_count
        )


def _next_available_output_dir(parent_dir: Path, folder_name: str) -> Path:
    candidate = parent_dir / folder_name
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = parent_dir / f"{folder_name}{index:04d}"
        if not candidate.exists():
            return candidate
        index += 1


def _safe_folder_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return cleaned or "structure_folder"


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8f}"


__all__ = [
    "RepresentativeFinderCandidate",
    "RepresentativeFinderFolderInspection",
    "RepresentativeFinderInputInspection",
    "RepresentativeFinderOperationCancelled",
    "RepresentativeFinderPlotSeries",
    "RepresentativeFinderResult",
    "RepresentativeFinderSettings",
    "analyze_representative_structure_folder",
    "estimate_representativefinder_total_work",
    "inspect_representative_structure_input",
    "inspect_representative_structure_folder",
    "load_representativefinder_result",
    "persist_representativefinder_result_to_project",
    "representativefinder_result_from_dict",
    "representativefinder_settings_from_dict",
    "suggest_representativefinder_output_dir",
    "suggest_representativefinder_target_output_dir",
]
