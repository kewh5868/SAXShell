from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from saxshell.saxs.debye.profiles import (
    b_factor_from_sigma,
    discover_cluster_bins,
)
from saxshell.saxs.project_manager import build_project_paths

DebyeWallerProgressCallback = Callable[[int, int, str], None]
DebyeWallerLogCallback = Callable[[str], None]
DebyeWallerStoichiometryCallback = Callable[
    ["DebyeWallerStoichiometryResult"], None
]
_FRAME_ID_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
        if token
    ]


def _safe_float_mean(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_float_std(values: list[float] | tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _safe_output_basename(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return safe_value.strip("._") or "debye_waller_analysis"


def _scope_display_name(scope: str) -> str:
    if scope == "intra_molecular":
        return "Intra-molecular"
    if scope == "inter_molecular":
        return "Inter-molecular"
    return str(scope)


def _frame_set_range_label(
    frame_set: "DebyeWallerContiguousFrameSetSummary",
) -> str:
    if frame_set.frame_start is None or frame_set.frame_end is None:
        return f"{frame_set.frame_count} frame(s)"
    if frame_set.frame_start == frame_set.frame_end:
        return (
            f"frame {frame_set.frame_start} "
            f"({frame_set.frame_count} frame(s))"
        )
    return (
        f"frames {frame_set.frame_start}-{frame_set.frame_end} "
        f"({frame_set.frame_count} frame(s))"
    )


@dataclass(slots=True, frozen=True)
class DebyeWallerInputInspection:
    clusters_dir: Path
    stoichiometry_labels: tuple[str, ...]
    total_structure_files: int
    invalid_xyz_files: tuple[Path, ...]

    @property
    def stoichiometry_count(self) -> int:
        return len(self.stoichiometry_labels)

    @property
    def is_pdb_only(self) -> bool:
        return not self.invalid_xyz_files

    def to_dict(self) -> dict[str, object]:
        return {
            "clusters_dir": str(self.clusters_dir),
            "stoichiometry_labels": list(self.stoichiometry_labels),
            "total_structure_files": int(self.total_structure_files),
            "invalid_xyz_files": [
                str(path) for path in self.invalid_xyz_files
            ],
            "is_pdb_only": bool(self.is_pdb_only),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerContiguousFrameSetSummary:
    series_label: str
    frame_ids: tuple[int, ...]
    frame_labels: tuple[str, ...]
    file_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return len(self.file_paths)

    @property
    def frame_start(self) -> int | None:
        return None if not self.frame_ids else int(self.frame_ids[0])

    @property
    def frame_end(self) -> int | None:
        return None if not self.frame_ids else int(self.frame_ids[-1])

    def to_dict(self) -> dict[str, object]:
        return {
            "series_label": self.series_label,
            "frame_ids": [int(value) for value in self.frame_ids],
            "frame_labels": [str(value) for value in self.frame_labels],
            "file_paths": [str(path) for path in self.file_paths],
            "frame_count": int(self.frame_count),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerSegmentStatistic:
    stoichiometry_label: str
    structure: str
    motif: str
    source_dir: Path
    segment_index: int
    series_label: str
    frame_start: int | None
    frame_end: int | None
    frame_count: int
    scope: str
    type_definition: str
    pair_label_a: str
    pair_label_b: str
    pair_label: str
    pair_count: int
    mean_distance_a: float
    sigma: float
    sigma_squared: float
    b_factor: float

    def to_dict(self) -> dict[str, object]:
        return {
            "stoichiometry_label": self.stoichiometry_label,
            "structure": self.structure,
            "motif": self.motif,
            "source_dir": str(self.source_dir),
            "segment_index": int(self.segment_index),
            "series_label": self.series_label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frame_count": int(self.frame_count),
            "scope": self.scope,
            "type_definition": self.type_definition,
            "pair_label_a": self.pair_label_a,
            "pair_label_b": self.pair_label_b,
            "pair_label": self.pair_label,
            "pair_count": int(self.pair_count),
            "mean_distance_a": float(self.mean_distance_a),
            "sigma": float(self.sigma),
            "sigma_squared": float(self.sigma_squared),
            "b_factor": float(self.b_factor),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerPairSummary:
    stoichiometry_label: str
    structure: str
    motif: str
    source_dir: Path
    scope: str
    type_definition: str
    pair_label_a: str
    pair_label_b: str
    pair_label: str
    segment_count: int
    mean_pair_count: float
    mean_distance_a: float
    sigma_mean: float
    sigma_std: float
    sigma_squared_mean: float
    sigma_squared_std: float
    b_factor_mean: float
    b_factor_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "stoichiometry_label": self.stoichiometry_label,
            "structure": self.structure,
            "motif": self.motif,
            "source_dir": str(self.source_dir),
            "scope": self.scope,
            "type_definition": self.type_definition,
            "pair_label_a": self.pair_label_a,
            "pair_label_b": self.pair_label_b,
            "pair_label": self.pair_label,
            "segment_count": int(self.segment_count),
            "mean_pair_count": float(self.mean_pair_count),
            "mean_distance_a": float(self.mean_distance_a),
            "sigma_mean": float(self.sigma_mean),
            "sigma_std": float(self.sigma_std),
            "sigma_squared_mean": float(self.sigma_squared_mean),
            "sigma_squared_std": float(self.sigma_squared_std),
            "b_factor_mean": float(self.b_factor_mean),
            "b_factor_std": float(self.b_factor_std),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerScopeSummary:
    stoichiometry_label: str
    structure: str
    motif: str
    source_dir: Path
    scope: str
    segment_count: int
    mean_pair_count: float
    sigma_mean: float
    sigma_std: float
    sigma_squared_mean: float
    sigma_squared_std: float
    b_factor_mean: float
    b_factor_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "stoichiometry_label": self.stoichiometry_label,
            "structure": self.structure,
            "motif": self.motif,
            "source_dir": str(self.source_dir),
            "scope": self.scope,
            "segment_count": int(self.segment_count),
            "mean_pair_count": float(self.mean_pair_count),
            "sigma_mean": float(self.sigma_mean),
            "sigma_std": float(self.sigma_std),
            "sigma_squared_mean": float(self.sigma_squared_mean),
            "sigma_squared_std": float(self.sigma_squared_std),
            "b_factor_mean": float(self.b_factor_mean),
            "b_factor_std": float(self.b_factor_std),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerAggregatedPairSummary:
    scope: str
    type_definition: str
    pair_label_a: str
    pair_label_b: str
    pair_label: str
    stoichiometry_count: int
    segment_count: int
    mean_pair_count: float
    mean_distance_a: float
    sigma_mean: float
    sigma_std: float
    sigma_squared_mean: float
    sigma_squared_std: float
    b_factor_mean: float
    b_factor_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "type_definition": self.type_definition,
            "pair_label_a": self.pair_label_a,
            "pair_label_b": self.pair_label_b,
            "pair_label": self.pair_label,
            "stoichiometry_count": int(self.stoichiometry_count),
            "segment_count": int(self.segment_count),
            "mean_pair_count": float(self.mean_pair_count),
            "mean_distance_a": float(self.mean_distance_a),
            "sigma_mean": float(self.sigma_mean),
            "sigma_std": float(self.sigma_std),
            "sigma_squared_mean": float(self.sigma_squared_mean),
            "sigma_squared_std": float(self.sigma_squared_std),
            "b_factor_mean": float(self.b_factor_mean),
            "b_factor_std": float(self.b_factor_std),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerStoichiometryInfoSummary:
    stoichiometry_label: str
    structure: str
    motif: str
    source_dir: Path
    input_file_count: int
    processed_frame_count: int
    total_frame_count: int
    processed_frame_set_count: int
    total_frame_set_count: int
    average_frames_per_set: float
    min_frames_per_set: int
    max_frames_per_set: int
    average_atoms_per_frame: float
    average_molecules_per_frame: float
    average_solvent_like_molecules_per_frame: float
    average_shared_atom_sites_per_set: float
    unique_residue_name_count: int
    unique_residue_names: tuple[str, ...]
    unique_element_count: int
    unique_elements: tuple[str, ...]
    most_common_molecule_signature: str
    solvent_like_molecule_signature: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stoichiometry_label": self.stoichiometry_label,
            "structure": self.structure,
            "motif": self.motif,
            "source_dir": str(self.source_dir),
            "input_file_count": int(self.input_file_count),
            "processed_frame_count": int(self.processed_frame_count),
            "total_frame_count": int(self.total_frame_count),
            "processed_frame_set_count": int(self.processed_frame_set_count),
            "total_frame_set_count": int(self.total_frame_set_count),
            "average_frames_per_set": float(self.average_frames_per_set),
            "min_frames_per_set": int(self.min_frames_per_set),
            "max_frames_per_set": int(self.max_frames_per_set),
            "average_atoms_per_frame": float(self.average_atoms_per_frame),
            "average_molecules_per_frame": float(
                self.average_molecules_per_frame
            ),
            "average_solvent_like_molecules_per_frame": float(
                self.average_solvent_like_molecules_per_frame
            ),
            "average_shared_atom_sites_per_set": float(
                self.average_shared_atom_sites_per_set
            ),
            "unique_residue_name_count": int(self.unique_residue_name_count),
            "unique_residue_names": [
                str(value) for value in self.unique_residue_names
            ],
            "unique_element_count": int(self.unique_element_count),
            "unique_elements": [str(value) for value in self.unique_elements],
            "most_common_molecule_signature": self.most_common_molecule_signature,
            "solvent_like_molecule_signature": self.solvent_like_molecule_signature,
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerStoichiometryResult:
    label: str
    structure: str
    motif: str
    source_dir: Path
    input_file_count: int
    contiguous_frame_sets: tuple[DebyeWallerContiguousFrameSetSummary, ...]
    pair_summaries: tuple[DebyeWallerPairSummary, ...]
    scope_summaries: tuple[DebyeWallerScopeSummary, ...]
    segment_statistics: tuple[DebyeWallerSegmentStatistic, ...]
    info_summary: DebyeWallerStoichiometryInfoSummary | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "structure": self.structure,
            "motif": self.motif,
            "source_dir": str(self.source_dir),
            "input_file_count": int(self.input_file_count),
            "contiguous_frame_sets": [
                entry.to_dict() for entry in self.contiguous_frame_sets
            ],
            "pair_summaries": [
                entry.to_dict() for entry in self.pair_summaries
            ],
            "scope_summaries": [
                entry.to_dict() for entry in self.scope_summaries
            ],
            "segment_statistics": [
                entry.to_dict() for entry in self.segment_statistics
            ],
            "info_summary": (
                None
                if self.info_summary is None
                else self.info_summary.to_dict()
            ),
            "notes": [str(note) for note in self.notes],
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerOutputArtifacts:
    output_dir: Path
    summary_json_path: Path
    aggregated_pair_summary_csv_path: Path
    pair_summary_csv_path: Path
    scope_summary_csv_path: Path
    segment_csv_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "summary_json_path": str(self.summary_json_path),
            "aggregated_pair_summary_csv_path": str(
                self.aggregated_pair_summary_csv_path
            ),
            "pair_summary_csv_path": str(self.pair_summary_csv_path),
            "scope_summary_csv_path": str(self.scope_summary_csv_path),
            "segment_csv_path": str(self.segment_csv_path),
        }


@dataclass(slots=True, frozen=True)
class DebyeWallerAnalysisResult:
    created_at: str
    clusters_dir: Path
    project_dir: Path | None
    output_dir: Path
    inspection: DebyeWallerInputInspection
    stoichiometry_results: tuple[DebyeWallerStoichiometryResult, ...]
    aggregated_pair_summaries: tuple[
        DebyeWallerAggregatedPairSummary, ...
    ] = ()
    notes: tuple[str, ...] = ()
    artifacts: DebyeWallerOutputArtifacts | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "clusters_dir": str(self.clusters_dir),
            "project_dir": (
                None if self.project_dir is None else str(self.project_dir)
            ),
            "output_dir": str(self.output_dir),
            "inspection": self.inspection.to_dict(),
            "stoichiometry_results": [
                entry.to_dict() for entry in self.stoichiometry_results
            ],
            "aggregated_pair_summaries": [
                entry.to_dict() for entry in self.aggregated_pair_summaries
            ],
            "notes": [str(note) for note in self.notes],
            "artifacts": (
                None if self.artifacts is None else self.artifacts.to_dict()
            ),
        }


_PROJECT_DEBYE_WALLER_DIRNAME = "debye_waller"
_PROJECT_DEBYE_WALLER_SAVED_DIRNAME = "saved_analysis"
_PROJECT_DEBYE_WALLER_SAVED_BASENAME = "debye_waller_analysis"


@dataclass(slots=True, frozen=True)
class _ContiguousFrameRecord:
    file_path: Path
    frame_id: int
    frame_label: str
    series_key: str
    series_label: str


@dataclass(slots=True, frozen=True)
class _PDBAtomRecord:
    site_key: tuple[str, str, str, str, str, str]
    molecule_key: tuple[str, str, str]
    element: str
    residue_name: str
    atom_name: str
    coordinates: np.ndarray


@dataclass(slots=True, frozen=True)
class _DebyeWallerFrameMetadata:
    atom_count: int
    molecule_count: int
    dominant_molecule_count: int
    solvent_like_molecule_count: int
    residue_names: tuple[str, ...]
    elements: tuple[str, ...]
    molecule_signature_counts: tuple[tuple[str, int], ...]


def inspect_debye_waller_input(
    clusters_dir: str | Path,
) -> DebyeWallerInputInspection:
    resolved_clusters_dir = Path(clusters_dir).expanduser().resolve()
    cluster_bins = discover_cluster_bins(resolved_clusters_dir)
    labels = tuple(
        (
            cluster_bin.structure
            if cluster_bin.motif == "no_motif"
            else f"{cluster_bin.structure}/{cluster_bin.motif}"
        )
        for cluster_bin in cluster_bins
    )
    invalid_xyz_files = tuple(
        sorted(
            (
                file_path
                for cluster_bin in cluster_bins
                for file_path in cluster_bin.files
                if file_path.suffix.lower() == ".xyz"
            ),
            key=lambda path: _natural_sort_key(str(path)),
        )
    )
    total_files = sum(len(cluster_bin.files) for cluster_bin in cluster_bins)
    return DebyeWallerInputInspection(
        clusters_dir=resolved_clusters_dir,
        stoichiometry_labels=labels,
        total_structure_files=int(total_files),
        invalid_xyz_files=invalid_xyz_files,
    )


def suggest_output_dir(
    selection_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> Path:
    if project_dir is not None:
        paths = build_project_paths(project_dir)
        return paths.exported_data_dir / "debye_waller"
    resolved = Path(selection_path).expanduser().resolve()
    base_dir = resolved if resolved.is_dir() else resolved.parent
    return base_dir / "debye_waller"


def _frame_series_label(prefix: str, suffix: str) -> str:
    parts = [
        part for part in (prefix.strip("_- "), suffix.strip("_- ")) if part
    ]
    if not parts:
        return "default"
    return " ".join(parts)


def _parse_contiguous_frame_record(
    file_path: str | Path,
) -> _ContiguousFrameRecord | None:
    resolved = Path(file_path).expanduser().resolve()
    stem = str(resolved.stem)
    match = _FRAME_ID_PATTERN.search(stem)
    if match is None:
        return None
    frame_label = str(match.group(1))
    prefix = stem[: match.start()]
    suffix = stem[match.end() :]
    return _ContiguousFrameRecord(
        file_path=resolved,
        frame_id=int(frame_label),
        frame_label=frame_label,
        series_key=f"{prefix}::{suffix}",
        series_label=_frame_series_label(prefix, suffix),
    )


def _detect_contiguous_frame_sets(
    structure_files: tuple[Path, ...],
) -> tuple[tuple[DebyeWallerContiguousFrameSetSummary, ...], str | None]:
    if not structure_files:
        return tuple(), None
    records: list[_ContiguousFrameRecord] = []
    for file_path in structure_files:
        parsed = _parse_contiguous_frame_record(file_path)
        if parsed is None:
            return (
                tuple(),
                "Contiguous-frame averaging requires every input file to "
                "contain a frame_<NNNN> identifier.",
            )
        records.append(parsed)

    series_map: dict[str, list[_ContiguousFrameRecord]] = {}
    for record in sorted(
        records,
        key=lambda item: (item.series_key, item.frame_id, item.file_path.name),
    ):
        series_map.setdefault(record.series_key, []).append(record)

    contiguous_sets: list[DebyeWallerContiguousFrameSetSummary] = []
    for series_records in series_map.values():
        current_records: list[_ContiguousFrameRecord] = []
        for record in series_records:
            if (
                current_records
                and record.frame_id != current_records[-1].frame_id + 1
            ):
                contiguous_sets.append(
                    DebyeWallerContiguousFrameSetSummary(
                        series_label=str(current_records[0].series_label),
                        frame_ids=tuple(
                            int(entry.frame_id) for entry in current_records
                        ),
                        frame_labels=tuple(
                            str(entry.frame_label) for entry in current_records
                        ),
                        file_paths=tuple(
                            entry.file_path for entry in current_records
                        ),
                    )
                )
                current_records = []
            current_records.append(record)
        if current_records:
            contiguous_sets.append(
                DebyeWallerContiguousFrameSetSummary(
                    series_label=str(current_records[0].series_label),
                    frame_ids=tuple(
                        int(entry.frame_id) for entry in current_records
                    ),
                    frame_labels=tuple(
                        str(entry.frame_label) for entry in current_records
                    ),
                    file_paths=tuple(
                        entry.file_path for entry in current_records
                    ),
                )
            )
    return tuple(contiguous_sets), None


def _pdb_record_name(line: str) -> str:
    return line[:6].strip().upper()


def _pdb_altloc_rank(value: str) -> int:
    altloc = value.strip().upper()
    if not altloc:
        return 0
    if altloc == "A":
        return 1
    if altloc == "1":
        return 2
    return 3


def _normalized_element_symbol(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:].lower()


def _pdb_atom_element(line: str) -> str:
    if len(line) >= 78:
        element = _normalized_element_symbol(line[76:78].strip())
        if element:
            return element
    atom_name_field = line[12:16] if len(line) >= 16 else ""
    letters = "".join(
        character for character in atom_name_field if character.isalpha()
    )
    if not letters:
        return "X"
    if atom_name_field[:1].isalpha() and len(letters) >= 2:
        return _normalized_element_symbol(letters[:2])
    return _normalized_element_symbol(letters[:1])


def _pdb_residue_name(line: str) -> str:
    return (line[17:20] if len(line) >= 20 else "").strip()


def _pdb_sequence_id(line: str) -> str:
    return (line[22:27] if len(line) >= 27 else "").strip()


def _pdb_segment_id(line: str) -> str:
    return (line[72:76] if len(line) >= 76 else "").strip()


def _pdb_atom_site_key(line: str) -> tuple[str, str, str, str, str, str]:
    return (
        _pdb_segment_id(line),
        (line[21:22] if len(line) >= 22 else "").strip(),
        _pdb_sequence_id(line),
        (line[26:27] if len(line) >= 27 else "").strip(),
        _pdb_residue_name(line),
        (line[12:16] if len(line) >= 16 else "").strip(),
    )


def _pdb_molecule_key(line: str) -> tuple[str, str, str]:
    # Mirror fullrmc's automatic molecule grouping:
    # contiguous atoms with the same residue, sequence, and segment belong
    # to the same molecule.
    return (
        _pdb_residue_name(line),
        _pdb_sequence_id(line),
        _pdb_segment_id(line),
    )


def _load_pdb_atom_records(path: str | Path) -> tuple[_PDBAtomRecord, ...]:
    pdb_path = Path(path).expanduser().resolve()
    chosen_sites: dict[
        tuple[str, str, str, str, str, str],
        tuple[int, str],
    ] = {}
    site_order: list[tuple[str, str, str, str, str, str]] = []
    lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        record_name = _pdb_record_name(line)
        if record_name == "ENDMDL" and chosen_sites:
            break
        if record_name not in {"ATOM", "HETATM"}:
            continue
        site_key = _pdb_atom_site_key(line)
        altloc_rank = _pdb_altloc_rank(line[16:17] if len(line) >= 17 else "")
        current = chosen_sites.get(site_key)
        if current is None:
            chosen_sites[site_key] = (altloc_rank, line)
            site_order.append(site_key)
        elif altloc_rank < current[0]:
            chosen_sites[site_key] = (altloc_rank, line)

    records: list[_PDBAtomRecord] = []
    for site_key in site_order:
        _altloc_rank, line = chosen_sites[site_key]
        try:
            x_coord = float(line[30:38].strip())
            y_coord = float(line[38:46].strip())
            z_coord = float(line[46:54].strip())
        except ValueError as exc:
            raise ValueError(
                f"Could not parse coordinates for atom site {site_key!r} "
                f"in {pdb_path.name}."
            ) from exc
        records.append(
            _PDBAtomRecord(
                site_key=site_key,
                molecule_key=_pdb_molecule_key(line),
                element=_pdb_atom_element(line),
                residue_name=_pdb_residue_name(line),
                atom_name=site_key[-1],
                coordinates=np.asarray(
                    [x_coord, y_coord, z_coord], dtype=float
                ),
            )
        )
    if not records:
        raise ValueError(f"No atoms were parsed from {pdb_path}.")
    return tuple(records)


def _hill_formula_from_counts(counts: Counter[str]) -> str:
    if not counts:
        return "unknown"
    ordered_symbols: list[str] = []
    if "C" in counts:
        ordered_symbols.append("C")
        if "H" in counts:
            ordered_symbols.append("H")
    ordered_symbols.extend(
        symbol
        for symbol in sorted(counts)
        if symbol not in {"C", "H"} and symbol
    )
    if "C" not in counts and "H" in counts:
        ordered_symbols.insert(0, "H")
    parts: list[str] = []
    for symbol in ordered_symbols:
        count = int(counts[symbol])
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts) or "unknown"


def _molecule_signature(records: list[_PDBAtomRecord]) -> str:
    if not records:
        return "unknown"
    residue_name = str(records[0].residue_name).strip() or "UNK"
    element_counts: Counter[str] = Counter()
    for record in records:
        symbol = _normalized_element_symbol(record.element) or "X"
        element_counts[symbol] += 1
    return f"{residue_name}:{_hill_formula_from_counts(element_counts)}"


def _summarize_frame_metadata(
    records: tuple[_PDBAtomRecord, ...],
) -> _DebyeWallerFrameMetadata:
    molecule_map: dict[tuple[str, str, str], list[_PDBAtomRecord]] = (
        defaultdict(list)
    )
    residue_names: set[str] = set()
    elements: set[str] = set()
    for record in records:
        molecule_map[record.molecule_key].append(record)
        residue_name = str(record.residue_name).strip()
        if residue_name:
            residue_names.add(residue_name)
        element = _normalized_element_symbol(record.element)
        if element:
            elements.add(element)

    signature_counts: Counter[str] = Counter()
    for molecule_records in molecule_map.values():
        signature_counts[_molecule_signature(molecule_records)] += 1

    dominant_signature = ""
    dominant_count = 0
    if signature_counts:
        dominant_signature, dominant_count = sorted(
            signature_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    return _DebyeWallerFrameMetadata(
        atom_count=len(records),
        molecule_count=len(molecule_map),
        dominant_molecule_count=int(dominant_count),
        solvent_like_molecule_count=(
            int(dominant_count) if int(dominant_count) > 1 else 0
        ),
        residue_names=tuple(sorted(residue_names)),
        elements=tuple(sorted(elements)),
        molecule_signature_counts=tuple(
            (str(signature), int(count))
            for signature, count in sorted(signature_counts.items())
        ),
    )


def _build_stoichiometry_info_summary(
    *,
    label: str,
    structure: str,
    motif: str,
    source_dir: Path,
    input_file_count: int,
    frame_sets: tuple[DebyeWallerContiguousFrameSetSummary, ...],
    processed_frame_metadata: list[_DebyeWallerFrameMetadata],
    processed_aligned_atom_counts: list[int],
) -> DebyeWallerStoichiometryInfoSummary:
    total_frame_count = sum(
        int(frame_set.frame_count) for frame_set in frame_sets
    )
    total_frame_set_count = len(frame_sets)
    frame_set_sizes = [int(frame_set.frame_count) for frame_set in frame_sets]
    processed_frame_count = len(processed_frame_metadata)

    unique_residue_names: set[str] = set()
    unique_elements: set[str] = set()
    overall_signature_counts: Counter[str] = Counter()
    repeated_signature_counts: Counter[str] = Counter()
    for metadata in processed_frame_metadata:
        unique_residue_names.update(metadata.residue_names)
        unique_elements.update(metadata.elements)
        for signature, count in metadata.molecule_signature_counts:
            overall_signature_counts[str(signature)] += int(count)
            if int(count) > 1:
                repeated_signature_counts[str(signature)] += int(count)

    most_common_molecule_signature = ""
    if overall_signature_counts:
        most_common_molecule_signature = sorted(
            overall_signature_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    solvent_like_molecule_signature = ""
    average_solvent_like_molecules_per_frame = 0.0
    if repeated_signature_counts:
        solvent_like_molecule_signature, solvent_like_total = sorted(
            repeated_signature_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        if processed_frame_count > 0:
            average_solvent_like_molecules_per_frame = float(
                solvent_like_total / processed_frame_count
            )

    return DebyeWallerStoichiometryInfoSummary(
        stoichiometry_label=label,
        structure=structure,
        motif=motif,
        source_dir=source_dir,
        input_file_count=int(input_file_count),
        processed_frame_count=int(processed_frame_count),
        total_frame_count=int(total_frame_count),
        processed_frame_set_count=int(len(processed_aligned_atom_counts)),
        total_frame_set_count=int(total_frame_set_count),
        average_frames_per_set=(
            float(total_frame_count / total_frame_set_count)
            if total_frame_set_count > 0
            else 0.0
        ),
        min_frames_per_set=min(frame_set_sizes, default=0),
        max_frames_per_set=max(frame_set_sizes, default=0),
        average_atoms_per_frame=_safe_float_mean(
            [
                float(metadata.atom_count)
                for metadata in processed_frame_metadata
            ]
        ),
        average_molecules_per_frame=_safe_float_mean(
            [
                float(metadata.molecule_count)
                for metadata in processed_frame_metadata
            ]
        ),
        average_solvent_like_molecules_per_frame=(
            average_solvent_like_molecules_per_frame
        ),
        average_shared_atom_sites_per_set=_safe_float_mean(
            [float(value) for value in processed_aligned_atom_counts]
        ),
        unique_residue_name_count=len(unique_residue_names),
        unique_residue_names=tuple(sorted(unique_residue_names)),
        unique_element_count=len(unique_elements),
        unique_elements=tuple(sorted(unique_elements)),
        most_common_molecule_signature=most_common_molecule_signature,
        solvent_like_molecule_signature=solvent_like_molecule_signature,
    )


def _intra_atom_label(record: _PDBAtomRecord) -> str:
    residue_name = str(record.residue_name).strip()
    atom_name = str(record.atom_name).strip()
    if residue_name:
        return f"{residue_name}:{atom_name}"
    return atom_name


def _aggregate_pair_sigma(pair_sigmas: list[float]) -> float:
    if not pair_sigmas:
        return 0.0
    sigma_values = np.asarray(pair_sigmas, dtype=float)
    return float(np.sqrt(np.mean(np.square(sigma_values), dtype=float)))


def _compute_segment_statistics(
    *,
    label: str,
    structure: str,
    motif: str,
    source_dir: Path,
    segment_index: int,
    frame_set: DebyeWallerContiguousFrameSetSummary,
) -> tuple[
    list[DebyeWallerSegmentStatistic],
    list[str],
    tuple[_DebyeWallerFrameMetadata, ...],
    int,
]:
    frame_records = [
        _load_pdb_atom_records(file_path) for file_path in frame_set.file_paths
    ]
    frame_metadata = tuple(
        _summarize_frame_metadata(records) for records in frame_records
    )
    if frame_set.frame_count < 2:
        return (
            [],
            [
                f"Skipped contiguous frame set {segment_index} for {label} "
                "because at least two frames are required to estimate "
                "pair-disorder statistics."
            ],
            frame_metadata,
            0,
        )

    first_frame = frame_records[0]
    common_site_keys = set(
        first_frame_record.site_key for first_frame_record in first_frame
    )
    for records in frame_records[1:]:
        common_site_keys &= {record.site_key for record in records}
    ordered_site_keys = [
        record.site_key
        for record in first_frame
        if record.site_key in common_site_keys
    ]
    if len(ordered_site_keys) < 2:
        return (
            [],
            [
                f"Skipped contiguous frame set {segment_index} for {label} "
                "because fewer than two aligned atoms were shared across the "
                "segment."
            ],
            frame_metadata,
            len(ordered_site_keys),
        )

    notes: list[str] = []
    if len(ordered_site_keys) != len(first_frame):
        notes.append(
            f"Frame set {segment_index} for {label} aligned "
            f"{len(ordered_site_keys)}/{len(first_frame)} atom sites shared "
            "across every frame."
        )

    frame_maps = [
        {record.site_key: record for record in records}
        for records in frame_records
    ]
    reference_records = [
        frame_maps[0][site_key] for site_key in ordered_site_keys
    ]
    coordinate_cube = np.asarray(
        [
            [frame_map[site_key].coordinates for site_key in ordered_site_keys]
            for frame_map in frame_maps
        ],
        dtype=float,
    )
    intra_labels = [_intra_atom_label(record) for record in reference_records]
    elements = [
        _normalized_element_symbol(record.element)
        for record in reference_records
    ]
    molecule_keys = [record.molecule_key for record in reference_records]

    grouped_pair_sigmas: dict[
        tuple[str, str, str, str, str],
        list[float],
    ] = defaultdict(list)
    grouped_pair_distances: dict[
        tuple[str, str, str, str, str],
        list[float],
    ] = defaultdict(list)

    atom_count = len(reference_records)
    for left_index in range(atom_count - 1):
        left_coords = coordinate_cube[:, left_index, :]
        left_molecule = molecule_keys[left_index]
        for right_index in range(left_index + 1, atom_count):
            deltas = left_coords - coordinate_cube[:, right_index, :]
            distances = np.linalg.norm(deltas, axis=1)
            pair_mean_distance = float(np.mean(distances, dtype=float))
            pair_sigma = float(np.std(distances, ddof=0))
            if left_molecule == molecule_keys[right_index]:
                scope = "intra_molecular"
                type_definition = "name"
                pair_label_a, pair_label_b = sorted(
                    (intra_labels[left_index], intra_labels[right_index])
                )
            else:
                scope = "inter_molecular"
                type_definition = "element"
                pair_label_a, pair_label_b = sorted(
                    (elements[left_index], elements[right_index])
                )
            grouped_key = (
                scope,
                type_definition,
                pair_label_a,
                pair_label_b,
                f"{pair_label_a}-{pair_label_b}",
            )
            grouped_pair_sigmas[grouped_key].append(pair_sigma)
            grouped_pair_distances[grouped_key].append(pair_mean_distance)

    rows: list[DebyeWallerSegmentStatistic] = []
    for grouped_key in sorted(
        grouped_pair_sigmas,
        key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
    ):
        pair_sigmas = grouped_pair_sigmas[grouped_key]
        pair_mean_distance = _safe_float_mean(
            grouped_pair_distances[grouped_key]
        )
        sigma = _aggregate_pair_sigma(pair_sigmas)
        rows.append(
            DebyeWallerSegmentStatistic(
                stoichiometry_label=label,
                structure=structure,
                motif=motif,
                source_dir=source_dir,
                segment_index=int(segment_index),
                series_label=frame_set.series_label,
                frame_start=frame_set.frame_start,
                frame_end=frame_set.frame_end,
                frame_count=int(frame_set.frame_count),
                scope=grouped_key[0],
                type_definition=grouped_key[1],
                pair_label_a=grouped_key[2],
                pair_label_b=grouped_key[3],
                pair_label=grouped_key[4],
                pair_count=len(pair_sigmas),
                mean_distance_a=pair_mean_distance,
                sigma=sigma,
                sigma_squared=float(sigma * sigma),
                b_factor=b_factor_from_sigma(sigma),
            )
        )
    return rows, notes, frame_metadata, len(ordered_site_keys)


def _build_pair_summaries(
    *,
    label: str,
    structure: str,
    motif: str,
    source_dir: Path,
    segment_rows: list[DebyeWallerSegmentStatistic],
) -> list[DebyeWallerPairSummary]:
    grouped_rows: dict[
        tuple[str, str, str, str, str],
        list[DebyeWallerSegmentStatistic],
    ] = defaultdict(list)
    for row in segment_rows:
        grouped_rows[
            (
                row.scope,
                row.type_definition,
                row.pair_label_a,
                row.pair_label_b,
                row.pair_label,
            )
        ].append(row)

    summaries: list[DebyeWallerPairSummary] = []
    for grouped_key in sorted(grouped_rows):
        rows = grouped_rows[grouped_key]
        sigma_values = [float(row.sigma) for row in rows]
        sigma_squared_values = [float(row.sigma_squared) for row in rows]
        b_values = [float(row.b_factor) for row in rows]
        pair_counts = [float(row.pair_count) for row in rows]
        mean_distances = [float(row.mean_distance_a) for row in rows]
        summaries.append(
            DebyeWallerPairSummary(
                stoichiometry_label=label,
                structure=structure,
                motif=motif,
                source_dir=source_dir,
                scope=grouped_key[0],
                type_definition=grouped_key[1],
                pair_label_a=grouped_key[2],
                pair_label_b=grouped_key[3],
                pair_label=grouped_key[4],
                segment_count=len(rows),
                mean_pair_count=_safe_float_mean(pair_counts),
                mean_distance_a=_safe_float_mean(mean_distances),
                sigma_mean=_safe_float_mean(sigma_values),
                sigma_std=_safe_float_std(sigma_values),
                sigma_squared_mean=_safe_float_mean(sigma_squared_values),
                sigma_squared_std=_safe_float_std(sigma_squared_values),
                b_factor_mean=_safe_float_mean(b_values),
                b_factor_std=_safe_float_std(b_values),
            )
        )
    return summaries


def _build_scope_summaries(
    *,
    label: str,
    structure: str,
    motif: str,
    source_dir: Path,
    segment_rows: list[DebyeWallerSegmentStatistic],
) -> list[DebyeWallerScopeSummary]:
    grouped_rows: dict[str, list[DebyeWallerSegmentStatistic]] = defaultdict(
        list
    )
    for row in segment_rows:
        grouped_rows[row.scope].append(row)

    scope_summaries: list[DebyeWallerScopeSummary] = []
    for scope in sorted(grouped_rows):
        rows = grouped_rows[scope]
        sigma_values = [float(row.sigma) for row in rows]
        sigma_squared_values = [float(row.sigma_squared) for row in rows]
        b_values = [float(row.b_factor) for row in rows]
        pair_counts = [float(row.pair_count) for row in rows]
        scope_summaries.append(
            DebyeWallerScopeSummary(
                stoichiometry_label=label,
                structure=structure,
                motif=motif,
                source_dir=source_dir,
                scope=scope,
                segment_count=len(rows),
                mean_pair_count=_safe_float_mean(pair_counts),
                sigma_mean=_safe_float_mean(sigma_values),
                sigma_std=_safe_float_std(sigma_values),
                sigma_squared_mean=_safe_float_mean(sigma_squared_values),
                sigma_squared_std=_safe_float_std(sigma_squared_values),
                b_factor_mean=_safe_float_mean(b_values),
                b_factor_std=_safe_float_std(b_values),
            )
        )
    return scope_summaries


def build_debye_waller_aggregated_pair_summaries(
    stoichiometry_results: (
        tuple[DebyeWallerStoichiometryResult, ...]
        | list[DebyeWallerStoichiometryResult]
    ),
) -> tuple[DebyeWallerAggregatedPairSummary, ...]:
    grouped_rows: dict[
        tuple[str, str, str, str, str],
        list[DebyeWallerSegmentStatistic],
    ] = defaultdict(list)
    contributing_stoichiometries: dict[
        tuple[str, str, str, str, str],
        set[str],
    ] = defaultdict(set)
    for stoichiometry in stoichiometry_results:
        seen_keys: set[tuple[str, str, str, str, str]] = set()
        for row in stoichiometry.segment_statistics:
            grouped_key = (
                row.scope,
                row.type_definition,
                row.pair_label_a,
                row.pair_label_b,
                row.pair_label,
            )
            grouped_rows[grouped_key].append(row)
            if grouped_key not in seen_keys:
                contributing_stoichiometries[grouped_key].add(
                    stoichiometry.label
                )
                seen_keys.add(grouped_key)

    summaries: list[DebyeWallerAggregatedPairSummary] = []
    for grouped_key in sorted(grouped_rows):
        rows = grouped_rows[grouped_key]
        sigma_values = [float(row.sigma) for row in rows]
        sigma_squared_values = [float(row.sigma_squared) for row in rows]
        b_values = [float(row.b_factor) for row in rows]
        pair_counts = [float(row.pair_count) for row in rows]
        mean_distances = [float(row.mean_distance_a) for row in rows]
        summaries.append(
            DebyeWallerAggregatedPairSummary(
                scope=grouped_key[0],
                type_definition=grouped_key[1],
                pair_label_a=grouped_key[2],
                pair_label_b=grouped_key[3],
                pair_label=grouped_key[4],
                stoichiometry_count=len(
                    contributing_stoichiometries[grouped_key]
                ),
                segment_count=len(rows),
                mean_pair_count=_safe_float_mean(pair_counts),
                mean_distance_a=_safe_float_mean(mean_distances),
                sigma_mean=_safe_float_mean(sigma_values),
                sigma_std=_safe_float_std(sigma_values),
                sigma_squared_mean=_safe_float_mean(sigma_squared_values),
                sigma_squared_std=_safe_float_std(sigma_squared_values),
                b_factor_mean=_safe_float_mean(b_values),
                b_factor_std=_safe_float_std(b_values),
            )
        )
    return tuple(summaries)


def write_debye_waller_outputs(
    result: DebyeWallerAnalysisResult,
    output_dir: str | Path,
    *,
    basename: str = "debye_waller_analysis",
) -> DebyeWallerOutputArtifacts:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    safe_basename = _safe_output_basename(basename)

    summary_json_path = resolved_output_dir / f"{safe_basename}.json"
    aggregated_pair_summary_csv_path = (
        resolved_output_dir / f"{safe_basename}_aggregated_pairs.csv"
    )
    pair_summary_csv_path = resolved_output_dir / f"{safe_basename}_pairs.csv"
    scope_summary_csv_path = (
        resolved_output_dir / f"{safe_basename}_scopes.csv"
    )
    segment_csv_path = resolved_output_dir / f"{safe_basename}_segments.csv"

    aggregated_pair_rows = [
        summary.to_dict() for summary in result.aggregated_pair_summaries
    ]
    pair_rows = [
        summary.to_dict()
        for stoichiometry in result.stoichiometry_results
        for summary in stoichiometry.pair_summaries
    ]
    scope_rows = [
        summary.to_dict()
        for stoichiometry in result.stoichiometry_results
        for summary in stoichiometry.scope_summaries
    ]
    segment_rows = [
        summary.to_dict()
        for stoichiometry in result.stoichiometry_results
        for summary in stoichiometry.segment_statistics
    ]

    with pair_summary_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stoichiometry_label",
                "structure",
                "motif",
                "source_dir",
                "scope",
                "type_definition",
                "pair_label_a",
                "pair_label_b",
                "pair_label",
                "segment_count",
                "mean_pair_count",
                "mean_distance_a",
                "sigma_mean",
                "sigma_std",
                "sigma_squared_mean",
                "sigma_squared_std",
                "b_factor_mean",
                "b_factor_std",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    with scope_summary_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stoichiometry_label",
                "structure",
                "motif",
                "source_dir",
                "scope",
                "segment_count",
                "mean_pair_count",
                "sigma_mean",
                "sigma_std",
                "sigma_squared_mean",
                "sigma_squared_std",
                "b_factor_mean",
                "b_factor_std",
            ],
        )
        writer.writeheader()
        writer.writerows(scope_rows)

    with segment_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stoichiometry_label",
                "structure",
                "motif",
                "source_dir",
                "segment_index",
                "series_label",
                "frame_start",
                "frame_end",
                "frame_count",
                "scope",
                "type_definition",
                "pair_label_a",
                "pair_label_b",
                "pair_label",
                "pair_count",
                "mean_distance_a",
                "sigma",
                "sigma_squared",
                "b_factor",
            ],
        )
        writer.writeheader()
        writer.writerows(segment_rows)

    with aggregated_pair_summary_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "type_definition",
                "pair_label_a",
                "pair_label_b",
                "pair_label",
                "stoichiometry_count",
                "segment_count",
                "mean_pair_count",
                "mean_distance_a",
                "sigma_mean",
                "sigma_std",
                "sigma_squared_mean",
                "sigma_squared_std",
                "b_factor_mean",
                "b_factor_std",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregated_pair_rows)

    output_payload = result.to_dict()
    output_payload["artifacts"] = {
        "output_dir": str(resolved_output_dir),
        "summary_json_path": str(summary_json_path),
        "aggregated_pair_summary_csv_path": str(
            aggregated_pair_summary_csv_path
        ),
        "pair_summary_csv_path": str(pair_summary_csv_path),
        "scope_summary_csv_path": str(scope_summary_csv_path),
        "segment_csv_path": str(segment_csv_path),
    }
    summary_json_path.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return DebyeWallerOutputArtifacts(
        output_dir=resolved_output_dir,
        summary_json_path=summary_json_path,
        aggregated_pair_summary_csv_path=aggregated_pair_summary_csv_path,
        pair_summary_csv_path=pair_summary_csv_path,
        scope_summary_csv_path=scope_summary_csv_path,
        segment_csv_path=segment_csv_path,
    )


def project_debye_waller_dir(project_dir: str | Path) -> Path:
    return (
        build_project_paths(project_dir).exported_data_dir
        / _PROJECT_DEBYE_WALLER_DIRNAME
    )


def project_saved_debye_waller_dir(project_dir: str | Path) -> Path:
    return (
        project_debye_waller_dir(project_dir)
        / _PROJECT_DEBYE_WALLER_SAVED_DIRNAME
    )


def _is_debye_waller_analysis_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    inspection = payload.get("inspection")
    stoichiometry_results = payload.get("stoichiometry_results")
    return isinstance(inspection, dict) and isinstance(
        stoichiometry_results,
        list,
    )


def _load_contiguous_frame_set_summary(
    payload: dict[str, object],
) -> DebyeWallerContiguousFrameSetSummary:
    return DebyeWallerContiguousFrameSetSummary(
        series_label=str(payload.get("series_label", "")),
        frame_ids=tuple(
            int(value) for value in (payload.get("frame_ids") or [])
        ),
        frame_labels=tuple(
            str(value) for value in (payload.get("frame_labels") or [])
        ),
        file_paths=tuple(
            Path(str(value)).expanduser().resolve()
            for value in (payload.get("file_paths") or [])
        ),
    )


def _load_segment_statistic(
    payload: dict[str, object],
) -> DebyeWallerSegmentStatistic:
    return DebyeWallerSegmentStatistic(
        stoichiometry_label=str(payload.get("stoichiometry_label", "")),
        structure=str(payload.get("structure", "")),
        motif=str(payload.get("motif", "")),
        source_dir=Path(str(payload.get("source_dir", ".")))
        .expanduser()
        .resolve(),
        segment_index=int(payload.get("segment_index", 0) or 0),
        series_label=str(payload.get("series_label", "")),
        frame_start=(
            None
            if payload.get("frame_start") in (None, "")
            else int(payload.get("frame_start"))
        ),
        frame_end=(
            None
            if payload.get("frame_end") in (None, "")
            else int(payload.get("frame_end"))
        ),
        frame_count=int(payload.get("frame_count", 0) or 0),
        scope=str(payload.get("scope", "")),
        type_definition=str(payload.get("type_definition", "")),
        pair_label_a=str(payload.get("pair_label_a", "")),
        pair_label_b=str(payload.get("pair_label_b", "")),
        pair_label=str(payload.get("pair_label", "")),
        pair_count=int(payload.get("pair_count", 0) or 0),
        mean_distance_a=float(payload.get("mean_distance_a", 0.0) or 0.0),
        sigma=float(payload.get("sigma", 0.0) or 0.0),
        sigma_squared=float(payload.get("sigma_squared", 0.0) or 0.0),
        b_factor=float(payload.get("b_factor", 0.0) or 0.0),
    )


def _load_pair_summary(
    payload: dict[str, object],
) -> DebyeWallerPairSummary:
    return DebyeWallerPairSummary(
        stoichiometry_label=str(payload.get("stoichiometry_label", "")),
        structure=str(payload.get("structure", "")),
        motif=str(payload.get("motif", "")),
        source_dir=Path(str(payload.get("source_dir", ".")))
        .expanduser()
        .resolve(),
        scope=str(payload.get("scope", "")),
        type_definition=str(payload.get("type_definition", "")),
        pair_label_a=str(payload.get("pair_label_a", "")),
        pair_label_b=str(payload.get("pair_label_b", "")),
        pair_label=str(payload.get("pair_label", "")),
        segment_count=int(payload.get("segment_count", 0) or 0),
        mean_pair_count=float(payload.get("mean_pair_count", 0.0) or 0.0),
        mean_distance_a=float(payload.get("mean_distance_a", 0.0) or 0.0),
        sigma_mean=float(payload.get("sigma_mean", 0.0) or 0.0),
        sigma_std=float(payload.get("sigma_std", 0.0) or 0.0),
        sigma_squared_mean=float(
            payload.get("sigma_squared_mean", 0.0) or 0.0
        ),
        sigma_squared_std=float(payload.get("sigma_squared_std", 0.0) or 0.0),
        b_factor_mean=float(payload.get("b_factor_mean", 0.0) or 0.0),
        b_factor_std=float(payload.get("b_factor_std", 0.0) or 0.0),
    )


def _load_scope_summary(
    payload: dict[str, object],
) -> DebyeWallerScopeSummary:
    return DebyeWallerScopeSummary(
        stoichiometry_label=str(payload.get("stoichiometry_label", "")),
        structure=str(payload.get("structure", "")),
        motif=str(payload.get("motif", "")),
        source_dir=Path(str(payload.get("source_dir", ".")))
        .expanduser()
        .resolve(),
        scope=str(payload.get("scope", "")),
        segment_count=int(payload.get("segment_count", 0) or 0),
        mean_pair_count=float(payload.get("mean_pair_count", 0.0) or 0.0),
        sigma_mean=float(payload.get("sigma_mean", 0.0) or 0.0),
        sigma_std=float(payload.get("sigma_std", 0.0) or 0.0),
        sigma_squared_mean=float(
            payload.get("sigma_squared_mean", 0.0) or 0.0
        ),
        sigma_squared_std=float(payload.get("sigma_squared_std", 0.0) or 0.0),
        b_factor_mean=float(payload.get("b_factor_mean", 0.0) or 0.0),
        b_factor_std=float(payload.get("b_factor_std", 0.0) or 0.0),
    )


def _load_aggregated_pair_summary(
    payload: dict[str, object],
) -> DebyeWallerAggregatedPairSummary:
    return DebyeWallerAggregatedPairSummary(
        scope=str(payload.get("scope", "")),
        type_definition=str(payload.get("type_definition", "")),
        pair_label_a=str(payload.get("pair_label_a", "")),
        pair_label_b=str(payload.get("pair_label_b", "")),
        pair_label=str(payload.get("pair_label", "")),
        stoichiometry_count=int(payload.get("stoichiometry_count", 0) or 0),
        segment_count=int(payload.get("segment_count", 0) or 0),
        mean_pair_count=float(payload.get("mean_pair_count", 0.0) or 0.0),
        mean_distance_a=float(payload.get("mean_distance_a", 0.0) or 0.0),
        sigma_mean=float(payload.get("sigma_mean", 0.0) or 0.0),
        sigma_std=float(payload.get("sigma_std", 0.0) or 0.0),
        sigma_squared_mean=float(
            payload.get("sigma_squared_mean", 0.0) or 0.0
        ),
        sigma_squared_std=float(payload.get("sigma_squared_std", 0.0) or 0.0),
        b_factor_mean=float(payload.get("b_factor_mean", 0.0) or 0.0),
        b_factor_std=float(payload.get("b_factor_std", 0.0) or 0.0),
    )


def _load_stoichiometry_info_summary(
    payload: dict[str, object] | None,
) -> DebyeWallerStoichiometryInfoSummary | None:
    if not isinstance(payload, dict):
        return None
    return DebyeWallerStoichiometryInfoSummary(
        stoichiometry_label=str(payload.get("stoichiometry_label", "")),
        structure=str(payload.get("structure", "")),
        motif=str(payload.get("motif", "")),
        source_dir=Path(str(payload.get("source_dir", ".")))
        .expanduser()
        .resolve(),
        input_file_count=int(payload.get("input_file_count", 0) or 0),
        processed_frame_count=int(
            payload.get("processed_frame_count", 0) or 0
        ),
        total_frame_count=int(payload.get("total_frame_count", 0) or 0),
        processed_frame_set_count=int(
            payload.get("processed_frame_set_count", 0) or 0
        ),
        total_frame_set_count=int(
            payload.get("total_frame_set_count", 0) or 0
        ),
        average_frames_per_set=float(
            payload.get("average_frames_per_set", 0.0) or 0.0
        ),
        min_frames_per_set=int(payload.get("min_frames_per_set", 0) or 0),
        max_frames_per_set=int(payload.get("max_frames_per_set", 0) or 0),
        average_atoms_per_frame=float(
            payload.get("average_atoms_per_frame", 0.0) or 0.0
        ),
        average_molecules_per_frame=float(
            payload.get("average_molecules_per_frame", 0.0) or 0.0
        ),
        average_solvent_like_molecules_per_frame=float(
            payload.get("average_solvent_like_molecules_per_frame", 0.0) or 0.0
        ),
        average_shared_atom_sites_per_set=float(
            payload.get("average_shared_atom_sites_per_set", 0.0) or 0.0
        ),
        unique_residue_name_count=int(
            payload.get("unique_residue_name_count", 0) or 0
        ),
        unique_residue_names=tuple(
            str(value) for value in (payload.get("unique_residue_names") or [])
        ),
        unique_element_count=int(payload.get("unique_element_count", 0) or 0),
        unique_elements=tuple(
            str(value) for value in (payload.get("unique_elements") or [])
        ),
        most_common_molecule_signature=str(
            payload.get("most_common_molecule_signature", "")
        ),
        solvent_like_molecule_signature=str(
            payload.get("solvent_like_molecule_signature", "")
        ),
    )


def _load_stoichiometry_result(
    payload: dict[str, object],
) -> DebyeWallerStoichiometryResult:
    return DebyeWallerStoichiometryResult(
        label=str(payload.get("label", "")),
        structure=str(payload.get("structure", "")),
        motif=str(payload.get("motif", "")),
        source_dir=Path(str(payload.get("source_dir", ".")))
        .expanduser()
        .resolve(),
        input_file_count=int(payload.get("input_file_count", 0) or 0),
        contiguous_frame_sets=tuple(
            _load_contiguous_frame_set_summary(entry)
            for entry in (payload.get("contiguous_frame_sets") or [])
            if isinstance(entry, dict)
        ),
        pair_summaries=tuple(
            _load_pair_summary(entry)
            for entry in (payload.get("pair_summaries") or [])
            if isinstance(entry, dict)
        ),
        scope_summaries=tuple(
            _load_scope_summary(entry)
            for entry in (payload.get("scope_summaries") or [])
            if isinstance(entry, dict)
        ),
        segment_statistics=tuple(
            _load_segment_statistic(entry)
            for entry in (payload.get("segment_statistics") or [])
            if isinstance(entry, dict)
        ),
        info_summary=_load_stoichiometry_info_summary(
            payload.get("info_summary")
        ),
        notes=tuple(str(note) for note in (payload.get("notes") or [])),
    )


def _load_input_inspection(
    payload: dict[str, object],
) -> DebyeWallerInputInspection:
    return DebyeWallerInputInspection(
        clusters_dir=Path(str(payload.get("clusters_dir", ".")))
        .expanduser()
        .resolve(),
        stoichiometry_labels=tuple(
            str(value) for value in (payload.get("stoichiometry_labels") or [])
        ),
        total_structure_files=int(
            payload.get("total_structure_files", 0) or 0
        ),
        invalid_xyz_files=tuple(
            Path(str(value)).expanduser().resolve()
            for value in (payload.get("invalid_xyz_files") or [])
        ),
    )


def _load_output_artifacts(
    payload: dict[str, object] | None,
) -> DebyeWallerOutputArtifacts | None:
    if not isinstance(payload, dict):
        return None
    summary_json_path = (
        Path(str(payload.get("summary_json_path", "."))).expanduser().resolve()
    )
    return DebyeWallerOutputArtifacts(
        output_dir=Path(str(payload.get("output_dir", ".")))
        .expanduser()
        .resolve(),
        summary_json_path=summary_json_path,
        aggregated_pair_summary_csv_path=Path(
            str(
                payload.get(
                    "aggregated_pair_summary_csv_path",
                    summary_json_path.with_name(
                        f"{summary_json_path.stem}_aggregated_pairs.csv"
                    ),
                )
            )
        )
        .expanduser()
        .resolve(),
        pair_summary_csv_path=Path(
            str(payload.get("pair_summary_csv_path", "."))
        )
        .expanduser()
        .resolve(),
        scope_summary_csv_path=Path(
            str(payload.get("scope_summary_csv_path", "."))
        )
        .expanduser()
        .resolve(),
        segment_csv_path=Path(str(payload.get("segment_csv_path", ".")))
        .expanduser()
        .resolve(),
    )


def load_debye_waller_analysis_result(
    summary_path: str | Path,
) -> DebyeWallerAnalysisResult:
    resolved_summary_path = Path(summary_path).expanduser().resolve()
    payload = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
    if not _is_debye_waller_analysis_payload(payload):
        raise ValueError(
            f"{resolved_summary_path} does not contain a Debye-Waller analysis payload."
        )
    inspection_payload = payload.get("inspection")
    stoichiometry_results = tuple(
        _load_stoichiometry_result(entry)
        for entry in (payload.get("stoichiometry_results") or [])
        if isinstance(entry, dict)
    )
    aggregated_pair_payloads = tuple(
        _load_aggregated_pair_summary(entry)
        for entry in (payload.get("aggregated_pair_summaries") or [])
        if isinstance(entry, dict)
    )
    return DebyeWallerAnalysisResult(
        created_at=str(payload.get("created_at", "")),
        clusters_dir=Path(str(payload.get("clusters_dir", ".")))
        .expanduser()
        .resolve(),
        project_dir=(
            None
            if payload.get("project_dir") in (None, "")
            else Path(str(payload.get("project_dir"))).expanduser().resolve()
        ),
        output_dir=Path(str(payload.get("output_dir", ".")))
        .expanduser()
        .resolve(),
        inspection=_load_input_inspection(
            inspection_payload if isinstance(inspection_payload, dict) else {}
        ),
        stoichiometry_results=stoichiometry_results,
        aggregated_pair_summaries=(
            aggregated_pair_payloads
            if aggregated_pair_payloads
            else build_debye_waller_aggregated_pair_summaries(
                stoichiometry_results
            )
        ),
        notes=tuple(str(note) for note in (payload.get("notes") or [])),
        artifacts=_load_output_artifacts(payload.get("artifacts")),
    )


def find_saved_project_debye_waller_analysis(
    project_dir: str | Path,
) -> Path | None:
    root_dir = project_debye_waller_dir(project_dir)
    if not root_dir.is_dir():
        return None
    preferred_path = (
        project_saved_debye_waller_dir(project_dir)
        / f"{_PROJECT_DEBYE_WALLER_SAVED_BASENAME}.json"
    )
    candidates: list[Path] = []
    if preferred_path.is_file():
        candidates.append(preferred_path)
    candidates.extend(
        sorted(
            (
                path
                for path in root_dir.rglob("*.json")
                if path.is_file() and path != preferred_path
            ),
            key=lambda path: (
                -path.stat().st_mtime,
                path.name.lower(),
            ),
        )
    )
    for candidate in candidates:
        try:
            load_debye_waller_analysis_result(candidate)
        except Exception:
            continue
        return candidate
    return None


def save_debye_waller_analysis_to_project(
    result: DebyeWallerAnalysisResult,
    project_dir: str | Path,
) -> DebyeWallerAnalysisResult:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    saved_dir = project_saved_debye_waller_dir(resolved_project_dir)
    base_result = DebyeWallerAnalysisResult(
        created_at=result.created_at,
        clusters_dir=result.clusters_dir,
        project_dir=resolved_project_dir,
        output_dir=saved_dir,
        inspection=result.inspection,
        stoichiometry_results=result.stoichiometry_results,
        aggregated_pair_summaries=result.aggregated_pair_summaries,
        notes=result.notes,
        artifacts=None,
    )
    artifacts = write_debye_waller_outputs(
        base_result,
        saved_dir,
        basename=_PROJECT_DEBYE_WALLER_SAVED_BASENAME,
    )
    return DebyeWallerAnalysisResult(
        created_at=base_result.created_at,
        clusters_dir=base_result.clusters_dir,
        project_dir=base_result.project_dir,
        output_dir=base_result.output_dir,
        inspection=base_result.inspection,
        stoichiometry_results=base_result.stoichiometry_results,
        aggregated_pair_summaries=base_result.aggregated_pair_summaries,
        notes=base_result.notes,
        artifacts=artifacts,
    )


class DebyeWallerWorkflow:
    """Estimate segment-aware Debye-Waller coefficients from cluster
    PDBs."""

    def __init__(
        self,
        clusters_dir: str | Path,
        *,
        project_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        output_basename: str = "debye_waller_analysis",
    ) -> None:
        self.clusters_dir = Path(clusters_dir).expanduser().resolve()
        self.project_dir = (
            None
            if project_dir is None
            else Path(project_dir).expanduser().resolve()
        )
        self.output_dir = (
            suggest_output_dir(self.clusters_dir, project_dir=self.project_dir)
            if output_dir is None
            else Path(output_dir).expanduser().resolve()
        )
        self.output_basename = _safe_output_basename(output_basename)

    def inspect(self) -> DebyeWallerInputInspection:
        return inspect_debye_waller_input(self.clusters_dir)

    def run(
        self,
        *,
        progress_callback: DebyeWallerProgressCallback | None = None,
        log_callback: DebyeWallerLogCallback | None = None,
        stoichiometry_callback: DebyeWallerStoichiometryCallback | None = None,
    ) -> DebyeWallerAnalysisResult:
        inspection = self.inspect()
        cluster_bins = discover_cluster_bins(self.clusters_dir)
        if not cluster_bins:
            raise ValueError(
                "No recognized stoichiometry folders were found in the "
                "selected clusters directory."
            )
        if inspection.invalid_xyz_files:
            file_list = "\n".join(
                f"- {path}" for path in inspection.invalid_xyz_files[:10]
            )
            raise ValueError(
                "Debye-Waller analysis requires PDB cluster files only. "
                "Remove or convert the following XYZ inputs before running:\n"
                f"{file_list}"
            )

        total_steps = len(cluster_bins) + 1
        if progress_callback is not None:
            progress_callback(
                0,
                total_steps,
                "Inspecting sorted PDB cluster bins and validating input.",
            )
        if log_callback is not None:
            log_callback("Starting Debye-Waller analysis.")
            log_callback(
                "Validated "
                f"{inspection.total_structure_files} structure file(s) across "
                f"{len(cluster_bins)} stoichiometry label(s) in "
                f"{self.clusters_dir}."
            )
            log_callback(
                "Output will be written to "
                f"{self.output_dir} with basename "
                f"{self.output_basename}."
            )

        stoichiometry_results: list[DebyeWallerStoichiometryResult] = []
        global_notes: list[str] = [
            "Intra/inter molecule membership follows fullrmc-style PDB "
            "grouping by residue name, sequence identifier, and segment "
            "identifier.",
            "Intra-molecular pair types are grouped by residue-qualified atom "
            "names, while inter-molecular pair types are grouped by element "
            "pairs.",
        ]

        for step_index, cluster_bin in enumerate(cluster_bins, start=1):
            label = (
                cluster_bin.structure
                if cluster_bin.motif == "no_motif"
                else f"{cluster_bin.structure}/{cluster_bin.motif}"
            )
            message = (
                f"Preparing Debye-Waller evaluation for {label} "
                f"({len(cluster_bin.files)} frame files)."
            )
            if progress_callback is not None:
                progress_callback(step_index, total_steps, message)
            if log_callback is not None:
                log_callback(message)

            frame_sets, contiguous_note = _detect_contiguous_frame_sets(
                tuple(cluster_bin.files)
            )
            notes: list[str] = []
            if contiguous_note is not None:
                notes.append(contiguous_note)
            if not frame_sets:
                notes.append(
                    f"No contiguous frame sets were detected for {label}."
                )
            if log_callback is not None:
                log_callback(
                    f"{label}: found {len(frame_sets)} contiguous frame set(s)."
                )
                for frame_set_index, frame_set in enumerate(
                    frame_sets,
                    start=1,
                ):
                    log_callback(
                        f"{label}: frame set {frame_set_index}/"
                        f"{len(frame_sets)} spans "
                        f"{_frame_set_range_label(frame_set)} in "
                        f"series '{frame_set.series_label}'."
                    )
                if contiguous_note is not None:
                    log_callback(f"{label}: {contiguous_note}")

            segment_rows: list[DebyeWallerSegmentStatistic] = []
            processed_frame_metadata: list[_DebyeWallerFrameMetadata] = []
            processed_aligned_atom_counts: list[int] = []
            partial_result: DebyeWallerStoichiometryResult | None = None
            for segment_index, frame_set in enumerate(frame_sets, start=1):
                if progress_callback is not None:
                    progress_callback(
                        step_index,
                        total_steps,
                        f"{label}: analyzing frame set {segment_index}/"
                        f"{len(frame_sets)} spanning "
                        f"{_frame_set_range_label(frame_set)}.",
                    )
                if log_callback is not None:
                    log_callback(
                        f"{label}: analyzing frame set {segment_index}/"
                        f"{len(frame_sets)} spanning "
                        f"{_frame_set_range_label(frame_set)}."
                    )
                (
                    computed_rows,
                    segment_notes,
                    frame_metadata,
                    aligned_atom_count,
                ) = _compute_segment_statistics(
                    label=label,
                    structure=cluster_bin.structure,
                    motif=cluster_bin.motif,
                    source_dir=cluster_bin.source_dir,
                    segment_index=segment_index,
                    frame_set=frame_set,
                )
                segment_rows.extend(computed_rows)
                notes.extend(segment_notes)
                processed_frame_metadata.extend(frame_metadata)
                processed_aligned_atom_counts.append(aligned_atom_count)
                partial_result = DebyeWallerStoichiometryResult(
                    label=label,
                    structure=cluster_bin.structure,
                    motif=cluster_bin.motif,
                    source_dir=cluster_bin.source_dir,
                    input_file_count=len(cluster_bin.files),
                    contiguous_frame_sets=frame_sets,
                    pair_summaries=tuple(
                        _build_pair_summaries(
                            label=label,
                            structure=cluster_bin.structure,
                            motif=cluster_bin.motif,
                            source_dir=cluster_bin.source_dir,
                            segment_rows=segment_rows,
                        )
                    ),
                    scope_summaries=tuple(
                        _build_scope_summaries(
                            label=label,
                            structure=cluster_bin.structure,
                            motif=cluster_bin.motif,
                            source_dir=cluster_bin.source_dir,
                            segment_rows=segment_rows,
                        )
                    ),
                    segment_statistics=tuple(
                        sorted(
                            segment_rows,
                            key=lambda row: (
                                row.segment_index,
                                row.scope,
                                row.type_definition,
                                row.pair_label,
                            ),
                        )
                    ),
                    info_summary=_build_stoichiometry_info_summary(
                        label=label,
                        structure=cluster_bin.structure,
                        motif=cluster_bin.motif,
                        source_dir=cluster_bin.source_dir,
                        input_file_count=len(cluster_bin.files),
                        frame_sets=frame_sets,
                        processed_frame_metadata=processed_frame_metadata,
                        processed_aligned_atom_counts=processed_aligned_atom_counts,
                    ),
                    notes=tuple(notes),
                )
                if log_callback is not None:
                    if computed_rows:
                        log_callback(
                            f"{label}: frame set {segment_index} produced "
                            f"{len(computed_rows)} segment-level "
                            "coefficient row(s)."
                        )
                        log_callback(
                            f"{label}: accumulated "
                            f"{len(partial_result.pair_summaries)} pair-type "
                            f"summary row(s), "
                            f"{len(partial_result.scope_summaries)} scope "
                            f"summary row(s), and "
                            f"{len(partial_result.segment_statistics)} "
                            "segment row(s) so far."
                        )
                    else:
                        log_callback(
                            f"{label}: frame set {segment_index} produced no "
                            "usable pair-disorder rows."
                        )
                    info_summary = partial_result.info_summary
                    if info_summary is not None:
                        residue_preview = ", ".join(
                            info_summary.unique_residue_names[:4]
                        )
                        if len(info_summary.unique_residue_names) > 4:
                            residue_preview += ", ..."
                        log_callback(
                            f"{label}: metadata so far -> "
                            f"{info_summary.processed_frame_count}/"
                            f"{info_summary.total_frame_count} frame(s) "
                            f"profiled, "
                            f"{info_summary.average_atoms_per_frame:.1f} "
                            "atom(s)/frame, "
                            f"{info_summary.average_molecules_per_frame:.1f} "
                            "molecule group(s)/frame, residues "
                            f"{residue_preview or 'n/a'}."
                        )
                if (
                    stoichiometry_callback is not None
                    and partial_result is not None
                ):
                    stoichiometry_callback(partial_result)

            stoichiometry_result = partial_result
            if stoichiometry_result is None:
                stoichiometry_result = DebyeWallerStoichiometryResult(
                    label=label,
                    structure=cluster_bin.structure,
                    motif=cluster_bin.motif,
                    source_dir=cluster_bin.source_dir,
                    input_file_count=len(cluster_bin.files),
                    contiguous_frame_sets=frame_sets,
                    pair_summaries=tuple(),
                    scope_summaries=tuple(),
                    segment_statistics=tuple(),
                    info_summary=_build_stoichiometry_info_summary(
                        label=label,
                        structure=cluster_bin.structure,
                        motif=cluster_bin.motif,
                        source_dir=cluster_bin.source_dir,
                        input_file_count=len(cluster_bin.files),
                        frame_sets=frame_sets,
                        processed_frame_metadata=processed_frame_metadata,
                        processed_aligned_atom_counts=processed_aligned_atom_counts,
                    ),
                    notes=tuple(notes),
                )
            stoichiometry_results.append(stoichiometry_result)
            if log_callback is not None:
                log_callback(
                    f"{label}: completed with "
                    f"{len(stoichiometry_result.pair_summaries)} pair-type "
                    f"summary row(s), "
                    f"{len(stoichiometry_result.scope_summaries)} scope "
                    f"summary row(s), and "
                    f"{len(stoichiometry_result.segment_statistics)} "
                    "segment row(s)."
                )
            if stoichiometry_callback is not None and partial_result is None:
                stoichiometry_callback(stoichiometry_result)

        created_at = datetime.now(timezone.utc).isoformat()
        result = DebyeWallerAnalysisResult(
            created_at=created_at,
            clusters_dir=self.clusters_dir,
            project_dir=self.project_dir,
            output_dir=self.output_dir,
            inspection=inspection,
            stoichiometry_results=tuple(stoichiometry_results),
            aggregated_pair_summaries=build_debye_waller_aggregated_pair_summaries(
                stoichiometry_results
            ),
            notes=tuple(global_notes),
            artifacts=None,
        )
        artifacts = write_debye_waller_outputs(
            result,
            self.output_dir,
            basename=self.output_basename,
        )
        finalized = DebyeWallerAnalysisResult(
            created_at=result.created_at,
            clusters_dir=result.clusters_dir,
            project_dir=result.project_dir,
            output_dir=result.output_dir,
            inspection=result.inspection,
            stoichiometry_results=result.stoichiometry_results,
            aggregated_pair_summaries=result.aggregated_pair_summaries,
            notes=result.notes,
            artifacts=artifacts,
        )
        if progress_callback is not None:
            progress_callback(
                total_steps,
                total_steps,
                "Debye-Waller analysis complete.",
            )
        if log_callback is not None:
            log_callback(
                "Wrote Debye-Waller outputs to " f"{artifacts.output_dir}."
            )
        return finalized


__all__ = [
    "DebyeWallerAggregatedPairSummary",
    "DebyeWallerAnalysisResult",
    "DebyeWallerContiguousFrameSetSummary",
    "DebyeWallerInputInspection",
    "DebyeWallerOutputArtifacts",
    "DebyeWallerPairSummary",
    "DebyeWallerScopeSummary",
    "DebyeWallerSegmentStatistic",
    "DebyeWallerStoichiometryInfoSummary",
    "DebyeWallerStoichiometryResult",
    "DebyeWallerWorkflow",
    "DebyeWallerStoichiometryCallback",
    "inspect_debye_waller_input",
    "build_debye_waller_aggregated_pair_summaries",
    "find_saved_project_debye_waller_analysis",
    "load_debye_waller_analysis_result",
    "project_debye_waller_dir",
    "project_saved_debye_waller_dir",
    "save_debye_waller_analysis_to_project",
    "suggest_output_dir",
]
