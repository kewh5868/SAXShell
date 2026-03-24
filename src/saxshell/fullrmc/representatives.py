from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from saxshell.bondanalysis.bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
)
from saxshell.bondanalysis.results import (
    LEGACY_RESULTS_INDEX_FILENAME,
    RESULTS_INDEX_FILENAME,
    load_result_index,
)
from saxshell.saxs.debye import load_structure_file
from saxshell.saxs.dream.results import SAXSDreamResultsLoader
from saxshell.saxs.project_manager import DreamBestFitSelection

if TYPE_CHECKING:
    from .project_loader import RMCDreamProjectSource

_STRUCTURE_SUFFIXES = {".pdb", ".xyz"}
_DEFAULT_QUANTILES = tuple(np.linspace(0.0, 1.0, 11).tolist())
RepresentativeSelectionProgressCallback = Callable[[int, int, str], None]
RepresentativeSelectionLogCallback = Callable[[str], None]


def _emit_selection_progress(
    callback: RepresentativeSelectionProgressCallback | None,
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


def _emit_selection_log(
    callback: RepresentativeSelectionLogCallback | None,
    message: str,
) -> None:
    if callback is None:
        return
    text = str(message).strip()
    if text:
        callback(text)


@dataclass(slots=True)
class DistributionSelectionEntry:
    param: str
    structure: str
    motif: str
    selected_weight: float
    vary: bool
    cluster_count: int
    source_dir: str | None
    source_file: str | None
    source_file_name: str | None
    source_kind: str | None
    is_active: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def effective_cluster_count(self) -> int:
        if int(self.cluster_count) > 0:
            return int(self.cluster_count)
        if self.source_file:
            return 1
        return int(self.cluster_count)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "DistributionSelectionEntry":
        return cls(
            param=str(payload.get("param", "")).strip(),
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            selected_weight=float(payload.get("selected_weight", 0.0)),
            vary=bool(payload.get("vary", False)),
            cluster_count=int(payload.get("cluster_count", 0)),
            source_dir=_optional_text(payload.get("source_dir")),
            source_file=_optional_text(payload.get("source_file")),
            source_file_name=_optional_text(payload.get("source_file_name")),
            source_kind=_optional_text(payload.get("source_kind")),
            is_active=bool(payload.get("is_active", False)),
        )


@dataclass(slots=True)
class DistributionSelectionMetadata:
    selection_mode: str
    selection: DreamBestFitSelection
    run_dir: str
    updated_at: str
    entries: list[DistributionSelectionEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "selection_mode": self.selection_mode,
            "selection": self.selection.to_dict(),
            "run_dir": self.run_dir,
            "updated_at": self.updated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "DistributionSelectionMetadata | None":
        if not payload:
            return None
        selection_payload = payload.get("selection")
        if not isinstance(selection_payload, dict):
            return None
        return cls(
            selection_mode=str(
                payload.get("selection_mode", "first_file")
            ).strip()
            or "first_file",
            selection=DreamBestFitSelection.from_dict(selection_payload),
            run_dir=str(payload.get("run_dir", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            entries=[
                DistributionSelectionEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
        )

    def active_entries(
        self,
        minimum_cluster_count: int = 1,
    ) -> list[DistributionSelectionEntry]:
        cutoff = max(int(minimum_cluster_count), 1)
        return [
            entry
            for entry in self.entries
            if entry.is_active and entry.effective_cluster_count() >= cutoff
        ]

    def skipped_by_cluster_count(
        self,
        minimum_cluster_count: int = 1,
    ) -> list[DistributionSelectionEntry]:
        cutoff = max(int(minimum_cluster_count), 1)
        return [
            entry
            for entry in self.entries
            if entry.is_active and entry.effective_cluster_count() < cutoff
        ]


@dataclass(slots=True)
class RepresentativeSelectionSettings:
    selection_mode: str = "first_file"
    selection_algorithm: str = "target_distribution_quantile_distance"
    minimum_cluster_count_for_analysis: int = 1
    bond_weight: float = 1.0
    angle_weight: float = 1.0
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    bond_pairs: tuple[BondPairDefinition, ...] = ()
    angle_triplets: tuple[AngleTripletDefinition, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_mode": self.selection_mode,
            "selection_algorithm": self.selection_algorithm,
            "minimum_cluster_count_for_analysis": (
                self.minimum_cluster_count_for_analysis
            ),
            "bond_weight": self.bond_weight,
            "angle_weight": self.angle_weight,
            "quantiles": list(self.quantiles),
            "bond_pairs": [
                definition.to_dict() for definition in self.bond_pairs
            ],
            "angle_triplets": [
                definition.to_dict() for definition in self.angle_triplets
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "RepresentativeSelectionSettings":
        source = dict(payload or {})
        quantiles = tuple(
            float(value)
            for value in source.get("quantiles", _DEFAULT_QUANTILES)
        )
        return cls(
            selection_mode=str(
                source.get("selection_mode", "first_file")
            ).strip()
            or "first_file",
            selection_algorithm=str(
                source.get(
                    "selection_algorithm",
                    "target_distribution_quantile_distance",
                )
            ).strip()
            or "target_distribution_quantile_distance",
            minimum_cluster_count_for_analysis=max(
                1,
                int(source.get("minimum_cluster_count_for_analysis", 1)),
            ),
            bond_weight=float(source.get("bond_weight", 1.0)),
            angle_weight=float(source.get("angle_weight", 1.0)),
            quantiles=quantiles or _DEFAULT_QUANTILES,
            bond_pairs=tuple(
                BondPairDefinition(**dict(entry))
                for entry in source.get("bond_pairs", [])
                if isinstance(entry, dict)
            ),
            angle_triplets=tuple(
                AngleTripletDefinition(**dict(entry))
                for entry in source.get("angle_triplets", [])
                if isinstance(entry, dict)
            ),
        )


@dataclass(slots=True)
class RepresentativeSelectionEntry:
    structure: str
    motif: str
    param: str
    selected_weight: float
    cluster_count: int
    source_dir: str
    source_file: str
    source_file_name: str
    atom_count: int
    element_counts: dict[str, int]
    analysis_source: str = "first_valid_file"
    score_total: float | None = None
    score_bond: float | None = None
    score_angle: float | None = None
    cached_results_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "RepresentativeSelectionEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            selected_weight=float(payload.get("selected_weight", 0.0)),
            cluster_count=int(payload.get("cluster_count", 0)),
            source_dir=str(payload.get("source_dir", "")).strip(),
            source_file=str(payload.get("source_file", "")).strip(),
            source_file_name=str(payload.get("source_file_name", "")).strip(),
            atom_count=int(payload.get("atom_count", 0)),
            element_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("element_counts", {})
                ).items()
            },
            analysis_source=str(
                payload.get("analysis_source", "first_valid_file")
            ).strip()
            or "first_valid_file",
            score_total=_optional_float(payload.get("score_total")),
            score_bond=_optional_float(payload.get("score_bond")),
            score_angle=_optional_float(payload.get("score_angle")),
            cached_results_path=_optional_text(
                payload.get("cached_results_path")
            ),
        )


@dataclass(slots=True)
class RepresentativeSelectionIssue:
    structure: str
    motif: str
    param: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "RepresentativeSelectionIssue":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            message=str(payload.get("message", "")).strip(),
        )


@dataclass(slots=True)
class RepresentativeSelectionMetadata:
    selection_mode: str
    selection: DreamBestFitSelection
    distribution_selection: DistributionSelectionMetadata
    settings: RepresentativeSelectionSettings
    updated_at: str
    representative_entries: list[RepresentativeSelectionEntry]
    missing_bins: list[RepresentativeSelectionIssue]
    invalid_bins: list[RepresentativeSelectionIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "selection_mode": self.selection_mode,
            "selection": self.selection.to_dict(),
            "distribution_selection": self.distribution_selection.to_dict(),
            "settings": self.settings.to_dict(),
            "updated_at": self.updated_at,
            "representative_entries": [
                entry.to_dict() for entry in self.representative_entries
            ],
            "missing_bins": [issue.to_dict() for issue in self.missing_bins],
            "invalid_bins": [issue.to_dict() for issue in self.invalid_bins],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "RepresentativeSelectionMetadata | None":
        if not payload:
            return None
        selection_payload = payload.get("selection")
        distribution_payload = payload.get("distribution_selection")
        settings_payload = payload.get("settings")
        if not isinstance(selection_payload, dict) or not isinstance(
            distribution_payload,
            dict,
        ):
            return None
        distribution = DistributionSelectionMetadata.from_dict(
            distribution_payload
        )
        if distribution is None:
            return None
        return cls(
            selection_mode=str(
                payload.get("selection_mode", "first_file")
            ).strip()
            or "first_file",
            selection=DreamBestFitSelection.from_dict(selection_payload),
            distribution_selection=distribution,
            settings=RepresentativeSelectionSettings.from_dict(
                settings_payload
                if isinstance(settings_payload, dict)
                else None
            ),
            updated_at=str(payload.get("updated_at", "")).strip(),
            representative_entries=[
                RepresentativeSelectionEntry.from_dict(dict(entry))
                for entry in payload.get("representative_entries", [])
                if isinstance(entry, dict)
            ],
            missing_bins=[
                RepresentativeSelectionIssue.from_dict(dict(entry))
                for entry in payload.get("missing_bins", [])
                if isinstance(entry, dict)
            ],
            invalid_bins=[
                RepresentativeSelectionIssue.from_dict(dict(entry))
                for entry in payload.get("invalid_bins", [])
                if isinstance(entry, dict)
            ],
        )

    def summary_text(self) -> str:
        lines = [
            f"Selection mode: {self.selection_mode}",
            f"Saved at: {self.updated_at}",
            f"Run: {self.selection.run_name}",
            f"Best-fit method: {self.selection.bestfit_method}",
            f"Algorithm: {self.settings.selection_algorithm}",
            (
                "Active bins in selected DREAM distribution: "
                f"{len(self.distribution_selection.active_entries(self.settings.minimum_cluster_count_for_analysis))}"
            ),
            (
                "Cluster count cutoff: "
                f"{self.settings.minimum_cluster_count_for_analysis}"
            ),
            (
                "Skipped by count cutoff: "
                f"{len(self.distribution_selection.skipped_by_cluster_count(self.settings.minimum_cluster_count_for_analysis))}"
            ),
            (
                "Representative files resolved: "
                f"{len(self.representative_entries)}"
            ),
            f"Missing bins: {len(self.missing_bins)}",
            f"Invalid bins: {len(self.invalid_bins)}",
        ]
        if self.representative_entries:
            first = self.representative_entries[0]
            lines.extend(
                [
                    "",
                    "Example representative:",
                    f"  {first.structure}/{first.motif}",
                    f"  source file: {first.source_file_name}",
                    f"  analysis source: {first.analysis_source}",
                    f"  selected weight: {first.selected_weight:.6g}",
                    f"  atom count: {first.atom_count}",
                ]
            )
            if first.score_total is not None:
                lines.append(f"  score: {first.score_total:.6g}")
        return "\n".join(lines)


@dataclass(slots=True)
class RepresentativePreviewSeries:
    category: str
    display_label: str
    xlabel: str
    distribution_values: np.ndarray
    representative_values: tuple[float, ...]


@dataclass(slots=True)
class RepresentativePreviewCluster:
    structure: str
    motif: str
    param: str
    selected_weight: float
    source_file_name: str
    analysis_source: str
    score_total: float | None
    bond_series: tuple[RepresentativePreviewSeries, ...]
    angle_series: tuple[RepresentativePreviewSeries, ...]

    @property
    def tab_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    @property
    def title(self) -> str:
        return f"{self.tab_label} • {self.source_file_name}"

    def all_series(self) -> tuple[RepresentativePreviewSeries, ...]:
        return self.bond_series + self.angle_series


def build_distribution_selection(
    project_source: RMCDreamProjectSource,
    selection: DreamBestFitSelection,
    *,
    selection_mode: str = "first_file",
) -> DistributionSelectionMetadata:
    run = project_source.find_run_for_selection(selection)
    if run is None:
        raise ValueError(
            "The selected DREAM run is not available in the current project."
        )
    loader = SAXSDreamResultsLoader(run.run_dir)
    summary = loader.get_summary(
        bestfit_method=selection.bestfit_method,
        posterior_filter_mode=selection.posterior_filter_mode,
        posterior_top_percent=selection.posterior_top_percent,
        posterior_top_n=selection.posterior_top_n,
        credible_interval_low=selection.credible_interval_low,
        credible_interval_high=selection.credible_interval_high,
    )
    parameter_lookup = _build_parameter_lookup(loader)
    cluster_lookup = _cluster_lookup(project_source)
    entries: list[DistributionSelectionEntry] = []

    for index, param_name in enumerate(summary.full_parameter_names):
        if not param_name.startswith("w"):
            continue
        parameter_entry = parameter_lookup.get(param_name, {})
        structure = str(parameter_entry.get("structure", "")).strip()
        if not structure:
            continue
        motif = (
            str(parameter_entry.get("motif", "no_motif")).strip() or "no_motif"
        )
        cluster_row = cluster_lookup.get((structure, motif), {})
        selected_weight = float(summary.bestfit_params[index])
        source_file = _resolve_cluster_row_source_file(cluster_row)
        source_file_name = _resolve_cluster_row_source_file_name(
            cluster_row,
            source_file=source_file,
        )
        source_dir = _optional_text(cluster_row.get("source_dir"))
        if source_dir is None and source_file is not None:
            source_dir = str(Path(source_file).expanduser().resolve().parent)
        cluster_count = int(cluster_row.get("count", 0))
        if cluster_count <= 0 and source_file is not None:
            cluster_count = 1
        entries.append(
            DistributionSelectionEntry(
                param=param_name,
                structure=structure,
                motif=motif,
                selected_weight=selected_weight,
                vary=bool(parameter_entry.get("vary", False)),
                cluster_count=cluster_count,
                source_dir=source_dir,
                source_file=source_file,
                source_file_name=source_file_name,
                source_kind=_resolve_cluster_row_source_kind(
                    cluster_row,
                    source_dir=source_dir,
                    source_file=source_file,
                ),
                is_active=selected_weight > 0.0,
            )
        )

    if not entries:
        raise ValueError(
            "No DREAM weight parameters with structure metadata were found "
            "for the selected run."
        )

    entries.sort(
        key=lambda entry: (
            -entry.selected_weight,
            _natural_sort_key(entry.structure),
            _natural_sort_key(entry.motif),
            entry.param,
        )
    )
    return DistributionSelectionMetadata(
        selection_mode=selection_mode,
        selection=selection,
        run_dir=str(run.run_dir),
        updated_at=datetime.now().isoformat(timespec="seconds"),
        entries=entries,
    )


def save_distribution_selection_metadata(
    output_path: str | Path,
    metadata: DistributionSelectionMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_distribution_selection_metadata(
    metadata_path: str | Path,
) -> DistributionSelectionMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return DistributionSelectionMetadata.from_dict(payload)


def parse_bond_pair_text(text: str) -> tuple[BondPairDefinition, ...]:
    definitions: list[BondPairDefinition] = []
    for chunk in _definition_chunks(text):
        match = re.fullmatch(
            r"([A-Za-z][A-Za-z0-9]*)\s*-\s*([A-Za-z][A-Za-z0-9]*)\s*:\s*([0-9.+-eE]+)",
            chunk,
        )
        if match is None:
            raise ValueError(
                "Bond pairs must be written like 'Pb-I:3.5; Pb-O:3.0'."
            )
        definitions.append(
            BondPairDefinition(
                atom1=match.group(1),
                atom2=match.group(2),
                cutoff_angstrom=float(match.group(3)),
            )
        )
    return tuple(definitions)


def parse_angle_triplet_text(text: str) -> tuple[AngleTripletDefinition, ...]:
    definitions: list[AngleTripletDefinition] = []
    for chunk in _definition_chunks(text):
        match = re.fullmatch(
            (
                r"([A-Za-z][A-Za-z0-9]*)\s*-\s*([A-Za-z][A-Za-z0-9]*)\s*-\s*"
                r"([A-Za-z][A-Za-z0-9]*)\s*:\s*([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)"
            ),
            chunk,
        )
        if match is None:
            raise ValueError(
                "Angle triplets must be written like "
                "'I-Pb-I:3.5,3.5; O-Pb-I:3.0,3.5'."
            )
        definitions.append(
            AngleTripletDefinition(
                vertex=match.group(2),
                arm1=match.group(1),
                arm2=match.group(3),
                cutoff1_angstrom=float(match.group(4)),
                cutoff2_angstrom=float(match.group(5)),
            )
        )
    return tuple(definitions)


def select_first_file_representatives(
    project_source: RMCDreamProjectSource,
    selection: DreamBestFitSelection,
    *,
    settings: RepresentativeSelectionSettings | None = None,
    progress_callback: RepresentativeSelectionProgressCallback | None = None,
    log_callback: RepresentativeSelectionLogCallback | None = None,
) -> RepresentativeSelectionMetadata:
    active_settings = settings or RepresentativeSelectionSettings(
        selection_mode="first_file"
    )
    _emit_selection_log(
        log_callback,
        "Loading DREAM distribution for representative selection.",
    )
    _emit_selection_progress(
        progress_callback,
        0,
        1,
        "Loading DREAM distribution...",
    )
    distribution = build_distribution_selection(
        project_source,
        selection,
        selection_mode="first_file",
    )
    save_distribution_selection_metadata(
        project_source.rmcsetup_paths.distribution_selection_path,
        distribution,
    )

    representative_entries: list[RepresentativeSelectionEntry] = []
    missing_bins: list[RepresentativeSelectionIssue] = []
    invalid_bins: list[RepresentativeSelectionIssue] = []
    active_entries = distribution.active_entries(
        active_settings.minimum_cluster_count_for_analysis
    )
    work_plan, total_work = _prepare_entry_work(active_entries)
    processed_work = 0
    for entry in active_entries:
        label = _distribution_entry_label(entry)
        prepared = work_plan[_distribution_entry_key(entry)]
        source_dir = prepared.source_dir
        source_file = prepared.source_file
        structure_files = prepared.structure_files
        work_units = prepared.work_units
        _emit_selection_log(
            log_callback,
            f"Selecting representative for {label}.",
        )
        _emit_selection_progress(
            progress_callback,
            processed_work,
            total_work,
            f"Selecting {label}...",
        )
        if source_dir is None and source_file is None:
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "No source directory or source structure file was "
                        "resolved for this component."
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no source file",
            )
            continue
        if source_file is not None and not source_file.is_file():
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "The source structure file does not exist: "
                        f"{source_file}"
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: missing source file",
            )
            continue
        if (
            source_dir is not None
            and source_file is None
            and not source_dir.is_dir()
        ):
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "The cluster source directory does not exist: "
                        f"{source_dir}"
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: missing source directory",
            )
            continue

        selected_file: Path | None = None
        selected_elements: list[str] | None = None
        last_error: Exception | None = None
        for local_index, candidate in enumerate(structure_files, start=1):
            _emit_selection_progress(
                progress_callback,
                processed_work + local_index - 1,
                total_work,
                f"{label}: reading {candidate.name}",
            )
            try:
                _positions, elements = load_structure_file(candidate)
            except Exception as exc:
                last_error = exc
                continue
            selected_file = candidate
            selected_elements = list(elements)
            break

        if selected_file is None or selected_elements is None:
            message = "No valid .xyz or .pdb structure file could be parsed."
            if last_error is not None:
                message = f"{message} Last error: {last_error}"
            invalid_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=message,
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no readable structures",
            )
            continue

        representative_entries.append(
            RepresentativeSelectionEntry(
                structure=entry.structure,
                motif=entry.motif,
                param=entry.param,
                selected_weight=entry.selected_weight,
                cluster_count=entry.cluster_count,
                source_dir=str(source_dir or selected_file.parent),
                source_file=str(selected_file),
                source_file_name=selected_file.name,
                atom_count=len(selected_elements),
                element_counts=dict(Counter(selected_elements)),
            )
        )
        processed_work += work_units
        _emit_selection_progress(
            progress_callback,
            processed_work,
            total_work,
            f"Finished {label}",
        )

    metadata = RepresentativeSelectionMetadata(
        selection_mode="first_file",
        selection=selection,
        distribution_selection=distribution,
        settings=active_settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_entries=representative_entries,
        missing_bins=missing_bins,
        invalid_bins=invalid_bins,
    )
    save_representative_selection_metadata(
        project_source.rmcsetup_paths.representative_selection_path,
        metadata,
    )
    _emit_selection_log(
        log_callback,
        "Representative selection finished: "
        f"{len(representative_entries)} representative file(s) resolved.",
    )
    _emit_selection_progress(
        progress_callback,
        total_work,
        total_work,
        "Representative selection complete.",
    )
    return metadata


@dataclass(slots=True)
class _CandidateMeasurement:
    path: Path
    atom_count: int
    element_counts: dict[str, int]
    bond_values: dict[BondPairDefinition, list[float]]
    angle_values: dict[AngleTripletDefinition, list[float]]


@dataclass(slots=True)
class _PreparedEntryWork:
    source_dir: Path | None
    source_file: Path | None
    structure_files: list[Path]
    work_units: int


@dataclass(slots=True)
class _CachedDistributionContext:
    bond_pairs: tuple[BondPairDefinition, ...]
    angle_triplets: tuple[AngleTripletDefinition, ...]
    bond_values: dict[BondPairDefinition, np.ndarray]
    angle_values: dict[AngleTripletDefinition, np.ndarray]
    results_index_path: Path


def select_distribution_representatives(
    project_source: RMCDreamProjectSource,
    selection: DreamBestFitSelection,
    *,
    settings: RepresentativeSelectionSettings,
    progress_callback: RepresentativeSelectionProgressCallback | None = None,
    log_callback: RepresentativeSelectionLogCallback | None = None,
) -> RepresentativeSelectionMetadata:
    if settings.selection_mode != "bond_angle_distribution":
        raise ValueError(
            "Distribution representative selection requires "
            "selection_mode='bond_angle_distribution'."
        )
    _emit_selection_log(
        log_callback,
        "Loading DREAM distribution for representative selection.",
    )
    _emit_selection_progress(
        progress_callback,
        0,
        1,
        "Loading DREAM distribution...",
    )
    distribution = build_distribution_selection(
        project_source,
        selection,
        selection_mode="bond_angle_distribution",
    )
    save_distribution_selection_metadata(
        project_source.rmcsetup_paths.distribution_selection_path,
        distribution,
    )

    cached_indices = _discover_project_bondanalysis_indices(
        project_source.paths.project_dir
    )
    representative_entries: list[RepresentativeSelectionEntry] = []
    missing_bins: list[RepresentativeSelectionIssue] = []
    invalid_bins: list[RepresentativeSelectionIssue] = []
    active_entries = distribution.active_entries(
        settings.minimum_cluster_count_for_analysis
    )
    work_plan, total_work = _prepare_entry_work(active_entries)
    processed_work = 0

    for entry in active_entries:
        label = _distribution_entry_label(entry)
        prepared = work_plan[_distribution_entry_key(entry)]
        source_dir = prepared.source_dir
        source_file = prepared.source_file
        structure_files = prepared.structure_files
        work_units = prepared.work_units
        _emit_selection_log(
            log_callback,
            f"Selecting representative for {label}.",
        )
        _emit_selection_progress(
            progress_callback,
            processed_work,
            total_work,
            f"Analyzing {label}...",
        )
        if source_dir is None and source_file is None:
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "No source directory or source structure file was "
                        "resolved for this component."
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no source file",
            )
            continue
        if source_file is not None and not source_file.is_file():
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "The source structure file does not exist: "
                        f"{source_file}"
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: missing source file",
            )
            continue
        if (
            source_dir is not None
            and source_file is None
            and not source_dir.is_dir()
        ):
            missing_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "The cluster source directory does not exist: "
                        f"{source_dir}"
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: missing source directory",
            )
            continue

        cached = _find_cached_distribution_context(
            entry,
            source_dir=source_dir,
            cached_indices=cached_indices,
        )
        active_settings = settings
        if cached is not None and (cached.bond_pairs or cached.angle_triplets):
            active_settings = RepresentativeSelectionSettings(
                selection_mode=settings.selection_mode,
                selection_algorithm=settings.selection_algorithm,
                bond_weight=settings.bond_weight,
                angle_weight=settings.angle_weight,
                quantiles=settings.quantiles,
                bond_pairs=cached.bond_pairs,
                angle_triplets=cached.angle_triplets,
            )
        if (
            not active_settings.bond_pairs
            and not active_settings.angle_triplets
        ):
            invalid_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "No bond pairs or angle triplets are available for "
                        "distribution-based representative selection."
                    ),
                )
            )
            processed_work += work_units
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no measurement definitions",
            )
            continue

        analyzer = BondAnalyzer(
            bond_pairs=active_settings.bond_pairs,
            angle_triplets=active_settings.angle_triplets,
        )
        if cached is None:
            _emit_selection_log(
                log_callback,
                f"{label}: recomputing target bond and angle distributions.",
            )
        else:
            _emit_selection_log(
                log_callback,
                f"{label}: reusing cached bondanalysis distributions.",
            )
        candidates, last_error = _measure_candidate_structures(
            source_dir,
            analyzer,
            structure_files=structure_files,
            progress_callback=lambda local_processed, local_total, message: _emit_selection_progress(
                progress_callback,
                min(
                    processed_work + local_processed,
                    total_work,
                ),
                total_work,
                f"{label}: {message}",
            ),
        )
        processed_work += work_units
        if not candidates:
            message = "No valid structure files could be measured for this cluster bin."
            if last_error is not None:
                message = f"{message} Last error: {last_error}"
            invalid_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=message,
                )
            )
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no measurable structures",
            )
            continue

        if cached is None:
            target_bonds, target_angles = _aggregate_candidate_values(
                candidates,
                active_settings.bond_pairs,
                active_settings.angle_triplets,
            )
            analysis_source = "recomputed"
            cached_results_path = None
        else:
            target_bonds = dict(cached.bond_values)
            target_angles = dict(cached.angle_values)
            analysis_source = "cached_bondanalysis"
            cached_results_path = str(cached.results_index_path)

        if not any(
            values.size > 0 for values in target_bonds.values()
        ) and not any(values.size > 0 for values in target_angles.values()):
            invalid_bins.append(
                RepresentativeSelectionIssue(
                    structure=entry.structure,
                    motif=entry.motif,
                    param=entry.param,
                    message=(
                        "No bond or angle values were available for the "
                        "selected measurement definitions."
                    ),
                )
            )
            _emit_selection_progress(
                progress_callback,
                processed_work,
                total_work,
                f"Skipped {label}: no bond or angle values",
            )
            continue

        best_candidate, score_bond, score_angle, score_total = (
            _select_best_distribution_candidate(
                candidates,
                target_bonds=target_bonds,
                target_angles=target_angles,
                settings=active_settings,
            )
        )
        representative_entries.append(
            RepresentativeSelectionEntry(
                structure=entry.structure,
                motif=entry.motif,
                param=entry.param,
                selected_weight=entry.selected_weight,
                cluster_count=entry.cluster_count,
                source_dir=str(source_dir or best_candidate.path.parent),
                source_file=str(best_candidate.path),
                source_file_name=best_candidate.path.name,
                atom_count=best_candidate.atom_count,
                element_counts=dict(best_candidate.element_counts),
                analysis_source=analysis_source,
                score_total=score_total,
                score_bond=score_bond,
                score_angle=score_angle,
                cached_results_path=cached_results_path,
            )
        )
        _emit_selection_progress(
            progress_callback,
            processed_work,
            total_work,
            f"Finished {label}",
        )

    metadata = RepresentativeSelectionMetadata(
        selection_mode="bond_angle_distribution",
        selection=selection,
        distribution_selection=distribution,
        settings=settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_entries=representative_entries,
        missing_bins=missing_bins,
        invalid_bins=invalid_bins,
    )
    save_representative_selection_metadata(
        project_source.rmcsetup_paths.representative_selection_path,
        metadata,
    )
    _emit_selection_log(
        log_callback,
        "Representative selection finished: "
        f"{len(representative_entries)} representative file(s) resolved.",
    )
    _emit_selection_progress(
        progress_callback,
        total_work,
        total_work,
        "Representative selection complete.",
    )
    return metadata


def save_representative_selection_metadata(
    output_path: str | Path,
    metadata: RepresentativeSelectionMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_representative_selection_metadata(
    metadata_path: str | Path,
) -> RepresentativeSelectionMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return RepresentativeSelectionMetadata.from_dict(payload)


def build_representative_preview_clusters(
    project_source: RMCDreamProjectSource,
    metadata: RepresentativeSelectionMetadata | None = None,
) -> list[RepresentativePreviewCluster]:
    active_metadata = metadata or project_source.representative_selection
    if active_metadata is None:
        raise ValueError(
            "No representative selection metadata is available for preview."
        )

    preview_clusters: list[RepresentativePreviewCluster] = []
    cached_index_cache: dict[Path, object] = {}
    for entry in active_metadata.representative_entries:
        source_dir = Path(entry.source_dir).expanduser().resolve()
        source_file = Path(entry.source_file).expanduser().resolve()
        distribution_entry = DistributionSelectionEntry(
            param=entry.param,
            structure=entry.structure,
            motif=entry.motif,
            selected_weight=entry.selected_weight,
            vary=True,
            cluster_count=entry.cluster_count,
            source_dir=str(source_dir),
            source_file=entry.source_file,
            source_file_name=entry.source_file_name,
            source_kind=None,
            is_active=True,
        )

        cached = _load_cached_distribution_context_for_preview(
            distribution_entry,
            source_dir=source_dir,
            cached_results_path=entry.cached_results_path,
            cache=cached_index_cache,
        )
        active_settings = active_metadata.settings
        if cached is not None and (cached.bond_pairs or cached.angle_triplets):
            active_settings = RepresentativeSelectionSettings(
                selection_mode=active_metadata.settings.selection_mode,
                selection_algorithm=(
                    active_metadata.settings.selection_algorithm
                ),
                bond_weight=active_metadata.settings.bond_weight,
                angle_weight=active_metadata.settings.angle_weight,
                quantiles=active_metadata.settings.quantiles,
                bond_pairs=cached.bond_pairs,
                angle_triplets=cached.angle_triplets,
            )

        analyzer = BondAnalyzer(
            bond_pairs=active_settings.bond_pairs,
            angle_triplets=active_settings.angle_triplets,
        )
        try:
            representative_bonds, representative_angles = (
                analyzer.measure_structure(source_file)
            )
        except Exception:
            representative_bonds = {
                definition: [] for definition in active_settings.bond_pairs
            }
            representative_angles = {
                definition: [] for definition in active_settings.angle_triplets
            }

        if cached is None:
            candidates, _last_error = _measure_candidate_structures(
                source_dir,
                analyzer,
            )
            target_bonds, target_angles = _aggregate_candidate_values(
                candidates,
                active_settings.bond_pairs,
                active_settings.angle_triplets,
            )
        else:
            target_bonds = dict(cached.bond_values)
            target_angles = dict(cached.angle_values)

        bond_series = tuple(
            RepresentativePreviewSeries(
                category="bond",
                display_label=definition.display_label,
                xlabel=f"{definition.display_label} distance (Angstrom)",
                distribution_values=np.asarray(
                    target_bonds.get(definition, np.array([], dtype=float)),
                    dtype=float,
                ),
                representative_values=_representative_line_values(
                    representative_bonds.get(definition, [])
                ),
            )
            for definition in active_settings.bond_pairs
        )
        angle_series = tuple(
            RepresentativePreviewSeries(
                category="angle",
                display_label=definition.display_label,
                xlabel=f"{definition.display_label} angle (deg)",
                distribution_values=np.asarray(
                    target_angles.get(definition, np.array([], dtype=float)),
                    dtype=float,
                ),
                representative_values=_representative_line_values(
                    representative_angles.get(definition, [])
                ),
            )
            for definition in active_settings.angle_triplets
        )
        preview_clusters.append(
            RepresentativePreviewCluster(
                structure=entry.structure,
                motif=entry.motif,
                param=entry.param,
                selected_weight=entry.selected_weight,
                source_file_name=entry.source_file_name,
                analysis_source=entry.analysis_source,
                score_total=entry.score_total,
                bond_series=bond_series,
                angle_series=angle_series,
            )
        )
    return preview_clusters


def _build_parameter_lookup(
    loader: SAXSDreamResultsLoader,
) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for entry in loader.parameter_map_entries:
        name = str(entry.get("param", "")).strip()
        if name and name not in lookup:
            lookup[name] = dict(entry)
    for entry in loader.active_entries:
        name = str(entry.get("param", "")).strip()
        if name and name not in lookup:
            lookup[name] = dict(entry)
    return lookup


def _cluster_lookup(
    project_source: RMCDreamProjectSource,
) -> dict[tuple[str, str], dict[str, object]]:
    validation = project_source.cluster_validation
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in validation.expected_rows:
        _merge_cluster_lookup_row(lookup, row)
    for row in validation.current_rows:
        _merge_cluster_lookup_row(lookup, row)
    return lookup


def _discover_project_bondanalysis_indices(project_dir: Path) -> list[object]:
    indices: list[object] = []
    for filename in (RESULTS_INDEX_FILENAME, LEGACY_RESULTS_INDEX_FILENAME):
        for index_path in sorted(project_dir.rglob(filename)):
            try:
                indices.append(load_result_index(index_path.parent))
            except Exception:
                continue
    unique: list[object] = []
    seen: set[Path] = set()
    for result_index in indices:
        path = Path(result_index.results_index_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(result_index)
    unique.sort(
        key=lambda result_index: (
            result_index.results_index_path.stat().st_mtime,
            str(result_index.results_index_path),
        ),
        reverse=True,
    )
    return unique


def _find_cached_distribution_context(
    distribution_entry: DistributionSelectionEntry,
    *,
    source_dir: Path,
    cached_indices: list[object],
) -> _CachedDistributionContext | None:
    leaf_name_candidates = {
        distribution_entry.structure,
        distribution_entry.motif,
        source_dir.name,
    }
    for result_index in cached_indices:
        cluster_dir = Path(result_index.clusters_dir).resolve()
        if cluster_dir != source_dir and cluster_dir not in source_dir.parents:
            continue
        bond_values: dict[BondPairDefinition, np.ndarray] = {}
        angle_values: dict[AngleTripletDefinition, np.ndarray] = {}
        matched = False
        for definition, group in zip(
            result_index.bond_pairs,
            result_index.bond_groups,
            strict=True,
        ):
            leaf = next(
                (
                    candidate
                    for candidate in group.cluster_leaves
                    if candidate.scope_name in leaf_name_candidates
                ),
                None,
            )
            if (
                leaf is None
                or leaf.npy_path is None
                or not leaf.npy_path.is_file()
            ):
                continue
            bond_values[definition] = _load_npy_values(leaf.npy_path)
            matched = True
        for definition, group in zip(
            result_index.angle_triplets,
            result_index.angle_groups,
            strict=True,
        ):
            leaf = next(
                (
                    candidate
                    for candidate in group.cluster_leaves
                    if candidate.scope_name in leaf_name_candidates
                ),
                None,
            )
            if (
                leaf is None
                or leaf.npy_path is None
                or not leaf.npy_path.is_file()
            ):
                continue
            angle_values[definition] = _load_npy_values(leaf.npy_path)
            matched = True
        if matched:
            return _CachedDistributionContext(
                bond_pairs=tuple(result_index.bond_pairs),
                angle_triplets=tuple(result_index.angle_triplets),
                bond_values=bond_values,
                angle_values=angle_values,
                results_index_path=result_index.results_index_path,
            )
    return None


def _load_cached_distribution_context_for_preview(
    distribution_entry: DistributionSelectionEntry,
    *,
    source_dir: Path,
    cached_results_path: str | None,
    cache: dict[Path, object],
) -> _CachedDistributionContext | None:
    if not cached_results_path:
        return None
    results_index_path = Path(cached_results_path).expanduser().resolve()
    if not results_index_path.is_file():
        return None
    result_index = cache.get(results_index_path)
    if result_index is None:
        try:
            result_index = load_result_index(results_index_path.parent)
        except Exception:
            return None
        cache[results_index_path] = result_index
    return _find_cached_distribution_context(
        distribution_entry,
        source_dir=source_dir,
        cached_indices=[result_index],
    )


def _load_npy_values(npy_path: Path) -> np.ndarray:
    payload = np.load(npy_path, allow_pickle=False)
    if getattr(payload, "dtype", None) is not None and payload.dtype.names:
        return np.asarray(payload["value"], dtype=float)
    return np.asarray(payload, dtype=float)


def _measure_candidate_structures(
    source_dir: Path,
    analyzer: BondAnalyzer,
    *,
    structure_files: list[Path] | None = None,
    progress_callback: RepresentativeSelectionProgressCallback | None = None,
) -> tuple[list[_CandidateMeasurement], Exception | None]:
    candidates: list[_CandidateMeasurement] = []
    last_error: Exception | None = None
    files = (
        list(structure_files)
        if structure_files is not None
        else _sorted_structure_files(source_dir)
    )
    total_files = max(len(files), 1)
    if not files:
        _emit_selection_progress(
            progress_callback,
            total_files,
            total_files,
            "No structure files found",
        )
        return candidates, last_error
    for index, candidate in enumerate(files, start=1):
        try:
            atoms = analyzer.read_structure(candidate)
            bond_values, angle_values = analyzer.measure_atoms(atoms)
        except Exception as exc:
            last_error = exc
            _emit_selection_progress(
                progress_callback,
                index,
                total_files,
                f"Skipped unreadable file {candidate.name}",
            )
            continue
        element_counts = Counter(atom.element for atom in atoms)
        candidates.append(
            _CandidateMeasurement(
                path=candidate,
                atom_count=len(atoms),
                element_counts=dict(element_counts),
                bond_values=bond_values,
                angle_values=angle_values,
            )
        )
        _emit_selection_progress(
            progress_callback,
            index,
            total_files,
            f"Measured {candidate.name}",
        )
    return candidates, last_error


def _prepare_entry_work(
    entries: list[DistributionSelectionEntry],
) -> tuple[
    dict[tuple[str, str, str], _PreparedEntryWork],
    int,
]:
    work_plan: dict[tuple[str, str, str], _PreparedEntryWork] = {}
    total_work = 0
    for entry in entries:
        source_file = (
            Path(entry.source_file).expanduser().resolve()
            if entry.source_file
            else None
        )
        source_dir = (
            Path(entry.source_dir).expanduser().resolve()
            if entry.source_dir
            else None
        )
        if source_dir is None and source_file is not None:
            source_dir = source_file.parent
        if (
            source_file is not None
            and source_file.is_file()
            and source_file.suffix.lower() in _STRUCTURE_SUFFIXES
        ):
            structure_files = [source_file]
        elif source_dir is not None and source_dir.is_dir():
            structure_files = _sorted_structure_files(source_dir)
        else:
            structure_files = []
        work_units = max(len(structure_files), 1)
        work_plan[_distribution_entry_key(entry)] = _PreparedEntryWork(
            source_dir,
            source_file,
            structure_files,
            work_units,
        )
        total_work += work_units
    return work_plan, max(total_work, 1)


def _distribution_entry_key(
    entry: DistributionSelectionEntry,
) -> tuple[str, str, str]:
    return (entry.structure, entry.motif, entry.param)


def _distribution_entry_label(entry: DistributionSelectionEntry) -> str:
    if entry.motif == "no_motif":
        return entry.structure
    return f"{entry.structure}/{entry.motif}"


def _merge_cluster_lookup_row(
    lookup: dict[tuple[str, str], dict[str, object]],
    row: dict[str, object],
) -> None:
    key = (
        str(row.get("structure", "")).strip(),
        str(row.get("motif", "no_motif")).strip() or "no_motif",
    )
    if not key[0]:
        return
    merged = dict(lookup.get(key, {}))
    merged.update(dict(row))
    lookup[key] = merged


def _resolve_cluster_row_source_file(
    row: dict[str, object],
) -> str | None:
    source_file = _optional_text(row.get("source_file"))
    if source_file is not None:
        return source_file
    source_dir = _optional_text(row.get("source_dir"))
    representative = _optional_text(row.get("representative"))
    if source_dir is None or representative is None:
        return None
    return str(
        (Path(source_dir).expanduser().resolve() / representative).resolve()
    )


def _resolve_cluster_row_source_file_name(
    row: dict[str, object],
    *,
    source_file: str | None,
) -> str | None:
    source_file_name = _optional_text(row.get("source_file_name"))
    if source_file_name is not None:
        return source_file_name
    representative = _optional_text(row.get("representative"))
    if representative is not None:
        return representative
    if source_file is None:
        return None
    return Path(source_file).name


def _resolve_cluster_row_source_kind(
    row: dict[str, object],
    *,
    source_dir: str | None,
    source_file: str | None,
) -> str | None:
    source_kind = _optional_text(row.get("source_kind"))
    if source_kind is not None:
        return source_kind
    raw_source_file = _optional_text(row.get("source_file"))
    raw_source_dir = _optional_text(row.get("source_dir"))
    if raw_source_file is not None and raw_source_dir is None:
        return "single_structure_file"
    if source_file is not None and source_dir is None:
        return "single_structure_file"
    if source_dir is not None:
        return "cluster_dir"
    return None


def _aggregate_candidate_values(
    candidates: list[_CandidateMeasurement],
    bond_pairs: tuple[BondPairDefinition, ...],
    angle_triplets: tuple[AngleTripletDefinition, ...],
) -> tuple[
    dict[BondPairDefinition, np.ndarray],
    dict[AngleTripletDefinition, np.ndarray],
]:
    bond_values: dict[BondPairDefinition, np.ndarray] = {}
    angle_values: dict[AngleTripletDefinition, np.ndarray] = {}
    for definition in bond_pairs:
        merged = [
            value
            for candidate in candidates
            for value in candidate.bond_values.get(definition, [])
        ]
        bond_values[definition] = np.asarray(merged, dtype=float)
    for definition in angle_triplets:
        merged = [
            value
            for candidate in candidates
            for value in candidate.angle_values.get(definition, [])
        ]
        angle_values[definition] = np.asarray(merged, dtype=float)
    return bond_values, angle_values


def _select_best_distribution_candidate(
    candidates: list[_CandidateMeasurement],
    *,
    target_bonds: dict[BondPairDefinition, np.ndarray],
    target_angles: dict[AngleTripletDefinition, np.ndarray],
    settings: RepresentativeSelectionSettings,
) -> tuple[_CandidateMeasurement, float, float, float]:
    best_candidate = candidates[0]
    best_score = float("inf")
    best_bond_score = 0.0
    best_angle_score = 0.0
    for candidate in candidates:
        if (
            settings.selection_algorithm
            == "target_distribution_moment_distance"
        ):
            bond_score = _category_moment_distance(
                target_bonds,
                candidate.bond_values,
            )
            angle_score = _category_moment_distance(
                target_angles,
                candidate.angle_values,
            )
        else:
            bond_score = _category_distance(
                target_bonds,
                candidate.bond_values,
                quantiles=settings.quantiles,
            )
            angle_score = _category_distance(
                target_angles,
                candidate.angle_values,
                quantiles=settings.quantiles,
            )
        total_score = (
            settings.bond_weight * bond_score
            + settings.angle_weight * angle_score
        )
        if total_score < best_score:
            best_candidate = candidate
            best_score = total_score
            best_bond_score = bond_score
            best_angle_score = angle_score
    return best_candidate, best_bond_score, best_angle_score, best_score


def _category_distance(
    target_values: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
    *,
    quantiles: tuple[float, ...],
) -> float:
    distances: list[float] = []
    for definition, target in target_values.items():
        target_vector = _quantile_feature_vector(target, quantiles=quantiles)
        candidate_vector = _quantile_feature_vector(
            candidate_values.get(definition, []),
            quantiles=quantiles,
        )
        distances.append(
            float(np.linalg.norm(candidate_vector - target_vector))
        )
    if not distances:
        return 0.0
    return float(np.mean(distances))


def _category_moment_distance(
    target_values: dict[object, np.ndarray],
    candidate_values: dict[object, list[float]],
) -> float:
    distances: list[float] = []
    for definition, target in target_values.items():
        target_vector = _moment_feature_vector(target)
        candidate_vector = _moment_feature_vector(
            candidate_values.get(definition, [])
        )
        distances.append(
            float(np.linalg.norm(candidate_vector - target_vector))
        )
    if not distances:
        return 0.0
    return float(np.mean(distances))


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


def _representative_line_values(
    values: list[float] | np.ndarray,
) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.size <= 0:
        return ()
    return tuple(float(value) for value in np.unique(array))


def _sorted_structure_files(source_dir: Path) -> list[Path]:
    return sorted(
        [
            candidate
            for candidate in source_dir.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in _STRUCTURE_SUFFIXES
        ],
        key=lambda path: _natural_sort_key(path.name),
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


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _definition_chunks(text: str) -> list[str]:
    return [
        chunk.strip() for chunk in re.split(r"[;\n]+", text) if chunk.strip()
    ]


__all__ = [
    "DistributionSelectionEntry",
    "DistributionSelectionMetadata",
    "RepresentativeSelectionSettings",
    "RepresentativeSelectionEntry",
    "RepresentativeSelectionIssue",
    "RepresentativeSelectionMetadata",
    "RepresentativePreviewCluster",
    "RepresentativePreviewSeries",
    "build_distribution_selection",
    "build_representative_preview_clusters",
    "load_distribution_selection_metadata",
    "load_representative_selection_metadata",
    "parse_angle_triplet_text",
    "parse_bond_pair_text",
    "save_distribution_selection_metadata",
    "save_representative_selection_metadata",
    "select_distribution_representatives",
    "select_first_file_representatives",
]
