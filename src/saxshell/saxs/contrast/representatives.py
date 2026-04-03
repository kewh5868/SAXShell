from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from saxshell.saxs.contrast.descriptors import (
    ContrastStructureDescriptor,
    ParsedContrastStructure,
    describe_parsed_contrast_structure,
    estimate_pair_contact_distance_medians,
    load_parsed_contrast_structure,
)
from saxshell.saxs.contrast.settings import (
    ContrastRepresentativeSamplerSettings,
)
from saxshell.saxs.debye import ClusterBin, discover_cluster_bins
from saxshell.saxs.stoichiometry import parse_stoich_label

RepresentativeAnalysisProgressCallback = Callable[[int, int, str], None]
RepresentativeAnalysisLogCallback = Callable[[str], None]


def _emit_progress(
    callback: RepresentativeAnalysisProgressCallback | None,
    processed: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(max(int(processed), 0), max(int(total), 1), str(message).strip())


def _emit_log(
    callback: RepresentativeAnalysisLogCallback | None,
    message: str,
) -> None:
    if callback is None:
        return
    text = str(message).strip()
    if text:
        callback(text)


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _cluster_bin_label(cluster_bin: ClusterBin) -> str:
    if cluster_bin.motif == "no_motif":
        return cluster_bin.structure
    return f"{cluster_bin.structure}/{cluster_bin.motif}"


def _cluster_bin_slug(cluster_bin: ClusterBin) -> str:
    if cluster_bin.motif == "no_motif":
        return cluster_bin.structure
    return f"{cluster_bin.structure}__{cluster_bin.motif}"


def _median_summary(
    descriptor_getter: Callable[
        [ContrastStructureDescriptor], dict[str, float]
    ],
    descriptors: tuple[ContrastStructureDescriptor, ...],
) -> dict[str, float]:
    values_by_key: defaultdict[str, list[float]] = defaultdict(list)
    for descriptor in descriptors:
        for key, value in descriptor_getter(descriptor).items():
            values_by_key[str(key)].append(float(value))
    return {
        key: (
            float(sorted(values)[len(values) // 2])
            if len(values) % 2 == 1
            else float(
                (
                    sorted(values)[len(values) // 2 - 1]
                    + sorted(values)[len(values) // 2]
                )
                / 2.0
            )
        )
        for key, values in sorted(values_by_key.items())
        if values
    }


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
            target_values[key],
            scale_floor=default_scale,
            missing_penalty=missing_penalty,
        )
        for key in sorted(target_values)
    ]
    return float(sum(deltas) / len(deltas)) if deltas else 0.0


@dataclass(slots=True, frozen=True)
class ContrastRepresentativeTargetSummary:
    pair_contact_distance_medians: dict[str, float]
    bond_length_medians: dict[str, float]
    angle_medians: dict[str, float]
    coordination_medians: dict[str, float]
    solvent_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_contact_distance_medians": dict(
                sorted(self.pair_contact_distance_medians.items())
            ),
            "bond_length_medians": dict(
                sorted(self.bond_length_medians.items())
            ),
            "angle_medians": dict(sorted(self.angle_medians.items())),
            "coordination_medians": dict(
                sorted(self.coordination_medians.items())
            ),
            "solvent_metrics": dict(sorted(self.solvent_metrics.items())),
        }


@dataclass(slots=True, frozen=True)
class ContrastRepresentativeCandidate:
    descriptor: ContrastStructureDescriptor
    score_total: float
    score_bond: float
    score_angle: float
    score_coordination: float
    score_solvent: float

    def to_dict(self) -> dict[str, object]:
        return {
            "file_path": str(self.descriptor.file_path),
            "score_total": float(self.score_total),
            "score_bond": float(self.score_bond),
            "score_angle": float(self.score_angle),
            "score_coordination": float(self.score_coordination),
            "score_solvent": float(self.score_solvent),
            "descriptor": self.descriptor.to_dict(),
        }


class ContrastRepresentativeIssue:
    structure: str
    motif: str
    source: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "motif": self.motif,
            "source": self.source,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class ContrastRepresentativeBinResult:
    structure: str
    motif: str
    source_dir: Path
    file_count: int
    selected_file: Path
    copied_representative_file: Path
    target_summary: ContrastRepresentativeTargetSummary
    selected_candidate: ContrastRepresentativeCandidate
    candidates: tuple[ContrastRepresentativeCandidate, ...]
    selection_strategy: str
    distribution_sample_count: int
    sampled_candidate_count: int
    sampler_settings: ContrastRepresentativeSamplerSettings
    screening_json_path: Path
    screening_table_path: Path
    notes: tuple[str, ...]

    @property
    def display_label(self) -> str:
        if self.motif == "no_motif":
            return self.structure
        return f"{self.structure}/{self.motif}"

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "motif": self.motif,
            "display_label": self.display_label,
            "source_dir": str(self.source_dir),
            "file_count": self.file_count,
            "selected_file": str(self.selected_file),
            "copied_representative_file": str(self.copied_representative_file),
            "target_summary": self.target_summary.to_dict(),
            "selected_candidate": self.selected_candidate.to_dict(),
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "selection_strategy": self.selection_strategy,
            "distribution_sample_count": self.distribution_sample_count,
            "sampled_candidate_count": self.sampled_candidate_count,
            "sampler_settings": self.sampler_settings.to_dict(),
            "screening_json_path": str(self.screening_json_path),
            "screening_table_path": str(self.screening_table_path),
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class ContrastRepresentativeSelectionResult:
    project_dir: Path
    clusters_dir: Path
    output_dir: Path
    representative_structures_dir: Path
    screening_dir: Path
    generated_at: str
    bin_results: tuple[ContrastRepresentativeBinResult, ...]
    issues: tuple[ContrastRepresentativeIssue, ...]
    summary_json_path: Path
    summary_table_path: Path
    summary_text_path: Path

    def summary_text(self) -> str:
        lines = [
            "Contrast representative selection complete",
            f"Generated at: {self.generated_at}",
            f"Project folder: {self.project_dir}",
            f"Clusters folder: {self.clusters_dir}",
            f"Output folder: {self.output_dir}",
            f"Processed bins: {len(self.bin_results)}",
            f"Issues: {len(self.issues)}",
        ]
        if self.bin_results:
            lines.extend(["", "Selected representatives"])
            for bin_result in self.bin_results:
                lines.append(
                    "  "
                    + f"{bin_result.display_label}: {bin_result.selected_file.name} "
                    + "(descriptor-distance score "
                    + f"{bin_result.selected_candidate.score_total:.4f}, lower is better)"
                )
        if self.issues:
            lines.extend(["", "Issues"])
            for issue in self.issues:
                lines.append(
                    "  " + f"{issue.structure}/{issue.motif}: {issue.message}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "generated_at": self.generated_at,
            "project_dir": str(self.project_dir),
            "clusters_dir": str(self.clusters_dir),
            "output_dir": str(self.output_dir),
            "representative_structures_dir": str(
                self.representative_structures_dir
            ),
            "screening_dir": str(self.screening_dir),
            "summary_json_path": str(self.summary_json_path),
            "summary_table_path": str(self.summary_table_path),
            "summary_text_path": str(self.summary_text_path),
            "bin_results": [result.to_dict() for result in self.bin_results],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _output_dirs(
    project_dir: Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (project_dir / "contrast_workflow" / "representatives").resolve()
    )
    representative_structures_dir = root / "representative_structures"
    screening_dir = root / "screening"
    representative_structures_dir.mkdir(parents=True, exist_ok=True)
    screening_dir.mkdir(parents=True, exist_ok=True)
    return root, representative_structures_dir, screening_dir


def _target_summary_for_descriptors(
    descriptors: tuple[ContrastStructureDescriptor, ...],
    pair_contact_distance_medians: dict[str, float],
) -> ContrastRepresentativeTargetSummary:
    return ContrastRepresentativeTargetSummary(
        pair_contact_distance_medians=dict(
            sorted(pair_contact_distance_medians.items())
        ),
        bond_length_medians=_median_summary(
            lambda descriptor: descriptor.bond_length_medians,
            descriptors,
        ),
        angle_medians=_median_summary(
            lambda descriptor: descriptor.angle_medians,
            descriptors,
        ),
        coordination_medians=_median_summary(
            lambda descriptor: descriptor.coordination_medians,
            descriptors,
        ),
        solvent_metrics=_median_summary(
            lambda descriptor: descriptor.solvent_metrics(),
            descriptors,
        ),
    )


def _candidate_for_descriptor(
    descriptor: ContrastStructureDescriptor,
    *,
    target_summary: ContrastRepresentativeTargetSummary,
) -> ContrastRepresentativeCandidate:
    score_bond = _score_feature_map(
        descriptor.bond_length_medians,
        target_summary.bond_length_medians,
        default_scale=1.0,
    )
    score_angle = _score_feature_map(
        descriptor.angle_medians,
        target_summary.angle_medians,
        default_scale=180.0,
    )
    score_coordination = _score_feature_map(
        descriptor.coordination_medians,
        target_summary.coordination_medians,
        default_scale=1.0,
    )
    score_solvent = _score_feature_map(
        descriptor.solvent_metrics(),
        target_summary.solvent_metrics,
        default_scale=1.0,
    )
    score_total = float(
        (score_bond + score_angle + score_coordination + score_solvent) / 4.0
    )
    return ContrastRepresentativeCandidate(
        descriptor=descriptor,
        score_total=score_total,
        score_bond=score_bond,
        score_angle=score_angle,
        score_coordination=score_coordination,
        score_solvent=score_solvent,
    )


def _candidate_rank_key(
    candidate: ContrastRepresentativeCandidate,
) -> tuple[float, float, float, str]:
    return (
        float(candidate.score_total),
        float(candidate.score_solvent),
        float(candidate.score_bond),
        str(candidate.descriptor.file_path),
    )


def _evaluate_descriptor_candidates(
    parsed_structures: tuple[ParsedContrastStructure, ...],
    *,
    expected_core_counts: dict[str, int],
) -> tuple[
    dict[str, float],
    ContrastRepresentativeTargetSummary,
    tuple[ContrastRepresentativeCandidate, ...],
]:
    pair_contact_distance_medians = estimate_pair_contact_distance_medians(
        parsed_structures
    )
    descriptors = tuple(
        describe_parsed_contrast_structure(
            parsed_structure,
            expected_core_counts=expected_core_counts,
            pair_contact_distance_medians=pair_contact_distance_medians,
        )
        for parsed_structure in parsed_structures
    )
    target_summary = _target_summary_for_descriptors(
        descriptors,
        pair_contact_distance_medians,
    )
    candidates = tuple(
        sorted(
            (
                _candidate_for_descriptor(
                    descriptor,
                    target_summary=target_summary,
                )
                for descriptor in descriptors
            ),
            key=_candidate_rank_key,
        )
    )
    return pair_contact_distance_medians, target_summary, candidates


def _sample_indices(
    total_count: int,
    sample_count: int,
    *,
    rng: np.random.Generator,
    stratify: bool,
    shuffle: bool,
) -> tuple[int, ...]:
    total = max(int(total_count), 0)
    limit = min(max(int(sample_count), 0), total)
    if limit <= 0:
        return tuple()
    if limit >= total:
        indices = list(range(total))
        if shuffle and len(indices) > 1:
            rng.shuffle(indices)
        return tuple(int(index) for index in indices)
    if not stratify:
        sampled = np.atleast_1d(
            rng.choice(total, size=limit, replace=False)
        ).tolist()
        indices = [int(index) for index in sampled]
        if not shuffle:
            indices.sort()
        return tuple(indices)

    chosen: list[int] = []
    edges = np.linspace(0, total, limit + 1, dtype=int)
    used: set[int] = set()
    for start, end in zip(edges[:-1], edges[1:], strict=False):
        if end <= start:
            continue
        local_candidates = [
            index for index in range(int(start), int(end)) if index not in used
        ]
        if not local_candidates:
            continue
        local_choice = int(rng.integers(0, len(local_candidates)))
        selected_index = int(local_candidates[local_choice])
        used.add(selected_index)
        chosen.append(selected_index)
    if len(chosen) < limit:
        remaining = sorted(set(range(total)) - used)
        if remaining:
            extra_offsets = np.atleast_1d(
                rng.choice(
                    len(remaining),
                    size=min(limit - len(chosen), len(remaining)),
                    replace=False,
                )
            ).tolist()
            for offset in extra_offsets:
                chosen.append(int(remaining[int(offset)]))
    if shuffle and len(chosen) > 1:
        rng.shuffle(chosen)
    elif not shuffle:
        chosen.sort()
    return tuple(chosen[:limit])


def _load_parsed_structure_cached(
    file_path: Path,
    *,
    parsed_cache: dict[Path, ParsedContrastStructure],
    issues: list[ContrastRepresentativeIssue],
    cluster_bin: ClusterBin,
) -> ParsedContrastStructure | None:
    cached = parsed_cache.get(file_path)
    if cached is not None:
        return cached
    try:
        parsed = load_parsed_contrast_structure(file_path)
    except Exception as exc:
        issues.append(
            ContrastRepresentativeIssue(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                source=str(file_path),
                message=f"Unable to parse structure file: {exc}",
            )
        )
        return None
    parsed_cache[file_path] = parsed
    return parsed


def _describe_candidate_cached(
    file_path: Path,
    *,
    parsed_cache: dict[Path, ParsedContrastStructure],
    descriptor_cache: dict[Path, ContrastStructureDescriptor],
    pair_contact_distance_medians: dict[str, float],
    expected_core_counts: dict[str, int],
    issues: list[ContrastRepresentativeIssue],
    cluster_bin: ClusterBin,
) -> ContrastStructureDescriptor | None:
    cached = descriptor_cache.get(file_path)
    if cached is not None:
        return cached
    parsed_structure = _load_parsed_structure_cached(
        file_path,
        parsed_cache=parsed_cache,
        issues=issues,
        cluster_bin=cluster_bin,
    )
    if parsed_structure is None:
        return None
    descriptor = describe_parsed_contrast_structure(
        parsed_structure,
        expected_core_counts=expected_core_counts,
        pair_contact_distance_medians=pair_contact_distance_medians,
    )
    descriptor_cache[file_path] = descriptor
    return descriptor


def _sampled_descriptor_candidates(
    cluster_bin: ClusterBin,
    *,
    expected_core_counts: dict[str, int],
    sampler_settings: ContrastRepresentativeSamplerSettings,
    log_callback: RepresentativeAnalysisLogCallback | None,
    parsed_cache: dict[Path, ParsedContrastStructure],
    issues: list[ContrastRepresentativeIssue],
) -> tuple[
    ContrastRepresentativeTargetSummary,
    tuple[ContrastRepresentativeCandidate, ...],
    str,
    int,
    int,
]:
    file_paths = tuple(cluster_bin.files)
    total_candidates = len(file_paths)
    distribution_rng = np.random.default_rng(int(sampler_settings.random_seed))
    distribution_limit = sampler_settings.distribution_sample_limit(
        total_candidates
    )
    candidate_limit = sampler_settings.candidate_sample_limit(total_candidates)

    if (
        not sampler_settings.should_use_sampling(total_candidates)
        or candidate_limit >= total_candidates
    ):
        parsed_structures = tuple(
            parsed_structure
            for file_path in file_paths
            if (
                parsed_structure := _load_parsed_structure_cached(
                    file_path,
                    parsed_cache=parsed_cache,
                    issues=issues,
                    cluster_bin=cluster_bin,
                )
            )
            is not None
        )
        if not parsed_structures:
            raise ValueError(
                f"No valid candidate structures were available in {_cluster_bin_label(cluster_bin)}."
            )
        (
            _pair_contact_distance_medians,
            target_summary,
            candidates,
        ) = _evaluate_descriptor_candidates(
            parsed_structures,
            expected_core_counts=expected_core_counts,
        )
        return (
            target_summary,
            candidates,
            "full_scan",
            len(parsed_structures),
            len(candidates),
        )

    distribution_indices = _sample_indices(
        total_candidates,
        distribution_limit,
        rng=distribution_rng,
        stratify=bool(sampler_settings.stratify_sampling),
        shuffle=False,
    )
    distribution_structures = tuple(
        parsed_structure
        for index in distribution_indices
        if (
            parsed_structure := _load_parsed_structure_cached(
                file_paths[index],
                parsed_cache=parsed_cache,
                issues=issues,
                cluster_bin=cluster_bin,
            )
        )
        is not None
    )
    if not distribution_structures:
        raise ValueError(
            f"Unable to build a target descriptor distribution for {_cluster_bin_label(cluster_bin)}."
        )
    (
        pair_contact_distance_medians,
        target_summary,
        _distribution_candidates,
    ) = _evaluate_descriptor_candidates(
        distribution_structures,
        expected_core_counts=expected_core_counts,
    )
    distribution_sample_count = len(distribution_structures)
    _emit_log(
        log_callback,
        (
            f"{_cluster_bin_label(cluster_bin)}: estimated the fixed median "
            f"target from {distribution_sample_count}/{total_candidates} sampled "
            "structure(s). Lower scores mean a closer descriptor match to that "
            "fixed target."
        ),
    )

    candidate_rng = np.random.default_rng(
        int(sampler_settings.random_seed) + 1
    )
    candidate_indices = list(
        _sample_indices(
            total_candidates,
            candidate_limit,
            rng=candidate_rng,
            stratify=bool(sampler_settings.stratify_sampling),
            shuffle=True,
        )
    )
    descriptor_cache: dict[Path, ContrastStructureDescriptor] = {}
    evaluated_candidates: list[ContrastRepresentativeCandidate] = []
    stable_rounds = 0
    minimum_candidate_samples = min(
        max(int(sampler_settings.minimum_candidate_samples), 1),
        max(len(candidate_indices), 1),
    )
    current_best: ContrastRepresentativeCandidate | None = None
    significant_best_score: float | None = None

    _emit_log(
        log_callback,
        (
            f"{_cluster_bin_label(cluster_bin)}: evaluating up to "
            f"{len(candidate_indices)} random candidate sample(s) against the "
            f"fixed target with seed {sampler_settings.random_seed}. The score "
            "is a normalized descriptor-distance, so lower is better."
        ),
    )
    while candidate_indices:
        batch_size = min(
            int(sampler_settings.candidate_batch_size),
            len(candidate_indices),
        )
        current_batch = [candidate_indices.pop(0) for _ in range(batch_size)]
        batch_improved = False
        for index in current_batch:
            file_path = file_paths[index]
            descriptor = _describe_candidate_cached(
                file_path,
                parsed_cache=parsed_cache,
                descriptor_cache=descriptor_cache,
                pair_contact_distance_medians=pair_contact_distance_medians,
                expected_core_counts=expected_core_counts,
                issues=issues,
                cluster_bin=cluster_bin,
            )
            if descriptor is None:
                continue
            candidate = _candidate_for_descriptor(
                descriptor,
                target_summary=target_summary,
            )
            evaluated_candidates.append(candidate)
            if current_best is None or _candidate_rank_key(
                candidate
            ) < _candidate_rank_key(current_best):
                current_best = candidate
            if significant_best_score is None or candidate.score_total < float(
                significant_best_score
            ) - float(sampler_settings.improvement_tolerance):
                batch_improved = True
                significant_best_score = float(candidate.score_total)
        if current_best is None:
            continue
        if len(evaluated_candidates) >= minimum_candidate_samples:
            if batch_improved:
                stable_rounds = 0
            else:
                stable_rounds += 1
        _emit_log(
            log_callback,
            (
                f"{_cluster_bin_label(cluster_bin)}: evaluated "
                f"{len(evaluated_candidates)}/"
                f"{len(evaluated_candidates) + len(candidate_indices)} candidate "
                f"sample(s) against a fixed target built from "
                f"{distribution_sample_count} sampled structure(s); current best "
                "descriptor-distance match is "
                f"{current_best.descriptor.file_path.name} with score "
                f"{current_best.score_total:.4f} (lower is better)."
            ),
        )
        if len(
            evaluated_candidates
        ) >= minimum_candidate_samples and stable_rounds >= int(
            sampler_settings.convergence_patience
        ):
            break

    if not evaluated_candidates:
        raise ValueError(
            f"No valid candidate structures were evaluated for {_cluster_bin_label(cluster_bin)}."
        )
    candidates = tuple(
        sorted(
            evaluated_candidates,
            key=_candidate_rank_key,
        )
    )
    return (
        target_summary,
        candidates,
        "monte_carlo_sampling",
        distribution_sample_count,
        len(candidates),
    )


def _copy_selected_representative(
    *,
    cluster_bin: ClusterBin,
    selected_file: Path,
    representative_structures_dir: Path,
) -> Path:
    destination = (
        representative_structures_dir
        / f"{_cluster_bin_slug(cluster_bin)}__{selected_file.name}"
    )
    shutil.copy2(selected_file, destination)
    return destination


def _write_screening_outputs(
    *,
    cluster_bin: ClusterBin,
    screening_dir: Path,
    target_summary: ContrastRepresentativeTargetSummary,
    candidates: tuple[ContrastRepresentativeCandidate, ...],
    selected_candidate: ContrastRepresentativeCandidate,
    selection_strategy: str,
    distribution_sample_count: int,
    sampled_candidate_count: int,
    sampler_settings: ContrastRepresentativeSamplerSettings,
    notes: tuple[str, ...],
) -> tuple[Path, Path]:
    screening_json_path = (
        screening_dir / f"{_cluster_bin_slug(cluster_bin)}.json"
    )
    screening_table_path = (
        screening_dir / f"{_cluster_bin_slug(cluster_bin)}.tsv"
    )
    screening_payload = {
        "structure": cluster_bin.structure,
        "motif": cluster_bin.motif,
        "display_label": _cluster_bin_label(cluster_bin),
        "source_dir": str(cluster_bin.source_dir),
        "file_count": len(cluster_bin.files),
        "selected_file": str(selected_candidate.descriptor.file_path),
        "selected_score_total": float(selected_candidate.score_total),
        "selection_strategy": selection_strategy,
        "distribution_sample_count": int(distribution_sample_count),
        "sampled_candidate_count": int(sampled_candidate_count),
        "sampler_settings": sampler_settings.to_dict(),
        "target_summary": target_summary.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "notes": list(notes),
    }
    screening_json_path.write_text(
        json.dumps(screening_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with screening_table_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "file_name",
                "selection_strategy",
                "score_total",
                "score_bond",
                "score_angle",
                "score_coordination",
                "score_solvent",
                "atom_count",
                "solvent_atoms",
                "direct_solvent_atoms",
                "outer_solvent_atoms",
                "mean_direct_solvent_coordination",
            ]
        )
        for candidate in candidates:
            descriptor = candidate.descriptor
            writer.writerow(
                [
                    descriptor.file_path.name,
                    selection_strategy,
                    f"{candidate.score_total:.8f}",
                    f"{candidate.score_bond:.8f}",
                    f"{candidate.score_angle:.8f}",
                    f"{candidate.score_coordination:.8f}",
                    f"{candidate.score_solvent:.8f}",
                    descriptor.atom_count,
                    descriptor.solvent_atom_count,
                    descriptor.direct_solvent_atom_count,
                    descriptor.outer_solvent_atom_count,
                    f"{descriptor.mean_direct_solvent_coordination:.8f}",
                ]
            )
    return screening_json_path, screening_table_path


def _write_selection_summary(
    *,
    result: ContrastRepresentativeSelectionResult,
) -> None:
    result.summary_json_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    with result.summary_table_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "structure",
                "motif",
                "display_label",
                "selected_file_name",
                "copied_representative_file",
                "selection_strategy",
                "distribution_sample_count",
                "sampled_candidate_count",
                "score_total",
                "score_bond",
                "score_angle",
                "score_coordination",
                "score_solvent",
                "atom_count",
                "solvent_atoms",
                "direct_solvent_atoms",
                "outer_solvent_atoms",
            ]
        )
        for bin_result in result.bin_results:
            descriptor = bin_result.selected_candidate.descriptor
            writer.writerow(
                [
                    bin_result.structure,
                    bin_result.motif,
                    bin_result.display_label,
                    bin_result.selected_file.name,
                    str(bin_result.copied_representative_file),
                    bin_result.selection_strategy,
                    bin_result.distribution_sample_count,
                    bin_result.sampled_candidate_count,
                    f"{bin_result.selected_candidate.score_total:.8f}",
                    f"{bin_result.selected_candidate.score_bond:.8f}",
                    f"{bin_result.selected_candidate.score_angle:.8f}",
                    f"{bin_result.selected_candidate.score_coordination:.8f}",
                    f"{bin_result.selected_candidate.score_solvent:.8f}",
                    descriptor.atom_count,
                    descriptor.solvent_atom_count,
                    descriptor.direct_solvent_atom_count,
                    descriptor.outer_solvent_atom_count,
                ]
            )
    result.summary_text_path.write_text(
        result.summary_text() + "\n",
        encoding="utf-8",
    )


def _analyze_cluster_bin(
    cluster_bin: ClusterBin,
    *,
    representative_structures_dir: Path,
    screening_dir: Path,
    sampler_settings: ContrastRepresentativeSamplerSettings,
    log_callback: RepresentativeAnalysisLogCallback | None,
) -> tuple[
    ContrastRepresentativeBinResult | None, list[ContrastRepresentativeIssue]
]:
    issues: list[ContrastRepresentativeIssue] = []
    if not cluster_bin.files:
        issues.append(
            ContrastRepresentativeIssue(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                source=str(cluster_bin.source_dir),
                message="No valid structure files were available for representative screening.",
            )
        )
        return None, issues

    expected_core_counts = parse_stoich_label(cluster_bin.structure)
    parsed_cache: dict[Path, ParsedContrastStructure] = {}
    (
        target_summary,
        candidates,
        selection_strategy,
        distribution_sample_count,
        sampled_candidate_count,
    ) = _sampled_descriptor_candidates(
        cluster_bin,
        expected_core_counts=expected_core_counts,
        sampler_settings=sampler_settings,
        log_callback=log_callback,
        parsed_cache=parsed_cache,
        issues=issues,
    )
    selected_candidate = candidates[0]
    copied_representative_file = _copy_selected_representative(
        cluster_bin=cluster_bin,
        selected_file=selected_candidate.descriptor.file_path,
        representative_structures_dir=representative_structures_dir,
    )
    if selection_strategy == "monte_carlo_sampling":
        notes = (
            "Estimated a fixed bin-wide target from a sampled subset of the "
            "complete structure distribution, then used random Monte Carlo "
            "candidate sampling to find the representative structure that best "
            "matched that fixed median target.",
            "The descriptor-distance score is the average normalized mismatch "
            "across bond lengths, bond angles, coordination counts, and "
            "solvent-shell metrics, so lower scores indicate a closer match.",
            (
                f"Target metrics were estimated from {distribution_sample_count} "
                f"sampled structure(s), and {sampled_candidate_count} "
                f"candidate structure(s) were scored against that fixed target "
                f"out of {len(cluster_bin.files)} total candidate structure(s)."
            ),
            "Descriptor scoring uses bond lengths, bond angles, coordination "
            "numbers, and solvent-shell counts from the isolated contrast backend.",
            *selected_candidate.descriptor.notes,
        )
    else:
        notes = (
            "Selected the lowest-distance existing structure against the bin-wide "
            "median descriptor summary.",
            "The descriptor-distance score is the average normalized mismatch "
            "across bond lengths, bond angles, coordination counts, and "
            "solvent-shell metrics, so lower scores indicate a closer match.",
            (
                f"Full contrast descriptors were evaluated for all "
                f"{len(cluster_bin.files)} candidate structure(s) in this bin "
                "to estimate the full distribution and select the best match."
            ),
            "Descriptor scoring uses bond lengths, bond angles, coordination "
            "numbers, and solvent-shell counts from the isolated contrast backend.",
            *selected_candidate.descriptor.notes,
        )
    screening_json_path, screening_table_path = _write_screening_outputs(
        cluster_bin=cluster_bin,
        screening_dir=screening_dir,
        target_summary=target_summary,
        candidates=candidates,
        selected_candidate=selected_candidate,
        selection_strategy=selection_strategy,
        distribution_sample_count=distribution_sample_count,
        sampled_candidate_count=sampled_candidate_count,
        sampler_settings=sampler_settings,
        notes=notes,
    )
    return (
        ContrastRepresentativeBinResult(
            structure=cluster_bin.structure,
            motif=cluster_bin.motif,
            source_dir=cluster_bin.source_dir,
            file_count=len(cluster_bin.files),
            selected_file=selected_candidate.descriptor.file_path,
            copied_representative_file=copied_representative_file,
            target_summary=target_summary,
            selected_candidate=selected_candidate,
            candidates=candidates,
            selection_strategy=selection_strategy,
            distribution_sample_count=distribution_sample_count,
            sampled_candidate_count=sampled_candidate_count,
            sampler_settings=sampler_settings,
            screening_json_path=screening_json_path,
            screening_table_path=screening_table_path,
            notes=notes,
        ),
        issues,
    )


def analyze_contrast_representatives(
    project_dir: str | Path,
    clusters_dir: str | Path,
    *,
    cluster_bins: tuple[ClusterBin, ...] | list[ClusterBin] | None = None,
    output_dir: str | Path | None = None,
    sampler_settings: ContrastRepresentativeSamplerSettings | None = None,
    progress_callback: RepresentativeAnalysisProgressCallback | None = None,
    log_callback: RepresentativeAnalysisLogCallback | None = None,
) -> ContrastRepresentativeSelectionResult:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    resolved_clusters_dir = Path(clusters_dir).expanduser().resolve()
    if not resolved_project_dir.is_dir():
        raise ValueError(
            f"Project directory does not exist: {resolved_project_dir}"
        )
    cluster_bins = (
        tuple(cluster_bins)
        if cluster_bins is not None
        else tuple(discover_cluster_bins(resolved_clusters_dir))
    )
    if not cluster_bins:
        raise ValueError(
            f"No recognized cluster bins were found in {resolved_clusters_dir}"
        )

    root_output_dir, representative_structures_dir, screening_dir = (
        _output_dirs(
            resolved_project_dir,
            output_dir=output_dir,
        )
    )
    normalized_sampler_settings = (
        sampler_settings
        if sampler_settings is not None
        else ContrastRepresentativeSamplerSettings()
    )
    _emit_log(
        log_callback,
        f"Starting contrast representative screening for {len(cluster_bins)} cluster bin(s).",
    )
    bin_results: list[ContrastRepresentativeBinResult] = []
    issues: list[ContrastRepresentativeIssue] = []

    for index, cluster_bin in enumerate(cluster_bins, start=1):
        _emit_progress(
            progress_callback,
            index - 1,
            len(cluster_bins),
            f"Analyzing {_cluster_bin_label(cluster_bin)} ({index}/{len(cluster_bins)})",
        )
        bin_result, bin_issues = _analyze_cluster_bin(
            cluster_bin,
            representative_structures_dir=representative_structures_dir,
            screening_dir=screening_dir,
            sampler_settings=normalized_sampler_settings,
            log_callback=log_callback,
        )
        issues.extend(bin_issues)
        if bin_result is None:
            _emit_log(
                log_callback,
                f"Skipped {_cluster_bin_label(cluster_bin)} because no valid candidate structures were available.",
            )
            continue
        bin_results.append(bin_result)
        _emit_log(
            log_callback,
            f"Selected {bin_result.selected_file.name} for {bin_result.display_label} "
            "with descriptor-distance score "
            f"{bin_result.selected_candidate.score_total:.4f} (lower is better).",
        )

    if not bin_results:
        raise ValueError(
            "Representative screening did not produce any valid outputs."
        )

    result = ContrastRepresentativeSelectionResult(
        project_dir=resolved_project_dir,
        clusters_dir=resolved_clusters_dir,
        output_dir=root_output_dir,
        representative_structures_dir=representative_structures_dir,
        screening_dir=screening_dir,
        generated_at=_now_text(),
        bin_results=tuple(bin_results),
        issues=tuple(issues),
        summary_json_path=root_output_dir / "selection_summary.json",
        summary_table_path=root_output_dir / "selection_summary.tsv",
        summary_text_path=root_output_dir / "selection_summary.txt",
    )
    _write_selection_summary(result=result)
    _emit_progress(
        progress_callback,
        len(cluster_bins),
        len(cluster_bins),
        f"Representative screening complete ({len(bin_results)} bin(s) selected)",
    )
    _emit_log(
        log_callback,
        f"Wrote contrast representative outputs to {result.output_dir}.",
    )
    return result
