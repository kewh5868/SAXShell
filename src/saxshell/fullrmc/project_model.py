from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from saxshell.saxs.project_manager import (
    ProjectPaths,
    ProjectSettings,
    build_project_paths,
)

_STRUCTURE_SUFFIXES = {".pdb", ".xyz"}


@dataclass(slots=True)
class RMCSetupPaths:
    project_dir: Path
    rmcsetup_dir: Path
    representative_clusters_dir: Path
    representative_selection_path: Path
    pdb_no_solvent_dir: Path
    pdb_with_solvent_dir: Path
    packmol_inputs_dir: Path
    constraints_dir: Path
    reports_dir: Path
    distribution_selection_path: Path
    solution_properties_path: Path
    solvent_handling_path: Path
    packmol_plan_path: Path
    packmol_setup_path: Path
    constraint_generation_path: Path
    packmol_plan_report_path: Path
    packmol_audit_report_path: Path
    cluster_counts_csv_path: Path
    planned_count_weights_csv_path: Path
    planned_atom_weights_csv_path: Path
    packmol_input_path: Path
    packmol_output_pdb_path: Path
    merged_constraints_path: Path


@dataclass(slots=True)
class ClusterSourceValidationResult:
    clusters_dir: Path | None
    expected_rows: list[dict[str, object]]
    current_rows: list[dict[str, object]]
    missing_bins: list[dict[str, object]]
    extra_bins: list[dict[str, object]]
    count_mismatches: list[dict[str, object]]
    is_valid: bool
    message: str


def build_rmcsetup_paths(
    project_dir_or_paths: ProjectPaths | str | Path,
) -> RMCSetupPaths:
    if isinstance(project_dir_or_paths, ProjectPaths):
        project_dir = project_dir_or_paths.project_dir
    else:
        project_dir = Path(project_dir_or_paths).expanduser().resolve()
    rmcsetup_dir = project_dir / "rmcsetup"
    return RMCSetupPaths(
        project_dir=project_dir,
        rmcsetup_dir=rmcsetup_dir,
        representative_clusters_dir=rmcsetup_dir / "representative_clusters",
        representative_selection_path=rmcsetup_dir
        / "representative_clusters"
        / "representative_selection.json",
        pdb_no_solvent_dir=rmcsetup_dir / "pdb_no_solvent",
        pdb_with_solvent_dir=rmcsetup_dir / "pdb_with_solvent",
        packmol_inputs_dir=rmcsetup_dir / "packmol_inputs",
        constraints_dir=rmcsetup_dir / "constraints",
        reports_dir=rmcsetup_dir / "reports",
        distribution_selection_path=rmcsetup_dir
        / "distribution_selection.json",
        solution_properties_path=rmcsetup_dir / "solution_properties.json",
        solvent_handling_path=rmcsetup_dir / "solvent_handling.json",
        packmol_plan_path=rmcsetup_dir / "packmol_plan.json",
        packmol_setup_path=rmcsetup_dir / "packmol_setup.json",
        constraint_generation_path=rmcsetup_dir / "constraints.json",
        packmol_plan_report_path=rmcsetup_dir
        / "reports"
        / "packmol_planning_report.txt",
        packmol_audit_report_path=rmcsetup_dir
        / "reports"
        / "packmol_audit.md",
        cluster_counts_csv_path=rmcsetup_dir
        / "reports"
        / "cluster_counts.csv",
        planned_count_weights_csv_path=rmcsetup_dir
        / "reports"
        / "planned_count_weights.csv",
        planned_atom_weights_csv_path=rmcsetup_dir
        / "reports"
        / "planned_atom_weights.csv",
        packmol_input_path=rmcsetup_dir
        / "packmol_inputs"
        / "packmol_combined.inp",
        packmol_output_pdb_path=rmcsetup_dir
        / "packmol_inputs"
        / "packed_combined.pdb",
        merged_constraints_path=rmcsetup_dir
        / "constraints"
        / "merged_fullrmc_constraints.py",
    )


def ensure_rmcsetup_structure(
    project_dir_or_paths: ProjectPaths | str | Path,
) -> RMCSetupPaths:
    paths = build_rmcsetup_paths(project_dir_or_paths)
    for directory in (
        paths.rmcsetup_dir,
        paths.representative_clusters_dir,
        paths.pdb_no_solvent_dir,
        paths.pdb_with_solvent_dir,
        paths.packmol_inputs_dir,
        paths.constraints_dir,
        paths.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _ensure_json_file(paths.representative_selection_path)
    _ensure_json_file(paths.distribution_selection_path)
    _ensure_json_file(paths.solution_properties_path)
    _ensure_json_file(paths.solvent_handling_path)
    _ensure_json_file(paths.packmol_plan_path)
    _ensure_json_file(paths.packmol_setup_path)
    _ensure_json_file(paths.constraint_generation_path)
    return paths


def validate_cluster_source(
    settings: ProjectSettings,
    *,
    project_paths: ProjectPaths | None = None,
) -> ClusterSourceValidationResult:
    expected_rows = expected_cluster_inventory_rows(
        settings,
        project_paths=project_paths,
    )
    expected_rows_for_validation = [
        dict(row) for row in expected_rows if _row_uses_cluster_directory(row)
    ]
    clusters_dir = settings.resolved_clusters_dir
    if clusters_dir is None:
        return ClusterSourceValidationResult(
            clusters_dir=None,
            expected_rows=expected_rows,
            current_rows=[],
            missing_bins=[],
            extra_bins=[],
            count_mismatches=[],
            is_valid=False,
            message="No cluster source path is saved in the SAXS project.",
        )
    if not clusters_dir.is_dir():
        return ClusterSourceValidationResult(
            clusters_dir=clusters_dir,
            expected_rows=expected_rows,
            current_rows=[],
            missing_bins=[],
            extra_bins=[],
            count_mismatches=[],
            is_valid=False,
            message=(
                "The saved cluster source path does not exist or is no "
                f"longer a directory: {clusters_dir}"
            ),
        )

    current_rows = collect_cluster_count_rows(clusters_dir)
    expected_map = {
        (str(row["structure"]), str(row["motif"])): int(row["count"])
        for row in expected_rows_for_validation
    }
    current_map = {
        (str(row["structure"]), str(row["motif"])): int(row["count"])
        for row in current_rows
    }
    missing_bins: list[dict[str, object]] = []
    extra_bins: list[dict[str, object]] = []
    count_mismatches: list[dict[str, object]] = []

    for key, expected_count in expected_map.items():
        if key not in current_map:
            missing_bins.append(
                {
                    "structure": key[0],
                    "motif": key[1],
                    "expected_count": expected_count,
                }
            )
            continue
        actual_count = current_map[key]
        if actual_count != expected_count:
            count_mismatches.append(
                {
                    "structure": key[0],
                    "motif": key[1],
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                }
            )
    for key, actual_count in current_map.items():
        if key not in expected_map:
            extra_bins.append(
                {
                    "structure": key[0],
                    "motif": key[1],
                    "actual_count": actual_count,
                }
            )

    if not expected_rows_for_validation:
        is_valid = bool(current_rows)
        message = (
            "No saved cluster inventory snapshot was available, so the "
            f"current cluster folder was only checked for readable bin counts. "
            f"Detected {len(current_rows)} non-empty bins."
        )
    else:
        is_valid = not missing_bins and not extra_bins and not count_mismatches
        if is_valid:
            message = (
                "Saved cluster source matches the project snapshot by "
                f"bin name and file count ({len(current_rows)} bins)."
            )
        else:
            message = (
                "Saved cluster source differs from the project snapshot. "
                f"Missing bins: {len(missing_bins)}, "
                f"extra bins: {len(extra_bins)}, "
                f"count mismatches: {len(count_mismatches)}."
            )

    return ClusterSourceValidationResult(
        clusters_dir=clusters_dir,
        expected_rows=expected_rows,
        current_rows=current_rows,
        missing_bins=missing_bins,
        extra_bins=extra_bins,
        count_mismatches=count_mismatches,
        is_valid=is_valid,
        message=message,
    )


def expected_cluster_inventory_rows(
    settings: ProjectSettings,
    *,
    project_paths: ProjectPaths | None = None,
) -> list[dict[str, object]]:
    if settings.cluster_inventory_rows:
        return [dict(row) for row in settings.cluster_inventory_rows]
    paths = (
        project_paths
        if project_paths is not None
        else build_project_paths(settings.project_dir)
    )
    prior_weights_path = paths.project_dir / "md_prior_weights.json"
    if not prior_weights_path.is_file():
        return []
    try:
        payload = json.loads(prior_weights_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows: list[dict[str, object]] = []
    for structure, motifs in dict(payload.get("structures", {})).items():
        if not isinstance(motifs, dict):
            continue
        for motif, details in motifs.items():
            if not isinstance(details, dict):
                continue
            row: dict[str, object] = {
                "structure": str(structure),
                "motif": str(motif),
                "count": int(details.get("count", 0)),
            }
            for key in (
                "source_kind",
                "source_dir",
                "source_file",
                "source_file_name",
                "representative",
                "profile_file",
            ):
                value = _optional_text(details.get(key))
                if value is not None:
                    row[key] = value
            rows.append(row)
    rows.sort(
        key=lambda row: (
            _natural_sort_key(str(row["structure"])),
            _natural_sort_key(str(row["motif"])),
        )
    )
    return rows


def collect_cluster_count_rows(
    clusters_dir: str | Path,
) -> list[dict[str, object]]:
    clusters_path = Path(clusters_dir).expanduser().resolve()
    if not clusters_path.is_dir():
        raise ValueError(f"Clusters directory does not exist: {clusters_path}")
    rows: list[dict[str, object]] = []
    for structure_dir in sorted(
        clusters_path.iterdir(),
        key=lambda path: _natural_sort_key(path.name),
    ):
        if not structure_dir.is_dir():
            continue
        if structure_dir.name.startswith("representative_"):
            continue

        motif_dirs = sorted(
            [
                candidate
                for candidate in structure_dir.iterdir()
                if candidate.is_dir() and candidate.name.startswith("motif_")
            ],
            key=lambda path: _natural_sort_key(path.name),
        )
        if motif_dirs:
            for motif_dir in motif_dirs:
                count = _count_structure_files_in_dir(motif_dir)
                if count <= 0:
                    continue
                rows.append(
                    {
                        "structure": structure_dir.name,
                        "motif": motif_dir.name,
                        "count": count,
                        "source_dir": str(motif_dir),
                    }
                )
            continue

        count = _count_structure_files_in_dir(structure_dir)
        if count <= 0:
            continue
        rows.append(
            {
                "structure": structure_dir.name,
                "motif": "no_motif",
                "count": count,
                "source_dir": str(structure_dir),
            }
        )
    return rows


def _count_structure_files_in_dir(directory: Path) -> int:
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _STRUCTURE_SUFFIXES
    )


def _row_uses_cluster_directory(row: dict[str, object]) -> bool:
    source_kind = _optional_text(row.get("source_kind"))
    if source_kind == "single_structure_file":
        return False
    source_dir = _optional_text(row.get("source_dir"))
    source_file = _optional_text(row.get("source_file"))
    if source_file is not None and source_dir is None:
        return False
    return True


def _ensure_json_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text("{}\n", encoding="utf-8")


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


__all__ = [
    "ClusterSourceValidationResult",
    "RMCSetupPaths",
    "build_rmcsetup_paths",
    "collect_cluster_count_rows",
    "ensure_rmcsetup_structure",
    "expected_cluster_inventory_rows",
    "validate_cluster_source",
]
