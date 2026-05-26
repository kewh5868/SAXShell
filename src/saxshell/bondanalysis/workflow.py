from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from re import sub
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from .bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
    CoordinationNumberDefinition,
)
from .results import RESULTS_INDEX_FILENAME

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ClusterTypeSummary:
    """One discovered stoichiometry-level cluster folder."""

    name: str
    path: Path
    structure_files: tuple[Path, ...]

    @property
    def structure_count(self) -> int:
        return len(self.structure_files)


@dataclass(slots=True)
class BondAnalysisClusterResult:
    """Per-cluster-type output summary."""

    cluster_type: str
    structure_count: int
    output_dir: Path
    bond_value_counts: dict[str, int]
    angle_value_counts: dict[str, int]
    coordination_value_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_type": self.cluster_type,
            "structure_count": self.structure_count,
            "output_dir": str(self.output_dir),
            "bond_value_counts": dict(self.bond_value_counts),
            "angle_value_counts": dict(self.angle_value_counts),
            "coordination_value_counts": dict(self.coordination_value_counts),
        }


@dataclass(slots=True)
class BondAnalysisBatchResult:
    """Top-level output summary for one run."""

    clusters_dir: Path
    output_dir: Path
    selected_cluster_types: tuple[str, ...]
    total_structure_files: int
    cluster_results: list[BondAnalysisClusterResult]
    results_index_path: Path

    @property
    def manifest_path(self) -> Path:
        """Backward-compatible alias for older callers."""
        return self.results_index_path

    def to_dict(self) -> dict[str, object]:
        return {
            "clusters_dir": str(self.clusters_dir),
            "output_dir": str(self.output_dir),
            "selected_cluster_types": list(self.selected_cluster_types),
            "total_structure_files": self.total_structure_files,
            "results_index_path": str(self.results_index_path),
            "cluster_results": [
                result.to_dict() for result in self.cluster_results
            ],
        }


def next_available_output_dir(parent_dir: Path, folder_name: str) -> Path:
    """Return the next available output directory beside the source."""
    candidate = parent_dir / folder_name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = parent_dir / f"{folder_name}{index:04d}"
        if not candidate.exists():
            return candidate
        index += 1


def suggest_bondanalysis_output_dir(clusters_dir: str | Path) -> Path:
    """Suggest a sibling directory for bond-analysis output."""
    source_path = Path(clusters_dir)
    folder_name = _base_output_dir_name(source_path)
    return next_available_output_dir(source_path.parent, folder_name)


def _base_output_dir_name(clusters_dir: Path) -> str:
    folder_label = sub(r"[^0-9A-Za-z]+", "_", clusters_dir.name).strip("_")
    if not folder_label:
        folder_label = "clusters"
    return f"bondanalysis_{folder_label}"


def discover_cluster_types(
    clusters_dir: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[ClusterTypeSummary]:
    """Discover stoichiometry-level cluster folders.

    The expected layout is a root directory containing one folder per
    cluster type, with structure files directly inside each of those
    folders. If the selected directory itself contains structure files
    directly, it is treated as one single cluster type.
    """

    source_path = Path(clusters_dir)
    if not source_path.is_dir():
        raise ValueError(f"Clusters directory does not exist: {source_path}")

    analyzer = BondAnalyzer()
    if progress_callback is not None:
        progress_callback(
            0,
            1,
            f"Scanning selected clusters directory: {source_path}",
        )
    direct_files = tuple(analyzer.structure_files(source_path))
    if direct_files:
        if progress_callback is not None:
            progress_callback(
                1,
                1,
                f"Detected {len(direct_files)} structure file(s) directly.",
            )
        return [
            ClusterTypeSummary(
                name=source_path.name,
                path=source_path,
                structure_files=direct_files,
            )
        ]

    summaries: list[ClusterTypeSummary] = []
    child_dirs = [
        child for child in sorted(source_path.iterdir()) if child.is_dir()
    ]
    total_children = max(len(child_dirs), 1)
    if progress_callback is not None:
        progress_callback(
            0,
            total_children,
            f"Inspecting {len(child_dirs)} cluster folder(s).",
        )
    for index, child in enumerate(child_dirs, start=1):
        if progress_callback is not None:
            progress_callback(
                index,
                total_children,
                (
                    f"Inspecting cluster folder {child.name} "
                    f"({index}/{total_children})."
                ),
            )
        structure_files = tuple(analyzer.structure_files(child))
        if not structure_files:
            continue
        summaries.append(
            ClusterTypeSummary(
                name=child.name,
                path=child,
                structure_files=structure_files,
            )
        )
    if progress_callback is not None:
        progress_callback(
            total_children,
            total_children,
            f"Discovered {len(summaries)} cluster type(s).",
        )
    return summaries


class BondAnalysisWorkflow:
    """Shared workflow used by the UI, CLI, and notebook entry
    points."""

    def __init__(
        self,
        clusters_dir: str | Path,
        *,
        bond_pairs: Iterable[BondPairDefinition] | None = None,
        angle_triplets: Iterable[AngleTripletDefinition] | None = None,
        coordination_numbers: (
            Iterable[CoordinationNumberDefinition] | None
        ) = None,
        output_dir: str | Path | None = None,
        selected_cluster_types: Sequence[str] | None = None,
        structure_distribution_store_dir: str | Path | None = None,
        generate_preview_plots: bool = True,
    ) -> None:
        self.clusters_dir = Path(clusters_dir)
        self.bond_pairs = tuple(dict.fromkeys(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))
        self.coordination_numbers = tuple(
            dict.fromkeys(coordination_numbers or ())
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.structure_distribution_store_dir = (
            None
            if structure_distribution_store_dir is None
            else Path(structure_distribution_store_dir)
        )
        self.generate_preview_plots = bool(generate_preview_plots)
        self.selected_cluster_types = (
            tuple(selected_cluster_types)
            if selected_cluster_types is not None
            else None
        )

    def inspect(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        cluster_types = discover_cluster_types(
            self.clusters_dir,
            progress_callback=progress_callback,
        )
        return {
            "clusters_dir": str(self.clusters_dir),
            "cluster_types": [summary.name for summary in cluster_types],
            "cluster_type_count": len(cluster_types),
            "total_structure_files": sum(
                summary.structure_count for summary in cluster_types
            ),
            "suggested_output_dir": str(
                self.output_dir
                if self.output_dir is not None
                else suggest_bondanalysis_output_dir(self.clusters_dir)
            ),
        }

    def run(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> BondAnalysisBatchResult:
        if (
            not self.bond_pairs
            and not self.angle_triplets
            and not self.coordination_numbers
        ):
            raise ValueError(
                "Provide at least one bond pair, angle triplet, or "
                "coordination-number definition."
            )

        cluster_summaries = self._selected_cluster_summaries()
        if not cluster_summaries:
            raise ValueError("No cluster types were selected for analysis.")

        output_dir = (
            self.output_dir
            if self.output_dir is not None
            else suggest_bondanalysis_output_dir(self.clusters_dir)
        )
        from saxshell.structure_distributions import StructureDistributionStore

        store_dir = (
            self.structure_distribution_store_dir
            if self.structure_distribution_store_dir is not None
            else output_dir / "structure_distribution_store"
        )
        distribution_store = StructureDistributionStore(store_dir)
        cluster_root = output_dir / "cluster_types"
        aggregate_root = output_dir / "all_clusters"
        comparison_root = output_dir / "comparisons"
        cluster_root.mkdir(parents=True, exist_ok=True)
        aggregate_root.mkdir(parents=True, exist_ok=True)
        comparison_root.mkdir(parents=True, exist_ok=True)

        total_files = sum(
            summary.structure_count for summary in cluster_summaries
        )
        post_processing_steps = len(cluster_summaries) + 6
        total_work_units = max(total_files + post_processing_steps, 1)
        completed_units = 0
        if progress_callback is not None:
            progress_callback(
                0,
                total_work_units,
                "Preparing bond analysis.",
            )

        aggregate_bond_rows = {
            definition: [] for definition in self.bond_pairs
        }
        aggregate_angle_rows = {
            definition: [] for definition in self.angle_triplets
        }
        aggregate_coordination_rows = {
            definition: [] for definition in self.coordination_numbers
        }
        comparison_bonds = {definition: {} for definition in self.bond_pairs}
        comparison_angles = {
            definition: {} for definition in self.angle_triplets
        }
        comparison_coordination = {
            definition: {} for definition in self.coordination_numbers
        }
        cluster_results: list[BondAnalysisClusterResult] = []
        cache_hit_count = 0
        cache_miss_count = 0

        for summary in cluster_summaries:
            if log_callback is not None:
                log_callback(
                    "Analyzing cluster type "
                    f"{summary.name} ({summary.structure_count} files)."
                )
            cluster_output_dir = cluster_root / summary.name
            cluster_output_dir.mkdir(parents=True, exist_ok=True)

            cluster_bond_rows = {
                definition: [] for definition in self.bond_pairs
            }
            cluster_angle_rows = {
                definition: [] for definition in self.angle_triplets
            }
            cluster_coordination_rows = {
                definition: [] for definition in self.coordination_numbers
            }

            for structure_file in summary.structure_files:
                measurement = distribution_store.measure_structure_file(
                    structure_file,
                    bond_pairs=self.bond_pairs,
                    angle_triplets=self.angle_triplets,
                    coordination_numbers=self.coordination_numbers,
                    cluster_label=summary.name,
                    relative_label=structure_file.name,
                    autosave=False,
                )
                if measurement.from_cache:
                    cache_hit_count += 1
                else:
                    cache_miss_count += 1
                bond_values = measurement.bond_values
                angle_values = measurement.angle_values
                coordination_values = measurement.coordination_values
                for definition, values in bond_values.items():
                    cluster_bond_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_bond_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                for definition, values in coordination_values.items():
                    cluster_coordination_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_coordination_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                for definition, values in angle_values.items():
                    cluster_angle_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_angle_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )

                completed_units += 1
                if progress_callback is not None:
                    progress_callback(
                        completed_units,
                        total_work_units,
                        (
                            f"Processed {summary.name}/"
                            f"{structure_file.name}"
                        ),
                    )

            if progress_callback is not None:
                progress_callback(
                    completed_units,
                    total_work_units,
                    f"Writing distributions for {summary.name}.",
                )
            cluster_bond_counts = self._write_bond_outputs(
                cluster_output_dir,
                cluster_bond_rows,
                title_prefix=summary.name,
            )
            cluster_angle_counts = self._write_angle_outputs(
                cluster_output_dir,
                cluster_angle_rows,
                title_prefix=summary.name,
            )
            cluster_coordination_counts = self._write_coordination_outputs(
                cluster_output_dir,
                cluster_coordination_rows,
                title_prefix=summary.name,
            )
            completed_units += 1
            if progress_callback is not None:
                progress_callback(
                    completed_units,
                    total_work_units,
                    f"Wrote distributions for {summary.name}.",
                )

            for definition, rows in cluster_bond_rows.items():
                comparison_bonds[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_angle_rows.items():
                comparison_angles[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_coordination_rows.items():
                comparison_coordination[definition][summary.name] = [
                    row[2] for row in rows
                ]

            cluster_results.append(
                BondAnalysisClusterResult(
                    cluster_type=summary.name,
                    structure_count=summary.structure_count,
                    output_dir=cluster_output_dir,
                    bond_value_counts=cluster_bond_counts,
                    angle_value_counts=cluster_angle_counts,
                    coordination_value_counts=cluster_coordination_counts,
                )
            )

        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Saving cached structure measurements.",
            )
        distribution_store.flush()
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing all-cluster bond distributions.",
            )

        self._write_bond_outputs(
            aggregate_root,
            aggregate_bond_rows,
            title_prefix="All selected clusters",
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing all-cluster angle distributions.",
            )
        self._write_angle_outputs(
            aggregate_root,
            aggregate_angle_rows,
            title_prefix="All selected clusters",
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing all-cluster coordination distributions.",
            )
        self._write_coordination_outputs(
            aggregate_root,
            aggregate_coordination_rows,
            title_prefix="All selected clusters",
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing cluster comparison overlays.",
            )
        self._write_comparison_bond_outputs(comparison_root, comparison_bonds)
        self._write_comparison_angle_outputs(
            comparison_root,
            comparison_angles,
        )
        self._write_comparison_coordination_outputs(
            comparison_root,
            comparison_coordination,
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Writing bond-analysis results index.",
            )

        results_index_path = output_dir / RESULTS_INDEX_FILENAME
        results_index_path.write_text(
            json.dumps(
                {
                    "clusters_dir": str(self.clusters_dir),
                    "output_dir": str(output_dir),
                    "selected_cluster_types": [
                        summary.name for summary in cluster_summaries
                    ],
                    "total_structure_files": total_files,
                    "bond_pairs": [
                        definition.to_dict() for definition in self.bond_pairs
                    ],
                    "angle_triplets": [
                        definition.to_dict()
                        for definition in self.angle_triplets
                    ],
                    "coordination_numbers": [
                        definition.to_dict()
                        for definition in self.coordination_numbers
                    ],
                    "cluster_results": [
                        result.to_dict() for result in cluster_results
                    ],
                    "aggregate_output_dir": str(aggregate_root),
                    "comparison_output_dir": str(comparison_root),
                    "structure_distribution_store_dir": str(
                        distribution_store.root_dir
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        completed_units += 1
        if progress_callback is not None:
            progress_callback(
                completed_units,
                total_work_units,
                "Bond analysis complete.",
            )

        if log_callback is not None:
            log_callback(
                "Wrote bond-analysis results index to "
                f"{results_index_path}."
            )
            log_callback(
                "Structure distribution cache: "
                f"{cache_hit_count} hit(s), {cache_miss_count} miss(es) at "
                f"{distribution_store.root_dir}."
            )

        return BondAnalysisBatchResult(
            clusters_dir=self.clusters_dir,
            output_dir=output_dir,
            selected_cluster_types=tuple(
                summary.name for summary in cluster_summaries
            ),
            total_structure_files=total_files,
            cluster_results=cluster_results,
            results_index_path=results_index_path,
        )

    def _selected_cluster_summaries(self) -> list[ClusterTypeSummary]:
        summaries = discover_cluster_types(self.clusters_dir)
        if self.selected_cluster_types is None:
            return summaries

        selected_names = set(self.selected_cluster_types)
        selected = [
            summary for summary in summaries if summary.name in selected_names
        ]
        missing = selected_names.difference(
            summary.name for summary in selected
        )
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Unknown cluster type selection: " f"{missing_text}"
            )
        return selected

    def _write_bond_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            BondPairDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = (
                output_dir / f"{definition.filename_stem}_distribution.csv"
            )
            npy_path = (
                output_dir / f"{definition.filename_stem}_distribution.npy"
            )
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Distance (A)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="bond",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Distance (A)",
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "bond distribution"
                        ),
                        xlabel="Distance (A)",
                        png_path=png_path,
                    )
        return counts

    def _write_angle_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            AngleTripletDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = output_dir / f"{definition.filename_stem}_angles.csv"
            npy_path = output_dir / f"{definition.filename_stem}_angles.npy"
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Angle (deg)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="angle",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Angle (deg)",
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "angle distribution"
                        ),
                        xlabel="Angle (deg)",
                        png_path=png_path,
                    )
        return counts

    def _write_coordination_outputs(
        self,
        output_dir: Path,
        rows_by_definition: dict[
            CoordinationNumberDefinition,
            list[tuple[str, str, float]],
        ],
        *,
        title_prefix: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for definition, rows in rows_by_definition.items():
            csv_path = (
                output_dir / f"{definition.filename_stem}_coordination.csv"
            )
            npy_path = (
                output_dir / f"{definition.filename_stem}_coordination.npy"
            )
            self._write_distribution_csv(
                csv_path,
                rows,
                header=(
                    "Cluster Type",
                    "Structure File",
                    "Coordination Number",
                ),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                histogram_csv_path = (
                    output_dir / f"{definition.filename_stem}_histogram.csv"
                )
                self._write_histogram_csv(
                    histogram_csv_path,
                    values,
                    distribution_type="coordination",
                    distribution_label=definition.display_label,
                    scope_label=title_prefix,
                    value_label="Coordination Number",
                    metadata={
                        "center_atom": definition.center_atom,
                        "atom_of_interest": definition.neighbor_atom,
                        "cutoff_angstrom": definition.cutoff_angstrom,
                    },
                    bins=self._integer_histogram_edges(values),
                )
                if self.generate_preview_plots:
                    png_path = (
                        output_dir
                        / f"{definition.filename_stem}_histogram.png"
                    )
                    self._save_histogram(
                        values,
                        title=(
                            f"{title_prefix} • {definition.display_label} "
                            "coordination-number distribution"
                        ),
                        xlabel="Coordination Number",
                        png_path=png_path,
                    )
        return counts

    def _write_comparison_bond_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            BondPairDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} bond distribution"
                    ),
                    xlabel="Distance (A)",
                    png_path=png_path,
                )

    def _write_comparison_coordination_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            CoordinationNumberDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} "
                        "coordination-number distribution"
                    ),
                    xlabel="Coordination Number",
                    png_path=png_path,
                )

    def _write_comparison_angle_outputs(
        self,
        output_dir: Path,
        values_by_definition: dict[
            AngleTripletDefinition,
            dict[str, list[float]],
        ],
    ) -> None:
        for definition, values_by_cluster in values_by_definition.items():
            if not any(values_by_cluster.values()):
                continue
            csv_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.csv"
            )
            npy_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.npy"
            )
            png_path = (
                output_dir
                / f"{definition.filename_stem}_cluster_type_overlay.png"
            )
            self._write_overlay_csv(csv_path, values_by_cluster)
            self._write_overlay_npy(npy_path, values_by_cluster)
            if self.generate_preview_plots:
                self._save_overlay_histogram(
                    values_by_cluster,
                    title=(
                        "Cluster-type comparison • "
                        f"{definition.display_label} angle distribution"
                    ),
                    xlabel="Angle (deg)",
                    png_path=png_path,
                )

    @staticmethod
    def _write_distribution_csv(
        csv_path: Path,
        rows: list[tuple[str, str, float]],
        *,
        header: tuple[str, str, str],
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for cluster_type, structure_file, value in rows:
                writer.writerow([cluster_type, structure_file, f"{value:.6f}"])

    @staticmethod
    def _write_distribution_npy(
        npy_path: Path,
        rows: list[tuple[str, str, float]],
    ) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_width = max((len(row[0]) for row in rows), default=1)
        structure_width = max((len(row[1]) for row in rows), default=1)
        payload = np.empty(
            len(rows),
            dtype=[
                ("cluster_type", f"U{cluster_width}"),
                ("structure_file", f"U{structure_width}"),
                ("value", np.float64),
            ],
        )
        for index, (cluster_type, structure_file, value) in enumerate(rows):
            payload[index] = (cluster_type, structure_file, float(value))
        np.save(npy_path, payload)

    @classmethod
    def _write_histogram_csv(
        cls,
        csv_path: Path,
        values: list[float],
        *,
        distribution_type: str,
        distribution_label: str,
        scope_label: str,
        value_label: str,
        metadata: Mapping[str, object] | None = None,
        bins: int | Sequence[float] = 60,
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        numeric_values = np.asarray(values, dtype=float)
        counts, edges = np.histogram(numeric_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]
        densities = np.divide(
            counts,
            widths * max(float(numeric_values.size), 1.0),
            out=np.zeros_like(centers, dtype=float),
            where=widths > 0,
        )
        stats = cls._distribution_statistics(
            numeric_values,
            counts=counts,
            centers=centers,
        )
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["# SAXSShell bondanalysis histogram distribution"]
            )
            for key, value in (
                ("distribution_type", distribution_type),
                ("distribution_label", distribution_label),
                ("scope", scope_label),
                ("value_label", value_label),
                *((metadata or {}).items()),
                ("histogram_bins", int(counts.size)),
                ("point_count", int(numeric_values.size)),
                ("mean", stats["mean"]),
                ("median", stats["median"]),
                ("mode", stats["mode"]),
                ("sigma", stats["sigma"]),
                ("standard_deviation", stats["standard_deviation"]),
                ("sample_sigma", stats["sample_sigma"]),
                ("variance", stats["variance"]),
                ("minimum", stats["minimum"]),
                ("maximum", stats["maximum"]),
                ("q1", stats["q1"]),
                ("q3", stats["q3"]),
            ):
                writer.writerow([f"# {key}", value])
            writer.writerow(
                (
                    "bin_left",
                    "bin_right",
                    "bin_center",
                    "count",
                    "density",
                )
            )
            for left, right, center, count, density in zip(
                edges[:-1],
                edges[1:],
                centers,
                counts,
                densities,
            ):
                writer.writerow(
                    (
                        f"{float(left):.8g}",
                        f"{float(right):.8g}",
                        f"{float(center):.8g}",
                        int(count),
                        f"{float(density):.8g}",
                    )
                )

    @staticmethod
    def _distribution_statistics(
        values: np.ndarray,
        *,
        counts: np.ndarray,
        centers: np.ndarray,
    ) -> dict[str, float]:
        if values.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "sigma": 0.0,
                "standard_deviation": 0.0,
                "sample_sigma": 0.0,
                "variance": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "q1": 0.0,
                "q3": 0.0,
            }
        if np.all(np.isclose(values, np.round(values))):
            unique_values, unique_counts = np.unique(
                values, return_counts=True
            )
            peak_count = int(np.max(unique_counts))
            mode_value = float(unique_values[unique_counts == peak_count][0])
        else:
            peak_index = int(np.argmax(counts)) if counts.size else 0
            mode_value = (
                float(centers[peak_index])
                if centers.size
                else float(np.median(values))
            )
        sigma = float(np.std(values, ddof=0))
        sample_sigma = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "mode": mode_value,
            "sigma": sigma,
            "standard_deviation": sigma,
            "sample_sigma": sample_sigma,
            "variance": float(np.var(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
        }

    @staticmethod
    def _integer_histogram_edges(values: list[float]) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.size == 0:
            return np.asarray([0.0, 1.0], dtype=float)
        left = math.floor(float(np.min(numeric_values))) - 0.5
        right = math.ceil(float(np.max(numeric_values))) + 0.5
        edges = np.arange(left, right + 1.0, 1.0, dtype=float)
        if edges.size < 2:
            return np.asarray([left, right], dtype=float)
        return edges

    @staticmethod
    def _write_overlay_csv(
        csv_path: Path,
        values_by_cluster: dict[str, list[float]],
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Cluster Type", "Value"))
            for cluster_type, values in sorted(values_by_cluster.items()):
                for value in values:
                    writer.writerow([cluster_type, f"{value:.6f}"])

    @staticmethod
    def _write_overlay_npy(
        npy_path: Path,
        values_by_cluster: dict[str, list[float]],
    ) -> None:
        rows = [
            (cluster_type, float(value))
            for cluster_type, values in sorted(values_by_cluster.items())
            for value in values
        ]
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_width = max((len(row[0]) for row in rows), default=1)
        payload = np.empty(
            len(rows),
            dtype=[
                ("cluster_type", f"U{cluster_width}"),
                ("value", np.float64),
            ],
        )
        for index, (cluster_type, value) in enumerate(rows):
            payload[index] = (cluster_type, value)
        np.save(npy_path, payload)

    @staticmethod
    def _save_histogram(
        values: list[float],
        *,
        title: str,
        xlabel: str,
        png_path: Path,
    ) -> None:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        axis.hist(values, bins=60, color="#355070", edgecolor="white")
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        figure.clear()

    @staticmethod
    def _save_overlay_histogram(
        values_by_cluster: dict[str, list[float]],
        *,
        title: str,
        xlabel: str,
        png_path: Path,
    ) -> None:
        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        for cluster_type, values in sorted(values_by_cluster.items()):
            if not values:
                continue
            axis.hist(
                values,
                bins=60,
                histtype="step",
                linewidth=1.5,
                label=cluster_type,
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(png_path, dpi=200)
        figure.clear()


__all__ = [
    "BondAnalysisBatchResult",
    "BondAnalysisClusterResult",
    "BondAnalysisWorkflow",
    "ClusterTypeSummary",
    "discover_cluster_types",
    "next_available_output_dir",
    "suggest_bondanalysis_output_dir",
]
