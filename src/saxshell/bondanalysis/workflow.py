from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from re import sub
from typing import Callable, Iterable, Sequence

import numpy as np
from matplotlib.figure import Figure

from .bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
)

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

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_type": self.cluster_type,
            "structure_count": self.structure_count,
            "output_dir": str(self.output_dir),
            "bond_value_counts": dict(self.bond_value_counts),
            "angle_value_counts": dict(self.angle_value_counts),
        }


@dataclass(slots=True)
class BondAnalysisBatchResult:
    """Top-level output summary for one run."""

    clusters_dir: Path
    output_dir: Path
    selected_cluster_types: tuple[str, ...]
    total_structure_files: int
    cluster_results: list[BondAnalysisClusterResult]
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "clusters_dir": str(self.clusters_dir),
            "output_dir": str(self.output_dir),
            "selected_cluster_types": list(self.selected_cluster_types),
            "total_structure_files": self.total_structure_files,
            "manifest_path": str(self.manifest_path),
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


def discover_cluster_types(clusters_dir: str | Path) -> list[ClusterTypeSummary]:
    """Discover stoichiometry-level cluster folders.

    The expected layout is a root directory containing one folder per cluster
    type, with structure files directly inside each of those folders. If the
    selected directory itself contains structure files directly, it is treated
    as one single cluster type.
    """

    source_path = Path(clusters_dir)
    if not source_path.is_dir():
        raise ValueError(f"Clusters directory does not exist: {source_path}")

    analyzer = BondAnalyzer()
    direct_files = tuple(analyzer.structure_files(source_path))
    if direct_files:
        return [
            ClusterTypeSummary(
                name=source_path.name,
                path=source_path,
                structure_files=direct_files,
            )
        ]

    summaries: list[ClusterTypeSummary] = []
    for child in sorted(source_path.iterdir()):
        if not child.is_dir():
            continue
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
    return summaries


class BondAnalysisWorkflow:
    """Shared workflow used by the UI, CLI, and notebook entry points."""

    def __init__(
        self,
        clusters_dir: str | Path,
        *,
        bond_pairs: Iterable[BondPairDefinition] | None = None,
        angle_triplets: Iterable[AngleTripletDefinition] | None = None,
        output_dir: str | Path | None = None,
        selected_cluster_types: Sequence[str] | None = None,
    ) -> None:
        self.clusters_dir = Path(clusters_dir)
        self.bond_pairs = tuple(dict.fromkeys(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))
        self.output_dir = (
            Path(output_dir) if output_dir is not None else None
        )
        self.selected_cluster_types = (
            tuple(selected_cluster_types)
            if selected_cluster_types is not None
            else None
        )

    def inspect(self) -> dict[str, object]:
        cluster_types = discover_cluster_types(self.clusters_dir)
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
        if not self.bond_pairs and not self.angle_triplets:
            raise ValueError(
                "Provide at least one bond pair or one angle triplet."
            )

        cluster_summaries = self._selected_cluster_summaries()
        if not cluster_summaries:
            raise ValueError("No cluster types were selected for analysis.")

        analyzer = BondAnalyzer(
            bond_pairs=self.bond_pairs,
            angle_triplets=self.angle_triplets,
        )
        output_dir = (
            self.output_dir
            if self.output_dir is not None
            else suggest_bondanalysis_output_dir(self.clusters_dir)
        )
        cluster_root = output_dir / "cluster_types"
        aggregate_root = output_dir / "all_clusters"
        comparison_root = output_dir / "comparisons"
        cluster_root.mkdir(parents=True, exist_ok=True)
        aggregate_root.mkdir(parents=True, exist_ok=True)
        comparison_root.mkdir(parents=True, exist_ok=True)

        total_files = sum(
            summary.structure_count for summary in cluster_summaries
        )
        processed_files = 0
        if progress_callback is not None:
            progress_callback(0, total_files, "Preparing bond analysis.")

        aggregate_bond_rows = {
            definition: [] for definition in self.bond_pairs
        }
        aggregate_angle_rows = {
            definition: [] for definition in self.angle_triplets
        }
        comparison_bonds = {
            definition: {} for definition in self.bond_pairs
        }
        comparison_angles = {
            definition: {} for definition in self.angle_triplets
        }
        cluster_results: list[BondAnalysisClusterResult] = []

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

            for structure_file in summary.structure_files:
                bond_values, angle_values = analyzer.measure_structure(
                    structure_file
                )
                for definition, values in bond_values.items():
                    cluster_bond_rows[definition].extend(
                        (summary.name, structure_file.name, value)
                        for value in values
                    )
                    aggregate_bond_rows[definition].extend(
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

                processed_files += 1
                if progress_callback is not None:
                    progress_callback(
                        processed_files,
                        total_files,
                        (
                            f"Processed {summary.name}/"
                            f"{structure_file.name}"
                        ),
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

            for definition, rows in cluster_bond_rows.items():
                comparison_bonds[definition][summary.name] = [
                    row[2] for row in rows
                ]
            for definition, rows in cluster_angle_rows.items():
                comparison_angles[definition][summary.name] = [
                    row[2] for row in rows
                ]

            cluster_results.append(
                BondAnalysisClusterResult(
                    cluster_type=summary.name,
                    structure_count=summary.structure_count,
                    output_dir=cluster_output_dir,
                    bond_value_counts=cluster_bond_counts,
                    angle_value_counts=cluster_angle_counts,
                )
            )

        self._write_bond_outputs(
            aggregate_root,
            aggregate_bond_rows,
            title_prefix="All selected clusters",
        )
        self._write_angle_outputs(
            aggregate_root,
            aggregate_angle_rows,
            title_prefix="All selected clusters",
        )
        self._write_comparison_bond_outputs(comparison_root, comparison_bonds)
        self._write_comparison_angle_outputs(
            comparison_root,
            comparison_angles,
        )

        manifest_path = output_dir / "bondanalysis_manifest.json"
        manifest_path.write_text(
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
                    "cluster_results": [
                        result.to_dict() for result in cluster_results
                    ],
                    "aggregate_output_dir": str(aggregate_root),
                    "comparison_output_dir": str(comparison_root),
                },
                indent=2,
            )
            + "\n"
        )

        if log_callback is not None:
            log_callback(f"Wrote bond-analysis manifest to {manifest_path}.")

        return BondAnalysisBatchResult(
            clusters_dir=self.clusters_dir,
            output_dir=output_dir,
            selected_cluster_types=tuple(
                summary.name for summary in cluster_summaries
            ),
            total_structure_files=total_files,
            cluster_results=cluster_results,
            manifest_path=manifest_path,
        )

    def _selected_cluster_summaries(self) -> list[ClusterTypeSummary]:
        summaries = discover_cluster_types(self.clusters_dir)
        if self.selected_cluster_types is None:
            return summaries

        selected_names = set(self.selected_cluster_types)
        selected = [
            summary
            for summary in summaries
            if summary.name in selected_names
        ]
        missing = selected_names.difference(summary.name for summary in selected)
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
            csv_path = output_dir / f"{definition.filename_stem}_distribution.csv"
            npy_path = output_dir / f"{definition.filename_stem}_distribution.npy"
            self._write_distribution_csv(
                csv_path,
                rows,
                header=("Cluster Type", "Structure File", "Distance (A)"),
            )
            self._write_distribution_npy(npy_path, rows)
            values = [row[2] for row in rows]
            counts[definition.display_label] = len(values)
            if values:
                png_path = output_dir / f"{definition.filename_stem}_histogram.png"
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
                png_path = output_dir / f"{definition.filename_stem}_histogram.png"
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
            self._save_overlay_histogram(
                values_by_cluster,
                title=(
                    "Cluster-type comparison • "
                    f"{definition.display_label} bond distribution"
                ),
                xlabel="Distance (A)",
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
                writer.writerow(
                    [cluster_type, structure_file, f"{value:.6f}"]
                )

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


__all__ = [
    "BondAnalysisBatchResult",
    "BondAnalysisClusterResult",
    "BondAnalysisWorkflow",
    "ClusterTypeSummary",
    "discover_cluster_types",
    "next_available_output_dir",
    "suggest_bondanalysis_output_dir",
]
