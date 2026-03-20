from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.axes import Axes

from .bondanalyzer import AngleTripletDefinition, BondPairDefinition

DistributionCategory = Literal["bond", "angle"]
RESULTS_INDEX_FILENAME = "bondanalysis_results_index.json"
LEGACY_RESULTS_INDEX_FILENAME = "bondanalysis_manifest.json"


@dataclass(frozen=True, slots=True)
class BondAnalysisResultLeaf:
    """One browsable distribution leaf shown in the UI tree."""

    category: DistributionCategory
    display_label: str
    scope_name: str
    npy_path: Path | None
    point_count: int
    is_all: bool = False


@dataclass(frozen=True, slots=True)
class BondAnalysisResultGroup:
    """All cluster-level leaves for one bond pair or angle triplet."""

    category: DistributionCategory
    display_label: str
    xlabel: str
    cluster_leaves: tuple[BondAnalysisResultLeaf, ...]
    all_leaf: BondAnalysisResultLeaf


@dataclass(frozen=True, slots=True)
class BondAnalysisDistributionSeries:
    """One plotted histogram series."""

    label: str
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class BondAnalysisPlotRequest:
    """Resolved plotting payload for one UI selection."""

    category: DistributionCategory
    display_label: str
    title: str
    xlabel: str
    series: tuple[BondAnalysisDistributionSeries, ...]


@dataclass(frozen=True, slots=True)
class BondAnalysisResultIndex:
    """Results-index-backed tree of computed bondanalysis
    distributions."""

    results_index_path: Path
    output_dir: Path
    clusters_dir: Path
    selected_cluster_types: tuple[str, ...]
    cluster_type_names: tuple[str, ...]
    bond_pairs: tuple[BondPairDefinition, ...]
    angle_triplets: tuple[AngleTripletDefinition, ...]
    bond_groups: tuple[BondAnalysisResultGroup, ...]
    angle_groups: tuple[BondAnalysisResultGroup, ...]

    @property
    def manifest_path(self) -> Path:
        """Backward-compatible alias for older callers."""
        return self.results_index_path

    def find_group(
        self,
        category: DistributionCategory,
        display_label: str,
    ) -> BondAnalysisResultGroup:
        groups = self.bond_groups if category == "bond" else self.angle_groups
        for group in groups:
            if group.display_label == display_label:
                return group
        raise KeyError(
            f"Unknown {category} distribution label: {display_label}"
        )


def draw_plot_request(
    axis: Axes,
    request: BondAnalysisPlotRequest,
) -> int:
    """Render one bondanalysis plot request onto a Matplotlib axis."""

    axis.clear()
    non_empty_series = [
        series for series in request.series if series.values.size > 0
    ]
    if not non_empty_series:
        axis.text(
            0.5,
            0.5,
            "No computed values were found for this selection.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    elif len(non_empty_series) == 1:
        axis.hist(
            non_empty_series[0].values,
            bins=60,
            color="#355070",
            edgecolor="white",
        )
    else:
        for series in non_empty_series:
            axis.hist(
                series.values,
                bins=60,
                histtype="step",
                linewidth=1.7,
                label=series.label,
            )
        axis.legend(frameon=False)

    axis.set_title(request.title)
    axis.set_xlabel(request.xlabel)
    axis.set_ylabel("Count")
    return len(non_empty_series)


def load_result_index(output_dir: str | Path) -> BondAnalysisResultIndex:
    """Load the computed result tree from a bondanalysis results
    index."""

    output_path = Path(output_dir)
    results_index_path = output_path / RESULTS_INDEX_FILENAME
    if not results_index_path.exists():
        results_index_path = output_path / LEGACY_RESULTS_INDEX_FILENAME
    if not results_index_path.exists():
        raise ValueError(
            "No bondanalysis results index file was found in the selected "
            f"output directory: {output_path}"
        )

    payload = json.loads(results_index_path.read_text())
    cluster_results = tuple(payload.get("cluster_results", []))
    bond_pairs = tuple(
        BondPairDefinition(**entry) for entry in payload.get("bond_pairs", [])
    )
    angle_triplets = tuple(
        AngleTripletDefinition(**entry)
        for entry in payload.get("angle_triplets", [])
    )
    bond_groups = tuple(
        _build_bond_group(definition, cluster_results)
        for definition in bond_pairs
    )
    angle_groups = tuple(
        _build_angle_group(definition, cluster_results)
        for definition in angle_triplets
    )
    return BondAnalysisResultIndex(
        results_index_path=results_index_path,
        output_dir=output_path,
        clusters_dir=Path(str(payload.get("clusters_dir", output_path))),
        selected_cluster_types=tuple(
            payload.get("selected_cluster_types", [])
        ),
        cluster_type_names=tuple(
            str(entry.get("cluster_type", ""))
            for entry in cluster_results
            if str(entry.get("cluster_type", "")).strip()
        ),
        bond_pairs=bond_pairs,
        angle_triplets=angle_triplets,
        bond_groups=bond_groups,
        angle_groups=angle_groups,
    )


def build_plot_request(
    result_index: BondAnalysisResultIndex,
    leaves: list[BondAnalysisResultLeaf] | tuple[BondAnalysisResultLeaf, ...],
) -> BondAnalysisPlotRequest:
    """Convert one or more selected tree leaves into plotted series."""

    selected_leaves = tuple(leaves)
    if not selected_leaves:
        raise ValueError("Select at least one computed distribution to plot.")

    first_leaf = selected_leaves[0]
    if any(
        leaf.category != first_leaf.category
        or leaf.display_label != first_leaf.display_label
        for leaf in selected_leaves[1:]
    ):
        raise ValueError(
            "Select bond pairs or bond angles of the same type before "
            "plotting them together."
        )

    if len(selected_leaves) > 1 and any(
        leaf.is_all for leaf in selected_leaves
    ):
        raise ValueError(
            "Select either the 'all' entry or one or more individual "
            "cluster distributions, but not both together."
        )

    group = result_index.find_group(
        first_leaf.category,
        first_leaf.display_label,
    )
    if len(selected_leaves) == 1 and first_leaf.is_all:
        cluster_values = [
            _load_distribution_values(leaf) for leaf in group.cluster_leaves
        ]
        non_empty_cluster_values = [
            values for values in cluster_values if values.size > 0
        ]
        merged_values = np.concatenate(
            non_empty_cluster_values or [np.array([], dtype=float)]
        )
        series = (
            BondAnalysisDistributionSeries(
                label="all",
                values=merged_values,
            ),
        )
        title = f"{group.display_label} " f"across all cluster types"
    elif len(selected_leaves) == 1:
        source_leaves = selected_leaves
        title = f"{selected_leaves[0].scope_name} • {group.display_label}"
        series = tuple(
            BondAnalysisDistributionSeries(
                label=leaf.scope_name,
                values=_load_distribution_values(leaf),
            )
            for leaf in source_leaves
        )
    else:
        source_leaves = selected_leaves
        title = f"{group.display_label} across selected cluster types"
        series = tuple(
            BondAnalysisDistributionSeries(
                label=leaf.scope_name,
                values=_load_distribution_values(leaf),
            )
            for leaf in source_leaves
        )
    return BondAnalysisPlotRequest(
        category=group.category,
        display_label=group.display_label,
        title=title,
        xlabel=group.xlabel,
        series=series,
    )


def export_plot_request_csv(
    request: BondAnalysisPlotRequest,
    output_path: str | Path,
) -> Path:
    """Save one plotted distribution or overlay request as CSV."""

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Series", "Value"))
        for series in request.series:
            for value in series.values:
                writer.writerow((series.label, f"{float(value):.6f}"))
    return csv_path


def recommended_plot_request_filename(request: BondAnalysisPlotRequest) -> str:
    """Return a user-friendly default CSV name for one plot request."""

    label_text = _slugify_filename_fragment(request.display_label)
    if len(request.series) == 1:
        scope_text = _slugify_filename_fragment(request.series[0].label)
    else:
        scope_text = "overlay"
    return f"{request.category}_{label_text}_{scope_text}.csv"


def _build_bond_group(
    definition: BondPairDefinition,
    cluster_results: tuple[dict[str, object], ...],
) -> BondAnalysisResultGroup:
    cluster_leaves = tuple(
        BondAnalysisResultLeaf(
            category="bond",
            display_label=definition.display_label,
            scope_name=str(cluster_result["cluster_type"]),
            npy_path=(
                Path(str(cluster_result["output_dir"]))
                / f"{definition.filename_stem}_distribution.npy"
            ),
            point_count=int(
                cluster_result.get("bond_value_counts", {}).get(
                    definition.display_label,
                    0,
                )
            ),
        )
        for cluster_result in cluster_results
    )
    return BondAnalysisResultGroup(
        category="bond",
        display_label=definition.display_label,
        xlabel="Distance (A)",
        cluster_leaves=cluster_leaves,
        all_leaf=BondAnalysisResultLeaf(
            category="bond",
            display_label=definition.display_label,
            scope_name="all",
            npy_path=None,
            point_count=sum(leaf.point_count for leaf in cluster_leaves),
            is_all=True,
        ),
    )


def _build_angle_group(
    definition: AngleTripletDefinition,
    cluster_results: tuple[dict[str, object], ...],
) -> BondAnalysisResultGroup:
    cluster_leaves = tuple(
        BondAnalysisResultLeaf(
            category="angle",
            display_label=definition.display_label,
            scope_name=str(cluster_result["cluster_type"]),
            npy_path=(
                Path(str(cluster_result["output_dir"]))
                / f"{definition.filename_stem}_angles.npy"
            ),
            point_count=int(
                cluster_result.get("angle_value_counts", {}).get(
                    definition.display_label,
                    0,
                )
            ),
        )
        for cluster_result in cluster_results
    )
    return BondAnalysisResultGroup(
        category="angle",
        display_label=definition.display_label,
        xlabel="Angle (deg)",
        cluster_leaves=cluster_leaves,
        all_leaf=BondAnalysisResultLeaf(
            category="angle",
            display_label=definition.display_label,
            scope_name="all",
            npy_path=None,
            point_count=sum(leaf.point_count for leaf in cluster_leaves),
            is_all=True,
        ),
    )


def _load_distribution_values(leaf: BondAnalysisResultLeaf) -> np.ndarray:
    if leaf.is_all or leaf.npy_path is None or not leaf.npy_path.exists():
        return np.array([], dtype=float)
    payload = np.load(leaf.npy_path, allow_pickle=False)
    if (
        getattr(payload.dtype, "names", None)
        and "value" in payload.dtype.names
    ):
        return np.asarray(payload["value"], dtype=float)
    return np.asarray(payload, dtype=float)


def _slugify_filename_fragment(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return text or "distribution"


__all__ = [
    "BondAnalysisDistributionSeries",
    "BondAnalysisPlotRequest",
    "BondAnalysisResultGroup",
    "BondAnalysisResultIndex",
    "BondAnalysisResultLeaf",
    "build_plot_request",
    "draw_plot_request",
    "export_plot_request_csv",
    "load_result_index",
    "recommended_plot_request_filename",
]
