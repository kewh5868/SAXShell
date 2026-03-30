from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from saxshell.saxs.stoichiometry import (
    format_stoich_for_axis,
    parse_stoich_label,
    sort_stoich_labels,
)


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _load_prior_payload(
    json_path: str | Path | dict[str, object],
) -> dict[str, object]:
    if isinstance(json_path, dict):
        return dict(json_path)
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def export_prior_plot_data(
    json_path: str | Path | dict[str, object],
    output_path: str | Path,
    *,
    mode: str = "structure_fraction",
) -> Path:
    payload = _load_prior_payload(json_path)
    structures = payload.get("structures", {})
    total_files = float(payload.get("total_files", 0) or 0.0)
    labels = sort_stoich_labels(structures)

    entries: list[dict[str, object]] = []
    for structure in labels:
        motifs = structures[structure]
        atom_count = max(
            sum(int(token) for token in re.findall(r"(\d+)", structure)),
            1,
        )
        for motif in sorted(motifs, key=_natural_sort_key):
            motif_payload = motifs[motif]
            count = float(motif_payload.get("count", 0.0) or 0.0)
            structure_fraction = count / total_files if total_files else 0.0
            atom_fraction = (
                (count * atom_count)
                / sum(
                    max(
                        sum(
                            int(token)
                            for token in re.findall(r"(\d+)", other_label)
                        ),
                        1,
                    )
                    * float(other_payload.get("count", 0.0) or 0.0)
                    for other_label, motif_dict in structures.items()
                    for other_payload in motif_dict.values()
                )
                if structures
                else 0.0
            )
            entries.append(
                {
                    "structure": structure,
                    "motif": motif,
                    "count": count,
                    "weight": float(motif_payload.get("weight", 0.0)),
                    "structure_fraction": structure_fraction,
                    "atom_fraction": atom_fraction,
                    "selected_mode_value": (
                        atom_fraction
                        if mode == "atom_fraction"
                        else structure_fraction
                    ),
                    "profile_file": motif_payload.get("profile_file"),
                    "representative": motif_payload.get("representative"),
                }
            )

    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "total_files": payload.get("total_files", 0),
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def build_prior_histogram_export_payload(
    json_path: str | Path | dict[str, object],
    *,
    mode: str = "structure_fraction",
    value_mode: str = "percent",
    secondary_element: str | None = None,
) -> dict[str, object]:
    payload = _load_prior_payload(json_path)
    structures = payload.get("structures", {})
    labels = sort_stoich_labels(structures)
    normalized_mode = _normalize_prior_mode(mode)
    if normalized_mode not in {
        "structure_fraction",
        "atom_fraction",
        "solvent_sort_structure_fraction",
        "solvent_sort_atom_fraction",
    }:
        raise ValueError(
            "Unsupported prior plot mode: "
            f"{mode}. Choose structure_fraction, atom_fraction, "
            "solvent_sort_structure_fraction, or solvent_sort_atom_fraction."
        )
    is_atom_fraction = normalized_mode in {
        "atom_fraction",
        "solvent_sort_atom_fraction",
    }
    is_solvent_sort = normalized_mode.startswith("solvent_sort")

    total_files = float(payload.get("total_files", 0) or 0.0)
    atom_weight_total = sum(
        max(sum(int(token) for token in re.findall(r"(\d+)", label)), 1)
        * float(motif_payload.get("count", 0.0) or 0.0)
        for label, motif_dict in structures.items()
        for motif_payload in motif_dict.values()
    )

    if is_solvent_sort:
        if not secondary_element:
            raise ValueError(
                "Select a secondary atom filter before using Solvent Sort."
            )
        segments = _secondary_count_segments(structures, secondary_element)
        segment_labels = [
            f"{segment} {secondary_element}" for segment in segments
        ]
    else:
        segments = sorted(
            {
                motif
                for structure in structures.values()
                for motif in structure
            },
            key=_natural_sort_key,
        )
        segment_labels = [str(segment) for segment in segments]

    matrix = np.zeros((len(labels), len(segments)), dtype=float)
    for row_index, label in enumerate(labels):
        atom_count = max(
            sum(int(token) for token in re.findall(r"(\d+)", label)),
            1,
        )
        for column_index, segment in enumerate(segments):
            if is_solvent_sort:
                count = _secondary_segment_count(
                    structures[label],
                    secondary_element,
                    int(segment),
                )
            else:
                count = float(
                    structures[label].get(str(segment), {}).get("count", 0.0)
                    or 0.0
                )
            if is_atom_fraction:
                base_value = float(count * atom_count)
                denominator = atom_weight_total
            else:
                base_value = float(count)
                denominator = total_files
            matrix[row_index, column_index] = _scaled_prior_value(
                base_value,
                denominator,
                value_mode,
            )

    return {
        "plot_mode": normalized_mode,
        "value_mode": value_mode,
        "labels": labels,
        "axis_labels": [format_stoich_for_axis(label) for label in labels],
        "segments": [str(segment) for segment in segments],
        "segment_labels": segment_labels,
        "secondary_element": secondary_element,
        "matrix": matrix,
        "totals": matrix.sum(axis=1),
    }


def export_prior_histogram_table(
    json_path: str | Path | dict[str, object],
    output_path: str | Path,
    *,
    mode: str = "structure_fraction",
    value_mode: str = "percent",
    secondary_element: str | None = None,
) -> Path:
    payload = build_prior_histogram_export_payload(
        json_path,
        mode=mode,
        value_mode=value_mode,
        secondary_element=secondary_element,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(payload["matrix"], dtype=float)
    labels = [str(label) for label in payload["labels"]]
    axis_labels = [str(label) for label in payload["axis_labels"]]
    segments = [str(segment) for segment in payload["segments"]]
    segment_labels = [str(label) for label in payload["segment_labels"]]
    totals = np.asarray(payload["totals"], dtype=float)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        segment_prefix = (
            "secondary_count"
            if str(payload["plot_mode"]).startswith("solvent_sort")
            else "motif"
        )
        writer.writerow(
            [
                "structure",
                "axis_label",
                *[
                    (
                        f"{segment_prefix}_{segment}"
                        if segment_prefix == "motif"
                        else f"{segment_prefix}_{segment}_{segment_label}"
                    )
                    for segment, segment_label in zip(segments, segment_labels)
                ],
                "total",
            ]
        )
        for row_index, label in enumerate(labels):
            writer.writerow(
                [
                    label,
                    axis_labels[row_index],
                    *[f"{value:.10g}" for value in matrix[row_index]],
                    f"{totals[row_index]:.10g}",
                ]
            )
    return output_path


def export_prior_histogram_npy(
    json_path: str | Path | dict[str, object],
    output_path: str | Path,
    *,
    mode: str = "structure_fraction",
    value_mode: str = "percent",
    secondary_element: str | None = None,
) -> Path:
    payload = build_prior_histogram_export_payload(
        json_path,
        mode=mode,
        value_mode=value_mode,
        secondary_element=secondary_element,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)
    return output_path


def plot_md_prior_histogram(
    json_path: str | Path | dict[str, object],
    *,
    mode: str = "structure_fraction",
    secondary_element: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    cmap: str = "summer",
    structure_motif_colors: dict[str, str] | None = None,
    show_percent: bool = True,
    ax=None,
):
    small_total_threshold = 1.0
    export_payload = build_prior_histogram_export_payload(
        json_path,
        mode=mode,
        value_mode="percent",
        secondary_element=secondary_element,
    )
    labels = [str(label) for label in export_payload["labels"]]
    segments = [str(segment) for segment in export_payload["segments"]]
    segment_labels = [str(label) for label in export_payload["segment_labels"]]
    plot_mode = str(export_payload["plot_mode"])
    matrix = np.asarray(export_payload["matrix"], dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax.clear()

    if not labels:
        ax.set_title("No prior-weight data available")
        ax.set_xlabel("Structure")
        ax.set_ylabel("Fraction")
        return fig, ax

    colors = plt.get_cmap(cmap)(
        np.linspace(0.1, 0.9, max(len(segment_labels), 1), endpoint=True)
    )

    bottoms = np.zeros(len(labels), dtype=float)
    for index, segment_label in enumerate(segment_labels):
        heights_array = matrix[:, index]
        bar_colors = colors[index]
        if structure_motif_colors and not plot_mode.startswith("solvent_sort"):
            bar_colors = [
                structure_motif_colors.get(
                    f"{label}_{segments[index]}",
                    fallback_color,
                )
                for label, fallback_color in zip(
                    labels,
                    [colors[index]] * len(labels),
                )
            ]
        ax.bar(
            labels,
            heights_array,
            bottom=bottoms,
            label=segment_label,
            color=bar_colors,
            edgecolor="white",
            width=0.8,
        )
        bottoms += heights_array

    if show_percent:
        showed_small_total_marker = False
        for index, total in enumerate(bottoms):
            if total >= small_total_threshold:
                ax.text(
                    index,
                    total + 1.0,
                    f"{total:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            else:
                ax.scatter(index, total + 1.0, color="red", s=16, zorder=4)
                showed_small_total_marker = True

    ax.set_ylim(0.0, max(bottoms.max(initial=0.0) + 4.0, 10.0))
    ax.set_xlabel("Structure")
    ax.set_ylabel(
        "Percentage of Total Atom-Weighted Count (%)"
        if plot_mode in {"atom_fraction", "solvent_sort_atom_fraction"}
        else "Percentage of Total Structures (%)"
    )
    ax.set_title(_prior_plot_title(plot_mode, secondary_element))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [format_stoich_for_axis(label) for label in labels],
        rotation=45,
        ha="right",
    )
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if show_percent and showed_small_total_marker:
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                color="red",
                linestyle="None",
                markersize=5,
            )
        )
        legend_labels.append("< 1% total")
    ax.legend(
        legend_handles,
        legend_labels,
        title=(
            "Motif"
            if not plot_mode.startswith("solvent_sort")
            else f"{secondary_element} count"
        ),
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    fig.tight_layout()
    return fig, ax


def list_secondary_filter_elements(
    json_path: str | Path | dict[str, object],
) -> list[str]:
    payload = _load_prior_payload(json_path)
    return _payload_secondary_filter_elements(payload)


def _payload_secondary_filter_elements(
    payload: dict[str, object]
) -> list[str]:
    structures = payload.get("structures", {})
    structure_element_sets = [
        set(parse_stoich_label(str(label)).keys()) for label in structures
    ]
    axis_elements = (
        set.intersection(*structure_element_sets)
        if structure_element_sets
        else set()
    )
    available = {
        str(element)
        for element in payload.get("available_elements", [])
        if str(element).strip()
    }
    discovered = {
        str(element)
        for motif_dict in structures.values()
        for motif_payload in motif_dict.values()
        for element in motif_payload.get("secondary_atom_distributions", {})
    }
    secondary_elements = (available | discovered) - axis_elements
    return sorted(secondary_elements, key=_natural_sort_key)


def _secondary_count_segments(
    structures: dict[str, object],
    secondary_element: str,
) -> list[int]:
    segments = {
        int(count_key)
        for motif_dict in structures.values()
        for motif_payload in motif_dict.values()
        for count_key in (
            motif_payload.get("secondary_atom_distributions", {}).get(
                secondary_element,
                {},
            )
        )
    }
    if not segments:
        segments.add(0)
    return sorted(segments)


def _secondary_segment_count(
    motif_payloads: dict[str, object],
    secondary_element: str | None,
    segment_value: int,
) -> float:
    if secondary_element is None:
        return 0.0
    count_key = str(int(segment_value))
    total = 0.0
    for motif_payload in motif_payloads.values():
        secondary_distributions = motif_payload.get(
            "secondary_atom_distributions",
            {},
        )
        total += float(
            secondary_distributions.get(secondary_element, {}).get(
                count_key,
                0.0,
            )
        )
    return total


def _scaled_prior_value(
    base_value: float,
    denominator: float,
    value_mode: str,
) -> float:
    if value_mode == "count":
        return base_value
    if value_mode == "fraction":
        return base_value / denominator if denominator else 0.0
    if value_mode == "percent":
        return 100.0 * base_value / denominator if denominator else 0.0
    raise ValueError("value_mode must be 'count', 'fraction', or 'percent'.")


def _normalize_prior_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower().replace("-", "_")
    normalized = normalized.replace(" ", "_")
    if normalized == "solvent_sort":
        return "solvent_sort_structure_fraction"
    return normalized or "structure_fraction"


def _prior_plot_title(
    plot_mode: str,
    secondary_element: str | None,
) -> str:
    if plot_mode == "atom_fraction":
        return "Atom-Weighted Fraction Prior Histogram"
    if plot_mode == "structure_fraction":
        return "Structure-Fraction Prior Histogram"
    if plot_mode == "solvent_sort_atom_fraction":
        return (
            f"Solvent-Sort Atom Fraction Prior Histogram ({secondary_element})"
        )
    return f"Solvent-Sort Structure Fraction Prior Histogram ({secondary_element})"
