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
    value_kind = _prior_value_kind(payload)
    labels = sort_stoich_labels(structures)
    total_structure_value = _prior_total_value(
        structures,
        value_kind=value_kind,
    )
    total_atom_value = _prior_total_atom_value(
        structures,
        value_kind=value_kind,
    )

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
            normalized_weight = _prior_entry_value(
                motif_payload,
                value_kind=value_kind,
            )
            structure_fraction = (
                normalized_weight / total_structure_value
                if total_structure_value
                else 0.0
            )
            atom_fraction = (
                (normalized_weight * atom_count) / total_atom_value
                if total_atom_value
                else 0.0
            )
            entries.append(
                {
                    "structure": structure,
                    "motif": motif,
                    "count": count,
                    "weight": float(motif_payload.get("weight", 0.0)),
                    "normalized_weight": normalized_weight,
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
    normalized_mode = _normalize_prior_mode(mode)
    structures = _histogram_structures(
        payload,
        plot_mode=normalized_mode,
    )
    labels = sort_stoich_labels(structures)
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

    value_kind = _prior_value_kind(payload)
    total_structure_value = _prior_total_value(
        structures,
        value_kind=value_kind,
    )
    atom_weight_total = _prior_total_atom_value(
        structures,
        value_kind=value_kind,
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
        segment_labels = [
            _histogram_segment_label(structures, str(segment))
            for segment in segments
        ]

    matrix = np.zeros((len(labels), len(segments)), dtype=float)
    color_keys: list[list[str | None]] = [
        [None] * len(segments) for _ in range(len(labels))
    ]
    for row_index, label in enumerate(labels):
        for column_index, segment in enumerate(segments):
            motif_payload = structures[label].get(str(segment), {})
            if isinstance(motif_payload, dict):
                color_keys[row_index][column_index] = (
                    str(
                        motif_payload.get(
                            "histogram_component_key",
                            f"{label}_{segment}",
                        )
                    ).strip()
                    or None
                )
            if is_solvent_sort:
                count = _secondary_segment_count(
                    structures[label],
                    secondary_element,
                    int(segment),
                    value_kind=value_kind,
                )
            else:
                count = _prior_entry_value(
                    motif_payload,
                    value_kind=value_kind,
                )
            if is_atom_fraction:
                base_value = float(
                    count
                    * _prior_atom_weight_for_entry(
                        label,
                        motif_payload,
                    )
                )
                denominator = atom_weight_total
            else:
                base_value = float(count)
                denominator = total_structure_value
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
        "color_keys": color_keys,
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
    color_keys = [
        [None if key is None else str(key) for key in row]
        for row in export_payload.get("color_keys", [])
    ]
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
                    (
                        color_keys[row_index][index]
                        if row_index < len(color_keys)
                        and index < len(color_keys[row_index])
                        else f"{label}_{segments[index]}"
                    ),
                    fallback_color,
                )
                for row_index, (label, fallback_color) in enumerate(
                    zip(
                        labels,
                        [colors[index]] * len(labels),
                    )
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
    *,
    value_kind: str,
) -> float:
    if secondary_element is None:
        return 0.0
    count_key = str(int(segment_value))
    total = 0.0
    for motif_payload in motif_payloads.values():
        weight = _prior_entry_value(
            motif_payload,
            value_kind=value_kind,
        )
        secondary_distributions = motif_payload.get(
            "secondary_atom_distributions",
            {},
        )
        distribution = secondary_distributions.get(secondary_element, {})
        if not isinstance(distribution, dict):
            continue
        segment_matches = float(distribution.get(count_key, 0.0) or 0.0)
        distribution_total = float(
            sum(float(value or 0.0) for value in distribution.values())
        )
        if distribution_total <= 0.0:
            continue
        total += weight * (segment_matches / distribution_total)
    return total


def _prior_value_kind(payload: dict[str, object]) -> str:
    normalized = str(payload.get("value_kind", "count")).strip().lower()
    return normalized or "count"


def _prior_entry_value(
    motif_payload: object,
    *,
    value_kind: str,
) -> float:
    if not isinstance(motif_payload, dict):
        return 0.0
    if value_kind == "normalized_weight":
        return float(
            motif_payload.get(
                "normalized_weight",
                motif_payload.get("weight", 0.0),
            )
            or 0.0
        )
    return float(motif_payload.get("count", 0.0) or 0.0)


def _prior_total_value(
    structures: dict[str, dict[str, object]],
    *,
    value_kind: str,
) -> float:
    return float(
        sum(
            _prior_entry_value(motif_payload, value_kind=value_kind)
            for motif_dict in structures.values()
            for motif_payload in motif_dict.values()
        )
    )


def _prior_total_atom_value(
    structures: dict[str, dict[str, object]],
    *,
    value_kind: str,
) -> float:
    return float(
        sum(
            _prior_atom_weight_for_entry(label, motif_payload)
            * _prior_entry_value(motif_payload, value_kind=value_kind)
            for label, motif_dict in structures.items()
            for motif_payload in motif_dict.values()
        )
    )


def _histogram_structures(
    payload: dict[str, object],
    *,
    plot_mode: str,
) -> dict[str, dict[str, dict[str, object]]]:
    raw_structures = payload.get("structures", {})
    structures = {
        str(label): {
            str(motif): dict(motif_payload)
            for motif, motif_payload in dict(motif_dict).items()
            if isinstance(motif_payload, dict)
        }
        for label, motif_dict in dict(raw_structures).items()
    }
    del plot_mode
    if not bool(payload.get("includes_predicted_structures", False)):
        return structures

    observed_labels = [
        label
        for label, motif_dict in structures.items()
        if any(
            str(motif_payload.get("source_kind", "cluster_dir"))
            != "predicted_structure"
            for motif_payload in motif_dict.values()
        )
    ]
    if not observed_labels:
        return structures

    observed_element_basis = {
        str(element)
        for label in observed_labels
        for element in parse_stoich_label(label)
    }
    remapped: dict[str, dict[str, dict[str, object]]] = {}
    for label, motif_dict in structures.items():
        for motif, motif_payload in motif_dict.items():
            source_kind = str(motif_payload.get("source_kind", "cluster_dir"))
            payload_copy = dict(motif_payload)
            payload_copy["histogram_component_key"] = _histogram_component_key(
                label,
                motif,
                payload_copy,
            )
            target_label = label
            target_motif = str(motif)
            target_counts = _filtered_histogram_counts(
                label,
                observed_element_basis=observed_element_basis,
            )
            if source_kind == "predicted_structure":
                filtered_label = _histogram_label_from_counts(target_counts)
                if filtered_label:
                    target_label = filtered_label
                if target_label != label:
                    target_motif = f"{motif}::{label}"
                    payload_copy["histogram_segment_label"] = (
                        f"{motif} ({label})"
                    )
            payload_copy["histogram_atom_weight"] = float(
                max(sum(target_counts.values()), 1)
            )
            remapped.setdefault(target_label, {})
            if target_motif in remapped[target_label]:
                _merge_histogram_motif_payload(
                    remapped[target_label][target_motif],
                    payload_copy,
                )
            else:
                remapped[target_label][target_motif] = payload_copy
    return remapped


def _histogram_segment_label(
    structures: dict[str, dict[str, object]],
    segment: str,
) -> str:
    for motif_dict in structures.values():
        motif_payload = motif_dict.get(segment)
        if isinstance(motif_payload, dict):
            label = str(
                motif_payload.get("histogram_segment_label", "")
            ).strip()
            if label:
                return label
    return str(segment)


def _filtered_histogram_counts(
    label: str,
    *,
    observed_element_basis: set[str],
) -> dict[str, int]:
    parsed = {
        str(element): int(count)
        for element, count in parse_stoich_label(label).items()
        if int(count) > 0
    }
    filtered = {
        element: count
        for element, count in parsed.items()
        if not observed_element_basis or element in observed_element_basis
    }
    return filtered or parsed


def _histogram_label_from_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    if "Pb" in counts or "I" in counts:
        elements = ["Pb", "I"] + sorted(
            element for element in counts if element not in {"Pb", "I"}
        )
    else:
        elements = sorted(counts)
    return "".join(
        (
            element
            if int(counts.get(element, 0)) == 1
            else f"{element}{counts[element]}"
        )
        for element in elements
        if int(counts.get(element, 0)) > 0
    )


def _merge_histogram_motif_payload(
    existing: dict[str, object],
    incoming: dict[str, object],
) -> None:
    for key in (
        "count",
        "weight",
        "normalized_weight",
        "observed_only_weight",
    ):
        existing[key] = float(existing.get(key, 0.0) or 0.0) + float(
            incoming.get(key, 0.0) or 0.0
        )
    existing["histogram_atom_weight"] = max(
        float(existing.get("histogram_atom_weight", 1.0) or 1.0),
        float(incoming.get("histogram_atom_weight", 1.0) or 1.0),
    )
    existing.setdefault(
        "histogram_component_key",
        incoming.get("histogram_component_key"),
    )
    existing.setdefault(
        "histogram_segment_label",
        incoming.get("histogram_segment_label", ""),
    )
    existing_secondary = dict(existing.get("secondary_atom_distributions", {}))
    incoming_secondary = dict(incoming.get("secondary_atom_distributions", {}))
    for element, distribution in incoming_secondary.items():
        merged_distribution = {
            str(segment): float(value)
            for segment, value in dict(
                existing_secondary.get(element, {})
            ).items()
        }
        for segment, value in dict(distribution).items():
            segment_key = str(segment)
            merged_distribution[segment_key] = merged_distribution.get(
                segment_key,
                0.0,
            ) + float(value or 0.0)
        existing_secondary[str(element)] = merged_distribution
    existing["secondary_atom_distributions"] = existing_secondary


def _histogram_component_key(
    label: str,
    motif: str,
    motif_payload: dict[str, object],
) -> str:
    profile_file = str(motif_payload.get("profile_file", "")).strip()
    if profile_file:
        return Path(profile_file).stem
    return f"{label}_{motif}"


def _prior_atom_weight_for_entry(
    label: str,
    motif_payload: object,
) -> float:
    if isinstance(motif_payload, dict):
        override = motif_payload.get("histogram_atom_weight", None)
        if override is not None:
            return float(override)
    return float(
        max(sum(int(token) for token in re.findall(r"(\d+)", label)), 1)
    )


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
