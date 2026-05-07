from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from saxshell.plotting.igor_inline import (
    apply_igor_inline_text_artist,
    igor_inline_to_mathtext,
    prepare_igor_inline_segments,
)
from saxshell.saxs.stoichiometry import (
    format_stoich_for_axis,
    sort_stoich_labels,
)

STACKED_HISTOGRAM_LEGEND_LOCATIONS = (
    ("Outside Upper Right", "outside_upper_right"),
    ("Upper Right", "upper_right"),
    ("Upper Left", "upper_left"),
    ("Lower Right", "lower_right"),
    ("Lower Left", "lower_left"),
    ("Best", "best"),
)


@dataclass(slots=True)
class StackedHistogramPlotDefaults:
    title: str
    x_label: str
    y_label: str
    legend_title: str
    title_position_x: float = 0.5
    title_position_y: float = 1.0
    default_colormap_name: str = ""
    available_colormap_names: tuple[str, ...] = ()
    raw_category_labels: tuple[str, ...] = ()
    default_label_entries: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True)
class StackedHistogramPlotSettings:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend_title: str | None = None
    title_position_x: float | None = None
    title_position_y: float | None = None
    font_family: str = ""
    title_font_size: float = 12.0
    axis_label_font_size: float = 11.0
    tick_label_font_size: float = 9.0
    legend_font_size: float = 8.5
    annotation_font_size: float = 9.0
    max_y_ticks: int = 8
    x_tick_rotation: int = 45
    y_tick_rotation: int = 0
    show_minor_y_ticks: bool = False
    show_total_annotations: bool = True
    show_legend: bool = True
    legend_location: str = "outside_upper_right"
    label_order: list[str] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)

    def resolve_title(self, defaults: StackedHistogramPlotDefaults) -> str:
        return defaults.title if self.title is None else self.title

    def resolve_x_label(self, defaults: StackedHistogramPlotDefaults) -> str:
        return defaults.x_label if self.x_label is None else self.x_label

    def resolve_y_label(self, defaults: StackedHistogramPlotDefaults) -> str:
        return defaults.y_label if self.y_label is None else self.y_label

    def resolve_legend_title(
        self,
        defaults: StackedHistogramPlotDefaults,
    ) -> str:
        return (
            defaults.legend_title
            if self.legend_title is None
            else self.legend_title
        )

    def resolve_title_position_x(
        self,
        defaults: StackedHistogramPlotDefaults,
    ) -> float:
        return (
            defaults.title_position_x
            if self.title_position_x is None
            else self.title_position_x
        )

    def resolve_title_position_y(
        self,
        defaults: StackedHistogramPlotDefaults,
    ) -> float:
        return (
            defaults.title_position_y
            if self.title_position_y is None
            else self.title_position_y
        )

    def sync_labels(
        self,
        raw_labels: Sequence[str],
        *,
        default_label_entries: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        default_entries = (
            [
                (str(raw_label), format_stoich_for_axis(str(raw_label)))
                for raw_label in raw_labels
            ]
            if default_label_entries is None
            else [
                (str(raw_label), str(display_label))
                for raw_label, display_label in default_label_entries
            ]
        )
        default_map = {
            raw_label: display_label
            for raw_label, display_label in default_entries
        }
        existing = dict(self.label_map)
        preserved_order = [
            raw_label
            for raw_label in self.label_order
            if raw_label in default_map
        ]
        remaining = [
            raw_label
            for raw_label, _display_label in default_entries
            if raw_label not in preserved_order
        ]
        self.label_order = preserved_order + remaining
        self.label_map = {
            raw_label: existing.get(raw_label, default_map[raw_label])
            for raw_label in self.label_order
        }

    def display_label(self, raw_label: str) -> str:
        return self.label_map.get(raw_label, raw_label)

    def ordered_raw_labels(
        self,
        defaults: StackedHistogramPlotDefaults,
    ) -> list[str]:
        if self.label_order:
            available = set(defaults.raw_category_labels)
            ordered = [raw for raw in self.label_order if raw in available]
            remaining = [
                raw
                for raw in defaults.raw_category_labels
                if raw not in ordered
            ]
            return ordered + remaining
        return list(defaults.raw_category_labels)

    def ordered_label_entries(
        self,
        defaults: StackedHistogramPlotDefaults,
    ) -> list[tuple[str, str]]:
        return [
            (raw_label, self.display_label(raw_label))
            for raw_label in self.ordered_raw_labels(defaults)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "legend_title": self.legend_title,
            "title_position_x": self.title_position_x,
            "title_position_y": self.title_position_y,
            "font_family": self.font_family,
            "title_font_size": self.title_font_size,
            "axis_label_font_size": self.axis_label_font_size,
            "tick_label_font_size": self.tick_label_font_size,
            "legend_font_size": self.legend_font_size,
            "annotation_font_size": self.annotation_font_size,
            "max_y_ticks": self.max_y_ticks,
            "x_tick_rotation": self.x_tick_rotation,
            "y_tick_rotation": self.y_tick_rotation,
            "show_minor_y_ticks": self.show_minor_y_ticks,
            "show_total_annotations": self.show_total_annotations,
            "show_legend": self.show_legend,
            "legend_location": self.legend_location,
            "label_order": list(self.label_order),
            "label_map": dict(self.label_map),
        }

    def update_from_dict(self, payload: Mapping[str, object]) -> None:
        for field_name in (
            "title",
            "x_label",
            "y_label",
            "legend_title",
            "title_position_x",
            "title_position_y",
        ):
            if field_name in payload:
                setattr(self, field_name, payload[field_name])
        if "font_family" in payload:
            self.font_family = str(payload["font_family"] or "")
        for field_name in (
            "title_font_size",
            "axis_label_font_size",
            "tick_label_font_size",
            "legend_font_size",
            "annotation_font_size",
        ):
            if field_name in payload:
                setattr(self, field_name, float(payload[field_name]))
        for field_name in (
            "max_y_ticks",
            "x_tick_rotation",
            "y_tick_rotation",
        ):
            if field_name in payload:
                setattr(self, field_name, int(payload[field_name]))
        for field_name in (
            "show_minor_y_ticks",
            "show_total_annotations",
            "show_legend",
        ):
            if field_name in payload:
                setattr(self, field_name, bool(payload[field_name]))
        if "legend_location" in payload:
            self.legend_location = str(payload["legend_location"] or "best")
        if "label_order" in payload:
            self.label_order = [str(value) for value in payload["label_order"]]
        if "label_map" in payload:
            label_map = payload["label_map"]
            if isinstance(label_map, Mapping):
                self.label_map = {
                    str(key): str(value) for key, value in label_map.items()
                }


def render_stacked_histogram_export_payload(
    export_payload: Mapping[str, object],
    *,
    ax,
    defaults: StackedHistogramPlotDefaults,
    settings: StackedHistogramPlotSettings | None = None,
    cmap: str | None = None,
    structure_segment_colors: Mapping[str, str] | None = None,
    show_percent: bool = True,
):
    resolved_settings = (
        StackedHistogramPlotSettings() if settings is None else settings
    )
    fig = ax.figure
    ax.clear()

    labels = [str(label) for label in export_payload.get("labels", ())]
    axis_labels = [
        str(label) for label in export_payload.get("axis_labels", ())
    ]
    segments = [str(segment) for segment in export_payload.get("segments", ())]
    segment_labels = [
        str(label) for label in export_payload.get("segment_labels", ())
    ]
    plot_mode = str(export_payload.get("plot_mode", "structure_fraction"))
    matrix = np.asarray(export_payload.get("matrix", []), dtype=float)
    color_keys = [
        [None if key is None else str(key) for key in row]
        for row in export_payload.get("color_keys", [])
    ]

    if not labels:
        ax.set_title("No prior-weight data available")
        ax.set_xlabel(resolved_settings.resolve_x_label(defaults))
        ax.set_ylabel(resolved_settings.resolve_y_label(defaults))
        return fig, ax

    cmap_name = str(cmap or defaults.default_colormap_name or "summer")
    colors = plt.get_cmap(cmap_name)(
        np.linspace(0.1, 0.9, max(len(segment_labels), 1), endpoint=True)
    )

    x_positions = np.arange(len(labels), dtype=float)
    bottoms = np.zeros(len(labels), dtype=float)
    for index, segment_label in enumerate(segment_labels):
        heights_array = matrix[:, index]
        bar_colors = colors[index]
        if structure_segment_colors and not plot_mode.startswith(
            "solvent_sort"
        ):
            bar_colors = [
                structure_segment_colors.get(
                    (
                        color_keys[row_index][index]
                        if row_index < len(color_keys)
                        and index < len(color_keys[row_index])
                        else f"{label}_{segments[index]}"
                    ),
                    fallback_color,
                )
                for row_index, (label, fallback_color) in enumerate(
                    zip(labels, [colors[index]] * len(labels), strict=False)
                )
            ]
        ax.bar(
            x_positions,
            heights_array,
            bottom=bottoms,
            label=segment_label,
            color=bar_colors,
            edgecolor="white",
            width=0.8,
        )
        bottoms += heights_array

    showed_small_total_marker = False
    if show_percent and resolved_settings.show_total_annotations:
        for index, total in enumerate(bottoms):
            if total >= 1.0:
                ax.text(
                    x_positions[index],
                    total + 1.0,
                    f"{total:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=resolved_settings.annotation_font_size,
                    **_font_kwargs(resolved_settings.font_family),
                )
            else:
                ax.scatter(
                    x_positions[index],
                    total + 1.0,
                    color="red",
                    s=16,
                    zorder=4,
                )
                showed_small_total_marker = True

    max_total = float(np.max(bottoms)) if bottoms.size else 0.0
    ax.set_ylim(0.0, max(max_total + 4.0, 10.0))
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel(
        resolved_settings.resolve_x_label(defaults),
        fontsize=resolved_settings.axis_label_font_size,
        **_font_kwargs(resolved_settings.font_family),
    )
    apply_igor_inline_text_artist(
        ax.xaxis.label,
        resolved_settings.resolve_x_label(defaults),
        default_font_size=resolved_settings.axis_label_font_size,
        gid_prefix="stacked-histogram-x-label",
        target_axes=ax,
    )
    ax.set_ylabel(
        resolved_settings.resolve_y_label(defaults),
        fontsize=resolved_settings.axis_label_font_size,
        **_font_kwargs(resolved_settings.font_family),
    )
    apply_igor_inline_text_artist(
        ax.yaxis.label,
        resolved_settings.resolve_y_label(defaults),
        default_font_size=resolved_settings.axis_label_font_size,
        gid_prefix="stacked-histogram-y-label",
        target_axes=ax,
    )
    ax.set_title(
        resolved_settings.resolve_title(defaults),
        y=resolved_settings.resolve_title_position_y(defaults),
        fontsize=resolved_settings.title_font_size,
        **_font_kwargs(resolved_settings.font_family),
    )
    ax.title.set_x(resolved_settings.resolve_title_position_x(defaults))
    apply_igor_inline_text_artist(
        ax.title,
        resolved_settings.resolve_title(defaults),
        default_font_size=resolved_settings.title_font_size,
        gid_prefix="stacked-histogram-title",
        target_axes=ax,
    )

    rendered_x_tick_labels: list[str] = []
    composite_x_tick_labels: dict[int, str] = {}
    for tick_index, axis_label in enumerate(axis_labels):
        segments_for_label, has_markup = prepare_igor_inline_segments(
            axis_label,
            default_font_size=resolved_settings.tick_label_font_size,
        )
        if not has_markup:
            rendered_x_tick_labels.append(axis_label)
            continue
        if any(
            not math.isclose(
                segment.font_size,
                resolved_settings.tick_label_font_size,
            )
            for segment in segments_for_label
        ):
            rendered_x_tick_labels.append(" ")
            composite_x_tick_labels[tick_index] = axis_label
            continue
        rendered_x_tick_labels.append(
            igor_inline_to_mathtext(
                axis_label,
                default_font_size=resolved_settings.tick_label_font_size,
            )
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        rendered_x_tick_labels,
        rotation=resolved_settings.x_tick_rotation,
        ha="right",
    )
    ax.tick_params(axis="x", labelsize=resolved_settings.tick_label_font_size)
    ax.tick_params(axis="y", labelsize=resolved_settings.tick_label_font_size)
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=max(resolved_settings.max_y_ticks, 2))
    )
    if resolved_settings.show_minor_y_ticks:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.minorticks_off()

    for tick_label in ax.get_xticklabels():
        _apply_font_to_text(
            tick_label,
            font_family=resolved_settings.font_family,
            rotation=resolved_settings.x_tick_rotation,
        )
    for tick_label in ax.get_yticklabels():
        _apply_font_to_text(
            tick_label,
            font_family=resolved_settings.font_family,
            rotation=resolved_settings.y_tick_rotation,
        )
    for tick_index, tick_label in enumerate(ax.get_xticklabels()):
        if tick_index not in composite_x_tick_labels:
            continue
        apply_igor_inline_text_artist(
            tick_label,
            composite_x_tick_labels[tick_index],
            default_font_size=resolved_settings.tick_label_font_size,
            gid_prefix=f"stacked-histogram-x-tick-{tick_index}",
            target_axes=ax,
        )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if (
        show_percent
        and resolved_settings.show_total_annotations
        and showed_small_total_marker
    ):
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

    if resolved_settings.show_legend and legend_handles:
        legend = ax.legend(
            legend_handles,
            legend_labels,
            title=resolved_settings.resolve_legend_title(defaults),
            fontsize=resolved_settings.legend_font_size,
            **_legend_kwargs(resolved_settings.legend_location),
        )
        for legend_text in legend.get_texts():
            _apply_font_to_text(
                legend_text,
                font_family=resolved_settings.font_family,
            )
        _apply_font_to_text(
            legend.get_title(),
            font_family=resolved_settings.font_family,
        )
        apply_igor_inline_text_artist(
            legend.get_title(),
            resolved_settings.resolve_legend_title(defaults),
            default_font_size=resolved_settings.legend_font_size,
            gid_prefix="stacked-histogram-legend-title",
            target_axes=ax,
        )

    fig.tight_layout()
    return fig, ax


def default_histogram_label_entries(
    raw_labels: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(raw_label), format_stoich_for_axis(str(raw_label)))
        for raw_label in raw_labels
    )


def sorted_histogram_label_entries(
    raw_labels: Sequence[str],
) -> list[tuple[str, str]]:
    return [
        (raw_label, format_stoich_for_axis(raw_label))
        for raw_label in sort_stoich_labels(raw_labels)
    ]


def _font_kwargs(font_family: str) -> dict[str, str]:
    return {} if not font_family else {"fontfamily": font_family}


def _apply_font_to_text(
    text_artist,
    *,
    font_family: str,
    rotation: float | None = None,
) -> None:
    if font_family:
        text_artist.set_fontfamily(font_family)
    if rotation is not None:
        text_artist.set_rotation(rotation)


def _legend_kwargs(location: str) -> dict[str, object]:
    if location == "outside_upper_right":
        return {"bbox_to_anchor": (1.02, 1.0), "loc": "upper left"}
    return {"loc": location.replace("_", " ")}


__all__ = [
    "STACKED_HISTOGRAM_LEGEND_LOCATIONS",
    "StackedHistogramPlotDefaults",
    "StackedHistogramPlotSettings",
    "default_histogram_label_entries",
    "render_stacked_histogram_export_payload",
    "sorted_histogram_label_entries",
]
