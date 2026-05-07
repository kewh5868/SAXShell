from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from saxshell.clusterdynamics.workflow import ClusterDynamicsResult
from saxshell.plotting.igor_inline import (
    apply_igor_inline_text_artist,
    igor_inline_to_mathtext,
    prepare_igor_inline_segments,
)
from saxshell.plotting.plot_editor import (
    HeatmapPlotDefaults,
    HeatmapPlotEditorControls,
    HeatmapPlotSettings,
    PlotEditorWindow,
)
from saxshell.saxs.stoichiometry import format_stoich_for_axis

PLOT_COLORMAPS = ("viridis", "magma", "cividis", "inferno", "turbo")
DISPLAY_MODE_LABELS = {
    "count": "Counts / bin",
    "fraction": "Fraction / bin",
    "mean_count": "Mean count / frame",
}
DISPLAY_MODE_COLORBAR_LABELS = {
    "count": "Clusters in bin",
    "fraction": "Cluster fraction",
    "mean_count": "Mean clusters per frame",
}
OVERLAY_SERIES = (
    ("None", None),
    ("Temperature", "temperature"),
    ("Potential Energy", "potential"),
    ("Kinetic Energy", "kinetic"),
)
OVERLAY_COLORS = {
    "temperature": "#1f77b4",
    "potential": "#2e8b57",
    "kinetic": "#c0392b",
}


class ClusterDynamicsPlotPanel(QWidget):
    """Interactive time-binned cluster heatmap panel."""

    _MIN_PANEL_HEIGHT = 420
    _MIN_CANVAS_HEIGHT = 300

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        enable_plot_editor: bool = False,
    ) -> None:
        super().__init__(parent)
        self._enable_plot_editor = bool(enable_plot_editor)
        self._result: ClusterDynamicsResult | None = None
        self._plot_settings = HeatmapPlotSettings()
        self._plot_editor_window: PlotEditorWindow | None = None
        self._plot_editor_controls: HeatmapPlotEditorControls | None = None
        self.plot_editor_button: QPushButton | None = None
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        self.setMinimumHeight(self._MIN_PANEL_HEIGHT)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        if self._enable_plot_editor:
            editor_row = QHBoxLayout()
            editor_row.setContentsMargins(0, 0, 0, 0)
            editor_row.setSpacing(8)
            self.plot_editor_button = QPushButton("Open Plot Editor")
            self.plot_editor_button.clicked.connect(self.open_plot_editor)
            editor_row.addWidget(self.plot_editor_button)
            editor_row.addStretch(1)
            root.addLayout(editor_row)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        controls.addWidget(QLabel("Heatmap"))
        self.display_mode_combo = QComboBox()
        for mode, label in DISPLAY_MODE_LABELS.items():
            self.display_mode_combo.addItem(label, mode)
        self.display_mode_combo.setCurrentIndex(1)
        self.display_mode_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.display_mode_combo)

        controls.addWidget(QLabel("Units"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItem("fs", "fs")
        self.time_unit_combo.addItem("ps", "ps")
        self.time_unit_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.time_unit_combo)

        controls.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        for cmap_name in PLOT_COLORMAPS:
            self.colormap_combo.addItem(cmap_name, cmap_name)
        self.colormap_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.colormap_combo)

        controls.addWidget(QLabel("Lower q"))
        self.lower_quantile_spin = QDoubleSpinBox()
        self.lower_quantile_spin.setDecimals(2)
        self.lower_quantile_spin.setRange(0.0, 0.95)
        self.lower_quantile_spin.setSingleStep(0.05)
        self.lower_quantile_spin.setValue(0.05)
        self.lower_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.lower_quantile_spin)

        controls.addWidget(QLabel("Upper q"))
        self.upper_quantile_spin = QDoubleSpinBox()
        self.upper_quantile_spin.setDecimals(2)
        self.upper_quantile_spin.setRange(0.05, 1.0)
        self.upper_quantile_spin.setSingleStep(0.05)
        self.upper_quantile_spin.setValue(0.95)
        self.upper_quantile_spin.valueChanged.connect(
            self._on_quantile_changed
        )
        controls.addWidget(self.upper_quantile_spin)

        controls.addWidget(QLabel("Overlay"))
        self.overlay_combo = QComboBox()
        for label, data in OVERLAY_SERIES:
            self.overlay_combo.addItem(label, data)
        self.overlay_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.overlay_combo)
        controls.addStretch(1)

        root.addWidget(controls_widget)

        self.figure = Figure(figsize=(9.2, 7.2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(self._MIN_CANVAS_HEIGHT)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(NavigationToolbar(self.canvas, self))
        root.addWidget(self.canvas, stretch=1)

    def set_result(self, result: ClusterDynamicsResult | None) -> None:
        self._result = result
        has_energy = bool(
            result is not None and result.energy_data is not None
        )
        self.overlay_combo.setEnabled(has_energy)
        if not has_energy:
            self.overlay_combo.setCurrentIndex(0)
        self.refresh_plot()

    def open_plot_editor(self) -> None:
        if not self._enable_plot_editor:
            return
        if self._plot_editor_window is not None:
            self._plot_editor_window.show()
            self._plot_editor_window.raise_()
            self._plot_editor_window.activateWindow()
            self._plot_editor_window.refresh_preview()
            return

        defaults = self._current_plot_defaults()
        self._plot_editor_controls = HeatmapPlotEditorControls(
            settings=self._plot_settings,
            defaults=defaults,
            parent=self,
        )
        self._plot_editor_controls.settings_changed.connect(self.refresh_plot)
        self._plot_editor_controls.x_axis_unit_changed.connect(
            self._on_plot_editor_x_axis_unit_changed
        )
        self._plot_editor_controls.colormap_changed.connect(
            self._on_plot_editor_colormap_changed
        )
        self._plot_editor_window = PlotEditorWindow(
            window_title="Cluster Dynamics Colormap Editor",
            controls_widget=self._plot_editor_controls,
            render_preview=self._render_plot_figure,
            pickle_state_provider=self._plot_editor_pickle_state,
            apply_loaded_pickle_state=self._apply_loaded_plot_editor_pickle_state,
            parent=self,
        )
        self._plot_editor_window.closed.connect(self._on_plot_editor_closed)
        self._plot_editor_window.refresh_preview()
        self._plot_editor_window.show()
        self._plot_editor_window.raise_()
        self._plot_editor_window.activateWindow()

    def refresh_plot(self) -> None:
        self._render_plot_figure(self.figure)
        self.canvas.draw_idle()
        if self._plot_editor_window is not None:
            self._plot_editor_window.refresh_preview()

    def _on_plot_editor_closed(self) -> None:
        self._plot_editor_window = None
        self._plot_editor_controls = None

    def _on_plot_editor_colormap_changed(self, colormap_name: str) -> None:
        index = self.colormap_combo.findData(colormap_name)
        if index < 0 or index == self.colormap_combo.currentIndex():
            return
        self.colormap_combo.setCurrentIndex(index)

    def _on_plot_editor_x_axis_unit_changed(self, unit_name: str) -> None:
        index = self.time_unit_combo.findData(unit_name)
        if index < 0 or index == self.time_unit_combo.currentIndex():
            return
        self.time_unit_combo.setCurrentIndex(index)

    def _sync_plot_editor_defaults(
        self, defaults: HeatmapPlotDefaults
    ) -> None:
        if (
            self._plot_editor_controls is not None
            and self._plot_editor_controls.needs_default_sync(defaults)
        ):
            self._plot_editor_controls.sync_defaults(defaults)

    def _plot_editor_pickle_state(self) -> dict[str, object]:
        return {
            "plot_editor_state": {
                "kind": "heatmap_plot_editor_state",
                "version": 1,
                "heatmap_settings": self._plot_settings.to_dict(),
                "panel_state": {
                    "display_mode": self._display_mode(),
                    "time_unit": str(self.time_unit_combo.currentData() or ""),
                    "colormap_name": str(
                        self.colormap_combo.currentData() or ""
                    ),
                    "lower_quantile": float(self.lower_quantile_spin.value()),
                    "upper_quantile": float(self.upper_quantile_spin.value()),
                    "overlay_name": self.overlay_combo.currentData(),
                },
            }
        }

    def _apply_loaded_plot_editor_pickle_state(
        self,
        payload: Mapping[str, object],
    ) -> bool:
        editor_state = payload.get("plot_editor_state")
        if not isinstance(editor_state, Mapping):
            return False
        if str(editor_state.get("kind")) != "heatmap_plot_editor_state":
            return False

        heatmap_settings = editor_state.get("heatmap_settings")
        if isinstance(heatmap_settings, Mapping):
            self._plot_settings.update_from_dict(heatmap_settings)

        panel_state = editor_state.get("panel_state")
        if isinstance(panel_state, Mapping):
            self._apply_panel_state_from_pickle(panel_state)

        defaults = self._current_plot_defaults()
        self._plot_settings.sync_labels(
            defaults.raw_cluster_labels,
            default_label_entries=defaults.default_label_entries,
        )
        if self._plot_editor_controls is not None:
            self._plot_editor_controls.sync_defaults(defaults)
        self.refresh_plot()
        return True

    def _apply_panel_state_from_pickle(
        self,
        panel_state: Mapping[str, object],
    ) -> None:
        self.display_mode_combo.blockSignals(True)
        self.time_unit_combo.blockSignals(True)
        self.colormap_combo.blockSignals(True)
        self.lower_quantile_spin.blockSignals(True)
        self.upper_quantile_spin.blockSignals(True)
        self.overlay_combo.blockSignals(True)
        try:
            self._set_combo_data_if_present(
                self.display_mode_combo,
                panel_state.get("display_mode"),
            )
            self._set_combo_data_if_present(
                self.time_unit_combo,
                panel_state.get("time_unit"),
            )
            self._set_combo_data_if_present(
                self.colormap_combo,
                panel_state.get("colormap_name"),
            )
            if "lower_quantile" in panel_state:
                self.lower_quantile_spin.setValue(
                    float(panel_state["lower_quantile"])
                )
            if "upper_quantile" in panel_state:
                self.upper_quantile_spin.setValue(
                    float(panel_state["upper_quantile"])
                )
            self._ensure_valid_quantiles()
            self._set_combo_data_if_present(
                self.overlay_combo,
                panel_state.get("overlay_name"),
            )
        finally:
            self.display_mode_combo.blockSignals(False)
            self.time_unit_combo.blockSignals(False)
            self.colormap_combo.blockSignals(False)
            self.lower_quantile_spin.blockSignals(False)
            self.upper_quantile_spin.blockSignals(False)
            self.overlay_combo.blockSignals(False)

    @staticmethod
    def _set_combo_data_if_present(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _render_plot_figure(self, figure: Figure) -> None:
        defaults = self._current_plot_defaults()
        self._plot_settings.sync_labels(
            defaults.raw_cluster_labels,
            default_label_entries=defaults.default_label_entries,
        )

        figure.clear()
        if self._result is None:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "Run the analysis to render the cluster-distribution heatmap.",
            )
            return

        if self._result.bin_count == 0:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No time bins are available for the current selection.",
            )
            return

        matrix = self._result.matrix(self._display_mode())
        if matrix.size == 0 or len(self._result.cluster_labels) == 0:
            self._sync_plot_editor_defaults(defaults)
            axis = figure.add_subplot(111)
            self._draw_placeholder(
                axis,
                "No clusters were detected in the selected time window.",
            )
            return

        overlay_name = self.overlay_combo.currentData()
        show_overlay = bool(
            overlay_name is not None and self._result.energy_data is not None
        )

        if show_overlay:
            grid = figure.add_gridspec(
                2,
                1,
                height_ratios=[4.0, 1.2],
                hspace=0.08,
            )
            heatmap_axis = figure.add_subplot(grid[0, 0])
            overlay_axis = figure.add_subplot(
                grid[1, 0],
                sharex=heatmap_axis,
            )
        else:
            heatmap_axis = figure.add_subplot(111)
            overlay_axis = None

        time_unit = self.time_unit_combo.currentData()
        time_edges = self._result.time_edges(time_unit)
        cmap = colormaps[self.colormap_combo.currentData()]
        ordered_labels = self._plot_settings.ordered_raw_labels(defaults)
        ordered_index_lookup = {
            str(label): index
            for index, label in enumerate(self._result.cluster_labels)
        }
        ordered_indices = [
            ordered_index_lookup[label]
            for label in ordered_labels
            if label in ordered_index_lookup
        ]
        display_matrix = (
            matrix
            if not ordered_indices
            else np.asarray(matrix, dtype=float)[ordered_indices, :]
        )
        auto_vmin, auto_vmax = self._auto_color_limits(display_matrix)
        defaults = replace(
            defaults,
            auto_color_limit_min=auto_vmin,
            auto_color_limit_max=auto_vmax,
        )
        self._sync_plot_editor_defaults(defaults)
        norm = self._heatmap_norm(defaults)

        image = heatmap_axis.imshow(
            display_matrix,
            aspect=self._resolved_aspect(),
            origin="lower",
            interpolation="nearest",
            extent=(
                float(time_edges[0]),
                float(time_edges[-1]),
                -0.5,
                len(ordered_labels) - 0.5,
            ),
            cmap=cmap,
            norm=norm,
        )
        colorbar = figure.colorbar(image, ax=heatmap_axis, pad=0.02)
        colorbar.set_label(
            self._plot_settings.resolve_colorbar_label(defaults),
            fontsize=self._plot_settings.axis_label_font_size,
            **self._font_kwargs(),
        )
        apply_igor_inline_text_artist(
            colorbar.ax.yaxis.label,
            self._plot_settings.resolve_colorbar_label(defaults),
            default_font_size=self._plot_settings.axis_label_font_size,
            gid_prefix="heatmap-colorbar-label",
            target_axes=colorbar.ax,
        )
        colorbar.ax.tick_params(
            labelsize=self._plot_settings.tick_label_font_size
        )
        for tick_label in colorbar.ax.get_yticklabels():
            self._apply_font_to_text(tick_label)

        label_count = len(ordered_labels)
        label_step = max(
            1,
            int(
                math.ceil(
                    label_count / max(self._plot_settings.max_y_ticks, 1)
                )
            ),
        )
        tick_positions = np.arange(0, len(ordered_labels), label_step)
        y_tick_labels = [
            self._plot_settings.display_label(ordered_labels[index])
            for index in tick_positions
        ]
        rendered_y_tick_labels: list[str] = []
        composite_y_tick_labels: dict[int, str] = {}
        for tick_index, tick_label in enumerate(y_tick_labels):
            segments, has_markup = prepare_igor_inline_segments(
                tick_label,
                default_font_size=self._plot_settings.cluster_label_font_size,
            )
            if not has_markup:
                rendered_y_tick_labels.append(tick_label)
                continue
            if any(
                not math.isclose(
                    segment.font_size,
                    self._plot_settings.cluster_label_font_size,
                )
                for segment in segments
            ):
                rendered_y_tick_labels.append(" ")
                composite_y_tick_labels[tick_index] = tick_label
                continue
            rendered_y_tick_labels.append(
                igor_inline_to_mathtext(
                    tick_label,
                    default_font_size=self._plot_settings.cluster_label_font_size,
                )
            )
        heatmap_axis.set_yticks(tick_positions)
        heatmap_axis.set_yticklabels(rendered_y_tick_labels)
        heatmap_axis.set_ylabel(
            self._plot_settings.resolve_y_label(defaults),
            fontsize=self._plot_settings.axis_label_font_size,
            **self._font_kwargs(),
        )
        apply_igor_inline_text_artist(
            heatmap_axis.yaxis.label,
            self._plot_settings.resolve_y_label(defaults),
            default_font_size=self._plot_settings.axis_label_font_size,
            gid_prefix="heatmap-y-label",
            target_axes=heatmap_axis,
        )
        heatmap_axis.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        heatmap_axis.set_title(
            self._plot_settings.resolve_title(defaults),
            y=self._plot_settings.resolve_title_position_y(defaults),
            fontsize=self._plot_settings.title_font_size,
            **self._font_kwargs(),
        )
        heatmap_axis.title.set_x(
            self._plot_settings.resolve_title_position_x(defaults)
        )
        apply_igor_inline_text_artist(
            heatmap_axis.title,
            self._plot_settings.resolve_title(defaults),
            default_font_size=self._plot_settings.title_font_size,
            gid_prefix="heatmap-title",
            target_axes=heatmap_axis,
        )
        heatmap_axis.xaxis.set_major_locator(
            MaxNLocator(nbins=max(self._plot_settings.max_x_ticks, 2))
        )

        if overlay_axis is None:
            heatmap_axis.set_xlabel(
                self._plot_settings.resolve_x_label(defaults),
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            apply_igor_inline_text_artist(
                heatmap_axis.xaxis.label,
                self._plot_settings.resolve_x_label(defaults),
                default_font_size=self._plot_settings.axis_label_font_size,
                gid_prefix="heatmap-x-label",
                target_axes=heatmap_axis,
            )
        else:
            heatmap_axis.tick_params(labelbottom=False)

        self._style_heatmap_ticks(heatmap_axis)
        for tick_index, tick_label in enumerate(
            heatmap_axis.get_yticklabels()
        ):
            if tick_index not in composite_y_tick_labels:
                continue
            apply_igor_inline_text_artist(
                tick_label,
                composite_y_tick_labels[tick_index],
                default_font_size=self._plot_settings.cluster_label_font_size,
                gid_prefix=f"heatmap-y-tick-{tick_index}",
                target_axes=heatmap_axis,
            )

        if overlay_axis is not None and overlay_name is not None:
            x_values, y_values, y_label = self._result.energy_series(
                overlay_name,
                unit=time_unit,
            )
            overlay_axis.plot(
                x_values,
                y_values,
                color=OVERLAY_COLORS.get(overlay_name, "#333333"),
                linewidth=1.5,
            )
            overlay_axis.set_ylabel(
                y_label,
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            overlay_axis.set_xlabel(
                self._plot_settings.resolve_x_label(defaults),
                fontsize=self._plot_settings.axis_label_font_size,
                **self._font_kwargs(),
            )
            apply_igor_inline_text_artist(
                overlay_axis.xaxis.label,
                self._plot_settings.resolve_x_label(defaults),
                default_font_size=self._plot_settings.axis_label_font_size,
                gid_prefix="overlay-x-label",
                target_axes=overlay_axis,
            )
            overlay_axis.grid(alpha=0.25, linestyle=":")
            overlay_axis.xaxis.set_major_locator(
                MaxNLocator(nbins=max(self._plot_settings.max_x_ticks, 2))
            )
            self._style_overlay_ticks(overlay_axis)

        figure.tight_layout()

    def _current_plot_defaults(self) -> HeatmapPlotDefaults:
        time_unit = self.time_unit_combo.currentData()
        raw_labels = (
            ()
            if self._result is None
            else tuple(str(label) for label in self._result.cluster_labels)
        )
        current_colormap = self.colormap_combo.currentData()
        default_label_entries = tuple(
            (raw_label, self._format_cluster_axis_label(raw_label))
            for raw_label in raw_labels
        )
        return HeatmapPlotDefaults(
            title=(
                "Time-Binned Cluster Distribution "
                f"({DISPLAY_MODE_LABELS[self._display_mode()]})"
            ),
            x_label=f"Time ({time_unit})",
            y_label="Cluster label",
            colorbar_label=DISPLAY_MODE_COLORBAR_LABELS[self._display_mode()],
            default_x_axis_unit_name=(
                "" if time_unit is None else str(time_unit)
            ),
            available_x_axis_unit_names=("fs", "ps"),
            default_colormap_name=(
                "" if current_colormap is None else str(current_colormap)
            ),
            available_colormap_names=tuple(PLOT_COLORMAPS),
            raw_cluster_labels=raw_labels,
            default_label_entries=default_label_entries,
        )

    def _resolved_aspect(self) -> str | float:
        if self._plot_settings.aspect_mode == "equal":
            return "equal"
        if self._plot_settings.aspect_mode == "custom":
            return float(self._plot_settings.custom_aspect)
        return "auto"

    def _font_kwargs(self) -> dict[str, str]:
        if not self._plot_settings.font_family:
            return {}
        return {"fontfamily": self._plot_settings.font_family}

    @staticmethod
    def _format_cluster_axis_label(label: str) -> str:
        return format_stoich_for_axis(label)

    def _apply_font_to_text(self, text_artist) -> None:
        if self._plot_settings.font_family:
            text_artist.set_fontfamily(self._plot_settings.font_family)

    def _style_heatmap_ticks(self, axis) -> None:
        axis.tick_params(
            axis="x",
            labelsize=self._plot_settings.tick_label_font_size,
            labelrotation=self._plot_settings.x_tick_rotation,
        )
        axis.tick_params(
            axis="y",
            labelsize=self._plot_settings.cluster_label_font_size,
            labelrotation=self._plot_settings.y_tick_rotation,
        )
        if (
            self._plot_settings.show_minor_x_ticks
            or self._plot_settings.show_minor_y_ticks
        ):
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=self._plot_settings.show_minor_x_ticks,
            top=False,
        )
        axis.tick_params(
            axis="y",
            which="minor",
            left=self._plot_settings.show_minor_y_ticks,
            right=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_font_to_text(tick_label)

    def _style_overlay_ticks(self, axis) -> None:
        axis.tick_params(
            axis="both",
            labelsize=self._plot_settings.tick_label_font_size,
        )
        axis.tick_params(
            axis="x",
            labelrotation=self._plot_settings.x_tick_rotation,
        )
        if self._plot_settings.show_minor_x_ticks:
            axis.minorticks_on()
        else:
            axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="minor",
            bottom=self._plot_settings.show_minor_x_ticks,
            top=False,
        )
        for tick_label in axis.get_xticklabels():
            self._apply_font_to_text(tick_label)
        for tick_label in axis.get_yticklabels():
            self._apply_font_to_text(tick_label)

    def _display_mode(self) -> str:
        value = self.display_mode_combo.currentData()
        return "fraction" if value is None else str(value)

    def _ensure_valid_quantiles(self) -> None:
        lower = self.lower_quantile_spin.value()
        upper = self.upper_quantile_spin.value()
        if lower >= upper:
            self.upper_quantile_spin.blockSignals(True)
            self.upper_quantile_spin.setValue(min(lower + 0.05, 1.0))
            self.upper_quantile_spin.blockSignals(False)
            lower = self.lower_quantile_spin.value()
            upper = self.upper_quantile_spin.value()
        if lower >= upper:
            self.lower_quantile_spin.blockSignals(True)
            self.lower_quantile_spin.setValue(max(upper - 0.05, 0.0))
            self.lower_quantile_spin.blockSignals(False)

    def _on_quantile_changed(self) -> None:
        self._ensure_valid_quantiles()
        self.refresh_plot()

    def _heatmap_norm(
        self,
        defaults: HeatmapPlotDefaults,
    ) -> mcolors.Normalize:
        vmin = float(self._plot_settings.resolve_color_limit_min(defaults))
        vmax = float(self._plot_settings.resolve_color_limit_max(defaults))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _auto_color_limits(self, matrix: np.ndarray) -> tuple[float, float]:
        values = np.asarray(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 1.0)

        positive = finite[finite > 0.0]
        if positive.size:
            finite = positive

        lower_q = float(self.lower_quantile_spin.value())
        upper_q = float(self.upper_quantile_spin.value())
        vmin = float(np.quantile(finite, lower_q))
        vmax = float(np.quantile(finite, upper_q))
        if vmax <= vmin:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return (vmin, vmax)

    @staticmethod
    def _draw_placeholder(axis, message: str) -> None:
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)


__all__ = ["ClusterDynamicsPlotPanel"]
