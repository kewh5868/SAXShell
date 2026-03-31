from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster.clusternetwork import stoichiometry_label
from saxshell.clusterdynamicsml.workflow import (
    ClusterDynamicsMLResult,
    _resolved_population_weights,
)
from saxshell.saxs.debye.profiles import scan_structure_element_counts
from saxshell.saxs.project_manager.prior_plot import (
    list_secondary_filter_elements,
    plot_md_prior_histogram,
)
from saxshell.saxs.stoichiometry import parse_stoich_label

_EXPERIMENTAL_COLOR = "#111111"
_OBSERVED_MODEL_COLOR = "#1f77b4"
_COMBINED_MODEL_COLOR = "#ff7f0e"
_HISTOGRAM_CMAP = "summer"
_HISTOGRAM_MODES = (
    ("Structure Fraction", "structure_fraction"),
    ("Atom Fraction", "atom_fraction"),
    ("Solvent Sort - Structure Fraction", "solvent_sort_structure_fraction"),
    ("Solvent Sort - Atom Fraction", "solvent_sort_atom_fraction"),
)
_STRUCTURE_FILE_SUFFIXES = {".xyz", ".pdb"}


class ClusterDynamicsMLHistogramPanel(QWidget):
    """Plot Project Setup-style stoichiometry histograms for Cluster
    Dynamics ML results."""

    def __init__(
        self,
        *,
        include_predictions: bool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fixed_include_predictions = include_predictions
        self._include_predictions = (
            False if include_predictions is None else bool(include_predictions)
        )
        self._result: ClusterDynamicsMLResult | None = None
        self._histogram_payloads: dict[bool, dict[str, object] | None] = {
            False: None,
            True: None,
        }
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        self.population_label = QLabel("Population set")
        controls.addWidget(self.population_label)
        self.population_combo = QComboBox()
        self.population_combo.addItem("Observed", False)
        self.population_combo.addItem("Observed + Predicted Structures", True)
        self.population_combo.currentIndexChanged.connect(
            self._on_population_changed
        )
        controls.addWidget(self.population_combo)

        controls.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        for label, mode in _HISTOGRAM_MODES:
            self.mode_combo.addItem(label, mode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self.mode_combo)

        self.secondary_label = QLabel("Secondary atom")
        controls.addWidget(self.secondary_label)
        self.secondary_combo = QComboBox()
        self.secondary_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_plot()
        )
        controls.addWidget(self.secondary_combo)
        controls.addStretch(1)
        layout.addWidget(controls_widget)

        self.figure = Figure(figsize=(9.2, 7.0))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)
        self._update_population_control_state()

    def set_result(self, result: ClusterDynamicsMLResult | None) -> None:
        self._result = result
        if result is None:
            self._histogram_payloads = {False: None, True: None}
        else:
            self._histogram_payloads = {
                False: _build_population_histogram_payload(
                    result,
                    include_predictions=False,
                ),
                True: _build_population_histogram_payload(
                    result,
                    include_predictions=True,
                ),
            }
        self._refresh_secondary_elements()
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self.figure.clear()
        if self._result is None:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Run the prediction workflow to plot cluster stoichiometry\n"
                "histograms for the current result.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        include_predictions = self._current_include_predictions()
        histogram_payload = self._histogram_payloads.get(include_predictions)
        if histogram_payload is None:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "No stoichiometry histogram data are available for the current result.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        axis = self.figure.add_subplot(111)
        mode = self._mode()
        secondary_element = self._secondary_element()
        if mode.startswith("solvent_sort") and secondary_element is None:
            axis.text(
                0.5,
                0.5,
                "No secondary atom counts are available for solvent-sort histograms.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        try:
            plot_md_prior_histogram(
                histogram_payload,
                mode=mode,
                secondary_element=secondary_element,
                cmap=_HISTOGRAM_CMAP,
                ax=axis,
            )
        except Exception as exc:
            axis.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.canvas.draw_idle()
            return

        summary_label = (
            "Observed + Predicted Structures populations"
            if include_predictions
            else "Observed populations"
        )
        self.figure.suptitle(summary_label, y=0.995)
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        self.canvas.draw_idle()

    def _mode(self) -> str:
        current = self.mode_combo.currentData()
        return "structure_fraction" if current is None else str(current)

    def _secondary_element(self) -> str | None:
        text = self.secondary_combo.currentText().strip()
        return text or None

    def _current_include_predictions(self) -> bool:
        if self._fixed_include_predictions is not None:
            return bool(self._fixed_include_predictions)
        current = self.population_combo.currentData()
        return bool(self._include_predictions if current is None else current)

    def _on_mode_changed(self) -> None:
        self._update_secondary_control_state()
        self.refresh_plot()

    def _on_population_changed(self) -> None:
        self._include_predictions = self._current_include_predictions()
        self._refresh_secondary_elements()
        self.refresh_plot()

    def _refresh_secondary_elements(self) -> None:
        current = self._secondary_element()
        elements = (
            []
            if self._histogram_payloads.get(
                self._current_include_predictions()
            )
            is None
            else list_secondary_filter_elements(
                self._histogram_payloads[self._current_include_predictions()]
            )
        )
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        self.secondary_combo.addItems(elements)
        if current and current in elements:
            self.secondary_combo.setCurrentText(current)
        elif elements:
            self.secondary_combo.setCurrentIndex(0)
        self.secondary_combo.blockSignals(False)
        self._update_secondary_control_state()

    def _update_secondary_control_state(self) -> None:
        needs_secondary = self._mode().startswith("solvent_sort")
        has_options = self.secondary_combo.count() > 0
        enabled = needs_secondary and has_options
        self.secondary_label.setEnabled(enabled)
        self.secondary_combo.setEnabled(enabled)

    def _update_population_control_state(self) -> None:
        visible = self._fixed_include_predictions is None
        self.population_label.setVisible(visible)
        self.population_combo.setVisible(visible)


class ClusterDynamicsMLPlotPanel(QWidget):
    """Plot observed-only, Predicted Structures, and component SAXS
    traces."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: ClusterDynamicsMLResult | None = None
        self._legend_line_map: dict[object, object] = {}
        self._legend_handle_lookup: dict[str, object] = {}
        self._trace_line_lookup: dict[str, object] = {}
        self._trace_visibility: dict[str, bool] = {}
        self._component_trace_keys: list[str] = []
        self._observed_trace_keys: list[str] = []
        self._predicted_trace_keys: list[str] = []
        self._build_ui()
        self.refresh_plot()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_widget = QWidget()
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        self.log_x_checkbox = QCheckBox("Log X")
        self.log_x_checkbox.setChecked(True)
        self.log_x_checkbox.toggled.connect(self.refresh_plot)
        controls.addWidget(self.log_x_checkbox)
        self.log_y_checkbox = QCheckBox("Log Y")
        self.log_y_checkbox.setChecked(True)
        self.log_y_checkbox.toggled.connect(self.refresh_plot)
        controls.addWidget(self.log_y_checkbox)
        self.legend_toggle_button = QPushButton("Legend")
        self.legend_toggle_button.setCheckable(True)
        self.legend_toggle_button.setChecked(True)
        self.legend_toggle_button.toggled.connect(self.refresh_plot)
        controls.addWidget(self.legend_toggle_button)
        self.model_range_button = QPushButton("Autoscale to Model Range")
        self.model_range_button.setCheckable(True)
        self.model_range_button.toggled.connect(self.refresh_plot)
        controls.addWidget(self.model_range_button)
        self.observed_traces_button = QPushButton("Hide Observed Traces")
        self.observed_traces_button.clicked.connect(
            self._toggle_observed_traces
        )
        controls.addWidget(self.observed_traces_button)
        self.predicted_traces_button = QPushButton("Hide Predicted Traces")
        self.predicted_traces_button.clicked.connect(
            self._toggle_predicted_traces
        )
        controls.addWidget(self.predicted_traces_button)
        self.component_traces_button = QPushButton("Hide Component Traces")
        self.component_traces_button.clicked.connect(
            self._toggle_all_component_traces
        )
        controls.addWidget(self.component_traces_button)
        controls.addStretch(1)
        layout.addWidget(controls_widget)

        self.figure = Figure(figsize=(9.2, 7.0))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self._handle_legend_pick)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)
        self._update_trace_control_state()

    def set_result(self, result: ClusterDynamicsMLResult | None) -> None:
        self._result = result
        self.refresh_plot()

    def refresh_plot(self) -> None:
        for axis in list(self.figure.axes):
            try:
                axis.set_xscale("linear")
                axis.set_yscale("linear")
            except Exception:
                continue
        self.figure.clear()
        self._legend_line_map.clear()
        self._legend_handle_lookup.clear()
        self._trace_line_lookup.clear()
        self._component_trace_keys = []
        self._observed_trace_keys = []
        self._predicted_trace_keys = []
        if self._result is None:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Run the prediction workflow to plot the SAXS\n"
                "form-factor comparison.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self._update_trace_control_state()
            self.canvas.draw_idle()
            return

        experimental_axis = self.figure.add_subplot(111)
        plotted_lines: list[object] = []

        observed_model = _build_saxs_model(
            self._result,
            include_predictions=False,
        )
        combined_model = _build_saxs_model(
            self._result,
            include_predictions=True,
        )
        if observed_model is None and combined_model is None:
            experimental_axis.text(
                0.5,
                0.5,
                "No SAXS form-factor model is available for the current result.",
                ha="center",
                va="center",
                transform=experimental_axis.transAxes,
            )
            experimental_axis.set_axis_off()
        else:
            has_experimental_trace = any(
                model is not None
                and model["experimental_intensity"] is not None
                for model in (observed_model, combined_model)
            )
            if has_experimental_trace:
                model_axis = experimental_axis.twinx()
            else:
                model_axis = experimental_axis
            plotted_any = False
            experimental_model = (
                observed_model
                if observed_model is not None
                and observed_model["experimental_intensity"] is not None
                else combined_model
            )
            if (
                experimental_model is not None
                and experimental_model["experimental_intensity"] is not None
            ):
                line = _plot_saxs_trace_line(
                    experimental_axis,
                    q_values=experimental_model["q_values"],
                    intensity=experimental_model["experimental_intensity"],
                    color=_EXPERIMENTAL_COLOR,
                    linewidth=1.2,
                    alpha=0.75,
                    label="experimental",
                    gid="experimental",
                    visible=self._trace_visibility.get("experimental", True),
                )
                if line is not None:
                    plotted_lines.append(line)
                    self._trace_line_lookup["experimental"] = line
                    self._trace_visibility.setdefault("experimental", True)

            if observed_model is not None:
                line = _plot_saxs_trace_line(
                    model_axis,
                    q_values=observed_model["q_values"],
                    intensity=observed_model["model_intensity"],
                    color=_OBSERVED_MODEL_COLOR,
                    linewidth=1.4,
                    linestyle="--",
                    label="observed-only model",
                    gid="observed-only model",
                    visible=self._trace_visibility.get(
                        "observed-only model", True
                    ),
                )
                if line is not None:
                    plotted_any = True
                    plotted_lines.append(line)
                    self._trace_line_lookup["observed-only model"] = line
                    self._trace_visibility.setdefault(
                        "observed-only model",
                        True,
                    )
                    self._observed_trace_keys.append("observed-only model")
            if combined_model is not None:
                line = _plot_saxs_trace_line(
                    model_axis,
                    q_values=combined_model["q_values"],
                    intensity=combined_model["model_intensity"],
                    color=_COMBINED_MODEL_COLOR,
                    linewidth=1.8,
                    label="observed + predicted structures model",
                    gid="observed + predicted structures model",
                    visible=self._trace_visibility.get(
                        "observed + predicted structures model",
                        True,
                    ),
                )
                if line is not None:
                    plotted_any = True
                    plotted_lines.append(line)
                    self._trace_line_lookup[
                        "observed + predicted structures model"
                    ] = line
                    self._trace_visibility.setdefault(
                        "observed + predicted structures model",
                        True,
                    )
                    self._predicted_trace_keys.append(
                        "observed + predicted structures model"
                    )

            for component_trace in _build_saxs_component_traces(self._result):
                line = _plot_saxs_trace_line(
                    model_axis,
                    q_values=component_trace["q_values"],
                    intensity=component_trace["intensity"],
                    color=str(component_trace["color"]),
                    linewidth=float(component_trace["linewidth"]),
                    linestyle=str(component_trace["linestyle"]),
                    alpha=float(component_trace["alpha"]),
                    label=str(component_trace["label"]),
                    gid=str(component_trace["key"]),
                    visible=self._trace_visibility.get(
                        str(component_trace["key"]),
                        False,
                    ),
                )
                if line is None:
                    continue
                plotted_any = True
                component_key = str(component_trace["key"])
                self._trace_line_lookup[component_key] = line
                self._trace_visibility.setdefault(
                    component_key, line.get_visible()
                )
                self._component_trace_keys.append(component_key)
                if str(component_trace["source"]) == "predicted":
                    self._predicted_trace_keys.append(component_key)
                else:
                    self._observed_trace_keys.append(component_key)
                plotted_lines.append(line)

            if not plotted_any:
                experimental_axis.text(
                    0.5,
                    0.5,
                    "No positive SAXS values are available for the current result.",
                    ha="center",
                    va="center",
                    transform=experimental_axis.transAxes,
                )
                experimental_axis.set_axis_off()
                if (
                    has_experimental_trace
                    and model_axis is not experimental_axis
                ):
                    model_axis.set_axis_off()
            else:
                self._apply_saxs_axis_style(
                    experimental_axis,
                    is_model_axis=False,
                    has_separate_model_axis=model_axis
                    is not experimental_axis,
                )
                if model_axis is not experimental_axis:
                    self._apply_saxs_axis_style(
                        model_axis,
                        is_model_axis=True,
                        has_separate_model_axis=True,
                    )
                title = "SAXS Form Factor Models"
                title_bits: list[str] = []
                if (
                    observed_model is not None
                    and observed_model["rmse"] is not None
                ):
                    title_bits.append(
                        f"observed-only RMSE {observed_model['rmse']:.4g}"
                    )
                if (
                    combined_model is not None
                    and combined_model["rmse"] is not None
                ):
                    title_bits.append(
                        "with predicted structures RMSE "
                        f"{combined_model['rmse']:.4g}"
                    )
                if title_bits:
                    title += " (" + "; ".join(title_bits) + ")"
                experimental_axis.set_title(title)
                if self.legend_toggle_button.isChecked():
                    self._build_interactive_legend(
                        experimental_axis, plotted_lines
                    )
                self._refresh_axes()

        self._update_trace_control_state()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _build_interactive_legend(self, axis, lines: list[object]) -> None:
        legend_columns = max(1, int(np.ceil(len(lines) / 5.0)))
        legend = axis.legend(
            lines,
            [line.get_label() for line in lines],
            fontsize="small",
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            borderaxespad=0.3,
            framealpha=0.9,
            ncols=legend_columns,
            columnspacing=0.9,
            handlelength=1.5,
        )
        if legend is None:
            return
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = getattr(legend, "legendHandles", [])
        for legend_handle, original_line in zip(legend_handles, lines):
            if hasattr(legend_handle, "set_picker"):
                legend_handle.set_picker(True)
                legend_handle.set_pickradius(6)
            if hasattr(legend_handle, "set_color"):
                legend_handle.set_color(original_line.get_color())
            if hasattr(legend_handle, "set_linestyle"):
                legend_handle.set_linestyle(original_line.get_linestyle())
            if hasattr(legend_handle, "set_linewidth"):
                legend_handle.set_linewidth(original_line.get_linewidth())
            legend_handle.set_alpha(
                1.0 if original_line.get_visible() else 0.25
            )
            self._legend_line_map[legend_handle] = original_line
            line_key = str(original_line.get_gid() or "").strip()
            if line_key:
                self._legend_handle_lookup[line_key] = legend_handle

    def _handle_legend_pick(self, event) -> None:
        original_line = self._legend_line_map.get(event.artist)
        if original_line is None:
            return
        is_visible = not original_line.get_visible()
        original_line.set_visible(is_visible)
        line_key = str(original_line.get_gid() or "").strip()
        if line_key:
            self._trace_visibility[line_key] = is_visible
        if hasattr(event.artist, "set_alpha"):
            event.artist.set_alpha(1.0 if is_visible else 0.25)
        self._update_trace_control_state()
        self._refresh_axes()
        self.canvas.draw_idle()

    def _toggle_all_component_traces(self) -> None:
        if not self._component_trace_keys:
            return
        self._toggle_trace_keys(self._component_trace_keys)
        self._update_trace_control_state()
        self._refresh_axes()
        self.canvas.draw_idle()

    def _toggle_observed_traces(self) -> None:
        if not self._observed_trace_keys:
            return
        self._toggle_trace_keys(self._observed_trace_keys)
        self._update_trace_control_state()
        self._refresh_axes()
        self.canvas.draw_idle()

    def _toggle_predicted_traces(self) -> None:
        if not self._predicted_trace_keys:
            return
        self._toggle_trace_keys(self._predicted_trace_keys)
        self._update_trace_control_state()
        self._refresh_axes()
        self.canvas.draw_idle()

    def _toggle_trace_keys(self, trace_keys: list[str]) -> None:
        any_visible = any(
            self._trace_visibility.get(trace_key, False)
            for trace_key in trace_keys
        )
        target_visible = not any_visible
        for trace_key in trace_keys:
            self._trace_visibility[trace_key] = target_visible
            line = self._trace_line_lookup.get(trace_key)
            if line is not None:
                line.set_visible(target_visible)
            legend_line = self._legend_handle_lookup.get(trace_key)
            if legend_line is not None and hasattr(legend_line, "set_alpha"):
                legend_line.set_alpha(1.0 if target_visible else 0.25)

    def _update_trace_control_state(self) -> None:
        has_components = bool(self._component_trace_keys)
        any_visible = any(
            self._trace_visibility.get(component_key, False)
            for component_key in self._component_trace_keys
        )
        has_observed = bool(self._observed_trace_keys)
        observed_visible = any(
            self._trace_visibility.get(trace_key, False)
            for trace_key in self._observed_trace_keys
        )
        has_predicted = bool(self._predicted_trace_keys)
        predicted_visible = any(
            self._trace_visibility.get(trace_key, False)
            for trace_key in self._predicted_trace_keys
        )
        self.observed_traces_button.setEnabled(has_observed)
        self.observed_traces_button.setText(
            "Hide Observed Traces"
            if observed_visible
            else "Show Observed Traces"
        )
        self.predicted_traces_button.setEnabled(has_predicted)
        self.predicted_traces_button.setText(
            "Hide Predicted Traces"
            if predicted_visible
            else "Show Predicted Traces"
        )
        self.component_traces_button.setEnabled(has_components)
        self.component_traces_button.setText(
            "Hide Component Traces" if any_visible else "Show Component Traces"
        )

    def _apply_saxs_axis_style(
        self,
        axis,
        *,
        is_model_axis: bool,
        has_separate_model_axis: bool,
    ) -> None:
        axis.set_xscale("log" if self.log_x_checkbox.isChecked() else "linear")
        axis.set_yscale("log" if self.log_y_checkbox.isChecked() else "linear")
        if not is_model_axis or not has_separate_model_axis:
            axis.set_xlabel("q (Å⁻¹)")
        if not is_model_axis:
            axis.set_ylabel("Intensity (arb. units)")
        elif has_separate_model_axis:
            axis.set_ylabel("Model Intensity (arb. units)")

    def _refresh_axes(self) -> None:
        axes = self.figure.axes
        if not axes:
            return
        for axis in axes:
            if not hasattr(axis, "relim"):
                continue
            try:
                axis.relim(visible_only=True)
                axis.autoscale_view()
            except Exception:
                continue
        if len(axes) == 2:
            experimental_axis, model_axis = axes
            if self.model_range_button.isChecked():
                self._autoscale_to_model_range(experimental_axis, model_axis)
            else:
                self._normalize_model_axis(experimental_axis, model_axis)
        elif self.model_range_button.isChecked():
            self._autoscale_to_model_range(None, axes[0])

    def _normalize_model_axis(self, experimental_axis, model_axis) -> None:
        experimental_lines = [
            line
            for line in experimental_axis.get_lines()
            if line.get_visible()
        ]
        model_lines = [
            line for line in model_axis.get_lines() if line.get_visible()
        ]
        if not experimental_lines or not model_lines:
            return
        experimental_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in experimental_lines
            ]
        )
        experimental_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in experimental_lines
            ]
        )
        model_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in model_lines
            ]
        )
        model_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in model_lines
            ]
        )
        overlap_mask = (experimental_q >= float(np.nanmin(model_q))) & (
            experimental_q <= float(np.nanmax(model_q))
        )
        if np.any(overlap_mask):
            experimental_i = experimental_i[overlap_mask]
        experimental_i = experimental_i[np.isfinite(experimental_i)]
        model_i = model_i[np.isfinite(model_i)]
        if self.log_y_checkbox.isChecked():
            experimental_i = experimental_i[experimental_i > 0.0]
            model_i = model_i[model_i > 0.0]
        if experimental_i.size == 0 or model_i.size == 0:
            return
        model_axis.set_ylim(
            self._aligned_y_limits(
                experimental_axis.get_ylim(),
                float(np.nanmin(experimental_i)),
                float(np.nanmax(experimental_i)),
                float(np.nanmin(model_i)),
                float(np.nanmax(model_i)),
                log_scale=self.log_y_checkbox.isChecked(),
            )
        )

    def _autoscale_to_model_range(self, experimental_axis, model_axis) -> None:
        model_lines = [
            line for line in model_axis.get_lines() if line.get_visible()
        ]
        if not model_lines:
            return
        model_q_values = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in model_lines
            ]
        )
        model_q_values = model_q_values[np.isfinite(model_q_values)]
        if model_q_values.size == 0:
            return
        q_min = float(np.nanmin(model_q_values))
        q_max = float(np.nanmax(model_q_values))
        model_axis.set_xlim(q_min, q_max)
        if experimental_axis is not None:
            experimental_axis.set_xlim(q_min, q_max)
            self._autoscale_axis_y(experimental_axis, q_min, q_max)
            self._normalize_model_axis(experimental_axis, model_axis)
            return
        self._autoscale_axis_y(model_axis, q_min, q_max)

    def _autoscale_axis_y(
        self,
        axis,
        q_min: float,
        q_max: float,
    ) -> None:
        y_segments: list[np.ndarray] = []
        log_scale = self.log_y_checkbox.isChecked()
        for line in axis.get_lines():
            if not line.get_visible():
                continue
            x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
            y_data = np.asarray(line.get_ydata(orig=False), dtype=float)
            mask = (
                np.isfinite(x_data)
                & np.isfinite(y_data)
                & (x_data >= q_min)
                & (x_data <= q_max)
            )
            if log_scale:
                mask &= y_data > 0.0
            if np.any(mask):
                y_segments.append(y_data[mask])
        if not y_segments:
            return
        y_values = np.concatenate(y_segments)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            padding = max(abs(y_min) * 0.05, 1e-12)
            axis.set_ylim(y_min - padding, y_max + padding)
            return
        if log_scale:
            axis.set_ylim(y_min / 1.15, y_max * 1.15)
        else:
            padding = 0.05 * (y_max - y_min)
            axis.set_ylim(y_min - padding, y_max + padding)

    @staticmethod
    def _aligned_y_limits(
        left_limits: tuple[float, float],
        experimental_min: float,
        experimental_max: float,
        model_min: float,
        model_max: float,
        *,
        log_scale: bool,
    ) -> tuple[float, float]:
        if log_scale:
            if (
                min(
                    left_limits[0],
                    left_limits[1],
                    experimental_min,
                    experimental_max,
                    model_min,
                    model_max,
                )
                <= 0.0
            ):
                log_scale = False
        if not log_scale:
            left_low, left_high = left_limits
            exp_low, exp_high = sorted((experimental_min, experimental_max))
            model_low, model_high = sorted((model_min, model_max))
            if np.isclose(left_high, left_low) or np.isclose(
                exp_high, exp_low
            ):
                padding = max(abs(model_low) * 0.1, 1e-12)
                return model_low - padding, model_high + padding
            p0 = (exp_low - left_low) / (left_high - left_low)
            p1 = (exp_high - left_low) / (left_high - left_low)
            if np.isclose(p1, p0):
                padding = max(abs(model_low) * 0.1, 1e-12)
                return model_low - padding, model_high + padding
            delta = (model_high - model_low) / (p1 - p0)
            right_low = model_low - p0 * delta
            right_high = right_low + delta
            return right_low, right_high

        left_logs = np.log10(np.asarray(left_limits, dtype=float))
        experimental_logs = np.log10(
            np.asarray(
                sorted((experimental_min, experimental_max)),
                dtype=float,
            )
        )
        model_logs = np.log10(
            np.asarray(sorted((model_min, model_max)), dtype=float)
        )
        if np.isclose(left_logs[1], left_logs[0]) or np.isclose(
            experimental_logs[1],
            experimental_logs[0],
        ):
            return model_min / 1.2, model_max * 1.2
        p0 = (experimental_logs[0] - left_logs[0]) / (
            left_logs[1] - left_logs[0]
        )
        p1 = (experimental_logs[1] - left_logs[0]) / (
            left_logs[1] - left_logs[0]
        )
        if np.isclose(p1, p0):
            return model_min / 1.2, model_max * 1.2
        delta = (model_logs[1] - model_logs[0]) / (p1 - p0)
        right_low_log = model_logs[0] - p0 * delta
        right_high_log = right_low_log + delta
        return 10**right_low_log, 10**right_high_log


def _plot_saxs_trace_line(
    axis,
    *,
    q_values: np.ndarray,
    intensity: np.ndarray,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    gid: str | None = None,
    visible: bool = True,
):
    q_array = np.asarray(q_values, dtype=float)
    intensity_array = np.asarray(intensity, dtype=float)
    mask = np.isfinite(q_array) & np.isfinite(intensity_array)
    mask &= q_array > 0.0
    mask &= intensity_array > 0.0
    if not np.any(mask):
        return None
    (line,) = axis.plot(
        q_array[mask],
        intensity_array[mask],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
        visible=visible,
    )
    if gid:
        line.set_gid(gid)
    return line


def _build_saxs_component_traces(
    result: ClusterDynamicsMLResult,
) -> list[dict[str, object]]:
    comparison = result.saxs_comparison
    if comparison is None:
        return []
    component_entries = [
        entry
        for entry in comparison.component_weights
        if entry.profile_path is not None
        and Path(entry.profile_path).is_file()
    ]
    if not component_entries:
        return []

    component_colors = _component_trace_colors(len(component_entries))
    scale_factor = (
        float(comparison.scale_factor)
        if float(comparison.scale_factor) > 0.0
        else 1.0
    )
    traces: list[dict[str, object]] = []
    for index, entry in enumerate(component_entries):
        try:
            raw_data = np.loadtxt(entry.profile_path, comments="#")
        except Exception:
            continue
        if raw_data.size == 0:
            continue
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        q_values = np.asarray(raw_data[:, 0], dtype=float)
        intensity = np.asarray(raw_data[:, 1], dtype=float)
        weighted_intensity = (
            intensity * max(float(entry.weight), 0.0) * scale_factor
        )
        source = str(entry.source)
        color = component_colors[index]
        if source == "predicted":
            label = f"predicted structure component: {entry.label}"
            linestyle = "-"
        else:
            label = f"observed component: {entry.label}"
            linestyle = ":"
        traces.append(
            {
                "key": f"component:{index}:{source}:{entry.label}",
                "source": source,
                "label": label,
                "q_values": q_values,
                "intensity": weighted_intensity,
                "color": color,
                "linestyle": linestyle,
                "linewidth": 1.0,
                "alpha": 0.85 if source == "predicted" else 0.65,
            }
        )
    return traces


def _component_trace_colors(total: int) -> list[str]:
    if total <= 0:
        return []
    default_cycle = list(
        rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    )
    return [
        default_cycle[index % len(default_cycle)] for index in range(total)
    ]


def _distribution_entries(
    result: ClusterDynamicsMLResult,
    *,
    include_predictions: bool,
) -> list[dict[str, float | None]]:
    entries: list[dict[str, float | None]] = []
    observed_weights, predicted_weights = _resolved_population_weights(
        result.training_observations,
        result.predictions,
        frame_timestep_fs=float(
            result.dynamics_result.preview.frame_timestep_fs
        ),
    )
    for row, weight in zip(
        result.training_observations,
        observed_weights,
        strict=False,
    ):
        if weight <= 0.0:
            continue
        entries.append(
            {
                "node_count": float(row.node_count),
                "mean_lifetime_fs": row.mean_lifetime_fs,
                "mean_count_per_frame": float(row.mean_count_per_frame),
                "mean_max_radius": float(row.mean_max_radius),
                "weight": weight,
            }
        )
    if include_predictions:
        for item, weight in zip(
            result.predictions,
            predicted_weights,
            strict=False,
        ):
            if weight <= 0.0:
                continue
            entries.append(
                {
                    "node_count": float(item.target_node_count),
                    "mean_lifetime_fs": float(item.predicted_mean_lifetime_fs),
                    "mean_count_per_frame": float(
                        item.predicted_mean_count_per_frame
                    ),
                    "mean_max_radius": float(item.predicted_mean_max_radius),
                    "weight": weight,
                }
            )
    total_weight = sum(float(entry["weight"]) for entry in entries)
    if total_weight <= 0.0:
        return []
    for entry in entries:
        entry["normalized_weight"] = float(entry["weight"]) / total_weight
    return entries


def _build_population_histogram_payload(
    result: ClusterDynamicsMLResult,
    *,
    include_predictions: bool,
) -> dict[str, object] | None:
    structures: dict[str, dict[str, dict[str, object]]] = {}
    secondary_elements: set[str] = set()
    total_population = 0.0
    observed_label_elements = _observed_label_elements(result)
    observed_weights, predicted_weights = _resolved_population_weights(
        result.training_observations,
        result.predictions,
        frame_timestep_fs=float(
            result.dynamics_result.preview.frame_timestep_fs
        ),
    )

    for observation, base_count in zip(
        result.training_observations,
        observed_weights,
        strict=False,
    ):
        base_count = max(float(base_count), 0.0)
        if base_count <= 0.0:
            continue
        raw_payloads = _observed_structure_payloads(observation)
        if not raw_payloads:
            continue
        scaled_payloads = _scale_motif_payloads(
            raw_payloads, base_count=base_count
        )
        if not scaled_payloads:
            continue
        structures.setdefault(observation.label, {}).update(scaled_payloads)
        total_population += sum(
            float(payload["count"]) for payload in scaled_payloads.values()
        )
        secondary_elements.update(
            _secondary_elements_from_payloads(scaled_payloads)
        )

    if include_predictions:
        for prediction, base_count in zip(
            result.predictions,
            predicted_weights,
            strict=False,
        ):
            base_count = max(float(base_count), 0.0)
            if base_count <= 0.0:
                continue
            motif_name = f"predicted_rank_{int(prediction.rank):02d}"
            structure_label = _predicted_structure_label(
                prediction,
                observed_label_elements=observed_label_elements,
            )
            motif_payload = _predicted_structure_payload(
                prediction,
                count=base_count,
                structure_label=structure_label,
            )
            structures.setdefault(structure_label, {})[
                motif_name
            ] = motif_payload
            total_population += float(motif_payload["count"])
            secondary_elements.update(
                _secondary_elements_from_payloads({motif_name: motif_payload})
            )

    if total_population <= 0.0 or not structures:
        return None

    if secondary_elements:
        for motif_payloads in structures.values():
            for payload in motif_payloads.values():
                distributions = dict(
                    payload.get("secondary_atom_distributions", {})
                )
                count = float(payload.get("count", 0.0) or 0.0)
                for element in secondary_elements:
                    distributions.setdefault(element, {"0": count})
                payload["secondary_atom_distributions"] = distributions

    for motif_payloads in structures.values():
        for payload in motif_payloads.values():
            payload["weight"] = float(payload["count"]) / total_population

    return {
        "origin": "clusterdynamicsml",
        "total_files": float(total_population),
        "available_elements": sorted(secondary_elements),
        "structures": structures,
    }


def _observed_structure_payloads(
    observation,
) -> dict[str, dict[str, object]]:
    structure_dir = Path(observation.structure_dir).expanduser()
    raw_payloads: dict[str, dict[str, object]] = {}

    if structure_dir.is_dir():
        motif_dirs = sorted(
            path
            for path in structure_dir.iterdir()
            if path.is_dir() and path.name.startswith("motif_")
        )
        if motif_dirs:
            for motif_dir in motif_dirs:
                payload = _structure_payload_from_files(
                    _structure_files_in_dir(motif_dir),
                    label=observation.label,
                )
                if payload is not None:
                    raw_payloads[motif_dir.name] = payload
        else:
            payload = _structure_payload_from_files(
                _structure_files_in_dir(structure_dir),
                label=observation.label,
            )
            if payload is not None:
                raw_payloads["no_motif"] = payload

    representative_path = (
        None
        if observation.representative_path is None
        else Path(observation.representative_path).expanduser()
    )
    if (
        not raw_payloads
        and representative_path is not None
        and representative_path.is_file()
    ):
        payload = _structure_payload_from_files(
            [representative_path],
            label=observation.label,
        )
        if payload is not None:
            raw_payloads["no_motif"] = payload

    return raw_payloads


def _structure_files_in_dir(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _STRUCTURE_FILE_SUFFIXES
    )


def _structure_payload_from_files(
    file_paths: list[Path],
    *,
    label: str,
) -> dict[str, object] | None:
    element_counts = []
    for file_path in file_paths:
        try:
            element_counts.append(scan_structure_element_counts(file_path))
        except Exception:
            continue
    if not element_counts:
        return None

    label_elements = set(parse_stoich_label(label).keys())
    secondary_elements = sorted(
        {
            element
            for counts in element_counts
            for element in counts
            if element not in label_elements
        }
    )
    secondary_distributions: dict[str, dict[str, float]] = {}
    for element in secondary_elements:
        buckets: Counter[str] = Counter()
        for counts in element_counts:
            buckets[str(int(counts.get(element, 0)))] += 1
        secondary_distributions[element] = {
            segment: float(buckets[segment])
            for segment in sorted(buckets, key=lambda value: int(value))
        }

    return {
        "count": float(len(element_counts)),
        "weight": 0.0,
        "secondary_atom_distributions": secondary_distributions,
    }


def _scale_motif_payloads(
    raw_payloads: dict[str, dict[str, object]],
    *,
    base_count: float,
) -> dict[str, dict[str, object]]:
    total_raw_count = sum(
        float(payload.get("count", 0.0) or 0.0)
        for payload in raw_payloads.values()
    )
    if total_raw_count <= 0.0 or base_count <= 0.0:
        return {}

    scale = float(base_count) / total_raw_count
    scaled_payloads: dict[str, dict[str, object]] = {}
    for motif_name, payload in raw_payloads.items():
        scaled_payloads[motif_name] = {
            "count": float(payload.get("count", 0.0) or 0.0) * scale,
            "weight": 0.0,
            "secondary_atom_distributions": {
                element: {
                    segment: float(value) * scale
                    for segment, value in dict(distribution).items()
                }
                for element, distribution in dict(
                    payload.get("secondary_atom_distributions", {})
                ).items()
            },
        }
    return scaled_payloads


def _predicted_structure_payload(
    prediction,
    *,
    count: float,
    structure_label: str,
) -> dict[str, object]:
    element_counts = Counter(
        str(element) for element in prediction.generated_elements
    )
    label_elements = set(parse_stoich_label(structure_label).keys())
    secondary_distributions = {
        element: {str(int(element_counts[element])): float(count)}
        for element in sorted(element_counts)
        if element not in label_elements
    }
    return {
        "count": float(count),
        "weight": 0.0,
        "secondary_atom_distributions": secondary_distributions,
    }


def _secondary_elements_from_payloads(
    payloads: dict[str, dict[str, object]],
) -> set[str]:
    return {
        str(element)
        for payload in payloads.values()
        for element in dict(payload.get("secondary_atom_distributions", {}))
    }


def _observed_label_elements(result: ClusterDynamicsMLResult) -> set[str]:
    return {
        str(element)
        for observation in result.training_observations
        for element in parse_stoich_label(observation.label)
    }


def _predicted_structure_label(
    prediction,
    *,
    observed_label_elements: set[str],
) -> str:
    primary_counts = {
        str(element): int(count)
        for element, count in dict(prediction.element_counts).items()
        if int(count) > 0
        and (
            not observed_label_elements
            or str(element) in observed_label_elements
        )
    }
    if not primary_counts:
        primary_counts = {
            str(element): int(count)
            for element, count in dict(prediction.element_counts).items()
            if int(count) > 0
        }
    return stoichiometry_label(primary_counts)


def _build_saxs_model(
    result: ClusterDynamicsMLResult,
    *,
    include_predictions: bool,
) -> dict[str, np.ndarray | float | None] | None:
    comparison = result.saxs_comparison
    if comparison is None:
        return None
    q_values = np.asarray(comparison.q_values, dtype=float)
    if q_values.size == 0:
        return None
    experimental_intensity = (
        None
        if comparison.experimental_intensity is None
        else np.asarray(comparison.experimental_intensity, dtype=float)
    )
    if include_predictions:
        fitted_model = np.asarray(
            comparison.fitted_model_intensity, dtype=float
        )
        rmse = comparison.rmse
    else:
        if comparison.observed_fitted_model_intensity is None:
            return None
        fitted_model = np.asarray(
            comparison.observed_fitted_model_intensity,
            dtype=float,
        )
        rmse = comparison.observed_rmse
    return {
        "q_values": q_values,
        "model_intensity": fitted_model,
        "experimental_intensity": experimental_intensity,
        "rmse": rmse,
    }
