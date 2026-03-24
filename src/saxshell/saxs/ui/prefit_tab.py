from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs._model_templates import TemplateSpec
from saxshell.saxs.prefit import (
    ClusterGeometryMetadataRow,
    PrefitEvaluation,
    PrefitParameterEntry,
)
from saxshell.saxs.ui.template_help import (
    TEMPLATE_HELP_TEXT,
    show_template_help,
)


@dataclass(slots=True)
class PrefitRunConfig:
    method: str
    max_nfev: int


class PrefitTab(QWidget):
    template_changed = Signal(str)
    autosave_toggled = Signal(bool)
    update_model_requested = Signal()
    run_fit_requested = Signal()
    apply_recommended_scale_requested = Signal()
    set_best_prefit_requested = Signal()
    reset_best_prefit_requested = Signal()
    save_fit_requested = Signal()
    save_plot_data_requested = Signal()
    restore_state_requested = Signal()
    reset_requested = Signal()
    compute_cluster_geometry_requested = Signal()
    cluster_geometry_mapping_changed = Signal()

    PREFIT_HELP_TEXT = (
        "Recommended prefit workflow:\n"
        "- Refine scale before the other model parameters.\n"
        "- After scale is stable, refine scale and offset together.\n"
        "- Component weights w<##> are not recommended for prefit "
        "refinement or manual adjustment."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_evaluation: PrefitEvaluation | None = None
        self._summary_text = ""
        self._base_log_text = ""
        self._history_messages: list[str] = []
        self._legend_line_map: dict[object, object] = {}
        self._legend_handle_lookup: dict[str, object] = {}
        self._cluster_geometry_rows: list[ClusterGeometryMetadataRow] = []
        self._cluster_geometry_mapping_options: list[tuple[str, str]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(10)

        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pane_splitter.setChildrenCollapsible(False)
        self._pane_splitter.setHandleWidth(10)

        self._left_panel = QWidget()
        left_layout = QVBoxLayout(self._left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_controls_group())
        self._cluster_geometry_group = self._build_cluster_geometry_group()
        left_layout.addWidget(self._cluster_geometry_group)
        left_layout.addWidget(self._build_parameter_group(), stretch=1)
        self._plot_group = self._build_plot_group()
        self._pane_splitter.addWidget(self._left_panel)
        self._pane_splitter.addWidget(self._plot_group)
        self._pane_splitter.setStretchFactor(0, 5)
        self._pane_splitter.setStretchFactor(1, 7)
        self._pane_splitter.setSizes([520, 760])
        self._output_group = self._build_output_group()
        self._main_splitter.addWidget(self._pane_splitter)
        self._main_splitter.addWidget(self._output_group)
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 2)
        self._main_splitter.setSizes([760, 260])
        content_layout.addWidget(self._main_splitter, stretch=1)

        self._scroll_area.setWidget(content)
        root.addWidget(self._scroll_area)

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Prefit Controls")
        layout = QGridLayout(group)

        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(320)
        self.template_combo.currentIndexChanged.connect(
            self._on_template_index_changed
        )
        self.template_help_button = QToolButton()
        self.template_help_button.setText("?")
        self.template_help_button.setToolTip(TEMPLATE_HELP_TEXT)
        self.template_help_button.clicked.connect(
            lambda: show_template_help(self)
        )
        layout.addWidget(QLabel("Template"), 0, 0)
        layout.addWidget(self.template_combo, 0, 1)
        layout.addWidget(self.template_help_button, 0, 2)

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["leastsq", "nelder", "powell", "differential_evolution"]
        )
        layout.addWidget(QLabel("Minimizer"), 1, 0)
        layout.addWidget(self.method_combo, 1, 1, 1, 2)

        self.nfev_spin = QSpinBox()
        self.nfev_spin.setRange(100, 10_000_000)
        self.nfev_spin.setValue(10_000)
        layout.addWidget(QLabel("Max nfev"), 2, 0)
        layout.addWidget(self.nfev_spin, 2, 1, 1, 2)

        self.saved_state_combo = QComboBox()
        self.saved_state_combo.setToolTip(
            "Choose one of the timestamped prefit snapshot folders saved in "
            "the project's prefit directory."
        )
        self.restore_state_button = QPushButton("Restore Prefit State")
        self.restore_state_button.setToolTip(
            "Load the selected saved prefit snapshot folder and restore its "
            "parameter table plus saved prefit run settings."
        )
        self.restore_state_button.setEnabled(False)
        self.restore_state_button.clicked.connect(
            self.restore_state_requested.emit
        )
        restore_row = QWidget()
        restore_layout = QHBoxLayout(restore_row)
        restore_layout.setContentsMargins(0, 0, 0, 0)
        restore_layout.addWidget(self.saved_state_combo, stretch=1)
        restore_layout.addWidget(self.restore_state_button)
        layout.addWidget(QLabel("Saved states"), 3, 0)
        layout.addWidget(restore_row, 3, 1, 1, 2)

        button_grid = QGridLayout()
        self.update_button = QPushButton("Update Model")
        self.update_button.setToolTip(
            "Recalculate the current SAXS model preview with the parameter "
            "values shown in the table."
        )
        self.update_button.clicked.connect(self.update_model_requested.emit)
        self.run_button = QPushButton("Run Prefit")
        self.run_button.setToolTip(
            "Run the lmfit prefit refinement using the selected minimizer, "
            "max nfev, and current parameter table."
        )
        self.run_button.clicked.connect(self.run_fit_requested.emit)
        self.prefit_help_button = QToolButton()
        self.prefit_help_button.setText("?")
        self.prefit_help_button.setToolTip(self.PREFIT_HELP_TEXT)
        self.recommended_scale_button = QPushButton(
            "Use Recommended Scale Settings"
        )
        self.recommended_scale_button.setToolTip(
            "Estimate the multiplicative scale from the current model and "
            "experimental intensities, then update the scale value and its "
            "refinement bounds."
        )
        self.recommended_scale_button.clicked.connect(
            self.apply_recommended_scale_requested.emit
        )
        self.set_best_button = QPushButton("Set Best Prefit Params")
        self.set_best_button.setToolTip(
            "Save the current prefit parameter table into the project file as "
            "the Best Prefit preset for future reloads and quick restores."
        )
        self.set_best_button.clicked.connect(
            self.set_best_prefit_requested.emit
        )
        self.reset_best_button = QPushButton("Reset Parameters to Best Prefit")
        self.reset_best_button.setToolTip(
            "Replace the current table values with the Best Prefit preset "
            "saved in the project file."
        )
        self.reset_best_button.clicked.connect(
            self.reset_best_prefit_requested.emit
        )
        self.autosave_checkbox = QCheckBox("Autosave fit results")
        self.autosave_checkbox.setChecked(False)
        self.autosave_checkbox.setToolTip(
            "Automatically write the current fit report and parameter state "
            "to the project after each prefit run."
        )
        self.autosave_checkbox.toggled.connect(self.autosave_toggled.emit)
        self.save_button = QPushButton("Save Fit")
        self.save_button.setToolTip(
            "Write the current working prefit report, curve, and parameter "
            "state to the project without changing the Best Prefit preset."
        )
        self.save_button.clicked.connect(self.save_fit_requested.emit)
        self.reset_button = QPushButton("Reset Parameters to Template")
        self.reset_button.setToolTip(
            "Restore the parameter table to the template-default prefit "
            "preset saved in the project file."
        )
        self.reset_button.clicked.connect(self.reset_requested.emit)
        run_cell = QWidget()
        run_cell_layout = QVBoxLayout(run_cell)
        run_cell_layout.setContentsMargins(0, 0, 0, 0)
        run_cell_layout.setSpacing(4)
        run_button_row = QHBoxLayout()
        run_button_row.setContentsMargins(0, 0, 0, 0)
        run_button_row.addWidget(self.run_button)
        run_button_row.addWidget(self.prefit_help_button)
        run_cell_layout.addLayout(run_button_row)
        run_cell_layout.addWidget(self.autosave_checkbox)
        button_grid.addWidget(self.update_button, 0, 0)
        button_grid.addWidget(run_cell, 0, 1)
        button_grid.addWidget(self.save_button, 1, 0)
        button_grid.addWidget(self.reset_button, 1, 1)
        button_grid.addWidget(self.set_best_button, 2, 0)
        button_grid.addWidget(self.reset_best_button, 2, 1)
        layout.addLayout(button_grid, 4, 0, 1, 3)
        layout.addWidget(self.recommended_scale_button, 5, 0, 1, 3)
        return group

    def _build_plot_group(self) -> QGroupBox:
        group = QGroupBox("Model vs Experimental")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.log_x_checkbox = QCheckBox("Log X")
        self.log_x_checkbox.setChecked(True)
        self.log_x_checkbox.toggled.connect(self._redraw_current_plot)
        self.log_y_checkbox = QCheckBox("Log Y")
        self.log_y_checkbox.setChecked(True)
        self.log_y_checkbox.toggled.connect(self._redraw_current_plot)
        self.save_plot_data_button = QPushButton("Export Plot Data")
        self.save_plot_data_button.clicked.connect(
            self.save_plot_data_requested.emit
        )
        controls.addWidget(self.log_x_checkbox)
        controls.addWidget(self.log_y_checkbox)
        controls.addWidget(self.save_plot_data_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(9.6, 5.6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("pick_event", self._handle_legend_pick)
        self.plot_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.setMinimumHeight(340)
        layout.addWidget(self.plot_toolbar)
        layout.addWidget(self.canvas)
        return group

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Parameters")
        layout = QVBoxLayout(group)
        self.parameter_table = QTableWidget(0, 7)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Structure", "Motif", "Param", "Value", "Vary", "Min", "Max"]
        )
        header = self.parameter_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parameter_table.setMinimumWidth(520)
        layout.addWidget(self.parameter_table)
        return group

    def _build_cluster_geometry_group(self) -> QGroupBox:
        group = QGroupBox("Cluster Geometry Metadata")
        layout = QVBoxLayout(group)

        header_row = QHBoxLayout()
        self.cluster_geometry_status_label = QLabel(
            "This template does not use per-cluster geometry metadata."
        )
        self.cluster_geometry_status_label.setWordWrap(True)
        self.compute_cluster_geometry_button = QPushButton(
            "Compute Cluster Geometry"
        )
        self.compute_cluster_geometry_button.setToolTip(
            "Compute average cluster geometry descriptors for each cluster "
            "folder and map the results to the generated component weight "
            "parameters."
        )
        self.compute_cluster_geometry_button.clicked.connect(
            self.compute_cluster_geometry_requested.emit
        )
        header_row.addWidget(self.cluster_geometry_status_label, stretch=1)
        header_row.addWidget(self.compute_cluster_geometry_button)
        layout.addLayout(header_row)

        self.cluster_geometry_table = QTableWidget(0, 8)
        self.cluster_geometry_table.setHorizontalHeaderLabels(
            [
                "Cluster",
                "Path",
                "Avg Size",
                "Effective Radius",
                "S.F. Approx.",
                "Anisotropy",
                "Map To",
                "Notes",
            ]
        )
        header = self.cluster_geometry_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cluster_geometry_table.setMinimumHeight(220)
        layout.addWidget(self.cluster_geometry_table)
        group.setVisible(False)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Prefit Output")
        layout = QVBoxLayout(group)
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(220)
        layout.addWidget(self.output_box)
        self.log_box = self.output_box
        self.summary_box = self.output_box
        return group

    def set_templates(
        self,
        template_specs: list[TemplateSpec],
        selected_name: str | None,
    ) -> None:
        current_name = selected_name or self.selected_template_name()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for spec in template_specs:
            self.template_combo.addItem(spec.display_name, userData=spec.name)
            index = self.template_combo.count() - 1
            self.template_combo.setItemData(
                index,
                spec.description,
                Qt.ItemDataRole.ToolTipRole,
            )
        if current_name:
            index = self._find_template_index(current_name)
            if index >= 0:
                self.template_combo.setCurrentIndex(index)
        self.template_combo.blockSignals(False)
        self._update_template_tooltip()

    def selected_template_name(self) -> str | None:
        return str(self.template_combo.currentData() or "").strip() or None

    def set_autosave(self, enabled: bool) -> None:
        self.autosave_checkbox.blockSignals(True)
        self.autosave_checkbox.setChecked(enabled)
        self.autosave_checkbox.blockSignals(False)

    def set_run_config(self, *, method: str, max_nfev: int) -> None:
        method_index = self.method_combo.findText(method)
        if method_index >= 0:
            self.method_combo.setCurrentIndex(method_index)
        self.nfev_spin.setValue(int(max_nfev))

    def set_saved_states(
        self,
        state_names: list[str],
        selected_name: str | None = None,
    ) -> None:
        current_name = selected_name or self.selected_saved_state_name()
        self.saved_state_combo.blockSignals(True)
        self.saved_state_combo.clear()
        self.saved_state_combo.addItems(state_names)
        if current_name:
            index = self.saved_state_combo.findText(current_name)
            if index >= 0:
                self.saved_state_combo.setCurrentIndex(index)
        if self.saved_state_combo.currentIndex() < 0 and state_names:
            self.saved_state_combo.setCurrentIndex(0)
        self.saved_state_combo.blockSignals(False)
        self.restore_state_button.setEnabled(bool(state_names))

    def selected_saved_state_name(self) -> str | None:
        text = self.saved_state_combo.currentText().strip()
        return text or None

    def populate_parameter_table(
        self,
        entries: list[PrefitParameterEntry],
    ) -> None:
        self.parameter_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self.parameter_table.setItem(
                row, 0, QTableWidgetItem(entry.structure)
            )
            self.parameter_table.setItem(row, 1, QTableWidgetItem(entry.motif))
            self.parameter_table.setItem(row, 2, QTableWidgetItem(entry.name))
            self.parameter_table.setItem(
                row,
                3,
                QTableWidgetItem(f"{entry.value:.6g}"),
            )
            vary_item = QTableWidgetItem()
            vary_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            vary_item.setCheckState(
                Qt.CheckState.Checked
                if entry.vary
                else Qt.CheckState.Unchecked
            )
            self.parameter_table.setItem(row, 4, vary_item)
            self.parameter_table.setItem(
                row,
                5,
                QTableWidgetItem(f"{entry.minimum:.6g}"),
            )
            self.parameter_table.setItem(
                row,
                6,
                QTableWidgetItem(f"{entry.maximum:.6g}"),
            )
        self.parameter_table.resizeRowsToContents()

    def set_cluster_geometry_visible(self, visible: bool) -> None:
        self._cluster_geometry_group.setVisible(bool(visible))

    def set_cluster_geometry_status_text(self, text: str) -> None:
        self.cluster_geometry_status_label.setText(text.strip())

    def populate_cluster_geometry_table(
        self,
        rows: list[ClusterGeometryMetadataRow],
        *,
        mapping_options: list[tuple[str, str]],
    ) -> None:
        self._cluster_geometry_rows = [
            ClusterGeometryMetadataRow.from_dict(row.to_dict()) for row in rows
        ]
        self._cluster_geometry_mapping_options = list(mapping_options)
        self.cluster_geometry_table.setRowCount(
            len(self._cluster_geometry_rows)
        )
        for row_index, row in enumerate(self._cluster_geometry_rows):
            self.cluster_geometry_table.setItem(
                row_index,
                0,
                QTableWidgetItem(row.cluster_id),
            )
            path_item = QTableWidgetItem(row.cluster_path)
            path_item.setToolTip(row.cluster_path)
            self.cluster_geometry_table.setItem(row_index, 1, path_item)
            self.cluster_geometry_table.setItem(
                row_index,
                2,
                QTableWidgetItem(f"{row.avg_size_metric:.4g}"),
            )
            self.cluster_geometry_table.setItem(
                row_index,
                3,
                QTableWidgetItem(f"{row.effective_radius:.4g}"),
            )
            self.cluster_geometry_table.setItem(
                row_index,
                4,
                QTableWidgetItem(row.structure_factor_recommendation),
            )
            self.cluster_geometry_table.setItem(
                row_index,
                5,
                QTableWidgetItem(f"{row.anisotropy_metric:.4g}"),
            )
            mapping_combo = QComboBox()
            mapping_combo.addItem("", userData=None)
            selected_index = 0
            for option_index, (param_name, label) in enumerate(
                self._cluster_geometry_mapping_options,
                start=1,
            ):
                mapping_combo.addItem(label, userData=param_name)
                if (
                    row.mapped_parameter is not None
                    and row.mapped_parameter == param_name
                ):
                    selected_index = option_index
            mapping_combo.setCurrentIndex(selected_index)
            mapping_combo.currentIndexChanged.connect(
                lambda _index, row_idx=row_index, combo=mapping_combo: (
                    self._on_cluster_geometry_mapping_changed(row_idx, combo)
                )
            )
            self.cluster_geometry_table.setCellWidget(
                row_index,
                6,
                mapping_combo,
            )
            notes_item = QTableWidgetItem(row.notes)
            notes_item.setToolTip(row.notes)
            self.cluster_geometry_table.setItem(row_index, 7, notes_item)
        self.cluster_geometry_table.resizeRowsToContents()

    def cluster_geometry_rows(self) -> list[ClusterGeometryMetadataRow]:
        return [
            ClusterGeometryMetadataRow.from_dict(row.to_dict())
            for row in self._cluster_geometry_rows
        ]

    def parameter_entries(self) -> list[PrefitParameterEntry]:
        entries: list[PrefitParameterEntry] = []
        for row in range(self.parameter_table.rowCount()):
            entries.append(
                PrefitParameterEntry(
                    structure=self._item_text(row, 0),
                    motif=self._item_text(row, 1),
                    name=self._item_text(row, 2),
                    value=float(self._item_text(row, 3)),
                    vary=(
                        self.parameter_table.item(row, 4).checkState()
                        == Qt.CheckState.Checked
                    ),
                    minimum=float(self._item_text(row, 5)),
                    maximum=float(self._item_text(row, 6)),
                    category=(
                        "weight"
                        if self._item_text(row, 2).startswith("w")
                        else "fit"
                    ),
                )
            )
        return entries

    def find_parameter_row(self, parameter_name: str) -> int:
        for row in range(self.parameter_table.rowCount()):
            if self._item_text(row, 2) == parameter_name:
                return row
        return -1

    def set_parameter_row(
        self,
        parameter_name: str,
        *,
        value: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        vary: bool | None = None,
    ) -> None:
        row = self.find_parameter_row(parameter_name)
        if row < 0:
            raise ValueError(f"Parameter {parameter_name} was not found.")
        if value is not None:
            self.parameter_table.setItem(
                row,
                3,
                QTableWidgetItem(f"{float(value):.6g}"),
            )
        if vary is not None:
            vary_item = self.parameter_table.item(row, 4)
            if vary_item is not None:
                vary_item.setCheckState(
                    Qt.CheckState.Checked if vary else Qt.CheckState.Unchecked
                )
        if minimum is not None:
            self.parameter_table.setItem(
                row,
                5,
                QTableWidgetItem(f"{float(minimum):.6g}"),
            )
        if maximum is not None:
            self.parameter_table.setItem(
                row,
                6,
                QTableWidgetItem(f"{float(maximum):.6g}"),
            )

    def run_config(self) -> PrefitRunConfig:
        return PrefitRunConfig(
            method=self.method_combo.currentText(),
            max_nfev=int(self.nfev_spin.value()),
        )

    def plot_evaluation(
        self,
        evaluation: PrefitEvaluation | None,
    ) -> None:
        self._current_evaluation = evaluation
        self._legend_line_map.clear()
        self._legend_handle_lookup.clear()
        for axis in self.figure.axes:
            axis.set_xscale("linear")
        self.figure.clear()
        if evaluation is None:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Build a project and load the prefit workflow to preview the model.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            self.canvas.draw()
            return

        grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
        top = self.figure.add_subplot(grid[0, 0])
        bottom = self.figure.add_subplot(grid[1, 0], sharex=top)

        plotted_lines = []

        (experimental_line,) = top.plot(
            evaluation.q_values,
            evaluation.experimental_intensities,
            color="black",
            label="Experimental",
        )
        plotted_lines.append(experimental_line)

        if evaluation.solvent_contribution is not None:
            solvent_values = np.asarray(
                evaluation.solvent_contribution,
                dtype=float,
            )
            solvent_mask = np.isfinite(solvent_values)
            if self.log_y_checkbox.isChecked():
                solvent_mask &= solvent_values > 0.0
            if np.any(solvent_mask):
                (solvent_line,) = top.plot(
                    np.asarray(evaluation.q_values, dtype=float)[solvent_mask],
                    solvent_values[solvent_mask],
                    color="green",
                    linewidth=1.5,
                    label="Solvent contribution",
                )
                plotted_lines.append(solvent_line)

        (model_line,) = top.plot(
            evaluation.q_values,
            evaluation.model_intensities,
            color="tab:red",
            label="Model",
        )
        plotted_lines.append(model_line)
        top.set_xscale("log" if self.log_x_checkbox.isChecked() else "linear")
        top.set_yscale("log" if self.log_y_checkbox.isChecked() else "linear")
        top.set_ylabel("Intensity (arb. units)")
        self._build_interactive_legend(top, plotted_lines)

        bottom.axhline(0.0, color="0.5", linewidth=1.0)
        bottom.plot(
            evaluation.q_values,
            evaluation.residuals,
            color="tab:blue",
        )
        bottom.set_xscale(
            "log" if self.log_x_checkbox.isChecked() else "linear"
        )
        bottom.set_xlabel("q (Å⁻¹)")
        bottom.set_ylabel("Residual")
        self.figure.tight_layout()
        self.canvas.draw()

    def current_evaluation(self) -> PrefitEvaluation | None:
        return self._current_evaluation

    def append_log(self, message: str) -> None:
        stripped = message.strip()
        if stripped:
            self._history_messages.append(stripped)
        self._render_output(scroll_to_end=True)

    def set_log_text(self, text: str) -> None:
        self._base_log_text = text.strip()
        self._render_output()

    def set_summary_text(self, text: str) -> None:
        self._summary_text = text.strip()
        self._render_output()

    def _item_text(self, row: int, column: int) -> str:
        item = self.parameter_table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _redraw_current_plot(self) -> None:
        self.plot_evaluation(self._current_evaluation)

    def _on_cluster_geometry_mapping_changed(
        self,
        row_index: int,
        combo: QComboBox,
    ) -> None:
        if row_index < 0 or row_index >= len(self._cluster_geometry_rows):
            return
        mapped_parameter = combo.currentData()
        self._cluster_geometry_rows[row_index].mapped_parameter = (
            str(mapped_parameter).strip()
            if mapped_parameter not in (None, "")
            else None
        )
        self.cluster_geometry_mapping_changed.emit()

    def _build_interactive_legend(self, axis, lines: list[object]) -> None:
        legend = axis.legend()
        if legend is None:
            return
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = getattr(legend, "legendHandles", [])
        for legend_handle, original_line in zip(legend_handles, lines):
            if hasattr(legend_handle, "set_picker"):
                legend_handle.set_picker(True)
                legend_handle.set_pickradius(6)
            legend_handle.set_alpha(
                1.0 if original_line.get_visible() else 0.25
            )
            self._legend_line_map[legend_handle] = original_line
            label = str(original_line.get_label()).strip()
            if label:
                self._legend_handle_lookup[label] = legend_handle

    def _handle_legend_pick(self, event) -> None:
        original_line = self._legend_line_map.get(event.artist)
        if original_line is None:
            return
        is_visible = not original_line.get_visible()
        original_line.set_visible(is_visible)
        if hasattr(event.artist, "set_alpha"):
            event.artist.set_alpha(1.0 if is_visible else 0.25)
        for axis in self.figure.axes:
            try:
                axis.relim(visible_only=True)
                axis.autoscale_view()
            except Exception:
                continue
        self.canvas.draw_idle()

    def set_selected_template(
        self,
        template_name: str | None,
        *,
        emit_signal: bool = False,
    ) -> None:
        selected = template_name or ""
        index = self._find_template_index(selected)
        if index < 0:
            return
        if emit_signal:
            self.template_combo.setCurrentIndex(index)
            return
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(index)
        self.template_combo.blockSignals(False)
        self._update_template_tooltip()

    def _render_output(self, *, scroll_to_end: bool = False) -> None:
        sections: list[str] = []
        if self._summary_text:
            sections.append("Prefit Summary\n" + self._summary_text)
        history_parts = [
            part
            for part in [self._base_log_text, *self._history_messages]
            if part
        ]
        if history_parts:
            sections.append("Prefit Console\n" + "\n\n".join(history_parts))
        self.output_box.setPlainText("\n\n".join(sections).strip())
        if scroll_to_end:
            scrollbar = self.output_box.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _on_template_index_changed(self) -> None:
        self._update_template_tooltip()
        selected_name = self.selected_template_name()
        if selected_name:
            self.template_changed.emit(selected_name)

    def _update_template_tooltip(self) -> None:
        description = str(
            self.template_combo.currentData(Qt.ItemDataRole.ToolTipRole) or ""
        ).strip()
        self.template_combo.setToolTip(description)

    def _find_template_index(self, template_name: str) -> int:
        for index in range(self.template_combo.count()):
            if (
                str(self.template_combo.itemData(index) or "").strip()
                == template_name
            ):
                return index
        return -1
