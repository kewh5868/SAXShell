from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.plotting import Q_A_INVERSE_LABEL
from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    load_experimental_data_file,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.experimental_data_loader import (
    ExperimentalDataHeaderDialog,
)


@dataclass(slots=True)
class ExperimentalOverlayTrace:
    path: Path
    summary: ExperimentalDataSummary
    label: str
    color: str
    visible: bool = True
    axis: str = "left"


class ExperimentalDataOverlayWindow(QMainWindow):
    """Overlay multiple experimental data files with shared header
    parsing."""

    SHOW_COLUMN = 0
    LABEL_COLUMN = 1
    AXIS_COLUMN = 2
    COLOR_COLUMN = 3
    POINTS_COLUMN = 4
    Q_RANGE_COLUMN = 5
    COLUMNS_COLUMN = 6

    def __init__(
        self,
        *,
        initial_paths: Iterable[str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Experimental Data Overlay")
        self.resize(1180, 760)
        self.traces: list[ExperimentalOverlayTrace] = []
        self._updating_table = False
        self._left_axis = None
        self._right_axis = None

        self._build_ui()
        self._refresh_q_range_controls()
        self._refresh_trace_table()
        self._refresh_plot()

        if initial_paths is not None:
            self.add_data_files(initial_paths)

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(root)

        controls = QWidget()
        controls.setMinimumWidth(255)
        controls.setMaximumWidth(340)
        controls_layout = QVBoxLayout(controls)

        file_group = QGroupBox("Data Files")
        file_layout = QVBoxLayout(file_group)
        self.add_files_button = QPushButton("Add Data Files...")
        self.add_files_button.clicked.connect(self._choose_data_files)
        self.remove_files_button = QPushButton("Remove Selected")
        self.remove_files_button.clicked.connect(self._remove_selected_traces)
        self.clear_files_button = QPushButton("Clear Traces")
        self.clear_files_button.clicked.connect(self._clear_traces)
        file_layout.addWidget(self.add_files_button)
        file_layout.addWidget(self.remove_files_button)
        file_layout.addWidget(self.clear_files_button)
        controls_layout.addWidget(file_group)

        range_group = QGroupBox("q-Range")
        range_layout = QFormLayout(range_group)
        self.full_q_range_checkbox = QCheckBox("Use full loaded range")
        self.full_q_range_checkbox.setChecked(True)
        self.full_q_range_checkbox.toggled.connect(
            self._on_full_q_range_toggled
        )
        range_layout.addRow(self.full_q_range_checkbox)

        self.q_min_spin = QDoubleSpinBox()
        self.q_min_spin.setDecimals(6)
        self.q_min_spin.setRange(-1.0e12, 1.0e12)
        self.q_min_spin.setSingleStep(0.01)
        self.q_min_spin.valueChanged.connect(self._on_q_range_changed)
        range_layout.addRow("q min", self.q_min_spin)

        self.q_max_spin = QDoubleSpinBox()
        self.q_max_spin.setDecimals(6)
        self.q_max_spin.setRange(-1.0e12, 1.0e12)
        self.q_max_spin.setSingleStep(0.01)
        self.q_max_spin.valueChanged.connect(self._on_q_range_changed)
        range_layout.addRow("q max", self.q_max_spin)

        self.use_loaded_range_button = QPushButton("Use Loaded Range")
        self.use_loaded_range_button.clicked.connect(self._use_loaded_q_range)
        range_layout.addRow(self.use_loaded_range_button)
        controls_layout.addWidget(range_group)

        axes_group = QGroupBox("Axes")
        axes_layout = QVBoxLayout(axes_group)
        scale_button_row = QHBoxLayout()
        self.log_x_axis_button = QPushButton("Log X: On")
        self.log_x_axis_button.setCheckable(True)
        self.log_x_axis_button.setChecked(True)
        self.log_x_axis_button.toggled.connect(self._set_log_x_axis_enabled)
        self.log_y_axis_button = QPushButton("Log Y: On")
        self.log_y_axis_button.setCheckable(True)
        self.log_y_axis_button.setChecked(True)
        self.log_y_axis_button.toggled.connect(self._set_log_y_axis_enabled)
        scale_button_row.addWidget(self.log_x_axis_button)
        scale_button_row.addWidget(self.log_y_axis_button)
        axes_layout.addLayout(scale_button_row)
        self.align_y_axes_checkbox = QCheckBox(
            "Rescale right axis to left data"
        )
        self.align_y_axes_checkbox.setChecked(True)
        self.align_y_axes_checkbox.toggled.connect(self._refresh_plot)
        self.rescale_axes_button = QPushButton("Rescale Axes")
        self.rescale_axes_button.clicked.connect(
            self._rescale_axes_to_current_q_range
        )
        axes_layout.addWidget(self.align_y_axes_checkbox)
        axes_layout.addWidget(self.rescale_axes_button)
        controls_layout.addWidget(axes_group)

        self.status_label = QLabel("Open experimental data files to overlay.")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)
        root.addWidget(controls)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(7.6, 5.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)

        self.trace_table = QTableWidget(0, 7)
        self.trace_table.setHorizontalHeaderLabels(
            [
                "Show",
                "Dataset",
                "Axis",
                "Color",
                "Points",
                "q range",
                "Columns",
            ]
        )
        self.trace_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.trace_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.trace_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.trace_table.itemChanged.connect(self._on_trace_item_changed)
        self.trace_table.cellClicked.connect(self._on_trace_cell_clicked)
        self.trace_table.itemSelectionChanged.connect(
            self._refresh_action_state
        )
        horizontal_header = self.trace_table.horizontalHeader()
        horizontal_header.setSectionResizeMode(
            self.SHOW_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        horizontal_header.setSectionResizeMode(
            self.LABEL_COLUMN,
            QHeaderView.ResizeMode.Stretch,
        )
        horizontal_header.setSectionResizeMode(
            self.AXIS_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        horizontal_header.setSectionResizeMode(
            self.COLOR_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        horizontal_header.setSectionResizeMode(
            self.POINTS_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        horizontal_header.setSectionResizeMode(
            self.Q_RANGE_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        horizontal_header.setSectionResizeMode(
            self.COLUMNS_COLUMN,
            QHeaderView.ResizeMode.Stretch,
        )
        self.trace_table.setMinimumHeight(170)
        plot_layout.addWidget(self.trace_table)
        root.addWidget(plot_panel)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)

    def _choose_data_files(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Open Experimental Data Files",
            "",
            "Data files (*.txt *.dat *.iq);;All files (*)",
        )
        if paths:
            self.add_data_files(paths)

    def add_data_files(self, paths: Iterable[str | Path]) -> int:
        added = 0
        failures: list[str] = []
        for raw_path in paths:
            file_path = Path(raw_path).expanduser().resolve()
            summary = self._load_data_file(file_path)
            if summary is None:
                failures.append(file_path.name)
                continue
            self.traces.append(
                ExperimentalOverlayTrace(
                    path=file_path,
                    summary=summary,
                    label=file_path.name,
                    color=self._next_trace_color(),
                )
            )
            added += 1

        if added:
            self._refresh_q_range_controls()
            self._refresh_trace_table()
            self._refresh_plot()
            self.status_label.setText(
                f"Loaded {added} data file{'s' if added != 1 else ''}."
            )
        if failures:
            QMessageBox.warning(
                self,
                "Experimental Data Load",
                "Could not load: " + ", ".join(failures),
            )
        self._refresh_action_state()
        return added

    def _load_data_file(
        self,
        file_path: Path,
    ) -> ExperimentalDataSummary | None:
        try:
            return load_experimental_data_file(file_path, skiprows=0)
        except Exception:
            dialog = ExperimentalDataHeaderDialog(file_path, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            return dialog.accepted_summary

    def _next_trace_color(self) -> str:
        color = colormaps["tab10"](len(self.traces) % 10)
        return str(to_hex(color))

    def _refresh_trace_table(self) -> None:
        self._updating_table = True
        self.trace_table.blockSignals(True)
        try:
            self.trace_table.setRowCount(len(self.traces))
            for row, trace in enumerate(self.traces):
                self.trace_table.setItem(
                    row,
                    self.SHOW_COLUMN,
                    self._build_visibility_item(trace),
                )
                self.trace_table.setItem(
                    row,
                    self.LABEL_COLUMN,
                    self._build_label_item(trace),
                )
                self.trace_table.setCellWidget(
                    row,
                    self.AXIS_COLUMN,
                    self._build_axis_combo(row, trace),
                )
                self.trace_table.setItem(
                    row,
                    self.COLOR_COLUMN,
                    self._build_color_item(trace),
                )
                self.trace_table.setItem(
                    row,
                    self.POINTS_COLUMN,
                    self._read_only_item(str(len(trace.summary.q_values))),
                )
                self.trace_table.setItem(
                    row,
                    self.Q_RANGE_COLUMN,
                    self._read_only_item(self._trace_q_range_text(trace)),
                )
                self.trace_table.setItem(
                    row,
                    self.COLUMNS_COLUMN,
                    self._read_only_item(self._trace_column_text(trace)),
                )
        finally:
            self.trace_table.blockSignals(False)
            self._updating_table = False
        self._refresh_action_state()

    def _build_visibility_item(
        self,
        trace: ExperimentalOverlayTrace,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(
            Qt.CheckState.Checked if trace.visible else Qt.CheckState.Unchecked
        )
        return item

    def _build_label_item(
        self,
        trace: ExperimentalOverlayTrace,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(trace.label)
        item.setToolTip(str(trace.path))
        item.setFlags(
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
        )
        return item

    def _build_axis_combo(
        self,
        row: int,
        trace: ExperimentalOverlayTrace,
    ) -> QComboBox:
        combo = QComboBox()
        combo.addItem("Left Y", "left")
        combo.addItem("Right Y", "right")
        combo.setCurrentIndex(combo.findData(trace.axis))
        combo.currentIndexChanged.connect(
            lambda _index, trace_index=row, widget=combo: self._set_trace_axis(
                trace_index,
                str(widget.currentData()),
            )
        )
        return combo

    def _build_color_item(
        self,
        trace: ExperimentalOverlayTrace,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(trace.color)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setBackground(QColor(trace.color))
        item.setToolTip("Click to choose a custom trace color.")
        return item

    def _read_only_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        return item

    def _on_trace_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table:
            return
        row = item.row()
        if row < 0 or row >= len(self.traces):
            return
        trace = self.traces[row]
        if item.column() == self.SHOW_COLUMN:
            trace.visible = item.checkState() == Qt.CheckState.Checked
            self._refresh_plot()
            return
        if item.column() == self.LABEL_COLUMN:
            updated = item.text().strip()
            if updated:
                trace.label = updated
                self._refresh_plot()
            else:
                self._refresh_trace_table()

    def _on_trace_cell_clicked(self, row: int, column: int) -> None:
        if column != self.COLOR_COLUMN:
            return
        if row < 0 or row >= len(self.traces):
            return
        initial_color = QColor(self.traces[row].color)
        chosen = QColorDialog.getColor(
            initial_color,
            self,
            f"Choose color for {self.traces[row].label}",
        )
        if chosen.isValid():
            self._set_trace_color(row, chosen.name())

    def _set_trace_axis(self, row: int, axis: str) -> None:
        if self._updating_table or row < 0 or row >= len(self.traces):
            return
        if axis not in {"left", "right"}:
            return
        self.traces[row].axis = axis
        self._refresh_plot()

    def _set_trace_color(self, row: int, color: str) -> None:
        if row < 0 or row >= len(self.traces):
            return
        self.traces[row].color = color
        self._refresh_trace_table()
        self._refresh_plot()

    def _remove_selected_traces(self) -> None:
        selected_rows = {
            index.row()
            for index in self.trace_table.selectionModel().selectedRows()
        }
        if not selected_rows and self.trace_table.currentRow() >= 0:
            selected_rows = {self.trace_table.currentRow()}
        if not selected_rows:
            return
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < len(self.traces):
                del self.traces[row]
        self._refresh_q_range_controls()
        self._refresh_trace_table()
        self._refresh_plot()
        self.status_label.setText("Removed selected trace(s).")

    def _clear_traces(self) -> None:
        if not self.traces:
            return
        self.traces.clear()
        self._refresh_q_range_controls()
        self._refresh_trace_table()
        self._refresh_plot()
        self.status_label.setText("Cleared plotted traces.")

    def _on_q_range_changed(self, *_args: object) -> None:
        self._refresh_plot()

    def _on_full_q_range_toggled(self, checked: bool) -> None:
        self.q_min_spin.setEnabled(not checked and bool(self.traces))
        self.q_max_spin.setEnabled(not checked and bool(self.traces))
        if checked:
            self._set_spin_values_to_loaded_q_range()
        self._on_q_range_changed()

    def _use_loaded_q_range(self) -> None:
        self.full_q_range_checkbox.setChecked(True)
        self._set_spin_values_to_loaded_q_range()
        self._on_q_range_changed()

    def _refresh_q_range_controls(self) -> None:
        has_traces = bool(self.traces)
        self.full_q_range_checkbox.setEnabled(has_traces)
        self.use_loaded_range_button.setEnabled(has_traces)
        self.q_min_spin.setEnabled(
            has_traces and not self.full_q_range_checkbox.isChecked()
        )
        self.q_max_spin.setEnabled(
            has_traces and not self.full_q_range_checkbox.isChecked()
        )
        self._set_spin_values_to_loaded_q_range()
        self._refresh_action_state()

    def _set_spin_values_to_loaded_q_range(self) -> None:
        q_bounds = self._loaded_q_bounds()
        self.q_min_spin.blockSignals(True)
        self.q_max_spin.blockSignals(True)
        try:
            if q_bounds is None:
                self.q_min_spin.setValue(0.0)
                self.q_max_spin.setValue(0.0)
                return
            q_min, q_max = q_bounds
            padding = max((q_max - q_min) * 0.05, 1.0e-9)
            self.q_min_spin.setRange(q_min - padding, q_max + padding)
            self.q_max_spin.setRange(q_min - padding, q_max + padding)
            if self.full_q_range_checkbox.isChecked():
                self.q_min_spin.setValue(q_min)
                self.q_max_spin.setValue(q_max)
            else:
                self.q_min_spin.setValue(
                    max(q_min, min(self.q_min_spin.value(), q_max))
                )
                self.q_max_spin.setValue(
                    min(q_max, max(self.q_max_spin.value(), q_min))
                )
        finally:
            self.q_min_spin.blockSignals(False)
            self.q_max_spin.blockSignals(False)

    def _loaded_q_bounds(self) -> tuple[float, float] | None:
        q_segments: list[np.ndarray] = []
        for trace in self.traces:
            q_values = np.asarray(trace.summary.q_values, dtype=float)
            q_values = q_values[np.isfinite(q_values)]
            if q_values.size:
                q_segments.append(q_values)
        if not q_segments:
            return None
        q_values = np.concatenate(q_segments)
        return float(np.nanmin(q_values)), float(np.nanmax(q_values))

    def _active_q_bounds(self) -> tuple[float, float] | None:
        loaded_bounds = self._loaded_q_bounds()
        if loaded_bounds is None:
            return None
        if self.full_q_range_checkbox.isChecked():
            return loaded_bounds
        q_min, q_max = sorted(
            (float(self.q_min_spin.value()), float(self.q_max_spin.value()))
        )
        return q_min, q_max

    def _refresh_action_state(self) -> None:
        has_traces = bool(self.traces)
        has_selection = False
        selection_model = self.trace_table.selectionModel()
        if selection_model is not None:
            has_selection = bool(selection_model.selectedRows())
        self.remove_files_button.setEnabled(has_traces and has_selection)
        self.clear_files_button.setEnabled(has_traces)
        has_right_axis = self._has_visible_right_axis_trace()
        self.align_y_axes_checkbox.setEnabled(has_right_axis)
        self.rescale_axes_button.setEnabled(has_right_axis)

    def _has_visible_right_axis_trace(self) -> bool:
        return any(
            trace.visible and trace.axis == "right" for trace in self.traces
        )

    def _rescale_axes_to_current_q_range(self) -> None:
        if not self._has_visible_right_axis_trace():
            return
        if not self.align_y_axes_checkbox.isChecked():
            self.align_y_axes_checkbox.setChecked(True)
            return
        self._refresh_plot()

    def _set_log_x_axis_enabled(self, checked: bool) -> None:
        self.log_x_axis_button.setText(f"Log X: {'On' if checked else 'Off'}")
        self._refresh_plot()

    def _set_log_y_axis_enabled(self, checked: bool) -> None:
        self.log_y_axis_button.setText(f"Log Y: {'On' if checked else 'Off'}")
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        for axis in list(self.figure.axes):
            try:
                axis.set_xscale("linear")
                axis.set_yscale("linear")
            except Exception:
                continue
        self.figure.clear()
        self._left_axis = None
        self._right_axis = None

        visible_traces = [trace for trace in self.traces if trace.visible]
        if not visible_traces:
            axis = self.figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Open experimental data files to overlay traces.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self._refresh_action_state()
            return

        self._left_axis = self.figure.add_subplot(111)
        right_traces = [
            trace for trace in visible_traces if trace.axis == "right"
        ]
        if right_traces:
            self._right_axis = self._left_axis.twinx()

        self._apply_axis_scale(self._left_axis)
        if self._right_axis is not None:
            self._apply_axis_scale(self._right_axis)

        plotted_lines: list[object] = []
        for trace in visible_traces:
            target_axis = (
                self._right_axis
                if trace.axis == "right" and self._right_axis is not None
                else self._left_axis
            )
            line = self._plot_trace(target_axis, trace)
            if line is not None:
                plotted_lines.append(line)

        q_bounds = self._active_plot_q_bounds()
        if q_bounds is not None:
            q_min, q_max = q_bounds
            self._left_axis.set_xlim(q_min, q_max)
            self._autoscale_axis_y(
                self._left_axis,
                q_min,
                q_max,
                log_scale=self._log_y_axis_enabled(),
            )
            if self._right_axis is not None:
                self._right_axis.set_xlim(q_min, q_max)
                self._autoscale_axis_y(
                    self._right_axis,
                    q_min,
                    q_max,
                    log_scale=self._log_y_axis_enabled(),
                )
                if self.align_y_axes_checkbox.isChecked():
                    self._normalize_secondary_axis(
                        self._left_axis,
                        self._right_axis,
                        q_min,
                        q_max,
                    )

        self._left_axis.set_xlabel(Q_A_INVERSE_LABEL)
        self._left_axis.set_ylabel("Left axis intensity")
        self._left_axis.grid(True, alpha=0.25, linewidth=0.8)
        if self._right_axis is not None:
            self._right_axis.set_ylabel("Right axis intensity")

        if plotted_lines:
            self._left_axis.legend(
                plotted_lines,
                [line.get_label() for line in plotted_lines],
                loc="best",
                fontsize=9,
                framealpha=0.9,
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._refresh_action_state()

    def _apply_axis_scale(self, axis) -> None:
        axis.set_xscale("log" if self._log_x_axis_enabled() else "linear")
        axis.set_yscale("log" if self._log_y_axis_enabled() else "linear")

    def _log_x_axis_enabled(self) -> bool:
        return self.log_x_axis_button.isChecked()

    def _log_y_axis_enabled(self) -> bool:
        return self.log_y_axis_button.isChecked()

    def _active_plot_q_bounds(self) -> tuple[float, float] | None:
        bounds = self._active_q_bounds()
        if bounds is None or not self._log_x_axis_enabled():
            return bounds
        q_min, q_max = bounds
        if q_max <= 0.0:
            return None
        positive_segments: list[np.ndarray] = []
        for trace in self.traces:
            if not trace.visible:
                continue
            q_values = np.asarray(trace.summary.q_values, dtype=float)
            q_values = q_values[np.isfinite(q_values) & (q_values > 0.0)]
            if q_values.size:
                positive_segments.append(q_values)
        if not positive_segments:
            return None
        positive_q = np.concatenate(positive_segments)
        lower = max(q_min, float(np.nanmin(positive_q)))
        upper = max(q_max, lower * (1.0 + 1.0e-9))
        return lower, upper

    def _plot_trace(
        self,
        axis,
        trace: ExperimentalOverlayTrace,
    ):
        q_values = np.asarray(trace.summary.q_values, dtype=float)
        intensities = np.asarray(trace.summary.intensities, dtype=float)
        mask = np.isfinite(q_values) & np.isfinite(intensities)
        if self._log_x_axis_enabled():
            mask &= q_values > 0.0
        if self._log_y_axis_enabled():
            mask &= intensities > 0.0
        if not np.any(mask):
            return None
        (line,) = axis.plot(
            q_values[mask],
            intensities[mask],
            color=trace.color,
            linewidth=1.6,
            label=trace.label,
        )
        return line

    def _normalize_secondary_axis(
        self,
        left_axis,
        right_axis,
        q_min: float,
        q_max: float,
    ) -> None:
        log_scale = self._log_y_axis_enabled()
        left_values = self._axis_y_values(
            left_axis,
            q_min,
            q_max,
            log_scale=log_scale,
        )
        right_values = self._axis_y_values(
            right_axis,
            q_min,
            q_max,
            log_scale=log_scale,
        )
        if left_values.size == 0 or right_values.size == 0:
            return
        right_limits = self._aligned_y_limits(
            left_axis.get_ylim(),
            float(np.nanmin(left_values)),
            float(np.nanmax(left_values)),
            float(np.nanmin(right_values)),
            float(np.nanmax(right_values)),
            log_scale=log_scale,
        )
        right_axis.set_ylim(right_limits)

    @staticmethod
    def _axis_y_values(
        axis,
        q_min: float,
        q_max: float,
        *,
        log_scale: bool = False,
    ) -> np.ndarray:
        y_segments: list[np.ndarray] = []
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
            return np.asarray([], dtype=float)
        return np.concatenate(y_segments)

    @staticmethod
    def _autoscale_axis_y(
        axis,
        q_min: float,
        q_max: float,
        *,
        log_scale: bool,
    ) -> None:
        y_values = ExperimentalDataOverlayWindow._axis_y_values(
            axis,
            q_min,
            q_max,
            log_scale=log_scale,
        )
        if y_values.size == 0:
            return
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            if log_scale and y_min > 0.0:
                axis.set_ylim(y_min / 1.15, y_max * 1.15)
            else:
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
        left_min: float,
        left_max: float,
        right_min: float,
        right_max: float,
        *,
        log_scale: bool,
    ) -> tuple[float, float]:
        if log_scale:
            if (
                min(
                    left_limits[0],
                    left_limits[1],
                    left_min,
                    left_max,
                    right_min,
                    right_max,
                )
                <= 0.0
            ):
                log_scale = False
        if not log_scale:
            left_low, left_high = left_limits
            data_low, data_high = sorted((left_min, left_max))
            right_low_data, right_high_data = sorted((right_min, right_max))
            if np.isclose(right_high_data, right_low_data):
                padding = max(abs(right_low_data) * 0.05, 1e-12)
                right_low_data -= padding
                right_high_data += padding
            if np.isclose(left_high, left_low) or np.isclose(
                data_high,
                data_low,
            ):
                padding = max(abs(right_low_data) * 0.1, 1e-12)
                return (
                    right_low_data - padding,
                    right_high_data + padding,
                )
            p0 = (data_low - left_low) / (left_high - left_low)
            p1 = (data_high - left_low) / (left_high - left_low)
            if np.isclose(p1, p0):
                padding = max(abs(right_low_data) * 0.1, 1e-12)
                return (
                    right_low_data - padding,
                    right_high_data + padding,
                )
            delta = (right_high_data - right_low_data) / (p1 - p0)
            right_low = right_low_data - p0 * delta
            right_high = right_low + delta
            return right_low, right_high

        left_logs = np.log10(np.asarray(left_limits, dtype=float))
        data_logs = np.log10(
            np.asarray(sorted((left_min, left_max)), dtype=float)
        )
        right_logs = np.log10(
            np.asarray(sorted((right_min, right_max)), dtype=float)
        )
        if np.isclose(left_logs[1], left_logs[0]) or np.isclose(
            data_logs[1],
            data_logs[0],
        ):
            return right_min / 1.2, right_max * 1.2
        p0 = (data_logs[0] - left_logs[0]) / (left_logs[1] - left_logs[0])
        p1 = (data_logs[1] - left_logs[0]) / (left_logs[1] - left_logs[0])
        if np.isclose(p1, p0):
            return right_min / 1.2, right_max * 1.2
        delta = (right_logs[1] - right_logs[0]) / (p1 - p0)
        right_low_log = right_logs[0] - p0 * delta
        right_high_log = right_low_log + delta
        return 10**right_low_log, 10**right_high_log

    @staticmethod
    def _summary_column_label(
        summary: ExperimentalDataSummary,
        column_index: int | None,
    ) -> str:
        if column_index is None:
            return "None"
        if 0 <= column_index < len(summary.column_names):
            return summary.column_names[column_index]
        return f"Column {column_index + 1}"

    def _trace_column_text(self, trace: ExperimentalOverlayTrace) -> str:
        summary = trace.summary
        q_label = self._summary_column_label(summary, summary.q_column)
        i_label = self._summary_column_label(
            summary,
            summary.intensity_column,
        )
        text = f"q={q_label}, I={i_label}"
        if summary.error_column is not None:
            error_label = self._summary_column_label(
                summary,
                summary.error_column,
            )
            text += f", err={error_label}"
        text += f"; header rows={summary.header_rows}"
        return text

    @staticmethod
    def _trace_q_range_text(trace: ExperimentalOverlayTrace) -> str:
        q_values = np.asarray(trace.summary.q_values, dtype=float)
        q_values = q_values[np.isfinite(q_values)]
        if q_values.size == 0:
            return "--"
        q_min = float(np.nanmin(q_values))
        q_max = float(np.nanmax(q_values))
        return f"{q_min:.6g} - {q_max:.6g}"


_OPEN_WINDOWS: list[ExperimentalDataOverlayWindow] = []


def _forget_open_window(window: ExperimentalDataOverlayWindow) -> None:
    _OPEN_WINDOWS[:] = [
        existing for existing in _OPEN_WINDOWS if existing is not window
    ]


def launch_experimental_data_overlay_ui(
    *,
    initial_paths: Iterable[str | Path] | None = None,
) -> ExperimentalDataOverlayWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ExperimentalDataOverlayWindow(initial_paths=initial_paths)
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window


__all__ = [
    "ExperimentalDataOverlayWindow",
    "ExperimentalOverlayTrace",
    "launch_experimental_data_overlay_ui",
]
