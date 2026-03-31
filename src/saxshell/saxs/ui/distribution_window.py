from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import stats

from saxshell.saxs.dream import BASE_DISTRIBUTIONS, DreamParameterEntry

SMART_PRIOR_PRESET_ITEMS: tuple[tuple[str, str], ...] = (
    ("Very Strict", "very_strict"),
    ("Strict", "strict"),
    ("Proportional (Current Default)", "proportional"),
    ("Lenient", "lenient"),
    ("Very Lenient", "very_lenient"),
    ("Strict Small / Lenient Large", "strict_small_lenient_large"),
    ("Lenient Small / Strict Large", "lenient_small_strict_large"),
)
SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS: tuple[tuple[str, str], ...] = (
    ("Custom / Manual", "custom"),
    ("Very Strict", "very_strict"),
    ("Strict", "strict"),
    ("Proportional", "proportional"),
    ("Lenient", "lenient"),
    ("Very Lenient", "very_lenient"),
)
SMART_PRIOR_APPLY_SCOPE_ITEMS: tuple[tuple[str, str], ...] = (
    ("All Parameters", "all"),
    ("Selected Parameters", "selected"),
)
SMART_PRIOR_SPREAD_FACTORS: dict[str, float] = {
    "very_strict": 0.4,
    "strict": 0.65,
    "proportional": 1.0,
    "lenient": 1.5,
    "very_lenient": 2.25,
}
SMART_PRIOR_STATUS_LABELS: dict[str, str] = {
    value: label for label, value in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS
}
GUIDE_INTERVAL_LOWER_Q = float(stats.norm.cdf(-3.0))
GUIDE_INTERVAL_UPPER_Q = float(stats.norm.cdf(3.0))
GUIDE_LOW_COLUMN = 9
GUIDE_HIGH_COLUMN = 10
RESET_COLUMN = 11
PLOT_DOMAIN_LOWER_Q = 1e-6
PLOT_DOMAIN_UPPER_Q = 1.0 - PLOT_DOMAIN_LOWER_Q
PLOT_RELATIVE_DENSITY_THRESHOLD = 0.01
PLOT_PADDING_FRACTION = 0.08
PLOT_WINDOW_MARGIN_FRACTION = 0.12
PLOT_SAMPLE_COUNT = 600
PLOT_LOG_SCALE_SPAN_RATIO = 25.0
GUIDE_INTERVAL_SIGMA = 3.0
INTERACTIVE_PARAMETER_EPSILON = 1e-6
INTERACTIVE_MAX_LOGNORM_SHAPE = 5.0
INTERACTIVE_PEAK_DRAG_SENSITIVITY = 3.0
INTERACTIVE_PEAK_HANDLE_SIZE = 90
INTERACTIVE_CENTER_HANDLE_SIZE = 78
INTERACTIVE_WIDTH_HANDLE_SIZE = 70
INTERACTIVE_CENTER_HANDLE_Y_FRACTION = 0.88
INTERACTIVE_WIDTH_HANDLE_Y_FRACTION = 0.18
INTERACTIVE_PREVIEW_THROTTLE_MS = 20


@dataclass(slots=True)
class _InteractiveHandleArtists:
    axis: object
    peak: object | None = None
    center: object | None = None
    left_width: object | None = None
    right_width: object | None = None


@dataclass(slots=True)
class _InteractiveDragState:
    row: int
    kind: str
    start_entry: DreamParameterEntry
    preview_entry: DreamParameterEntry
    start_y: float
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    x_scale: str


@dataclass(slots=True)
class _PlotWindowState:
    row: int
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    x_scale: str


class WeightDistributionPreviewWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[DreamParameterEntry] = []
        self._parameter_checkboxes: list[
            tuple[DreamParameterEntry, QCheckBox]
        ] = []
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("DREAM Prior Preview")
        self.resize(1080, 640)

        central = QWidget()
        root = QHBoxLayout(central)

        controls = QWidget()
        controls.setMinimumWidth(240)
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("Visible parameters"))
        helper = QLabel("Only w<##> priors are enabled by default.")
        helper.setWordWrap(True)
        controls_layout.addWidget(helper)
        self.parameter_scroll = QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.parameter_scroll.setMinimumWidth(240)
        self.parameter_checkbox_container = QWidget()
        self.parameter_checkbox_layout = QVBoxLayout(
            self.parameter_checkbox_container
        )
        self.parameter_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_checkbox_layout.setSpacing(4)
        self.parameter_checkbox_layout.addStretch(1)
        self.parameter_scroll.setWidget(self.parameter_checkbox_container)
        controls_layout.addWidget(self.parameter_scroll)

        plot_panel = QWidget()
        layout = QVBoxLayout(plot_panel)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        root.addWidget(controls, stretch=1)
        root.addWidget(plot_panel, stretch=3)
        self.setCentralWidget(central)

    def plot_entries(self, entries: list[DreamParameterEntry]) -> None:
        self._entries = list(entries)
        self._rebuild_parameter_checkboxes()
        self._refresh_plot()

    @staticmethod
    def _is_weight_parameter(param_name: str) -> bool:
        return re.fullmatch(r"w\d+", param_name.strip()) is not None

    @staticmethod
    def _entry_label(entry: DreamParameterEntry) -> str:
        legend_label = entry.param.strip() or "Unnamed parameter"
        if entry.structure.strip():
            legend_label = f"{legend_label} ({entry.structure.strip()})"
        return legend_label

    @staticmethod
    def _entry_toggle_key(entry: DreamParameterEntry) -> tuple[str, ...]:
        return (
            entry.structure.strip(),
            entry.motif.strip(),
            entry.param_type.strip(),
            entry.param.strip(),
        )

    def _rebuild_parameter_checkboxes(self) -> None:
        prior_states = {
            self._entry_toggle_key(entry): checkbox.isChecked()
            for entry, checkbox in self._parameter_checkboxes
        }
        self._parameter_checkboxes = []
        while self.parameter_checkbox_layout.count():
            item = self.parameter_checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self._entries:
            empty_label = QLabel("No prior distributions are available.")
            empty_label.setWordWrap(True)
            self.parameter_checkbox_layout.addWidget(empty_label)
            self.parameter_checkbox_layout.addStretch(1)
            return
        for entry in self._entries:
            checkbox = QCheckBox(self._entry_label(entry))
            checkbox.setChecked(
                prior_states.get(
                    self._entry_toggle_key(entry),
                    self._is_weight_parameter(entry.param),
                )
            )
            tooltip_parts = []
            if entry.structure.strip():
                tooltip_parts.append(f"Structure: {entry.structure.strip()}")
            if entry.motif.strip():
                tooltip_parts.append(f"Motif: {entry.motif.strip()}")
            if tooltip_parts:
                checkbox.setToolTip("\n".join(tooltip_parts))
            checkbox.toggled.connect(
                lambda _checked=False: self._refresh_plot()
            )
            self.parameter_checkbox_layout.addWidget(checkbox)
            self._parameter_checkboxes.append((entry, checkbox))
        self.parameter_checkbox_layout.addStretch(1)

    def _refresh_plot(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        x_limits: list[tuple[float, float]] = []
        plotted = 0
        selected_entries = [
            entry
            for entry, checkbox in self._parameter_checkboxes
            if checkbox.isChecked()
        ]

        for entry in selected_entries:
            try:
                distribution = getattr(stats, entry.distribution)
                x_min, x_max = _distribution_domain(entry)
                if not np.isfinite(x_min) or not np.isfinite(x_max):
                    continue
                x_values = np.linspace(x_min, x_max, 300)
                y_values = distribution.pdf(x_values, **entry.dist_params)
            except Exception:
                continue
            if not np.all(np.isfinite(y_values)):
                continue
            axis.plot(
                x_values,
                y_values,
                linewidth=1.6,
                label=self._entry_label(entry),
            )
            x_limits.append((x_min, x_max))
            plotted += 1

        if plotted == 0:
            message = (
                "No prior distributions are currently enabled in the preview."
                if not selected_entries
                else "No valid prior distributions are available for the selected parameters."
            )
            axis.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        else:
            axis.set_xlabel("Value")
            axis.set_ylabel("Density")
            axis.set_title("Prior distributions")
            axis.legend(loc="best", fontsize="small")
            axis.set_xlim(
                min(limit[0] for limit in x_limits),
                max(limit[1] for limit in x_limits),
            )

        self.figure.tight_layout()
        self.canvas.draw()


class DistributionSetupWindow(QMainWindow):
    saved = Signal(list)
    _session_skip_effective_radius_vary_warning = False

    def __init__(
        self,
        entries: list[DreamParameterEntry],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._entries = entries
        self._has_existing_parameter_map = False
        self._was_saved = False
        self._suppress_vary_warning = False
        self._suppress_status_change = False
        self._reset_entries: list[DreamParameterEntry] = []
        self._weight_preview_window: WeightDistributionPreviewWindow | None = (
            None
        )
        self._interactive_handles: _InteractiveHandleArtists | None = None
        self._drag_state: _InteractiveDragState | None = None
        self._plot_window_state: _PlotWindowState | None = None
        self._pending_drag_preview: tuple[int, DreamParameterEntry] | None = (
            None
        )
        self._interactive_preview_timer = QTimer(self)
        self._interactive_preview_timer.setSingleShot(True)
        self._interactive_preview_timer.timeout.connect(
            self._flush_interactive_drag_preview
        )
        self._build_ui()
        self.load_entries(entries)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXS DREAM Prior Setup")
        self.resize(1200, 720)

        central = QWidget()
        root = QVBoxLayout(central)

        self._left_panel = QWidget()
        left_layout = QVBoxLayout(self._left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        top_row = QHBoxLayout()
        self.preview_weight_priors_button = QPushButton("Preview Priors")
        self.preview_weight_priors_button.setToolTip(
            "Open a shared density plot of the current prior distributions. "
            "Only w<##> parameters are enabled by default in the preview."
        )
        self.preview_weight_priors_button.clicked.connect(
            self._show_weight_prior_preview
        )
        top_row.addWidget(self.preview_weight_priors_button)
        top_row.addWidget(QLabel("Smart prior preset"))
        self.smart_prior_preset_combo = QComboBox()
        for label, value in SMART_PRIOR_PRESET_ITEMS:
            self.smart_prior_preset_combo.addItem(label, userData=value)
        self.smart_prior_preset_combo.setToolTip(
            "Apply a preset tightening or relaxation pattern to the current "
            "prior distributions. The current table settings act as the "
            "baseline."
        )
        top_row.addWidget(self.smart_prior_preset_combo)
        top_row.addWidget(QLabel("Apply to"))
        self.smart_prior_apply_scope_combo = QComboBox()
        for label, value in SMART_PRIOR_APPLY_SCOPE_ITEMS:
            self.smart_prior_apply_scope_combo.addItem(label, userData=value)
        self.smart_prior_apply_scope_combo.setToolTip(
            "Choose whether the selected preset should affect every "
            "parameter row in the table or only the currently selected "
            "parameter rows. Size-aware mixed presets always apply across "
            "the full table so their relative ranking remains meaningful."
        )
        top_row.addWidget(self.smart_prior_apply_scope_combo)
        self.apply_smart_prior_preset_button = QPushButton(
            "Apply Smart Preset"
        )
        self.apply_smart_prior_preset_button.setToolTip(
            "Adjust the current prior widths using the selected preset. "
            "Size-aware presets use effective-radius parameters to identify "
            "relatively small and large cluster weights."
        )
        self.apply_smart_prior_preset_button.clicked.connect(
            self._apply_smart_prior_preset
        )
        top_row.addWidget(self.apply_smart_prior_preset_button)
        top_row.addStretch(1)
        left_layout.addLayout(top_row)
        self.table = QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(
            [
                "Structure",
                "Motif",
                "Param Type",
                "Param",
                "Value",
                "Vary",
                "Distribution",
                "Distribution Params",
                "Smart Preset Status",
                "Guide Low",
                "Guide High",
                "Reset",
            ]
        )
        guide_tooltip = (
            "Practical prior bounds for the current distribution. "
            "Bounded priors use their exact support, while unbounded priors "
            "use a central 99.73% interval (Gaussian 3sigma equivalent)."
        )
        low_header = self.table.horizontalHeaderItem(GUIDE_LOW_COLUMN)
        if low_header is not None:
            low_header.setToolTip(guide_tooltip)
        high_header = self.table.horizontalHeaderItem(GUIDE_HIGH_COLUMN)
        if high_header is not None:
            high_header.setToolTip(guide_tooltip)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.cellClicked.connect(self._on_row_selected)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)
        self.table.cellChanged.connect(self._on_table_changed)

        button_row = QHBoxLayout()
        self.select_recommended_vary_button = QPushButton(
            "Select Weights + Model Params"
        )
        self.select_recommended_vary_button.setToolTip(
            "Enable vary for weights and other model parameters, but keep "
            "effective-radius parameters turned off."
        )
        self.select_recommended_vary_button.clicked.connect(
            self._set_recommended_vary_selection
        )
        button_row.addWidget(self.select_recommended_vary_button)
        self.set_all_vary_on_button = QPushButton("Set All Vary On")
        self.set_all_vary_on_button.clicked.connect(
            lambda: self._set_all_vary(True)
        )
        button_row.addWidget(self.set_all_vary_on_button)
        self.set_all_vary_off_button = QPushButton("Set All Vary Off")
        self.set_all_vary_off_button.clicked.connect(
            lambda: self._set_all_vary(False)
        )
        button_row.addWidget(self.set_all_vary_off_button)
        save_button = QPushButton("Save Parameter Map")
        save_button.clicked.connect(self._emit_saved)
        button_row.addWidget(save_button)
        button_row.addStretch(1)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(160)
        self._editor_panel = QWidget()
        editor_layout = QVBoxLayout(self._editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.table, stretch=1)
        editor_layout.addLayout(button_row)

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.setHandleWidth(10)
        self._left_splitter.addWidget(self._editor_panel)
        self._left_splitter.addWidget(self.console)
        self._left_splitter.setStretchFactor(0, 5)
        self._left_splitter.setStretchFactor(1, 2)
        self._left_splitter.setSizes([520, 180])

        left_layout.addWidget(self._left_splitter, stretch=1)

        self._plot_panel = QWidget()
        right_layout = QVBoxLayout(self._plot_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        interaction_row = QHBoxLayout()
        interaction_row.addWidget(QLabel("Interactive plot editing"))
        self.rescale_axes_button = QPushButton("Rescale Axes")
        self.rescale_axes_button.setToolTip(
            "Refit the x- and y-axis limits to the currently selected prior."
        )
        self.rescale_axes_button.clicked.connect(self._rescale_current_plot)
        interaction_row.addWidget(self.rescale_axes_button)
        interaction_row.addStretch(1)
        self.lock_center_checkbox = QCheckBox("Lock center")
        self.lock_center_checkbox.setChecked(True)
        self.lock_center_checkbox.setToolTip(
            "Keep the prior center fixed while dragging width and peak "
            "handles. Uncheck this to enable dragging the red center "
            "marker."
        )
        self.lock_center_checkbox.toggled.connect(self._on_center_lock_toggled)
        interaction_row.addWidget(self.lock_center_checkbox)
        right_layout.addLayout(interaction_row)
        self.interactive_hint_label = QLabel()
        self.interactive_hint_label.setWordWrap(True)
        right_layout.addWidget(self.interactive_hint_label)
        self._refresh_interactive_hint()
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect(
            "button_press_event", self._on_plot_mouse_press
        )
        self.canvas.mpl_connect(
            "motion_notify_event", self._on_plot_mouse_move
        )
        self.canvas.mpl_connect(
            "button_release_event", self._on_plot_mouse_release
        )
        self.canvas.mpl_connect(
            "figure_leave_event", self._on_plot_mouse_leave
        )
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.setHandleWidth(10)
        self._main_splitter.addWidget(self._left_panel)
        self._main_splitter.addWidget(self._plot_panel)
        self._main_splitter.setStretchFactor(0, 4)
        self._main_splitter.setStretchFactor(1, 3)
        self._main_splitter.setSizes([720, 560])

        root.addWidget(self._main_splitter, stretch=1)
        self.setCentralWidget(central)

    def load_entries(
        self,
        entries: list[DreamParameterEntry],
        *,
        has_existing_parameter_map: bool | None = None,
        update_reset_entries: bool = True,
    ) -> None:
        if has_existing_parameter_map is not None:
            self._has_existing_parameter_map = bool(has_existing_parameter_map)
            self._was_saved = False
        normalized_entries = [
            self._normalized_entry_copy(entry) for entry in entries
        ]
        self._plot_window_state = None
        self._pending_drag_preview = None
        self._interactive_preview_timer.stop()
        if update_reset_entries:
            self._reset_entries = [
                self._normalized_entry_copy(entry)
                for entry in normalized_entries
            ]
        self.table.blockSignals(True)
        self.table.setRowCount(len(normalized_entries))
        for row, entry in enumerate(normalized_entries):
            params = dict(entry.dist_params)
            self.table.setItem(row, 0, QTableWidgetItem(entry.structure))
            self.table.setItem(row, 1, QTableWidgetItem(entry.motif))
            self.table.setItem(row, 2, QTableWidgetItem(entry.param_type))
            self.table.setItem(row, 3, QTableWidgetItem(entry.param))
            self.table.setItem(
                row,
                4,
                QTableWidgetItem(f"{entry.value:.6g}"),
            )
            vary_box = QCheckBox()
            vary_box.setChecked(entry.vary)
            vary_box.toggled.connect(
                lambda checked, selected_row=row: self._on_vary_toggled(
                    selected_row,
                    checked,
                )
            )
            self.table.setCellWidget(row, 5, vary_box)
            combo = QComboBox()
            combo.addItems(list(BASE_DISTRIBUTIONS))
            combo.setCurrentText(entry.distribution)
            combo.currentTextChanged.connect(
                lambda _text, selected_row=row: self._on_distribution_changed(
                    selected_row
                )
            )
            self.table.setCellWidget(row, 6, combo)
            self.table.setItem(
                row,
                7,
                QTableWidgetItem(json.dumps(params, sort_keys=True)),
            )
            status_combo = QComboBox()
            for label, value in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS:
                status_combo.addItem(label, userData=value)
            status_value = self._normalized_smart_status(
                getattr(entry, "smart_preset_status", "custom")
            )
            status_index = status_combo.findData(status_value)
            if status_index < 0:
                status_index = status_combo.findData("custom")
            self._suppress_status_change = True
            try:
                status_combo.setCurrentIndex(max(status_index, 0))
            finally:
                self._suppress_status_change = False
            status_combo.currentIndexChanged.connect(
                lambda _index, selected_row=row, combo=status_combo: (
                    self._on_smart_status_changed(selected_row, combo)
                )
            )
            self.table.setCellWidget(row, 8, status_combo)
            self._refresh_distribution_guides_for_row(
                row,
                entry=entry,
            )
            reset_button = QPushButton("Reset")
            reset_button.setToolTip(
                "Reset this prior row to the most recently loaded or saved "
                "parameter-map values."
            )
            reset_button.clicked.connect(
                lambda _checked=False, selected_row=row: (
                    self._reset_row_to_baseline(selected_row)
                )
            )
            self.table.setCellWidget(row, RESET_COLUMN, reset_button)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._entries = normalized_entries
        if normalized_entries:
            self.table.setCurrentCell(0, 0)
            self._plot_entry(
                normalized_entries[0],
                row=0,
                force_rescale=True,
            )
            return
        self.figure.clear()
        self._interactive_handles = None
        self.canvas.draw()

    def _entry_from_row(self, row: int) -> DreamParameterEntry:
        distribution_widget = self.table.cellWidget(row, 6)
        vary_widget = self.table.cellWidget(row, 5)
        distribution = (
            distribution_widget.currentText()
            if isinstance(distribution_widget, QComboBox)
            else "lognorm"
        )
        value_item = self.table.item(row, 4)
        params_item = self.table.item(row, 7)
        status_value = self._row_smart_status(row)
        value = float(value_item.text()) if value_item is not None else 0.0
        params = self._normalize_distribution_params(
            distribution,
            self._parse_params(params_item.text() if params_item else "{}"),
            value,
        )
        return DreamParameterEntry(
            structure=self.table.item(row, 0).text(),
            motif=self.table.item(row, 1).text(),
            param_type=self.table.item(row, 2).text(),
            param=self.table.item(row, 3).text(),
            value=value,
            vary=(
                vary_widget.isChecked()
                if isinstance(vary_widget, QCheckBox)
                else False
            ),
            distribution=distribution,
            dist_params=params,
            smart_preset_status=status_value,
        )

    def current_entries(self) -> list[DreamParameterEntry]:
        return [
            self._entry_from_row(row) for row in range(self.table.rowCount())
        ]

    def _normalized_entry_copy(
        self,
        entry: DreamParameterEntry,
    ) -> DreamParameterEntry:
        return DreamParameterEntry(
            structure=str(entry.structure),
            motif=str(entry.motif),
            param_type=str(entry.param_type),
            param=str(entry.param),
            value=float(entry.value),
            vary=bool(entry.vary),
            distribution=str(entry.distribution),
            dist_params=self._normalize_distribution_params(
                str(entry.distribution),
                dict(entry.dist_params),
                float(entry.value),
            ),
            smart_preset_status=self._normalized_smart_status(
                getattr(entry, "smart_preset_status", "custom")
            ),
        )

    def _emit_saved(self) -> None:
        entries = self.current_entries()
        self._was_saved = True
        self._has_existing_parameter_map = True
        self._reset_entries = [
            self._normalized_entry_copy(entry) for entry in entries
        ]
        self.saved.emit(entries)
        self.console.append("Saved current DREAM parameter map.")
        QMessageBox.information(
            self,
            "Saved",
            "The DREAM parameter map was updated.",
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._has_existing_parameter_map or self._was_saved:
            super().closeEvent(event)
            return
        response = QMessageBox.question(
            self,
            "Quit without saving parameter map?",
            (
                "No DREAM parameter map has been saved for this project yet. "
                "Are you sure you want to quit Edit Priors without saving "
                "the parameter map?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response == QMessageBox.StandardButton.Yes:
            super().closeEvent(event)
            return
        event.ignore()

    def _on_row_selected(self, row: int, _column: int) -> None:
        self._plot_entry(
            self._entry_from_row(row),
            row=row,
            force_rescale=True,
        )

    def _on_current_cell_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if current_row < 0 or current_row >= self.table.rowCount():
            return
        self._plot_entry(
            self._entry_from_row(current_row),
            row=current_row,
            force_rescale=True,
        )

    def _on_distribution_changed(self, row: int) -> None:
        entry = self._entry_from_row(row)
        self.table.blockSignals(True)
        self.table.item(row, 7).setText(
            json.dumps(entry.dist_params, sort_keys=True)
        )
        self.table.blockSignals(False)
        self._refresh_distribution_guides_for_row(row, entry=entry)
        self.console.append(
            f"Distribution for {entry.param} set to {entry.distribution}."
        )
        self._set_group_status_for_row(row, "custom")
        self._plot_entry(entry, row=row, force_rescale=True)

    def _on_table_changed(self, row: int, column: int) -> None:
        if column == 7:
            try:
                entry = self._entry_from_row(row)
            except Exception as exc:
                self._refresh_distribution_guides_for_row(row, entry=None)
                self.console.append(
                    f"Invalid distribution parameter JSON: {exc}"
                )
                return
            self._refresh_distribution_guides_for_row(row, entry=entry)
            self._set_group_status_for_row(row, "custom")
            if row == self.table.currentRow():
                self._plot_entry(entry, row=row)
            return
        if column == 4:
            try:
                entry = self._entry_from_row(row)
            except Exception:
                entry = None
            self._refresh_distribution_guides_for_row(row, entry=entry)
            self._set_group_status_for_row(row, "custom")
            if entry is not None and row == self.table.currentRow():
                self._plot_entry(entry, row=row)
            return
        if column in {GUIDE_LOW_COLUMN, GUIDE_HIGH_COLUMN}:
            self._on_distribution_guide_changed(row, column)

    def _reset_row_to_baseline(self, row: int) -> None:
        if row < 0 or row >= len(self._reset_entries):
            QMessageBox.warning(
                self,
                "Reset prior failed",
                "No baseline prior entry is available for the selected row.",
            )
            return
        self._apply_entry_to_row(row, self._reset_entries[row])
        self.console.append(
            "Reset prior row to the loaded/saved baseline: "
            f"{self._row_status_label(row)}."
        )

    def _apply_entry_to_row(
        self,
        row: int,
        entry: DreamParameterEntry,
    ) -> None:
        if row < 0 or row >= self.table.rowCount():
            return
        normalized_entry = self._normalized_entry_copy(entry)
        distribution_combo = self.table.cellWidget(row, 6)
        vary_box = self.table.cellWidget(row, 5)
        status_combo = self.table.cellWidget(row, 8)
        was_table_blocked = self.table.blockSignals(True)
        prior_vary_suppressed = self._suppress_vary_warning
        prior_status_suppressed = self._suppress_status_change
        self._suppress_vary_warning = True
        self._suppress_status_change = True
        try:
            self._set_editable_table_item(row, 0, normalized_entry.structure)
            self._set_editable_table_item(row, 1, normalized_entry.motif)
            self._set_editable_table_item(row, 2, normalized_entry.param_type)
            self._set_editable_table_item(row, 3, normalized_entry.param)
            self._set_editable_table_item(
                row,
                4,
                f"{normalized_entry.value:.6g}",
            )
            if isinstance(vary_box, QCheckBox):
                vary_box.blockSignals(True)
                vary_box.setChecked(normalized_entry.vary)
                vary_box.blockSignals(False)
            if isinstance(distribution_combo, QComboBox):
                distribution_combo.blockSignals(True)
                distribution_combo.setCurrentText(
                    normalized_entry.distribution
                )
                distribution_combo.blockSignals(False)
            self._set_editable_table_item(
                row,
                7,
                json.dumps(normalized_entry.dist_params, sort_keys=True),
            )
            if isinstance(status_combo, QComboBox):
                status_index = status_combo.findData(
                    normalized_entry.smart_preset_status
                )
                if status_index < 0:
                    status_index = status_combo.findData("custom")
                status_combo.blockSignals(True)
                status_combo.setCurrentIndex(max(status_index, 0))
                status_combo.blockSignals(False)
        finally:
            self._suppress_vary_warning = prior_vary_suppressed
            self._suppress_status_change = prior_status_suppressed
            self.table.blockSignals(was_table_blocked)
        self._refresh_distribution_guides_for_row(row, entry=normalized_entry)
        if row == self.table.currentRow():
            self._plot_entry(normalized_entry, row=row)

    def _set_editable_table_item(
        self,
        row: int,
        column: int,
        text: str,
    ) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(text)

    def _set_all_vary(self, enabled: bool) -> None:
        self._set_vary_state_for_rows(lambda _row_index, _param_name: enabled)
        self.console.append(
            "Set all DREAM vary flags " + ("on." if enabled else "off.")
        )

    def _apply_smart_prior_preset(self) -> None:
        preset_mode = (
            str(
                self.smart_prior_preset_combo.currentData() or "proportional"
            ).strip()
            or "proportional"
        )
        apply_scope = (
            str(
                self.smart_prior_apply_scope_combo.currentData() or "all"
            ).strip()
            or "all"
        )
        effective_scope = (
            "all"
            if preset_mode
            in {
                "strict_small_lenient_large",
                "lenient_small_strict_large",
            }
            else apply_scope
        )
        entries = self.current_entries()
        try:
            updated_entries = self._smart_adjusted_entries(
                entries,
                preset_mode=preset_mode,
                apply_scope=effective_scope,
                selected_rows=self._selected_row_indexes(),
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to apply smart prior preset",
                str(exc),
            )
            self.console.append(
                "Unable to apply smart prior preset: " + str(exc)
            )
            return
        self.load_entries(
            updated_entries,
            has_existing_parameter_map=self._has_existing_parameter_map,
            update_reset_entries=False,
        )
        preset_label = str(
            self.smart_prior_preset_combo.currentText() or preset_mode
        ).strip()
        scope_label = str(
            self.smart_prior_apply_scope_combo.currentText() or apply_scope
        ).strip()
        if preset_mode in {
            "strict_small_lenient_large",
            "lenient_small_strict_large",
        }:
            scope_label = "All Parameters"
        self.console.append(
            f"Applied smart prior preset: {preset_label} ({scope_label})."
        )

    def _set_recommended_vary_selection(self) -> None:
        self._set_vary_state_for_rows(
            lambda _row_index, param_name: (
                not self._is_effective_radius_parameter(param_name)
            )
        )
        self.console.append(
            "Enabled vary for weights and model parameters while keeping "
            "effective-radius parameters off."
        )

    def _set_vary_state_for_rows(
        self,
        selector,
    ) -> None:
        self._suppress_vary_warning = True
        try:
            for row in range(self.table.rowCount()):
                vary_box = self.table.cellWidget(row, 5)
                if not isinstance(vary_box, QCheckBox):
                    continue
                param_name = self.table.item(row, 3).text().strip()
                vary_box.setChecked(bool(selector(row, param_name)))
        finally:
            self._suppress_vary_warning = False

    def _on_vary_toggled(self, row: int, checked: bool) -> None:
        if self._suppress_vary_warning or not checked:
            return
        if row < 0 or row >= self.table.rowCount():
            return
        param_name = self.table.item(row, 3).text().strip()
        if not self._is_effective_radius_parameter(param_name):
            return
        self.console.append(
            "Warning: effective-radius parameters are not recommended for "
            f"DREAM variation ({param_name})."
        )
        self._show_effective_radius_vary_warning(param_name)

    def _show_effective_radius_vary_warning(self, param_name: str) -> None:
        if type(self)._session_skip_effective_radius_vary_warning:
            return
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Effective radius variation warning")
        dialog.setText(
            "It is not recommended to vary effective-radius parameters in "
            "the DREAM prior map."
        )
        dialog.setInformativeText(f"Selected parameter: {param_name}")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        skip_checkbox = QCheckBox(
            "Don't show this type of warning again during this session"
        )
        dialog.setCheckBox(skip_checkbox)
        dialog.exec()
        if skip_checkbox.isChecked():
            type(self)._session_skip_effective_radius_vary_warning = True

    def _show_weight_prior_preview(self) -> None:
        entries = self.current_entries()
        if not entries:
            QMessageBox.information(
                self,
                "No priors available",
                "No prior distributions are currently available in the table.",
            )
            return
        if self._weight_preview_window is None:
            self._weight_preview_window = WeightDistributionPreviewWindow()
        self._weight_preview_window.plot_entries(entries)
        self._weight_preview_window.show()
        self._weight_preview_window.raise_()
        self._weight_preview_window.activateWindow()
        self.console.append(
            "Opened shared prior preview with only w<##> parameters enabled by default."
        )

    def _refresh_interactive_hint(self) -> None:
        if self.lock_center_checkbox.isChecked():
            message = (
                "Drag the orange peak handle to sharpen or broaden the "
                "distribution, or drag the blue side handles to resize the "
                "width. The center is currently locked. The gray dashed "
                "curve shows the reset baseline."
            )
        else:
            message = (
                "Drag the orange peak handle to adjust peak height and "
                "width, drag the blue side handles to resize the width, or "
                "drag the red center handle to reposition the prior. The "
                "gray dashed curve shows the reset baseline."
            )
        self.interactive_hint_label.setText(message)

    def _on_center_lock_toggled(self, _checked: bool) -> None:
        if (
            self._drag_state is not None
            and self._drag_state.kind == "center"
            and self.lock_center_checkbox.isChecked()
        ):
            self._drag_state = None
        self._refresh_interactive_hint()
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= self.table.rowCount():
            return
        try:
            self._plot_entry(
                self._entry_from_row(current_row),
                row=current_row,
            )
        except Exception:
            return

    def _rescale_current_plot(self) -> None:
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= self.table.rowCount():
            return
        try:
            self._plot_entry(
                self._entry_from_row(current_row),
                row=current_row,
                force_rescale=True,
            )
        except Exception:
            return

    def _on_plot_mouse_press(self, event) -> None:
        if event is None or event.button != 1 or event.inaxes is None:
            return
        if self._interactive_handles is None:
            return
        if event.inaxes is not self._interactive_handles.axis:
            return
        handle_kind = self._interactive_handle_kind_at_event(event)
        if handle_kind is None:
            return
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount():
            return
        try:
            start_entry = self._entry_from_row(row)
        except Exception:
            return
        self._interactive_preview_timer.stop()
        self._pending_drag_preview = None
        y_limits = event.inaxes.get_ylim()
        self._drag_state = _InteractiveDragState(
            row=row,
            kind=handle_kind,
            start_entry=DreamParameterEntry.from_dict(start_entry.to_dict()),
            preview_entry=DreamParameterEntry.from_dict(start_entry.to_dict()),
            start_y=self._clamped_event_y(
                event.ydata,
                y_limits=y_limits,
            ),
            x_limits=tuple(float(limit) for limit in event.inaxes.get_xlim()),
            y_limits=tuple(float(limit) for limit in y_limits),
            x_scale=str(event.inaxes.get_xscale() or "linear"),
        )

    def _on_plot_mouse_move(self, event) -> None:
        if self._drag_state is None or event is None:
            return
        if event.inaxes is None:
            return
        preview_entry = self._interactive_drag_preview_entry(
            self._drag_state,
            event,
        )
        if preview_entry is None:
            return
        if self._entries_match(
            self._drag_state.preview_entry,
            preview_entry,
        ):
            return
        self._drag_state.preview_entry = preview_entry
        self._schedule_interactive_drag_preview(
            preview_entry,
            row=self._drag_state.row,
        )

    def _on_plot_mouse_release(self, _event) -> None:
        if self._drag_state is None:
            return
        self._interactive_preview_timer.stop()
        self._pending_drag_preview = None
        drag_state = self._drag_state
        self._drag_state = None
        if self._entries_match(
            drag_state.start_entry,
            drag_state.preview_entry,
        ):
            self._plot_entry(drag_state.start_entry, row=drag_state.row)
            return
        self._apply_entry_to_row(drag_state.row, drag_state.preview_entry)
        self._set_group_status_for_row(drag_state.row, "custom")
        self.console.append(
            self._interactive_drag_message(
                drag_state.start_entry,
                drag_state.preview_entry,
                handle_kind=drag_state.kind,
            )
        )

    def _on_plot_mouse_leave(self, _event) -> None:
        self._on_plot_mouse_release(None)

    def _schedule_interactive_drag_preview(
        self,
        entry: DreamParameterEntry,
        *,
        row: int,
    ) -> None:
        self._pending_drag_preview = (
            int(row),
            DreamParameterEntry.from_dict(entry.to_dict()),
        )
        if not self._interactive_preview_timer.isActive():
            self._interactive_preview_timer.start(
                INTERACTIVE_PREVIEW_THROTTLE_MS
            )

    def _flush_interactive_drag_preview(self) -> None:
        if self._pending_drag_preview is None:
            return
        row, entry = self._pending_drag_preview
        self._pending_drag_preview = None
        self._plot_entry(entry, row=row, interactive_preview=True)

    def _interactive_handle_kind_at_event(self, event) -> str | None:
        if self._interactive_handles is None:
            return None
        for handle_kind, artist in (
            ("peak", self._interactive_handles.peak),
            ("center", self._interactive_handles.center),
            ("left_width", self._interactive_handles.left_width),
            ("right_width", self._interactive_handles.right_width),
        ):
            if artist is None:
                continue
            contains, _details = artist.contains(event)
            if contains:
                return handle_kind
        return None

    def _interactive_drag_preview_entry(
        self,
        drag_state: _InteractiveDragState,
        event,
    ) -> DreamParameterEntry | None:
        if drag_state.kind == "peak":
            target_y = self._clamped_event_y(
                event.ydata,
                y_limits=drag_state.y_limits,
            )
            return self._peak_drag_adjusted_entry(
                drag_state.start_entry,
                start_y=drag_state.start_y,
                target_y=target_y,
                y_limits=drag_state.y_limits,
            )
        target_x = self._clamped_event_x(
            event.xdata,
            x_limits=drag_state.x_limits,
            x_scale=drag_state.x_scale,
        )
        if target_x is None:
            return None
        if drag_state.kind == "center":
            if self.lock_center_checkbox.isChecked():
                return None
            return self._center_drag_adjusted_entry(
                drag_state.start_entry,
                target_center=target_x,
            )
        if drag_state.kind in {"left_width", "right_width"}:
            return self._width_drag_adjusted_entry(
                drag_state.start_entry,
                handle_kind=drag_state.kind,
                target_x=target_x,
            )
        return None

    def _baseline_entry_for_row(
        self,
        row: int,
    ) -> DreamParameterEntry | None:
        if row < 0 or row >= len(self._reset_entries):
            return None
        return self._normalized_entry_copy(self._reset_entries[row])

    def _current_plot_window_state(
        self,
        row: int,
    ) -> _PlotWindowState | None:
        if (
            row < 0
            or self._plot_window_state is None
            or self._plot_window_state.row != row
        ):
            return None
        if self.figure.axes:
            axis = self.figure.axes[0]
            return _PlotWindowState(
                row=row,
                x_limits=tuple(float(limit) for limit in axis.get_xlim()),
                y_limits=tuple(float(limit) for limit in axis.get_ylim()),
                x_scale=str(axis.get_xscale() or "linear"),
            )
        return self._plot_window_state

    @staticmethod
    def _plot_window_requires_rescale(
        current_state: _PlotWindowState | None,
        *,
        preferred_x_limits: tuple[float, float],
        preferred_y_limits: tuple[float, float],
        x_scale: str,
    ) -> bool:
        if current_state is None or current_state.x_scale != x_scale:
            return True
        current_x_low, current_x_high = current_state.x_limits
        preferred_x_low, preferred_x_high = preferred_x_limits
        current_y_low, current_y_high = current_state.y_limits
        preferred_y_low, preferred_y_high = preferred_y_limits
        x_span = max(abs(current_x_high - current_x_low), 1.0)
        y_span = max(abs(current_y_high - current_y_low), 1.0)
        x_tolerance = max(x_span * 1e-9, 1e-9)
        y_tolerance = max(y_span * 1e-9, 1e-9)
        return bool(
            preferred_x_low < current_x_low - x_tolerance
            or preferred_x_high > current_x_high + x_tolerance
            or preferred_y_low < current_y_low - y_tolerance
            or preferred_y_high > current_y_high + y_tolerance
        )

    def _plot_entry(
        self,
        entry: DreamParameterEntry,
        *,
        row: int | None = None,
        force_rescale: bool = False,
        interactive_preview: bool = False,
    ) -> None:
        plot_row = row if row is not None else int(self.table.currentRow())
        current_window = self._current_plot_window_state(plot_row)
        self.figure.clear()
        self._interactive_handles = None
        axis = self.figure.add_subplot(111)
        try:
            required_x_limits, required_y_limits, x_scale = (
                _distribution_plot_bounds(entry)
            )
            preferred_x_limits, preferred_y_limits, _preferred_scale = (
                _distribution_plot_window(entry)
            )
            if force_rescale or self._plot_window_requires_rescale(
                current_window,
                preferred_x_limits=required_x_limits,
                preferred_y_limits=required_y_limits,
                x_scale=x_scale,
            ):
                plot_window = _PlotWindowState(
                    row=plot_row,
                    x_limits=preferred_x_limits,
                    y_limits=preferred_y_limits,
                    x_scale=x_scale,
                )
            else:
                plot_window = current_window
                assert plot_window is not None
            x_values, y_values = _distribution_plot_curve(
                entry,
                x_limits=plot_window.x_limits,
                x_scale=plot_window.x_scale,
            )
            baseline_entry = self._baseline_entry_for_row(plot_row)
            baseline_series: tuple[np.ndarray, np.ndarray] | None = None
            if baseline_entry is not None:
                try:
                    baseline_series = _distribution_plot_curve(
                        baseline_entry,
                        x_limits=plot_window.x_limits,
                        x_scale=plot_window.x_scale,
                    )
                except Exception:
                    baseline_series = None
            axis.set_box_aspect(1.0)
            axis.set_xscale(plot_window.x_scale)
            if baseline_series is not None:
                (baseline_line,) = axis.plot(
                    baseline_series[0],
                    baseline_series[1],
                    color="tab:gray",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.45,
                    zorder=1,
                )
                baseline_line.set_gid("reset-baseline")
            (current_line,) = axis.plot(
                x_values,
                y_values,
                color="black",
                linewidth=1.8,
                zorder=3,
            )
            current_line.set_gid("current-distribution")
            axis.axvline(entry.value, color="tab:red", linestyle="--")
            guide_low, guide_high = self._interactive_width_handle_positions(
                entry
            )
            if guide_low is not None:
                axis.axvline(
                    guide_low,
                    color="tab:gray",
                    linestyle=":",
                    linewidth=1.0,
                )
            if guide_high is not None and not np.isclose(
                guide_high,
                guide_low,
            ):
                axis.axvline(
                    guide_high,
                    color="tab:gray",
                    linestyle=":",
                    linewidth=1.0,
                )
            axis.set_title(f"{entry.param}: {entry.distribution}")
            axis.set_xlabel("Value")
            axis.set_ylabel("Density")
            axis.set_xlim(*plot_window.x_limits)
            axis.set_ylim(*plot_window.y_limits)
            self._interactive_handles = self._draw_interactive_handles(
                axis,
                entry,
                x_values=x_values,
                y_values=y_values,
                x_scale=plot_window.x_scale,
                y_limits=plot_window.y_limits,
            )
            self._plot_window_state = plot_window
        except Exception as exc:
            self._plot_window_state = None
            axis.text(
                0.5,
                0.5,
                f"Unable to plot distribution:\n{exc}",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        if not interactive_preview:
            self.figure.tight_layout()
        if self._drag_state is None:
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

    def _draw_interactive_handles(
        self,
        axis,
        entry: DreamParameterEntry,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_scale: str,
        y_limits: tuple[float, float],
    ) -> _InteractiveHandleArtists:
        top_y = float(y_limits[1])
        width_y = max(top_y * INTERACTIVE_WIDTH_HANDLE_Y_FRACTION, 1e-9)
        center_y = max(top_y * INTERACTIVE_CENTER_HANDLE_Y_FRACTION, 1e-9)
        peak_index = int(np.argmax(y_values))
        peak_artist = axis.scatter(
            [float(x_values[peak_index])],
            [float(y_values[peak_index])],
            s=INTERACTIVE_PEAK_HANDLE_SIZE,
            marker="^",
            facecolor="tab:orange",
            edgecolor="black",
            zorder=6,
        )

        left_width_artist = None
        right_width_artist = None
        width_low, width_high = self._interactive_width_handle_positions(entry)
        if _x_coordinate_is_valid_for_scale(width_low, x_scale):
            left_width_artist = axis.scatter(
                [float(width_low)],
                [width_y],
                s=INTERACTIVE_WIDTH_HANDLE_SIZE,
                marker="s",
                facecolor="tab:blue",
                edgecolor="white",
                zorder=6,
            )
        if _x_coordinate_is_valid_for_scale(width_high, x_scale):
            right_width_artist = axis.scatter(
                [float(width_high)],
                [width_y],
                s=INTERACTIVE_WIDTH_HANDLE_SIZE,
                marker="s",
                facecolor="tab:blue",
                edgecolor="white",
                zorder=6,
            )

        center_artist = None
        if _x_coordinate_is_valid_for_scale(float(entry.value), x_scale):
            if self.lock_center_checkbox.isChecked():
                axis.scatter(
                    [float(entry.value)],
                    [center_y],
                    s=INTERACTIVE_CENTER_HANDLE_SIZE,
                    marker="o",
                    facecolor="white",
                    edgecolor="tab:red",
                    alpha=0.55,
                    zorder=6,
                )
            else:
                center_artist = axis.scatter(
                    [float(entry.value)],
                    [center_y],
                    s=INTERACTIVE_CENTER_HANDLE_SIZE,
                    marker="o",
                    facecolor="tab:red",
                    edgecolor="white",
                    zorder=6,
                )
        return _InteractiveHandleArtists(
            axis=axis,
            peak=peak_artist,
            center=center_artist,
            left_width=left_width_artist,
            right_width=right_width_artist,
        )

    @staticmethod
    def _clamped_event_y(
        y_value: float | None,
        *,
        y_limits: tuple[float, float],
    ) -> float:
        lower, upper = sorted(float(limit) for limit in y_limits)
        if y_value is None or not np.isfinite(y_value):
            return lower
        return min(max(float(y_value), lower), upper)

    @staticmethod
    def _clamped_event_x(
        x_value: float | None,
        *,
        x_limits: tuple[float, float],
        x_scale: str,
    ) -> float | None:
        if x_value is None or not np.isfinite(x_value):
            return None
        lower, upper = sorted(float(limit) for limit in x_limits)
        bounded_value = min(max(float(x_value), lower), upper)
        if x_scale == "log":
            bounded_value = max(bounded_value, np.finfo(float).tiny)
        return bounded_value

    @staticmethod
    def _entries_match(
        previous: DreamParameterEntry,
        current: DreamParameterEntry,
    ) -> bool:
        return (
            math.isclose(float(previous.value), float(current.value))
            and str(previous.distribution) == str(current.distribution)
            and previous.vary == current.vary
            and all(
                math.isclose(
                    float(previous.dist_params.get(key, float("nan"))),
                    float(current.dist_params.get(key, float("nan"))),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for key in set(previous.dist_params) | set(current.dist_params)
            )
        )

    def _interactive_drag_message(
        self,
        previous: DreamParameterEntry,
        current: DreamParameterEntry,
        *,
        handle_kind: str,
    ) -> str:
        del previous
        action = {
            "peak": "Adjusted prior peak height and width",
            "center": "Moved prior center",
            "left_width": "Adjusted prior width from the low-side handle",
            "right_width": "Adjusted prior width from the high-side handle",
        }.get(handle_kind, "Adjusted prior")
        return (
            f"{action}: {current.param} -> value={current.value:.6g}, "
            f"params={json.dumps(current.dist_params, sort_keys=True)}"
        )

    @staticmethod
    def _parse_params(text: str) -> dict[str, float]:
        payload = json.loads(text or "{}")
        return {str(key): float(value) for key, value in payload.items()}

    @staticmethod
    def _normalize_distribution_params(
        distribution: str,
        raw_params: dict[str, float],
        value: float,
    ) -> dict[str, float]:
        if distribution not in BASE_DISTRIBUTIONS:
            return {
                str(key): float(param_value)
                for key, param_value in raw_params.items()
            }
        params = _distribution_defaults_for_value(distribution, value)
        for key in list(params):
            if key in raw_params:
                params[key] = float(raw_params[key])
        return params

    @classmethod
    def _center_drag_adjusted_entry(
        cls,
        entry: DreamParameterEntry,
        *,
        target_center: float,
    ) -> DreamParameterEntry:
        updated_entry = DreamParameterEntry.from_dict(entry.to_dict())
        params = copy.deepcopy(dict(updated_entry.dist_params))
        center_value = float(target_center)
        updated_entry.value = center_value
        if updated_entry.distribution == "norm":
            params["loc"] = center_value
            params["scale"] = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
        elif updated_entry.distribution == "uniform":
            width = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["scale"] = width
            params["loc"] = center_value - width / 2.0
        elif updated_entry.distribution == "lognorm":
            scale_value = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["scale"] = scale_value
            params["s"] = max(
                float(params.get("s", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["loc"] = center_value - scale_value
        updated_entry.dist_params = params
        return updated_entry

    @classmethod
    def _width_drag_adjusted_entry(
        cls,
        entry: DreamParameterEntry,
        *,
        handle_kind: str,
        target_x: float,
    ) -> DreamParameterEntry:
        updated_entry = DreamParameterEntry.from_dict(entry.to_dict())
        params = copy.deepcopy(dict(updated_entry.dist_params))
        center_value = float(updated_entry.value)
        bounded_target = float(target_x)
        if updated_entry.distribution == "norm":
            sigma = max(
                abs(bounded_target - center_value) / GUIDE_INTERVAL_SIGMA,
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["loc"] = center_value
            params["scale"] = sigma
        elif updated_entry.distribution == "uniform":
            width = max(
                2.0 * abs(bounded_target - center_value),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["scale"] = width
            params["loc"] = center_value - width / 2.0
        elif updated_entry.distribution == "lognorm":
            scale_value = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            if handle_kind == "left_width":
                normalized = max(
                    1.0
                    + (min(bounded_target, center_value) - center_value)
                    / scale_value,
                    INTERACTIVE_PARAMETER_EPSILON,
                )
                shape_value = max(
                    -math.log(normalized) / GUIDE_INTERVAL_SIGMA,
                    INTERACTIVE_PARAMETER_EPSILON,
                )
            else:
                normalized = max(
                    1.0
                    + (max(bounded_target, center_value) - center_value)
                    / scale_value,
                    1.0 + INTERACTIVE_PARAMETER_EPSILON,
                )
                shape_value = max(
                    math.log(normalized) / GUIDE_INTERVAL_SIGMA,
                    INTERACTIVE_PARAMETER_EPSILON,
                )
            params["scale"] = scale_value
            params["loc"] = center_value - scale_value
            params["s"] = min(
                shape_value,
                INTERACTIVE_MAX_LOGNORM_SHAPE,
            )
        updated_entry.dist_params = params
        return updated_entry

    @classmethod
    def _peak_drag_adjusted_entry(
        cls,
        entry: DreamParameterEntry,
        *,
        start_y: float,
        target_y: float,
        y_limits: tuple[float, float],
    ) -> DreamParameterEntry:
        updated_entry = DreamParameterEntry.from_dict(entry.to_dict())
        params = copy.deepcopy(dict(updated_entry.dist_params))
        span = max(float(y_limits[1]) - float(y_limits[0]), 1e-9)
        delta_fraction = (float(target_y) - float(start_y)) / span
        width_factor = math.exp(
            -INTERACTIVE_PEAK_DRAG_SENSITIVITY * delta_fraction
        )
        center_value = float(updated_entry.value)
        if updated_entry.distribution == "norm":
            sigma = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON))
                * width_factor,
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["loc"] = center_value
            params["scale"] = sigma
        elif updated_entry.distribution == "uniform":
            width = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON))
                * width_factor,
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["scale"] = width
            params["loc"] = center_value - width / 2.0
        elif updated_entry.distribution == "lognorm":
            scale_value = max(
                float(params.get("scale", INTERACTIVE_PARAMETER_EPSILON)),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            params["scale"] = scale_value
            params["loc"] = center_value - scale_value
            params["s"] = min(
                max(
                    float(params.get("s", INTERACTIVE_PARAMETER_EPSILON))
                    * width_factor,
                    INTERACTIVE_PARAMETER_EPSILON,
                ),
                INTERACTIVE_MAX_LOGNORM_SHAPE,
            )
        updated_entry.dist_params = params
        return updated_entry

    @staticmethod
    def _interactive_width_handle_positions(
        entry: DreamParameterEntry,
    ) -> tuple[float | None, float | None]:
        center_value = float(entry.value)
        if entry.distribution == "norm":
            sigma = max(
                float(
                    entry.dist_params.get(
                        "scale", INTERACTIVE_PARAMETER_EPSILON
                    )
                ),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            spread = GUIDE_INTERVAL_SIGMA * sigma
            return center_value - spread, center_value + spread
        if entry.distribution == "uniform":
            width = max(
                float(
                    entry.dist_params.get(
                        "scale", INTERACTIVE_PARAMETER_EPSILON
                    )
                ),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            return center_value - width / 2.0, center_value + width / 2.0
        if entry.distribution == "lognorm":
            scale_value = max(
                float(
                    entry.dist_params.get(
                        "scale", INTERACTIVE_PARAMETER_EPSILON
                    )
                ),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            shape_value = max(
                float(
                    entry.dist_params.get("s", INTERACTIVE_PARAMETER_EPSILON)
                ),
                INTERACTIVE_PARAMETER_EPSILON,
            )
            return (
                center_value
                + scale_value
                * (math.exp(-GUIDE_INTERVAL_SIGMA * shape_value) - 1.0),
                center_value
                + scale_value
                * (math.exp(GUIDE_INTERVAL_SIGMA * shape_value) - 1.0),
            )
        return None, None

    def _smart_adjusted_entries(
        self,
        entries: list[DreamParameterEntry],
        *,
        preset_mode: str,
        apply_scope: str = "all",
        selected_rows: list[int] | None = None,
    ) -> list[DreamParameterEntry]:
        updated_entries = [
            DreamParameterEntry.from_dict(entry.to_dict()) for entry in entries
        ]
        row_groups = self._row_group_keys(updated_entries)
        target_groups = self._target_group_keys(
            updated_entries,
            apply_scope=apply_scope,
            selected_rows=list(selected_rows or []),
        )
        if preset_mode in SMART_PRIOR_SPREAD_FACTORS:
            factor = SMART_PRIOR_SPREAD_FACTORS[preset_mode]
            for row_index, entry in enumerate(updated_entries):
                if row_groups[row_index] not in target_groups:
                    continue
                entry.dist_params = self._adjust_distribution_params(
                    entry,
                    factor=factor,
                )
                entry.smart_preset_status = preset_mode
            return updated_entries

        weight_radius_map = self._weight_radius_map(updated_entries)
        if len(weight_radius_map) < 2:
            raise ValueError(
                "Size-aware smart priors require at least two effective-"
                "radius values tied to weight parameters."
            )
        size_groups = self._weight_size_groups(weight_radius_map)
        if preset_mode == "strict_small_lenient_large":
            factor_by_group = {
                "small": SMART_PRIOR_SPREAD_FACTORS["strict"],
                "large": SMART_PRIOR_SPREAD_FACTORS["lenient"],
                "neutral": SMART_PRIOR_SPREAD_FACTORS["proportional"],
            }
        elif preset_mode == "lenient_small_strict_large":
            factor_by_group = {
                "small": SMART_PRIOR_SPREAD_FACTORS["lenient"],
                "large": SMART_PRIOR_SPREAD_FACTORS["strict"],
                "neutral": SMART_PRIOR_SPREAD_FACTORS["proportional"],
            }
        else:
            raise ValueError(f"Unknown smart prior preset: {preset_mode}")

        weight_by_group = self._group_weight_params(updated_entries)
        if preset_mode == "strict_small_lenient_large":
            status_by_group = {
                "small": "strict",
                "large": "lenient",
                "neutral": "proportional",
            }
        else:
            status_by_group = {
                "small": "lenient",
                "large": "strict",
                "neutral": "proportional",
            }

        for row_index, entry in enumerate(updated_entries):
            group_key = row_groups[row_index]
            weight_param = weight_by_group.get(group_key)
            if weight_param is None:
                size_group = "neutral"
                factor = SMART_PRIOR_SPREAD_FACTORS["proportional"]
            else:
                size_group = size_groups.get(weight_param, "neutral")
                factor = factor_by_group.get(
                    size_group,
                    SMART_PRIOR_SPREAD_FACTORS["proportional"],
                )
            entry.dist_params = self._adjust_distribution_params(
                entry,
                factor=factor,
            )
            entry.smart_preset_status = status_by_group.get(
                size_group,
                "proportional",
            )
        return updated_entries

    def _on_smart_status_changed(
        self,
        row: int,
        combo: QComboBox,
    ) -> None:
        if self._suppress_status_change:
            return
        preset_mode = self._normalized_smart_status(combo.currentData())
        if preset_mode == "custom":
            self._set_group_status_for_row(row, "custom")
            self.console.append(
                "Marked smart prior status as Custom / Manual for "
                f"{self._row_status_label(row)}."
            )
            return
        entries = self.current_entries()
        try:
            updated_entries = self._smart_adjusted_entries(
                entries,
                preset_mode=preset_mode,
                apply_scope="selected",
                selected_rows=[row],
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to apply row smart prior preset",
                str(exc),
            )
            self.console.append(
                "Unable to apply row smart prior preset: " + str(exc)
            )
            self._set_group_status_for_row(row, self._row_smart_status(row))
            return
        self.load_entries(
            updated_entries,
            has_existing_parameter_map=self._has_existing_parameter_map,
            update_reset_entries=False,
        )
        self.console.append(
            "Applied row smart prior preset: "
            f"{SMART_PRIOR_STATUS_LABELS.get(preset_mode, preset_mode)} "
            f"for {self._row_status_label(row)}."
        )

    def _selected_row_indexes(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _row_group_keys(
        self,
        entries: list[DreamParameterEntry],
    ) -> list[tuple[str, str, str]]:
        return [
            self._entry_group_key(entry, row_index)
            for row_index, entry in enumerate(entries)
        ]

    def _target_group_keys(
        self,
        entries: list[DreamParameterEntry],
        *,
        apply_scope: str,
        selected_rows: list[int],
    ) -> set[tuple[str, str, str]]:
        row_groups = self._row_group_keys(entries)
        if apply_scope == "all":
            return set(row_groups)
        if apply_scope != "selected":
            raise ValueError(f"Unknown smart prior apply scope: {apply_scope}")
        if not selected_rows:
            raise ValueError(
                "Select one or more parameter rows before applying a smart "
                "prior preset to selected parameters."
            )
        return {
            row_groups[row]
            for row in selected_rows
            if 0 <= row < len(row_groups)
        }

    def _group_weight_params(
        self,
        entries: list[DreamParameterEntry],
    ) -> dict[tuple[str, str, str], str]:
        mapping: dict[tuple[str, str, str], str] = {}
        for row_index, entry in enumerate(entries):
            weight_name = self._weight_param_name_for_entry(entry)
            if weight_name is None:
                continue
            mapping[self._entry_group_key(entry, row_index)] = weight_name
        return mapping

    @staticmethod
    def _entry_group_key(
        entry: DreamParameterEntry,
        row_index: int,
    ) -> tuple[str, str, str]:
        structure = str(entry.structure).strip()
        motif = str(entry.motif).strip()
        if structure or motif:
            return ("structure", structure, motif)
        return ("row", str(row_index), "")

    @staticmethod
    def _normalized_smart_status(value: object) -> str:
        text = str(value or "custom").strip() or "custom"
        valid = {
            status for _label, status in SMART_PRIOR_INDIVIDUAL_STATUS_ITEMS
        }
        return text if text in valid else "custom"

    def _row_smart_status(self, row: int) -> str:
        combo = self.table.cellWidget(row, 8)
        if isinstance(combo, QComboBox):
            return self._normalized_smart_status(combo.currentData())
        return "custom"

    def _set_group_status_for_row(
        self,
        row: int,
        status: str,
    ) -> None:
        entries = self.current_entries()
        if row < 0 or row >= len(entries):
            return
        target_key = self._entry_group_key(entries[row], row)
        normalized_status = self._normalized_smart_status(status)
        self._suppress_status_change = True
        try:
            for row_index, entry in enumerate(entries):
                if self._entry_group_key(entry, row_index) != target_key:
                    continue
                combo = self.table.cellWidget(row_index, 8)
                if not isinstance(combo, QComboBox):
                    continue
                combo_index = combo.findData(normalized_status)
                if combo_index >= 0:
                    combo.setCurrentIndex(combo_index)
        finally:
            self._suppress_status_change = False

    def _row_status_label(self, row: int) -> str:
        structure_item = self.table.item(row, 0)
        motif_item = self.table.item(row, 1)
        param_item = self.table.item(row, 3)
        structure = (
            structure_item.text().strip() if structure_item is not None else ""
        )
        motif = motif_item.text().strip() if motif_item is not None else ""
        param = param_item.text().strip() if param_item is not None else ""
        if structure or motif:
            if motif:
                return f"{structure or 'No structure'} / {motif}"
            return structure
        return param or f"row {row + 1}"

    @staticmethod
    def _adjust_distribution_params(
        entry: DreamParameterEntry,
        *,
        factor: float,
    ) -> dict[str, float]:
        params = copy.deepcopy(dict(entry.dist_params))
        bounded_factor = max(float(factor), 1e-6)
        epsilon = 1e-9
        if entry.distribution == "lognorm":
            params["s"] = max(
                float(params.get("s", epsilon)) * bounded_factor, epsilon
            )
            return params
        if entry.distribution == "norm":
            params["scale"] = max(
                float(params.get("scale", epsilon)) * bounded_factor,
                epsilon,
            )
            return params
        if entry.distribution == "uniform":
            current_scale = max(float(params.get("scale", epsilon)), epsilon)
            center = float(params.get("loc", 0.0)) + current_scale / 2.0
            updated_scale = max(current_scale * bounded_factor, epsilon)
            params["scale"] = updated_scale
            params["loc"] = center - updated_scale / 2.0
            return params
        return params

    @classmethod
    def _weight_radius_map(
        cls,
        entries: list[DreamParameterEntry],
    ) -> dict[str, float]:
        sphere_map: dict[str, float] = {}
        ellipsoid_axes: dict[str, dict[str, float]] = {}
        for entry in entries:
            param_name = str(entry.param).strip()
            sphere_match = re.fullmatch(r"r_eff_(w\d+)", param_name)
            if sphere_match:
                sphere_map[sphere_match.group(1)] = max(
                    float(entry.value), 0.0
                )
                continue
            ellipsoid_match = re.fullmatch(r"([abc])_eff_(w\d+)", param_name)
            if ellipsoid_match:
                axis_name = ellipsoid_match.group(1)
                weight_name = ellipsoid_match.group(2)
                ellipsoid_axes.setdefault(weight_name, {})[axis_name] = max(
                    float(entry.value),
                    0.0,
                )
        for weight_name, axes in ellipsoid_axes.items():
            if {"a", "b", "c"} <= set(axes):
                sphere_map[weight_name] = float(
                    np.cbrt(axes["a"] * axes["b"] * axes["c"])
                )
        return sphere_map

    @staticmethod
    def _weight_size_groups(
        weight_radius_map: dict[str, float],
    ) -> dict[str, str]:
        radii = np.asarray(list(weight_radius_map.values()), dtype=float)
        median_radius = float(np.median(radii))
        tolerance = max(median_radius * 1e-9, 1e-9)
        groups: dict[str, str] = {}
        for weight_name, radius in weight_radius_map.items():
            if radius < median_radius - tolerance:
                groups[weight_name] = "small"
            elif radius > median_radius + tolerance:
                groups[weight_name] = "large"
            else:
                groups[weight_name] = "neutral"
        return groups

    @staticmethod
    def _weight_param_name_for_entry(
        entry: DreamParameterEntry,
    ) -> str | None:
        param_name = str(entry.param).strip()
        if re.fullmatch(r"w\d+", param_name):
            return param_name
        return None

    @staticmethod
    def _is_effective_radius_parameter(param_name: str) -> bool:
        name = str(param_name).strip()
        return bool(
            name == "eff_r"
            or name.startswith("r_eff_")
            or name.startswith("a_eff_")
            or name.startswith("b_eff_")
            or name.startswith("c_eff_")
        )

    def _refresh_distribution_guides_for_row(
        self,
        row: int,
        *,
        entry: DreamParameterEntry | None,
    ) -> None:
        guide_low_text = "n/a"
        guide_high_text = "n/a"
        guide_tooltip = (
            "Practical prior bound is unavailable until the distribution "
            "parameters are valid."
        )
        if entry is not None:
            guide_low, guide_high, guide_kind = _distribution_guide_bounds(
                entry
            )
            if guide_low is not None and guide_high is not None:
                guide_low_text = _format_distribution_guide_value(guide_low)
                guide_high_text = _format_distribution_guide_value(guide_high)
                guide_tooltip = (
                    f"{guide_kind} for the current {entry.distribution} prior."
                )
        self._set_distribution_guide_item(
            row,
            GUIDE_LOW_COLUMN,
            guide_low_text,
            tooltip=guide_tooltip,
        )
        self._set_distribution_guide_item(
            row,
            GUIDE_HIGH_COLUMN,
            guide_high_text,
            tooltip=guide_tooltip,
        )

    def _set_distribution_guide_item(
        self,
        row: int,
        column: int,
        text: str,
        *,
        tooltip: str,
    ) -> None:
        was_blocked = self.table.blockSignals(True)
        try:
            item = self.table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row, column, item)
            item.setText(text)
            item.setToolTip(tooltip)
        finally:
            self.table.blockSignals(was_blocked)

    def _on_distribution_guide_changed(self, row: int, column: int) -> None:
        guide_item = self.table.item(row, column)
        guide_text = "" if guide_item is None else guide_item.text().strip()
        guide_label = "low" if column == GUIDE_LOW_COLUMN else "high"
        handle_kind = (
            "left_width" if column == GUIDE_LOW_COLUMN else "right_width"
        )
        try:
            target_value = float(guide_text)
            entry = self._entry_from_row(row)
            updated_entry = self._width_drag_adjusted_entry(
                entry,
                handle_kind=handle_kind,
                target_x=target_value,
            )
        except Exception as exc:
            self._refresh_distribution_guides_for_row(
                row,
                entry=(
                    self._entry_from_row(row)
                    if 0 <= row < self.table.rowCount()
                    else None
                ),
            )
            self.console.append(
                "Invalid guide "
                f"{guide_label} value for {self._row_status_label(row)}: {exc}"
            )
            return
        self._apply_entry_to_row(row, updated_entry)
        self._set_group_status_for_row(row, "custom")
        self.console.append(
            "Updated guide "
            f"{guide_label} for {updated_entry.param} -> "
            f"{target_value:.6g}."
        )


def _x_coordinate_is_valid_for_scale(
    x_value: float | None,
    x_scale: str,
) -> bool:
    if x_value is None or not np.isfinite(x_value):
        return False
    if x_scale == "log" and float(x_value) <= 0.0:
        return False
    return True


def _distribution_domain(entry: DreamParameterEntry) -> tuple[float, float]:
    return _distribution_domain_quantiles(
        entry,
        lower_q=0.001,
        upper_q=0.999,
    )


def _distribution_domain_quantiles(
    entry: DreamParameterEntry,
    *,
    lower_q: float,
    upper_q: float,
) -> tuple[float, float]:
    distribution = getattr(stats, entry.distribution)
    x_min = distribution.ppf(lower_q, **entry.dist_params)
    x_max = distribution.ppf(upper_q, **entry.dist_params)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return entry.value - 1.0, entry.value + 1.0
    return float(x_min), float(x_max)


def _distribution_plot_bounds(
    entry: DreamParameterEntry,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    distribution = getattr(stats, entry.distribution)
    sample_low, sample_high = _distribution_domain_quantiles(
        entry,
        lower_q=PLOT_DOMAIN_LOWER_Q,
        upper_q=PLOT_DOMAIN_UPPER_Q,
    )
    if not np.isfinite(sample_low) or not np.isfinite(sample_high):
        raise ValueError("Plot domain is not finite.")
    if sample_low == sample_high:
        span = max(abs(float(entry.value)) * 0.5, 1.0)
        sample_low = float(entry.value) - span
        sample_high = float(entry.value) + span
    if sample_high < sample_low:
        sample_low, sample_high = sample_high, sample_low

    use_log_scale = (
        entry.distribution == "lognorm"
        and sample_low > 0.0
        and sample_high / sample_low >= PLOT_LOG_SCALE_SPAN_RATIO
    )
    sample_x = _distribution_sample_grid(
        sample_low,
        sample_high,
        count=PLOT_SAMPLE_COUNT,
        x_scale="log" if use_log_scale else "linear",
    )
    sample_y = distribution.pdf(sample_x, **entry.dist_params)
    finite_mask = np.isfinite(sample_x) & np.isfinite(sample_y)
    if not np.any(finite_mask):
        raise ValueError("Distribution density is not finite.")
    sample_x = sample_x[finite_mask]
    sample_y = sample_y[finite_mask]
    peak_density = float(np.max(sample_y))
    if not np.isfinite(peak_density) or peak_density <= 0.0:
        raise ValueError("Distribution density could not be evaluated.")

    focus_mask = sample_y >= peak_density * PLOT_RELATIVE_DENSITY_THRESHOLD
    if not np.any(focus_mask):
        focus_mask = np.ones_like(sample_y, dtype=bool)
    focus_low = float(sample_x[focus_mask][0])
    focus_high = float(sample_x[focus_mask][-1])
    if np.isfinite(entry.value):
        focus_low = min(focus_low, float(entry.value))
        focus_high = max(focus_high, float(entry.value))

    focus_low, focus_high = _expand_plot_limits(
        focus_low,
        focus_high,
        x_scale="log" if use_log_scale else "linear",
        padding_fraction=PLOT_PADDING_FRACTION,
    )
    focus_x = _distribution_sample_grid(
        focus_low,
        focus_high,
        count=PLOT_SAMPLE_COUNT,
        x_scale="log" if use_log_scale else "linear",
    )
    focus_y = distribution.pdf(focus_x, **entry.dist_params)
    finite_focus_mask = np.isfinite(focus_x) & np.isfinite(focus_y)
    if not np.any(finite_focus_mask):
        raise ValueError("Focused plot density is not finite.")
    focus_x = focus_x[finite_focus_mask]
    focus_y = focus_y[finite_focus_mask]
    y_peak = float(np.max(focus_y))
    if not np.isfinite(y_peak) or y_peak <= 0.0:
        raise ValueError("Focused plot density could not be evaluated.")
    y_padding = max(y_peak * PLOT_PADDING_FRACTION, 1e-12)
    return (
        (float(focus_x[0]), float(focus_x[-1])),
        (0.0, max(y_peak + y_padding, 1e-12)),
        "log" if use_log_scale else "linear",
    )


def _distribution_plot_window(
    entry: DreamParameterEntry,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    required_x_limits, required_y_limits, x_scale = _distribution_plot_bounds(
        entry
    )
    window_low, window_high = _expand_plot_limits(
        required_x_limits[0],
        required_x_limits[1],
        x_scale=x_scale,
        padding_fraction=PLOT_WINDOW_MARGIN_FRACTION,
    )
    return (
        (window_low, window_high),
        (
            required_y_limits[0],
            max(required_y_limits[1], 1e-12)
            * (1.0 + PLOT_WINDOW_MARGIN_FRACTION),
        ),
        x_scale,
    )


def _distribution_plot_curve(
    entry: DreamParameterEntry,
    *,
    x_limits: tuple[float, float],
    x_scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    distribution = getattr(stats, entry.distribution)
    plot_x = _distribution_sample_grid(
        x_limits[0],
        x_limits[1],
        count=PLOT_SAMPLE_COUNT,
        x_scale=x_scale,
    )
    plot_y = distribution.pdf(plot_x, **entry.dist_params)
    finite_plot_mask = np.isfinite(plot_x) & np.isfinite(plot_y)
    if not np.any(finite_plot_mask):
        raise ValueError("Focused plot density is not finite.")
    return plot_x[finite_plot_mask], plot_y[finite_plot_mask]


def _distribution_sample_grid(
    lower: float,
    upper: float,
    *,
    count: int,
    x_scale: str,
) -> np.ndarray:
    if x_scale == "log":
        bounded_low = max(float(lower), np.finfo(float).tiny)
        bounded_high = max(float(upper), bounded_low * (1.0 + 1e-9))
        return np.geomspace(bounded_low, bounded_high, count)
    bounded_low = float(lower)
    bounded_high = float(upper)
    if bounded_high == bounded_low:
        bounded_high = bounded_low + 1.0
    return np.linspace(bounded_low, bounded_high, count)


def _expand_plot_limits(
    lower: float,
    upper: float,
    *,
    x_scale: str,
    padding_fraction: float = PLOT_PADDING_FRACTION,
) -> tuple[float, float]:
    if x_scale == "log":
        bounded_low = max(float(lower), np.finfo(float).tiny)
        bounded_high = max(float(upper), bounded_low * (1.0 + 1e-9))
        ratio = bounded_high / bounded_low
        padding_ratio = max(ratio**padding_fraction, 1.05)
        return bounded_low / padding_ratio, bounded_high * padding_ratio
    bounded_low = float(lower)
    bounded_high = float(upper)
    span = bounded_high - bounded_low
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(bounded_low), abs(bounded_high), 1.0)
    padding = max(span * padding_fraction, 1e-9)
    return bounded_low - padding, bounded_high + padding


def _distribution_guide_bounds(
    entry: DreamParameterEntry,
) -> tuple[float | None, float | None, str]:
    distribution = getattr(stats, entry.distribution)
    try:
        support_low, support_high = distribution.support(**entry.dist_params)
        support_low = float(support_low)
        support_high = float(support_high)
    except Exception:
        support_low = float("nan")
        support_high = float("nan")

    if np.isfinite(support_low) and np.isfinite(support_high):
        if support_low <= support_high:
            return support_low, support_high, "Exact support"

    try:
        guide_low = float(
            distribution.ppf(GUIDE_INTERVAL_LOWER_Q, **entry.dist_params)
        )
        guide_high = float(
            distribution.ppf(GUIDE_INTERVAL_UPPER_Q, **entry.dist_params)
        )
        if np.isfinite(support_low):
            guide_low = (
                max(guide_low, support_low)
                if np.isfinite(guide_low)
                else support_low
            )
        if np.isfinite(support_high):
            guide_high = (
                min(guide_high, support_high)
                if np.isfinite(guide_high)
                else support_high
            )
        if np.isfinite(guide_low) and np.isfinite(guide_high):
            if guide_low <= guide_high:
                return (
                    guide_low,
                    guide_high,
                    "Central 99.73% interval (3sigma equivalent)",
                )
    except Exception:
        pass

    try:
        domain_low, domain_high = _distribution_domain(entry)
    except Exception:
        return None, None, "Unavailable"
    if np.isfinite(domain_low) and np.isfinite(domain_high):
        if domain_low <= domain_high:
            return domain_low, domain_high, "Preview domain fallback"
    return None, None, "Unavailable"


def _format_distribution_guide_value(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.6g}"


def _distribution_defaults_for_value(
    distribution: str,
    value: float,
) -> dict[str, float]:
    eps = 1e-6
    cv_default = 1 / math.e
    if distribution == "lognorm":
        scale_val = value if value > 0 else eps
        return {
            "loc": 0.0,
            "scale": scale_val,
            "s": math.sqrt(math.log(1 + cv_default**2)),
        }
    if distribution == "norm":
        return {
            "loc": value,
            "scale": value / math.e if value > 0 else eps,
        }
    if distribution == "uniform":
        if value > 0:
            half_range = value * cv_default
            return {
                "loc": max(value - half_range, 0.0),
                "scale": 2 * half_range,
            }
        return {"loc": 0.0, "scale": eps}
    return dict(BASE_DISTRIBUTIONS[distribution])
