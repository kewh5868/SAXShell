from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QScrollBar,
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
    resolve_prefit_parameter_entries,
)
from saxshell.saxs.prefit.cluster_geometry import (
    DEFAULT_IONIC_RADIUS_TYPE,
    DEFAULT_RADIUS_TYPE,
    IONIC_RADIUS_TYPE_OPTIONS,
    RADIUS_TYPE_OPTIONS,
    STRUCTURE_FACTOR_RECOMMENDATIONS,
    synchronize_cluster_geometry_row,
)
from saxshell.saxs.ui.solute_volume_fraction_widget import (
    SOLUTE_VOLUME_FRACTION_HELP_TEXT as SOLUTE_VOLUME_FRACTION_HELP_MESSAGE,
)
from saxshell.saxs.ui.solute_volume_fraction_widget import (
    SoluteVolumeFractionWidget,
)
from saxshell.saxs.ui.template_help import (
    TEMPLATE_HELP_TEXT,
    show_template_help,
)


@dataclass(slots=True)
class PrefitRunConfig:
    method: str
    max_nfev: int


class TableCellComboBox(QComboBox):
    """QComboBox with popup alignment that stays anchored to table
    cells."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Use the Qt popup instead of platform-native popup placement so
        # table-cell combos remain anchored to the cell inside scroll areas.
        self.setStyleSheet("QComboBox { combobox-popup: 0; }")

    def showPopup(self) -> None:
        super().showPopup()
        popup = self.view().window()
        if popup is None:
            return

        popup_size = popup.sizeHint()
        popup_width = max(popup.width(), popup_size.width(), self.width())
        popup_height = max(popup.height(), popup_size.height())
        screen = self.screen()
        available = (
            screen.availableGeometry()
            if screen is not None
            else self.window().geometry()
        )

        below_left = self.mapToGlobal(QPoint(0, self.height()))
        above_left = self.mapToGlobal(QPoint(0, 0))
        x_pos = below_left.x()
        y_pos = below_left.y()

        max_x = available.x() + available.width() - popup_width
        if x_pos > max_x:
            x_pos = max(available.x(), max_x)

        if y_pos + popup_height > available.y() + available.height():
            above_y = above_left.y() - popup_height
            if above_y >= available.y():
                y_pos = above_y

        popup.resize(popup_width, popup_height)
        popup.move(x_pos, y_pos)


class PrefitTab(QWidget):
    PARAMETER_VALUE_ROLE = int(Qt.ItemDataRole.UserRole)
    PARAMETER_VARY_MEMORY_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    template_changed = Signal(str)
    show_deprecated_templates_changed = Signal(bool)
    autosave_toggled = Signal(bool)
    sequence_history_toggled = Signal(bool)
    field_interaction_requested = Signal()
    parameter_table_edited = Signal()
    parameter_reset_requested = Signal(str, str, str)
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
    update_cluster_geometry_requested = Signal()
    cluster_geometry_mapping_changed = Signal()
    cluster_geometry_sf_approximation_changed = Signal(str, str)
    cluster_geometry_radii_type_changed = Signal(str)
    cluster_geometry_ionic_radius_type_changed = Signal(str)

    PREFIT_HELP_TEXT = (
        "Recommended prefit workflow:\n"
        "- Refine scale before the other model parameters.\n"
        "- After scale is stable, refine scale and offset together.\n"
        "- Component weights w<##> are not recommended for prefit "
        "refinement or manual adjustment."
    )
    IONIC_RADIUS_HELP_TEXT = (
        "Ionic radius estimate source\n\n"
        "The ionic-radius modes in Cluster Geometry Metadata currently use "
        "SAXSShell's internal approximate element-level lookup table in "
        "`src/saxshell/saxs/prefit/cluster_geometry.py`.\n\n"
        "Literature anchor for the effective ionic-radius picture:\n"
        "Shannon, R. D. (1976). Revised effective ionic radii and "
        "systematic studies of interatomic distances in halides and "
        "chalcogenides. Acta Crystallographica Section A, 32(5), 751-767.\n"
        "https://doi.org/10.1107/S0567739476001551\n\n"
        "Current SAXSShell implementation notes:\n"
        "- Effective ionic: internal approximate element-level values meant "
        "to stay broadly consistent with Shannon-style effective ionic "
        "radii, but not a charge-state or coordination-specific lookup.\n"
        "- Crystal ionic: currently derived in SAXSShell by adding a fixed "
        "+0.14 A offset to the effective ionic table. This is an "
        "application-level approximation, not a direct crystal-radius "
        "database lookup.\n"
        "- Missing elements currently fall back to 1.5 A."
    )
    SOLUTE_VOLUME_FRACTION_HELP_TEXT = SOLUTE_VOLUME_FRACTION_HELP_MESSAGE
    CLUSTER_GEOMETRY_HEADERS = [
        "Cluster",
        "Path",
        "Avg Size",
        "Effective Radius",
        "Radii Type",
        "Semiaxis X",
        "Semiaxis Y",
        "Semiaxis Z",
        "S.F. Approx.",
        "Anisotropy",
        "Map To",
        "Notes",
    ]
    ACTIVE_CLUSTER_GEOMETRY_COLOR = QColor("#0057d9")
    INACTIVE_CLUSTER_GEOMETRY_COLOR = QColor("#404040")
    CLUSTER_COL_CLUSTER = 0
    CLUSTER_COL_PATH = 1
    CLUSTER_COL_AVG_SIZE = 2
    CLUSTER_COL_EFFECTIVE_RADIUS = 3
    CLUSTER_COL_RADII_TYPE = 4
    CLUSTER_COL_SEMIAXIS_X = 5
    CLUSTER_COL_SEMIAXIS_Y = 6
    CLUSTER_COL_SEMIAXIS_Z = 7
    CLUSTER_COL_SF_APPROX = 8
    CLUSTER_COL_ANISOTROPY = 9
    CLUSTER_COL_MAP_TO = 10
    CLUSTER_COL_NOTES = 11
    PARAMETER_SCROLL_RESOLUTION = 2000
    PARAMETER_SCROLL_LOG_DECADE_THRESHOLD = 2.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._console_autoscroll_enabled = True
        self._current_evaluation: PrefitEvaluation | None = None
        self._summary_text = ""
        self._base_log_text = ""
        self._history_messages: list[str] = []
        self._legend_line_map: dict[object, object] = {}
        self._legend_handle_lookup: dict[str, object] = {}
        self._cluster_geometry_rows: list[ClusterGeometryMetadataRow] = []
        self._cluster_geometry_mapping_options: list[tuple[str, str]] = []
        self._cluster_geometry_allowed_sf_approximations: tuple[str, ...] = (
            STRUCTURE_FACTOR_RECOMMENDATIONS
        )
        self._expanded_cluster_geometry_path_rows: set[int] = set()
        self._expanded_cluster_geometry_note_rows: set[int] = set()
        self._model_only_mode = False
        self._prefit_execution_enabled = True
        self._updating_parameter_table = False
        self._updating_parameter_scrollbar = False
        self._last_cluster_geometry_radii_type = DEFAULT_RADIUS_TYPE
        self._last_cluster_geometry_ionic_radius_type = (
            DEFAULT_IONIC_RADIUS_TYPE
        )
        self._build_ui()
        self._install_field_interaction_watchers()

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
        self._solute_volume_fraction_group = (
            self._build_solute_volume_fraction_group()
        )
        left_layout.addWidget(self._solute_volume_fraction_group)
        self._cluster_geometry_group = self._build_cluster_geometry_group()
        left_layout.addWidget(self._cluster_geometry_group)
        left_layout.addWidget(self._build_parameter_group())
        left_layout.addStretch(1)
        self._left_scroll_area = QScrollArea()
        self._left_scroll_area.setWidgetResizable(True)
        self._left_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._left_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._left_scroll_area.setWidget(self._left_panel)
        self._plot_group = self._build_plot_group()
        self._pane_splitter.addWidget(self._left_scroll_area)
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

    def _install_field_interaction_watchers(self) -> None:
        watched_widgets: list[QWidget] = []
        for child in self.findChildren(QWidget):
            if isinstance(
                child,
                (
                    QAbstractSpinBox,
                    QCheckBox,
                    QComboBox,
                    QLineEdit,
                ),
            ):
                watched_widgets.append(child)
        watched_widgets.extend(
            [
                self.parameter_table,
                self.parameter_table.viewport(),
                self.cluster_geometry_table,
                self.cluster_geometry_table.viewport(),
            ]
        )
        seen: set[int] = set()
        for widget in watched_widgets:
            widget_id = id(widget)
            if widget_id in seen:
                continue
            seen.add(widget_id)
            widget.installEventFilter(self)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if isinstance(watched, QWidget) and watched.isEnabled():
            if event.type() in (
                QEvent.Type.KeyPress,
                QEvent.Type.MouseButtonDblClick,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.Wheel,
            ):
                self.field_interaction_requested.emit()
        return super().eventFilter(watched, event)

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
        self.show_deprecated_templates_checkbox = QCheckBox("Show deprecated")
        self.show_deprecated_templates_checkbox.setChecked(False)
        self.show_deprecated_templates_checkbox.setToolTip(
            "Include deprecated and archived SAXS templates in the "
            "template dropdown."
        )
        self.show_deprecated_templates_checkbox.toggled.connect(
            self.show_deprecated_templates_changed.emit
        )
        layout.addWidget(QLabel("Template"), 0, 0)
        layout.addWidget(self.template_combo, 0, 1)
        layout.addWidget(self.template_help_button, 0, 2)
        layout.addWidget(self.show_deprecated_templates_checkbox, 0, 3)

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

        self.sequence_history_checkbox = QCheckBox("Sequence history logger")
        self.sequence_history_checkbox.setChecked(False)
        self.sequence_history_checkbox.setToolTip(
            "Record Prefit actions, manual parameter changes, autoscale "
            "updates, and fit runs into prefit_sequence_history.json in the "
            "active distribution Prefit folder."
        )
        self.sequence_history_checkbox.toggled.connect(
            self.sequence_history_toggled.emit
        )
        layout.addWidget(self.sequence_history_checkbox, 3, 0, 1, 3)

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
        layout.addWidget(QLabel("Saved states"), 4, 0)
        layout.addWidget(restore_row, 4, 1, 1, 2)

        self.stoichiometry_status_label = QLabel(
            "Stoichiometry monitor: configure target elements and ratio in "
            "DREAM > Posterior Filtering."
        )
        self.stoichiometry_status_label.setWordWrap(True)
        layout.addWidget(self.stoichiometry_status_label, 5, 0, 1, 3)

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
        self.recommended_scale_button = QPushButton("Autoscale")
        self.recommended_scale_button.setToolTip(
            "Estimate the scale from the current model and experimental "
            "intensities, then update the scale value and refinement bounds."
        )
        self.recommended_scale_button.clicked.connect(
            self.apply_recommended_scale_requested.emit
        )
        self.set_best_button = QPushButton("Set Best Prefit Params")
        self.set_best_button.setToolTip(
            "Save the current prefit parameter table into the project file as "
            "the Best Prefit preset for future reloads and quick restores, "
            "and update the DREAM parameter-map centers to match it."
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
        self._run_button_cell = run_cell
        run_cell_layout = QVBoxLayout(run_cell)
        run_cell_layout.setContentsMargins(0, 0, 0, 0)
        run_cell_layout.setSpacing(4)
        run_button_row = QHBoxLayout()
        run_button_row.setContentsMargins(0, 0, 0, 0)
        run_button_row.addWidget(self.run_button)
        run_button_row.addWidget(self.prefit_help_button)
        run_button_row.addStretch(1)
        run_cell_layout.addLayout(run_button_row)
        self._prefit_control_button_grid = button_grid
        button_grid.addWidget(run_cell, 0, 0)
        button_grid.addWidget(self.autosave_checkbox, 0, 1)
        button_grid.addWidget(self.save_button, 1, 0)
        button_grid.addWidget(self.reset_button, 1, 1)
        button_grid.addWidget(self.set_best_button, 2, 0)
        button_grid.addWidget(self.reset_best_button, 2, 1)
        layout.addLayout(button_grid, 6, 0, 1, 3)
        return group

    def _build_solute_volume_fraction_group(self) -> QGroupBox:
        group = QGroupBox("Solution Scattering Estimators")
        layout = QVBoxLayout(group)

        header_row = QHBoxLayout()
        self.solute_volume_fraction_collapse_button = QToolButton()
        self.solute_volume_fraction_collapse_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.solute_volume_fraction_collapse_button.setAutoRaise(True)
        self.solute_volume_fraction_collapse_button.clicked.connect(
            self._toggle_solute_volume_fraction_collapsed
        )
        header_row.addWidget(self.solute_volume_fraction_collapse_button)
        self.solute_volume_fraction_status_label = QLabel(
            "These estimators can calculate solution volume fractions, "
            "solvent attenuation scaling, and fluorescence background "
            "proxies."
        )
        self.solute_volume_fraction_status_label.setWordWrap(True)
        header_row.addWidget(
            self.solute_volume_fraction_status_label, stretch=1
        )
        self.solute_volume_fraction_help_button = QToolButton()
        self.solute_volume_fraction_help_button.setText("?")
        self.solute_volume_fraction_help_button.setToolTip(
            "How the solution-scattering estimators are defined. Click for "
            "the volume-fraction, attenuation, and fluorescence summary."
        )
        self.solute_volume_fraction_help_button.clicked.connect(
            self._show_solute_volume_fraction_help
        )
        header_row.addWidget(self.solute_volume_fraction_help_button)
        layout.addLayout(header_row)

        self.solute_volume_fraction_widget = SoluteVolumeFractionWidget(self)
        layout.addWidget(self.solute_volume_fraction_widget)
        self.set_solute_volume_fraction_collapsed(True)
        group.setVisible(False)
        return group

    def _build_plot_group(self) -> QGroupBox:
        group = QGroupBox("Model vs Experimental")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.show_experimental_trace_checkbox = QCheckBox("Experimental")
        self.show_experimental_trace_checkbox.setChecked(True)
        self.show_experimental_trace_checkbox.toggled.connect(
            self._redraw_current_plot
        )
        self.show_model_trace_checkbox = QCheckBox("Model")
        self.show_model_trace_checkbox.setChecked(True)
        self.show_model_trace_checkbox.toggled.connect(
            self._redraw_current_plot
        )
        self.show_solvent_trace_checkbox = QCheckBox("Solvent")
        self.show_solvent_trace_checkbox.setChecked(False)
        self.show_solvent_trace_checkbox.toggled.connect(
            self._redraw_current_plot
        )
        self.show_structure_factor_trace_checkbox = QCheckBox(
            "Structure factor"
        )
        self.show_structure_factor_trace_checkbox.setChecked(False)
        self.show_structure_factor_trace_checkbox.toggled.connect(
            self._redraw_current_plot
        )
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
        controls.addWidget(self.show_experimental_trace_checkbox)
        controls.addWidget(self.show_model_trace_checkbox)
        controls.addWidget(self.show_solvent_trace_checkbox)
        controls.addWidget(self.show_structure_factor_trace_checkbox)
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
        action_row = QHBoxLayout()
        self._parameter_action_layout = action_row
        action_row.addWidget(self.recommended_scale_button)
        action_row.addWidget(self.update_button)
        self.auto_update_checkbox = QCheckBox(
            "Auto-update on parameter change"
        )
        self.auto_update_checkbox.setChecked(False)
        self.auto_update_checkbox.setToolTip(
            "Automatically refresh the SAXS model preview when a parameter "
            "value in the table changes."
        )
        self.auto_update_checkbox.toggled.connect(
            self._on_auto_update_checkbox_toggled
        )
        action_row.addWidget(self.auto_update_checkbox)
        self.scrollable_parameter_checkbox = QCheckBox("Scrollable parameter")
        self.scrollable_parameter_checkbox.setChecked(False)
        self.scrollable_parameter_checkbox.setEnabled(False)
        self.scrollable_parameter_checkbox.setToolTip(
            "Show a scrollbar for the selected parameter and update the "
            "model as you scrub through the allowed range."
        )
        self.scrollable_parameter_checkbox.toggled.connect(
            self._on_scrollable_parameter_toggled
        )
        action_row.addWidget(self.scrollable_parameter_checkbox)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        self.parameter_scroll_panel = QWidget()
        self.parameter_scroll_panel.setVisible(False)
        scroll_panel_layout = QVBoxLayout(self.parameter_scroll_panel)
        scroll_panel_layout.setContentsMargins(0, 0, 0, 0)
        scroll_panel_layout.setSpacing(4)
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        self.parameter_scroll_name_label = QLabel(
            "Select a parameter row to scrub its value."
        )
        self.parameter_scroll_name_label.setWordWrap(True)
        info_row.addWidget(self.parameter_scroll_name_label, stretch=1)
        self.parameter_scroll_mode_label = QLabel("")
        info_row.addWidget(self.parameter_scroll_mode_label)
        self.parameter_scroll_value_label = QLabel("")
        info_row.addWidget(self.parameter_scroll_value_label)
        scroll_panel_layout.addLayout(info_row)
        self.parameter_scroll_bar = QScrollBar(Qt.Orientation.Horizontal)
        self.parameter_scroll_bar.setRange(
            0,
            self.PARAMETER_SCROLL_RESOLUTION,
        )
        self.parameter_scroll_bar.setSingleStep(1)
        self.parameter_scroll_bar.setPageStep(
            max(self.PARAMETER_SCROLL_RESOLUTION // 20, 1)
        )
        self.parameter_scroll_bar.valueChanged.connect(
            self._on_parameter_scrollbar_value_changed
        )
        scroll_panel_layout.addWidget(self.parameter_scroll_bar)
        layout.addWidget(self.parameter_scroll_panel)
        self.parameter_table = QTableWidget(0, 8)
        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Structure",
                "Motif",
                "Param",
                "Value",
                "Vary",
                "Min",
                "Max",
                "Reset",
            ]
        )
        self.parameter_table.itemChanged.connect(
            self._on_parameter_table_item_changed
        )
        self.parameter_table.currentCellChanged.connect(
            self._on_parameter_table_current_cell_changed
        )
        header = self.parameter_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parameter_table.setMinimumWidth(520)
        layout.addWidget(self.parameter_table)
        return group

    def _build_cluster_geometry_group(self) -> QGroupBox:
        group = QGroupBox("Cluster Geometry Metadata")
        layout = QVBoxLayout(group)

        self.cluster_geometry_status_label = QLabel(
            "This template does not use per-cluster geometry metadata."
        )
        self.cluster_geometry_status_label.setWordWrap(True)
        layout.addWidget(self.cluster_geometry_status_label)

        controls_row = QHBoxLayout()
        self.cluster_geometry_radii_type_combo = QComboBox()
        for radii_type in RADIUS_TYPE_OPTIONS:
            self.cluster_geometry_radii_type_combo.addItem(
                self._radii_type_label(radii_type),
                userData=radii_type,
            )
        self.cluster_geometry_radii_type_combo.setToolTip(
            "Choose which precomputed radii model is active in the cluster "
            "geometry table and runtime metadata."
        )
        self.cluster_geometry_radii_type_combo.currentIndexChanged.connect(
            self._on_cluster_geometry_radii_type_changed
        )
        self.cluster_geometry_ionic_radius_type_combo = QComboBox()
        for ionic_radius_type in IONIC_RADIUS_TYPE_OPTIONS:
            self.cluster_geometry_ionic_radius_type_combo.addItem(
                self._ionic_radius_type_label(ionic_radius_type),
                userData=ionic_radius_type,
            )
        self.cluster_geometry_ionic_radius_type_combo.setToolTip(
            "Choose which ionic-radius convention is active whenever the "
            "cluster geometry table is using ionic radii."
        )
        self.cluster_geometry_ionic_radius_type_combo.currentIndexChanged.connect(
            self._on_cluster_geometry_ionic_radius_type_changed
        )
        self.cluster_geometry_ionic_radius_help_button = QToolButton()
        self.cluster_geometry_ionic_radius_help_button.setText("?")
        self.cluster_geometry_ionic_radius_help_button.setToolTip(
            "How the ionic-radius estimate is sourced. Click for citations "
            "and implementation details."
        )
        self.cluster_geometry_ionic_radius_help_button.clicked.connect(
            self._show_ionic_radius_help
        )
        self.toggle_cluster_geometry_radii_button = QPushButton(
            "Toggle Active Radii"
        )
        self.toggle_cluster_geometry_radii_button.setToolTip(
            "Switch the active cluster geometry view between ionic-radii "
            "and bond-length calculations."
        )
        self.toggle_cluster_geometry_radii_button.clicked.connect(
            self._toggle_cluster_geometry_radii_type
        )
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
        self.update_cluster_geometry_button = QPushButton(
            "Update Cluster Geometry"
        )
        self.update_cluster_geometry_button.setToolTip(
            "Apply manual radii edits from the cluster geometry table and "
            "refresh the Prefit model preview."
        )
        self.update_cluster_geometry_button.clicked.connect(
            self.update_cluster_geometry_requested.emit
        )
        controls_row.addWidget(QLabel("Radii"))
        controls_row.addWidget(self.cluster_geometry_radii_type_combo)
        controls_row.addWidget(QLabel("Ionic Type"))
        controls_row.addWidget(self.cluster_geometry_ionic_radius_help_button)
        controls_row.addWidget(self.cluster_geometry_ionic_radius_type_combo)
        controls_row.addWidget(self.toggle_cluster_geometry_radii_button)
        controls_row.addWidget(self.compute_cluster_geometry_button)
        controls_row.addWidget(self.update_cluster_geometry_button)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.cluster_geometry_progress_label = QLabel("Progress: idle")
        layout.addWidget(self.cluster_geometry_progress_label)
        self.cluster_geometry_progress_bar = QProgressBar()
        self.cluster_geometry_progress_bar.setRange(0, 1)
        self.cluster_geometry_progress_bar.setValue(0)
        self.cluster_geometry_progress_bar.setFormat("%v / %m files")
        layout.addWidget(self.cluster_geometry_progress_bar)

        self.cluster_geometry_table = QTableWidget(
            0,
            len(self.CLUSTER_GEOMETRY_HEADERS),
        )
        self.cluster_geometry_table.setHorizontalHeaderLabels(
            self.CLUSTER_GEOMETRY_HEADERS
        )
        header = self.cluster_geometry_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cluster_geometry_table.setMinimumHeight(220)
        self.cluster_geometry_table.cellDoubleClicked.connect(
            self._on_cluster_geometry_cell_double_clicked
        )
        layout.addWidget(self.cluster_geometry_table)
        group.setVisible(False)
        self._update_cluster_geometry_ionic_radius_type_enabled_state()
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

    def show_deprecated_templates(self) -> bool:
        return self.show_deprecated_templates_checkbox.isChecked()

    def set_show_deprecated_templates(self, enabled: bool) -> None:
        self.show_deprecated_templates_checkbox.blockSignals(True)
        self.show_deprecated_templates_checkbox.setChecked(bool(enabled))
        self.show_deprecated_templates_checkbox.blockSignals(False)

    def set_autosave(self, enabled: bool) -> None:
        self.autosave_checkbox.blockSignals(True)
        self.autosave_checkbox.setChecked(enabled)
        self.autosave_checkbox.blockSignals(False)

    def set_sequence_history_enabled(self, enabled: bool) -> None:
        self.sequence_history_checkbox.blockSignals(True)
        self.sequence_history_checkbox.setChecked(bool(enabled))
        self.sequence_history_checkbox.blockSignals(False)

    def set_model_only_mode(self, enabled: bool) -> None:
        self._model_only_mode = bool(enabled)
        self._update_prefit_execution_control_state()
        self._update_plot_group_title()

    def set_prefit_execution_enabled(self, enabled: bool) -> None:
        self._prefit_execution_enabled = bool(enabled)
        self._update_prefit_execution_control_state()

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
        has_states = bool(state_names)
        self.saved_state_combo.setEnabled(
            has_states and self._prefit_execution_enabled
        )
        self.restore_state_button.setEnabled(
            has_states and self._prefit_execution_enabled
        )

    def selected_saved_state_name(self) -> str | None:
        text = self.saved_state_combo.currentText().strip()
        return text or None

    def populate_parameter_table(
        self,
        entries: list[PrefitParameterEntry],
    ) -> None:
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            self.parameter_table.setColumnCount(8)
            self.parameter_table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self.parameter_table.setItem(
                    row, 0, QTableWidgetItem(entry.structure)
                )
                self.parameter_table.setItem(
                    row, 1, QTableWidgetItem(entry.motif)
                )
                self.parameter_table.setItem(
                    row, 2, QTableWidgetItem(entry.name)
                )
                vary_item = QTableWidgetItem()
                vary_item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.vary
                    else Qt.CheckState.Unchecked
                )
                self.parameter_table.setItem(row, 4, vary_item)
                self._set_parameter_value_item(
                    row,
                    value=float(entry.value),
                    value_expression=entry.value_expression,
                    initial_value_expression=entry.initial_value_expression,
                )
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
                reset_button = QPushButton("Reset")
                reset_button.setToolTip(
                    "Reset this parameter to the template-default prefit "
                    "value, vary setting, and bounds."
                )
                reset_button.clicked.connect(
                    lambda _checked=False, structure=entry.structure, motif=entry.motif, name=entry.name: self.parameter_reset_requested.emit(
                        structure,
                        motif,
                        name,
                    )
                )
                self.parameter_table.setCellWidget(row, 7, reset_button)
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        self.parameter_table.resizeRowsToContents()
        self._refresh_parameter_scroll_panel()

    def set_cluster_geometry_visible(self, visible: bool) -> None:
        self._cluster_geometry_group.setVisible(bool(visible))

    def set_solute_volume_fraction_visible(self, visible: bool) -> None:
        was_visible = not self._solute_volume_fraction_group.isHidden()
        self._solute_volume_fraction_group.setVisible(bool(visible))
        if visible and not was_visible:
            self.set_solute_volume_fraction_collapsed(True)

    def set_solute_volume_fraction_target(
        self,
        parameter_name: str | None,
        fraction_kind: str | None,
        solvent_weight_parameter: str | None = None,
    ) -> None:
        target_messages: list[str] = []
        if parameter_name and fraction_kind:
            target_label = (
                "solute"
                if str(fraction_kind).strip() == "solute"
                else "solvent"
            )
            target_messages.append(
                f"{target_label} SAXS-effective interaction fraction -> {parameter_name}"
            )
        if solvent_weight_parameter:
            if parameter_name and fraction_kind:
                target_messages.append(
                    f"attenuation solvent scale -> {solvent_weight_parameter}"
                )
            else:
                target_messages.append(
                    "combined solvent background multiplier -> "
                    f"{solvent_weight_parameter}"
                )
        if target_messages:
            self.solute_volume_fraction_status_label.setText(
                "Automatic Prefit targets: " + "; ".join(target_messages) + "."
            )
        else:
            self.solute_volume_fraction_status_label.setText(
                "These estimators are available for diagnostics, but the "
                "active template does not expose an automatic Prefit target."
            )
        self.solute_volume_fraction_widget.set_target_parameter(
            parameter_name,
            fraction_kind,
            solvent_weight_parameter,
        )

    def solute_volume_fraction_is_collapsed(self) -> bool:
        return self.solute_volume_fraction_widget.isHidden()

    def set_solute_volume_fraction_collapsed(
        self,
        collapsed: bool,
    ) -> None:
        is_collapsed = bool(collapsed)
        self.solute_volume_fraction_widget.setVisible(not is_collapsed)
        self.solute_volume_fraction_collapse_button.setArrowType(
            Qt.ArrowType.RightArrow if is_collapsed else Qt.ArrowType.DownArrow
        )
        self.solute_volume_fraction_collapse_button.setText(
            "Show Estimator" if is_collapsed else "Hide Estimator"
        )

    def _toggle_solute_volume_fraction_collapsed(self) -> None:
        self.set_solute_volume_fraction_collapsed(
            not self.solute_volume_fraction_widget.isHidden()
        )

    def set_cluster_geometry_status_text(self, text: str) -> None:
        self.cluster_geometry_status_label.setText(text.strip())

    def populate_cluster_geometry_table(
        self,
        rows: list[ClusterGeometryMetadataRow],
        *,
        mapping_options: list[tuple[str, str]],
        active_radii_type: str | None = None,
        active_ionic_radius_type: str | None = None,
        allowed_sf_approximations: tuple[str, ...] | None = None,
    ) -> None:
        self._cluster_geometry_rows = [
            ClusterGeometryMetadataRow.from_dict(row.to_dict()) for row in rows
        ]
        self._cluster_geometry_mapping_options = list(mapping_options)
        self._cluster_geometry_allowed_sf_approximations = (
            tuple(allowed_sf_approximations)
            if allowed_sf_approximations
            else STRUCTURE_FACTOR_RECOMMENDATIONS
        )
        self._expanded_cluster_geometry_path_rows.clear()
        self._expanded_cluster_geometry_note_rows.clear()
        self.cluster_geometry_table.setRowCount(
            len(self._cluster_geometry_rows)
        )
        selected_radii_type = active_radii_type or (
            self._cluster_geometry_rows[0].radii_type_used
            if self._cluster_geometry_rows
            else self.cluster_geometry_active_radii_type()
        )
        self.set_cluster_geometry_active_radii_type(
            selected_radii_type,
            emit_signal=False,
        )
        selected_ionic_radius_type = active_ionic_radius_type or (
            self._cluster_geometry_rows[0].ionic_radius_type_used
            if self._cluster_geometry_rows
            else self.cluster_geometry_active_ionic_radius_type()
        )
        self.set_cluster_geometry_active_ionic_radius_type(
            selected_ionic_radius_type,
            emit_signal=False,
        )
        for row_index, row in enumerate(self._cluster_geometry_rows):
            sf_approx_combo = TableCellComboBox()
            for approximation in self._cluster_geometry_sf_approx_options():
                sf_approx_combo.addItem(
                    approximation.capitalize(),
                    userData=approximation,
                )
            selected_sf_index = sf_approx_combo.findData(row.sf_approximation)
            if selected_sf_index >= 0:
                sf_approx_combo.setCurrentIndex(selected_sf_index)
            sf_approx_combo.currentIndexChanged.connect(
                lambda _index, row_idx=row_index, combo=sf_approx_combo: (
                    self._on_cluster_geometry_sf_approximation_changed(
                        row_idx,
                        combo,
                    )
                )
            )
            self.cluster_geometry_table.setCellWidget(
                row_index,
                self.CLUSTER_COL_SF_APPROX,
                sf_approx_combo,
            )
            mapping_combo = TableCellComboBox()
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
                self.CLUSTER_COL_MAP_TO,
                mapping_combo,
            )
            self._refresh_cluster_geometry_row_display(row_index)
        self.cluster_geometry_table.resizeRowsToContents()
        self._apply_cluster_geometry_note_row_heights()

    def cluster_geometry_rows(self) -> list[ClusterGeometryMetadataRow]:
        self._commit_cluster_geometry_table_values(
            active_radii_type=self.cluster_geometry_active_radii_type(),
            active_ionic_radius_type=(
                self.cluster_geometry_active_ionic_radius_type()
            ),
        )
        return [
            ClusterGeometryMetadataRow.from_dict(row.to_dict())
            for row in self._cluster_geometry_rows
        ]

    def cluster_geometry_active_radii_type(self) -> str:
        return (
            str(self.cluster_geometry_radii_type_combo.currentData()).strip()
            or DEFAULT_RADIUS_TYPE
        )

    def cluster_geometry_active_ionic_radius_type(self) -> str:
        return (
            str(
                self.cluster_geometry_ionic_radius_type_combo.currentData()
            ).strip()
            or DEFAULT_IONIC_RADIUS_TYPE
        )

    def set_cluster_geometry_active_radii_type(
        self,
        radii_type: str,
        *,
        emit_signal: bool = False,
    ) -> None:
        index = self.cluster_geometry_radii_type_combo.findData(radii_type)
        if index < 0:
            index = self.cluster_geometry_radii_type_combo.findData(
                DEFAULT_RADIUS_TYPE
            )
        if index < 0:
            return
        if emit_signal:
            self.cluster_geometry_radii_type_combo.setCurrentIndex(index)
            return
        self.cluster_geometry_radii_type_combo.blockSignals(True)
        self.cluster_geometry_radii_type_combo.setCurrentIndex(index)
        self.cluster_geometry_radii_type_combo.blockSignals(False)
        self._apply_cluster_geometry_radii_type_to_rows(
            self.cluster_geometry_active_radii_type()
        )
        self._update_cluster_geometry_ionic_radius_type_enabled_state()

    def set_cluster_geometry_active_ionic_radius_type(
        self,
        ionic_radius_type: str,
        *,
        emit_signal: bool = False,
    ) -> None:
        index = self.cluster_geometry_ionic_radius_type_combo.findData(
            ionic_radius_type
        )
        if index < 0:
            index = self.cluster_geometry_ionic_radius_type_combo.findData(
                DEFAULT_IONIC_RADIUS_TYPE
            )
        if index < 0:
            return
        if emit_signal:
            self.cluster_geometry_ionic_radius_type_combo.setCurrentIndex(
                index
            )
            return
        self.cluster_geometry_ionic_radius_type_combo.blockSignals(True)
        self.cluster_geometry_ionic_radius_type_combo.setCurrentIndex(index)
        self.cluster_geometry_ionic_radius_type_combo.blockSignals(False)
        self._apply_cluster_geometry_ionic_radius_type_to_rows(
            self.cluster_geometry_active_ionic_radius_type()
        )
        self._update_cluster_geometry_ionic_radius_type_enabled_state()

    def parameter_entries(self) -> list[PrefitParameterEntry]:
        entries: list[PrefitParameterEntry] = []
        for row in range(self.parameter_table.rowCount()):
            value_text = self._item_text(row, 3)
            value_expression: str | None = None
            initial_value_expression: str | None = None
            vary = (
                self.parameter_table.item(row, 4).checkState()
                == Qt.CheckState.Checked
            )
            try:
                value = float(value_text)
            except (TypeError, ValueError):
                if not value_text:
                    raise ValueError(
                        "Each prefit parameter requires a numeric value or "
                        "a linked-parameter expression."
                    )
                if vary:
                    initial_value_expression = value_text
                else:
                    value_expression = value_text
                value = self._parameter_item_numeric_value(
                    self.parameter_table.item(row, 3)
                )
            entries.append(
                PrefitParameterEntry(
                    structure=self._item_text(row, 0),
                    motif=self._item_text(row, 1),
                    name=self._item_text(row, 2),
                    value=value,
                    vary=vary,
                    minimum=float(self._item_text(row, 5)),
                    maximum=float(self._item_text(row, 6)),
                    category=(
                        "weight"
                        if self._item_text(row, 2).startswith("w")
                        else "fit"
                    ),
                    value_expression=value_expression,
                    initial_value_expression=initial_value_expression,
                )
            )
        resolved_entries = resolve_prefit_parameter_entries(entries)
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            for row, entry in enumerate(resolved_entries):
                value_item = self.parameter_table.item(row, 3)
                if value_item is not None:
                    value_item.setData(
                        self.PARAMETER_VALUE_ROLE,
                        float(entry.value),
                    )
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        return resolved_entries

    def find_parameter_row(self, parameter_name: str) -> int:
        for row in range(self.parameter_table.rowCount()):
            if self._item_text(row, 2) == parameter_name:
                return row
        return -1

    def find_parameter_row_by_signature(
        self,
        structure: str,
        motif: str,
        parameter_name: str,
    ) -> int:
        for row in range(self.parameter_table.rowCount()):
            if (
                self._item_text(row, 0) == structure
                and self._item_text(row, 1) == motif
                and self._item_text(row, 2) == parameter_name
            ):
                return row
        return -1

    def set_parameter_row(
        self,
        parameter_name: str,
        *,
        structure: str | None = None,
        motif: str | None = None,
        value: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        vary: bool | None = None,
    ) -> None:
        row = (
            self.find_parameter_row_by_signature(
                structure,
                motif,
                parameter_name,
            )
            if structure is not None and motif is not None
            else self.find_parameter_row(parameter_name)
        )
        if row < 0:
            raise ValueError(f"Parameter {parameter_name} was not found.")
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            if value is not None:
                self._set_parameter_value_item(
                    row,
                    value=float(value),
                    value_expression=None,
                    initial_value_expression=None,
                )
            if vary is not None:
                vary_item = self.parameter_table.item(row, 4)
                if vary_item is not None:
                    vary_item.setCheckState(
                        Qt.CheckState.Checked
                        if vary
                        else Qt.CheckState.Unchecked
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
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        self._refresh_parameter_scroll_panel()

    def auto_update_on_parameter_change(self) -> bool:
        return bool(self.auto_update_checkbox.isChecked())

    def scrollable_parameter_enabled(self) -> bool:
        return bool(self.scrollable_parameter_checkbox.isChecked())

    def _on_parameter_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if item.column() in {3, 4}:
            self._sync_parameter_row_link_state(item.row())
        if not self._updating_parameter_table and item.column() in {
            3,
            4,
            5,
            6,
        }:
            self.parameter_table_edited.emit()
        if not self._updating_parameter_table and item.column() in {3, 4}:
            try:
                self.parameter_entries()
            except Exception:
                pass
        self._refresh_parameter_scroll_panel()
        if self._updating_parameter_table or item.column() != 3:
            return
        if not self.auto_update_on_parameter_change():
            return
        try:
            self.parameter_entries()
        except Exception:
            return
        self.update_model_requested.emit()

    def _on_auto_update_checkbox_toggled(self, enabled: bool) -> None:
        self.scrollable_parameter_checkbox.setEnabled(bool(enabled))
        if enabled:
            self._refresh_parameter_scroll_panel()
            return
        self.scrollable_parameter_checkbox.blockSignals(True)
        self.scrollable_parameter_checkbox.setChecked(False)
        self.scrollable_parameter_checkbox.blockSignals(False)
        self._refresh_parameter_scroll_panel()

    def _on_scrollable_parameter_toggled(self, enabled: bool) -> None:
        if enabled and not self.auto_update_on_parameter_change():
            self.scrollable_parameter_checkbox.blockSignals(True)
            self.scrollable_parameter_checkbox.setChecked(False)
            self.scrollable_parameter_checkbox.blockSignals(False)
        self._refresh_parameter_scroll_panel()

    def _on_parameter_table_current_cell_changed(
        self,
        current_row: int,
        current_column: int,
        previous_row: int,
        previous_column: int,
    ) -> None:
        del current_column, previous_row, previous_column
        if current_row < 0:
            return
        self._refresh_parameter_scroll_panel()

    def _refresh_parameter_scroll_panel(self) -> None:
        if (
            not self.auto_update_on_parameter_change()
            or not self.scrollable_parameter_enabled()
        ):
            self.parameter_scroll_panel.setVisible(False)
            return
        row = self.parameter_table.currentRow()
        if row < 0:
            self.parameter_scroll_panel.setVisible(False)
            return
        parameter_name = self._item_text(row, 2)
        value_item = self.parameter_table.item(row, 3)
        value_text = self._item_text(row, 3)
        uses_expression = self._parameter_value_uses_expression(value_text)
        try:
            value = (
                self._parameter_item_numeric_value(value_item)
                if uses_expression
                else float(value_text)
            )
            minimum = float(self._item_text(row, 5))
            maximum = float(self._item_text(row, 6))
        except (TypeError, ValueError):
            self.parameter_scroll_name_label.setText(
                f"{parameter_name or 'Selected parameter'} has no numeric range."
            )
            self.parameter_scroll_mode_label.setText("")
            self.parameter_scroll_value_label.setText("")
            self.parameter_scroll_bar.setEnabled(False)
            self.parameter_scroll_panel.setVisible(True)
            return
        lower = min(minimum, maximum)
        upper = max(minimum, maximum)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            self.parameter_scroll_name_label.setText(
                f"{parameter_name or 'Selected parameter'} has no usable range."
            )
            self.parameter_scroll_mode_label.setText("")
            self.parameter_scroll_value_label.setText(f"Value {value:.6g}")
            self.parameter_scroll_bar.setEnabled(False)
            self.parameter_scroll_panel.setVisible(True)
            return
        scroll_mode = self._parameter_scroll_mode(lower, upper)
        self.parameter_scroll_name_label.setText(
            f"{parameter_name} [{lower:.6g}, {upper:.6g}]"
        )
        if uses_expression:
            vary_item = self.parameter_table.item(row, 4)
            expression_mode = (
                "Initial expression seed"
                if vary_item is not None
                and vary_item.checkState() == Qt.CheckState.Checked
                else "Dependent expression"
            )
            self.parameter_scroll_mode_label.setText(
                f"{scroll_mode.capitalize()} scroll | {expression_mode}"
            )
        else:
            self.parameter_scroll_mode_label.setText(
                f"{scroll_mode.capitalize()} scroll"
            )
        self.parameter_scroll_value_label.setText(f"Value {value:.6g}")
        position = self._parameter_scroll_position_for_value(
            value,
            lower,
            upper,
            scroll_mode,
        )
        self._updating_parameter_scrollbar = True
        self.parameter_scroll_bar.blockSignals(True)
        try:
            self.parameter_scroll_bar.setEnabled(True)
            self.parameter_scroll_bar.setValue(position)
        finally:
            self.parameter_scroll_bar.blockSignals(False)
            self._updating_parameter_scrollbar = False
        self.parameter_scroll_panel.setVisible(True)

    def _on_parameter_scrollbar_value_changed(self, position: int) -> None:
        if self._updating_parameter_scrollbar:
            return
        row = self.parameter_table.currentRow()
        if row < 0:
            return
        try:
            minimum = float(self._item_text(row, 5))
            maximum = float(self._item_text(row, 6))
        except (TypeError, ValueError):
            return
        lower = min(minimum, maximum)
        upper = max(minimum, maximum)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            return
        scroll_mode = self._parameter_scroll_mode(lower, upper)
        value = self._parameter_scroll_value_for_position(
            position,
            lower,
            upper,
            scroll_mode,
        )
        item = self.parameter_table.item(row, 3)
        if item is None:
            item = QTableWidgetItem()
            self.parameter_table.setItem(row, 3, item)
        formatted = f"{float(value):.6g}"
        self.parameter_scroll_value_label.setText(f"Value {formatted}")
        if item.text().strip() == formatted:
            return
        item.setText(formatted)

    def _parameter_scroll_mode(
        self,
        minimum: float,
        maximum: float,
    ) -> str:
        if minimum == 0.0 or maximum == 0.0 or minimum * maximum < 0.0:
            return "linear"
        decade_span = abs(
            np.log10(abs(float(maximum))) - np.log10(abs(float(minimum)))
        )
        return (
            "log"
            if decade_span >= self.PARAMETER_SCROLL_LOG_DECADE_THRESHOLD
            else "linear"
        )

    def _parameter_scroll_position_for_value(
        self,
        value: float,
        minimum: float,
        maximum: float,
        scroll_mode: str,
    ) -> int:
        clamped_value = min(max(float(value), minimum), maximum)
        lower = self._parameter_scroll_transform(
            minimum,
            minimum,
            maximum,
            scroll_mode,
        )
        upper = self._parameter_scroll_transform(
            maximum,
            minimum,
            maximum,
            scroll_mode,
        )
        transformed_value = self._parameter_scroll_transform(
            clamped_value,
            minimum,
            maximum,
            scroll_mode,
        )
        if np.isclose(upper, lower):
            return 0
        fraction = (transformed_value - lower) / (upper - lower)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return int(round(fraction * self.PARAMETER_SCROLL_RESOLUTION))

    def _parameter_scroll_value_for_position(
        self,
        position: int,
        minimum: float,
        maximum: float,
        scroll_mode: str,
    ) -> float:
        lower = self._parameter_scroll_transform(
            minimum,
            minimum,
            maximum,
            scroll_mode,
        )
        upper = self._parameter_scroll_transform(
            maximum,
            minimum,
            maximum,
            scroll_mode,
        )
        fraction = float(position) / float(self.PARAMETER_SCROLL_RESOLUTION)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        transformed_value = lower + fraction * (upper - lower)
        value = self._parameter_scroll_inverse_transform(
            transformed_value,
            minimum,
            maximum,
            scroll_mode,
        )
        return float(np.clip(value, minimum, maximum))

    @staticmethod
    def _parameter_scroll_transform(
        value: float,
        minimum: float,
        maximum: float,
        scroll_mode: str,
    ) -> float:
        numeric_value = float(value)
        if scroll_mode != "log":
            return numeric_value
        if minimum < 0.0 and maximum < 0.0:
            return -float(np.log10(-numeric_value))
        return float(np.log10(numeric_value))

    @staticmethod
    def _parameter_scroll_inverse_transform(
        value: float,
        minimum: float,
        maximum: float,
        scroll_mode: str,
    ) -> float:
        numeric_value = float(value)
        if scroll_mode != "log":
            return numeric_value
        if minimum < 0.0 and maximum < 0.0:
            return -(10.0 ** (-numeric_value))
        return 10.0**numeric_value

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
        self._update_prefit_trace_toggle_state(evaluation)
        self._update_plot_group_title()
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

        has_experimental = evaluation.experimental_intensities is not None
        has_residuals = evaluation.residuals is not None
        if has_experimental and has_residuals:
            grid = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
            top = self.figure.add_subplot(grid[0, 0])
            bottom = self.figure.add_subplot(grid[1, 0], sharex=top)
        else:
            top = self.figure.add_subplot(111)
            bottom = None

        plotted_lines = []
        structure_axis = None

        if (
            has_experimental
            and self.show_experimental_trace_checkbox.isChecked()
        ):
            (experimental_line,) = top.plot(
                evaluation.q_values,
                evaluation.experimental_intensities,
                color="black",
                label="Experimental",
            )
            plotted_lines.append(experimental_line)

        if (
            self.show_solvent_trace_checkbox.isChecked()
            and evaluation.solvent_contribution is not None
        ):
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

        if (
            self.show_structure_factor_trace_checkbox.isChecked()
            and evaluation.structure_factor_trace is not None
        ):
            structure_values = np.asarray(
                evaluation.structure_factor_trace,
                dtype=float,
            )
            structure_mask = np.isfinite(structure_values)
            if np.any(structure_mask):
                structure_axis = top.twinx()
                structure_axis.set_xscale(
                    "log" if self.log_x_checkbox.isChecked() else "linear"
                )
                (structure_line,) = structure_axis.plot(
                    np.asarray(evaluation.q_values, dtype=float)[
                        structure_mask
                    ],
                    structure_values[structure_mask],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.5,
                    label="Structure factor S(q)",
                )
                structure_axis.set_ylabel("S(q)", color="tab:purple")
                structure_axis.tick_params(axis="y", colors="tab:purple")
                structure_axis.spines["right"].set_color("tab:purple")
                plotted_lines.append(structure_line)

        if self.show_model_trace_checkbox.isChecked():
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
        top.text(
            0.02,
            0.02,
            "\n".join(self._prefit_metric_lines(evaluation)),
            transform=top.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.85,
            },
        )
        if plotted_lines:
            self._build_interactive_legend(top, plotted_lines)

        if bottom is not None and evaluation.residuals is not None:
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
        else:
            top.set_xlabel("q (Å⁻¹)")
        self.figure.tight_layout()
        self.canvas.draw()

    def current_evaluation(self) -> PrefitEvaluation | None:
        return self._current_evaluation

    @staticmethod
    def _prefit_metric_lines(
        evaluation: PrefitEvaluation,
    ) -> list[str]:
        if (
            evaluation.experimental_intensities is None
            or evaluation.residuals is None
        ):
            return [
                "Model Only Mode",
                "Experimental fit metrics unavailable",
            ]
        experimental_values = np.asarray(
            evaluation.experimental_intensities,
            dtype=float,
        )
        model_values = np.asarray(evaluation.model_intensities, dtype=float)
        residuals = np.asarray(model_values - experimental_values, dtype=float)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mean_abs_residual = float(np.mean(np.abs(residuals)))
        experimental_mean = float(np.mean(experimental_values))
        total_sum_squares = float(
            np.sum((experimental_values - experimental_mean) ** 2)
        )
        residual_sum_squares = float(np.sum(residuals**2))
        r_squared = (
            float(1.0 - (residual_sum_squares / total_sum_squares))
            if total_sum_squares > 0.0
            else 1.0
        )
        metric_lines = [
            f"RMSE: {rmse:.4g}",
            f"Mean |res|: {mean_abs_residual:.4g}",
            f"R²: {r_squared:.4g}",
        ]
        non_positive_model_points = int(
            np.count_nonzero(np.isfinite(model_values) & (model_values <= 0.0))
        )
        if non_positive_model_points:
            metric_lines.append(
                f"Model <= 0 at {non_positive_model_points} q-points"
            )
        return metric_lines

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

    def set_stoichiometry_status_text(self, text: str) -> None:
        self.stoichiometry_status_label.setText(text.strip())

    def set_console_autoscroll_enabled(self, enabled: bool) -> None:
        self._console_autoscroll_enabled = bool(enabled)
        if self._console_autoscroll_enabled:
            self._scroll_output_to_end()

    def _item_text(self, row: int, column: int) -> str:
        item = self.parameter_table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _parameter_item_numeric_value(
        self,
        item: QTableWidgetItem | None,
    ) -> float:
        if item is None:
            return 0.0
        raw_value = item.data(self.PARAMETER_VALUE_ROLE)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 0.0

    def _parameter_value_uses_expression(self, value_text: str) -> bool:
        stripped = value_text.strip()
        if not stripped:
            return False
        try:
            float(stripped)
        except (TypeError, ValueError):
            return True
        return False

    def _set_parameter_value_item(
        self,
        row: int,
        *,
        value: float,
        value_expression: str | None,
        initial_value_expression: str | None,
    ) -> None:
        display_text = (
            value_expression.strip()
            if value_expression is not None and value_expression.strip()
            else (
                initial_value_expression.strip()
                if initial_value_expression is not None
                and initial_value_expression.strip()
                else f"{float(value):.6g}"
            )
        )
        value_item = QTableWidgetItem(display_text)
        value_item.setData(self.PARAMETER_VALUE_ROLE, float(value))
        self.parameter_table.setItem(row, 3, value_item)
        self._sync_parameter_row_link_state(row)

    def _sync_parameter_row_link_state(self, row: int) -> None:
        if row < 0:
            return
        vary_item = self.parameter_table.item(row, 4)
        value_item = self.parameter_table.item(row, 3)
        if vary_item is None or value_item is None:
            return
        linked = self._parameter_value_uses_expression(self._item_text(row, 3))
        resolved_value = self._parameter_item_numeric_value(value_item)
        was_updating = self._updating_parameter_table
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            vary_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            if not linked:
                vary_item.setToolTip(
                    "Enable to vary this parameter during fitting."
                )
                value_item.setToolTip("")
                return
            if vary_item.checkState() == Qt.CheckState.Checked:
                vary_item.setToolTip(
                    "Artemis-style guess behavior: with Vary enabled, the "
                    "Value expression is evaluated into the starting numeric "
                    "value and the parameter may still refine within Min/Max."
                )
                value_item.setToolTip(
                    "Initial expression seed. With Vary enabled, the Value "
                    "expression resolves to the current numeric value "
                    f"({resolved_value:.6g}) before fitting, but the "
                    "parameter then varies independently."
                )
                return
            vary_item.setToolTip(
                "Artemis-style def behavior: with Vary disabled, the Value "
                "expression is treated as a live dependent parameter and its "
                "Min/Max are ignored during fitting."
            )
            value_item.setToolTip(
                "Dependent expression. With Vary disabled, this parameter "
                "follows the expression entered in the Value column. Current "
                f"resolved value: {resolved_value:.6g}."
            )
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = was_updating

    def _redraw_current_plot(self) -> None:
        self.plot_evaluation(self._current_evaluation)

    def _update_prefit_execution_control_state(self) -> None:
        enabled = bool(self._prefit_execution_enabled)
        self.method_combo.setEnabled(enabled)
        self.nfev_spin.setEnabled(enabled)
        self.saved_state_combo.setEnabled(
            enabled and self.saved_state_combo.count() > 0
        )
        self.restore_state_button.setEnabled(
            enabled and self.saved_state_combo.count() > 0
        )
        self.run_button.setEnabled(enabled)
        self.recommended_scale_button.setEnabled(enabled)
        self.autosave_checkbox.setEnabled(enabled)
        self.save_button.setEnabled(enabled)

    def _update_plot_group_title(self) -> None:
        has_experimental = (
            self._current_evaluation is not None
            and self._current_evaluation.experimental_intensities is not None
        )
        if self._model_only_mode or not has_experimental:
            self._plot_group.setTitle("Model Preview")
            return
        self._plot_group.setTitle("Model vs Experimental")

    def _update_prefit_trace_toggle_state(
        self,
        evaluation: PrefitEvaluation | None,
    ) -> None:
        has_evaluation = evaluation is not None
        has_experimental = (
            has_evaluation and evaluation.experimental_intensities is not None
        )
        has_solvent = (
            has_evaluation and evaluation.solvent_contribution is not None
        )
        has_structure_factor = (
            has_evaluation and evaluation.structure_factor_trace is not None
        )
        self.show_experimental_trace_checkbox.setEnabled(
            bool(has_experimental)
        )
        self.show_model_trace_checkbox.setEnabled(has_evaluation)
        self.show_solvent_trace_checkbox.setEnabled(bool(has_solvent))
        self.show_structure_factor_trace_checkbox.setEnabled(
            bool(has_structure_factor)
        )

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

    def _on_cluster_geometry_sf_approximation_changed(
        self,
        row_index: int,
        combo: QComboBox,
    ) -> None:
        if row_index < 0 or row_index >= len(self._cluster_geometry_rows):
            return
        previous_approximation = str(
            self._cluster_geometry_rows[row_index].sf_approximation
        ).strip()
        self._commit_cluster_geometry_table_values(
            row_index=row_index,
            active_radii_type=self.cluster_geometry_active_radii_type(),
            active_ionic_radius_type=(
                self.cluster_geometry_active_ionic_radius_type()
            ),
        )
        self._cluster_geometry_rows[row_index].sf_approximation = (
            str(combo.currentData()).strip()
            or self._cluster_geometry_sf_approx_options()[0]
        )
        synchronize_cluster_geometry_row(
            self._cluster_geometry_rows[row_index],
            active_radii_type=self.cluster_geometry_active_radii_type(),
            active_ionic_radius_type=(
                self.cluster_geometry_active_ionic_radius_type()
            ),
        )
        self._refresh_cluster_geometry_row_display(row_index)
        self.cluster_geometry_sf_approximation_changed.emit(
            previous_approximation,
            self._cluster_geometry_rows[row_index].sf_approximation,
        )
        self.cluster_geometry_mapping_changed.emit()

    def _on_cluster_geometry_radii_type_changed(self) -> None:
        active_radii_type = self.cluster_geometry_active_radii_type()
        self._commit_cluster_geometry_table_values(
            active_radii_type=self._last_cluster_geometry_radii_type,
            active_ionic_radius_type=(
                self._last_cluster_geometry_ionic_radius_type
            ),
        )
        self._apply_cluster_geometry_radii_type_to_rows(active_radii_type)
        self.cluster_geometry_radii_type_changed.emit(active_radii_type)
        self._update_cluster_geometry_ionic_radius_type_enabled_state()

    def _on_cluster_geometry_ionic_radius_type_changed(self) -> None:
        active_ionic_radius_type = (
            self.cluster_geometry_active_ionic_radius_type()
        )
        self._commit_cluster_geometry_table_values(
            active_radii_type=self.cluster_geometry_active_radii_type(),
            active_ionic_radius_type=(
                self._last_cluster_geometry_ionic_radius_type
            ),
        )
        self._apply_cluster_geometry_ionic_radius_type_to_rows(
            active_ionic_radius_type
        )
        self.cluster_geometry_ionic_radius_type_changed.emit(
            active_ionic_radius_type
        )

    def _toggle_cluster_geometry_radii_type(self) -> None:
        current_type = self.cluster_geometry_active_radii_type()
        next_type = "bond_length" if current_type == "ionic" else "ionic"
        self.set_cluster_geometry_active_radii_type(
            next_type,
            emit_signal=True,
        )

    def _apply_cluster_geometry_radii_type_to_rows(
        self,
        active_radii_type: str,
    ) -> None:
        for row_index, row in enumerate(self._cluster_geometry_rows):
            synchronize_cluster_geometry_row(
                row,
                active_radii_type=active_radii_type,
                active_ionic_radius_type=(
                    self.cluster_geometry_active_ionic_radius_type()
                ),
            )
            self._refresh_cluster_geometry_row_display(row_index)
        self._last_cluster_geometry_radii_type = active_radii_type
        self.cluster_geometry_table.resizeRowsToContents()
        self._apply_cluster_geometry_note_row_heights()

    def _apply_cluster_geometry_ionic_radius_type_to_rows(
        self,
        active_ionic_radius_type: str,
    ) -> None:
        for row_index, row in enumerate(self._cluster_geometry_rows):
            synchronize_cluster_geometry_row(
                row,
                active_radii_type=self.cluster_geometry_active_radii_type(),
                active_ionic_radius_type=active_ionic_radius_type,
            )
            self._refresh_cluster_geometry_row_display(row_index)
        self._last_cluster_geometry_ionic_radius_type = (
            active_ionic_radius_type
        )
        self.cluster_geometry_table.resizeRowsToContents()
        self._apply_cluster_geometry_note_row_heights()

    def _apply_cluster_geometry_note_row_heights(self) -> None:
        collapsed_height = max(
            self.cluster_geometry_table.verticalHeader().defaultSectionSize(),
            30,
        )
        for row_index in range(len(self._cluster_geometry_rows)):
            if (
                row_index in self._expanded_cluster_geometry_note_rows
                or row_index in self._expanded_cluster_geometry_path_rows
            ):
                self.cluster_geometry_table.resizeRowToContents(row_index)
            else:
                self.cluster_geometry_table.setRowHeight(
                    row_index,
                    collapsed_height,
                )

    def _commit_cluster_geometry_table_values(
        self,
        *,
        active_radii_type: str,
        active_ionic_radius_type: str,
        row_index: int | None = None,
    ) -> None:
        if row_index is None:
            row_indices = range(len(self._cluster_geometry_rows))
        else:
            row_indices = (row_index,)
        pending_updates: list[tuple[int, dict[str, object]]] = []
        invalid_entries: list[str] = []
        for current_row_index in row_indices:
            update_payload, row_invalid_entries = (
                self._collect_cluster_geometry_row_update(
                    current_row_index,
                    active_radii_type=active_radii_type,
                    active_ionic_radius_type=active_ionic_radius_type,
                )
            )
            invalid_entries.extend(row_invalid_entries)
            if update_payload is not None:
                pending_updates.append((current_row_index, update_payload))
        if invalid_entries:
            raise ValueError(
                "Cluster geometry radii must be positive before updating "
                "the model. Invalid values: " + "; ".join(invalid_entries)
            )
        for current_row_index, update_payload in pending_updates:
            self._apply_cluster_geometry_row_update(
                current_row_index,
                update_payload=update_payload,
                active_radii_type=active_radii_type,
                active_ionic_radius_type=active_ionic_radius_type,
            )

    def _collect_cluster_geometry_row_update(
        self,
        row_index: int,
        *,
        active_radii_type: str,
        active_ionic_radius_type: str,
    ) -> tuple[dict[str, object] | None, list[str]]:
        del active_ionic_radius_type
        if row_index < 0 or row_index >= len(self._cluster_geometry_rows):
            return None, []
        row = self._cluster_geometry_rows[row_index]
        invalid_entries: list[str] = []
        if row.sf_approximation == "ellipsoid":
            semiaxes = [
                self._parse_cluster_geometry_numeric_item(
                    row_index,
                    self.CLUSTER_COL_SEMIAXIS_X,
                    label="Semiaxis X",
                    cluster_id=row.cluster_id,
                    invalid_entries=invalid_entries,
                ),
                self._parse_cluster_geometry_numeric_item(
                    row_index,
                    self.CLUSTER_COL_SEMIAXIS_Y,
                    label="Semiaxis Y",
                    cluster_id=row.cluster_id,
                    invalid_entries=invalid_entries,
                ),
                self._parse_cluster_geometry_numeric_item(
                    row_index,
                    self.CLUSTER_COL_SEMIAXIS_Z,
                    label="Semiaxis Z",
                    cluster_id=row.cluster_id,
                    invalid_entries=invalid_entries,
                ),
            ]
            if invalid_entries:
                return None, invalid_entries
            return (
                {
                    "kind": "ellipsoid",
                    "values": np.asarray(semiaxes, dtype=float),
                },
                [],
            )
        radius = self._parse_cluster_geometry_numeric_item(
            row_index,
            self.CLUSTER_COL_EFFECTIVE_RADIUS,
            label="Effective Radius",
            cluster_id=row.cluster_id,
            invalid_entries=invalid_entries,
        )
        if invalid_entries:
            return None, invalid_entries
        return (
            {
                "kind": "sphere",
                "value": float(radius),
            },
            [],
        )

    def _apply_cluster_geometry_row_update(
        self,
        row_index: int,
        *,
        update_payload: dict[str, object],
        active_radii_type: str,
        active_ionic_radius_type: str,
    ) -> None:
        row = self._cluster_geometry_rows[row_index]
        if str(update_payload.get("kind")) == "ellipsoid":
            semiaxes = np.asarray(
                update_payload.get("values", (0.0, 0.0, 0.0)),
                dtype=float,
            )
            if active_radii_type == "bond_length":
                row.bond_length_ellipsoid_semiaxis_a = float(semiaxes[0])
                row.bond_length_ellipsoid_semiaxis_b = float(semiaxes[1])
                row.bond_length_ellipsoid_semiaxis_c = float(semiaxes[2])
            elif active_ionic_radius_type == "crystal":
                row.crystal_ionic_ellipsoid_semiaxis_a = float(semiaxes[0])
                row.crystal_ionic_ellipsoid_semiaxis_b = float(semiaxes[1])
                row.crystal_ionic_ellipsoid_semiaxis_c = float(semiaxes[2])
            else:
                row.ionic_ellipsoid_semiaxis_a = float(semiaxes[0])
                row.ionic_ellipsoid_semiaxis_b = float(semiaxes[1])
                row.ionic_ellipsoid_semiaxis_c = float(semiaxes[2])
        else:
            radius = float(update_payload.get("value", 0.0))
            if active_radii_type == "bond_length":
                row.bond_length_sphere_effective_radius = float(radius)
            elif active_ionic_radius_type == "crystal":
                row.crystal_ionic_sphere_effective_radius = float(radius)
            else:
                row.ionic_sphere_effective_radius = float(radius)
        synchronize_cluster_geometry_row(
            row,
            active_radii_type=active_radii_type,
            active_ionic_radius_type=active_ionic_radius_type,
        )

    def _refresh_cluster_geometry_row_display(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._cluster_geometry_rows):
            return
        row = self._cluster_geometry_rows[row_index]
        is_ellipsoid = row.sf_approximation == "ellipsoid"
        path_text = row.cluster_path
        path_preview = (
            path_text
            if row_index in self._expanded_cluster_geometry_path_rows
            else self._collapsed_cluster_geometry_path_text(path_text)
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_CLUSTER,
            self._make_cluster_geometry_item(row.cluster_id),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_PATH,
            self._make_cluster_geometry_item(
                path_preview,
                tool_tip=path_text,
            ),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_AVG_SIZE,
            self._make_cluster_geometry_item(f"{row.avg_size_metric:.4g}"),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_EFFECTIVE_RADIUS,
            self._make_cluster_geometry_item(
                f"{row.effective_radius:.4g}",
                editable=not is_ellipsoid,
                color=(
                    self.ACTIVE_CLUSTER_GEOMETRY_COLOR
                    if not is_ellipsoid
                    else self.INACTIVE_CLUSTER_GEOMETRY_COLOR
                ),
            ),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_RADII_TYPE,
            self._make_cluster_geometry_item(
                self._radii_type_label(
                    row.radii_type_used,
                    ionic_radius_type=row.ionic_radius_type_used,
                )
            ),
        )
        semiaxis_color = (
            self.ACTIVE_CLUSTER_GEOMETRY_COLOR
            if is_ellipsoid
            else self.INACTIVE_CLUSTER_GEOMETRY_COLOR
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_SEMIAXIS_X,
            self._make_cluster_geometry_item(
                f"{row.active_semiaxis_a:.4g}",
                editable=is_ellipsoid,
                color=semiaxis_color,
            ),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_SEMIAXIS_Y,
            self._make_cluster_geometry_item(
                f"{row.active_semiaxis_b:.4g}",
                editable=is_ellipsoid,
                color=semiaxis_color,
            ),
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_SEMIAXIS_Z,
            self._make_cluster_geometry_item(
                f"{row.active_semiaxis_c:.4g}",
                editable=is_ellipsoid,
                color=semiaxis_color,
            ),
        )
        sf_approx_combo = self.cluster_geometry_table.cellWidget(
            row_index,
            self.CLUSTER_COL_SF_APPROX,
        )
        if isinstance(sf_approx_combo, QComboBox):
            sf_approx_combo.blockSignals(True)
            sf_index = sf_approx_combo.findData(row.sf_approximation)
            if sf_index >= 0:
                sf_approx_combo.setCurrentIndex(sf_index)
            sf_approx_combo.blockSignals(False)
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_ANISOTROPY,
            self._make_cluster_geometry_item(f"{row.anisotropy_metric:.4g}"),
        )
        notes_text = self._cluster_geometry_full_notes_text(row)
        notes_preview = (
            notes_text
            if row_index in self._expanded_cluster_geometry_note_rows
            else self._collapsed_cluster_geometry_notes_text(notes_text)
        )
        self.cluster_geometry_table.setItem(
            row_index,
            self.CLUSTER_COL_NOTES,
            self._make_cluster_geometry_item(
                notes_preview,
                tool_tip=notes_text,
            ),
        )
        if (
            row_index in self._expanded_cluster_geometry_note_rows
            or row_index in self._expanded_cluster_geometry_path_rows
        ):
            self.cluster_geometry_table.resizeRowToContents(row_index)
        else:
            self.cluster_geometry_table.setRowHeight(
                row_index,
                max(
                    self.cluster_geometry_table.verticalHeader().defaultSectionSize(),
                    30,
                ),
            )

    def _on_cluster_geometry_cell_double_clicked(
        self,
        row_index: int,
        column_index: int,
    ) -> None:
        if row_index < 0 or row_index >= len(self._cluster_geometry_rows):
            return
        if column_index == self.CLUSTER_COL_NOTES:
            if row_index in self._expanded_cluster_geometry_note_rows:
                self._expanded_cluster_geometry_note_rows.remove(row_index)
            else:
                self._expanded_cluster_geometry_note_rows.add(row_index)
        elif column_index == self.CLUSTER_COL_PATH:
            if row_index in self._expanded_cluster_geometry_path_rows:
                self._expanded_cluster_geometry_path_rows.remove(row_index)
            else:
                self._expanded_cluster_geometry_path_rows.add(row_index)
        else:
            return
        self._refresh_cluster_geometry_row_display(row_index)

    @staticmethod
    def _collapsed_cluster_geometry_notes_text(
        text: str,
        *,
        max_length: int = 88,
    ) -> str:
        collapsed = " ".join(str(text).split())
        if len(collapsed) > max_length:
            collapsed = collapsed[: max_length - 1].rstrip() + "..."
        return collapsed

    @staticmethod
    def _collapsed_cluster_geometry_path_text(
        text: str,
        *,
        max_length: int = 72,
    ) -> str:
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= max_length:
            return collapsed
        keep = max(max_length - 3, 8)
        tail_length = max(int(keep * 0.68), 8)
        head_length = max(keep - tail_length, 4)
        return (
            collapsed[:head_length].rstrip()
            + "..."
            + collapsed[-tail_length:].lstrip()
        )

    @staticmethod
    def _cluster_geometry_full_notes_text(
        row: ClusterGeometryMetadataRow,
    ) -> str:
        notes_text = row.notes
        if row.sf_approximation == "ellipsoid":
            return (
                f"{notes_text}\nActive ellipsoid semiaxes: "
                f"{row.active_semiaxis_a:.3f}, "
                f"{row.active_semiaxis_b:.3f}, "
                f"{row.active_semiaxis_c:.3f} A"
            )
        return (
            f"{notes_text}\nActive sphere radius: "
            f"{row.effective_radius:.3f} A"
        )

    def _cluster_geometry_sf_approx_options(self) -> tuple[str, ...]:
        if self._cluster_geometry_allowed_sf_approximations:
            return self._cluster_geometry_allowed_sf_approximations
        return STRUCTURE_FACTOR_RECOMMENDATIONS

    @staticmethod
    def _make_cluster_geometry_item(
        text: str,
        *,
        tool_tip: str | None = None,
        editable: bool = False,
        color: QColor | None = None,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        if tool_tip:
            item.setToolTip(tool_tip)
        if color is not None:
            item.setForeground(color)
        return item

    def _parse_cluster_geometry_numeric_item(
        self,
        row: int,
        column: int,
        *,
        label: str,
        cluster_id: str,
        invalid_entries: list[str],
    ) -> float | None:
        item = self.cluster_geometry_table.item(row, column)
        text = item.text().strip() if item is not None else ""
        try:
            value = float(text)
        except Exception:
            invalid_entries.append(f"{cluster_id} {label}={text!r}")
            return None
        if not np.isfinite(value) or value <= 0.0:
            invalid_entries.append(f"{cluster_id} {label}={value:.6g}")
            return None
        return value

    @staticmethod
    def _radii_type_label(
        radii_type: str,
        *,
        ionic_radius_type: str = DEFAULT_IONIC_RADIUS_TYPE,
    ) -> str:
        if str(radii_type).strip() == "bond_length":
            return "Bond length"
        if str(ionic_radius_type).strip() == "crystal":
            return "Ionic (crystal)"
        return "Ionic (effective)"

    @staticmethod
    def _ionic_radius_type_label(ionic_radius_type: str) -> str:
        if str(ionic_radius_type).strip() == "crystal":
            return "Crystal ionic"
        return "Effective ionic"

    def _update_cluster_geometry_ionic_radius_type_enabled_state(self) -> None:
        self.cluster_geometry_ionic_radius_type_combo.setEnabled(
            self.cluster_geometry_active_radii_type() == "ionic"
        )

    def start_cluster_geometry_progress(self, message: str) -> None:
        self.cluster_geometry_progress_label.setText(message)
        self.cluster_geometry_progress_bar.setRange(0, 0)
        self.cluster_geometry_progress_bar.setValue(0)
        self.cluster_geometry_progress_bar.setFormat("")

    def update_cluster_geometry_progress(
        self,
        processed: int,
        total: int,
        message: str,
        *,
        unit_label: str = "files",
    ) -> None:
        bounded_total = max(int(total), 1)
        bounded_processed = max(0, min(int(processed), bounded_total))
        self.cluster_geometry_progress_bar.setRange(0, bounded_total)
        self.cluster_geometry_progress_bar.setValue(bounded_processed)
        self.cluster_geometry_progress_bar.setFormat(f"%v / %m {unit_label}")
        self.cluster_geometry_progress_label.setText(message)

    def finish_cluster_geometry_progress(self, message: str) -> None:
        self.cluster_geometry_progress_label.setText(message)
        if self.cluster_geometry_progress_bar.maximum() <= 0:
            self.cluster_geometry_progress_bar.setRange(0, 1)
        self.cluster_geometry_progress_bar.setValue(
            self.cluster_geometry_progress_bar.maximum()
        )
        self.cluster_geometry_progress_bar.setFormat("%v / %m files")

    def reset_cluster_geometry_progress(self) -> None:
        self.cluster_geometry_progress_label.setText("Progress: idle")
        self.cluster_geometry_progress_bar.setRange(0, 1)
        self.cluster_geometry_progress_bar.setValue(0)
        self.cluster_geometry_progress_bar.setFormat("%v / %m files")

    def _build_interactive_legend(self, axis, lines: list[object]) -> None:
        legend = axis.legend(handles=lines, loc="best")
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
        del scroll_to_end
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
        scrollbar = self.output_box.verticalScrollBar()
        previous_value = scrollbar.value()
        previous_maximum = max(scrollbar.maximum(), 1)
        self.output_box.setPlainText("\n\n".join(sections).strip())
        if self._console_autoscroll_enabled:
            self._scroll_output_to_end()
            return
        updated_scrollbar = self.output_box.verticalScrollBar()
        if updated_scrollbar.maximum() > 0:
            position_fraction = previous_value / previous_maximum
            updated_scrollbar.setValue(
                int(round(position_fraction * updated_scrollbar.maximum()))
            )

    def _scroll_output_to_end(self) -> None:
        cursor = self.output_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_box.setTextCursor(cursor)
        self.output_box.ensureCursorVisible()
        scrollbar = self.output_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QTimer.singleShot(
            0,
            self._scroll_output_to_end_once,
        )

    def _scroll_output_to_end_once(self) -> None:
        cursor = self.output_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_box.setTextCursor(cursor)
        self.output_box.ensureCursorVisible()
        scrollbar = self.output_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_ionic_radius_help(self) -> None:
        QMessageBox.information(
            self,
            "Ionic Radius Estimate Help",
            self.IONIC_RADIUS_HELP_TEXT,
        )

    def _show_solute_volume_fraction_help(self) -> None:
        QMessageBox.information(
            self,
            "Solution Scattering Estimate Help",
            self.SOLUTE_VOLUME_FRACTION_HELP_TEXT,
        )

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
