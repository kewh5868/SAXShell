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
    QDoubleSpinBox,
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

from saxshell.plotting import (
    Q_A_INVERSE_LABEL,
    LinePlotDefaults,
    LinePlotEditorControls,
    LinePlotSeriesDefaults,
    LinePlotSettings,
    PlotEditorWindow,
    apply_axis_scales,
)
from saxshell.saxs._model_templates import TemplateSpec
from saxshell.saxs.dielectric_presets import (
    DIELECTRIC_CONSTANT_PRESETS,
    DIELECTRIC_CONSTANT_PRESETS_BY_KEY,
)
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
from saxshell.saxs.stoichiometry_compensator import (
    guess_single_atom_compensator_names,
)
from saxshell.saxs.ui._pane_snap import PaneSnapFilter
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
    PARAM_COL_STRUCTURE = 0
    PARAM_COL_MOTIF = 1
    PARAM_COL_NAME = 2
    PARAM_COL_VALUE = 3
    PARAM_COL_VARY = 4
    PARAM_COL_MIN = 5
    PARAM_COL_MAX = 6
    PARAM_COL_ACTIVE = 7
    PARAM_COL_RESET = 8

    template_changed = Signal(str)
    change_template_requested = Signal(str)
    show_deprecated_templates_changed = Signal(bool)
    autosave_toggled = Signal(bool)
    sequence_history_toggled = Signal(bool)
    field_interaction_requested = Signal()
    parameter_table_edited = Signal()
    parameter_reset_requested = Signal(str, str, str)
    update_model_requested = Signal()
    charge_estimate_requested = Signal()
    run_fit_requested = Signal()
    undo_fit_requested = Signal()
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
    stoichiometry_compensator_settings_changed = Signal()
    fit_range_changed = Signal(float, float)

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
    DIELECTRIC_PARAMETER_NAME = "dielectconst"
    CHARGE_PARAMETER_NAME = "charge"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._auto_snap_enabled = True
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
        self._fit_undo_available = False
        self._updating_parameter_table = False
        self._updating_parameter_scrollbar = False
        self._parameter_entries_dirty = True
        self._cached_parameter_entries: list[PrefitParameterEntry] | None = (
            None
        )
        self._stoichiometry_compensator_entries: list[PrefitParameterEntry] = (
            []
        )
        self._updating_stoichiometry_compensator = False
        self._updating_fit_range_controls = False
        self._prefit_range_drag_start: float | None = None
        self._last_cluster_geometry_radii_type = DEFAULT_RADIUS_TYPE
        self._last_cluster_geometry_ionic_radius_type = (
            DEFAULT_IONIC_RADIUS_TYPE
        )
        self._active_template_name: str | None = None
        self._line_plot_settings = LinePlotSettings()
        self._plot_editor_window: PlotEditorWindow | None = None
        self._plot_editor_controls: LinePlotEditorControls | None = None
        self._suspend_template_selection_signal = False
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
        self._stoichiometry_compensator_group = (
            self._build_stoichiometry_compensator_group()
        )
        left_layout.addWidget(self._stoichiometry_compensator_group)
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
        self._auto_snap_filter = PaneSnapFilter(
            self._pane_splitter,
            self._left_scroll_area,
            self._plot_group,
            parent=self,
        )
        self.set_auto_snap_enabled(self._auto_snap_enabled)
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

        self.active_template_edit = QLineEdit()
        self.active_template_edit.setReadOnly(True)
        self.change_template_button = QPushButton("Change Template")
        self.change_template_button.setEnabled(False)
        self.change_template_button.clicked.connect(
            self._emit_change_template_requested
        )
        active_row = QWidget()
        active_layout = QHBoxLayout(active_row)
        active_layout.setContentsMargins(0, 0, 0, 0)
        active_layout.addWidget(self.active_template_edit, stretch=1)
        active_layout.addWidget(self.change_template_button)
        layout.addWidget(QLabel("Active Template"), 1, 0)
        layout.addWidget(active_row, 1, 1, 1, 3)

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["leastsq", "nelder", "powell", "differential_evolution"]
        )
        layout.addWidget(QLabel("Minimizer"), 2, 0)
        layout.addWidget(self.method_combo, 2, 1, 1, 2)

        self.nfev_spin = QSpinBox()
        self.nfev_spin.setRange(100, 10_000_000)
        self.nfev_spin.setValue(10_000)
        layout.addWidget(QLabel("Max nfev"), 3, 0)
        layout.addWidget(self.nfev_spin, 3, 1, 1, 2)

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
        layout.addWidget(self.sequence_history_checkbox, 4, 0, 1, 3)

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
        layout.addWidget(QLabel("Saved states"), 5, 0)
        layout.addWidget(restore_row, 5, 1, 1, 2)

        self.stoichiometry_status_label = QLabel(
            "Stoichiometry monitor: configure target elements and ratio in "
            "DREAM > Posterior Filtering."
        )
        self.stoichiometry_status_label.setWordWrap(True)
        layout.addWidget(self.stoichiometry_status_label, 6, 0, 1, 3)

        self.fit_q_min_spin = QDoubleSpinBox()
        self.fit_q_min_spin.setDecimals(8)
        self.fit_q_min_spin.setKeyboardTracking(False)
        self.fit_q_min_spin.setEnabled(False)
        self.fit_q_min_spin.setToolTip(
            "Lower q bound for the active Prefit fit window."
        )
        self.fit_q_max_spin = QDoubleSpinBox()
        self.fit_q_max_spin.setDecimals(8)
        self.fit_q_max_spin.setKeyboardTracking(False)
        self.fit_q_max_spin.setEnabled(False)
        self.fit_q_max_spin.setToolTip(
            "Upper q bound for the active Prefit fit window."
        )
        self.fit_q_min_spin.valueChanged.connect(
            self._on_fit_range_spin_changed
        )
        self.fit_q_max_spin.valueChanged.connect(
            self._on_fit_range_spin_changed
        )
        self.fit_range_reset_button = QPushButton("Full Range")
        self.fit_range_reset_button.setEnabled(False)
        self.fit_range_reset_button.setToolTip(
            "Reset the active Prefit fit window to the full model trace."
        )
        self.fit_range_reset_button.clicked.connect(
            self._reset_fit_range_to_model_bounds
        )
        self.fit_range_status_label = QLabel("Fit range unavailable")
        self.fit_range_status_label.setWordWrap(True)
        fit_range_row = QWidget()
        self.fit_range_controls_row = fit_range_row
        fit_range_layout = QHBoxLayout(fit_range_row)
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(6)
        fit_range_layout.addWidget(QLabel("Fit q min"))
        fit_range_layout.addWidget(self.fit_q_min_spin)
        fit_range_layout.addWidget(QLabel("Fit q max"))
        fit_range_layout.addWidget(self.fit_q_max_spin)
        fit_range_layout.addWidget(self.fit_range_reset_button)
        fit_range_layout.addWidget(self.fit_range_status_label, stretch=1)
        layout.addWidget(fit_range_row, 7, 0, 1, 3)

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
        self.undo_fit_button = QPushButton("Undo Fit")
        self.undo_fit_button.setToolTip(
            "Restore the parameter table from before the most recent "
            "successful Prefit run."
        )
        self.undo_fit_button.setEnabled(False)
        self.undo_fit_button.clicked.connect(self.undo_fit_requested.emit)
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
        run_button_row.addWidget(self.undo_fit_button)
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
        layout.addLayout(button_grid, 8, 0, 1, 3)
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
        self.open_plot_editor_button = QPushButton("Open Plot Editor")
        self.open_plot_editor_button.clicked.connect(self.open_plot_editor)
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
        controls.addWidget(self.open_plot_editor_button)
        controls.addWidget(self.save_plot_data_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(9.6, 5.6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("pick_event", self._handle_legend_pick)
        self.canvas.mpl_connect(
            "button_press_event",
            self._handle_prefit_range_press,
        )
        self.canvas.mpl_connect(
            "button_release_event",
            self._handle_prefit_range_release,
        )
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
        self.dielectric_preset_row = QWidget()
        dielectric_preset_layout = QHBoxLayout(self.dielectric_preset_row)
        dielectric_preset_layout.setContentsMargins(0, 0, 0, 0)
        self.dielectric_preset_label = QLabel("Dielectric preset")
        self.dielectric_preset_combo = QComboBox()
        self.dielectric_preset_combo.addItem("Custom", userData=None)
        for preset in DIELECTRIC_CONSTANT_PRESETS:
            self.dielectric_preset_combo.addItem(
                preset.combo_label,
                userData=preset.key,
            )
        self.dielectric_preset_combo.setToolTip(
            "Apply a solvent relative dielectric constant to dielectconst. "
            "Choose Custom to keep the table value."
        )
        self.dielectric_preset_combo.currentIndexChanged.connect(
            self._on_dielectric_preset_changed
        )
        dielectric_preset_layout.addWidget(self.dielectric_preset_label)
        dielectric_preset_layout.addWidget(
            self.dielectric_preset_combo,
            stretch=1,
        )
        dielectric_preset_layout.addStretch(1)
        self.dielectric_preset_row.setVisible(False)
        layout.addWidget(self.dielectric_preset_row)
        self.charge_estimate_row = QWidget()
        charge_estimate_layout = QHBoxLayout(self.charge_estimate_row)
        charge_estimate_layout.setContentsMargins(0, 0, 0, 0)
        self.charge_estimate_label = QLabel("Charge estimate")
        self.charge_estimate_button = QPushButton("Estimate")
        self.charge_estimate_button.setToolTip(
            "Estimate the charged-sphere charge magnitude from the current "
            "component stoichiometries and weights."
        )
        self.charge_estimate_button.clicked.connect(
            self.charge_estimate_requested.emit
        )
        self.charge_estimate_status_label = QLabel("")
        self.charge_estimate_status_label.setWordWrap(True)
        charge_estimate_layout.addWidget(self.charge_estimate_label)
        charge_estimate_layout.addWidget(self.charge_estimate_button)
        charge_estimate_layout.addWidget(
            self.charge_estimate_status_label,
            stretch=1,
        )
        self.charge_estimate_row.setVisible(False)
        layout.addWidget(self.charge_estimate_row)
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
        self.parameter_table = QTableWidget(0, 9)
        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Structure",
                "Motif",
                "Param",
                "Value",
                "Vary",
                "Min",
                "Max",
                "Use",
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

    def _build_stoichiometry_compensator_group(self) -> QGroupBox:
        group = QGroupBox("Stoichiometry Compensator (Experimental)")
        layout = QVBoxLayout(group)

        header_row = QHBoxLayout()
        self.stoich_compensator_collapse_button = QToolButton()
        self.stoich_compensator_collapse_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.stoich_compensator_collapse_button.setAutoRaise(True)
        self.stoich_compensator_collapse_button.clicked.connect(
            self._toggle_stoichiometry_compensator_collapsed
        )
        header_row.addWidget(self.stoich_compensator_collapse_button)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        self.stoich_compensator_body = QWidget()
        body_layout = QVBoxLayout(self.stoich_compensator_body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        status = QLabel(
            "Selected compensator weights are recomputed by the experimental "
            "stoichiometry-compensator template before the model curve is "
            "evaluated."
        )
        status.setWordWrap(True)
        body_layout.addWidget(status)

        fields = QGridLayout()
        self.stoich_compensator_elements_edit = QLineEdit()
        self.stoich_compensator_elements_edit.setPlaceholderText("e.g. Pb, I")
        self.stoich_compensator_elements_edit.textChanged.connect(
            self._on_stoichiometry_compensator_settings_changed
        )
        self.stoich_compensator_ratio_edit = QLineEdit()
        self.stoich_compensator_ratio_edit.setPlaceholderText("e.g. 1:2")
        self.stoich_compensator_ratio_edit.textChanged.connect(
            self._on_stoichiometry_compensator_settings_changed
        )
        fields.addWidget(QLabel("Target elements"), 0, 0)
        fields.addWidget(self.stoich_compensator_elements_edit, 0, 1)
        fields.addWidget(QLabel("Target ratio"), 0, 2)
        fields.addWidget(self.stoich_compensator_ratio_edit, 0, 3)
        body_layout.addLayout(fields)

        self.stoich_compensator_table = QTableWidget(0, 4)
        self.stoich_compensator_table.setHorizontalHeaderLabels(
            ["Param", "Structure", "Motif", "Compensator"]
        )
        self.stoich_compensator_table.itemChanged.connect(
            self._on_stoichiometry_compensator_table_changed
        )
        header = self.stoich_compensator_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stoich_compensator_table.setMinimumHeight(140)
        body_layout.addWidget(self.stoich_compensator_table)

        layout.addWidget(self.stoich_compensator_body)
        self.set_stoichiometry_compensator_collapsed(True)
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
        *,
        active_name: str | None = None,
    ) -> None:
        current_name = selected_name or self.selected_template_name()
        self._suspend_template_selection_signal = True
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
        self.set_active_template(active_name or self._active_template_name)
        self._update_template_tooltip()
        self._suspend_template_selection_signal = False

    def selected_template_name(self) -> str | None:
        return str(self.template_combo.currentData() or "").strip() or None

    def active_template_name(self) -> str | None:
        return str(self._active_template_name or "").strip() or None

    def set_active_template(
        self,
        template_name: str | None,
        *,
        sync_selected: bool = False,
    ) -> None:
        self._active_template_name = str(template_name or "").strip() or None
        active_text = self._template_display_text(self._active_template_name)
        self.active_template_edit.setText(active_text)
        self.active_template_edit.setToolTip(active_text)
        if sync_selected and self._active_template_name:
            self.set_selected_template(self._active_template_name)
        self._update_template_change_state()

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

    def set_fit_undo_available(self, available: bool) -> None:
        self._fit_undo_available = bool(available)
        self._update_prefit_execution_control_state()

    def set_run_config(self, *, method: str, max_nfev: int) -> None:
        method_index = self.method_combo.findText(method)
        if method_index >= 0:
            self.method_combo.setCurrentIndex(method_index)
        self.nfev_spin.setValue(int(max_nfev))

    def set_saved_states(
        self,
        state_names: list[str] | list[tuple[str, str]],
        selected_name: str | None = None,
    ) -> None:
        current_name = selected_name or self.selected_saved_state_name()
        self.saved_state_combo.blockSignals(True)
        self.saved_state_combo.clear()
        for state_option in state_names:
            if isinstance(state_option, tuple):
                label, state_name = state_option
            else:
                label = state_name = state_option
            self.saved_state_combo.addItem(label, userData=state_name)
        if current_name:
            index = self.saved_state_combo.findData(current_name)
            if index < 0:
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
        data = self.saved_state_combo.currentData()
        if data is not None:
            text = str(data).strip()
            if text:
                return text
        text = self.saved_state_combo.currentText().strip()
        return text or None

    def populate_parameter_table(
        self,
        entries: list[PrefitParameterEntry],
    ) -> None:
        cached_entries = [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in entries
        ]
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            self.parameter_table.setColumnCount(9)
            self.parameter_table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_STRUCTURE,
                    QTableWidgetItem(entry.structure),
                )
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_MOTIF,
                    QTableWidgetItem(entry.motif),
                )
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_NAME,
                    QTableWidgetItem(entry.name),
                )
                self._set_parameter_active_widget(row, entry)
                vary_item = QTableWidgetItem()
                vary_item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.vary
                    else Qt.CheckState.Unchecked
                )
                self.parameter_table.setItem(
                    row, self.PARAM_COL_VARY, vary_item
                )
                self._set_parameter_value_item(
                    row,
                    value=float(entry.value),
                    value_expression=entry.value_expression,
                    initial_value_expression=entry.initial_value_expression,
                )
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_MIN,
                    QTableWidgetItem(f"{entry.minimum:.6g}"),
                )
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_MAX,
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
                self.parameter_table.setCellWidget(
                    row,
                    self.PARAM_COL_RESET,
                    reset_button,
                )
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        self.parameter_table.resizeRowsToContents()
        self._cached_parameter_entries = cached_entries
        self._parameter_entries_dirty = False
        self._refresh_dielectric_preset_controls()
        self._refresh_charge_estimate_controls()
        if not self._stoichiometry_compensator_group.isHidden():
            elements_text, ratio_text, selected_names = (
                self.stoichiometry_compensator_settings()
            )
            self.set_stoichiometry_compensator_settings(
                target_elements_text=elements_text,
                target_ratio_text=ratio_text,
                compensator_weight_names=selected_names,
                parameter_entries=cached_entries,
            )
        self._refresh_parameter_scroll_panel()

    def set_cluster_geometry_visible(self, visible: bool) -> None:
        self._cluster_geometry_group.setVisible(bool(visible))

    def set_stoichiometry_compensator_visible(self, visible: bool) -> None:
        was_visible = not self._stoichiometry_compensator_group.isHidden()
        self._stoichiometry_compensator_group.setVisible(bool(visible))
        if visible and not was_visible:
            self.set_stoichiometry_compensator_collapsed(True)

    def set_stoichiometry_compensator_settings(
        self,
        *,
        target_elements_text: str,
        target_ratio_text: str,
        compensator_weight_names: list[str] | tuple[str, ...],
        parameter_entries: list[PrefitParameterEntry],
    ) -> None:
        selected_names = {
            str(name).strip()
            for name in compensator_weight_names
            if str(name).strip()
        }
        weight_entries = [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in parameter_entries
            if self._is_component_weight_entry(entry)
            and bool(getattr(entry, "active", True))
        ]
        if not selected_names:
            selected_names = set(
                self._guessed_stoichiometry_compensator_names(
                    target_elements_text,
                    weight_entries,
                )
            )
        self._updating_stoichiometry_compensator = True
        self.stoich_compensator_table.blockSignals(True)
        self.stoich_compensator_elements_edit.blockSignals(True)
        self.stoich_compensator_ratio_edit.blockSignals(True)
        try:
            self.stoich_compensator_elements_edit.setText(
                str(target_elements_text or "")
            )
            self.stoich_compensator_ratio_edit.setText(
                str(target_ratio_text or "")
            )
            self._stoichiometry_compensator_entries = weight_entries
            self.stoich_compensator_table.setRowCount(len(weight_entries))
            for row, entry in enumerate(weight_entries):
                self.stoich_compensator_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(entry.name),
                )
                self.stoich_compensator_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(entry.structure),
                )
                self.stoich_compensator_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(entry.motif),
                )
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if entry.name in selected_names
                    else Qt.CheckState.Unchecked
                )
                self.stoich_compensator_table.setItem(row, 3, item)
        finally:
            self.stoich_compensator_ratio_edit.blockSignals(False)
            self.stoich_compensator_elements_edit.blockSignals(False)
            self.stoich_compensator_table.blockSignals(False)
            self._updating_stoichiometry_compensator = False
        self.stoich_compensator_table.resizeRowsToContents()

    def stoichiometry_compensator_settings(
        self,
    ) -> tuple[str, str, list[str]]:
        selected_names: list[str] = []
        for row in range(self.stoich_compensator_table.rowCount()):
            item = self.stoich_compensator_table.item(row, 3)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            name_item = self.stoich_compensator_table.item(row, 0)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if name:
                selected_names.append(name)
        return (
            self.stoich_compensator_elements_edit.text().strip(),
            self.stoich_compensator_ratio_edit.text().strip(),
            selected_names,
        )

    def stoichiometry_compensator_is_collapsed(self) -> bool:
        return self.stoich_compensator_body.isHidden()

    def set_stoichiometry_compensator_collapsed(
        self,
        collapsed: bool,
    ) -> None:
        is_collapsed = bool(collapsed)
        self.stoich_compensator_body.setVisible(not is_collapsed)
        self.stoich_compensator_collapse_button.setArrowType(
            Qt.ArrowType.RightArrow if is_collapsed else Qt.ArrowType.DownArrow
        )
        self.stoich_compensator_collapse_button.setText(
            "Show Compensator" if is_collapsed else "Hide Compensator"
        )

    def _toggle_stoichiometry_compensator_collapsed(self) -> None:
        self.set_stoichiometry_compensator_collapsed(
            not self.stoich_compensator_body.isHidden()
        )

    def set_solute_volume_fraction_visible(self, visible: bool) -> None:
        was_visible = not self._solute_volume_fraction_group.isHidden()
        self._solute_volume_fraction_group.setVisible(bool(visible))
        if visible and not was_visible:
            self.set_solute_volume_fraction_collapsed(True)

    def set_solute_volume_fraction_target(
        self,
        parameter_name: str | None,
        fraction_kind: str | None,
        fraction_source: str = "saxs_effective",
        solvent_weight_parameter: str | None = None,
        molar_concentration_parameter: str | None = None,
    ) -> None:
        target_messages: list[str] = []
        if parameter_name and fraction_kind:
            target_label = (
                "solute"
                if str(fraction_kind).strip() == "solute"
                else "solvent"
            )
            source_label = (
                "physical volume fraction"
                if str(fraction_source).strip() == "physical"
                else "SAXS-effective interaction fraction"
            )
            target_messages.append(
                f"{target_label} {source_label} -> {parameter_name}"
            )
        if solvent_weight_parameter:
            if (
                parameter_name
                and fraction_kind
                and str(fraction_source).strip() == "saxs_effective"
            ):
                target_messages.append(
                    f"attenuation solvent scale -> {solvent_weight_parameter}"
                )
            else:
                target_messages.append(
                    "combined solvent background multiplier -> "
                    f"{solvent_weight_parameter}"
                )
        if molar_concentration_parameter:
            target_messages.append(
                f"solute molar concentration -> {molar_concentration_parameter}"
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
            fraction_source,
            solvent_weight_parameter,
            molar_concentration_parameter,
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
        if (
            not self._parameter_entries_dirty
            and self._cached_parameter_entries is not None
        ):
            return [
                PrefitParameterEntry.from_dict(entry.to_dict())
                for entry in self._cached_parameter_entries
            ]
        entries: list[PrefitParameterEntry] = []
        for row in range(self.parameter_table.rowCount()):
            value_text = self._item_text(row, self.PARAM_COL_VALUE)
            value_expression: str | None = None
            initial_value_expression: str | None = None
            vary = (
                self.parameter_table.item(
                    row, self.PARAM_COL_VARY
                ).checkState()
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
                    self.parameter_table.item(row, self.PARAM_COL_VALUE)
                )
            name = self._item_text(row, self.PARAM_COL_NAME)
            category = (
                "weight" if self._is_weight_parameter_name(name) else "fit"
            )
            entries.append(
                PrefitParameterEntry(
                    structure=self._item_text(row, self.PARAM_COL_STRUCTURE),
                    motif=self._item_text(row, self.PARAM_COL_MOTIF),
                    name=name,
                    value=value,
                    vary=vary,
                    minimum=float(self._item_text(row, self.PARAM_COL_MIN)),
                    maximum=float(self._item_text(row, self.PARAM_COL_MAX)),
                    category=category,
                    value_expression=value_expression,
                    initial_value_expression=initial_value_expression,
                    active=self._parameter_row_active(row, category, name),
                )
            )
        resolved_entries = resolve_prefit_parameter_entries(entries)
        self._updating_parameter_table = True
        self.parameter_table.blockSignals(True)
        try:
            for row, entry in enumerate(resolved_entries):
                value_item = self.parameter_table.item(
                    row, self.PARAM_COL_VALUE
                )
                if value_item is not None:
                    value_item.setData(
                        self.PARAMETER_VALUE_ROLE,
                        float(entry.value),
                    )
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        self._cached_parameter_entries = [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in resolved_entries
        ]
        self._parameter_entries_dirty = False
        return resolved_entries

    def find_parameter_row(self, parameter_name: str) -> int:
        for row in range(self.parameter_table.rowCount()):
            if self._item_text(row, self.PARAM_COL_NAME) == parameter_name:
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
                and self._item_text(row, self.PARAM_COL_NAME) == parameter_name
            ):
                return row
        return -1

    def _parameter_row_active(
        self,
        row: int,
        category: str,
        name: str,
    ) -> bool:
        if str(
            category
        ).strip() != "weight" or not self._is_weight_parameter_name(name):
            return True
        widget = self.parameter_table.cellWidget(row, self.PARAM_COL_ACTIVE)
        if isinstance(widget, QPushButton):
            return bool(widget.isChecked())
        return True

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
                vary_item = self.parameter_table.item(row, self.PARAM_COL_VARY)
                if vary_item is not None:
                    vary_item.setCheckState(
                        Qt.CheckState.Checked
                        if vary
                        else Qt.CheckState.Unchecked
                    )
            if minimum is not None:
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_MIN,
                    QTableWidgetItem(f"{float(minimum):.6g}"),
                )
            if maximum is not None:
                self.parameter_table.setItem(
                    row,
                    self.PARAM_COL_MAX,
                    QTableWidgetItem(f"{float(maximum):.6g}"),
                )
        finally:
            self.parameter_table.blockSignals(False)
            self._updating_parameter_table = False
        self._invalidate_parameter_entries_cache()
        self._refresh_dielectric_preset_controls()
        self._refresh_charge_estimate_controls()
        self._refresh_parameter_scroll_panel()

    def auto_update_on_parameter_change(self) -> bool:
        return bool(self.auto_update_checkbox.isChecked())

    def scrollable_parameter_enabled(self) -> bool:
        return bool(self.scrollable_parameter_checkbox.isChecked())

    def _on_parameter_table_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if item.column() in {
            self.PARAM_COL_VALUE,
            self.PARAM_COL_VARY,
            self.PARAM_COL_MIN,
            self.PARAM_COL_MAX,
        }:
            self._invalidate_parameter_entries_cache()
        if item.column() in {self.PARAM_COL_VALUE, self.PARAM_COL_VARY}:
            self._sync_parameter_row_link_state(item.row())
        if not self._updating_parameter_table and item.column() in {
            self.PARAM_COL_VALUE,
            self.PARAM_COL_VARY,
            self.PARAM_COL_MIN,
            self.PARAM_COL_MAX,
        }:
            self.parameter_table_edited.emit()
        resolved_entries_valid = False
        if not self._updating_parameter_table and item.column() in {
            self.PARAM_COL_VALUE,
            self.PARAM_COL_VARY,
        }:
            try:
                self.parameter_entries()
                resolved_entries_valid = True
            except Exception:
                pass
        if (
            not self._updating_parameter_table
            and item.column() == self.PARAM_COL_VALUE
            and self._item_text(item.row(), self.PARAM_COL_NAME)
            == self.DIELECTRIC_PARAMETER_NAME
        ):
            self._refresh_dielectric_preset_controls()
        if (
            not self._updating_parameter_table
            and item.column()
            in {
                self.PARAM_COL_VALUE,
                self.PARAM_COL_VARY,
                self.PARAM_COL_MIN,
                self.PARAM_COL_MAX,
            }
            and self._item_text(item.row(), self.PARAM_COL_NAME)
            == self.CHARGE_PARAMETER_NAME
        ):
            self._refresh_charge_estimate_controls()
        self._refresh_parameter_scroll_panel()
        if (
            self._updating_parameter_table
            or item.column() != self.PARAM_COL_VALUE
        ):
            return
        if not self.auto_update_on_parameter_change():
            return
        if not resolved_entries_valid:
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

    def _on_dielectric_preset_changed(self, _index: int) -> None:
        if self._updating_parameter_table:
            return
        preset_key = self.dielectric_preset_combo.currentData()
        if not preset_key:
            return
        preset = DIELECTRIC_CONSTANT_PRESETS_BY_KEY.get(str(preset_key))
        if preset is None:
            return
        row = self.find_parameter_row(self.DIELECTRIC_PARAMETER_NAME)
        if row < 0:
            return
        self.set_parameter_row(
            self.DIELECTRIC_PARAMETER_NAME,
            value=preset.value,
        )
        self.parameter_table_edited.emit()
        if self.auto_update_on_parameter_change():
            self.update_model_requested.emit()

    def _refresh_dielectric_preset_controls(self) -> None:
        if not hasattr(self, "dielectric_preset_row"):
            return
        row = self.find_parameter_row(self.DIELECTRIC_PARAMETER_NAME)
        has_dielectric_parameter = row >= 0
        self.dielectric_preset_row.setVisible(has_dielectric_parameter)
        if not has_dielectric_parameter:
            self.dielectric_preset_combo.blockSignals(True)
            self.dielectric_preset_combo.setCurrentIndex(0)
            self.dielectric_preset_combo.blockSignals(False)
            return
        try:
            value = float(self._item_text(row, self.PARAM_COL_VALUE))
        except (TypeError, ValueError):
            value = None
        selected_key = None
        if value is not None:
            for preset in DIELECTRIC_CONSTANT_PRESETS:
                if abs(value - preset.value) <= 1.0e-6:
                    selected_key = preset.key
                    break
        selected_index = (
            self.dielectric_preset_combo.findData(selected_key)
            if selected_key
            else 0
        )
        if selected_index < 0:
            selected_index = 0
        self.dielectric_preset_combo.blockSignals(True)
        self.dielectric_preset_combo.setCurrentIndex(selected_index)
        self.dielectric_preset_combo.blockSignals(False)

    def set_charge_estimate_status_text(self, text: str) -> None:
        self.charge_estimate_status_label.setText(str(text or "").strip())

    def _refresh_charge_estimate_controls(self) -> None:
        if not hasattr(self, "charge_estimate_row"):
            return
        row = self.find_parameter_row(self.CHARGE_PARAMETER_NAME)
        has_charge_parameter = row >= 0
        self.charge_estimate_row.setVisible(has_charge_parameter)
        self.charge_estimate_button.setEnabled(has_charge_parameter)
        if not has_charge_parameter:
            self.charge_estimate_status_label.clear()

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
        parameter_name = self._item_text(row, self.PARAM_COL_NAME)
        value_item = self.parameter_table.item(row, self.PARAM_COL_VALUE)
        value_text = self._item_text(row, self.PARAM_COL_VALUE)
        uses_expression = self._parameter_value_uses_expression(value_text)
        try:
            value = (
                self._parameter_item_numeric_value(value_item)
                if uses_expression
                else float(value_text)
            )
            minimum = float(self._item_text(row, self.PARAM_COL_MIN))
            maximum = float(self._item_text(row, self.PARAM_COL_MAX))
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
            vary_item = self.parameter_table.item(row, self.PARAM_COL_VARY)
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
            minimum = float(self._item_text(row, self.PARAM_COL_MIN))
            maximum = float(self._item_text(row, self.PARAM_COL_MAX))
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
        item = self.parameter_table.item(row, self.PARAM_COL_VALUE)
        if item is None:
            item = QTableWidgetItem()
            self.parameter_table.setItem(row, self.PARAM_COL_VALUE, item)
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

    def fit_q_range(self) -> tuple[float | None, float | None]:
        if not self.fit_q_min_spin.isEnabled():
            return (None, None)
        return (
            float(self.fit_q_min_spin.value()),
            float(self.fit_q_max_spin.value()),
        )

    def set_fit_range_controls(
        self,
        *,
        model_q_min: float | None,
        model_q_max: float | None,
        fit_q_min: float | None,
        fit_q_max: float | None,
    ) -> None:
        if (
            model_q_min is None
            or model_q_max is None
            or not np.isfinite(float(model_q_min))
            or not np.isfinite(float(model_q_max))
            or float(model_q_max) < float(model_q_min)
        ):
            self._updating_fit_range_controls = True
            self.fit_q_min_spin.setEnabled(False)
            self.fit_q_max_spin.setEnabled(False)
            self.fit_range_reset_button.setEnabled(False)
            self.fit_range_status_label.setText("Fit range unavailable")
            self._updating_fit_range_controls = False
            return
        lower = float(model_q_min)
        upper = float(model_q_max)
        selected_lower = (
            lower if fit_q_min is None else max(lower, float(fit_q_min))
        )
        selected_upper = (
            upper if fit_q_max is None else min(upper, float(fit_q_max))
        )
        if selected_lower > selected_upper:
            selected_lower, selected_upper = lower, upper
        step = max((upper - lower) / 100.0, 1.0e-6)
        self._updating_fit_range_controls = True
        try:
            for spin in (self.fit_q_min_spin, self.fit_q_max_spin):
                spin.blockSignals(True)
                spin.setRange(lower, upper)
                spin.setSingleStep(step)
                spin.setEnabled(True)
            self.fit_q_min_spin.setValue(selected_lower)
            self.fit_q_max_spin.setValue(selected_upper)
            for spin in (self.fit_q_min_spin, self.fit_q_max_spin):
                spin.blockSignals(False)
            self.fit_range_reset_button.setEnabled(True)
            self.fit_range_status_label.setText(
                f"Active fit: {selected_lower:.6g} to {selected_upper:.6g} A^-1"
            )
        finally:
            self._updating_fit_range_controls = False

    def _set_fit_range_control_values(
        self,
        q_min: float,
        q_max: float,
        *,
        emit_signal: bool,
    ) -> None:
        if not self.fit_q_min_spin.isEnabled():
            return
        lower = float(self.fit_q_min_spin.minimum())
        upper = float(self.fit_q_max_spin.maximum())
        selected_lower = min(max(float(q_min), lower), upper)
        selected_upper = min(max(float(q_max), lower), upper)
        if selected_lower > selected_upper:
            selected_lower, selected_upper = selected_upper, selected_lower
        self._updating_fit_range_controls = True
        try:
            self.fit_q_min_spin.blockSignals(True)
            self.fit_q_max_spin.blockSignals(True)
            self.fit_q_min_spin.setValue(selected_lower)
            self.fit_q_max_spin.setValue(selected_upper)
            self.fit_q_min_spin.blockSignals(False)
            self.fit_q_max_spin.blockSignals(False)
            self.fit_range_status_label.setText(
                f"Active fit: {selected_lower:.6g} to {selected_upper:.6g} A^-1"
            )
        finally:
            self._updating_fit_range_controls = False
        if emit_signal:
            self.fit_range_changed.emit(selected_lower, selected_upper)

    def _on_fit_range_spin_changed(self, _value: float) -> None:
        if self._updating_fit_range_controls:
            return
        q_min = float(self.fit_q_min_spin.value())
        q_max = float(self.fit_q_max_spin.value())
        sender = self.sender()
        if q_min > q_max:
            if sender is self.fit_q_min_spin:
                q_max = q_min
            else:
                q_min = q_max
            self._set_fit_range_control_values(
                q_min,
                q_max,
                emit_signal=False,
            )
        self.fit_range_status_label.setText(
            f"Active fit: {q_min:.6g} to {q_max:.6g} A^-1"
        )
        self.fit_range_changed.emit(q_min, q_max)

    def _reset_fit_range_to_model_bounds(self) -> None:
        if not self.fit_q_min_spin.isEnabled():
            return
        self._set_fit_range_control_values(
            float(self.fit_q_min_spin.minimum()),
            float(self.fit_q_max_spin.maximum()),
            emit_signal=True,
        )

    def open_plot_editor(self) -> None:
        if self._plot_editor_window is not None:
            self._plot_editor_window.show()
            self._plot_editor_window.raise_()
            self._plot_editor_window.activateWindow()
            self._plot_editor_window.refresh_preview()
            return

        defaults = self._current_plot_defaults(self._current_evaluation)
        self._line_plot_settings.sync_series(defaults.series_defaults)
        self._plot_editor_controls = LinePlotEditorControls(
            settings=self._line_plot_settings,
            defaults=defaults,
            parent=self,
        )
        self._plot_editor_controls.label_settings_changed.connect(
            self._redraw_current_plot
        )
        self._plot_editor_controls.settings_changed.connect(
            self._redraw_current_plot
        )
        self._plot_editor_window = PlotEditorWindow(
            window_title="SAXS Prefit Plot Editor",
            controls_widget=self._plot_editor_controls,
            render_preview=self._render_plot_editor_preview,
            pickle_state_provider=self._plot_editor_pickle_state,
            apply_loaded_pickle_state=self._apply_loaded_plot_editor_pickle_state,
            parent=self,
        )
        self._plot_editor_window.closed.connect(self._on_plot_editor_closed)
        self._plot_editor_window.refresh_preview()
        self._plot_editor_window.show()
        self._plot_editor_window.raise_()
        self._plot_editor_window.activateWindow()

    def _on_plot_editor_closed(self) -> None:
        self._plot_editor_window = None
        self._plot_editor_controls = None

    def _current_plot_defaults(
        self,
        evaluation: PrefitEvaluation | None,
    ) -> LinePlotDefaults:
        has_evaluation = evaluation is not None
        has_experimental = (
            has_evaluation and evaluation.experimental_intensities is not None
        )
        has_residuals = has_evaluation and evaluation.residuals is not None
        structure_values = (
            np.asarray(evaluation.structure_factor_trace, dtype=float)
            if has_evaluation and evaluation.structure_factor_trace is not None
            else np.asarray([], dtype=float)
        )
        has_structure_factor_axis = bool(
            has_evaluation
            and self.show_structure_factor_trace_checkbox.isChecked()
            and np.any(np.isfinite(structure_values))
        )
        series_defaults: list[LinePlotSeriesDefaults] = []
        if (
            has_experimental
            and self.show_experimental_trace_checkbox.isChecked()
        ):
            series_defaults.append(
                LinePlotSeriesDefaults(
                    key="experimental",
                    label="Experimental",
                    axis_label="Main",
                )
            )
        if (
            has_evaluation
            and self.show_solvent_trace_checkbox.isChecked()
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
                series_defaults.append(
                    LinePlotSeriesDefaults(
                        key="solvent_contribution",
                        label="Solvent contribution",
                        axis_label="Main",
                    )
                )
        if has_structure_factor_axis:
            series_defaults.append(
                LinePlotSeriesDefaults(
                    key="structure_factor",
                    label="Structure factor S(q)",
                    axis_label="Structure Factor",
                )
            )
        if has_evaluation and self.show_model_trace_checkbox.isChecked():
            series_defaults.append(
                LinePlotSeriesDefaults(
                    key="model",
                    label="Model",
                    axis_label="Main",
                )
            )
        if has_experimental and has_residuals:
            series_defaults.append(
                LinePlotSeriesDefaults(
                    key="residual",
                    label="Residual",
                    axis_label="Residual",
                )
            )
        return LinePlotDefaults(
            title="",
            x_label=Q_A_INVERSE_LABEL,
            primary_y_label="Intensity (arb. units)",
            secondary_y_label="S(q)",
            residual_y_label="Residual",
            has_secondary_y_axis=has_structure_factor_axis,
            has_residual_y_axis=bool(has_experimental and has_residuals),
            has_annotation=has_evaluation,
            default_legend_location="best",
            default_show_annotation=True,
            series_defaults=tuple(series_defaults),
        )

    def _refresh_plot_editor_controls(self, *, force: bool = False) -> None:
        if self._plot_editor_controls is None:
            return
        defaults = self._current_plot_defaults(self._current_evaluation)
        self._line_plot_settings.sync_series(defaults.series_defaults)
        if force or self._plot_editor_controls.needs_default_sync(defaults):
            self._plot_editor_controls.sync_defaults(defaults)

    def _plot_editor_pickle_state(self) -> dict[str, object]:
        return {
            "plot_editor_state": {
                "kind": "line_plot_editor_state",
                "version": 1,
                "plot_context": "saxs_prefit_preview",
                "line_plot_settings": self._line_plot_settings.to_dict(),
                "panel_state": {
                    "show_experimental": bool(
                        self.show_experimental_trace_checkbox.isChecked()
                    ),
                    "show_model": bool(
                        self.show_model_trace_checkbox.isChecked()
                    ),
                    "show_solvent": bool(
                        self.show_solvent_trace_checkbox.isChecked()
                    ),
                    "show_structure_factor": bool(
                        self.show_structure_factor_trace_checkbox.isChecked()
                    ),
                    "log_x": bool(self.log_x_checkbox.isChecked()),
                    "log_y": bool(self.log_y_checkbox.isChecked()),
                },
            }
        }

    def _apply_loaded_plot_editor_pickle_state(
        self,
        payload: dict[str, object],
    ) -> bool:
        editor_state = payload.get("plot_editor_state")
        if not isinstance(editor_state, dict):
            return False
        if str(editor_state.get("kind")) != "line_plot_editor_state":
            return False
        if str(editor_state.get("plot_context")) != "saxs_prefit_preview":
            return False
        plot_settings = editor_state.get("line_plot_settings")
        if isinstance(plot_settings, dict):
            self._line_plot_settings.update_from_dict(plot_settings)
        panel_state = editor_state.get("panel_state")
        if isinstance(panel_state, dict):
            self.show_experimental_trace_checkbox.setChecked(
                bool(panel_state.get("show_experimental", True))
            )
            self.show_model_trace_checkbox.setChecked(
                bool(panel_state.get("show_model", True))
            )
            self.show_solvent_trace_checkbox.setChecked(
                bool(panel_state.get("show_solvent", False))
            )
            self.show_structure_factor_trace_checkbox.setChecked(
                bool(panel_state.get("show_structure_factor", False))
            )
            self.log_x_checkbox.setChecked(
                bool(panel_state.get("log_x", True))
            )
            self.log_y_checkbox.setChecked(
                bool(panel_state.get("log_y", True))
            )
        self._refresh_plot_editor_controls(force=True)
        self._redraw_current_plot()
        return True

    def _render_plot_editor_preview(self, figure: Figure) -> None:
        self._render_evaluation_figure(
            figure,
            self._current_evaluation,
            interactive=False,
        )

    def _prefit_range_drag_is_enabled(self) -> bool:
        if not self.fit_q_min_spin.isEnabled():
            return False
        mode_value = getattr(self.plot_toolbar, "mode", "")
        if hasattr(mode_value, "value"):
            mode_value = mode_value.value
        mode = str(mode_value or "")
        return not mode

    def _handle_prefit_range_press(self, event) -> None:
        if not self._prefit_range_drag_is_enabled():
            self._prefit_range_drag_start = None
            return
        if getattr(event, "button", None) != 1:
            self._prefit_range_drag_start = None
            return
        if getattr(event, "inaxes", None) not in self.figure.axes:
            self._prefit_range_drag_start = None
            return
        x_value = getattr(event, "xdata", None)
        if x_value is None or not np.isfinite(float(x_value)):
            self._prefit_range_drag_start = None
            return
        self._prefit_range_drag_start = float(x_value)

    def _handle_prefit_range_release(self, event) -> None:
        start = self._prefit_range_drag_start
        self._prefit_range_drag_start = None
        if start is None or not self._prefit_range_drag_is_enabled():
            return
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "inaxes", None) not in self.figure.axes:
            return
        end = getattr(event, "xdata", None)
        if end is None or not np.isfinite(float(end)):
            return
        q_min, q_max = sorted((float(start), float(end)))
        model_span = float(self.fit_q_max_spin.maximum()) - float(
            self.fit_q_min_spin.minimum()
        )
        if abs(q_max - q_min) <= max(model_span * 1.0e-4, 1.0e-8):
            return
        self._set_fit_range_control_values(q_min, q_max, emit_signal=True)

    def _render_evaluation_figure(
        self,
        figure: Figure,
        evaluation: PrefitEvaluation | None,
        *,
        interactive: bool,
    ) -> None:
        if interactive:
            self._legend_line_map.clear()
            self._legend_handle_lookup.clear()
        for axis in figure.axes:
            try:
                axis.set_xscale("linear")
                axis.set_yscale("linear")
            except Exception:
                continue
        figure.clear()
        self._update_prefit_trace_toggle_state(evaluation)
        self._update_plot_group_title()
        defaults = self._current_plot_defaults(evaluation)
        self._line_plot_settings.sync_series(defaults.series_defaults)
        if evaluation is None:
            axis = figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Build a project and load the prefit workflow to preview the model.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            if interactive:
                self._refresh_plot_editor_controls(force=True)
            figure.tight_layout()
            return

        q_values = np.asarray(evaluation.q_values, dtype=float)
        fit_q_min = (
            float(evaluation.fit_q_min)
            if evaluation.fit_q_min is not None
            else (float(np.min(q_values)) if q_values.size else None)
        )
        fit_q_max = (
            float(evaluation.fit_q_max)
            if evaluation.fit_q_max is not None
            else (float(np.max(q_values)) if q_values.size else None)
        )
        has_experimental = evaluation.experimental_intensities is not None
        has_residuals = evaluation.residuals is not None
        if has_experimental and has_residuals:
            grid = figure.add_gridspec(2, 1, height_ratios=[3, 1])
            top = figure.add_subplot(grid[0, 0])
            bottom = figure.add_subplot(grid[1, 0], sharex=top)
        else:
            top = figure.add_subplot(111)
            bottom = None

        if fit_q_min is not None and fit_q_max is not None:
            top.axvspan(
                fit_q_min,
                fit_q_max,
                color="tab:blue",
                alpha=0.08,
                linewidth=0.0,
                zorder=0,
            )

        plotted_lines: list[object] = []
        structure_axis = None
        font_family = self._line_plot_settings.font_family.strip()

        if (
            has_experimental
            and self.show_experimental_trace_checkbox.isChecked()
        ):
            (experimental_line,) = top.plot(
                evaluation.q_values,
                evaluation.experimental_intensities,
                color="black",
                label=self._line_plot_settings.display_series_label(
                    "experimental",
                    "Experimental",
                ),
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
                    label=self._line_plot_settings.display_series_label(
                        "solvent_contribution",
                        "Solvent contribution",
                    ),
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
                apply_axis_scales(
                    structure_axis,
                    log_x=self.log_x_checkbox.isChecked(),
                    log_y=False,
                )
                (structure_line,) = structure_axis.plot(
                    np.asarray(evaluation.q_values, dtype=float)[
                        structure_mask
                    ],
                    structure_values[structure_mask],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.5,
                    label=self._line_plot_settings.display_series_label(
                        "structure_factor",
                        "Structure factor S(q)",
                    ),
                )
                structure_axis.set_ylabel(
                    self._line_plot_settings.resolve_secondary_y_label(
                        defaults
                    ),
                    color="tab:purple",
                )
                structure_axis.tick_params(axis="y", colors="tab:purple")
                structure_axis.spines["right"].set_color("tab:purple")
                plotted_lines.append(structure_line)

        if self.show_model_trace_checkbox.isChecked():
            (model_line,) = top.plot(
                evaluation.q_values,
                evaluation.model_intensities,
                color="tab:red",
                label=self._line_plot_settings.display_series_label(
                    "model",
                    "Model",
                ),
            )
            plotted_lines.append(model_line)

        apply_axis_scales(
            top,
            log_x=self.log_x_checkbox.isChecked(),
            log_y=self.log_y_checkbox.isChecked(),
        )
        top.set_ylabel(
            self._line_plot_settings.resolve_primary_y_label(defaults)
        )
        if self._line_plot_settings.resolve_show_annotation(defaults):
            annotation_kwargs: dict[str, object] = {
                "transform": top.transAxes,
                "ha": "left",
                "va": "bottom",
                "fontsize": self._line_plot_settings.annotation_font_size,
                "bbox": {
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.6",
                    "alpha": 0.85,
                },
            }
            if font_family:
                annotation_kwargs["fontfamily"] = font_family
            top.text(
                0.02,
                0.02,
                "\n".join(self._prefit_metric_lines(evaluation)),
                **annotation_kwargs,
            )

        title = self._line_plot_settings.resolve_title(defaults)
        if title:
            title_kwargs: dict[str, object] = {
                "x": self._line_plot_settings.resolve_title_position_x(
                    defaults
                ),
                "y": self._line_plot_settings.resolve_title_position_y(
                    defaults
                ),
                "fontsize": self._line_plot_settings.title_font_size,
            }
            if font_family:
                title_kwargs["fontfamily"] = font_family
            top.set_title(title, **title_kwargs)
        else:
            top.set_title("")

        if plotted_lines and self._line_plot_settings.resolve_show_legend(
            defaults
        ):
            legend_location = self._line_plot_settings.resolve_legend_location(
                defaults
            )
            legend_font_size = self._line_plot_settings.legend_font_size
            if interactive:
                self._build_interactive_legend(
                    top,
                    plotted_lines,
                    location=legend_location,
                    font_size=legend_font_size,
                    font_family=font_family,
                )
            else:
                preview_legend = top.legend(
                    handles=plotted_lines,
                    loc=legend_location,
                    fontsize=legend_font_size,
                )
                if preview_legend is not None and font_family:
                    for text in preview_legend.get_texts():
                        text.set_fontfamily(font_family)

        if bottom is not None and evaluation.residuals is not None:
            bottom.axhline(0.0, color="0.5", linewidth=1.0)
            if fit_q_min is not None and fit_q_max is not None:
                bottom.axvspan(
                    fit_q_min,
                    fit_q_max,
                    color="tab:blue",
                    alpha=0.08,
                    linewidth=0.0,
                    zorder=0,
                )
            bottom.plot(
                evaluation.q_values,
                evaluation.residuals,
                color="tab:blue",
                label=self._line_plot_settings.display_series_label(
                    "residual",
                    "Residual",
                ),
            )
            apply_axis_scales(
                bottom,
                log_x=self.log_x_checkbox.isChecked(),
                log_y=False,
            )
            bottom.set_xlabel(
                self._line_plot_settings.resolve_x_label(defaults)
            )
            bottom.set_ylabel(
                self._line_plot_settings.resolve_residual_y_label(defaults)
            )
        else:
            top.set_xlabel(self._line_plot_settings.resolve_x_label(defaults))

        x_axis_label_font_size = self._line_plot_settings.axis_label_font_size
        x_tick_label_font_size = self._line_plot_settings.tick_label_font_size
        primary_axis_label_font_size = (
            self._line_plot_settings.resolve_primary_axis_label_font_size(
                defaults
            )
        )
        primary_tick_label_font_size = (
            self._line_plot_settings.resolve_primary_tick_label_font_size(
                defaults
            )
        )
        secondary_axis_label_font_size = (
            self._line_plot_settings.resolve_secondary_axis_label_font_size(
                defaults
            )
        )
        secondary_tick_label_font_size = (
            self._line_plot_settings.resolve_secondary_tick_label_font_size(
                defaults
            )
        )

        for axis in figure.axes:
            axis.xaxis.label.set_fontsize(x_axis_label_font_size)
            axis.tick_params(
                axis="x",
                which="both",
                labelsize=x_tick_label_font_size,
            )
            if font_family:
                axis.xaxis.label.set_fontfamily(font_family)
            for label in list(axis.get_xticklabels()) + list(
                axis.get_xticklabels(minor=True)
            ):
                if font_family:
                    label.set_fontfamily(font_family)

        for axis in (top, bottom):
            if axis is None:
                continue
            axis.yaxis.label.set_fontsize(primary_axis_label_font_size)
            axis.tick_params(
                axis="y",
                which="both",
                labelsize=primary_tick_label_font_size,
            )
            axis.yaxis.get_offset_text().set_fontsize(
                primary_tick_label_font_size
            )
            if font_family:
                axis.yaxis.label.set_fontfamily(font_family)
                axis.yaxis.get_offset_text().set_fontfamily(font_family)
            for label in list(axis.get_yticklabels()) + list(
                axis.get_yticklabels(minor=True)
            ):
                if font_family:
                    label.set_fontfamily(font_family)

        if structure_axis is not None:
            structure_axis.yaxis.label.set_fontsize(
                secondary_axis_label_font_size
            )
            structure_axis.tick_params(
                axis="y",
                which="both",
                labelsize=secondary_tick_label_font_size,
            )
            structure_axis.yaxis.get_offset_text().set_fontsize(
                secondary_tick_label_font_size
            )
            if font_family:
                structure_axis.yaxis.label.set_fontfamily(font_family)
                structure_axis.yaxis.get_offset_text().set_fontfamily(
                    font_family
                )
            for label in list(structure_axis.get_yticklabels()) + list(
                structure_axis.get_yticklabels(minor=True)
            ):
                if font_family:
                    label.set_fontfamily(font_family)

        if interactive:
            self._refresh_plot_editor_controls()
        figure.tight_layout()

    def plot_evaluation(
        self,
        evaluation: PrefitEvaluation | None,
    ) -> None:
        self._current_evaluation = evaluation
        if evaluation is None:
            self.set_fit_range_controls(
                model_q_min=None,
                model_q_max=None,
                fit_q_min=None,
                fit_q_max=None,
            )
        else:
            q_values = np.asarray(evaluation.q_values, dtype=float)
            self.set_fit_range_controls(
                model_q_min=(
                    float(np.min(q_values)) if q_values.size else None
                ),
                model_q_max=(
                    float(np.max(q_values)) if q_values.size else None
                ),
                fit_q_min=evaluation.fit_q_min,
                fit_q_max=evaluation.fit_q_max,
            )
        self._render_evaluation_figure(
            self.figure,
            evaluation,
            interactive=True,
        )
        self.canvas.draw()
        if self._plot_editor_window is not None:
            self._plot_editor_window.refresh_preview()

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
        residuals = np.asarray(evaluation.residuals, dtype=float)
        fit_mask = np.ones(residuals.shape, dtype=bool)
        if evaluation.fit_mask is not None:
            candidate_mask = np.asarray(evaluation.fit_mask, dtype=bool)
            if candidate_mask.shape == residuals.shape:
                fit_mask &= candidate_mask
        fit_mask &= (
            np.isfinite(residuals)
            & np.isfinite(experimental_values)
            & np.isfinite(model_values)
        )
        if not np.any(fit_mask):
            return [
                "Fit metrics unavailable",
                "No finite points in active fit range",
            ]
        residuals = residuals[fit_mask]
        experimental_values = experimental_values[fit_mask]
        model_values = model_values[fit_mask]
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
        if evaluation.fitted_stoichiometry_text:
            metric_lines.append(evaluation.fitted_stoichiometry_text)
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

    def set_auto_snap_enabled(self, enabled: bool) -> None:
        self._auto_snap_enabled = bool(enabled)
        self._auto_snap_filter.set_enabled(self._auto_snap_enabled)

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
        self.parameter_table.setItem(row, self.PARAM_COL_VALUE, value_item)
        self._sync_parameter_row_link_state(row)

    def _set_parameter_active_widget(
        self,
        row: int,
        entry: PrefitParameterEntry,
    ) -> None:
        if not self._is_component_weight_entry(entry):
            item = QTableWidgetItem("")
            item.setFlags(
                Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            )
            item.setToolTip(
                "Only component weights w<NN> can be turned on or off."
            )
            self.parameter_table.setItem(row, self.PARAM_COL_ACTIVE, item)
            return
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(bool(getattr(entry, "active", True)))
        self._update_parameter_active_button(button)
        button.clicked.connect(
            lambda checked=False, row_index=row, control=button: (
                self._on_parameter_active_toggled(row_index, control)
            )
        )
        self.parameter_table.setCellWidget(row, self.PARAM_COL_ACTIVE, button)

    def _on_parameter_active_toggled(
        self,
        row: int,
        button: QPushButton,
    ) -> None:
        self._update_parameter_active_button(button)
        self._invalidate_parameter_entries_cache()
        self._refresh_parameter_scroll_panel()
        if not self._updating_parameter_table:
            self.parameter_table_edited.emit()
            if self.auto_update_on_parameter_change():
                self.update_model_requested.emit()

    @staticmethod
    def _update_parameter_active_button(button: QPushButton) -> None:
        enabled = bool(button.isChecked())
        button.setText("On" if enabled else "Off")
        button.setToolTip(
            "This component weight is included in Prefit and DREAM."
            if enabled
            else "This component weight is excluded from Prefit and DREAM."
        )

    @staticmethod
    def _is_component_weight_entry(entry: PrefitParameterEntry) -> bool:
        return str(
            entry.category
        ).strip() == "weight" and PrefitTab._is_weight_parameter_name(
            entry.name
        )

    @staticmethod
    def _is_weight_parameter_name(name: str) -> bool:
        text = str(name or "").strip()
        return text.startswith("w") and text[1:].isdigit()

    def _sync_parameter_row_link_state(self, row: int) -> None:
        if row < 0:
            return
        vary_item = self.parameter_table.item(row, self.PARAM_COL_VARY)
        value_item = self.parameter_table.item(row, self.PARAM_COL_VALUE)
        if vary_item is None or value_item is None:
            return
        linked = self._parameter_value_uses_expression(
            self._item_text(row, self.PARAM_COL_VALUE)
        )
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

    def _invalidate_parameter_entries_cache(self) -> None:
        self._parameter_entries_dirty = True
        self._cached_parameter_entries = None

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
        self.undo_fit_button.setEnabled(enabled and self._fit_undo_available)
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

    def _on_stoichiometry_compensator_settings_changed(self) -> None:
        if self._updating_stoichiometry_compensator:
            return
        self.stoichiometry_compensator_settings_changed.emit()

    def _on_stoichiometry_compensator_table_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if self._updating_stoichiometry_compensator or item.column() != 3:
            return
        self.stoichiometry_compensator_settings_changed.emit()

    def _guess_stoichiometry_compensators(self) -> None:
        guesses = set(
            self._guessed_stoichiometry_compensator_names(
                self.stoich_compensator_elements_edit.text().strip()
            )
        )
        self._updating_stoichiometry_compensator = True
        self.stoich_compensator_table.blockSignals(True)
        try:
            for row in range(self.stoich_compensator_table.rowCount()):
                name_item = self.stoich_compensator_table.item(row, 0)
                item = self.stoich_compensator_table.item(row, 3)
                if name_item is None or item is None:
                    continue
                item.setCheckState(
                    Qt.CheckState.Checked
                    if name_item.text().strip() in guesses
                    else Qt.CheckState.Unchecked
                )
        finally:
            self.stoich_compensator_table.blockSignals(False)
            self._updating_stoichiometry_compensator = False
        self.stoichiometry_compensator_settings_changed.emit()

    def _guessed_stoichiometry_compensator_names(
        self,
        elements_text: str,
        entries: list[PrefitParameterEntry] | None = None,
    ) -> tuple[str, ...]:
        elements = [
            token.strip()
            for token in str(elements_text or "")
            .replace(":", ",")
            .replace(";", ",")
            .split(",")
            if token.strip()
        ]
        source_entries = (
            self._stoichiometry_compensator_entries
            if entries is None
            else entries
        )
        return guess_single_atom_compensator_names(
            tuple((entry.name, entry.structure) for entry in source_entries),
            tuple(elements),
        )

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

    def _build_interactive_legend(
        self,
        axis,
        lines: list[object],
        *,
        location: str = "best",
        font_size: float = 9.0,
        font_family: str = "",
    ) -> None:
        legend = axis.legend(handles=lines, loc=location, fontsize=font_size)
        if legend is None:
            return
        if font_family:
            for text in legend.get_texts():
                text.set_fontfamily(font_family)
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
        self._suspend_template_selection_signal = True
        self.template_combo.blockSignals(True)
        self.template_combo.setCurrentIndex(index)
        self.template_combo.blockSignals(False)
        self._suspend_template_selection_signal = False
        self._update_template_tooltip()
        self._update_template_change_state()

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
        self._update_template_change_state()
        selected_name = self.selected_template_name()
        if selected_name and not self._suspend_template_selection_signal:
            self.template_changed.emit(selected_name)

    def _update_template_tooltip(self) -> None:
        description = str(
            self.template_combo.currentData(Qt.ItemDataRole.ToolTipRole) or ""
        ).strip()
        self.template_combo.setToolTip(description)

    def _template_display_text(self, template_name: str | None) -> str:
        normalized_name = str(template_name or "").strip()
        if not normalized_name:
            return ""
        index = self._find_template_index(normalized_name)
        if index >= 0:
            return self.template_combo.itemText(index).strip()
        return normalized_name

    def _update_template_change_state(self) -> None:
        selected_name = self.selected_template_name()
        active_name = self.active_template_name()
        self.change_template_button.setEnabled(
            bool(
                active_name and selected_name and selected_name != active_name
            )
        )

    def _emit_change_template_requested(self) -> None:
        selected_name = self.selected_template_name()
        if not selected_name:
            return
        self.change_template_requested.emit(selected_name)

    def _find_template_index(self, template_name: str) -> int:
        for index in range(self.template_combo.count()):
            if (
                str(self.template_combo.itemData(index) or "").strip()
                == template_name
            ):
                return index
        return -1
