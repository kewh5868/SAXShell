from __future__ import annotations

import re
from pathlib import Path
from textwrap import fill

import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from saxshell.saxs.debye import discover_cluster_bins
from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    ProjectSettings,
    build_project_paths,
    load_experimental_data_file,
    plot_md_prior_histogram,
)
from saxshell.saxs.stoichiometry import parse_stoich_label
from saxshell.saxs.ui.experimental_data_loader import (
    ExperimentalDataHeaderDialog,
)
from saxshell.saxs.ui.template_help import (
    TEMPLATE_HELP_TEXT,
    show_template_help,
)

HISTOGRAM_COLORMAP_NAMES = [
    "summer",
    "viridis",
    "plasma",
    "cividis",
    "Greens",
    "Blues",
    "magma",
]


class ProjectSetupTab(QWidget):
    create_project_requested = Signal()
    open_project_requested = Signal()
    save_project_requested = Signal()
    autosave_project_requested = Signal(str)
    scan_clusters_requested = Signal()
    build_components_requested = Signal()
    build_prior_weights_requested = Signal()
    install_model_requested = Signal()
    generate_prior_plot_requested = Signal()
    save_prior_png_requested = Signal()
    save_component_plot_data_requested = Signal()
    save_prior_plot_data_requested = Signal()
    show_deprecated_templates_changed = Signal(bool)
    model_only_mode_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._console_autoscroll_enabled = True
        self._summary_text = ""
        self._experimental_header_rows = 0
        self._experimental_q_column: int | None = None
        self._experimental_intensity_column: int | None = None
        self._experimental_error_column: int | None = None
        self._experimental_summary: ExperimentalDataSummary | None = None
        self._solvent_header_rows = 0
        self._solvent_q_column: int | None = None
        self._solvent_intensity_column: int | None = None
        self._solvent_error_column: int | None = None
        self._solvent_summary: ExperimentalDataSummary | None = None
        self._component_paths: list[Path] | None = None
        self._current_prior_json_path: Path | None = None
        self._recognized_cluster_rows: list[dict[str, object]] = []
        self._legend_line_map: dict[object, object] = {}
        self._component_legend_lookup: dict[str, object] = {}
        self._component_line_lookup: dict[str, object] = {}
        self._component_color_lookup: dict[str, str] = {}
        self._component_color_overrides: dict[str, str] = {}
        self._component_visibility: dict[str, bool] = {}
        self._component_row_lookup: dict[str, int] = {}
        self._updating_cluster_table = False
        self._build_ui()
        self._update_data_trace_control_state()
        self._update_component_trace_control_state()
        self.set_project_selected(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        self.project_group = self._build_project_group()
        self.forward_model_group = self._build_inputs_group()
        self.model_group = self._build_model_group()
        self.activity_group = self._build_activity_group()
        left_layout.addWidget(self.project_group)
        left_layout.addWidget(self.forward_model_group)
        left_layout.addWidget(self.model_group)
        left_layout.addWidget(self.activity_group, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        self.component_group = self._build_component_group()
        self.prior_group = self._build_prior_group()
        right_layout.addWidget(self.component_group)
        right_layout.addWidget(self.prior_group)
        right_layout.addStretch(1)

        self._left_scroll_area = self._wrap_pane_in_scroll_area(left)
        self._right_scroll_area = self._wrap_pane_in_scroll_area(right)
        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._pane_splitter.setChildrenCollapsible(False)
        self._pane_splitter.setHandleWidth(10)
        self._pane_splitter.addWidget(self._left_scroll_area)
        self._pane_splitter.addWidget(self._right_scroll_area)
        self._pane_splitter.setStretchFactor(0, 6)
        self._pane_splitter.setStretchFactor(1, 7)
        self._pane_splitter.setSizes([720, 840])
        root.addWidget(self._pane_splitter)

    @staticmethod
    def _wrap_pane_in_scroll_area(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll_area.setWidget(widget)
        return scroll_area

    def _build_project_group(self) -> QGroupBox:
        group = QGroupBox("Project")
        layout = QVBoxLayout(group)

        create_group = QGroupBox("Create New Project")
        create_layout = QFormLayout(create_group)
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Example: PbI2-DMF SAXS fit")
        create_layout.addRow("Project folder name", self.project_name_edit)

        self.project_dir_edit = QLineEdit()
        create_layout.addRow(
            "Project directory",
            self._path_action_row(
                self.project_dir_edit,
                dialog_title="Select parent directory for the new SAXS project",
                action_button_text="Create Project",
                action=self.create_project_requested.emit,
            ),
        )
        layout.addWidget(create_group)

        open_group = QGroupBox("Open Existing Project")
        open_layout = QFormLayout(open_group)
        self.open_project_dir_edit = QLineEdit()
        open_layout.addRow(
            "Existing project folder",
            self._path_action_row(
                self.open_project_dir_edit,
                dialog_title="Select an existing SAXS project folder",
                action_button_text="Open Project",
                action=self.open_project_requested.emit,
                browse_handler=self._browse_existing_project_directory,
            ),
        )
        layout.addWidget(open_group)

        helper = QLabel(
            "Create a new project by choosing the parent directory and the "
            "project folder name, or open an existing complete project folder. "
            "Forward-model inputs stay locked until a project is selected."
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        self.save_project_button = QPushButton("Save Project State")
        self.save_project_button.clicked.connect(
            self.save_project_requested.emit
        )
        save_row.addWidget(self.save_project_button)
        layout.addLayout(save_row)
        return group

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Forward Model Inputs")
        layout = QFormLayout(group)

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.editingFinished.connect(
            self._on_clusters_dir_edited
        )
        layout.addRow("Clusters folder", self._clusters_row())

        self.model_only_mode_checkbox = QCheckBox("Model Only Mode")
        self.model_only_mode_checkbox.setChecked(False)
        self.model_only_mode_checkbox.setToolTip(
            "Disable experimental-data-dependent fitting and use the project "
            "as a forward-model-only SAXS simulator."
        )
        self.model_only_mode_checkbox.toggled.connect(
            self._on_model_only_mode_toggled
        )
        layout.addRow("", self.model_only_mode_checkbox)

        self.experimental_data_edit = QLineEdit()
        self.experimental_data_edit.setReadOnly(True)
        layout.addRow("Experimental data", self._experimental_data_row())
        self.solvent_data_edit = QLineEdit()
        self.solvent_data_edit.setReadOnly(True)
        layout.addRow("Solvent data", self._solvent_data_row())

        self.data_status_label = QLabel(
            "Choose an experimental SAXS file or folder after opening a "
            "project.\n"
            "The selected file, columns, q-range, and import settings will "
            "be summarized here."
        )
        self.data_status_label.setWordWrap(True)
        self.data_status_label.setMinimumHeight(72)
        self.data_status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        layout.addRow("", self.data_status_label)
        self.solvent_status_label = QLabel(
            "Optional solvent SAXS data can be loaded here and will be "
            "carried into prefit and DREAM if the active model uses "
            "solvent intensities."
        )
        self.solvent_status_label.setWordWrap(True)
        self.solvent_status_label.setMinimumHeight(52)
        self.solvent_status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        layout.addRow("", self.solvent_status_label)

        self.qmin_edit = QLineEdit()
        self.qmax_edit = QLineEdit()
        self.qmin_edit.textChanged.connect(self._redraw_saxs_preview)
        self.qmax_edit.textChanged.connect(self._redraw_saxs_preview)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("q min"))
        q_row.addWidget(self.qmin_edit)
        q_row.addWidget(QLabel("q max"))
        q_row.addWidget(self.qmax_edit)
        layout.addRow("q-range", q_row)

        self.use_experimental_grid_checkbox = QCheckBox(
            "Use experimental grid"
        )
        self.use_experimental_grid_checkbox.setChecked(True)
        self.use_experimental_grid_checkbox.setToolTip(
            "Use the experimental q-grid directly inside the selected q-range. "
            "Disable this option to resample the experimental data onto a "
            "custom evenly spaced grid for the forward model."
        )
        self.use_experimental_grid_checkbox.toggled.connect(
            self._update_resample_grid_state
        )
        self.resample_points_spin = QSpinBox()
        self.resample_points_spin.setRange(2, 50000)
        self.resample_points_spin.setValue(500)
        self.resample_points_spin.setToolTip(
            "Number of evenly spaced q-points to generate between q min and "
            "q max when resampling the model and experimental data."
        )
        grid_row = QHBoxLayout()
        grid_row.addWidget(self.use_experimental_grid_checkbox)
        grid_row.addStretch(1)
        grid_row.addWidget(QLabel("Resample grid"))
        grid_row.addWidget(self.resample_points_spin)
        layout.addRow("Grid", grid_row)
        self._update_resample_grid_state()

        self.available_elements_list = QListWidget()
        self.available_elements_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.available_elements_list.setMinimumHeight(104)
        self.available_elements_list.setMaximumHeight(132)
        self.available_elements_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        layout.addRow("Recognized elements", self._elements_row())

        self.exclude_elements_edit = QLineEdit()
        self.exclude_elements_edit.setPlaceholderText("Example: H O")
        self.exclude_elements_edit.textChanged.connect(
            self._sync_available_element_selection
        )
        layout.addRow("Exclude elements", self.exclude_elements_edit)
        return group

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model and Build")
        layout = QVBoxLayout(group)

        header_widget = QWidget()
        self._model_build_header_widget = header_widget
        header_layout = QFormLayout(header_widget)
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(420)
        self.template_combo.currentIndexChanged.connect(
            self._on_template_combo_changed
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
        header_layout.addRow(
            "Selected template",
            self._template_row(),
        )

        self.active_template_edit = QLineEdit()
        self.active_template_edit.setReadOnly(True)
        self.active_template_edit.setMinimumWidth(420)
        header_layout.addRow("Active template", self.active_template_edit)
        layout.addWidget(header_widget)

        lower_layout = QHBoxLayout()
        self._model_build_lower_layout = lower_layout
        button_widget = QWidget()
        self._model_build_button_widget = button_widget
        button_layout = QVBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        self.build_components_button = QPushButton("Build SAXS Components")
        self.build_components_button.clicked.connect(
            self.build_components_requested.emit
        )
        self.build_prior_weights_button = QPushButton("Generate Prior Weights")
        self.build_prior_weights_button.clicked.connect(
            self.build_prior_weights_requested.emit
        )
        self.install_model_button = QPushButton("Install Model")
        self.install_model_button.clicked.connect(
            self.install_model_requested.emit
        )
        button_layout.addWidget(self.build_prior_weights_button)
        button_layout.addWidget(self.build_components_button)
        button_layout.addWidget(self.install_model_button)
        button_layout.addStretch(1)
        button_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        button_widget.setMinimumWidth(220)
        lower_layout.addWidget(button_widget, stretch=0)

        clusters_group = QGroupBox("Recognized Clusters")
        self._recognized_clusters_group = clusters_group
        clusters_layout = QVBoxLayout(clusters_group)
        self.recognized_clusters_table = QTableWidget(0, 8)
        self.recognized_clusters_table.setHorizontalHeaderLabels(
            [
                "Stoichiometry",
                "Cluster Type",
                "Count",
                "Weight",
                "Atom %",
                "Structure %",
                "Visible",
                "Color",
            ]
        )
        self.recognized_clusters_table.setMinimumHeight(220)
        self.recognized_clusters_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.recognized_clusters_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.recognized_clusters_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.recognized_clusters_table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.recognized_clusters_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.recognized_clusters_table.setWordWrap(False)
        self.recognized_clusters_table.itemChanged.connect(
            self._on_recognized_cluster_item_changed
        )
        self.recognized_clusters_table.cellClicked.connect(
            self._on_recognized_cluster_cell_clicked
        )
        header = self.recognized_clusters_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        header.resizeSection(0, 150)
        header.resizeSection(1, 130)
        header.resizeSection(2, 78)
        header.resizeSection(3, 78)
        header.resizeSection(4, 84)
        header.resizeSection(5, 102)
        header.resizeSection(6, 72)
        header.resizeSection(7, 92)
        clusters_layout.addWidget(self.recognized_clusters_table)
        lower_layout.addWidget(clusters_group, stretch=1)
        layout.addLayout(lower_layout, stretch=1)
        return group

    def _template_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.template_combo, stretch=1)
        layout.addWidget(self.template_help_button)
        layout.addWidget(self.show_deprecated_templates_checkbox)
        return row

    def _build_activity_group(self) -> QGroupBox:
        group = QGroupBox("Project Activity")
        layout = QVBoxLayout(group)
        self.activity_progress_label = QLabel("Progress: idle")
        layout.addWidget(self.activity_progress_label)
        self.activity_progress_bar = QProgressBar()
        self.activity_progress_bar.setRange(0, 1)
        self.activity_progress_bar.setValue(0)
        self.activity_progress_bar.setFormat("%v / %m items")
        layout.addWidget(self.activity_progress_bar)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(220)
        self.summary_box.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.summary_box)
        return group

    def _build_component_group(self) -> QGroupBox:
        group = QGroupBox("Experimental Data and SAXS Components")
        group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.component_log_x_checkbox = QCheckBox("Log X")
        self.component_log_x_checkbox.setChecked(True)
        self.component_log_x_checkbox.toggled.connect(
            self._redraw_saxs_preview
        )
        self.component_log_y_checkbox = QCheckBox("Log Y")
        self.component_log_y_checkbox.setChecked(True)
        self.component_log_y_checkbox.toggled.connect(
            self._redraw_saxs_preview
        )
        self.component_legend_toggle_button = QPushButton("Legend")
        self.component_legend_toggle_button.setCheckable(True)
        self.component_legend_toggle_button.setChecked(True)
        self.component_legend_toggle_button.toggled.connect(
            self._redraw_saxs_preview
        )
        self.component_model_range_button = QPushButton(
            "Autoscale to Model Range"
        )
        self.component_model_range_button.setCheckable(True)
        self.component_model_range_button.toggled.connect(
            self._redraw_saxs_preview
        )
        self.component_all_traces_button = QPushButton("Hide Computed Traces")
        self.component_all_traces_button.clicked.connect(
            self._toggle_all_component_traces
        )
        self.component_trace_color_scheme_combo = QComboBox()
        self.component_trace_color_scheme_combo.addItem(
            "Current",
            userData="default",
        )
        for cmap_name in HISTOGRAM_COLORMAP_NAMES:
            self.component_trace_color_scheme_combo.addItem(
                cmap_name,
                userData=cmap_name,
            )
        self.component_trace_color_scheme_combo.currentIndexChanged.connect(
            self._on_component_trace_color_scheme_changed
        )
        self.save_component_plot_data_button = QPushButton("Export Plot Data")
        self.save_component_plot_data_button.clicked.connect(
            self.save_component_plot_data_requested.emit
        )
        controls.addWidget(self.component_log_x_checkbox)
        controls.addWidget(self.component_log_y_checkbox)
        controls.addWidget(self.component_legend_toggle_button)
        controls.addWidget(self.component_model_range_button)
        controls.addWidget(self.component_all_traces_button)
        controls.addWidget(QLabel("Trace Colors"))
        controls.addWidget(self.component_trace_color_scheme_combo)
        controls.addWidget(self.save_component_plot_data_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.component_figure = Figure(figsize=(7.6, 5.8))
        self.component_canvas = FigureCanvasQTAgg(self.component_figure)
        self.component_canvas.mpl_connect(
            "pick_event",
            self._handle_component_legend_pick,
        )
        self.component_toolbar = NavigationToolbar2QT(
            self.component_canvas,
            self,
        )
        layout.addWidget(self.component_toolbar)
        self.component_canvas.setMinimumHeight(420)
        layout.addWidget(self.component_canvas)
        return group

    def _build_prior_group(self) -> QGroupBox:
        group = QGroupBox("Prior Histograms")
        group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.prior_mode_combo = QComboBox()
        self.prior_mode_combo.addItem(
            "Structure Fraction",
            userData="structure_fraction",
        )
        self.prior_mode_combo.addItem(
            "Atom Fraction",
            userData="atom_fraction",
        )
        self.prior_mode_combo.addItem(
            "Solvent Sort - Structure Fraction",
            userData="solvent_sort_structure_fraction",
        )
        self.prior_mode_combo.addItem(
            "Solvent Sort - Atom Fraction",
            userData="solvent_sort_atom_fraction",
        )
        self.prior_mode_combo.currentTextChanged.connect(
            self._update_prior_control_state
        )
        self.secondary_filter_label = QLabel("Secondary atom")
        self.secondary_filter_combo = QComboBox()
        self.secondary_filter_combo.currentTextChanged.connect(
            self._redraw_prior_preview_if_needed
        )
        self.prior_color_combo = QComboBox()
        self.prior_color_combo.addItems(HISTOGRAM_COLORMAP_NAMES)
        self.prior_color_combo.currentTextChanged.connect(
            self._redraw_prior_preview_if_needed
        )
        self.prior_match_trace_colors_checkbox = QCheckBox(
            "Match Trace Colors"
        )
        self.prior_match_trace_colors_checkbox.toggled.connect(
            self._redraw_prior_preview_if_needed
        )
        self.generate_prior_plot_button = QPushButton("Generate Plot")
        self.generate_prior_plot_button.clicked.connect(
            self.generate_prior_plot_requested.emit
        )
        self.save_prior_png_button = QPushButton("Save Plot PNG")
        self.save_prior_png_button.clicked.connect(
            self.save_prior_png_requested.emit
        )
        self.save_prior_plot_data_button = QPushButton("Export Plot Data")
        self.save_prior_plot_data_button.clicked.connect(
            self.save_prior_plot_data_requested.emit
        )
        controls.addWidget(QLabel("Mode"))
        controls.addWidget(self.prior_mode_combo)
        controls.addWidget(self.secondary_filter_label)
        controls.addWidget(self.secondary_filter_combo)
        controls.addWidget(self.generate_prior_plot_button)
        controls.addWidget(QLabel("Color"))
        controls.addWidget(self.prior_color_combo)
        controls.addWidget(self.prior_match_trace_colors_checkbox)
        controls.addWidget(self.save_prior_png_button)
        controls.addWidget(self.save_prior_plot_data_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.prior_figure = Figure(figsize=(7.6, 3.8))
        self.prior_canvas = FigureCanvasQTAgg(self.prior_figure)
        self.prior_canvas.setMinimumHeight(300)
        layout.addWidget(self.prior_canvas)
        self._update_prior_control_state()
        return group

    def _clusters_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.clusters_dir_edit, stretch=1)

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._choose_clusters_directory)
        refresh_button = QPushButton("Refresh Elements")
        refresh_button.clicked.connect(self.request_cluster_scan)
        layout.addWidget(browse_button)
        layout.addWidget(refresh_button)
        return row

    def _elements_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.available_elements_list, stretch=1)

        button_row = QVBoxLayout()
        add_button = QPushButton("Add Selected")
        add_button.clicked.connect(self._add_selected_elements_to_exclude)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(
            self._remove_selected_elements_from_exclude
        )
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return row

    def set_project_selected(self, selected: bool) -> None:
        self.forward_model_group.setEnabled(selected)
        self.model_group.setEnabled(selected)
        self.prior_mode_combo.setEnabled(selected)
        self.generate_prior_plot_button.setEnabled(selected)
        self.save_prior_png_button.setEnabled(selected)
        self.save_component_plot_data_button.setEnabled(selected)
        self.save_prior_plot_data_button.setEnabled(selected)
        self.save_project_button.setEnabled(selected)
        self._update_prior_control_state()
        if not selected:
            self._experimental_summary = None
            self._solvent_summary = None
        self._refresh_data_status_labels()
        self._apply_model_only_mode_state()

    def set_project_settings(
        self,
        settings: ProjectSettings,
        template_specs: list[TemplateSpec],
    ) -> None:
        resolved_project_dir = Path(settings.project_dir).expanduser()
        self.project_name_edit.setText(resolved_project_dir.name)
        self.project_dir_edit.setText(str(resolved_project_dir.parent))
        self.open_project_dir_edit.setText(settings.project_dir)
        self.clusters_dir_edit.setText(settings.clusters_dir or "")
        displayed_data_path = settings.experimental_data_path or (
            settings.copied_experimental_data_file or ""
        )
        loadable_data_path = (
            settings.copied_experimental_data_file
            or settings.experimental_data_path
            or ""
        )
        displayed_solvent_path = settings.solvent_data_path or (
            settings.copied_solvent_data_file or ""
        )
        loadable_solvent_path = (
            settings.copied_solvent_data_file
            or settings.solvent_data_path
            or ""
        )
        self.experimental_data_edit.setText(displayed_data_path)
        self.solvent_data_edit.setText(displayed_solvent_path)
        self._experimental_header_rows = int(
            settings.experimental_header_rows or 0
        )
        self._experimental_q_column = settings.experimental_q_column
        self._experimental_intensity_column = (
            settings.experimental_intensity_column
        )
        self._experimental_error_column = settings.experimental_error_column
        self._solvent_header_rows = int(settings.solvent_header_rows or 0)
        self._solvent_q_column = settings.solvent_q_column
        self._solvent_intensity_column = settings.solvent_intensity_column
        self._solvent_error_column = settings.solvent_error_column
        self.set_model_only_mode(settings.model_only_mode)
        self.qmin_edit.setText(
            "" if settings.q_min is None else f"{settings.q_min:g}"
        )
        self.qmax_edit.setText(
            "" if settings.q_max is None else f"{settings.q_max:g}"
        )
        self.use_experimental_grid_checkbox.setChecked(
            settings.use_experimental_grid
        )
        self.resample_points_spin.setValue(settings.q_points or 500)
        self.exclude_elements_edit.setText(" ".join(settings.exclude_elements))
        self._component_color_overrides = dict(settings.component_trace_colors)
        self.set_component_trace_color_scheme(
            settings.component_trace_color_scheme
        )
        self.set_experimental_trace_settings(
            visible=settings.experimental_trace_visible,
            color=settings.experimental_trace_color,
        )
        self.set_solvent_trace_settings(
            visible=settings.solvent_trace_visible,
            color=settings.solvent_trace_color,
        )
        self.set_available_elements(settings.available_elements)
        self.set_available_templates(
            template_specs,
            settings.selected_model_template,
        )

        if settings.cluster_inventory_rows:
            self._recognized_cluster_rows = list(
                settings.cluster_inventory_rows
            )
            self._populate_recognized_clusters_table(
                self._recognized_cluster_rows
            )
        elif settings.clusters_dir:
            self._recognized_cluster_rows = []
            self._populate_recognized_clusters_table([])
        else:
            self._recognized_cluster_rows = []
            self.set_available_elements([])
            self._populate_recognized_clusters_table([])
        self._update_secondary_filter_options(
            settings.available_elements,
            self._recognized_cluster_rows,
        )

        if loadable_data_path:
            try:
                summary = self._load_experimental_summary_from_path(
                    Path(loadable_data_path),
                    self._experimental_header_rows,
                    q_column=self._experimental_q_column,
                    intensity_column=self._experimental_intensity_column,
                    error_column=self._experimental_error_column,
                )
            except Exception:
                self._experimental_summary = None
                self.data_status_label.setText(
                    "Experimental data selected.\n"
                    "Parsing will be retried when you build the project."
                )
            else:
                self._apply_experimental_file(
                    Path(displayed_data_path),
                    summary,
                    force_recommended_q_range=(
                        settings.q_min is None and settings.q_max is None
                    ),
                    log_to_activity=False,
                    emit_project_change=False,
                )
        else:
            self._experimental_summary = None
        if loadable_solvent_path:
            try:
                solvent_summary = self._load_solvent_summary_from_path(
                    Path(loadable_solvent_path),
                    self._solvent_header_rows,
                    q_column=self._solvent_q_column,
                    intensity_column=self._solvent_intensity_column,
                    error_column=self._solvent_error_column,
                )
            except Exception:
                self._solvent_summary = None
                self.solvent_status_label.setText(
                    "Solvent data selected.\n"
                    "Parsing will be retried when the project is built."
                )
            else:
                self._apply_solvent_file(
                    Path(displayed_solvent_path),
                    solvent_summary,
                    log_to_activity=False,
                    emit_project_change=False,
                )
        else:
            self._solvent_summary = None
        self._refresh_data_status_labels()
        self._apply_model_only_mode_state()
        self._update_data_trace_control_state()
        self._redraw_saxs_preview()

    def set_available_templates(
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
        self._on_template_combo_changed()

    def set_available_elements(self, elements: list[str]) -> None:
        current_exclude = set(self.exclude_elements())
        self.available_elements_list.clear()
        for element in elements:
            item = QListWidgetItem(element)
            if element in current_exclude:
                item.setSelected(True)
            self.available_elements_list.addItem(item)
        self._update_secondary_filter_options(
            elements,
            self._recognized_cluster_rows,
        )

    def _update_active_template_field(self) -> None:
        active_text = self.template_combo.currentText().strip()
        self.active_template_edit.setText(active_text)

    def selected_template_name(self) -> str | None:
        if self.template_combo.count() == 0:
            return None
        return str(self.template_combo.currentData() or "").strip() or None

    def show_deprecated_templates(self) -> bool:
        return self.show_deprecated_templates_checkbox.isChecked()

    def set_show_deprecated_templates(self, enabled: bool) -> None:
        self.show_deprecated_templates_checkbox.blockSignals(True)
        self.show_deprecated_templates_checkbox.setChecked(bool(enabled))
        self.show_deprecated_templates_checkbox.blockSignals(False)

    def project_dir(self) -> Path | None:
        parent_dir = self.project_directory()
        project_name = self.project_name()
        if parent_dir is None or project_name is None:
            return None
        return parent_dir / project_name

    def project_directory(self) -> Path | None:
        text = self.project_dir_edit.text().strip()
        return Path(text).expanduser() if text else None

    def project_name(self) -> str | None:
        text = self.project_name_edit.text().strip()
        return text or None

    def model_only_mode(self) -> bool:
        return bool(self.model_only_mode_checkbox.isChecked())

    def set_model_only_mode(self, enabled: bool) -> None:
        self.model_only_mode_checkbox.blockSignals(True)
        self.model_only_mode_checkbox.setChecked(bool(enabled))
        self.model_only_mode_checkbox.blockSignals(False)
        if enabled and self.use_experimental_grid_checkbox.isChecked():
            self.use_experimental_grid_checkbox.setChecked(False)
        self._apply_model_only_mode_state()

    def open_project_dir(self) -> Path | None:
        text = self.open_project_dir_edit.text().strip()
        return Path(text).expanduser() if text else None

    def clusters_dir(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        return Path(text).expanduser() if text else None

    def experimental_data_path(self) -> Path | None:
        text = self.experimental_data_edit.text().strip()
        return Path(text).expanduser() if text else None

    def solvent_data_path(self) -> Path | None:
        text = self.solvent_data_edit.text().strip()
        return Path(text).expanduser() if text else None

    def _default_experimental_status_text(self) -> str:
        return (
            "Choose an experimental SAXS file or folder after opening a "
            "project.\n"
            "The selected file, columns, q-range, and import settings will "
            "be summarized here."
        )

    def _default_solvent_status_text(self) -> str:
        return (
            "Optional solvent SAXS data can be loaded here and will be "
            "carried into prefit and DREAM if the active model uses "
            "solvent intensities."
        )

    def _refresh_data_status_labels(self) -> None:
        if self.model_only_mode():
            self.data_status_label.setText(
                "Model Only Mode is enabled.\n"
                "Experimental data input is locked and hidden from the "
                "plots until Model Only Mode is turned off."
            )
            self.solvent_status_label.setText(
                "Model Only Mode is enabled.\n"
                "Solvent data input is locked and hidden from the plots "
                "until Model Only Mode is turned off."
            )
            return
        if self._experimental_summary is not None:
            self.data_status_label.setText(
                self._experimental_import_summary(self._experimental_summary)
            )
        else:
            self.data_status_label.setText(
                self._default_experimental_status_text()
            )
        if self._solvent_summary is not None:
            self.solvent_status_label.setText(
                self._experimental_import_summary(self._solvent_summary)
            )
        else:
            self.solvent_status_label.setText(
                self._default_solvent_status_text()
            )

    def _apply_model_only_mode_state(self) -> None:
        locked = self.model_only_mode()
        self.experimental_data_edit.setEnabled(not locked)
        self.experimental_file_button.setEnabled(not locked)
        self.experimental_folder_button.setEnabled(not locked)
        self.experimental_columns_button.setEnabled(not locked)
        self.experimental_clear_button.setEnabled(not locked)
        self.solvent_data_edit.setEnabled(not locked)
        self.solvent_file_button.setEnabled(not locked)
        self.solvent_columns_button.setEnabled(not locked)
        self.solvent_clear_button.setEnabled(not locked)
        self.use_experimental_grid_checkbox.setEnabled(not locked)
        if locked and self.use_experimental_grid_checkbox.isChecked():
            self.use_experimental_grid_checkbox.setChecked(False)
        self._update_resample_grid_state()
        self._refresh_data_status_labels()
        self._update_data_trace_control_state()
        self._redraw_saxs_preview()

    def experimental_header_rows(self) -> int:
        return int(self._experimental_header_rows)

    def experimental_q_column(self) -> int | None:
        return self._experimental_q_column

    def experimental_intensity_column(self) -> int | None:
        return self._experimental_intensity_column

    def experimental_error_column(self) -> int | None:
        return self._experimental_error_column

    def solvent_header_rows(self) -> int:
        return int(self._solvent_header_rows)

    def solvent_q_column(self) -> int | None:
        return self._solvent_q_column

    def solvent_intensity_column(self) -> int | None:
        return self._solvent_intensity_column

    def solvent_error_column(self) -> int | None:
        return self._solvent_error_column

    def available_elements(self) -> list[str]:
        return [
            self.available_elements_list.item(index).text().strip()
            for index in range(self.available_elements_list.count())
            if self.available_elements_list.item(index).text().strip()
        ]

    def recognized_cluster_rows(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._recognized_cluster_rows]

    def exclude_elements(self) -> list[str]:
        return self._parse_elements(self.exclude_elements_edit.text())

    def component_trace_colors(self) -> dict[str, str]:
        return dict(self._component_color_overrides)

    def component_trace_color_scheme(self) -> str:
        return (
            str(
                self.component_trace_color_scheme_combo.currentData()
                or "default"
            ).strip()
            or "default"
        )

    def experimental_trace_visible(self) -> bool:
        return bool(self.experimental_trace_visible_checkbox.isChecked())

    def experimental_trace_color(self) -> str:
        return self._trace_color_button_value(
            self.experimental_trace_color_button,
            default="#000000",
        )

    def solvent_trace_visible(self) -> bool:
        return bool(self.solvent_trace_visible_checkbox.isChecked())

    def solvent_trace_color(self) -> str:
        return self._trace_color_button_value(
            self.solvent_trace_color_button,
            default="#008000",
        )

    def q_min(self) -> float | None:
        return self._optional_float(self.qmin_edit.text())

    def q_max(self) -> float | None:
        return self._optional_float(self.qmax_edit.text())

    def default_experimental_q_range(self) -> tuple[float, float] | None:
        if self.model_only_mode() or self._experimental_summary is None:
            return None
        q_values = np.asarray(self._experimental_summary.q_values, dtype=float)
        if q_values.size == 0:
            return None
        return float(q_values.min()), float(q_values.max())

    def q_range_matches_loaded_experimental_defaults(self) -> bool:
        default_range = self.default_experimental_q_range()
        if default_range is None:
            return False
        q_min = self.q_min()
        q_max = self.q_max()
        if q_min is None or q_max is None:
            return False
        default_q_min, default_q_max = default_range
        return bool(
            np.isclose(q_min, default_q_min, rtol=0.0, atol=1e-9)
            and np.isclose(q_max, default_q_max, rtol=0.0, atol=1e-9)
        )

    def use_experimental_grid(self) -> bool:
        if self.model_only_mode():
            return False
        return bool(self.use_experimental_grid_checkbox.isChecked())

    def q_points(self) -> int | None:
        if self.use_experimental_grid():
            return None
        return int(self.resample_points_spin.value())

    def prior_mode(self) -> str:
        return str(self.prior_mode_combo.currentData() or "structure_fraction")

    def prior_secondary_element(self) -> str | None:
        if not self._prior_mode_uses_secondary_filter():
            return None
        text = self.secondary_filter_combo.currentText().strip()
        return text or None

    def selected_prior_secondary_element(self) -> str | None:
        text = self.secondary_filter_combo.currentText().strip()
        return text or None

    def prior_cmap(self) -> str:
        return self.prior_color_combo.currentText().strip() or "summer"

    def prior_match_trace_colors(self) -> bool:
        return bool(self.prior_match_trace_colors_checkbox.isChecked())

    def _prior_uses_component_color_sync(self) -> bool:
        if (
            self._prior_mode_uses_secondary_filter()
            or not self._component_paths
        ):
            return False
        return self.prior_match_trace_colors() or (
            self.component_trace_color_scheme() != "default"
        )

    def current_prior_json_path(self) -> Path | None:
        return self._current_prior_json_path

    def prior_structure_motif_colors(self) -> dict[str, str] | None:
        if not self._prior_uses_component_color_sync():
            return None
        scheme_colors = self._component_scheme_colors(self._component_paths)
        colors: dict[str, str] = {}
        for component_path in self._component_paths:
            component_key = component_path.stem
            colors[component_key] = (
                self._component_color_overrides.get(component_key)
                or self._component_color_lookup.get(component_key)
                or scheme_colors.get(component_key)
                or "#1f77b4"
            )
        return colors or None

    def append_summary(self, message: str) -> None:
        stripped = str(message).strip()
        if not stripped:
            return
        if self._summary_text:
            self._summary_text += "\n" + stripped
        else:
            self._summary_text = stripped
        self._render_summary_text()

    def component_plot_export_payload(self) -> dict[str, object]:
        traces: list[dict[str, object]] = []
        for axis_index, axis in enumerate(self.component_figure.axes):
            for line in axis.get_lines():
                traces.append(
                    {
                        "series": str(line.get_label()),
                        "component_key": str(line.get_gid() or ""),
                        "axis_index": axis_index,
                        "axis_ylabel": str(axis.get_ylabel()),
                        "color": str(line.get_color()),
                        "visible": bool(line.get_visible()),
                        "x": np.asarray(
                            line.get_xdata(orig=False),
                            dtype=float,
                        ),
                        "y": np.asarray(
                            line.get_ydata(orig=False),
                            dtype=float,
                        ),
                    }
                )
        return {
            "title": (
                str(self.component_figure.axes[0].get_title())
                if self.component_figure.axes
                else ""
            ),
            "trace_color_scheme": self.component_trace_color_scheme(),
            "log_x": bool(self.component_log_x_checkbox.isChecked()),
            "log_y": bool(self.component_log_y_checkbox.isChecked()),
            "traces": traces,
        }

    def set_component_trace_colors(
        self,
        colors: dict[str, str] | None,
    ) -> None:
        self._component_color_overrides = dict(colors or {})
        self._redraw_saxs_preview()

    def set_component_trace_color_scheme(
        self,
        scheme: str | None,
    ) -> None:
        normalized = str(scheme or "default").strip() or "default"
        index = self.component_trace_color_scheme_combo.findData(normalized)
        if index < 0:
            index = self.component_trace_color_scheme_combo.findData("default")
        self.component_trace_color_scheme_combo.setCurrentIndex(max(index, 0))

    def set_experimental_trace_settings(
        self,
        *,
        visible: bool,
        color: str,
    ) -> None:
        self.experimental_trace_visible_checkbox.setChecked(bool(visible))
        self._configure_trace_color_button(
            self.experimental_trace_color_button,
            color or "#000000",
            label="Experimental",
        )
        self._update_data_trace_control_state()

    def set_solvent_trace_settings(
        self,
        *,
        visible: bool,
        color: str,
    ) -> None:
        self.solvent_trace_visible_checkbox.setChecked(bool(visible))
        self._configure_trace_color_button(
            self.solvent_trace_color_button,
            color or "#008000",
            label="Solvent",
        )
        self._update_data_trace_control_state()

    def set_summary_text(self, text: str) -> None:
        self._summary_text = str(text).strip()
        self._render_summary_text()

    def set_console_autoscroll_enabled(self, enabled: bool) -> None:
        self._console_autoscroll_enabled = bool(enabled)
        if self._console_autoscroll_enabled:
            self._scroll_summary_to_end()

    def _render_summary_text(self) -> None:
        scrollbar = self.summary_box.verticalScrollBar()
        previous_value = scrollbar.value()
        previous_maximum = max(scrollbar.maximum(), 1)
        self.summary_box.setPlainText(self._summary_text)
        if self._console_autoscroll_enabled:
            self._scroll_summary_to_end()
            return
        updated_scrollbar = self.summary_box.verticalScrollBar()
        if updated_scrollbar.maximum() > 0:
            position_fraction = previous_value / previous_maximum
            updated_scrollbar.setValue(
                int(round(position_fraction * updated_scrollbar.maximum()))
            )

    def _scroll_summary_to_end(self) -> None:
        cursor = self.summary_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.summary_box.setTextCursor(cursor)
        self.summary_box.ensureCursorVisible()
        scrollbar = self.summary_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QTimer.singleShot(
            0,
            self._scroll_summary_to_end_once,
        )

    def _scroll_summary_to_end_once(self) -> None:
        cursor = self.summary_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.summary_box.setTextCursor(cursor)
        self.summary_box.ensureCursorVisible()
        scrollbar = self.summary_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def draw_component_plot(self, component_paths: list[Path] | None) -> None:
        self._component_paths = component_paths
        self._redraw_saxs_preview()
        self._redraw_prior_preview_if_needed()

    def request_cluster_scan(self) -> None:
        clusters_dir = self.clusters_dir()
        if clusters_dir is None or not clusters_dir.exists():
            self._recognized_cluster_rows = []
            self.set_available_elements([])
            self._populate_recognized_clusters_table([])
            return
        self.scan_clusters_requested.emit()

    def apply_cluster_import_data(
        self,
        available_elements: list[str],
        cluster_rows: list[dict[str, object]],
    ) -> None:
        self._recognized_cluster_rows = list(cluster_rows)
        self.set_available_elements(available_elements)
        self._populate_recognized_clusters_table(cluster_rows)
        self._update_secondary_filter_options(
            available_elements,
            cluster_rows,
        )
        self._redraw_prior_preview_if_needed()

    def start_activity_progress(
        self,
        total: int,
        message: str,
        *,
        unit_label: str = "items",
    ) -> None:
        total = max(int(total), 1)
        self.activity_progress_bar.setRange(0, total)
        self.activity_progress_bar.setValue(0)
        self.activity_progress_bar.setFormat(f"%v / %m {unit_label}")
        self.activity_progress_label.setText(message)

    def update_activity_progress(
        self,
        processed: int,
        total: int,
        message: str,
        *,
        unit_label: str = "items",
    ) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.activity_progress_bar.setRange(0, total)
        self.activity_progress_bar.setValue(processed)
        self.activity_progress_bar.setFormat(f"%v / %m {unit_label}")
        self.activity_progress_label.setText(message)

    def finish_activity_progress(
        self,
        message: str = "Progress: complete",
    ) -> None:
        self.activity_progress_label.setText(message)
        if self.activity_progress_bar.maximum() > 0:
            self.activity_progress_bar.setValue(
                self.activity_progress_bar.maximum()
            )

    def reset_activity_progress(self) -> None:
        self.activity_progress_label.setText("Progress: idle")
        self.activity_progress_bar.setRange(0, 1)
        self.activity_progress_bar.setValue(0)
        self.activity_progress_bar.setFormat("%v / %m items")

    def _redraw_saxs_preview(self) -> None:
        self.component_figure.clear()
        self._legend_line_map.clear()
        self._component_legend_lookup.clear()
        self._component_line_lookup.clear()
        self._component_color_lookup.clear()
        show_data_preview = not self.model_only_mode()
        has_data_preview = show_data_preview and (
            self._experimental_summary is not None
            or self._solvent_summary is not None
        )
        has_components = bool(self._component_paths)
        if not has_data_preview and not has_components:
            axis = self.component_figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                fill(
                    (
                        "Build SAXS components to preview averaged cluster "
                        "profiles in Model Only Mode."
                        if self.model_only_mode()
                        else "Select experimental data and build SAXS "
                        "components to preview the experimental range and "
                        "averaged cluster profiles."
                    ),
                    width=42,
                ),
                ha="center",
                va="center",
                transform=axis.transAxes,
                wrap=True,
            )
            axis.set_axis_off()
            self._update_component_table_visuals()
            self._update_component_trace_control_state()
            self.component_figure.tight_layout()
            self.component_canvas.draw()
            return

        base_axis = self.component_figure.add_subplot(111)
        experimental_axis = base_axis if has_data_preview else None
        component_axis = (
            base_axis if has_components and not has_data_preview else None
        )
        plotted_lines: list[object] = []

        if experimental_axis is not None:
            plotted_lines.extend(
                self._draw_experimental_preview(
                    experimental_axis,
                    self._experimental_summary if show_data_preview else None,
                )
            )

        if has_data_preview and has_components:
            component_axis = base_axis.twinx()

        if component_axis is not None:
            plotted_lines.extend(
                self._draw_component_profiles(
                    component_axis,
                    self._component_paths or [],
                )
            )
            if (
                self._experimental_summary is not None
                and experimental_axis is not None
            ):
                self._normalize_component_axis(
                    experimental_axis,
                    component_axis,
                )
                experimental_axis.set_ylabel(
                    "Experimental Intensity (arb. units)"
                )
                component_axis.set_ylabel("Model Intensity (arb. units)")
                base_axis.set_title("Experimental Data and SAXS Components")
            elif has_data_preview and experimental_axis is not None:
                experimental_axis.set_ylabel("Intensity (arb. units)")
                component_axis.set_ylabel("Model Intensity (arb. units)")
                base_axis.set_title("Data and SAXS Components")
            else:
                component_axis.set_ylabel("Model Intensity (arb. units)")
                base_axis.set_title("SAXS Component Preview")
        elif experimental_axis is not None:
            experimental_axis.set_ylabel("Intensity (arb. units)")
            if self._experimental_summary is not None:
                base_axis.set_title("Experimental Data Preview")
            else:
                base_axis.set_title("Data Preview")

        if (
            component_axis is not None
            and self.component_model_range_button.isChecked()
        ):
            self._autoscale_to_model_range(
                experimental_axis,
                component_axis,
            )

        anchor_axis = experimental_axis or component_axis
        if (
            anchor_axis is not None
            and plotted_lines
            and self.component_legend_toggle_button.isChecked()
        ):
            self._build_interactive_legend(anchor_axis, plotted_lines)

        self._update_component_table_visuals()
        self._update_component_trace_control_state()
        self.component_figure.tight_layout()
        self.component_canvas.draw()

    def draw_prior_plot(self, json_path: str | Path | None) -> None:
        self._current_prior_json_path = (
            Path(json_path).expanduser().resolve()
            if json_path is not None
            else None
        )
        self.prior_figure.clear()
        if json_path is None:
            axis = self.prior_figure.add_subplot(111)
            axis.text(
                0.5,
                0.5,
                "Generate prior weights to preview the selected prior histogram.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
        else:
            axis = self.prior_figure.add_subplot(111)
            try:
                plot_md_prior_histogram(
                    json_path,
                    mode=self.prior_mode(),
                    secondary_element=self.prior_secondary_element(),
                    cmap=self.prior_cmap(),
                    structure_motif_colors=self.prior_structure_motif_colors(),
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
                )
                axis.set_axis_off()
        self.prior_canvas.draw()

    def refresh_available_elements(self) -> None:
        self.request_cluster_scan()

    def _refresh_recognized_clusters(self) -> None:
        clusters_dir = self.clusters_dir()
        if clusters_dir is None or not clusters_dir.exists():
            self._recognized_cluster_rows = []
            self._populate_recognized_clusters_table([])
            return
        try:
            cluster_bins = discover_cluster_bins(clusters_dir)
        except Exception:
            self._recognized_cluster_rows = []
            self._populate_recognized_clusters_table([])
            return

        total_count = sum(
            len(cluster_bin.files) for cluster_bin in cluster_bins
        )
        total_atom_weight = sum(
            len(cluster_bin.files)
            * self._structure_atom_weight(cluster_bin.structure)
            for cluster_bin in cluster_bins
        )
        rows: list[dict[str, object]] = []
        for cluster_bin in cluster_bins:
            count = len(cluster_bin.files)
            structure_weight = count / total_count if total_count else 0.0
            atom_fraction = (
                count
                * self._structure_atom_weight(cluster_bin.structure)
                / total_atom_weight
                * 100.0
                if total_atom_weight
                else 0.0
            )
            rows.append(
                {
                    "structure": cluster_bin.structure,
                    "motif": cluster_bin.motif,
                    "count": count,
                    "weight": structure_weight,
                    "atom_fraction_percent": atom_fraction,
                    "structure_fraction_percent": structure_weight * 100.0,
                }
            )
        self._recognized_cluster_rows = list(rows)
        self._populate_recognized_clusters_table(rows)
        self._update_secondary_filter_options(
            self.available_elements(),
            rows,
        )

    def _populate_recognized_clusters_table(
        self,
        rows: list[dict[str, object]],
    ) -> None:
        self._updating_cluster_table = True
        self.recognized_clusters_table.blockSignals(True)
        self._component_row_lookup = {}
        self.recognized_clusters_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            component_key = self._component_key(
                str(row["structure"]),
                str(row["motif"]),
            )
            self._component_row_lookup[component_key] = row_index
            values = [
                str(row["structure"]),
                str(row["motif"]),
                str(row["count"]),
                self._format_truncated_decimal(float(row["weight"])),
                f"{float(row['atom_fraction_percent']):.2f}",
                f"{float(row['structure_fraction_percent']):.2f}",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.recognized_clusters_table.setItem(
                    row_index,
                    column_index,
                    item,
                )
            self.recognized_clusters_table.setItem(
                row_index,
                6,
                self._build_visibility_table_item(component_key),
            )
            self.recognized_clusters_table.setItem(
                row_index,
                7,
                self._build_color_table_item(component_key),
            )
        self.recognized_clusters_table.resizeRowsToContents()
        self.recognized_clusters_table.blockSignals(False)
        self._updating_cluster_table = False

    def _path_action_row(
        self,
        line_edit: QLineEdit,
        *,
        dialog_title: str,
        action_button_text: str,
        action,
        browse_handler=None,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, stretch=1)
        browse_button = QPushButton("Browse…")
        if browse_handler is None:
            browse_button.clicked.connect(
                lambda: self._browse_directory(
                    line_edit,
                    dialog_title=dialog_title,
                )
            )
        else:
            browse_button.clicked.connect(browse_handler)
        layout.addWidget(browse_button)
        action_button = QPushButton(action_button_text)
        action_button.clicked.connect(action)
        layout.addWidget(action_button)
        return row

    @staticmethod
    def _configure_trace_color_button(
        button: QPushButton,
        color: str,
        *,
        label: str,
    ) -> None:
        normalized = str(color).strip() or "#000000"
        qcolor = QColor(normalized)
        foreground = "#ffffff"
        if qcolor.isValid() and qcolor.lightness() > 128:
            foreground = "#000000"
        button.setText(normalized)
        button.setToolTip(f"{label} trace color: {normalized}")
        button.setMinimumWidth(88)
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: {normalized};"
            f"color: {foreground};"
            "border: 1px solid #666666;"
            "padding: 2px 8px;"
            "}"
        )

    @staticmethod
    def _trace_color_button_value(
        button: QPushButton,
        *,
        default: str,
    ) -> str:
        text = button.text().strip()
        return text or default

    def _experimental_data_row(self) -> QWidget:
        row = QWidget()
        self.experimental_data_row_widget = row
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.experimental_data_edit, stretch=1)

        self.experimental_file_button = QPushButton("File…")
        self.experimental_file_button.clicked.connect(
            self._choose_experimental_file
        )
        self.experimental_folder_button = QPushButton("Folder…")
        self.experimental_folder_button.clicked.connect(
            self._choose_experimental_folder
        )
        self.experimental_columns_button = QPushButton("Columns…")
        self.experimental_columns_button.clicked.connect(
            self._configure_experimental_columns
        )
        self.experimental_clear_button = QPushButton("Clear")
        self.experimental_clear_button.clicked.connect(
            self._clear_experimental_selection
        )
        layout.addWidget(self.experimental_file_button)
        layout.addWidget(self.experimental_folder_button)
        layout.addWidget(self.experimental_columns_button)
        layout.addWidget(self.experimental_clear_button)
        self.experimental_trace_visible_checkbox = QCheckBox()
        self.experimental_trace_visible_checkbox.setChecked(True)
        self.experimental_trace_visible_checkbox.toggled.connect(
            self._redraw_saxs_preview
        )
        self.experimental_trace_color_button = QPushButton()
        self.experimental_trace_color_button.clicked.connect(
            lambda: self._choose_data_trace_color("experimental")
        )
        self._configure_trace_color_button(
            self.experimental_trace_color_button,
            "#000000",
            label="Experimental",
        )
        layout.addWidget(QLabel("Visible"))
        layout.addWidget(self.experimental_trace_visible_checkbox)
        layout.addWidget(QLabel("Color"))
        layout.addWidget(self.experimental_trace_color_button)
        return row

    def _solvent_data_row(self) -> QWidget:
        row = QWidget()
        self.solvent_data_row_widget = row
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.solvent_data_edit, stretch=1)

        self.solvent_file_button = QPushButton("File…")
        self.solvent_file_button.clicked.connect(self._choose_solvent_file)
        self.solvent_columns_button = QPushButton("Columns…")
        self.solvent_columns_button.clicked.connect(
            self._configure_solvent_columns
        )
        self.solvent_clear_button = QPushButton("Clear")
        self.solvent_clear_button.clicked.connect(
            self._clear_solvent_selection
        )
        layout.addWidget(self.solvent_file_button)
        layout.addWidget(self.solvent_columns_button)
        layout.addWidget(self.solvent_clear_button)
        self.solvent_trace_visible_checkbox = QCheckBox()
        self.solvent_trace_visible_checkbox.setChecked(True)
        self.solvent_trace_visible_checkbox.toggled.connect(
            self._redraw_saxs_preview
        )
        self.solvent_trace_color_button = QPushButton()
        self.solvent_trace_color_button.clicked.connect(
            lambda: self._choose_data_trace_color("solvent")
        )
        self._configure_trace_color_button(
            self.solvent_trace_color_button,
            "#008000",
            label="Solvent",
        )
        layout.addWidget(QLabel("Visible"))
        layout.addWidget(self.solvent_trace_visible_checkbox)
        layout.addWidget(QLabel("Color"))
        layout.addWidget(self.solvent_trace_color_button)
        return row

    def _browse_directory(
        self,
        line_edit: QLineEdit,
        *,
        dialog_title: str,
    ) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            dialog_title,
            line_edit.text().strip() or str(Path.home()),
        )
        if selected:
            line_edit.setText(selected)

    def _browse_existing_project_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select an existing SAXS project folder",
            self.open_project_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        project_dir = Path(selected).expanduser()
        project_file = build_project_paths(project_dir).project_file
        if not project_file.is_file():
            self.open_project_dir_edit.clear()
            self.append_summary(
                "Select a complete SAXS project folder that contains "
                "saxs_project.json, not a parent directory of multiple projects."
            )
            return
        self.open_project_dir_edit.setText(str(project_dir))

    def _choose_clusters_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select clusters folder",
            self.clusters_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.clusters_dir_edit.setText(selected)
        self.request_cluster_scan()
        self.autosave_project_requested.emit("selected a new clusters folder")

    def _choose_experimental_file(self) -> None:
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select experimental SAXS data file",
            self.experimental_data_edit.text().strip() or str(Path.home()),
            "Data files (*.txt *.dat *.iq);;All files (*)",
        )
        if not file_path:
            return
        selected_path = Path(file_path).expanduser()
        try:
            summary = load_experimental_data_file(selected_path, skiprows=0)
            self._apply_experimental_file(selected_path, summary)
            return
        except Exception:
            pass
        dialog = ExperimentalDataHeaderDialog(
            selected_path,
            self,
            initial_header_rows=self._experimental_header_rows,
            initial_q_column=self._experimental_q_column,
            initial_intensity_column=self._experimental_intensity_column,
            initial_error_column=self._experimental_error_column,
        )
        if dialog.exec():
            accepted_summary = dialog.accepted_summary
            if accepted_summary is None:
                return
            self._apply_experimental_file(selected_path, accepted_summary)

    def _choose_experimental_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select experimental SAXS data folder",
            self.experimental_data_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        selected_path = Path(selected).expanduser()
        self.experimental_data_edit.setText(str(selected_path))
        self._experimental_header_rows = 0
        try:
            summary = self._load_experimental_summary_from_path(
                selected_path,
                self._experimental_header_rows,
                q_column=self._experimental_q_column,
                intensity_column=self._experimental_intensity_column,
                error_column=self._experimental_error_column,
            )
        except Exception:
            self._experimental_summary = None
            self.data_status_label.setText(
                "Experimental data folder selected.\n"
                "The first matching text file will be loaded when the project "
                "is built."
            )
            self._redraw_saxs_preview()
            self.autosave_project_requested.emit(
                "selected a new experimental data folder"
            )
            return
        self._apply_experimental_file(selected_path, summary)

    def _clear_experimental_selection(self) -> None:
        self.experimental_data_edit.clear()
        self._experimental_header_rows = 0
        self._experimental_q_column = None
        self._experimental_intensity_column = None
        self._experimental_error_column = None
        self._experimental_summary = None
        self._refresh_data_status_labels()
        self._update_data_trace_control_state()
        self._redraw_saxs_preview()
        self.autosave_project_requested.emit("cleared experimental data")

    def _choose_solvent_file(self) -> None:
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select solvent SAXS data file",
            self.solvent_data_edit.text().strip() or str(Path.home()),
            "Data files (*.txt *.dat *.iq);;All files (*)",
        )
        if not file_path:
            return
        selected_path = Path(file_path).expanduser()
        try:
            summary = load_experimental_data_file(selected_path, skiprows=0)
            self._apply_solvent_file(selected_path, summary)
            return
        except Exception:
            pass
        dialog = ExperimentalDataHeaderDialog(
            selected_path,
            self,
            initial_header_rows=self._solvent_header_rows,
            initial_q_column=self._solvent_q_column,
            initial_intensity_column=self._solvent_intensity_column,
            initial_error_column=self._solvent_error_column,
        )
        if dialog.exec():
            accepted_summary = dialog.accepted_summary
            if accepted_summary is None:
                return
            self._apply_solvent_file(selected_path, accepted_summary)

    def _clear_solvent_selection(self) -> None:
        self.solvent_data_edit.clear()
        self._solvent_header_rows = 0
        self._solvent_q_column = None
        self._solvent_intensity_column = None
        self._solvent_error_column = None
        self._solvent_summary = None
        self._refresh_data_status_labels()
        self._update_data_trace_control_state()
        self._redraw_saxs_preview()
        self.autosave_project_requested.emit("cleared solvent data")

    def _apply_experimental_file(
        self,
        selected_path: Path,
        summary: ExperimentalDataSummary,
        *,
        force_recommended_q_range: bool = True,
        log_to_activity: bool = True,
        emit_project_change: bool = True,
    ) -> None:
        self.experimental_data_edit.setText(str(selected_path))
        self._experimental_header_rows = int(summary.header_rows)
        self._experimental_q_column = int(summary.q_column)
        self._experimental_intensity_column = int(summary.intensity_column)
        self._experimental_error_column = summary.error_column
        self._experimental_summary = summary
        self._set_experimental_status(summary)
        self._apply_recommended_q_range(
            summary,
            force=force_recommended_q_range,
        )
        self._update_data_trace_control_state()
        if log_to_activity:
            self.append_summary(
                "Experimental data import\n"
                + self._experimental_import_summary(summary)
            )
        if emit_project_change:
            self.autosave_project_requested.emit(
                "selected a new experimental data source"
            )

    def _apply_solvent_file(
        self,
        selected_path: Path,
        summary: ExperimentalDataSummary,
        *,
        log_to_activity: bool = True,
        emit_project_change: bool = True,
    ) -> None:
        self.solvent_data_edit.setText(str(selected_path))
        self._solvent_header_rows = int(summary.header_rows)
        self._solvent_q_column = int(summary.q_column)
        self._solvent_intensity_column = int(summary.intensity_column)
        self._solvent_error_column = summary.error_column
        self._solvent_summary = summary
        self._set_solvent_status(summary)
        self._update_data_trace_control_state()
        if log_to_activity:
            self.append_summary(
                "Solvent data import\n"
                + self._experimental_import_summary(summary)
            )
        self._redraw_saxs_preview()
        if emit_project_change:
            self.autosave_project_requested.emit(
                "selected a new solvent data source"
            )

    def _apply_recommended_q_range(
        self,
        summary: ExperimentalDataSummary,
        *,
        force: bool,
    ) -> None:
        if force or not self.qmin_edit.text().strip():
            self.qmin_edit.setText(f"{float(summary.q_values.min()):g}")
        if force or not self.qmax_edit.text().strip():
            self.qmax_edit.setText(f"{float(summary.q_values.max()):g}")

    def _set_experimental_status(
        self,
        summary: ExperimentalDataSummary,
    ) -> None:
        del summary
        self._refresh_data_status_labels()

    def _set_solvent_status(
        self,
        summary: ExperimentalDataSummary,
    ) -> None:
        del summary
        self._refresh_data_status_labels()

    def _load_experimental_summary_from_path(
        self,
        selected_path: Path,
        header_rows: int,
        *,
        q_column: int | None = None,
        intensity_column: int | None = None,
        error_column: int | None = None,
    ) -> ExperimentalDataSummary:
        if selected_path.is_file():
            return load_experimental_data_file(
                selected_path,
                skiprows=header_rows,
                q_column=q_column,
                intensity_column=intensity_column,
                error_column=error_column,
            )
        candidate = self._first_experimental_candidate(selected_path)
        if candidate is not None:
            return load_experimental_data_file(
                candidate,
                skiprows=header_rows,
                q_column=q_column,
                intensity_column=intensity_column,
                error_column=error_column,
            )
        raise FileNotFoundError(
            "No supported experimental data file was found in the selected folder."
        )

    def _load_solvent_summary_from_path(
        self,
        selected_path: Path,
        header_rows: int,
        *,
        q_column: int | None = None,
        intensity_column: int | None = None,
        error_column: int | None = None,
    ) -> ExperimentalDataSummary:
        if selected_path.is_file():
            return load_experimental_data_file(
                selected_path,
                skiprows=header_rows,
                q_column=q_column,
                intensity_column=intensity_column,
                error_column=error_column,
            )
        candidate = self._first_solvent_candidate(selected_path)
        if candidate is not None:
            return load_experimental_data_file(
                candidate,
                skiprows=header_rows,
                q_column=q_column,
                intensity_column=intensity_column,
                error_column=error_column,
            )
        raise FileNotFoundError(
            "No supported solvent data file was found in the selected folder."
        )

    def _configure_experimental_columns(self) -> None:
        selected_path = self.experimental_data_path()
        if selected_path is None:
            self.data_status_label.setText(
                "Select an experimental data file or folder before configuring columns."
            )
            return

        file_path = None
        if self._experimental_summary is not None:
            file_path = self._experimental_summary.path
        elif selected_path.is_file():
            file_path = selected_path
        elif selected_path.is_dir():
            file_path = self._first_experimental_candidate(selected_path)

        if file_path is None:
            self.data_status_label.setText(
                "No supported experimental data file was found to configure columns."
            )
            return

        dialog = ExperimentalDataHeaderDialog(
            file_path,
            self,
            initial_header_rows=self._experimental_header_rows,
            initial_q_column=self._experimental_q_column,
            initial_intensity_column=self._experimental_intensity_column,
            initial_error_column=self._experimental_error_column,
        )
        if not dialog.exec():
            return
        accepted_summary = dialog.accepted_summary
        if accepted_summary is None:
            return
        self._apply_experimental_file(selected_path, accepted_summary)

    def _configure_solvent_columns(self) -> None:
        selected_path = self.solvent_data_path()
        if selected_path is None:
            self.solvent_status_label.setText(
                "Select a solvent data file before configuring columns."
            )
            return

        file_path = None
        if self._solvent_summary is not None:
            file_path = self._solvent_summary.path
        elif selected_path.is_file():
            file_path = selected_path
        elif selected_path.is_dir():
            file_path = self._first_solvent_candidate(selected_path)

        if file_path is None:
            self.solvent_status_label.setText(
                "No supported solvent data file was found to configure columns."
            )
            return

        dialog = ExperimentalDataHeaderDialog(
            file_path,
            self,
            initial_header_rows=self._solvent_header_rows,
            initial_q_column=self._solvent_q_column,
            initial_intensity_column=self._solvent_intensity_column,
            initial_error_column=self._solvent_error_column,
        )
        if not dialog.exec():
            return
        accepted_summary = dialog.accepted_summary
        if accepted_summary is None:
            return
        self._apply_solvent_file(selected_path, accepted_summary)

    def _update_data_trace_control_state(self) -> None:
        if self.model_only_mode():
            has_experimental = False
            has_solvent = False
        else:
            has_experimental = self._experimental_summary is not None
            has_solvent = self._solvent_summary is not None
        self.experimental_trace_visible_checkbox.setEnabled(has_experimental)
        self.experimental_trace_color_button.setEnabled(has_experimental)
        self.solvent_trace_visible_checkbox.setEnabled(has_solvent)
        self.solvent_trace_color_button.setEnabled(has_solvent)

    def _choose_data_trace_color(self, trace_name: str) -> None:
        if trace_name == "experimental":
            button = self.experimental_trace_color_button
            current_color = self.experimental_trace_color()
            label = "Experimental"
        elif trace_name == "solvent":
            button = self.solvent_trace_color_button
            current_color = self.solvent_trace_color()
            label = "Solvent"
        else:
            raise ValueError(f"Unknown data trace name: {trace_name}")
        chosen = QColorDialog.getColor(
            QColor(current_color),
            self,
            f"Choose color for {label} trace",
        )
        if not chosen.isValid():
            return
        self._configure_trace_color_button(
            button,
            chosen.name(),
            label=label,
        )
        self._redraw_saxs_preview()

    def _on_clusters_dir_edited(self) -> None:
        self.request_cluster_scan()
        if self.clusters_dir() is not None:
            self.autosave_project_requested.emit(
                "updated the clusters folder reference"
            )
        else:
            self.autosave_project_requested.emit(
                "cleared the clusters folder reference"
            )

    def _on_model_only_mode_toggled(self, enabled: bool) -> None:
        if enabled and self.use_experimental_grid_checkbox.isChecked():
            self.use_experimental_grid_checkbox.setChecked(False)
        self._apply_model_only_mode_state()
        self.model_only_mode_changed.emit(bool(enabled))
        self.autosave_project_requested.emit(
            "enabled Model Only Mode"
            if enabled
            else "disabled Model Only Mode"
        )

    def _update_resample_grid_state(self) -> None:
        self.resample_points_spin.setEnabled(
            not self.use_experimental_grid_checkbox.isChecked()
        )

    def _update_secondary_filter_options(
        self,
        available_elements: list[str] | None = None,
        cluster_rows: list[dict[str, object]] | None = None,
    ) -> None:
        elements = {
            str(element).strip()
            for element in (available_elements or self.available_elements())
            if str(element).strip()
        }
        rows = (
            list(cluster_rows)
            if cluster_rows is not None
            else list(self._recognized_cluster_rows)
        )
        if not rows:
            secondary_elements: list[str] = []
        else:
            structure_element_sets = [
                set(
                    parse_stoich_label(
                        str(row.get("structure", "")).strip()
                    ).keys()
                )
                for row in rows
                if str(row.get("structure", "")).strip()
            ]
            axis_elements = (
                set.intersection(*structure_element_sets)
                if structure_element_sets
                else set()
            )
            secondary_elements = sorted(
                elements - axis_elements,
                key=self._natural_sort_key,
            )
        current_text = self.secondary_filter_combo.currentText().strip()
        self.secondary_filter_combo.blockSignals(True)
        self.secondary_filter_combo.clear()
        self.secondary_filter_combo.addItems(secondary_elements)
        if current_text:
            index = self.secondary_filter_combo.findText(current_text)
            if index >= 0:
                self.secondary_filter_combo.setCurrentIndex(index)
        if (
            self.secondary_filter_combo.currentIndex() < 0
            and secondary_elements
        ):
            self.secondary_filter_combo.setCurrentIndex(0)
        self.secondary_filter_combo.blockSignals(False)
        self._update_prior_control_state()

    def _prior_mode_uses_secondary_filter(self) -> bool:
        return self.prior_mode().startswith("solvent_sort")

    def _update_prior_control_state(self) -> None:
        uses_secondary = self._prior_mode_uses_secondary_filter()
        has_secondary_options = self.secondary_filter_combo.count() > 0
        can_match_trace_colors = not uses_secondary
        self.secondary_filter_label.setVisible(uses_secondary)
        self.secondary_filter_combo.setVisible(uses_secondary)
        self.secondary_filter_combo.setEnabled(
            uses_secondary
            and self.prior_mode_combo.isEnabled()
            and has_secondary_options
        )
        self.prior_match_trace_colors_checkbox.setVisible(
            can_match_trace_colors
        )
        self.prior_match_trace_colors_checkbox.setEnabled(
            can_match_trace_colors and bool(self._component_paths)
        )
        if uses_secondary and not has_secondary_options:
            self.secondary_filter_combo.setToolTip(
                "No secondary atom filters are available for the current "
                "recognized cluster stoichiometries."
            )
        else:
            self.secondary_filter_combo.setToolTip("")

    def _redraw_prior_preview_if_needed(self) -> None:
        if self._current_prior_json_path is None:
            return
        self.draw_prior_plot(self._current_prior_json_path)

    def _on_component_trace_color_scheme_changed(self) -> None:
        self._redraw_saxs_preview()
        self._redraw_prior_preview_if_needed()

    def _draw_experimental_preview(
        self,
        axis,
        summary: ExperimentalDataSummary | None,
    ) -> list[object]:
        lines: list[object] = []
        if summary is not None and self.experimental_trace_visible():
            q_values = np.asarray(summary.q_values, dtype=float)
            intensities = np.asarray(summary.intensities, dtype=float)
            exp_color = self.experimental_trace_color()
            (full_line,) = axis.plot(
                q_values,
                intensities,
                color=exp_color,
                alpha=0.35,
                linewidth=1.3,
                label="Experimental data",
            )
            lines.append(full_line)

            selected_mask = self._selected_q_mask(q_values)
            if selected_mask is not None and np.any(selected_mask):
                if not np.all(selected_mask):
                    (selected_line,) = axis.plot(
                        q_values[selected_mask],
                        intensities[selected_mask],
                        color=exp_color,
                        linewidth=1.8,
                        label="Selected q-range",
                    )
                    lines.append(selected_line)
                else:
                    full_line.set_alpha(1.0)
                    full_line.set_linewidth(1.8)
                    full_line.set_label("Experimental data")
            else:
                axis.text(
                    0.5,
                    0.08,
                    "Selected q-range does not overlap the loaded experimental data.",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize="small",
                )

        if self._solvent_summary is not None and self.solvent_trace_visible():
            solvent_q_values = np.asarray(
                self._solvent_summary.q_values,
                dtype=float,
            )
            solvent_intensities = np.asarray(
                self._solvent_summary.intensities,
                dtype=float,
            )
            solvent_color = self.solvent_trace_color()
            (solvent_line,) = axis.plot(
                solvent_q_values,
                solvent_intensities,
                color=solvent_color,
                alpha=0.45,
                linewidth=1.3,
                label="Solvent data",
            )
            lines.append(solvent_line)

            solvent_selected_mask = self._selected_q_mask(solvent_q_values)
            if solvent_selected_mask is not None and np.any(
                solvent_selected_mask
            ):
                if not np.all(solvent_selected_mask):
                    (selected_solvent_line,) = axis.plot(
                        solvent_q_values[solvent_selected_mask],
                        solvent_intensities[solvent_selected_mask],
                        color=solvent_color,
                        linewidth=1.8,
                        label="Selected solvent q-range",
                    )
                    lines.append(selected_solvent_line)
                else:
                    solvent_line.set_alpha(1.0)
                    solvent_line.set_linewidth(1.8)
                    solvent_line.set_label("Solvent data")

        self._apply_saxs_axis_style(axis, is_component_axis=False)
        return lines

    def _draw_component_profiles(
        self,
        axis,
        component_paths: list[Path],
    ) -> list[object]:
        if not component_paths:
            axis.text(
                0.5,
                0.5,
                "Build SAXS components to preview the averaged cluster profiles.",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            return []

        lines: list[object] = []
        scheme_colors = self._component_scheme_colors(component_paths)
        for component_path in component_paths:
            data = np.loadtxt(component_path, comments="#")
            q_values = np.asarray(data[:, 0], dtype=float)
            intensities = np.asarray(data[:, 1], dtype=float)
            component_key = component_path.stem
            visible = self._component_visibility.get(component_key, True)
            color_override = self._component_color_overrides.get(component_key)
            line_color = color_override or scheme_colors.get(component_key)
            (line,) = axis.plot(
                q_values,
                intensities,
                label=component_path.stem,
                linewidth=1.4,
                visible=visible,
                color=line_color,
            )
            line.set_gid(component_key)
            self._component_visibility.setdefault(component_key, visible)
            self._component_line_lookup[component_key] = line
            self._component_color_lookup[component_key] = str(line.get_color())
            lines.append(line)
        self._apply_saxs_axis_style(axis, is_component_axis=True)
        return lines

    def _component_scheme_colors(
        self,
        component_paths: list[Path],
    ) -> dict[str, str]:
        scheme = self.component_trace_color_scheme()
        if not component_paths:
            return {}
        if scheme == "default":
            default_cycle = list(
                rcParams["axes.prop_cycle"]
                .by_key()
                .get(
                    "color",
                    ["#1f77b4"],
                )
            )
            return {
                path.stem: default_cycle[index % len(default_cycle)]
                for index, path in enumerate(component_paths)
            }
        cmap = colormaps[scheme]
        if len(component_paths) == 1:
            positions = np.asarray([0.6], dtype=float)
        else:
            positions = np.linspace(0.15, 0.85, len(component_paths))
        return {
            path.stem: to_hex(cmap(float(position)), keep_alpha=False)
            for path, position in zip(component_paths, positions)
        }

    def _apply_saxs_axis_style(self, axis, *, is_component_axis: bool) -> None:
        axis.set_xscale(
            "log" if self.component_log_x_checkbox.isChecked() else "linear"
        )
        axis.set_yscale(
            "log" if self.component_log_y_checkbox.isChecked() else "linear"
        )
        if not is_component_axis or self._experimental_summary is None:
            axis.set_xlabel("q (Å⁻¹)")
        if not is_component_axis:
            axis.set_ylabel("Intensity (arb. units)")

    def _normalize_component_axis(
        self, experimental_axis, component_axis
    ) -> None:
        if self._experimental_summary is None:
            return
        component_lines = [
            line for line in component_axis.get_lines() if line.get_visible()
        ]
        if not component_lines:
            return
        exp_q = np.asarray(self._experimental_summary.q_values, dtype=float)
        exp_i = np.asarray(self._experimental_summary.intensities, dtype=float)
        exp_mask = self._selected_q_mask(exp_q)
        if exp_mask is None or not np.any(exp_mask):
            return
        filtered_q = exp_q[exp_mask]
        filtered_i = exp_i[exp_mask]

        component_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in component_lines
            ]
        )
        component_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in component_lines
            ]
        )
        overlap_mask = (filtered_q >= float(component_q.min())) & (
            filtered_q <= float(component_q.max())
        )
        if np.any(overlap_mask):
            filtered_i = filtered_i[overlap_mask]
        component_i = component_i[np.isfinite(component_i)]
        filtered_i = filtered_i[np.isfinite(filtered_i)]
        if self.component_log_y_checkbox.isChecked():
            filtered_i = filtered_i[filtered_i > 0.0]
            component_i = component_i[component_i > 0.0]
        if filtered_i.size == 0 or component_i.size == 0:
            return

        left_limits = experimental_axis.get_ylim()
        right_limits = self._aligned_y_limits(
            left_limits,
            float(np.nanmin(filtered_i)),
            float(np.nanmax(filtered_i)),
            float(np.nanmin(component_i)),
            float(np.nanmax(component_i)),
            log_scale=self.component_log_y_checkbox.isChecked(),
        )
        component_axis.set_ylim(right_limits)

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
        self._legend_line_map.clear()
        self._component_legend_lookup.clear()
        legend_handles = getattr(legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = getattr(legend, "legendHandles", [])
        for legend_line, original_line in zip(legend_handles, lines):
            if hasattr(legend_line, "set_picker"):
                legend_line.set_picker(True)
                legend_line.set_pickradius(6)
            legend_line.set_alpha(1.0 if original_line.get_visible() else 0.25)
            self._legend_line_map[legend_line] = original_line
            line_key = str(original_line.get_gid() or "").strip()
            if line_key:
                self._component_legend_lookup[line_key] = legend_line

    def _handle_component_legend_pick(self, event) -> None:
        original_line = self._legend_line_map.get(event.artist)
        if original_line is None:
            return
        is_visible = not original_line.get_visible()
        original_line.set_visible(is_visible)
        line_key = str(original_line.get_gid() or "").strip()
        if line_key:
            self._component_visibility[line_key] = is_visible
        if hasattr(event.artist, "set_alpha"):
            event.artist.set_alpha(1.0 if is_visible else 0.25)
        self._update_component_table_visuals()
        self._update_component_trace_control_state()
        self._refresh_component_axes()
        self.component_canvas.draw_idle()

    def _on_recognized_cluster_item_changed(
        self,
        item: QTableWidgetItem,
    ) -> None:
        if self._updating_cluster_table or item.column() != 6:
            return
        component_key = item.data(Qt.ItemDataRole.UserRole)
        if not component_key:
            return
        if component_key not in self._component_visibility:
            return
        visible = item.checkState() == Qt.CheckState.Checked
        self._component_visibility[str(component_key)] = visible
        line = self._component_line_lookup.get(str(component_key))
        if line is not None:
            line.set_visible(visible)
        legend_line = self._component_legend_lookup.get(str(component_key))
        if legend_line is not None and hasattr(legend_line, "set_alpha"):
            legend_line.set_alpha(1.0 if visible else 0.25)
        self._update_component_trace_control_state()
        self._refresh_component_axes()
        self.component_canvas.draw_idle()

    def _on_recognized_cluster_cell_clicked(
        self,
        row: int,
        column: int,
    ) -> None:
        if column != 7:
            return
        item = self.recognized_clusters_table.item(row, column)
        if item is None:
            return
        component_key = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not component_key or not self._has_component_trace(component_key):
            return
        initial_color = QColor(
            self._component_color_lookup.get(component_key, "#1f77b4")
        )
        chosen = QColorDialog.getColor(
            initial_color,
            self,
            f"Choose color for {component_key}",
        )
        if not chosen.isValid():
            return
        self._set_component_trace_color(component_key, chosen.name())

    def _set_component_trace_color(
        self,
        component_key: str,
        color: str,
    ) -> None:
        self._component_color_overrides[component_key] = color
        line = self._component_line_lookup.get(component_key)
        if line is not None:
            line.set_color(color)
        self._component_color_lookup[component_key] = color
        legend_line = self._component_legend_lookup.get(component_key)
        if legend_line is not None and hasattr(legend_line, "set_color"):
            legend_line.set_color(color)
        self._update_component_table_visuals()
        self.component_canvas.draw_idle()
        self._redraw_prior_preview_if_needed()

    def _toggle_all_component_traces(self) -> None:
        if not self._component_paths:
            return
        component_keys = [path.stem for path in self._component_paths]
        any_visible = any(
            self._component_visibility.get(component_key, True)
            for component_key in component_keys
        )
        target_visible = not any_visible
        for component_key in component_keys:
            self._component_visibility[component_key] = target_visible
            line = self._component_line_lookup.get(component_key)
            if line is not None:
                line.set_visible(target_visible)
            legend_line = self._component_legend_lookup.get(component_key)
            if legend_line is not None and hasattr(legend_line, "set_alpha"):
                legend_line.set_alpha(1.0 if target_visible else 0.25)
        self._update_component_table_visuals()
        self._update_component_trace_control_state()
        self._refresh_component_axes()
        self.component_canvas.draw_idle()

    def _update_component_trace_control_state(self) -> None:
        has_components = bool(self._component_paths)
        component_keys = (
            [path.stem for path in self._component_paths]
            if self._component_paths
            else []
        )
        any_visible = any(
            self._component_visibility.get(component_key, True)
            for component_key in component_keys
        )
        self.component_all_traces_button.setEnabled(has_components)
        self.component_trace_color_scheme_combo.setEnabled(has_components)
        self.component_all_traces_button.setText(
            "Hide Computed Traces" if any_visible else "Show Computed Traces"
        )
        self._update_prior_control_state()

    def _build_visibility_table_item(
        self,
        component_key: str,
    ) -> QTableWidgetItem:
        has_component = self._has_component_trace(component_key)
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, component_key)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if has_component:
            item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if self._component_visibility.get(component_key, True)
                else Qt.CheckState.Unchecked
            )
        else:
            item.setFlags(
                Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            )
            item.setText("--")
        return item

    def _build_color_table_item(
        self,
        component_key: str,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, component_key)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        color = (
            self._component_color_lookup.get(component_key)
            if self._has_component_trace(component_key)
            else None
        )
        if color:
            item.setText(color)
            item.setBackground(QColor(color))
            item.setToolTip("Click to choose a custom trace color.")
        else:
            item.setText("--")
        return item

    def _update_component_table_visuals(self) -> None:
        if not self._component_row_lookup:
            return
        self._updating_cluster_table = True
        self.recognized_clusters_table.blockSignals(True)
        try:
            for component_key, row_index in self._component_row_lookup.items():
                self.recognized_clusters_table.setItem(
                    row_index,
                    6,
                    self._build_visibility_table_item(component_key),
                )
                self.recognized_clusters_table.setItem(
                    row_index,
                    7,
                    self._build_color_table_item(component_key),
                )
        finally:
            self.recognized_clusters_table.blockSignals(False)
            self._updating_cluster_table = False

    def _refresh_component_axes(self) -> None:
        axes = self.component_figure.axes
        if not axes:
            return
        for axis in axes:
            try:
                axis.relim(visible_only=True)
                axis.autoscale_view()
            except Exception:
                continue
        if len(axes) == 2:
            experimental_axis, component_axis = axes
            if self.component_model_range_button.isChecked():
                self._autoscale_to_model_range(
                    experimental_axis,
                    component_axis,
                )
            else:
                self._normalize_component_axis(
                    experimental_axis,
                    component_axis,
                )
        elif self.component_model_range_button.isChecked():
            self._autoscale_to_model_range(None, axes[0])

    def _autoscale_to_model_range(
        self,
        experimental_axis,
        component_axis,
    ) -> None:
        component_lines = [
            line
            for line in self._component_line_lookup.values()
            if line.get_visible()
        ]
        if not component_lines:
            return
        model_q_values = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in component_lines
            ]
        )
        model_q_values = model_q_values[np.isfinite(model_q_values)]
        if model_q_values.size == 0:
            return
        q_min = float(np.nanmin(model_q_values))
        q_max = float(np.nanmax(model_q_values))
        component_axis.set_xlim(q_min, q_max)
        if experimental_axis is not None:
            experimental_axis.set_xlim(q_min, q_max)
            self._autoscale_axis_y(experimental_axis, q_min, q_max)
            self._normalize_component_axis(
                experimental_axis,
                component_axis,
            )
            return
        self._autoscale_axis_y(component_axis, q_min, q_max)

    def _autoscale_axis_y(
        self,
        axis,
        q_min: float,
        q_max: float,
    ) -> None:
        y_segments: list[np.ndarray] = []
        log_scale = self.component_log_y_checkbox.isChecked()
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

    def _has_component_trace(self, component_key: str) -> bool:
        if component_key in self._component_line_lookup:
            return True
        if self._component_paths is None:
            return False
        return any(
            path.stem == component_key for path in self._component_paths
        )

    @staticmethod
    def _component_key(structure: str, motif: str) -> str:
        return f"{structure}_{motif}".replace("/", "_")

    def _selected_q_mask(self, q_values: np.ndarray) -> np.ndarray | None:
        q_min, q_max = self._preview_q_range()
        if q_min is None and q_max is None:
            return np.ones_like(q_values, dtype=bool)
        lower = q_min if q_min is not None else float(q_values.min())
        upper = q_max if q_max is not None else float(q_values.max())
        if lower > upper:
            return np.zeros_like(q_values, dtype=bool)
        if self.use_experimental_grid():
            start_index = int(np.argmin(np.abs(q_values - lower)))
            end_index = int(np.argmin(np.abs(q_values - upper)))
            lo_index, hi_index = sorted((start_index, end_index))
            mask = np.zeros_like(q_values, dtype=bool)
            mask[lo_index : hi_index + 1] = True
            return mask
        return (q_values >= lower) & (q_values <= upper)

    def _preview_q_range(self) -> tuple[float | None, float | None]:
        return (
            self._safe_optional_float(self.qmin_edit.text()),
            self._safe_optional_float(self.qmax_edit.text()),
        )

    @staticmethod
    def _aligned_y_limits(
        left_limits: tuple[float, float],
        experimental_min: float,
        experimental_max: float,
        component_min: float,
        component_max: float,
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
                    component_min,
                    component_max,
                )
                <= 0.0
            ):
                log_scale = False
        if not log_scale:
            left_low, left_high = left_limits
            exp_low, exp_high = sorted((experimental_min, experimental_max))
            comp_low, comp_high = sorted((component_min, component_max))
            if np.isclose(left_high, left_low) or np.isclose(
                exp_high, exp_low
            ):
                padding = max(abs(comp_low) * 0.1, 1e-12)
                return comp_low - padding, comp_high + padding
            p0 = (exp_low - left_low) / (left_high - left_low)
            p1 = (exp_high - left_low) / (left_high - left_low)
            if np.isclose(p1, p0):
                padding = max(abs(comp_low) * 0.1, 1e-12)
                return comp_low - padding, comp_high + padding
            delta = (comp_high - comp_low) / (p1 - p0)
            right_low = comp_low - p0 * delta
            right_high = right_low + delta
            return right_low, right_high

        left_logs = np.log10(np.asarray(left_limits, dtype=float))
        exp_logs = np.log10(
            np.asarray(
                sorted((experimental_min, experimental_max)), dtype=float
            )
        )
        comp_logs = np.log10(
            np.asarray(sorted((component_min, component_max)), dtype=float)
        )
        if np.isclose(left_logs[1], left_logs[0]) or np.isclose(
            exp_logs[1],
            exp_logs[0],
        ):
            return component_min / 1.2, component_max * 1.2
        p0 = (exp_logs[0] - left_logs[0]) / (left_logs[1] - left_logs[0])
        p1 = (exp_logs[1] - left_logs[0]) / (left_logs[1] - left_logs[0])
        if np.isclose(p1, p0):
            return component_min / 1.2, component_max * 1.2
        delta = (comp_logs[1] - comp_logs[0]) / (p1 - p0)
        right_low_log = comp_logs[0] - p0 * delta
        right_high_log = right_low_log + delta
        return 10**right_low_log, 10**right_high_log

    def _experimental_import_summary(
        self,
        summary: ExperimentalDataSummary,
    ) -> str:
        errors_label = (
            "with uncertainties"
            if summary.errors is not None
            else "without uncertainties"
        )
        q_label = self._summary_column_label(summary, summary.q_column)
        intensity_label = self._summary_column_label(
            summary,
            summary.intensity_column,
        )
        column_text = f"Columns: q={q_label}, intensity={intensity_label}"
        if summary.error_column is not None:
            error_label = self._summary_column_label(
                summary,
                summary.error_column,
            )
            column_text += f", error={error_label}"
        return (
            f"Loaded: {summary.path.name}\n"
            f"Points: {len(summary.q_values)} ({errors_label})\n"
            f"Skipped header rows: {summary.header_rows}\n"
            f"q-range: {float(summary.q_values.min()):.6g} to "
            f"{float(summary.q_values.max()):.6g}\n"
            f"{column_text}"
        )

    @staticmethod
    def _structure_atom_weight(structure: str) -> int:
        return max(
            sum(int(token) for token in re.findall(r"(\d+)", structure)), 1
        )

    def _add_selected_elements_to_exclude(self) -> None:
        updated = set(self.exclude_elements())
        for item in self.available_elements_list.selectedItems():
            updated.add(item.text().strip())
        self.exclude_elements_edit.setText(" ".join(sorted(updated)))

    def _remove_selected_elements_from_exclude(self) -> None:
        updated = set(self.exclude_elements())
        for item in self.available_elements_list.selectedItems():
            updated.discard(item.text().strip())
        self.exclude_elements_edit.setText(" ".join(sorted(updated)))

    def _sync_available_element_selection(self) -> None:
        excluded = set(self.exclude_elements())
        self.available_elements_list.blockSignals(True)
        for index in range(self.available_elements_list.count()):
            item = self.available_elements_list.item(index)
            item.setSelected(item.text().strip() in excluded)
        self.available_elements_list.blockSignals(False)

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

    @staticmethod
    def _first_experimental_candidate(directory: Path) -> Path | None:
        for pattern in ("exp_*", "*.dat", "*.txt", "*.iq"):
            for candidate in sorted(directory.glob(pattern)):
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _first_solvent_candidate(directory: Path) -> Path | None:
        for pattern in ("solv_*", "*.dat", "*.txt", "*.iq"):
            for candidate in sorted(directory.glob(pattern)):
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _parse_elements(text: str) -> list[str]:
        return [
            token.strip()
            for token in text.replace(",", " ").split()
            if token.strip()
        ]

    @staticmethod
    def _optional_float(text: str) -> float | None:
        stripped = text.strip()
        return float(stripped) if stripped else None

    @staticmethod
    def _safe_optional_float(text: str) -> float | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    @staticmethod
    def _format_truncated_decimal(value: float, places: int = 3) -> str:
        scale = 10**places
        truncated = np.trunc(float(value) * scale) / scale
        return f"{truncated:.{places}f}"

    def _on_template_combo_changed(self) -> None:
        self._update_active_template_field()
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

    @staticmethod
    def _natural_sort_key(value: str) -> list[object]:
        return [
            int(token) if token.isdigit() else token.lower()
            for token in re.split(r"(\d+)", value)
            if token
        ]
