from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.contrast.descriptors import ContrastStructureDescriptor
from saxshell.saxs.contrast.electron_density import (
    CONTRAST_SOLVENT_METHOD_DIRECT,
    CONTRAST_SOLVENT_METHOD_NEAT,
    CONTRAST_SOLVENT_METHOD_REFERENCE,
    ContrastGeometryDensityResult,
    ContrastGeometryDensitySettings,
    ContrastSolventDensitySettings,
    compute_contrast_geometry_and_electron_density,
)
from saxshell.saxs.contrast.representatives import (
    ContrastRepresentativeBinResult,
    ContrastRepresentativeCandidate,
    ContrastRepresentativeIssue,
    ContrastRepresentativeSelectionResult,
    ContrastRepresentativeTargetSummary,
    analyze_contrast_representatives,
)
from saxshell.saxs.contrast.settings import (
    COMPONENT_BUILD_MODE_CONTRAST,
    ContrastModeLaunchContext,
    ContrastRepresentativeSamplerSettings,
    component_build_mode_label,
)
from saxshell.saxs.contrast.solvents import (
    ContrastSolventPreset,
    delete_custom_solvent_preset,
    load_solvent_presets,
    ordered_solvent_preset_names,
    save_custom_solvent_preset,
)
from saxshell.saxs.contrast.ui.structure_viewer import (
    ContrastRepresentativeViewer,
    load_contrast_representative_preview,
)
from saxshell.saxs.contrast.workflow import build_contrast_workflow_preview
from saxshell.saxs.debye import ClusterBin, discover_cluster_bins
from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
    load_experimental_data_file,
    project_artifact_paths,
)
from saxshell.saxs.stoichiometry import parse_stoich_label
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_OPEN_WINDOWS: list["ContrastModeMainWindow"] = []
_TRACE_COLOR_SCHEME_OPTIONS = (
    "Current",
    "summer",
    "viridis",
    "plasma",
    "cividis",
    "Greens",
    "Blues",
    "magma",
)
_REPRESENTATIVE_ROW_METADATA_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_UI_REFRESH_DELAY_MS = 225
_WORKFLOW_STAGE_TOTAL = 5
_REPRESENTATIVE_COLUMN_STOICHIOMETRY = 0
_REPRESENTATIVE_COLUMN_SELECTION = 1
_REPRESENTATIVE_COLUMN_FILE = 2
_REPRESENTATIVE_COLUMN_COLOR = 3
_REPRESENTATIVE_COLUMN_SOLVENT = 4
_REPRESENTATIVE_COLUMN_MESH = 5
_REPRESENTATIVE_COLUMN_DENSITY = 6
_REPRESENTATIVE_COLUMN_TRACE_STATUS = 7
_REPRESENTATIVE_COLUMN_NOTES = 8


def _normalized_hex_color(value: object, *, fallback: str = "#1f77b4") -> str:
    candidate = QColor(str(value or "").strip())
    if candidate.isValid():
        return candidate.name()
    return QColor(fallback).name()


def _color_text_hex(color_value: str) -> str:
    color = QColor(color_value)
    if not color.isValid():
        return "#000000"
    luminance = (
        0.299 * color.redF() + 0.587 * color.greenF() + 0.114 * color.blueF()
    )
    return "#111111" if luminance >= 0.58 else "#f9fafb"


def _color_display_value(color_value: str) -> str:
    return _normalized_hex_color(color_value).upper()


class _ContrastWorkflowWorker(QObject):
    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        task: Callable[..., object],
    ) -> None:
        super().__init__()
        self._task = task

    @Slot()
    def run(self) -> None:
        try:
            result = self._task(
                progress_callback=self._emit_progress,
                log_callback=self._emit_log,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(
            max(int(processed), 0),
            max(int(total), 1),
            str(message).strip(),
        )

    def _emit_log(self, message: str) -> None:
        text = str(message).strip()
        if text:
            self.log.emit(text)


class ContrastModeMainWindow(QMainWindow):
    """Dedicated contrast-mode workspace scaffold."""

    project_paths_registered = Signal(object)
    contrast_components_built = Signal(object)

    def __init__(
        self,
        *,
        initial_project_dir: Path | None = None,
        initial_clusters_dir: Path | None = None,
        initial_experimental_data_file: Path | None = None,
        initial_q_min: float | None = None,
        initial_q_max: float | None = None,
        initial_template_name: str | None = None,
        initial_distribution_id: str | None = None,
        initial_distribution_root_dir: Path | None = None,
        initial_contrast_artifact_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._launch_context = ContrastModeLaunchContext.from_values(
            project_dir=initial_project_dir,
            clusters_dir=initial_clusters_dir,
            experimental_data_file=initial_experimental_data_file,
            q_min=initial_q_min,
            q_max=initial_q_max,
            active_template_name=initial_template_name,
            active_distribution_id=initial_distribution_id,
            distribution_root_dir=initial_distribution_root_dir,
            contrast_artifact_dir=initial_contrast_artifact_dir,
        )
        self._recognized_cluster_bins: list[ClusterBin] = []
        self._experimental_summary: ExperimentalDataSummary | None = None
        self._representative_analysis_result: (
            ContrastRepresentativeSelectionResult | None
        ) = None
        self._density_result: ContrastGeometryDensityResult | None = None
        self._generated_trace_profiles: list[
            tuple[str, np.ndarray, np.ndarray]
        ] = []
        self._generated_traces_visible = True
        self._active_distribution_id: str | None = None
        self._distribution_root_dir: Path | None = None
        self._contrast_artifact_dir_override: Path | None = None
        self._last_build_distribution_id: str | None = None
        self._workflow_thread: QThread | None = None
        self._workflow_worker: _ContrastWorkflowWorker | None = None
        self._active_workflow_task: str | None = None
        self._workflow_completion_handler: Callable[[object], None] | None = (
            None
        )
        self._pending_launch_context_refresh = False
        self._pending_saved_distribution_restore = False
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.setInterval(_UI_REFRESH_DELAY_MS)
        self._preview_refresh_timer.timeout.connect(
            self._flush_scheduled_preview_refresh
        )
        self._app_event_filter_installed = False
        self._solvent_presets: dict[str, ContrastSolventPreset] = {}
        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_event_filter_installed = True
        self.apply_launch_context(self._launch_context)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (Contrast Debye Workflow)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1600, 980)

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        intro = QLabel(
            "This dedicated workspace hosts the future contrast-enabled Debye "
            "workflow. The existing no-contrast SAXS component builder remains "
            "unchanged while the contrast pipeline is developed in separate "
            "modules and tested here."
        )
        intro.setWordWrap(True)
        intro.setFrameShape(QFrame.Shape.StyledPanel)
        root_layout.addWidget(intro)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.left_scroll_area = self._build_left_pane()
        self.right_scroll_area = self._build_right_pane()
        self.main_splitter.addWidget(self.left_scroll_area)
        self.main_splitter.addWidget(self.right_scroll_area)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([620, 960])
        root_layout.addWidget(self.main_splitter, stretch=1)

        self.setCentralWidget(central)
        self._reload_solvent_presets(selected_name="Water")
        self._sync_density_method_controls()
        self._update_trace_control_state()
        self.statusBar().showMessage("Contrast-mode workspace ready")

    def _build_left_pane(self) -> QScrollArea:
        content = QWidget()
        content.setMinimumWidth(580)
        content.setMinimumHeight(1380)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_context_group())
        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_cluster_table_group())
        layout.addWidget(self._build_representative_table_group())
        layout.addWidget(self._build_controls_group())
        layout.addWidget(self._build_console_group())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_right_pane(self) -> QScrollArea:
        content = QWidget()
        content.setMinimumWidth(820)
        content.setMinimumHeight(980)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.addWidget(self._build_trace_plot_group())
        self.right_splitter.addWidget(self._build_visualizer_group())
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 2)
        self.right_splitter.setSizes([560, 380])
        layout.addWidget(self.right_splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_context_group(self) -> QGroupBox:
        group = QGroupBox("Project Context")
        layout = QVBoxLayout(group)

        self.context_status_label = QLabel()
        self.context_status_label.setWordWrap(True)
        layout.addWidget(self.context_status_label)

        form = QFormLayout()
        self.mode_edit = self._build_readonly_line_edit()
        self.project_dir_edit = self._build_readonly_line_edit()
        self.template_edit = self._build_readonly_line_edit()
        self.q_range_edit = self._build_readonly_line_edit()
        form.addRow("Build mode", self.mode_edit)
        form.addRow("Project folder", self.project_dir_edit)
        form.addRow("Active template", self.template_edit)
        form.addRow("Current q-range", self.q_range_edit)
        layout.addLayout(form)

        self.workflow_summary_box = QPlainTextEdit()
        self.workflow_summary_box.setReadOnly(True)
        self.workflow_summary_box.setMinimumHeight(170)
        layout.addWidget(self.workflow_summary_box)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Inputs")
        layout = QFormLayout(group)

        self.experimental_data_edit = QLineEdit()
        self.experimental_data_edit.setPlaceholderText(
            "Seeded from the active SAXS project, or choose an override file"
        )
        self.experimental_data_edit.editingFinished.connect(
            self._schedule_preview_refresh
        )
        layout.addRow(
            "Experimental data",
            self._build_path_input_row(
                self.experimental_data_edit,
                browse_slot=self._choose_experimental_data_file,
            ),
        )

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.setPlaceholderText(
            "Inherited cluster folder or choose a contrast-mode reference folder"
        )
        self.clusters_dir_edit.editingFinished.connect(
            self._schedule_preview_refresh
        )
        layout.addRow(
            "Reference clusters folder",
            self._build_path_input_row(
                self.clusters_dir_edit,
                browse_slot=self._choose_clusters_directory,
            ),
        )

        self.q_min_spin = self._build_q_range_spinbox()
        self.q_max_spin = self._build_q_range_spinbox()
        self.q_min_spin.valueChanged.connect(self._on_q_range_controls_changed)
        self.q_max_spin.valueChanged.connect(self._on_q_range_controls_changed)
        q_row = QWidget()
        q_layout = QHBoxLayout(q_row)
        q_layout.setContentsMargins(0, 0, 0, 0)
        q_layout.setSpacing(6)
        q_layout.addWidget(QLabel("q min"))
        q_layout.addWidget(self.q_min_spin)
        q_layout.addWidget(QLabel("q max"))
        q_layout.addWidget(self.q_max_spin)
        q_layout.addStretch(1)
        layout.addRow("Scattering q-range", q_row)

        self.input_hint_label = QLabel(
            "These values are seeded from the main SAXS window when the "
            "contrast workflow opens. You can then edit the experimental file, "
            "q-range, and reference clusters folder here without changing the "
            "main UI."
        )
        self.input_hint_label.setWordWrap(True)
        layout.addRow("", self.input_hint_label)
        return group

    def _build_cluster_table_group(self) -> QGroupBox:
        group = QGroupBox("Recognized Clusters")
        layout = QVBoxLayout(group)

        self.cluster_table_status_label = QLabel(
            "Point the reference clusters folder at a valid extracted-clusters "
            "directory to inspect the recognized stoichiometry bins."
        )
        self.cluster_table_status_label.setWordWrap(True)
        layout.addWidget(self.cluster_table_status_label)

        self.recognized_clusters_table = QTableWidget(0, 7)
        self.recognized_clusters_table.setHorizontalHeaderLabels(
            [
                "Cluster",
                "Stoichiometry",
                "Type",
                "Count",
                "Weight",
                "Atom %",
                "Representative",
            ]
        )
        self.recognized_clusters_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.recognized_clusters_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.recognized_clusters_table.verticalHeader().setVisible(False)
        header = self.recognized_clusters_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for column in (3, 4, 5, 6):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.recognized_clusters_table.setMinimumHeight(220)
        layout.addWidget(self.recognized_clusters_table)
        return group

    def _build_representative_table_group(self) -> QGroupBox:
        group = QGroupBox("Representative Structures")
        layout = QVBoxLayout(group)

        top_controls = QHBoxLayout()
        self.analyze_representatives_button = QPushButton(
            "Analyze Representative Structures"
        )
        self.analyze_representatives_button.clicked.connect(
            self._run_representative_analysis
        )
        top_controls.addWidget(self.analyze_representatives_button)
        top_controls.addStretch(1)
        layout.addLayout(top_controls)

        advanced_header = QHBoxLayout()
        self.sampler_settings_toggle_button = QToolButton()
        self.sampler_settings_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.sampler_settings_toggle_button.setAutoRaise(True)
        self.sampler_settings_toggle_button.clicked.connect(
            self._toggle_sampler_settings_collapsed
        )
        advanced_header.addWidget(self.sampler_settings_toggle_button)
        self.sampler_settings_status_label = QLabel(
            "For large cluster bins, a fast geometry/count screen inspired by "
            "Cluster Dynamics ML and bond-analysis preprocessing can limit the "
            "full contrast descriptor search to a seeded Monte Carlo shortlist."
        )
        self.sampler_settings_status_label.setWordWrap(True)
        advanced_header.addWidget(
            self.sampler_settings_status_label,
            stretch=1,
        )
        layout.addLayout(advanced_header)

        self.sampler_settings_widget = self._build_sampler_settings_widget()
        layout.addWidget(self.sampler_settings_widget)
        self.set_sampler_settings_collapsed(True)

        self.representative_table_status_label = QLabel(
            "Representative structures appear here after the screening step. "
            "Manual custom files can also be staged below, but any changes to "
            "the representative list will require the downstream contrast "
            "workflow to be rerun."
        )
        self.representative_table_status_label.setWordWrap(True)
        layout.addWidget(self.representative_table_status_label)

        self.representative_table = QTableWidget(0, 9)
        self.representative_table.setHorizontalHeaderLabels(
            [
                "Stoichiometry",
                "Selection",
                "Representative File",
                "Trace Color",
                "Solvent Coord.",
                "Mesh Volume",
                "Electron Density",
                "Trace Status",
                "Notes",
            ]
        )
        self.representative_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.representative_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.representative_table.verticalHeader().setVisible(False)
        self.representative_table.itemSelectionChanged.connect(
            self._sync_visualizer_preview
        )
        self.representative_table.cellClicked.connect(
            self._on_representative_table_cell_clicked
        )
        header = self.representative_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for column in (3, 4, 5, 6, 7, 8):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.representative_table.setMinimumHeight(220)
        layout.addWidget(self.representative_table)

        controls = QHBoxLayout()
        self.add_representative_button = QPushButton(
            "Add Representative (Custom)…"
        )
        self.add_representative_button.clicked.connect(
            self._add_manual_representatives
        )
        controls.addWidget(self.add_representative_button)
        self.remove_representative_button = QPushButton("Remove Selected")
        self.remove_representative_button.clicked.connect(
            self._remove_selected_representative
        )
        controls.addWidget(self.remove_representative_button)
        self.inspect_representative_button = QPushButton("Inspect Selected")
        self.inspect_representative_button.clicked.connect(
            self._inspect_selected_representative
        )
        controls.addWidget(self.inspect_representative_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        return group

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Electron Density and Build")
        layout = QVBoxLayout(group)

        density_group = QGroupBox("Electron Density Options")
        density_layout = QFormLayout(density_group)

        self.solvent_method_combo = QComboBox()
        self.solvent_method_combo.addItem(
            "Estimate from Solvent Formula and Density",
            userData=CONTRAST_SOLVENT_METHOD_NEAT,
        )
        self.solvent_method_combo.addItem(
            "Reference Solvent Structure (XYZ/PDB)",
            userData=CONTRAST_SOLVENT_METHOD_REFERENCE,
        )
        self.solvent_method_combo.addItem(
            "Direct Electron Density Value",
            userData=CONTRAST_SOLVENT_METHOD_DIRECT,
        )
        self.solvent_method_combo.currentIndexChanged.connect(
            self._sync_density_method_controls
        )
        density_layout.addRow("Compute option", self.solvent_method_combo)

        self.solvent_preset_combo = QComboBox()
        self.solvent_preset_combo.currentIndexChanged.connect(
            self._load_selected_solvent_preset
        )
        self.save_custom_solvent_button = QPushButton("Save Custom Solvent")
        self.save_custom_solvent_button.clicked.connect(
            self._save_current_solvent_preset
        )
        self.delete_custom_solvent_button = QPushButton(
            "Delete Custom Solvent"
        )
        self.delete_custom_solvent_button.clicked.connect(
            self._delete_current_solvent_preset
        )
        solvent_preset_row = QWidget()
        solvent_preset_layout = QHBoxLayout(solvent_preset_row)
        solvent_preset_layout.setContentsMargins(0, 0, 0, 0)
        solvent_preset_layout.setSpacing(6)
        solvent_preset_layout.addWidget(self.solvent_preset_combo, stretch=1)
        solvent_preset_layout.addWidget(self.save_custom_solvent_button)
        solvent_preset_layout.addWidget(self.delete_custom_solvent_button)
        density_layout.addRow("Saved solvents", solvent_preset_row)

        self.solvent_formula_edit = QLineEdit()
        self.solvent_formula_edit.setPlaceholderText(
            "Examples: H2O, Vacuum, C3H7NO (DMF), C2H6OS (DMSO)"
        )
        density_layout.addRow("Solvent formula", self.solvent_formula_edit)

        self.solvent_density_spin = QDoubleSpinBox()
        self.solvent_density_spin.setDecimals(6)
        self.solvent_density_spin.setRange(0.0, 100.0)
        self.solvent_density_spin.setSingleStep(0.01)
        self.solvent_density_spin.setValue(1.0)
        density_layout.addRow("Density (g/mL)", self.solvent_density_spin)

        self.exclude_hydrogen_checkbox = QCheckBox(
            "Exclude hydrogen in mesh and Debye"
        )
        self.exclude_hydrogen_checkbox.setToolTip(
            "Exclude hydrogen atoms consistently from the retained-volume mesh, "
            "electron-density calculation, reference-solvent sampling, and "
            "contrast Debye traces."
        )
        density_layout.addRow("", self.exclude_hydrogen_checkbox)

        self.direct_density_spin = QDoubleSpinBox()
        self.direct_density_spin.setDecimals(6)
        self.direct_density_spin.setRange(0.0, 100.0)
        self.direct_density_spin.setSingleStep(0.001)
        self.direct_density_spin.setValue(0.334)
        density_layout.addRow(
            "Direct density (e/A^3)",
            self.direct_density_spin,
        )

        self.reference_solvent_file_edit = QLineEdit()
        self.reference_solvent_file_edit.setPlaceholderText(
            "Choose a reference solvent XYZ or PDB file"
        )
        reference_row = QWidget()
        reference_layout = QHBoxLayout(reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(6)
        reference_layout.addWidget(self.reference_solvent_file_edit, stretch=1)
        self.reference_solvent_browse_button = QPushButton("Browse…")
        self.reference_solvent_browse_button.clicked.connect(
            self._choose_reference_solvent_file
        )
        reference_layout.addWidget(self.reference_solvent_browse_button)
        density_layout.addRow("Reference solvent file", reference_row)

        self.solvent_method_hint_label = QLabel()
        self.solvent_method_hint_label.setWordWrap(True)
        density_layout.addRow("", self.solvent_method_hint_label)
        layout.addWidget(density_group)

        action_row = QHBoxLayout()
        self.compute_density_button = QPushButton("Compute Electron Density")
        self.compute_density_button.setToolTip(
            "Build the retained mesh and electron-density contrast terms for "
            "the current representative structure list."
        )
        self.compute_density_button.clicked.connect(
            self._compute_electron_density
        )
        action_row.addWidget(self.compute_density_button)
        self.build_components_button = QPushButton(
            "Build Contrast SAXS Components"
        )
        self.build_components_button.setToolTip(
            "Build contrast SAXS components from the active representative "
            "structures after electron density has been computed."
        )
        self.build_components_button.clicked.connect(
            self._build_contrast_components
        )
        action_row.addWidget(self.build_components_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.workflow_progress_label = QLabel("Progress: scaffold idle")
        layout.addWidget(self.workflow_progress_label)
        self.workflow_progress_bar = QProgressBar()
        self.workflow_progress_bar.setRange(0, 5)
        self.workflow_progress_bar.setValue(1)
        self.workflow_progress_bar.setFormat("%v / %m workflow stages")
        layout.addWidget(self.workflow_progress_bar)

        layout.addWidget(QLabel("Saved output paths"))
        self.output_path_edit = self._build_readonly_line_edit()
        workflow_form = QFormLayout()
        workflow_form.setContentsMargins(0, 0, 0, 0)
        workflow_form.setSpacing(6)
        workflow_form.addRow("Workflow folder", self.output_path_edit)
        layout.addLayout(workflow_form)

        self.representatives_output_edit = self._build_readonly_line_edit()
        self.screening_output_edit = self._build_readonly_line_edit()
        self.summary_output_edit = self._build_readonly_line_edit()
        output_form = QFormLayout()
        output_form.setContentsMargins(0, 0, 0, 0)
        output_form.setSpacing(6)
        self.refresh_context_button = QPushButton("Refresh Preview")
        self.refresh_context_button.clicked.connect(self.refresh_preview)
        self.log_context_button = QPushButton("Log Context")
        self.log_context_button.clicked.connect(self.log_current_context)
        output_form.addRow(
            "Representative folder",
            self._build_output_path_row(
                self.representatives_output_edit,
                action_button=self.refresh_context_button,
                help_text=(
                    "Refresh the recognized-cluster table, experimental preview, "
                    "and current path summaries from the fields above."
                ),
            ),
        )
        output_form.addRow(
            "Screening folder",
            self._build_output_path_row(
                self.screening_output_edit,
                action_button=self.log_context_button,
                help_text=(
                    "Write the current contrast workflow context into the output "
                    "console without starting any new calculation."
                ),
            ),
        )
        output_form.addRow(
            "Summary file",
            self._build_output_path_row(
                self.summary_output_edit,
                help_text=(
                    "This summary file tracks the current representative-selection "
                    "outputs for the contrast workflow."
                ),
            ),
        )
        layout.addLayout(output_form)
        return group

    def _build_console_group(self) -> QGroupBox:
        group = QGroupBox("Contrast Tool Output")
        layout = QVBoxLayout(group)
        self.console_box = QPlainTextEdit()
        self.console_box.setReadOnly(True)
        self.console_box.setMinimumHeight(220)
        self.console_box.setMaximumBlockCount(2000)
        layout.addWidget(self.console_box)
        return group

    def _build_trace_plot_group(self) -> QGroupBox:
        group = QGroupBox("Experimental Data and Contrast Traces")
        group.setMinimumHeight(520)
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.trace_log_x_checkbox = QCheckBox("Log X")
        self.trace_log_x_checkbox.setChecked(True)
        self.trace_log_x_checkbox.toggled.connect(self._redraw_trace_plot)
        controls.addWidget(self.trace_log_x_checkbox)

        self.trace_log_y_checkbox = QCheckBox("Log Y")
        self.trace_log_y_checkbox.setChecked(True)
        self.trace_log_y_checkbox.toggled.connect(self._redraw_trace_plot)
        controls.addWidget(self.trace_log_y_checkbox)

        self.trace_legend_toggle_button = QPushButton("Legend")
        self.trace_legend_toggle_button.setCheckable(True)
        self.trace_legend_toggle_button.setChecked(True)
        self.trace_legend_toggle_button.toggled.connect(
            self._redraw_trace_plot
        )
        controls.addWidget(self.trace_legend_toggle_button)

        self.trace_q_range_button = QPushButton("Autoscale to Model Range")
        self.trace_q_range_button.setCheckable(True)
        self.trace_q_range_button.toggled.connect(self._redraw_trace_plot)
        controls.addWidget(self.trace_q_range_button)

        self.trace_generated_toggle_button = QPushButton(
            "Hide Computed Traces"
        )
        self.trace_generated_toggle_button.setEnabled(False)
        self.trace_generated_toggle_button.clicked.connect(
            self._toggle_generated_traces
        )
        controls.addWidget(self.trace_generated_toggle_button)

        controls.addWidget(QLabel("Trace Colors"))
        self.trace_color_scheme_combo = QComboBox()
        self.trace_color_scheme_combo.addItem("Current", userData="default")
        for option in _TRACE_COLOR_SCHEME_OPTIONS[1:]:
            self.trace_color_scheme_combo.addItem(option, userData=option)
        self.trace_color_scheme_combo.currentIndexChanged.connect(
            self._redraw_trace_plot
        )
        controls.addWidget(self.trace_color_scheme_combo)

        self.export_plot_data_button = QPushButton("Export Plot Data")
        self.export_plot_data_button.clicked.connect(self._export_plot_data)
        controls.addWidget(self.export_plot_data_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.trace_plot_status_label = QLabel(
            "No experimental data preview is loaded yet."
        )
        self.trace_plot_status_label.setWordWrap(True)
        layout.addWidget(self.trace_plot_status_label)

        self.trace_figure = Figure(figsize=(8.2, 4.8))
        self.trace_canvas = FigureCanvasQTAgg(self.trace_figure)
        self.trace_toolbar = NavigationToolbar2QT(self.trace_canvas, self)
        layout.addWidget(self.trace_toolbar)
        layout.addWidget(self.trace_canvas, stretch=1)
        return group

    def _build_visualizer_group(self) -> QGroupBox:
        group = QGroupBox("Representative Structure and Volume Preview")
        group.setMinimumHeight(560)
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.rotate_left_button = QPushButton("Rotate Left")
        self.rotate_left_button.clicked.connect(
            lambda: self._rotate_visualizer(-18.0, "Rotate Left")
        )
        controls.addWidget(self.rotate_left_button)
        self.rotate_right_button = QPushButton("Rotate Right")
        self.rotate_right_button.clicked.connect(
            lambda: self._rotate_visualizer(18.0, "Rotate Right")
        )
        controls.addWidget(self.rotate_right_button)
        self.pan_button = QPushButton("Pan")
        self.pan_button.setCheckable(True)
        self.pan_button.clicked.connect(self._toggle_visualizer_pan_mode)
        controls.addWidget(self.pan_button)
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(
            lambda: self._zoom_visualizer(0.86, "Zoomed in")
        )
        controls.addWidget(self.zoom_in_button)
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(
            lambda: self._zoom_visualizer(1.18, "Zoomed out")
        )
        controls.addWidget(self.zoom_out_button)
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self._reset_visualizer_view)
        controls.addWidget(self.reset_view_button)
        self.show_mesh_checkbox = QCheckBox("Show Mesh")
        self.show_mesh_checkbox.setChecked(True)
        self.show_mesh_checkbox.toggled.connect(self._sync_visualizer_preview)
        controls.addWidget(self.show_mesh_checkbox)
        self.show_structure_legend_checkbox = QCheckBox("Legend")
        self.show_structure_legend_checkbox.setChecked(True)
        self.show_structure_legend_checkbox.toggled.connect(
            self._sync_visualizer_preview
        )
        controls.addWidget(self.show_structure_legend_checkbox)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.visualizer_status_label = QLabel(
            "Select a representative structure row to preview its saved structure, "
            "retained mesh overlay, and electron-density metadata."
        )
        self.visualizer_status_label.setWordWrap(True)
        layout.addWidget(self.visualizer_status_label)

        self.structure_viewer = ContrastRepresentativeViewer(self)
        self.structure_viewer.setMinimumHeight(430)
        layout.addWidget(self.structure_viewer, stretch=1)
        self.visualizer_details_box = QPlainTextEdit()
        self.visualizer_details_box.setReadOnly(True)
        self.visualizer_details_box.setMinimumHeight(72)
        self.visualizer_details_box.setMaximumHeight(112)
        layout.addWidget(self.visualizer_details_box)
        return group

    @staticmethod
    def _build_readonly_line_edit() -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        return line_edit

    @staticmethod
    def _build_q_range_spinbox() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0, 1000.0)
        spin.setSingleStep(0.005)
        spin.setAccelerated(True)
        spin.setKeyboardTracking(False)
        spin.setSpecialValueText("Inherited")
        return spin

    def _build_sampler_settings_widget(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setContentsMargins(14, 0, 0, 6)
        layout.setSpacing(6)

        self.sampler_enabled_checkbox = QCheckBox(
            "Use Monte Carlo sampling for large bins"
        )
        self.sampler_enabled_checkbox.setChecked(True)
        layout.addRow("", self.sampler_enabled_checkbox)

        self.sampler_full_scan_threshold_spin = QSpinBox()
        self.sampler_full_scan_threshold_spin.setRange(0, 100_000)
        self.sampler_full_scan_threshold_spin.setValue(24)
        layout.addRow(
            "Full scan threshold",
            self.sampler_full_scan_threshold_spin,
        )

        self.sampler_distribution_samples_spin = QSpinBox()
        self.sampler_distribution_samples_spin.setRange(1, 100_000)
        self.sampler_distribution_samples_spin.setValue(64)
        layout.addRow(
            "Target distribution samples",
            self.sampler_distribution_samples_spin,
        )

        self.sampler_minimum_samples_spin = QSpinBox()
        self.sampler_minimum_samples_spin.setRange(1, 100_000)
        self.sampler_minimum_samples_spin.setValue(8)
        layout.addRow(
            "Minimum candidate samples",
            self.sampler_minimum_samples_spin,
        )

        self.sampler_max_samples_spin = QSpinBox()
        self.sampler_max_samples_spin.setRange(1, 100_000)
        self.sampler_max_samples_spin.setValue(32)
        layout.addRow(
            "Maximum candidate samples",
            self.sampler_max_samples_spin,
        )

        self.sampler_batch_size_spin = QSpinBox()
        self.sampler_batch_size_spin.setRange(1, 10_000)
        self.sampler_batch_size_spin.setValue(4)
        layout.addRow("Candidates per round", self.sampler_batch_size_spin)

        self.sampler_stratify_checkbox = QCheckBox(
            "Disperse samples across frame order"
        )
        self.sampler_stratify_checkbox.setChecked(True)
        layout.addRow("", self.sampler_stratify_checkbox)

        self.sampler_seed_spin = QSpinBox()
        self.sampler_seed_spin.setRange(0, 2_147_483_647)
        self.sampler_seed_spin.setValue(1337)
        layout.addRow("Random seed", self.sampler_seed_spin)

        self.sampler_patience_spin = QSpinBox()
        self.sampler_patience_spin.setRange(1, 10_000)
        self.sampler_patience_spin.setValue(2)
        layout.addRow(
            "Convergence rounds",
            self.sampler_patience_spin,
        )

        self.sampler_tolerance_spin = QDoubleSpinBox()
        self.sampler_tolerance_spin.setDecimals(6)
        self.sampler_tolerance_spin.setRange(0.0, 10.0)
        self.sampler_tolerance_spin.setSingleStep(0.0005)
        self.sampler_tolerance_spin.setValue(0.0025)
        layout.addRow(
            "Improvement tolerance",
            self.sampler_tolerance_spin,
        )

        self.sampler_hint_label = QLabel(
            "The analysis first estimates a fixed median target from a sampled "
            "subset of the full bin distribution, then scores randomly sampled "
            "candidate structures against that fixed target. The score is a "
            "normalized descriptor-distance across bond lengths, bond angles, "
            "coordination counts, and solvent-shell metrics, so lower is "
            "better. The spread option helps avoid picking adjacent frames from "
            "the same local region of the trajectory."
        )
        self.sampler_hint_label.setWordWrap(True)
        layout.addRow("", self.sampler_hint_label)
        return widget

    def set_sampler_settings_collapsed(self, collapsed: bool) -> None:
        self.sampler_settings_widget.setVisible(not collapsed)
        self.sampler_settings_toggle_button.setArrowType(
            Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.DownArrow
        )
        self.sampler_settings_toggle_button.setText(
            "Advanced Sampler Settings"
        )

    def _toggle_sampler_settings_collapsed(self) -> None:
        self.set_sampler_settings_collapsed(
            not self.sampler_settings_widget.isHidden()
        )

    def _representative_sampler_settings(
        self,
    ) -> ContrastRepresentativeSamplerSettings:
        return ContrastRepresentativeSamplerSettings.from_values(
            enabled=self.sampler_enabled_checkbox.isChecked(),
            full_scan_threshold=self.sampler_full_scan_threshold_spin.value(),
            target_distribution_samples=(
                self.sampler_distribution_samples_spin.value()
            ),
            minimum_candidate_samples=(
                self.sampler_minimum_samples_spin.value()
            ),
            max_candidate_samples=self.sampler_max_samples_spin.value(),
            candidate_batch_size=self.sampler_batch_size_spin.value(),
            random_seed=self.sampler_seed_spin.value(),
            convergence_patience=self.sampler_patience_spin.value(),
            improvement_tolerance=self.sampler_tolerance_spin.value(),
            stratify_sampling=self.sampler_stratify_checkbox.isChecked(),
        )

    def _build_path_input_row(
        self,
        line_edit: QLineEdit,
        *,
        browse_slot,
        browse_label: str = "Browse…",
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(line_edit, stretch=1)
        browse_button = QPushButton(browse_label)
        browse_button.clicked.connect(browse_slot)
        layout.addWidget(browse_button)
        return row

    def _build_output_path_row(
        self,
        line_edit: QLineEdit,
        *,
        action_button: QPushButton | None = None,
        help_text: str | None = None,
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(line_edit, stretch=1)
        if action_button is not None:
            layout.addWidget(action_button)
        if help_text:
            help_button = QPushButton("?")
            help_button.setFixedWidth(28)
            help_button.setToolTip(help_text)
            layout.addWidget(help_button)
        return row

    def _selected_solvent_preset_name(self) -> str | None:
        return self.solvent_preset_combo.currentData()

    def _reload_solvent_presets(
        self, *, selected_name: str | None = None
    ) -> None:
        previous_name = selected_name or self._selected_solvent_preset_name()
        self._solvent_presets = load_solvent_presets()
        self.solvent_preset_combo.blockSignals(True)
        self.solvent_preset_combo.clear()
        self.solvent_preset_combo.addItem("Custom entry", None)
        selected_index = 0
        for index, preset_name in enumerate(
            ordered_solvent_preset_names(self._solvent_presets),
            start=1,
        ):
            preset = self._solvent_presets[preset_name]
            label = (
                preset.name if preset.builtin else f"{preset.name} (Custom)"
            )
            self.solvent_preset_combo.addItem(label, preset_name)
            if previous_name == preset_name:
                selected_index = index
        self.solvent_preset_combo.setCurrentIndex(selected_index)
        self.solvent_preset_combo.blockSignals(False)
        self._load_selected_solvent_preset()

    @Slot()
    def _load_selected_solvent_preset(self) -> None:
        preset_name = self._selected_solvent_preset_name()
        preset = self._solvent_presets.get(preset_name or "")
        if preset is None:
            self.delete_custom_solvent_button.setEnabled(False)
            return
        self.solvent_formula_edit.setText(preset.formula)
        self.solvent_density_spin.setValue(preset.density_g_per_ml)
        self.delete_custom_solvent_button.setEnabled(not preset.builtin)

    @Slot()
    def _save_current_solvent_preset(self) -> None:
        suggested_name = self._selected_solvent_preset_name() or ""
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save Custom Solvent",
            "Custom solvent name:",
            text=suggested_name,
        )
        if not accepted:
            return
        name = str(preset_name).strip()
        if not name:
            QMessageBox.warning(
                self,
                "Save Custom Solvent",
                "Enter a solvent name before saving.",
            )
            return
        formula = self.solvent_formula_edit.text().strip()
        density = float(self.solvent_density_spin.value())
        try:
            preset = ContrastSolventPreset(
                name=name,
                formula=formula,
                density_g_per_ml=density,
                builtin=False,
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Save Custom Solvent",
                str(exc),
            )
            return
        if name in self._solvent_presets:
            response = QMessageBox.question(
                self,
                "Overwrite custom solvent?",
                f"A solvent named '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        save_custom_solvent_preset(preset)
        self._reload_solvent_presets(selected_name=name)
        self._append_console_line(f"Saved custom solvent preset: {name}.")

    @Slot()
    def _delete_current_solvent_preset(self) -> None:
        preset_name = self._selected_solvent_preset_name()
        if preset_name is None:
            QMessageBox.information(
                self,
                "Delete Custom Solvent",
                "Select a saved custom solvent first.",
            )
            return
        preset = self._solvent_presets.get(preset_name)
        if preset is None or preset.builtin:
            QMessageBox.information(
                self,
                "Delete Custom Solvent",
                "Only custom solvents can be deleted.",
            )
            return
        response = QMessageBox.question(
            self,
            "Delete Custom Solvent",
            f"Delete the custom solvent preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        delete_custom_solvent_preset(preset_name)
        self._reload_solvent_presets(selected_name="Water")
        self._append_console_line(
            f"Deleted custom solvent preset: {preset_name}."
        )

    @Slot()
    def _sync_density_method_controls(self) -> None:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        using_direct = method == CONTRAST_SOLVENT_METHOD_DIRECT
        using_reference = method == CONTRAST_SOLVENT_METHOD_REFERENCE
        using_neat = method == CONTRAST_SOLVENT_METHOD_NEAT
        for widget in (
            self.solvent_preset_combo,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.save_custom_solvent_button,
            self.delete_custom_solvent_button,
        ):
            widget.setEnabled(using_neat)
        self.reference_solvent_file_edit.setEnabled(using_reference)
        self.reference_solvent_browse_button.setEnabled(using_reference)
        self.direct_density_spin.setEnabled(using_direct)
        if using_reference:
            self.solvent_method_hint_label.setText(
                "Reference structure mode uses a free-solvent XYZ or PDB file. "
                "The file must span a larger box than the largest representative "
                "cluster volume."
            )
        elif using_direct:
            self.solvent_method_hint_label.setText(
                "Direct value mode uses the electron density you provide in "
                "e/A^3 without requiring a solvent structure file or solvent formula. "
                "Use 0.0 e/A^3 to model vacuum."
            )
        else:
            self.solvent_method_hint_label.setText(
                "Quick estimate mode uses the selected solvent stoichiometry and "
                "density. Built-in presets include Water, Vacuum, DMF, and DMSO, "
                "and custom solvents are saved to a reusable JSON file."
            )

    @Slot()
    def _choose_reference_solvent_file(self) -> None:
        start_dir = (
            str(
                Path(self.reference_solvent_file_edit.text())
                .expanduser()
                .resolve()
                .parent
            )
            if self.reference_solvent_file_edit.text().strip()
            else str(self._current_output_dir() or Path.cwd())
        )
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Reference Solvent Structure",
            start_dir,
            "Structure files (*.pdb *.xyz);;All files (*)",
        )
        if not selected_path:
            return
        self.reference_solvent_file_edit.setText(
            str(Path(selected_path).expanduser().resolve())
        )

    def _current_density_settings(self) -> ContrastGeometryDensitySettings:
        method = str(self.solvent_method_combo.currentData() or "").strip()
        if method == CONTRAST_SOLVENT_METHOD_REFERENCE:
            reference_path = self.reference_solvent_file_edit.text().strip()
            if not reference_path:
                raise ValueError(
                    "Choose a reference solvent XYZ or PDB file before computing "
                    "electron density in reference-structure mode."
                )
            solvent = ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_REFERENCE,
                reference_structure_file=reference_path,
            )
        elif method == CONTRAST_SOLVENT_METHOD_DIRECT:
            direct_density = float(self.direct_density_spin.value())
            if direct_density < 0.0:
                raise ValueError(
                    "Enter a non-negative direct electron density value before "
                    "computing electron density in direct-value mode."
                )
            solvent = ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_DIRECT,
                direct_electron_density_e_per_a3=direct_density,
            )
        else:
            formula = self.solvent_formula_edit.text().strip()
            if not formula:
                raise ValueError(
                    "Enter a solvent stoichiometry formula before computing "
                    "electron density in quick estimate mode."
                )
            solvent = ContrastSolventDensitySettings.from_values(
                method=CONTRAST_SOLVENT_METHOD_NEAT,
                solvent_formula=formula,
                solvent_density_g_per_ml=self.solvent_density_spin.value(),
            )
        exclude_elements = (
            ("H",) if self.exclude_hydrogen_checkbox.isChecked() else ()
        )
        return ContrastGeometryDensitySettings(
            solvent=solvent,
            exclude_elements=exclude_elements,
        )

    def _sync_exclude_hydrogen_from_project(self) -> None:
        project_dir = self.project_dir_edit.text().strip()
        if not project_dir:
            self.exclude_hydrogen_checkbox.setChecked(False)
            return
        try:
            settings = SAXSProjectManager().load_project(project_dir)
        except Exception:
            self.exclude_hydrogen_checkbox.setChecked(False)
            return
        excluded = {
            str(element).strip().upper()
            for element in (settings.exclude_elements or [])
            if str(element).strip()
        }
        self.exclude_hydrogen_checkbox.setChecked("H" in excluded)

    def apply_launch_context(
        self,
        launch_context: ContrastModeLaunchContext,
    ) -> None:
        previous_project_dir = getattr(
            self._launch_context,
            "project_dir",
            None,
        )
        previous_distribution_id = self._active_distribution_id
        previous_artifact_dir = self._contrast_artifact_dir_override
        self._launch_context = launch_context
        self._active_distribution_id = launch_context.active_distribution_id
        self._distribution_root_dir = launch_context.distribution_root_dir
        self._contrast_artifact_dir_override = (
            launch_context.contrast_artifact_dir
        )
        self._preview_refresh_timer.stop()
        context_changed = (
            previous_project_dir != launch_context.project_dir
            or previous_distribution_id
            != launch_context.active_distribution_id
            or previous_artifact_dir != launch_context.contrast_artifact_dir
        )
        if context_changed:
            self._representative_analysis_result = None
            self._density_result = None
            self._generated_trace_profiles = []
            self._generated_traces_visible = True
            self._last_build_distribution_id = None
        self.mode_edit.setText(
            component_build_mode_label(COMPONENT_BUILD_MODE_CONTRAST)
        )
        self.project_dir_edit.setText(
            ""
            if launch_context.project_dir is None
            else str(launch_context.project_dir)
        )
        self.template_edit.setText(launch_context.active_template_name or "")
        self.experimental_data_edit.setText(
            ""
            if launch_context.experimental_data_file is None
            else str(launch_context.experimental_data_file)
        )
        self.clusters_dir_edit.setText(
            ""
            if launch_context.clusters_dir is None
            else str(launch_context.clusters_dir)
        )
        self._sync_exclude_hydrogen_from_project()
        self._set_q_range_controls(
            launch_context.q_min,
            launch_context.q_max,
        )
        if launch_context.project_dir is not None:
            self.context_status_label.setText(
                "The contrast workspace inherited the active SAXS project. You "
                "can refine the reference folders and q-range here without "
                "changing the no-contrast builder."
            )
        else:
            self.context_status_label.setText(
                "No active SAXS project was inherited, so this workspace is "
                "running in standalone scaffold mode."
            )
        self.workflow_summary_box.setPlainText(
            build_contrast_workflow_preview(
                self._current_context()
            ).summary_text()
        )
        self._refresh_output_path()
        self._recognized_cluster_bins = []
        self._experimental_summary = None
        self.recognized_clusters_table.setRowCount(0)
        if self._contrast_artifact_dir_override is not None:
            self.cluster_table_status_label.setText(
                "Loading recognized cluster bins and saved contrast outputs..."
            )
            self.representative_table_status_label.setText(
                "Loading saved representative structures for the active "
                "contrast-mode distribution..."
            )
        else:
            self.cluster_table_status_label.setText(
                "Loading recognized cluster bins from the selected clusters folder..."
            )
            if context_changed:
                self.representative_table.setRowCount(0)
                self.representative_table_status_label.setText(
                    "Representative structures will appear here after the "
                    "screening step."
                )
                self._sync_visualizer_preview()
        self.trace_plot_status_label.setText(
            "Loading experimental data preview..."
            if self._current_experimental_file() is not None
            else "No experimental data preview is loaded yet."
        )
        self._redraw_trace_plot()
        self.workflow_progress_label.setText(
            "Progress: loading workspace preview"
        )
        self.workflow_progress_bar.setRange(0, 0)
        self.workflow_progress_bar.setFormat("Loading workspace preview…")
        self.statusBar().showMessage("Loading contrast-mode workspace preview")
        self._pending_launch_context_refresh = True
        self._pending_saved_distribution_restore = (
            self._contrast_artifact_dir_override is not None
        )
        self._queue_preview_refresh(delay_ms=0)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if self._should_block_guarded_field_wheel(watched, event):
            return True
        return super().eventFilter(watched, event)

    def _guarded_field_owner(self, watched: object) -> QWidget | None:
        widget = watched if isinstance(watched, QWidget) else None
        left_root = self.left_scroll_area.widget()
        while widget is not None:
            if isinstance(widget, (QAbstractSpinBox, QComboBox, QLineEdit)):
                if left_root is not None and (
                    widget is left_root or left_root.isAncestorOf(widget)
                ):
                    return widget
                return None
            widget = widget.parentWidget()
        return None

    def _should_block_guarded_field_wheel(
        self,
        watched: object,
        event: QEvent,
    ) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False
        owner = self._guarded_field_owner(watched)
        if owner is None:
            return False
        if isinstance(owner, QComboBox) and owner.view().isVisible():
            return False
        event.ignore()
        return True

    def _set_q_range_controls(
        self,
        q_min: float | None,
        q_max: float | None,
    ) -> None:
        self.q_min_spin.blockSignals(True)
        self.q_max_spin.blockSignals(True)
        self.q_min_spin.setValue(0.0 if q_min is None else float(q_min))
        self.q_max_spin.setValue(0.0 if q_max is None else float(q_max))
        self.q_min_spin.blockSignals(False)
        self.q_max_spin.blockSignals(False)
        self._update_q_range_summary()

    def _current_context(self) -> ContrastModeLaunchContext:
        return ContrastModeLaunchContext.from_values(
            project_dir=self.project_dir_edit.text(),
            clusters_dir=self.clusters_dir_edit.text(),
            experimental_data_file=self.experimental_data_edit.text(),
            q_min=self._current_q_min(),
            q_max=self._current_q_max(),
            active_template_name=self.template_edit.text(),
        )

    def _current_q_min(self) -> float | None:
        value = float(self.q_min_spin.value())
        return None if value <= 0.0 else value

    def _current_q_max(self) -> float | None:
        value = float(self.q_max_spin.value())
        return None if value <= 0.0 else value

    def _current_q_range_text(self) -> str:
        q_min = self._current_q_min()
        q_max = self._current_q_max()
        if q_min is None and q_max is None:
            return "Inherited from the main UI when available"
        if q_min is None or q_max is None:
            return "Incomplete q-range (enter both q min and q max)"
        return f"{q_min:.6g} to {q_max:.6g}"

    def _selected_q_mask(self, q_values: np.ndarray) -> np.ndarray | None:
        q_min = self._current_q_min()
        q_max = self._current_q_max()
        if q_min is None and q_max is None:
            return np.ones_like(q_values, dtype=bool)
        lower = q_min if q_min is not None else float(np.nanmin(q_values))
        upper = q_max if q_max is not None else float(np.nanmax(q_values))
        if lower > upper:
            return np.zeros_like(q_values, dtype=bool)
        return (q_values >= lower) & (q_values <= upper)

    def _current_clusters_dir(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _current_experimental_file(self) -> Path | None:
        text = self.experimental_data_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _current_output_dir(self) -> Path | None:
        if self._distribution_root_dir is not None:
            return self._distribution_root_dir
        project_dir = self.project_dir_edit.text().strip()
        if not project_dir:
            return None
        return Path(project_dir).expanduser().resolve() / "contrast_workflow"

    def _current_representative_output_dir(self) -> Path | None:
        if self._contrast_artifact_dir_override is not None:
            return self._contrast_artifact_dir_override
        output_dir = self._current_output_dir()
        if output_dir is None:
            return None
        return output_dir / "representatives"

    @staticmethod
    def _representative_slug(structure: str, motif: str) -> str:
        normalized_structure = str(structure).strip()
        normalized_motif = str(motif).strip()
        if not normalized_motif or normalized_motif == "no_motif":
            return normalized_structure
        return f"{normalized_structure}__{normalized_motif}"

    def _representative_output_paths(
        self,
        *,
        structure: str,
        motif: str,
    ) -> tuple[Path | None, Path | None]:
        representatives_dir = self._current_representative_output_dir()
        if representatives_dir is None:
            return None, None
        slug = self._representative_slug(structure, motif)
        return (
            representatives_dir / "geometry" / f"{slug}_mesh.json",
            representatives_dir / "electron_density" / f"{slug}_density.json",
        )

    def _representative_row_metadata(self, row: int) -> dict[str, object]:
        first_item = self.representative_table.item(
            row,
            _REPRESENTATIVE_COLUMN_STOICHIOMETRY,
        )
        if first_item is None:
            return {}
        data = first_item.data(_REPRESENTATIVE_ROW_METADATA_ROLE)
        return dict(data) if isinstance(data, dict) else {}

    def _set_representative_row_metadata(
        self,
        row: int,
        metadata: dict[str, object],
    ) -> None:
        first_item = self.representative_table.item(
            row,
            _REPRESENTATIVE_COLUMN_STOICHIOMETRY,
        )
        if first_item is None:
            return
        first_item.setData(
            _REPRESENTATIVE_ROW_METADATA_ROLE,
            dict(metadata),
        )

    def _build_representative_color_item(
        self,
        color_value: str,
    ) -> QTableWidgetItem:
        normalized = _normalized_hex_color(color_value)
        item = QTableWidgetItem(_color_display_value(normalized))
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setBackground(QColor(normalized))
        item.setForeground(QColor(_color_text_hex(normalized)))
        item.setToolTip("Click to choose a custom trace color.")
        return item

    def _representative_default_color(self, row: int) -> str:
        colors = self._trace_colors(
            max(self.representative_table.rowCount(), 1)
        )
        if not colors:
            return "#1f77b4"
        clamped_row = max(0, min(row, len(colors) - 1))
        return _normalized_hex_color(to_hex(colors[clamped_row]))

    def _representative_trace_color(self, row: int) -> str:
        metadata = self._representative_row_metadata(row)
        if bool(metadata.get("custom_trace_color")):
            return _normalized_hex_color(metadata.get("trace_color"))
        return self._representative_default_color(row)

    def _refresh_representative_color_cells(self) -> None:
        for row in range(self.representative_table.rowCount()):
            metadata = self._representative_row_metadata(row)
            color_value = self._representative_trace_color(row)
            metadata["trace_color"] = color_value
            self.representative_table.setItem(
                row,
                _REPRESENTATIVE_COLUMN_COLOR,
                self._build_representative_color_item(color_value),
            )
            self._set_representative_row_metadata(row, metadata)

    def _clear_representative_trace_metadata(self) -> None:
        for row in range(self.representative_table.rowCount()):
            metadata = self._representative_row_metadata(row)
            if not metadata:
                continue
            for key in (
                "trace_ready",
                "profile_file",
                "profile_path",
                "contrast_scale_factor",
                "trace_cluster_density_e_per_a3",
                "trace_solvent_density_e_per_a3",
                "trace_contrast_density_e_per_a3",
            ):
                metadata.pop(key, None)
            self._set_representative_row_metadata(row, metadata)

    @Slot(int, int)
    def _on_representative_table_cell_clicked(
        self, row: int, column: int
    ) -> None:
        if column != _REPRESENTATIVE_COLUMN_COLOR or row < 0:
            return
        current_color = self._representative_trace_color(row)
        chosen = QColorDialog.getColor(
            QColor(current_color),
            self,
            "Choose representative trace color",
        )
        if not chosen.isValid():
            return
        metadata = self._representative_row_metadata(row)
        metadata["trace_color"] = chosen.name()
        metadata["custom_trace_color"] = True
        self._set_representative_row_metadata(row, metadata)
        self.representative_table.setItem(
            row,
            _REPRESENTATIVE_COLUMN_COLOR,
            self._build_representative_color_item(chosen.name()),
        )
        self._redraw_trace_plot()

    def _refresh_representative_row_metadata(
        self,
        row: int,
    ) -> dict[str, object]:
        metadata = self._representative_row_metadata(row)
        if not metadata:
            return {}

        structure = str(metadata.get("structure") or "").strip()
        motif = str(metadata.get("motif") or "no_motif").strip() or "no_motif"
        source_kind = str(metadata.get("source_kind") or "manual").strip()

        mesh_json_path = str(metadata.get("mesh_json_path") or "").strip()
        density_json_path = str(
            metadata.get("density_json_path") or ""
        ).strip()
        if source_kind == "auto":
            default_mesh_path, default_density_path = (
                self._representative_output_paths(
                    structure=structure,
                    motif=motif,
                )
            )
            if default_mesh_path is not None and not mesh_json_path:
                mesh_json_path = str(default_mesh_path)
            if default_density_path is not None and not density_json_path:
                density_json_path = str(default_density_path)

        mesh_payload: dict[str, object] | None = None
        density_payload: dict[str, object] | None = None
        if mesh_json_path:
            mesh_path = Path(mesh_json_path).expanduser().resolve()
            if mesh_path.is_file():
                mesh_payload = json.loads(
                    mesh_path.read_text(encoding="utf-8")
                )
                metadata["mesh_json_path"] = str(mesh_path)
                metadata["mesh_volume_a3"] = mesh_payload.get("volume_a3")
                metadata["mesh_surface_area_a2"] = mesh_payload.get(
                    "surface_area_a2"
                )
        if density_json_path:
            density_path = Path(density_json_path).expanduser().resolve()
            if density_path.is_file():
                density_payload = json.loads(
                    density_path.read_text(encoding="utf-8")
                )
                metadata["density_json_path"] = str(density_path)
                metadata["cluster_density_e_per_a3"] = (
                    density_payload.get("cluster_electron_density", {}) or {}
                ).get("electron_density_e_per_a3")
                metadata["solvent_density_e_per_a3"] = (
                    density_payload.get("solvent_electron_density", {}) or {}
                ).get("electron_density_e_per_a3")
                metadata["contrast_density_e_per_a3"] = density_payload.get(
                    "contrast_electron_density_e_per_a3"
                )

        mesh_text = "Pending"
        density_text = "Pending"
        trace_status = (
            "Not computed" if source_kind == "manual" else "Selection ready"
        )
        if metadata.get("mesh_volume_a3") is not None:
            mesh_text = f"{float(metadata['mesh_volume_a3']):.2f} A^3"
            trace_status = "Mesh ready"
        if metadata.get("contrast_density_e_per_a3") is not None:
            density_text = (
                f"Δρ {float(metadata['contrast_density_e_per_a3']):+.4f} e/A^3"
            )
            trace_status = "Density ready"
        if metadata.get("trace_ready"):
            contrast_scale = metadata.get("contrast_scale_factor")
            if contrast_scale is not None:
                trace_status = (
                    f"Trace ready (scale {float(contrast_scale):.4f})"
                )
            else:
                trace_status = "Trace ready"

        for column, value in (
            (_REPRESENTATIVE_COLUMN_MESH, mesh_text),
            (_REPRESENTATIVE_COLUMN_DENSITY, density_text),
            (_REPRESENTATIVE_COLUMN_TRACE_STATUS, trace_status),
        ):
            item = self.representative_table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.representative_table.setItem(row, column, item)
            item.setText(value)
            if column == _REPRESENTATIVE_COLUMN_TRACE_STATUS:
                tooltip_lines = [trace_status]
                if metadata.get("contrast_scale_factor") is not None:
                    tooltip_lines.append(
                        "Contrast scale: "
                        f"{float(metadata['contrast_scale_factor']):.6f}"
                    )
                if metadata.get("trace_cluster_density_e_per_a3") is not None:
                    tooltip_lines.append(
                        "Cluster density: "
                        f"{float(metadata['trace_cluster_density_e_per_a3']):.6f} e/A^3"
                    )
                if metadata.get("trace_solvent_density_e_per_a3") is not None:
                    tooltip_lines.append(
                        "Solvent density: "
                        f"{float(metadata['trace_solvent_density_e_per_a3']):.6f} e/A^3"
                    )
                item.setToolTip("\n".join(tooltip_lines))

        self._set_representative_row_metadata(row, metadata)
        return metadata

    def _refresh_representative_metadata_from_saved_outputs(self) -> None:
        for row in range(self.representative_table.rowCount()):
            self._refresh_representative_row_metadata(row)

    def _update_q_range_summary(self) -> None:
        self.q_range_edit.setText(self._current_q_range_text())

    def _set_output_path_fields(
        self,
        *,
        workflow_dir: Path | None,
        representatives_dir: Path | None,
        screening_dir: Path | None,
        summary_path: Path | None,
    ) -> None:
        self.output_path_edit.setText(
            "" if workflow_dir is None else str(workflow_dir)
        )
        self.representatives_output_edit.setText(
            "" if representatives_dir is None else str(representatives_dir)
        )
        self.screening_output_edit.setText(
            "" if screening_dir is None else str(screening_dir)
        )
        self.summary_output_edit.setText(
            "" if summary_path is None else str(summary_path)
        )

    def _refresh_output_path(self) -> None:
        workflow_dir = self._current_output_dir()
        representatives_dir = self._current_representative_output_dir()
        self._set_output_path_fields(
            workflow_dir=workflow_dir,
            representatives_dir=representatives_dir,
            screening_dir=(
                None
                if representatives_dir is None
                else representatives_dir / "screening"
            ),
            summary_path=(
                None
                if representatives_dir is None
                else representatives_dir / "selection_summary.json"
            ),
        )

    def _set_workflow_stage_progress(
        self,
        stage_value: int,
        *,
        detail: str,
    ) -> None:
        normalized_stage = min(
            max(int(stage_value), 0),
            _WORKFLOW_STAGE_TOTAL,
        )
        self.workflow_progress_bar.setRange(0, _WORKFLOW_STAGE_TOTAL)
        self.workflow_progress_bar.setValue(normalized_stage)
        self.workflow_progress_bar.setFormat("%v / %m workflow stages")
        self.workflow_progress_label.setText(
            f"Progress: {detail} "
            f"({normalized_stage}/{_WORKFLOW_STAGE_TOTAL} workflow stages)"
        )

    def _rebuild_preview(
        self,
        *,
        log_to_console: bool,
        status_message: str,
    ) -> None:
        preview_context = self._current_context()
        preview = build_contrast_workflow_preview(preview_context)
        self.workflow_summary_box.setPlainText(preview.summary_text())
        self._update_q_range_summary()
        self._refresh_output_path()
        cluster_count = self._populate_cluster_inventory(
            log_to_console=log_to_console
        )
        experimental_loaded = self._load_experimental_preview(
            log_to_console=log_to_console
        )
        self._redraw_trace_plot()
        self._refresh_representative_metadata_from_saved_outputs()
        self._sync_visualizer_preview()

        progress_value = 1
        if preview_context.project_dir is not None:
            progress_value += 1
        if cluster_count > 0:
            progress_value += 1
        if experimental_loaded:
            progress_value += 1
        self._set_workflow_stage_progress(
            progress_value,
            detail="workspace prepared",
        )
        self.statusBar().showMessage(status_message)
        self.project_paths_registered.emit(
            {
                "project_dir": (
                    None
                    if preview_context.project_dir is None
                    else str(preview_context.project_dir)
                ),
                "clusters_dir": (
                    None
                    if preview_context.clusters_dir is None
                    else str(preview_context.clusters_dir)
                ),
                "experimental_data_file": (
                    None
                    if preview_context.experimental_data_file is None
                    else str(preview_context.experimental_data_file)
                ),
                "output_dir": (
                    None
                    if self._current_output_dir() is None
                    else str(self._current_output_dir())
                ),
            }
        )

    @Slot()
    def refresh_preview(self) -> None:
        self._rebuild_preview(
            log_to_console=True,
            status_message="Contrast-mode workspace preview refreshed",
        )
        self._append_console_line(
            "Refreshed the contrast-mode workspace preview."
        )

    def _queue_preview_refresh(self, *, delay_ms: int) -> None:
        self._preview_refresh_timer.start(max(int(delay_ms), 0))

    @Slot()
    def _schedule_preview_refresh(self) -> None:
        self._queue_preview_refresh(delay_ms=_UI_REFRESH_DELAY_MS)

    def _flush_scheduled_preview_refresh(self) -> None:
        status_message = "Contrast-mode workspace preview refreshed"
        if self._pending_launch_context_refresh:
            status_message = (
                "Contrast-mode workspace loaded"
                if self._contrast_artifact_dir_override is None
                else "Contrast-mode distribution view loaded"
            )
        self._rebuild_preview(
            log_to_console=False,
            status_message=status_message,
        )
        restored_saved_distribution = False
        if (
            self._pending_saved_distribution_restore
            and self._contrast_artifact_dir_override is not None
        ):
            restored_saved_distribution = (
                self._restore_saved_distribution_view()
            )
        if self._pending_launch_context_refresh:
            if restored_saved_distribution:
                self._append_console_line(
                    "Loaded the saved contrast-mode distribution view for the "
                    "active computed distribution."
                )
            else:
                self._append_console_line(
                    "Loaded contrast-mode workspace context."
                )
        self._pending_launch_context_refresh = False
        self._pending_saved_distribution_restore = False

    @Slot()
    def log_current_context(self) -> None:
        self._append_console_line(
            "Current contrast-mode workspace context:\n"
            + build_contrast_workflow_preview(
                self._current_context()
            ).summary_text()
        )
        self.statusBar().showMessage("Contrast-mode workspace context logged")

    def _contrast_cluster_inventory(
        self,
    ) -> tuple[list[ClusterBin], list[dict[str, object]], int]:
        project_dir = self.project_dir_edit.text().strip()
        clusters_dir = self._current_clusters_dir()
        if clusters_dir is None:
            raise ValueError(
                "Choose a reference clusters folder to inspect the recognized bins."
            )
        if not project_dir:
            cluster_bins = list(discover_cluster_bins(clusters_dir))
            return cluster_bins, [], 0
        if not build_project_paths(project_dir).project_file.is_file():
            cluster_bins = list(discover_cluster_bins(clusters_dir))
            return cluster_bins, [], 0
        manager = SAXSProjectManager()
        settings = self._contrast_project_settings()
        (
            cluster_inventory,
            _predicted_dataset_file,
            predicted_component_count,
        ) = manager.contrast_cluster_inventory(settings)
        return (
            list(cluster_inventory.cluster_bins),
            list(cluster_inventory.cluster_rows),
            int(predicted_component_count),
        )

    def _populate_cluster_inventory(self, *, log_to_console: bool) -> int:
        self.recognized_clusters_table.setRowCount(0)
        self._recognized_cluster_bins = []
        clusters_dir = self._current_clusters_dir()
        if clusters_dir is None:
            self.cluster_table_status_label.setText(
                "Choose a reference clusters folder to inspect the recognized bins."
            )
            return 0
        try:
            (
                cluster_bins,
                cluster_rows,
                predicted_component_count,
            ) = self._contrast_cluster_inventory()
        except Exception as exc:
            self.cluster_table_status_label.setText(
                f"Unable to inspect {clusters_dir}: {exc}"
            )
            if log_to_console:
                self._append_console_line(
                    f"Cluster validation failed for {clusters_dir}: {exc}"
                )
            return 0

        self._recognized_cluster_bins = cluster_bins
        row_lookup = {
            (
                str(row.get("structure") or "").strip(),
                str(row.get("motif") or "no_motif").strip() or "no_motif",
            ): row
            for row in cluster_rows
            if isinstance(row, dict)
        }
        total_files = sum(
            len(cluster_bin.files) for cluster_bin in cluster_bins
        )
        total_atoms = sum(
            max(sum(parse_stoich_label(cluster_bin.structure).values()), 1)
            * len(cluster_bin.files)
            for cluster_bin in cluster_bins
        )
        for row_index, cluster_bin in enumerate(cluster_bins):
            self.recognized_clusters_table.insertRow(row_index)
            stoich_counts = parse_stoich_label(cluster_bin.structure)
            atom_total = max(sum(stoich_counts.values()), 1)
            weighted_atom_total = atom_total * len(cluster_bin.files)
            cluster_row = row_lookup.get(
                (cluster_bin.structure, cluster_bin.motif),
                {},
            )
            weight_percent = float(
                cluster_row.get(
                    "structure_fraction_percent",
                    (
                        float(len(cluster_bin.files))
                        / float(total_files)
                        * 100.0
                        if total_files > 0
                        else 0.0
                    ),
                )
            )
            atom_percent = float(
                cluster_row.get(
                    "atom_fraction_percent",
                    (
                        float(weighted_atom_total) / float(total_atoms) * 100.0
                        if total_atoms > 0
                        else 0.0
                    ),
                )
            )
            source_kind = str(
                cluster_row.get("source_kind") or "cluster_dir"
            ).strip()
            cluster_label = (
                cluster_bin.structure
                if cluster_bin.motif == "no_motif"
                else f"{cluster_bin.structure} / {cluster_bin.motif}"
            )
            cluster_type = (
                "Predicted structure"
                if source_kind == "predicted_structure"
                else (
                    "No motif"
                    if cluster_bin.motif == "no_motif"
                    else cluster_bin.motif.replace("_", " ")
                )
            )
            values = (
                cluster_label,
                cluster_bin.structure,
                cluster_type,
                str(len(cluster_bin.files)),
                f"{weight_percent:.1f}%",
                f"{atom_percent:.1f}%",
                str(
                    cluster_row.get("representative")
                    or cluster_bin.representative
                    or "—"
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(cluster_bin.source_dir),
                    )
                self.recognized_clusters_table.setItem(row_index, column, item)

        if cluster_bins:
            self.recognized_clusters_table.selectRow(0)
            predicted_suffix = (
                f", including {predicted_component_count} predicted structure bin(s)"
                if predicted_component_count > 0
                else ""
            )
            self.cluster_table_status_label.setText(
                f"Loaded {len(cluster_bins)} recognized cluster bins from "
                f"{clusters_dir}{predicted_suffix}."
            )
            if log_to_console:
                self._append_console_line(
                    f"Loaded {len(cluster_bins)} recognized cluster bins from "
                    f"{clusters_dir}{predicted_suffix}."
                )
        else:
            self.cluster_table_status_label.setText(
                f"No cluster bins were recognized in {clusters_dir}."
            )
            if log_to_console:
                self._append_console_line(
                    f"No recognized cluster bins were found in {clusters_dir}."
                )
        return len(cluster_bins)

    def _load_experimental_preview(self, *, log_to_console: bool) -> bool:
        self._experimental_summary = None
        experimental_file = self._current_experimental_file()
        if experimental_file is None:
            self.trace_plot_status_label.setText(
                "No experimental data preview is loaded yet."
            )
            return False
        try:
            self._experimental_summary = load_experimental_data_file(
                experimental_file
            )
        except Exception as exc:
            self.trace_plot_status_label.setText(
                f"Unable to preview experimental data from {experimental_file}: {exc}"
            )
            if log_to_console:
                self._append_console_line(
                    f"Experimental preview failed for {experimental_file}: {exc}"
                )
            return False
        self.trace_plot_status_label.setText(
            f"Previewing {len(self._experimental_summary.q_values)} q points from "
            f"{self._experimental_summary.path}."
        )
        if log_to_console:
            self._append_console_line(
                f"Loaded experimental preview from {self._experimental_summary.path}."
            )
        return True

    def _trace_color(self) -> str:
        color_key = self.trace_color_scheme_combo.currentData()
        if color_key in (None, "", "default"):
            return "#1f77b4"
        cmap = colormaps.get_cmap(str(color_key))
        return cmap(0.68)

    def _update_trace_control_state(self) -> None:
        has_generated = bool(self._generated_trace_profiles)
        self.trace_generated_toggle_button.setEnabled(has_generated)
        self.trace_color_scheme_combo.setEnabled(has_generated)
        self.trace_generated_toggle_button.setText(
            "Hide Computed Traces"
            if self._generated_traces_visible
            else "Show Computed Traces"
        )

    def _trace_colors(self, count: int) -> list[object]:
        if count <= 0:
            return []
        color_key = self.trace_color_scheme_combo.currentData()
        if color_key in (None, "", "default"):
            cmap = colormaps.get_cmap("tab10")
        else:
            cmap = colormaps.get_cmap(str(color_key))
        if count == 1:
            return [cmap(0.68)]
        return [
            cmap(0.12 + 0.76 * (index / max(count - 1, 1)))
            for index in range(count)
        ]

    def _load_generated_trace_profiles(
        self,
        settings: ProjectSettings,
        component_entries: list[object],
    ) -> None:
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
            allow_legacy_fallback=False,
        )
        generated_profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
        for entry in component_entries:
            profile_file = str(getattr(entry, "profile_file", "")).strip()
            if not profile_file:
                continue
            profile_path = artifact_paths.component_dir / profile_file
            if not profile_path.is_file():
                continue
            data = np.loadtxt(profile_path, comments="#")
            data = np.atleast_2d(np.asarray(data, dtype=float))
            if data.shape[1] < 2:
                continue
            structure = str(getattr(entry, "structure", "")).strip()
            motif = str(getattr(entry, "motif", "")).strip() or "no_motif"
            display_label = (
                structure if motif == "no_motif" else f"{structure}/{motif}"
            )
            generated_profiles.append(
                (
                    display_label,
                    np.asarray(data[:, 0], dtype=float),
                    np.asarray(data[:, 1], dtype=float),
                )
            )
        self._generated_trace_profiles = generated_profiles
        self._generated_traces_visible = True

    def _load_generated_trace_profiles_from_trace_payloads(
        self,
        trace_payloads: (
            list[dict[str, object]] | tuple[dict[str, object], ...]
        ),
        *,
        default_component_dir: Path | None = None,
    ) -> None:
        generated_profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
        for payload in trace_payloads:
            if not isinstance(payload, dict):
                continue
            profile_path_text = str(payload.get("profile_path") or "").strip()
            profile_file = str(payload.get("profile_file") or "").strip()
            if profile_path_text:
                profile_path = Path(profile_path_text).expanduser().resolve()
            elif default_component_dir is not None and profile_file:
                profile_path = default_component_dir / profile_file
            else:
                continue
            if not profile_path.is_file():
                continue
            data = np.loadtxt(profile_path, comments="#")
            data = np.atleast_2d(np.asarray(data, dtype=float))
            if data.shape[1] < 2:
                continue
            display_label = (
                str(payload.get("display_label") or "").strip()
                or str(payload.get("structure") or "").strip()
                or profile_path.stem
            )
            generated_profiles.append(
                (
                    display_label,
                    np.asarray(data[:, 0], dtype=float),
                    np.asarray(data[:, 1], dtype=float),
                )
            )
        self._generated_trace_profiles = generated_profiles
        self._generated_traces_visible = True

    def _apply_saved_trace_metadata(
        self,
        trace_payloads: (
            list[dict[str, object]] | tuple[dict[str, object], ...]
        ),
    ) -> None:
        trace_by_key: dict[tuple[str, str], dict[str, object]] = {}
        for payload in trace_payloads:
            if not isinstance(payload, dict):
                continue
            structure = str(payload.get("structure") or "").strip()
            motif = (
                str(payload.get("motif") or "no_motif").strip() or "no_motif"
            )
            if not structure:
                continue
            trace_by_key[(structure, motif)] = payload

        for row in range(self.representative_table.rowCount()):
            metadata = self._representative_row_metadata(row)
            structure = str(metadata.get("structure") or "").strip()
            motif = (
                str(metadata.get("motif") or "no_motif").strip() or "no_motif"
            )
            trace_payload = trace_by_key.get((structure, motif))
            if trace_payload is None:
                metadata["trace_ready"] = False
                for key in (
                    "profile_file",
                    "profile_path",
                    "contrast_scale_factor",
                    "trace_cluster_density_e_per_a3",
                    "trace_solvent_density_e_per_a3",
                    "trace_contrast_density_e_per_a3",
                ):
                    metadata.pop(key, None)
                self._set_representative_row_metadata(row, metadata)
                self._refresh_representative_row_metadata(row)
                continue
            metadata["trace_ready"] = True
            metadata["profile_file"] = str(
                trace_payload.get("profile_file") or ""
            ).strip()
            profile_path_text = str(
                trace_payload.get("profile_path") or ""
            ).strip()
            if profile_path_text:
                metadata["profile_path"] = str(
                    Path(profile_path_text).expanduser().resolve()
                )
            if trace_payload.get("contrast_scale_factor") is not None:
                metadata["contrast_scale_factor"] = float(
                    trace_payload.get("contrast_scale_factor") or 0.0
                )
            if trace_payload.get("cluster_density_e_per_a3") is not None:
                metadata["trace_cluster_density_e_per_a3"] = float(
                    trace_payload.get("cluster_density_e_per_a3") or 0.0
                )
            if trace_payload.get("solvent_density_e_per_a3") is not None:
                metadata["trace_solvent_density_e_per_a3"] = float(
                    trace_payload.get("solvent_density_e_per_a3") or 0.0
                )
            if trace_payload.get("contrast_density_e_per_a3") is not None:
                metadata["trace_contrast_density_e_per_a3"] = float(
                    trace_payload.get("contrast_density_e_per_a3") or 0.0
                )
                metadata["contrast_density_e_per_a3"] = float(
                    trace_payload.get("contrast_density_e_per_a3") or 0.0
                )
            self._set_representative_row_metadata(row, metadata)
            self._refresh_representative_row_metadata(row)

    @staticmethod
    def _saved_counts_from_payload(
        payload: object,
    ) -> dict[str, int]:
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): int(value)
            for key, value in sorted(payload.items())
            if str(key).strip()
        }

    @staticmethod
    def _saved_float_map_from_payload(
        payload: object,
    ) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in sorted(payload.items())
            if str(key).strip()
        }

    @staticmethod
    def _saved_path_from_payload(
        payload_value: object,
        *,
        fallback_dir: Path | None = None,
    ) -> Path | None:
        text = str(payload_value or "").strip()
        if text:
            path = Path(text).expanduser()
            if path.is_absolute():
                return path.resolve()
            if fallback_dir is not None:
                return (fallback_dir / path).resolve()
            return path.resolve()
        return None

    def _restore_saved_candidate_descriptor(
        self,
        descriptor_payload: object,
        *,
        file_path: Path,
    ) -> ContrastStructureDescriptor:
        payload = (
            descriptor_payload if isinstance(descriptor_payload, dict) else {}
        )
        return ContrastStructureDescriptor(
            file_path=file_path,
            atom_count=int(payload.get("atom_count") or 0),
            element_counts=self._saved_counts_from_payload(
                payload.get("element_counts")
            ),
            core_atom_count=int(payload.get("core_atom_count") or 0),
            core_element_counts=self._saved_counts_from_payload(
                payload.get("core_element_counts")
            ),
            solvent_atom_count=int(payload.get("solvent_atom_count") or 0),
            solvent_element_counts=self._saved_counts_from_payload(
                payload.get("solvent_element_counts")
            ),
            direct_solvent_atom_count=int(
                payload.get("direct_solvent_atom_count") or 0
            ),
            outer_solvent_atom_count=int(
                payload.get("outer_solvent_atom_count") or 0
            ),
            direct_solvent_element_counts=self._saved_counts_from_payload(
                payload.get("direct_solvent_element_counts")
            ),
            outer_solvent_element_counts=self._saved_counts_from_payload(
                payload.get("outer_solvent_element_counts")
            ),
            mean_direct_solvent_coordination=float(
                payload.get("mean_direct_solvent_coordination") or 0.0
            ),
            direct_solvent_coordination_by_core_element=(
                self._saved_float_map_from_payload(
                    payload.get("direct_solvent_coordination_by_core_element")
                )
            ),
            bond_length_medians=self._saved_float_map_from_payload(
                payload.get("bond_length_medians")
            ),
            angle_medians=self._saved_float_map_from_payload(
                payload.get("angle_medians")
            ),
            coordination_medians=self._saved_float_map_from_payload(
                payload.get("coordination_medians")
            ),
            notes=tuple(str(note) for note in (payload.get("notes") or [])),
        )

    def _restore_saved_candidate(
        self,
        candidate_payload: object,
        *,
        fallback_file_path: Path,
    ) -> ContrastRepresentativeCandidate:
        payload = (
            candidate_payload if isinstance(candidate_payload, dict) else {}
        )
        descriptor_payload = (
            payload.get("descriptor") if isinstance(payload, dict) else {}
        )
        descriptor_file_path = self._saved_path_from_payload(
            (descriptor_payload or {}).get("file_path"),
            fallback_dir=fallback_file_path.parent,
        )
        if descriptor_file_path is None:
            descriptor_file_path = fallback_file_path
        descriptor = self._restore_saved_candidate_descriptor(
            descriptor_payload,
            file_path=descriptor_file_path,
        )
        return ContrastRepresentativeCandidate(
            descriptor=descriptor,
            score_total=float(payload.get("score_total") or 0.0),
            score_bond=float(payload.get("score_bond") or 0.0),
            score_angle=float(payload.get("score_angle") or 0.0),
            score_coordination=float(payload.get("score_coordination") or 0.0),
            score_solvent=float(payload.get("score_solvent") or 0.0),
        )

    def _restore_saved_representative_result(
        self,
        selection_payload: dict[str, object],
        *,
        contrast_dir: Path,
    ) -> ContrastRepresentativeSelectionResult | None:
        representative_structures_dir = (
            contrast_dir / "representative_structures"
        )
        screening_dir = contrast_dir / "screening"
        bin_results: list[ContrastRepresentativeBinResult] = []
        for raw_bin_payload in selection_payload.get("bin_results") or []:
            if not isinstance(raw_bin_payload, dict):
                continue
            structure = str(raw_bin_payload.get("structure") or "").strip()
            if not structure:
                continue
            motif = (
                str(raw_bin_payload.get("motif") or "no_motif").strip()
                or "no_motif"
            )
            copied_text = str(
                raw_bin_payload.get("copied_representative_file") or ""
            ).strip()
            representative_path = (
                representative_structures_dir / Path(copied_text).name
                if copied_text
                else representative_structures_dir / f"{structure}.xyz"
            )
            if not representative_path.is_file() and copied_text:
                representative_path = Path(copied_text).expanduser().resolve()
            selected_file = self._saved_path_from_payload(
                raw_bin_payload.get("selected_file"),
            )
            if selected_file is None:
                selected_file = representative_path
            source_dir = self._saved_path_from_payload(
                raw_bin_payload.get("source_dir"),
            )
            if source_dir is None:
                source_dir = selected_file.parent
            screening_json_path = (
                screening_dir
                / Path(
                    str(
                        raw_bin_payload.get("screening_json_path") or ""
                    ).strip()
                ).name
            )
            screening_table_path = (
                screening_dir
                / Path(
                    str(
                        raw_bin_payload.get("screening_table_path") or ""
                    ).strip()
                ).name
            )
            target_payload = raw_bin_payload.get("target_summary")
            target_summary = ContrastRepresentativeTargetSummary(
                pair_contact_distance_medians=self._saved_float_map_from_payload(
                    (target_payload or {}).get("pair_contact_distance_medians")
                ),
                bond_length_medians=self._saved_float_map_from_payload(
                    (target_payload or {}).get("bond_length_medians")
                ),
                angle_medians=self._saved_float_map_from_payload(
                    (target_payload or {}).get("angle_medians")
                ),
                coordination_medians=self._saved_float_map_from_payload(
                    (target_payload or {}).get("coordination_medians")
                ),
                solvent_metrics=self._saved_float_map_from_payload(
                    (target_payload or {}).get("solvent_metrics")
                ),
            )
            selected_candidate = self._restore_saved_candidate(
                raw_bin_payload.get("selected_candidate"),
                fallback_file_path=selected_file,
            )
            candidate_payloads = raw_bin_payload.get("candidates") or []
            restored_candidates = tuple(
                self._restore_saved_candidate(
                    candidate_payload,
                    fallback_file_path=selected_file,
                )
                for candidate_payload in candidate_payloads
                if isinstance(candidate_payload, dict)
            )
            if not restored_candidates:
                restored_candidates = (selected_candidate,)
            sampler_payload = raw_bin_payload.get("sampler_settings") or {}
            bin_results.append(
                ContrastRepresentativeBinResult(
                    structure=structure,
                    motif=motif,
                    source_dir=source_dir,
                    file_count=int(raw_bin_payload.get("file_count") or 0),
                    selected_file=selected_file,
                    copied_representative_file=representative_path,
                    target_summary=target_summary,
                    selected_candidate=selected_candidate,
                    candidates=restored_candidates,
                    selection_strategy=str(
                        raw_bin_payload.get("selection_strategy")
                        or "full_scan"
                    ).strip()
                    or "full_scan",
                    distribution_sample_count=int(
                        raw_bin_payload.get("distribution_sample_count") or 0
                    ),
                    sampled_candidate_count=int(
                        raw_bin_payload.get("sampled_candidate_count") or 0
                    ),
                    sampler_settings=ContrastRepresentativeSamplerSettings.from_values(
                        **(
                            sampler_payload
                            if isinstance(sampler_payload, dict)
                            else {}
                        )
                    ),
                    screening_json_path=screening_json_path.resolve(),
                    screening_table_path=screening_table_path.resolve(),
                    notes=tuple(
                        str(note)
                        for note in (raw_bin_payload.get("notes") or [])
                    ),
                )
            )
        if not bin_results:
            return None

        issues: tuple[ContrastRepresentativeIssue, ...] = tuple()
        project_dir = self._saved_path_from_payload(
            selection_payload.get("project_dir"),
            fallback_dir=(
                Path(self.project_dir_edit.text()).expanduser().resolve()
                if self.project_dir_edit.text().strip()
                else contrast_dir.parent
            ),
        )
        if project_dir is None:
            project_dir = contrast_dir.parent.resolve()
        clusters_dir = self._saved_path_from_payload(
            selection_payload.get("clusters_dir"),
            fallback_dir=(self._current_clusters_dir() or contrast_dir.parent),
        )
        if clusters_dir is None:
            clusters_dir = (
                self._current_clusters_dir() or contrast_dir.parent.resolve()
            )
        return ContrastRepresentativeSelectionResult(
            project_dir=project_dir,
            clusters_dir=clusters_dir,
            output_dir=contrast_dir.resolve(),
            representative_structures_dir=representative_structures_dir.resolve(),
            screening_dir=screening_dir.resolve(),
            generated_at=str(
                selection_payload.get("generated_at") or ""
            ).strip()
            or datetime.now().astimezone().isoformat(timespec="seconds"),
            bin_results=tuple(bin_results),
            issues=issues,
            summary_json_path=(
                contrast_dir / "selection_summary.json"
            ).resolve(),
            summary_table_path=(
                contrast_dir / "selection_summary.tsv"
            ).resolve(),
            summary_text_path=(
                contrast_dir / "selection_summary.txt"
            ).resolve(),
        )

    def _restore_saved_distribution_view(self) -> bool:
        contrast_dir = self._contrast_artifact_dir_override
        if contrast_dir is None or not contrast_dir.is_dir():
            return False

        selection_summary_path = contrast_dir / "selection_summary.json"
        if selection_summary_path.is_file():
            selection_payload = json.loads(
                selection_summary_path.read_text(encoding="utf-8")
            )
            self._representative_analysis_result = (
                self._restore_saved_representative_result(
                    selection_payload,
                    contrast_dir=contrast_dir,
                )
            )
            self._populate_saved_representative_table(
                selection_payload,
                contrast_dir=contrast_dir,
            )
        else:
            self._representative_analysis_result = None
            self.representative_table.setRowCount(0)
            self.representative_table_status_label.setText(
                "No saved representative summary is available for this "
                "contrast-mode distribution."
            )

        debye_summary_path = contrast_dir / "debye" / "component_summary.json"
        trace_payloads: list[dict[str, object]] = []
        if debye_summary_path.is_file():
            debye_payload = json.loads(
                debye_summary_path.read_text(encoding="utf-8")
            )
            trace_payloads = list(debye_payload.get("trace_results") or [])
            self._load_generated_trace_profiles_from_trace_payloads(
                trace_payloads,
                default_component_dir=(
                    self._current_output_dir() / "scattering_components"
                    if self._current_output_dir() is not None
                    else None
                ),
            )
            self._apply_saved_trace_metadata(trace_payloads)
        else:
            self._generated_trace_profiles = []
            self._generated_traces_visible = True

        self._refresh_representative_metadata_from_saved_outputs()
        self._redraw_trace_plot()
        self._sync_visualizer_preview()
        self._set_workflow_stage_progress(
            5 if trace_payloads else 4,
            detail="saved contrast distribution loaded",
        )
        self.statusBar().showMessage(
            "Loaded saved contrast-mode representative outputs"
        )
        return True

    def _populate_saved_representative_table(
        self,
        selection_payload: dict[str, object],
        *,
        contrast_dir: Path,
    ) -> None:
        self.representative_table.setRowCount(0)
        representative_structures_dir = (
            contrast_dir / "representative_structures"
        )
        screening_dir = contrast_dir / "screening"
        bin_results = list(selection_payload.get("bin_results") or [])
        for row_index, bin_payload in enumerate(bin_results):
            if not isinstance(bin_payload, dict):
                continue
            structure = str(bin_payload.get("structure") or "").strip()
            motif = (
                str(bin_payload.get("motif") or "no_motif").strip()
                or "no_motif"
            )
            display_label = (
                str(bin_payload.get("display_label") or "").strip()
                or structure
            )
            selected_candidate = dict(
                bin_payload.get("selected_candidate") or {}
            )
            descriptor = dict(selected_candidate.get("descriptor") or {})
            representative_name = Path(
                str(
                    bin_payload.get("copied_representative_file") or ""
                ).strip()
            ).name
            representative_path = (
                representative_structures_dir / representative_name
            )
            if not representative_path.is_file():
                fallback_text = str(
                    bin_payload.get("copied_representative_file") or ""
                ).strip()
                if fallback_text:
                    representative_path = (
                        Path(fallback_text).expanduser().resolve()
                    )
            screening_json_name = Path(
                str(bin_payload.get("screening_json_path") or "").strip()
            ).name
            screening_table_name = Path(
                str(bin_payload.get("screening_table_path") or "").strip()
            ).name
            screening_json_path = screening_dir / screening_json_name
            screening_table_path = screening_dir / screening_table_name
            direct_count = int(
                descriptor.get("direct_solvent_atom_count") or 0
            )
            outer_count = int(descriptor.get("outer_solvent_atom_count") or 0)
            score_total = float(selected_candidate.get("score_total") or 0.0)

            self.representative_table.insertRow(row_index)
            values = (
                display_label,
                "Auto",
                representative_path.name,
                "",
                f"Direct {direct_count}; Outer {outer_count}",
                "Pending",
                "Pending",
                "Selection ready",
                f"Score {score_total:.4f} | {screening_json_path.name}",
            )
            for column, value in enumerate(values):
                if column == _REPRESENTATIVE_COLUMN_COLOR:
                    item = self._build_representative_color_item(
                        self._representative_default_color(row_index)
                    )
                else:
                    item = QTableWidgetItem(value)
                if column == _REPRESENTATIVE_COLUMN_FILE:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(representative_path),
                    )
                if column == _REPRESENTATIVE_COLUMN_NOTES:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(screening_json_path),
                    )
                self.representative_table.setItem(row_index, column, item)
            self._set_representative_row_metadata(
                row_index,
                {
                    "source_kind": "auto",
                    "display_label": display_label,
                    "structure": structure,
                    "motif": motif,
                    "representative_file": str(representative_path),
                    "screening_json_path": str(screening_json_path),
                    "screening_table_path": str(screening_table_path),
                    "trace_color": self._representative_default_color(
                        row_index
                    ),
                    "custom_trace_color": False,
                },
            )
            self._refresh_representative_row_metadata(row_index)

        if self.representative_table.rowCount() > 0:
            self.representative_table.selectRow(0)
        self._refresh_representative_color_cells()
        self.representative_table_status_label.setText(
            f"Loaded {self.representative_table.rowCount()} saved representative "
            "structure(s) from the active contrast-mode distribution."
        )
        self._set_output_path_fields(
            workflow_dir=self._current_output_dir(),
            representatives_dir=contrast_dir,
            screening_dir=screening_dir,
            summary_path=contrast_dir / "selection_summary.json",
        )

    @Slot()
    def _redraw_trace_plot(self) -> None:
        for axis in list(self.trace_figure.axes):
            try:
                axis.set_xscale("linear")
                axis.set_yscale("linear")
            except Exception:
                continue
        self.trace_figure.clear()
        has_experimental = self._experimental_summary is not None
        has_generated = bool(self._generated_trace_profiles)
        self._update_trace_control_state()
        if self.representative_table.rowCount() > 0:
            self._refresh_representative_color_cells()
        if not has_experimental and not has_generated:
            axis = self.trace_figure.add_subplot(111)
            axis.text(
                0.5,
                0.55,
                "Choose or inherit an experimental data file to preview it here.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.text(
                0.5,
                0.42,
                "Generated contrast traces will appear after the contrast Debye "
                "workflow is built or reloaded.",
                ha="center",
                va="center",
                transform=axis.transAxes,
                alpha=0.7,
            )
            axis.set_axis_off()
            self.trace_canvas.draw_idle()
            return

        base_axis = self.trace_figure.add_subplot(111)
        experimental_axis = base_axis if has_experimental else None
        generated_axis = (
            base_axis if has_generated and not has_experimental else None
        )
        plotted_lines: list[object] = []
        experimental_q_values: np.ndarray | None = None
        experimental_intensities: np.ndarray | None = None
        if (
            experimental_axis is not None
            and self._experimental_summary is not None
        ):
            experimental_q_values = np.asarray(
                self._experimental_summary.q_values,
                dtype=float,
            )
            experimental_intensities = np.asarray(
                self._experimental_summary.intensities,
                dtype=float,
            )
            (experimental_line,) = experimental_axis.plot(
                experimental_q_values,
                experimental_intensities,
                label="Experimental data",
                color="#111827",
                alpha=0.35,
                linewidth=1.3,
            )
            plotted_lines.append(experimental_line)
            selected_mask = self._selected_q_mask(experimental_q_values)
            if selected_mask is not None and np.any(selected_mask):
                if not np.all(selected_mask):
                    (selected_line,) = experimental_axis.plot(
                        experimental_q_values[selected_mask],
                        experimental_intensities[selected_mask],
                        label="Selected q-range",
                        color="#111827",
                        linewidth=1.8,
                    )
                    plotted_lines.append(selected_line)
                else:
                    experimental_line.set_alpha(1.0)
                    experimental_line.set_linewidth(1.8)
                    experimental_line.set_label("Experimental data")
            else:
                experimental_axis.text(
                    0.5,
                    0.08,
                    "Selected q-range does not overlap the loaded experimental data.",
                    transform=experimental_axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize="small",
                )
            self._apply_trace_axis_style(
                experimental_axis,
                is_generated_axis=False,
            )

        generated_profiles = list(self._generated_trace_profiles)
        if has_experimental and has_generated:
            generated_axis = base_axis.twinx()

        if generated_axis is not None:
            representative_color_lookup: dict[str, str] = {}
            default_generated_colors = self._trace_colors(
                len(generated_profiles)
            )
            for row in range(self.representative_table.rowCount()):
                metadata = self._representative_row_metadata(row)
                display_label = str(
                    metadata.get("display_label") or ""
                ).strip()
                if display_label:
                    representative_color_lookup[display_label] = (
                        self._representative_trace_color(row)
                    )
            for index, (label, q_values, intensity) in enumerate(
                generated_profiles
            ):
                default_color = (
                    to_hex(default_generated_colors[index])
                    if index < len(default_generated_colors)
                    else "#1f77b4"
                )
                color = representative_color_lookup.get(
                    str(label).strip(),
                    _normalized_hex_color(default_color),
                )
                (line,) = generated_axis.plot(
                    np.asarray(q_values, dtype=float),
                    np.asarray(intensity, dtype=float),
                    label=label,
                    color=color,
                    linewidth=1.5,
                    alpha=0.95,
                    visible=self._generated_traces_visible,
                )
                plotted_lines.append(line)
            self._apply_trace_axis_style(
                generated_axis,
                is_generated_axis=True,
            )

        if experimental_axis is not None:
            if has_generated and generated_axis is not None:
                self._normalize_trace_axis(
                    experimental_axis,
                    generated_axis,
                )
                experimental_axis.set_ylabel(
                    "Experimental Intensity (arb. units)"
                )
                generated_axis.set_ylabel("Model Intensity (arb. units)")
                base_axis.set_title("Experimental Data and Contrast Traces")
            else:
                experimental_axis.set_ylabel("Intensity (arb. units)")
                base_axis.set_title("Experimental Data Preview")
        elif generated_axis is not None:
            generated_axis.set_ylabel("Model Intensity (arb. units)")
            base_axis.set_title("Contrast Trace Preview")

        if experimental_q_values is not None:
            if generated_profiles and self._generated_traces_visible:
                self.trace_plot_status_label.setText(
                    f"Previewing experimental data plus {len(generated_profiles)} "
                    "generated contrast trace(s)."
                )
            elif generated_profiles:
                self.trace_plot_status_label.setText(
                    f"Previewing experimental data with {len(generated_profiles)} "
                    "hidden generated contrast trace(s)."
                )
            else:
                self.trace_plot_status_label.setText(
                    f"Previewing {len(experimental_q_values)} q points from "
                    f"{self._experimental_summary.path}."
                )
        else:
            if self._generated_traces_visible:
                self.trace_plot_status_label.setText(
                    f"Previewing {len(generated_profiles)} generated contrast trace(s)."
                )
            else:
                self.trace_plot_status_label.setText(
                    f"Previewing {len(generated_profiles)} hidden generated contrast "
                    "trace(s)."
                )

        positive_q = True
        experimental_positive_i = True
        generated_positive_i = True
        if experimental_q_values is not None:
            positive_q = positive_q and bool(
                np.all(experimental_q_values > 0.0)
            )
            experimental_positive_i = experimental_positive_i and bool(
                np.all(experimental_intensities > 0.0)
            )
        for _label, q_values, intensity in generated_profiles:
            q_array = np.asarray(q_values, dtype=float)
            intensity_array = np.asarray(intensity, dtype=float)
            positive_q = positive_q and bool(np.all(q_array > 0.0))
            generated_positive_i = generated_positive_i and bool(
                np.all(intensity_array > 0.0)
            )

        if self.trace_log_x_checkbox.isChecked() and positive_q:
            base_axis.set_xscale("log")
            if generated_axis is not None and generated_axis is not base_axis:
                generated_axis.set_xscale("log")
        if experimental_axis is not None:
            experimental_axis.set_yscale(
                "log"
                if self.trace_log_y_checkbox.isChecked()
                and experimental_positive_i
                else "linear"
            )
        if generated_axis is not None:
            generated_axis.set_yscale(
                "log"
                if self.trace_log_y_checkbox.isChecked()
                and generated_positive_i
                else "linear"
            )

        if (
            self.trace_q_range_button.isChecked()
            and generated_axis is not None
        ):
            self._autoscale_to_generated_trace_range(
                experimental_axis,
                generated_axis,
            )

        anchor_axis = experimental_axis or generated_axis
        if (
            anchor_axis is not None
            and plotted_lines
            and self.trace_legend_toggle_button.isChecked()
        ):
            anchor_axis.legend(
                plotted_lines,
                [line.get_label() for line in plotted_lines],
                loc="best",
            )
        self.trace_figure.tight_layout()
        self.trace_canvas.draw_idle()

    def _apply_trace_axis_style(
        self, axis, *, is_generated_axis: bool
    ) -> None:
        if not is_generated_axis or self._experimental_summary is None:
            axis.set_xlabel("q (Å⁻¹)")
        if not is_generated_axis:
            axis.set_ylabel("Intensity (arb. units)")
        axis.grid(True, alpha=0.25)

    def _normalize_trace_axis(self, experimental_axis, generated_axis) -> None:
        if self._experimental_summary is None:
            return
        generated_lines = [
            line for line in generated_axis.get_lines() if line.get_visible()
        ]
        if not generated_lines:
            return
        exp_q = np.asarray(self._experimental_summary.q_values, dtype=float)
        exp_i = np.asarray(self._experimental_summary.intensities, dtype=float)
        filtered_q = exp_q[np.isfinite(exp_q) & np.isfinite(exp_i)]
        filtered_i = exp_i[np.isfinite(exp_q) & np.isfinite(exp_i)]
        if filtered_q.size == 0 or filtered_i.size == 0:
            return

        generated_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in generated_lines
            ]
        )
        generated_i = np.concatenate(
            [
                np.asarray(line.get_ydata(orig=False), dtype=float)
                for line in generated_lines
            ]
        )
        overlap_mask = (filtered_q >= float(np.nanmin(generated_q))) & (
            filtered_q <= float(np.nanmax(generated_q))
        )
        if np.any(overlap_mask):
            filtered_i = filtered_i[overlap_mask]
        generated_i = generated_i[np.isfinite(generated_i)]
        filtered_i = filtered_i[np.isfinite(filtered_i)]
        if self.trace_log_y_checkbox.isChecked():
            filtered_i = filtered_i[filtered_i > 0.0]
            generated_i = generated_i[generated_i > 0.0]
        if filtered_i.size == 0 or generated_i.size == 0:
            return
        left_limits = experimental_axis.get_ylim()
        right_limits = self._aligned_trace_y_limits(
            left_limits,
            float(np.nanmin(filtered_i)),
            float(np.nanmax(filtered_i)),
            float(np.nanmin(generated_i)),
            float(np.nanmax(generated_i)),
            log_scale=self.trace_log_y_checkbox.isChecked(),
        )
        generated_axis.set_ylim(right_limits)

    def _autoscale_to_generated_trace_range(
        self,
        experimental_axis,
        generated_axis,
    ) -> None:
        generated_lines = [
            line for line in generated_axis.get_lines() if line.get_visible()
        ]
        if not generated_lines:
            return
        generated_q = np.concatenate(
            [
                np.asarray(line.get_xdata(orig=False), dtype=float)
                for line in generated_lines
            ]
        )
        generated_q = generated_q[np.isfinite(generated_q)]
        if generated_q.size == 0:
            return
        q_min = float(np.nanmin(generated_q))
        q_max = float(np.nanmax(generated_q))
        generated_axis.set_xlim(q_min, q_max)
        if experimental_axis is not None:
            experimental_axis.set_xlim(q_min, q_max)
            self._autoscale_trace_axis_y(experimental_axis, q_min, q_max)
            self._normalize_trace_axis(experimental_axis, generated_axis)
            return
        self._autoscale_trace_axis_y(generated_axis, q_min, q_max)

    def _autoscale_trace_axis_y(
        self,
        axis,
        q_min: float,
        q_max: float,
    ) -> None:
        y_segments: list[np.ndarray] = []
        log_scale = self.trace_log_y_checkbox.isChecked()
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
    def _aligned_trace_y_limits(
        left_limits: tuple[float, float],
        experimental_min: float,
        experimental_max: float,
        generated_min: float,
        generated_max: float,
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
                    generated_min,
                    generated_max,
                )
                <= 0.0
            ):
                log_scale = False
        if not log_scale:
            left_low, left_high = left_limits
            exp_low, exp_high = sorted((experimental_min, experimental_max))
            model_low, model_high = sorted((generated_min, generated_max))
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
        exp_logs = np.log10(
            np.asarray(
                sorted((experimental_min, experimental_max)),
                dtype=float,
            )
        )
        model_logs = np.log10(
            np.asarray(sorted((generated_min, generated_max)), dtype=float)
        )
        if np.isclose(left_logs[1], left_logs[0]) or np.isclose(
            exp_logs[1],
            exp_logs[0],
        ):
            return generated_min / 1.2, generated_max * 1.2
        p0 = (exp_logs[0] - left_logs[0]) / (left_logs[1] - left_logs[0])
        p1 = (exp_logs[1] - left_logs[0]) / (left_logs[1] - left_logs[0])
        if np.isclose(p1, p0):
            return generated_min / 1.2, generated_max * 1.2
        delta = (model_logs[1] - model_logs[0]) / (p1 - p0)
        right_low_log = model_logs[0] - p0 * delta
        right_high_log = right_low_log + delta
        return 10**right_low_log, 10**right_high_log

    @Slot()
    def _toggle_generated_traces(self) -> None:
        if not self._generated_trace_profiles:
            return
        self._generated_traces_visible = not self._generated_traces_visible
        self._redraw_trace_plot()

    @Slot()
    def _validate_clusters(self) -> None:
        count = self._populate_cluster_inventory(log_to_console=True)
        self.statusBar().showMessage(
            f"Validated {count} recognized cluster bins"
            if count
            else "No recognized cluster bins were found"
        )

    @Slot()
    def _inspect_selected_cluster(self) -> None:
        row = self.recognized_clusters_table.currentRow()
        if row < 0:
            self._append_console_line(
                "No recognized cluster row is selected for inspection."
            )
            return
        values = [
            (
                self.recognized_clusters_table.item(row, column).text()
                if self.recognized_clusters_table.item(row, column) is not None
                else ""
            )
            for column in range(self.recognized_clusters_table.columnCount())
        ]
        source_dir = ""
        first_item = self.recognized_clusters_table.item(row, 0)
        if first_item is not None:
            source_dir = str(
                first_item.data(Qt.ItemDataRole.UserRole) or ""
            ).strip()
        self._append_console_line(
            "Selected cluster bin:\n"
            f"  Cluster: {values[0]}\n"
            f"  Stoichiometry: {values[1]}\n"
            f"  Type: {values[2]}\n"
            f"  Count: {values[3]}\n"
            f"  Weight: {values[4]}\n"
            f"  Atom %: {values[5]}\n"
            f"  Representative: {values[6]}\n"
            f"  Source directory: {source_dir or 'Unavailable'}"
        )
        self.statusBar().showMessage("Recognized cluster details logged")

    @Slot()
    def _add_manual_representatives(self) -> None:
        if not self._confirm_representative_list_edit(action_label="Adding"):
            return
        start_dir = (
            str(self._current_clusters_dir())
            if self._current_clusters_dir() is not None
            else str(Path.cwd())
        )
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Add Representative Structures",
            start_dir,
            "Structure files (*.pdb *.xyz);;All files (*)",
        )
        if not file_paths:
            return
        for file_path in file_paths:
            path = Path(file_path).expanduser().resolve()
            row_index = self.representative_table.rowCount()
            self.representative_table.insertRow(row_index)
            stoichiometry = (
                path.parent.name
                if path.parent.name and path.parent.name != "."
                else path.stem
            )
            custom_motif = f"custom_{path.stem}".lower().replace(" ", "_")
            values = (
                stoichiometry,
                "Manual",
                path.name,
                "",
                "Pending",
                "Pending",
                "Pending",
                "Not computed",
                "Custom representative staged",
            )
            for column, value in enumerate(values):
                if column == _REPRESENTATIVE_COLUMN_COLOR:
                    item = self._build_representative_color_item(
                        self._representative_default_color(row_index)
                    )
                else:
                    item = QTableWidgetItem(value)
                if column == _REPRESENTATIVE_COLUMN_FILE:
                    item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.representative_table.setItem(row_index, column, item)
            self._set_representative_row_metadata(
                row_index,
                {
                    "source_kind": "manual",
                    "display_label": stoichiometry,
                    "structure": stoichiometry,
                    "motif": custom_motif,
                    "representative_file": str(path),
                    "trace_color": self._representative_default_color(
                        row_index
                    ),
                    "custom_trace_color": False,
                },
            )
        self._refresh_representative_color_cells()
        self.representative_table.selectRow(
            self.representative_table.rowCount() - 1
        )
        self.representative_table_status_label.setText(
            f"Staged {self.representative_table.rowCount()} representative "
            "structure entries for the future contrast workflow."
        )
        self._redraw_trace_plot()
        self._sync_visualizer_preview()
        self._append_console_line(
            f"Added {len(file_paths)} custom representative structure file(s) to the contrast viewer."
        )

    @Slot()
    def _remove_selected_representative(self) -> None:
        row = self.representative_table.currentRow()
        if row < 0:
            self._append_console_line(
                "No representative structure row is selected for removal."
            )
            return
        if not self._confirm_representative_list_edit(action_label="Removing"):
            return
        removed_name = (
            self.representative_table.item(
                row, _REPRESENTATIVE_COLUMN_FILE
            ).text()
            if self.representative_table.item(row, _REPRESENTATIVE_COLUMN_FILE)
            is not None
            else "representative"
        )
        self.representative_table.removeRow(row)
        self._refresh_representative_color_cells()
        self._redraw_trace_plot()
        self.representative_table_status_label.setText(
            "Representative structure entry removed from the scaffold."
        )
        self._sync_visualizer_preview()
        self._append_console_line(
            f"Removed representative structure entry {removed_name}."
        )

    @Slot()
    def _inspect_selected_representative(self) -> None:
        row = self.representative_table.currentRow()
        if row < 0:
            self._append_console_line(
                "No representative structure row is selected for inspection."
            )
            return
        self._sync_visualizer_preview()
        self.right_scroll_area.ensureWidgetVisible(self.structure_viewer)
        values = [
            (
                self.representative_table.item(row, column).text()
                if self.representative_table.item(row, column) is not None
                else ""
            )
            for column in range(self.representative_table.columnCount())
        ]
        structure_path = ""
        file_item = self.representative_table.item(
            row, _REPRESENTATIVE_COLUMN_FILE
        )
        if file_item is not None:
            structure_path = str(
                file_item.data(Qt.ItemDataRole.UserRole) or ""
            ).strip()
        self._append_console_line(
            "Selected representative structure:\n"
            f"  Stoichiometry: {values[0]}\n"
            f"  Selection: {values[1]}\n"
            f"  File: {values[2]}\n"
            f"  Trace color: {values[3]}\n"
            f"  Solvent coordination: {values[4]}\n"
            f"  Mesh volume: {values[5]}\n"
            f"  Electron density: {values[6]}\n"
            f"  Trace status: {values[7]}\n"
            f"  Notes: {values[8]}\n"
            f"  Source path: {structure_path or 'Unavailable'}"
        )
        self.statusBar().showMessage(
            "Representative structure details loaded into the preview"
        )

    @Slot()
    def _sync_visualizer_preview(self) -> None:
        row = self.representative_table.currentRow()
        if row < 0:
            self.structure_viewer.set_preview(
                None,
                show_mesh=self.show_mesh_checkbox.isChecked(),
                show_legend=self.show_structure_legend_checkbox.isChecked(),
            )
            self.visualizer_status_label.setText(
                "Select a representative structure row to preview its saved file, "
                "retained mesh overlay, and electron-density metadata."
            )
            self.visualizer_details_box.setPlainText(
                "Viewer status\n"
                "  Selected representative: none\n"
                f"  Mesh overlay: {'On' if self.show_mesh_checkbox.isChecked() else 'Off'}\n"
                f"  Legend: {'On' if self.show_structure_legend_checkbox.isChecked() else 'Off'}"
            )
            return

        metadata = self._refresh_representative_row_metadata(row)
        file_item = self.representative_table.item(
            row, _REPRESENTATIVE_COLUMN_FILE
        )
        structure_path = (
            str(file_item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if file_item is not None
            else str(metadata.get("representative_file") or "").strip()
        )
        if not structure_path:
            self.structure_viewer.set_preview(
                None,
                show_mesh=self.show_mesh_checkbox.isChecked(),
                show_legend=self.show_structure_legend_checkbox.isChecked(),
            )
            self.visualizer_status_label.setText(
                "The selected representative row does not have a readable source file."
            )
            self.visualizer_details_box.setPlainText(
                "Viewer status\n" "  Source path: unavailable"
            )
            return

        display_label = (
            str(metadata.get("display_label") or "").strip()
            or self.representative_table.item(
                row,
                _REPRESENTATIVE_COLUMN_STOICHIOMETRY,
            ).text()
        )
        notes_text = (
            self.representative_table.item(
                row, _REPRESENTATIVE_COLUMN_NOTES
            ).text()
            if self.representative_table.item(
                row, _REPRESENTATIVE_COLUMN_NOTES
            )
            is not None
            else ""
        )
        mesh_json_path = (
            str(metadata.get("mesh_json_path") or "").strip() or None
        )
        density_json_path = (
            str(metadata.get("density_json_path") or "").strip() or None
        )
        try:
            preview = load_contrast_representative_preview(
                structure_path,
                display_label=display_label,
                mesh_json_path=mesh_json_path,
                density_json_path=density_json_path,
                notes=notes_text,
            )
        except Exception as exc:
            self.structure_viewer.set_preview(
                None,
                show_mesh=self.show_mesh_checkbox.isChecked(),
                show_legend=self.show_structure_legend_checkbox.isChecked(),
            )
            self.visualizer_status_label.setText(
                f"Unable to load the selected representative preview: {exc}"
            )
            self.visualizer_details_box.setPlainText(
                "Viewer status\n"
                f"  Source path: {structure_path}\n"
                f"  Error: {exc}"
            )
            return

        self.structure_viewer.set_preview(
            preview,
            show_mesh=self.show_mesh_checkbox.isChecked(),
            show_legend=self.show_structure_legend_checkbox.isChecked(),
        )
        if preview.has_mesh and self.show_mesh_checkbox.isChecked():
            self.visualizer_status_label.setText(
                f"Previewing {preview.display_label} with the retained mesh overlay."
            )
        elif preview.has_mesh:
            self.visualizer_status_label.setText(
                f"Previewing {preview.display_label}; saved retained mesh is available but hidden."
            )
        else:
            self.visualizer_status_label.setText(
                f"Previewing {preview.display_label}; saved retained mesh is not available yet."
            )
        self.visualizer_details_box.setPlainText(preview.details_text())

    def _placeholder_action(
        self,
        *,
        label: str,
        console_message: str,
        progress_value: int,
    ) -> None:
        self.workflow_progress_bar.setValue(progress_value)
        self.workflow_progress_label.setText(
            f"Progress: {label} placeholder triggered"
        )
        self._append_console_line(console_message)
        self.statusBar().showMessage(label)

    def _confirm_representative_list_edit(self, *, action_label: str) -> bool:
        response = QMessageBox.question(
            self,
            "Representative List Change",
            (
                f"{action_label} representative structures will change the current "
                "representative list.\n\n"
                "If you add or remove representatives, the prior weights will no "
                "longer match until they are reinitialized, including a new zero "
                "weight for any added representative. If contrast SAXS components "
                "have already been built, rerun the contrast build after the "
                "representative list changes.\n\n"
                "Continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _leave_saved_distribution_view(self) -> None:
        if (
            self._active_distribution_id is None
            and self._distribution_root_dir is None
            and self._contrast_artifact_dir_override is None
        ):
            return
        self._active_distribution_id = None
        self._distribution_root_dir = None
        self._contrast_artifact_dir_override = None
        self._generated_trace_profiles = []
        self._generated_traces_visible = True
        self._refresh_output_path()
        self._redraw_trace_plot()

    def _contrast_project_settings(self) -> ProjectSettings:
        project_dir = self.project_dir_edit.text().strip()
        if not project_dir:
            raise ValueError(
                "Contrast Mode requires an active project folder."
            )
        resolved_project_dir = Path(project_dir).expanduser().resolve()
        manager = SAXSProjectManager()
        try:
            settings = manager.load_project(project_dir)
        except Exception:
            settings = ProjectSettings(
                project_name=resolved_project_dir.name,
                project_dir=str(resolved_project_dir),
            )
        settings.project_name = resolved_project_dir.name
        settings.project_dir = str(resolved_project_dir)
        clusters_dir = self._current_clusters_dir()
        settings.clusters_dir = (
            None if clusters_dir is None else str(clusters_dir)
        )
        experimental_file = self._current_experimental_file()
        if experimental_file is not None:
            settings.experimental_data_path = str(experimental_file)
            settings.copied_experimental_data_file = None
        q_min = self._current_q_min()
        q_max = self._current_q_max()
        if q_min is not None:
            settings.q_min = q_min
        if q_max is not None:
            settings.q_max = q_max
        template_name = str(self.template_edit.text()).strip()
        if template_name:
            settings.selected_model_template = template_name
        exclude_elements = {
            str(element).strip().upper()
            for element in (settings.exclude_elements or [])
            if str(element).strip()
        }
        exclude_elements.discard("H")
        if self.exclude_hydrogen_checkbox.isChecked():
            exclude_elements.add("H")
        settings.exclude_elements = sorted(exclude_elements)
        settings.component_build_mode = COMPONENT_BUILD_MODE_CONTRAST
        return settings

    def _run_representative_analysis_task(
        self,
        settings: ProjectSettings,
        sampler_settings: ContrastRepresentativeSamplerSettings,
        *,
        progress_callback,
        log_callback,
    ) -> ContrastRepresentativeSelectionResult:
        manager = SAXSProjectManager()
        (
            cluster_inventory,
            _predicted_dataset_file,
            predicted_component_count,
        ) = manager.contrast_cluster_inventory(settings)
        if (
            settings.use_predicted_structure_weights
            and predicted_component_count > 0
        ):
            log_callback(
                "Predicted Structures mode is active, so the contrast representative "
                f"analysis will include {predicted_component_count} predicted "
                "structure bin(s) from the current Cluster Dynamics ML bundle."
            )
        if settings.resolved_clusters_dir is None:
            raise ValueError(
                "Contrast representative analysis requires a clusters directory."
            )
        return analyze_contrast_representatives(
            settings.project_dir,
            settings.resolved_clusters_dir,
            cluster_bins=tuple(cluster_inventory.cluster_bins),
            sampler_settings=sampler_settings,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    def _active_progress_format(self) -> str:
        if self._active_workflow_task == "analyze_representatives":
            return "%v / %m cluster bins analyzed"
        if self._active_workflow_task == "compute_density":
            return "%v / %m representative bins processed"
        return "%v / %m steps"

    def _set_workflow_controls_busy(self, busy: bool) -> None:
        enabled = not busy
        for widget in (
            self.add_representative_button,
            self.remove_representative_button,
            self.inspect_representative_button,
            self.refresh_context_button,
            self.log_context_button,
            self.analyze_representatives_button,
            self.compute_density_button,
            self.build_components_button,
            self.sampler_settings_toggle_button,
            self.sampler_enabled_checkbox,
            self.sampler_full_scan_threshold_spin,
            self.sampler_distribution_samples_spin,
            self.sampler_minimum_samples_spin,
            self.sampler_max_samples_spin,
            self.sampler_batch_size_spin,
            self.sampler_stratify_checkbox,
            self.sampler_seed_spin,
            self.sampler_patience_spin,
            self.sampler_tolerance_spin,
            self.solvent_method_combo,
            self.solvent_preset_combo,
            self.save_custom_solvent_button,
            self.delete_custom_solvent_button,
            self.solvent_formula_edit,
            self.solvent_density_spin,
            self.exclude_hydrogen_checkbox,
            self.direct_density_spin,
            self.reference_solvent_file_edit,
            self.reference_solvent_browse_button,
        ):
            widget.setEnabled(enabled)
        if enabled:
            self._sync_density_method_controls()

    def _start_workflow_task(
        self,
        *,
        task_name: str,
        start_message: str,
        console_message: str,
        initial_progress_label: str,
        task: Callable[..., object],
        on_success: Callable[[object], None],
    ) -> bool:
        if self._workflow_thread is not None:
            self._append_console_line(
                "A contrast workflow task is already running. Please wait "
                "for it to finish before starting another task."
            )
            self.statusBar().showMessage(
                "Contrast workflow task already running"
            )
            return False

        self._active_workflow_task = task_name
        self._workflow_completion_handler = on_success
        self._set_workflow_controls_busy(True)
        self.workflow_progress_bar.setRange(0, 1)
        self.workflow_progress_bar.setValue(0)
        self.workflow_progress_bar.setFormat(self._active_progress_format())
        self.workflow_progress_label.setText(initial_progress_label)
        self._append_console_line(console_message)
        self.statusBar().showMessage(start_message)

        self._workflow_thread = QThread(self)
        self._workflow_worker = _ContrastWorkflowWorker(task)
        self._workflow_worker.moveToThread(self._workflow_thread)
        self._workflow_thread.started.connect(self._workflow_worker.run)
        self._workflow_worker.progress.connect(self._on_workflow_progress)
        self._workflow_worker.log.connect(self._append_console_line)
        self._workflow_worker.finished.connect(self._on_workflow_task_finished)
        self._workflow_worker.failed.connect(self._on_workflow_task_failed)
        self._workflow_worker.finished.connect(self._workflow_thread.quit)
        self._workflow_worker.failed.connect(self._workflow_thread.quit)
        self._workflow_thread.finished.connect(self._cleanup_workflow_thread)
        self._workflow_thread.finished.connect(
            self._workflow_thread.deleteLater
        )
        self._workflow_thread.finished.connect(
            self._workflow_worker.deleteLater
        )
        self._workflow_thread.start(QThread.Priority.LowPriority)
        return True

    @Slot(int, int, str)
    def _on_workflow_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        normalized_total = max(int(total), 1)
        normalized_value = min(max(int(processed), 0), normalized_total)
        self.workflow_progress_bar.setRange(0, normalized_total)
        self.workflow_progress_bar.setValue(normalized_value)
        self.workflow_progress_bar.setFormat(self._active_progress_format())
        if message:
            self.workflow_progress_label.setText(f"Progress: {message}")
            self.statusBar().showMessage(message)

    @Slot(object)
    def _on_workflow_task_finished(self, result: object) -> None:
        handler = self._workflow_completion_handler
        self._workflow_completion_handler = None
        if handler is not None:
            handler(result)
        self._set_workflow_controls_busy(False)

    @Slot(str)
    def _on_workflow_task_failed(self, message: str) -> None:
        task_name = self._active_workflow_task or "contrast workflow"
        friendly_name = task_name.replace("_", " ")
        self.workflow_progress_bar.setRange(0, 1)
        self.workflow_progress_bar.setValue(0)
        self.workflow_progress_bar.setFormat("%v / %m steps")
        self.workflow_progress_label.setText(
            f"Progress: {friendly_name} failed"
        )
        self._append_console_line(
            f"{friendly_name.capitalize()} failed: {message}"
        )
        self.statusBar().showMessage(f"{friendly_name.capitalize()} failed")
        self._set_workflow_controls_busy(False)

    def _cleanup_workflow_thread(self) -> None:
        self._workflow_worker = None
        self._workflow_thread = None
        self._workflow_completion_handler = None
        self._active_workflow_task = None

    @Slot()
    def _populate_representative_analysis_result(
        self,
        result: ContrastRepresentativeSelectionResult,
    ) -> None:
        self.representative_table.setRowCount(0)
        for row_index, bin_result in enumerate(result.bin_results):
            descriptor = bin_result.selected_candidate.descriptor
            self.representative_table.insertRow(row_index)
            values = (
                bin_result.display_label,
                "Auto",
                bin_result.copied_representative_file.name,
                "",
                (
                    f"Direct {descriptor.direct_solvent_atom_count}; "
                    f"Outer {descriptor.outer_solvent_atom_count}"
                ),
                "Pending",
                "Pending",
                "Selection ready",
                (
                    f"Score {bin_result.selected_candidate.score_total:.4f} | "
                    f"{'MC' if bin_result.selection_strategy == 'monte_carlo_sampling' else 'Full'} "
                    f"target {bin_result.distribution_sample_count}, "
                    f"candidates {bin_result.sampled_candidate_count}/{bin_result.file_count} | "
                    f"{bin_result.screening_json_path.name}"
                ),
            )
            for column, value in enumerate(values):
                if column == _REPRESENTATIVE_COLUMN_COLOR:
                    item = self._build_representative_color_item(
                        self._representative_default_color(row_index)
                    )
                else:
                    item = QTableWidgetItem(value)
                if column == _REPRESENTATIVE_COLUMN_FILE:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(bin_result.copied_representative_file),
                    )
                if column == _REPRESENTATIVE_COLUMN_NOTES:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(bin_result.screening_json_path),
                    )
                self.representative_table.setItem(row_index, column, item)
            self._set_representative_row_metadata(
                row_index,
                {
                    "source_kind": "auto",
                    "display_label": bin_result.display_label,
                    "structure": bin_result.structure,
                    "motif": bin_result.motif,
                    "representative_file": str(
                        bin_result.copied_representative_file
                    ),
                    "screening_json_path": str(bin_result.screening_json_path),
                    "screening_table_path": str(
                        bin_result.screening_table_path
                    ),
                    "trace_color": self._representative_default_color(
                        row_index
                    ),
                    "custom_trace_color": False,
                },
            )
            self._refresh_representative_row_metadata(row_index)
        if result.bin_results:
            self.representative_table.selectRow(0)
        self._refresh_representative_color_cells()
        self.representative_table_status_label.setText(
            f"Loaded {len(result.bin_results)} selected representative "
            f"structure(s) from {result.output_dir}."
        )
        self._sync_visualizer_preview()
        self._set_output_path_fields(
            workflow_dir=result.output_dir.parent,
            representatives_dir=result.output_dir,
            screening_dir=result.screening_dir,
            summary_path=result.summary_json_path,
        )
        self._flush_ui()

    @Slot()
    def _run_representative_analysis(self) -> None:
        project_dir = self.project_dir_edit.text().strip()
        clusters_dir = self.clusters_dir_edit.text().strip()
        if not project_dir:
            self._append_console_line(
                "Representative screening needs an active project folder so the "
                "contrast outputs can be saved into the project."
            )
            self.statusBar().showMessage(
                "Contrast representative analysis requires a project"
            )
            return
        if not clusters_dir:
            self._append_console_line(
                "Representative screening needs a reference clusters folder."
            )
            self.statusBar().showMessage(
                "Contrast representative analysis requires clusters"
            )
            return
        try:
            settings = self._contrast_project_settings()
        except Exception as exc:
            self._append_console_line(str(exc))
            self.statusBar().showMessage(
                "Contrast representative analysis settings are incomplete"
            )
            return
        sampler_settings = self._representative_sampler_settings()

        self._leave_saved_distribution_view()
        self._density_result = None
        self._start_workflow_task(
            task_name="analyze_representatives",
            start_message="Analyzing representative structures...",
            console_message=(
                "Starting representative-structure analysis for the recognized "
                "cluster bins in the contrast workflow."
            ),
            initial_progress_label="Progress: analyzing representative structures",
            task=lambda progress_callback, log_callback: self._run_representative_analysis_task(
                settings,
                sampler_settings,
                progress_callback=progress_callback,
                log_callback=log_callback,
            ),
            on_success=self._on_representative_analysis_finished,
        )

    def _on_representative_analysis_finished(
        self,
        result: object,
    ) -> None:
        if not isinstance(result, ContrastRepresentativeSelectionResult):
            self._append_console_line(
                "Representative-structure analysis returned an unexpected result."
            )
            self.statusBar().showMessage(
                "Contrast representative analysis failed"
            )
            return
        self._representative_analysis_result = result
        self._density_result = None
        self._populate_representative_analysis_result(result)
        self._set_workflow_stage_progress(
            2,
            detail=(
                "representative analysis complete; "
                f"{len(result.bin_results)} bin(s) selected"
            ),
        )
        self._append_console_line(result.summary_text())
        self._append_console_line(
            f"Representative screening outputs were written to {result.output_dir}."
        )
        self.statusBar().showMessage(
            "Contrast representative analysis complete"
        )

    @Slot()
    def _compute_electron_density(self) -> None:
        representative_result = self._representative_analysis_result
        if representative_result is None:
            self._append_console_line(
                "Run Analyze Representative Structures before computing "
                "contrast geometry and electron density."
            )
            self.statusBar().showMessage(
                "Representative analysis is required first"
            )
            return
        try:
            density_settings = self._current_density_settings()
        except ValueError as exc:
            self._append_console_line(str(exc))
            self.statusBar().showMessage(
                "Electron-density settings are incomplete"
            )
            return

        self._start_workflow_task(
            task_name="compute_density",
            start_message="Computing contrast geometry and electron density...",
            console_message=(
                "Starting contrast geometry construction and electron-density "
                "estimation for the selected representative structures."
            ),
            initial_progress_label=(
                "Progress: computing contrast geometry and electron density"
            ),
            task=lambda progress_callback, log_callback: compute_contrast_geometry_and_electron_density(
                representative_result,
                density_settings,
                progress_callback=progress_callback,
                log_callback=log_callback,
            ),
            on_success=self._on_density_finished,
        )

    def _on_density_finished(self, result: object) -> None:
        if not isinstance(result, ContrastGeometryDensityResult):
            self._append_console_line(
                "Contrast electron-density calculation returned an unexpected result."
            )
            self.statusBar().showMessage(
                "Contrast electron-density calculation failed"
            )
            return
        self._density_result = result
        self._generated_trace_profiles = []
        self._generated_traces_visible = True
        self._clear_representative_trace_metadata()
        self._refresh_representative_metadata_from_saved_outputs()
        self._redraw_trace_plot()
        self._sync_visualizer_preview()
        self._set_workflow_stage_progress(
            3,
            detail="geometry and electron density complete",
        )
        self._set_output_path_fields(
            workflow_dir=result.output_dir.parent,
            representatives_dir=result.output_dir,
            screening_dir=result.output_dir / "screening",
            summary_path=result.summary_json_path,
        )
        self._append_console_line(result.summary_text())
        self._append_console_line(
            f"Contrast density outputs were written to {result.density_dir}."
        )
        self._append_console_line(
            "Existing contrast traces were cleared. Rebuild the contrast SAXS "
            "components to apply the updated solvent electron density."
        )
        self.statusBar().showMessage(
            "Contrast geometry and electron density complete"
        )

    @Slot()
    def start_contrast_component_build(self) -> None:
        self._build_contrast_components()

    @Slot()
    def _build_contrast_components(self) -> None:
        representative_result = self._representative_analysis_result
        if representative_result is None:
            self._append_console_line(
                "Run Analyze Representative Structures before building "
                "contrast SAXS components."
            )
            self.statusBar().showMessage(
                "Representative analysis is required first"
            )
            return
        density_result = self._density_result
        if density_result is None:
            self._append_console_line(
                "Run Compute Electron Density before building contrast SAXS "
                "components."
            )
            self.statusBar().showMessage(
                "Electron-density calculation is required first"
            )
            return

        try:
            settings = self._contrast_project_settings()
        except Exception as exc:
            self._append_console_line(
                f"Contrast SAXS component build could not start: {exc}"
            )
            self.statusBar().showMessage(
                "Contrast SAXS component build could not start"
            )
            return

        self._start_workflow_task(
            task_name="build_components",
            start_message="Building contrast Debye SAXS components...",
            console_message="Starting the contrast-specific Debye SAXS component build.",
            initial_progress_label=(
                "Progress: building contrast Debye SAXS components"
            ),
            task=lambda progress_callback, log_callback: self._build_components_task(
                settings=settings,
                representative_result=representative_result,
                density_result=density_result,
                progress_callback=progress_callback,
                log_callback=log_callback,
            ),
            on_success=self._on_build_components_finished,
        )

    def _build_components_task(
        self,
        *,
        settings: ProjectSettings,
        representative_result: ContrastRepresentativeSelectionResult,
        density_result: ContrastGeometryDensityResult,
        progress_callback,
        log_callback,
    ) -> dict[str, object]:
        manager = SAXSProjectManager()
        build_result = (
            manager.build_contrast_scattering_components_from_results(
                settings,
                representative_result=representative_result,
                density_result=density_result,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        )
        artifact_paths = project_artifact_paths(
            settings,
            storage_mode="distribution",
            allow_legacy_fallback=False,
        )
        return {
            "settings": settings,
            "build_result": build_result,
            "artifact_paths": artifact_paths,
            "representative_result": representative_result,
            "density_result": density_result,
        }

    def _on_build_components_finished(self, result: object) -> None:
        if not isinstance(result, dict):
            self._append_console_line(
                "Contrast SAXS component build returned an unexpected result."
            )
            self.statusBar().showMessage(
                "Contrast SAXS component build failed"
            )
            return
        settings = result.get("settings")
        build_result = result.get("build_result")
        artifact_paths = result.get("artifact_paths")
        representative_result = result.get("representative_result")
        density_result = result.get("density_result")
        if not isinstance(settings, ProjectSettings):
            self._append_console_line(
                "Contrast SAXS component build did not return project settings."
            )
            self.statusBar().showMessage(
                "Contrast SAXS component build failed"
            )
            return
        if not hasattr(build_result, "component_entries") or not hasattr(
            artifact_paths, "distribution_id"
        ):
            self._append_console_line(
                "Contrast SAXS component build did not return complete artifacts."
            )
            self.statusBar().showMessage(
                "Contrast SAXS component build failed"
            )
            return
        if isinstance(
            representative_result, ContrastRepresentativeSelectionResult
        ):
            self._representative_analysis_result = representative_result
            self._populate_representative_analysis_result(
                representative_result
            )
        if isinstance(density_result, ContrastGeometryDensityResult):
            self._density_result = density_result
        self._last_build_distribution_id = artifact_paths.distribution_id
        self._load_generated_trace_profiles(
            settings, build_result.component_entries
        )
        debye_summary_path = (
            artifact_paths.contrast_dir / "debye" / "component_summary.json"
        )
        if debye_summary_path.is_file():
            debye_payload = json.loads(
                debye_summary_path.read_text(encoding="utf-8")
            )
            self._apply_saved_trace_metadata(
                list(debye_payload.get("trace_results") or [])
            )
        self._refresh_representative_metadata_from_saved_outputs()
        self._redraw_trace_plot()
        self._sync_visualizer_preview()
        self._set_workflow_stage_progress(
            5,
            detail="contrast Debye component build complete",
        )
        self._set_output_path_fields(
            workflow_dir=artifact_paths.root_dir,
            representatives_dir=artifact_paths.contrast_dir,
            screening_dir=artifact_paths.contrast_dir / "screening",
            summary_path=artifact_paths.contrast_dir
            / "debye"
            / "component_summary.json",
        )
        self._append_console_line(
            "Contrast Debye SAXS component build complete.\n"
            f"Distribution folder: {artifact_paths.root_dir}\n"
            f"Component map: {build_result.model_map_path}"
        )
        self.statusBar().showMessage("Contrast SAXS components built")
        self.contrast_components_built.emit(
            {
                "project_dir": settings.project_dir,
                "distribution_id": artifact_paths.distribution_id,
                "distribution_dir": str(artifact_paths.root_dir),
                "component_dir": str(artifact_paths.component_dir),
                "component_map_path": (
                    None
                    if build_result.model_map_path is None
                    else str(build_result.model_map_path)
                ),
            }
        )

    @Slot()
    def _export_plot_data(self) -> None:
        payload = self._trace_plot_export_payload()
        traces = list(payload.get("traces", []))
        if not traces:
            self._append_console_line(
                "No experimental or contrast trace data is currently available "
                "to export."
            )
            return
        default_path = self._current_output_dir() or (
            self._experimental_summary.path.parent
            if self._experimental_summary is not None
            else Path.cwd()
        )
        default_file = default_path / (
            f"contrast_plot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        output_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Contrast Plot Data",
            str(default_file),
            "CSV files (*.csv);;NumPy files (*.npy)",
        )
        if not output_path:
            return
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() not in {".csv", ".npy"}:
            destination = destination.with_suffix(".csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".csv":
            self._write_trace_plot_csv(destination, payload)
        else:
            np.save(destination, payload, allow_pickle=True)
        self._append_console_line(
            f"Exported the current contrast plot data to {destination}."
        )
        self.statusBar().showMessage("Contrast plot data exported")

    def _trace_plot_export_payload(self) -> dict[str, object]:
        traces: list[dict[str, object]] = []
        for axis_index, axis in enumerate(self.trace_figure.axes):
            for line in axis.get_lines():
                traces.append(
                    {
                        "series": str(line.get_label()),
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
                str(self.trace_figure.axes[0].get_title())
                if self.trace_figure.axes
                else ""
            ),
            "trace_color_scheme": str(
                self.trace_color_scheme_combo.currentData() or "default"
            ),
            "log_x": bool(self.trace_log_x_checkbox.isChecked()),
            "log_y": bool(self.trace_log_y_checkbox.isChecked()),
            "traces": traces,
        }

    @staticmethod
    def _write_trace_plot_csv(
        output_path: Path,
        payload: dict[str, object],
    ) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "series",
                    "axis_index",
                    "axis_ylabel",
                    "color",
                    "visible",
                    "x",
                    "y",
                ]
            )
            for trace in payload.get("traces", []):
                x_values = np.asarray(trace.get("x", []), dtype=float)
                y_values = np.asarray(trace.get("y", []), dtype=float)
                count = min(len(x_values), len(y_values))
                for index in range(count):
                    writer.writerow(
                        [
                            str(trace.get("series", "")),
                            int(trace.get("axis_index", 0)),
                            str(trace.get("axis_ylabel", "")),
                            str(trace.get("color", "")),
                            bool(trace.get("visible", True)),
                            f"{x_values[index]:.10g}",
                            f"{y_values[index]:.10g}",
                        ]
                    )

    def _rotate_visualizer(self, delta_azimuth: float, label: str) -> None:
        self.structure_viewer.rotate(delta_azimuth)
        self.statusBar().showMessage(label)

    @Slot(bool)
    def _toggle_visualizer_pan_mode(self, enabled: bool) -> None:
        self.structure_viewer.toggle_pan_mode()
        state = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Visualizer pan mode {state}")

    def _zoom_visualizer(self, factor: float, label: str) -> None:
        self.structure_viewer.zoom_by(factor)
        self.statusBar().showMessage(label)

    @Slot()
    def _reset_visualizer_view(self) -> None:
        was_pan_enabled = self.pan_button.isChecked()
        self.pan_button.blockSignals(True)
        self.pan_button.setChecked(False)
        self.pan_button.blockSignals(False)
        if was_pan_enabled:
            self.structure_viewer.toggle_pan_mode()
        self.structure_viewer.reset_view()
        self.statusBar().showMessage("Visualizer view reset")

    @Slot(float)
    def _on_q_range_controls_changed(self, _value: float) -> None:
        self._update_q_range_summary()
        self._redraw_trace_plot()

    @Slot()
    def _choose_experimental_data_file(self) -> None:
        start_dir = (
            str(self._current_experimental_file().parent)
            if self._current_experimental_file() is not None
            else str(self._current_output_dir() or Path.cwd())
        )
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose Experimental Data File",
            start_dir,
            "Data files (*.txt *.dat *.csv);;All files (*)",
        )
        if not selected_path:
            return
        self.experimental_data_edit.setText(
            str(Path(selected_path).expanduser().resolve())
        )
        self._schedule_preview_refresh()

    @Slot()
    def _choose_clusters_directory(self) -> None:
        start_dir = (
            str(self._current_clusters_dir())
            if self._current_clusters_dir() is not None
            else str(self._current_output_dir() or Path.cwd())
        )
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Choose Reference Clusters Folder",
            start_dir,
        )
        if not selected_path:
            return
        self.clusters_dir_edit.setText(
            str(Path(selected_path).expanduser().resolve())
        )
        self._schedule_preview_refresh()

    @Slot()
    def _restore_inherited_experimental_data_file(self) -> None:
        self.experimental_data_edit.setText(
            ""
            if self._launch_context.experimental_data_file is None
            else str(self._launch_context.experimental_data_file)
        )
        self._schedule_preview_refresh()

    @Slot()
    def _restore_inherited_clusters_directory(self) -> None:
        self.clusters_dir_edit.setText(
            ""
            if self._launch_context.clusters_dir is None
            else str(self._launch_context.clusters_dir)
        )
        self._schedule_preview_refresh()

    @Slot()
    def _restore_inherited_q_range(self) -> None:
        self._set_q_range_controls(
            self._launch_context.q_min,
            self._launch_context.q_max,
        )
        self._schedule_preview_refresh()

    def _append_console_line(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        existing = self.console_box.toPlainText().strip()
        entry = f"[{timestamp}] {message}"
        if existing:
            self.console_box.appendPlainText("")
        self.console_box.appendPlainText(entry)

    def _flush_ui(self) -> None:
        return


def _forget_open_window(window: ContrastModeMainWindow) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def launch_contrast_mode_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_clusters_dir: str | Path | None = None,
    initial_experimental_data_file: str | Path | None = None,
    initial_q_min: float | None = None,
    initial_q_max: float | None = None,
    initial_template_name: str | None = None,
    initial_distribution_id: str | None = None,
    initial_distribution_root_dir: str | Path | None = None,
    initial_contrast_artifact_dir: str | Path | None = None,
) -> ContrastModeMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ContrastModeMainWindow(
        initial_project_dir=(
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        ),
        initial_clusters_dir=(
            None
            if initial_clusters_dir is None
            else Path(initial_clusters_dir).expanduser().resolve()
        ),
        initial_experimental_data_file=(
            None
            if initial_experimental_data_file is None
            else Path(initial_experimental_data_file).expanduser().resolve()
        ),
        initial_q_min=initial_q_min,
        initial_q_max=initial_q_max,
        initial_template_name=initial_template_name,
        initial_distribution_id=initial_distribution_id,
        initial_distribution_root_dir=(
            None
            if initial_distribution_root_dir is None
            else Path(initial_distribution_root_dir).expanduser().resolve()
        ),
        initial_contrast_artifact_dir=(
            None
            if initial_contrast_artifact_dir is None
            else Path(initial_contrast_artifact_dir).expanduser().resolve()
        ),
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window
