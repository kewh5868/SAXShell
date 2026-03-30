from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import (
    ExtractedFrameFolderClusterAnalyzer,
    detect_frame_folder_mode,
    format_box_dimensions,
    format_search_mode_label,
    frame_folder_label,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.ui.trajectory_panel import ClusterTrajectoryPanel
from saxshell.clusterdynamics.dataset import (
    export_cluster_dynamics_colormap_csv,
)
from saxshell.clusterdynamics.report import (
    default_powerpoint_report_path,
    export_cluster_dynamicsai_report_pptx,
)
from saxshell.clusterdynamics.ui.main_window import (
    ClusterDynamicsDatasetPanel,
    ClusterDynamicsRunPanel,
    ClusterDynamicsTimePanel,
)
from saxshell.clusterdynamics.ui.plot_panel import ClusterDynamicsPlotPanel
from saxshell.saxs.project_manager import (
    PowerPointExportSettings,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

from ..dataset import (
    LoadedClusterDynamicsMLDataset,
    SavedClusterDynamicsMLDataset,
    load_cluster_dynamicsai_dataset,
    save_cluster_dynamicsai_dataset,
)
from ..workflow import (
    ClusterDynamicsMLResult,
    ClusterDynamicsMLWorkflow,
    _resolved_population_weights,
)
from .plot_panel import (
    ClusterDynamicsMLHistogramPanel,
    ClusterDynamicsMLPlotPanel,
)

_OPEN_WINDOWS: list["ClusterDynamicsMLMainWindow"] = []


@dataclass(slots=True)
class ClusterDynamicsMLJobConfig:
    frames_dir: Path
    clusters_dir: Path | None
    project_dir: Path | None
    experimental_data_file: Path | None
    energy_file: Path | None
    atom_type_definitions: dict[str, list[tuple[str, str | None]]]
    pair_cutoff_definitions: dict[tuple[str, str], dict[int, float]]
    box_dimensions: tuple[float, float, float] | None
    use_pbc: bool
    default_cutoff: float | None
    shell_levels: tuple[int, ...]
    shared_shells: bool
    include_shell_atoms_in_stoichiometry: bool
    search_mode: str
    folder_start_time_fs: float | None
    first_frame_time_fs: float
    frame_timestep_fs: float
    frames_per_colormap_timestep: int
    analysis_start_fs: float | None
    analysis_stop_fs: float | None
    target_node_counts: tuple[int, ...]
    candidates_per_size: int
    prediction_population_share_threshold: float
    q_min: float | None
    q_max: float | None
    q_points: int


@dataclass(slots=True)
class _ProjectPredictionHistoryEntry:
    dataset_file: Path
    modified_time: float
    saved_label: str
    target_node_counts: tuple[int, ...]
    candidates_per_size: int
    prediction_population_share_threshold: float
    prediction_count: int
    max_predicted_node_count: int | None
    rmse: float | None


class ClusterDynamicsMLWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ClusterDynamicsMLJobConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            workflow = ClusterDynamicsMLWorkflow(
                self.config.frames_dir,
                atom_type_definitions=self.config.atom_type_definitions,
                pair_cutoff_definitions=self.config.pair_cutoff_definitions,
                clusters_dir=self.config.clusters_dir,
                project_dir=self.config.project_dir,
                experimental_data_file=self.config.experimental_data_file,
                box_dimensions=self.config.box_dimensions,
                use_pbc=self.config.use_pbc,
                default_cutoff=self.config.default_cutoff,
                shell_levels=self.config.shell_levels,
                shared_shells=self.config.shared_shells,
                include_shell_atoms_in_stoichiometry=(
                    self.config.include_shell_atoms_in_stoichiometry
                ),
                search_mode=self.config.search_mode,
                folder_start_time_fs=self.config.folder_start_time_fs,
                first_frame_time_fs=self.config.first_frame_time_fs,
                frame_timestep_fs=self.config.frame_timestep_fs,
                frames_per_colormap_timestep=(
                    self.config.frames_per_colormap_timestep
                ),
                analysis_start_fs=self.config.analysis_start_fs,
                analysis_stop_fs=self.config.analysis_stop_fs,
                energy_file=self.config.energy_file,
                target_node_counts=self.config.target_node_counts,
                candidates_per_size=self.config.candidates_per_size,
                prediction_population_share_threshold=(
                    self.config.prediction_population_share_threshold
                ),
                q_min=self.config.q_min,
                q_max=self.config.q_max,
                q_points=self.config.q_points,
            )
            preview = workflow.preview_selection()
            self.progress.emit(
                "Preparing clusterdynamicsml analysis.\n"
                f"Frames selected: {preview.dynamics_preview.selected_frames}\n"
                f"Observed node counts: {preview.observed_node_counts or ('n/a',)}\n"
                f"Target node counts: {preview.target_node_counts or ('n/a',)}"
            )
            result = workflow.analyze(progress_callback=self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ClusterDynamicsMLSettingsPanel(QGroupBox):
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Prediction Inputs")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        self.setToolTip(
            "Choose the observed smaller-cluster structures to learn from, "
            "optionally load experimental SAXS data for comparison, and set "
            "the larger node counts to predict."
        )

        clusters_tooltip = (
            "Folder of observed smaller-cluster structures. Use the cluster "
            "extraction output organized by stoichiometry label, for example "
            "Pb/, Pb2I/, Pb3I2/ (optionally with motif_* subfolders). "
            "When this tool is opened from the main SAXSShell UI, the active "
            "project's clusters folder can be filled automatically."
        )
        experimental_tooltip = (
            "Optional experimental SAXS data file used to fit and compare the "
            "cluster-only surrogate SAXS trace. Leave this blank to run the "
            "prediction workflow without an experimental comparison."
        )
        target_start_tooltip = (
            "Lowest node count to predict. clusterdynamicsml extrapolates "
            "beyond the observed node counts in the smaller-cluster training "
            "set."
        )
        target_stop_tooltip = (
            "Highest node count to predict. Every integer node count between "
            "the start and stop values is included."
        )
        candidates_tooltip = (
            "Number of ranked candidate stoichiometries to keep for each "
            "predicted node count."
        )
        share_threshold_tooltip = (
            "Minimum share among the predicted candidates used when "
            "highlighting the largest practical predicted node count and "
            "filtering out tiny predicted populations."
        )
        store_history_tooltip = (
            "When enabled and a project folder is set, each clusterdynamicsml "
            "run is cached as its own timestamped result bundle so you can "
            "compare prediction settings and fitted SAXS models later."
        )
        q_min_tooltip = (
            "Fallback minimum q value for the surrogate SAXS comparison when "
            "no experimental data file is loaded."
        )
        q_max_tooltip = (
            "Fallback maximum q value for the surrogate SAXS comparison when "
            "no experimental data file is loaded."
        )
        q_points_tooltip = (
            "Number of q samples in the fallback SAXS grid when no "
            "experimental data file is loaded."
        )

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.setToolTip(clusters_tooltip)
        self.clusters_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Clusters folder",
            self._make_dir_row(
                self.clusters_dir_edit, "Select clusters folder"
            ),
            clusters_tooltip,
            buddy=self.clusters_dir_edit,
        )

        self.experimental_data_edit = QLineEdit()
        self.experimental_data_edit.setToolTip(experimental_tooltip)
        self.experimental_data_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Experimental data",
            self._make_file_row(
                self.experimental_data_edit,
                "Select experimental SAXS data",
            ),
            experimental_tooltip,
            buddy=self.experimental_data_edit,
        )

        self.target_start_spin = QSpinBox()
        self.target_start_spin.setRange(1, 999)
        self.target_start_spin.setValue(4)
        self.target_start_spin.setToolTip(target_start_tooltip)
        self.target_start_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Predict from node count",
            self.target_start_spin,
            target_start_tooltip,
        )

        self.target_stop_spin = QSpinBox()
        self.target_stop_spin.setRange(1, 999)
        self.target_stop_spin.setValue(5)
        self.target_stop_spin.setToolTip(target_stop_tooltip)
        self.target_stop_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Predict through node count",
            self.target_stop_spin,
            target_stop_tooltip,
        )

        self.candidates_spin = QSpinBox()
        self.candidates_spin.setRange(1, 12)
        self.candidates_spin.setValue(3)
        self.candidates_spin.setToolTip(candidates_tooltip)
        self.candidates_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Candidates / size",
            self.candidates_spin,
            candidates_tooltip,
        )

        self.share_threshold_spin = QDoubleSpinBox()
        self.share_threshold_spin.setDecimals(3)
        self.share_threshold_spin.setRange(0.0, 1.0)
        self.share_threshold_spin.setSingleStep(0.01)
        self.share_threshold_spin.setValue(0.02)
        self.share_threshold_spin.setToolTip(share_threshold_tooltip)
        self.share_threshold_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Share threshold",
            self.share_threshold_spin,
            share_threshold_tooltip,
        )

        self.store_history_checkbox = QCheckBox(
            "Keep timestamped prediction history"
        )
        self.store_history_checkbox.setChecked(True)
        self.store_history_checkbox.setToolTip(store_history_tooltip)
        self.store_history_checkbox.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Store history",
            self.store_history_checkbox,
            store_history_tooltip,
        )

        self.q_min_spin = QDoubleSpinBox()
        self.q_min_spin.setDecimals(4)
        self.q_min_spin.setRange(0.0, 100.0)
        self.q_min_spin.setSingleStep(0.01)
        self.q_min_spin.setValue(0.02)
        self.q_min_spin.setToolTip(q_min_tooltip)
        self.q_min_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Fallback q min",
            self.q_min_spin,
            q_min_tooltip,
        )

        self.q_max_spin = QDoubleSpinBox()
        self.q_max_spin.setDecimals(4)
        self.q_max_spin.setRange(0.0, 100.0)
        self.q_max_spin.setSingleStep(0.05)
        self.q_max_spin.setValue(1.20)
        self.q_max_spin.setToolTip(q_max_tooltip)
        self.q_max_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Fallback q max",
            self.q_max_spin,
            q_max_tooltip,
        )

        self.q_points_spin = QSpinBox()
        self.q_points_spin.setRange(10, 20000)
        self.q_points_spin.setValue(250)
        self.q_points_spin.setToolTip(q_points_tooltip)
        self.q_points_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        self._add_tooltipped_row(
            layout,
            "Fallback q points",
            self.q_points_spin,
            q_points_tooltip,
        )

    def _add_tooltipped_row(
        self,
        layout: QFormLayout,
        label_text: str,
        field_widget: QWidget,
        tooltip: str,
        *,
        buddy: QWidget | None = None,
    ) -> None:
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        label.setBuddy(field_widget if buddy is None else buddy)
        field_widget.setToolTip(tooltip)
        layout.addRow(label, field_widget)

    def _make_dir_row(self, line_edit: QLineEdit, title: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")
        button.setToolTip(line_edit.toolTip())
        button.clicked.connect(
            lambda _checked=False: self._choose_directory(line_edit, title)
        )
        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _make_file_row(self, line_edit: QLineEdit, title: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")
        button.setToolTip(line_edit.toolTip())
        button.clicked.connect(
            lambda _checked=False: self._choose_file(line_edit, title)
        )
        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_directory(self, line_edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(
            self, title, line_edit.text().strip()
        )
        if path:
            line_edit.setText(path)

    def _choose_file(self, line_edit: QLineEdit, title: str) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            title,
            line_edit.text().strip(),
            "Data Files (*.dat *.txt *.csv);;All Files (*)",
        )
        if path:
            line_edit.setText(path)

    def clusters_dir(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        return None if not text else Path(text)

    def set_clusters_dir(
        self, path: Path | None, *, emit_signal: bool = True
    ) -> None:
        self.clusters_dir_edit.blockSignals(True)
        self.clusters_dir_edit.setText("" if path is None else str(path))
        self.clusters_dir_edit.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def experimental_data_file(self) -> Path | None:
        text = self.experimental_data_edit.text().strip()
        return None if not text else Path(text)

    def set_experimental_data_file(
        self,
        path: Path | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.experimental_data_edit.blockSignals(True)
        self.experimental_data_edit.setText("" if path is None else str(path))
        self.experimental_data_edit.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def target_node_counts(self) -> tuple[int, ...]:
        start = int(self.target_start_spin.value())
        stop = int(self.target_stop_spin.value())
        lower = min(start, stop)
        upper = max(start, stop)
        return tuple(range(lower, upper + 1))

    def set_target_node_counts(
        self,
        values: tuple[int, ...],
        *,
        emit_signal: bool = True,
    ) -> None:
        if not values:
            return
        self.target_start_spin.blockSignals(True)
        self.target_stop_spin.blockSignals(True)
        self.target_start_spin.setValue(min(values))
        self.target_stop_spin.setValue(max(values))
        self.target_start_spin.blockSignals(False)
        self.target_stop_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def candidates_per_size(self) -> int:
        return int(self.candidates_spin.value())

    def set_candidates_per_size(
        self,
        value: int,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.candidates_spin.blockSignals(True)
        self.candidates_spin.setValue(max(int(value), 1))
        self.candidates_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def prediction_population_share_threshold(self) -> float:
        return float(self.share_threshold_spin.value())

    def set_prediction_population_share_threshold(
        self,
        value: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.share_threshold_spin.blockSignals(True)
        self.share_threshold_spin.setValue(max(float(value), 0.0))
        self.share_threshold_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def store_prediction_history(self) -> bool:
        return bool(self.store_history_checkbox.isChecked())

    def set_store_prediction_history(
        self,
        value: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.store_history_checkbox.blockSignals(True)
        self.store_history_checkbox.setChecked(bool(value))
        self.store_history_checkbox.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def q_min(self) -> float:
        return float(self.q_min_spin.value())

    def q_max(self) -> float:
        return float(self.q_max_spin.value())

    def q_points(self) -> int:
        return int(self.q_points_spin.value())

    def set_q_settings(
        self,
        *,
        q_min: float,
        q_max: float,
        q_points: int,
        emit_signal: bool = True,
    ) -> None:
        self.q_min_spin.blockSignals(True)
        self.q_max_spin.blockSignals(True)
        self.q_points_spin.blockSignals(True)
        self.q_min_spin.setValue(float(q_min))
        self.q_max_spin.setValue(float(q_max))
        self.q_points_spin.setValue(max(int(q_points), 2))
        self.q_min_spin.blockSignals(False)
        self.q_max_spin.blockSignals(False)
        self.q_points_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()


class ClusterDynamicsMLMainWindow(QMainWindow):
    def __init__(
        self,
        initial_frames_dir: Path | None = None,
        initial_energy_file: Path | None = None,
        initial_project_dir: Path | None = None,
        initial_clusters_dir: Path | None = None,
        initial_experimental_data_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._project_manager = SAXSProjectManager()
        self._last_summary: dict[str, object] | None = None
        self._frame_format: str | None = None
        self._last_result: ClusterDynamicsMLResult | None = None
        self._last_dataset_file: Path | None = None
        self._project_history_entries: list[_ProjectPredictionHistoryEntry] = (
            []
        )
        self._run_thread: QThread | None = None
        self._run_worker: ClusterDynamicsMLWorker | None = None
        self._suspend_preview_refresh = False
        self._initializing = True
        self._restoring_project_dataset = False
        self._build_ui()

        if initial_frames_dir is not None:
            self.trajectory_panel.frames_dir_edit.setText(
                str(initial_frames_dir)
            )
        if initial_energy_file is not None:
            self.run_panel.energy_path_edit.setText(str(initial_energy_file))
        if initial_project_dir is not None:
            self.dataset_panel.set_project_dir(initial_project_dir)
        if initial_clusters_dir is not None:
            self.prediction_panel.set_clusters_dir(
                initial_clusters_dir, emit_signal=False
            )
        if initial_experimental_data_file is not None:
            self.prediction_panel.set_experimental_data_file(
                initial_experimental_data_file,
                emit_signal=False,
            )
        self._sync_project_defaults()
        restored = self._restore_latest_project_result(announce=False)
        self._initializing = False
        if not restored:
            self._refresh_project_history_view()
            self._refresh_selection_preview()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (clusterdynamicsml)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1640, 960)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.trajectory_panel = ClusterTrajectoryPanel()
        self.time_panel = ClusterDynamicsTimePanel()
        self.definitions_panel = ClusterDefinitionsPanel()
        self.prediction_panel = ClusterDynamicsMLSettingsPanel()
        self.run_panel = ClusterDynamicsRunPanel()
        self.dataset_panel = ClusterDynamicsDatasetPanel()

        self.run_panel.analyze_button.setText(
            "Analyze and Predict Larger Clusters"
        )

        left_layout.addWidget(self.trajectory_panel)
        left_layout.addWidget(self.time_panel)
        left_layout.addWidget(self.definitions_panel)
        left_layout.addWidget(self.prediction_panel)
        left_layout.addWidget(self.run_panel)
        left_layout.addWidget(self.dataset_panel)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.dynamics_plot_panel = ClusterDynamicsPlotPanel()
        right_layout.addWidget(self.dynamics_plot_panel, stretch=2)

        self.results_tabs = QTabWidget()
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(170)
        summary_layout.addWidget(self.summary_box, stretch=1)
        history_group = QGroupBox("Prediction History")
        history_layout = QVBoxLayout(history_group)
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_layout.setSpacing(8)
        self.history_status_label = QLabel(
            "Select a project folder to compare saved prediction runs. "
            "The most recent run is plotted by default."
        )
        self.history_status_label.setWordWrap(True)
        history_layout.addWidget(self.history_status_label)
        self.history_table = self._build_table(
            (
                "Loaded",
                "Saved",
                "Target Nodes",
                "Candidates / Size",
                "Share Threshold",
                "Predicted",
                "Max Pred Nodes",
                "RMSE",
                "Dataset",
            )
        )
        self.history_table.setMinimumHeight(220)
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.itemSelectionChanged.connect(
            self._update_history_controls
        )
        self.history_table.itemDoubleClicked.connect(
            lambda _item: self._load_selected_history_entry()
        )
        history_header = self.history_table.horizontalHeader()
        history_header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        history_header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        history_header.setSectionResizeMode(
            8,
            QHeaderView.ResizeMode.Stretch,
        )
        history_layout.addWidget(self.history_table, stretch=1)
        history_button_row = QHBoxLayout()
        history_button_row.setContentsMargins(0, 0, 0, 0)
        self.history_load_button = QPushButton("Plot Selected Prediction")
        self.history_load_button.clicked.connect(
            self._load_selected_history_entry
        )
        history_button_row.addWidget(self.history_load_button)
        self.history_refresh_button = QPushButton("Refresh History")
        self.history_refresh_button.clicked.connect(
            self._refresh_project_history_view
        )
        history_button_row.addWidget(self.history_refresh_button)
        history_button_row.addStretch(1)
        history_layout.addLayout(history_button_row)
        summary_layout.addWidget(history_group, stretch=1)
        self.lifetime_table = self._build_table(
            (
                "Type",
                "Nodes",
                "Size Rank",
                "Candidate Rank",
                "Label",
                "Observed-only Weight (%)",
                "Combined Weight (%)",
                "Share (%)",
                "Mean lifetime (fs)",
                "Std lifetime (fs)",
                "Completed",
                "Window-truncated",
                "Assoc. rate (1/ps)",
                "Dissoc. rate (1/ps)",
                "Occupancy (%)",
                "Mean count/frame",
                "Reference",
                "Notes",
            )
        )
        self.histogram_panel = ClusterDynamicsMLHistogramPanel()
        self.saxs_panel = ClusterDynamicsMLPlotPanel()
        self.observed_histogram_panel = self.histogram_panel
        self.combined_histogram_panel = self.histogram_panel
        self.surrogate_plot_panel = self.saxs_panel
        self.results_tabs.addTab(self.summary_tab, "Summary")
        self.results_tabs.addTab(self.lifetime_table, "Lifetimes")
        self.results_tabs.addTab(self.histogram_panel, "Histograms")
        self.results_tabs.addTab(self.saxs_panel, "SAXS")
        right_layout.addWidget(self.results_tabs, stretch=2)

        splitter.addWidget(self._wrap_scroll_area(left))
        splitter.addWidget(right)
        splitter.setSizes([530, 1110])
        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.trajectory_panel.inspect_requested.connect(
            self.inspect_frames_folder
        )
        self.trajectory_panel.frames_dir_changed.connect(
            self._on_frames_dir_changed
        )
        self.trajectory_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.time_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.definitions_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.prediction_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.run_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.run_panel.analyze_requested.connect(self.run_analysis)
        self.dataset_panel.settings_changed.connect(
            self._on_project_dir_changed
        )
        self.dataset_panel.save_dataset_requested.connect(self.save_dataset)
        self.dataset_panel.load_dataset_requested.connect(self.load_dataset)
        self.dataset_panel.save_colormap_requested.connect(
            self.save_colormap_data
        )
        self.dataset_panel.save_lifetime_requested.connect(
            self.save_lifetime_table
        )
        self.dataset_panel.save_powerpoint_requested.connect(
            self.save_powerpoint_report
        )

        self.run_panel.set_selection_summary(
            "Select an extracted frames folder and a smaller-cluster "
            "structures folder to preview the extrapolation workflow."
        )
        self.run_panel.set_log(
            "Ready. clusterdynamicsml reuses the time-binned cluster analysis "
            "from clusterdynamics, then fits an experimental surrogate to "
            "smaller-cluster lifetimes, populations, stoichiometries, and "
            "representative structures."
        )
        self._set_frame_format(None)
        self._update_history_controls()

    def inspect_frames_folder(self) -> None:
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            self._show_error("No extracted frames folder selected.")
            return
        self._inspect_frames_dir(frames_dir, announce=True)

    def run_analysis(self) -> None:
        try:
            if self._run_thread is not None:
                return
            config = self._build_job_config()
            self.run_panel.set_log(
                "clusterdynamicsml request received.\n"
                f"Frames folder: {config.frames_dir}\n"
                f"Clusters folder: {config.clusters_dir}\n"
                f"Targets: {config.target_node_counts}\n"
                f"Experimental data: {config.experimental_data_file}\n"
                f"Frame timestep: {config.frame_timestep_fs:.3f} fs"
            )
            self.run_panel.progress_label.setText(
                "Progress: running surrogate workflow"
            )
            self.run_panel.progress_bar.setRange(0, 0)
            self.dynamics_plot_panel.set_result(None)
            self.histogram_panel.set_result(None)
            self.saxs_panel.set_result(None)
            self.summary_box.clear()
            self.lifetime_table.setRowCount(0)
            self.statusBar().showMessage(
                "Analyzing and predicting larger clusters..."
            )
            self._start_worker(config)
        except Exception as exc:
            self._handle_error("clusterdynamicsml failed", str(exc))

    def _build_job_config(self) -> ClusterDynamicsMLJobConfig:
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            raise ValueError("No extracted frames folder selected.")
        atom_type_definitions = self.definitions_panel.atom_type_definitions()
        if not atom_type_definitions:
            raise ValueError(
                "Add at least one atom-type definition before running the analysis."
            )
        if not (
            atom_type_definitions.get("node")
            or atom_type_definitions.get("linker")
        ):
            raise ValueError("Define at least one node or linker atom type.")
        pair_cutoff_definitions = (
            self.definitions_panel.pair_cutoff_definitions()
        )
        default_cutoff = self.definitions_panel.default_cutoff()
        if not pair_cutoff_definitions and default_cutoff is None:
            raise ValueError(
                "Add at least one pair-cutoff definition or a default cutoff."
            )
        manual_box_dimensions = self.definitions_panel.box_dimensions()
        resolved_box_dimensions = manual_box_dimensions
        use_pbc = self.definitions_panel.use_pbc()
        if use_pbc and resolved_box_dimensions is None:
            resolved_box_dimensions = self._detected_box_dimensions()
            if resolved_box_dimensions is None:
                raise ValueError(
                    "Periodic boundary conditions are enabled, but no box "
                    "dimensions are available."
                )
        return ClusterDynamicsMLJobConfig(
            frames_dir=frames_dir,
            clusters_dir=self.prediction_panel.clusters_dir(),
            project_dir=self.dataset_panel.project_dir(),
            experimental_data_file=self.prediction_panel.experimental_data_file(),
            energy_file=self.run_panel.energy_file(),
            atom_type_definitions=atom_type_definitions,
            pair_cutoff_definitions=pair_cutoff_definitions,
            box_dimensions=resolved_box_dimensions,
            use_pbc=use_pbc,
            default_cutoff=default_cutoff,
            shell_levels=self.definitions_panel.shell_growth_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            search_mode=self.definitions_panel.search_mode(),
            folder_start_time_fs=self.time_panel.folder_start_time_fs(),
            first_frame_time_fs=self.time_panel.first_frame_time_fs(),
            frame_timestep_fs=self.time_panel.frame_timestep_fs(),
            frames_per_colormap_timestep=(
                self.time_panel.frames_per_colormap_timestep()
            ),
            analysis_start_fs=self.time_panel.analysis_start_fs(),
            analysis_stop_fs=self.time_panel.analysis_stop_fs(),
            target_node_counts=self.prediction_panel.target_node_counts(),
            candidates_per_size=self.prediction_panel.candidates_per_size(),
            prediction_population_share_threshold=(
                self.prediction_panel.prediction_population_share_threshold()
            ),
            q_min=self.prediction_panel.q_min(),
            q_max=self.prediction_panel.q_max(),
            q_points=self.prediction_panel.q_points(),
        )

    def _start_worker(self, config: ClusterDynamicsMLJobConfig) -> None:
        self._run_thread = QThread(self)
        self._run_worker = ClusterDynamicsMLWorker(config)
        self._run_worker.moveToThread(self._run_thread)

        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.progress.connect(self.run_panel.append_log)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.failed.connect(self._on_run_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.finished.connect(self._run_worker.deleteLater)
        self._run_thread.start()

    def _on_run_finished(self, result: ClusterDynamicsMLResult) -> None:
        self._last_result = result
        self.dynamics_plot_panel.set_result(result.dynamics_result)
        self.histogram_panel.set_result(result)
        self.saxs_panel.set_result(result)
        self.run_panel.progress_bar.setRange(
            0, max(result.dynamics_result.analyzed_frames, 1)
        )
        self.run_panel.progress_bar.setValue(
            result.dynamics_result.analyzed_frames
        )
        self.run_panel.progress_label.setText(
            f"Progress: completed {result.dynamics_result.analyzed_frames} frames"
        )
        self.run_panel.append_log(
            "clusterdynamicsml complete.\n"
            f"Observed node counts: {result.preview.observed_node_counts}\n"
            f"Predicted candidates: {len(result.predictions)}\n"
            f"Max predicted node count: {result.max_predicted_node_count}"
        )
        self._populate_summary_box(result)
        self._populate_lifetime_table(result)
        autosaved = self._autosave_project_result(result)
        if autosaved is not None:
            self._refresh_project_history_view(
                select_dataset=autosaved.dataset_file
            )
            self.run_panel.append_log(
                "Cached clusterdynamicsml result bundle in project folder.\n"
                f"Dataset: {autosaved.dataset_file}\n"
                f"Files written: {len(autosaved.written_files)}"
            )
            self.statusBar().showMessage(
                "clusterdynamicsml analysis complete and cached in project"
            )
        else:
            self._refresh_project_history_view(
                select_dataset=self._last_dataset_file
            )
            self.statusBar().showMessage("clusterdynamicsml analysis complete")

    def _on_run_failed(self, message: str) -> None:
        self.run_panel.progress_bar.setRange(0, 1)
        self.run_panel.progress_bar.setValue(0)
        self.run_panel.progress_label.setText("Progress: failed")
        self.statusBar().showMessage("clusterdynamicsml failed")
        self._handle_error("clusterdynamicsml failed", message)

    def _cleanup_run_thread(self) -> None:
        self._run_worker = None
        self._run_thread = None

    def save_dataset(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved result before exporting."
            )
            return
        default_path = self._default_dataset_file()
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save clusterdynamicsml dataset",
            str(default_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        saved = save_cluster_dynamicsai_dataset(
            self._last_result,
            path,
            analysis_settings=self._analysis_settings_payload(),
        )
        self._last_dataset_file = saved.dataset_file
        self.run_panel.append_log(
            "Saved clusterdynamicsml dataset to "
            f"{saved.dataset_file}\n"
            f"Wrote {len(saved.written_files)} file(s)."
        )
        self.statusBar().showMessage(
            f"Saved clusterdynamicsml dataset to {saved.dataset_file}"
        )

    def load_dataset(self) -> None:
        default_path = (
            self._last_dataset_file
            if self._last_dataset_file is not None
            else self._default_dataset_file()
        )
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load clusterdynamicsml dataset",
            str(default_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        loaded = load_cluster_dynamicsai_dataset(path)
        self._apply_loaded_dataset(
            loaded,
            announce=True,
            action_label="Loaded",
        )

    def save_colormap_data(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved result before exporting."
            )
            return
        default_path = self._default_export_file("cluster_distribution")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save colormap data",
            str(default_path),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        display_mode = (
            self.dynamics_plot_panel.display_mode_combo.currentData()
        )
        time_unit = self.dynamics_plot_panel.time_unit_combo.currentData()
        saved_path = export_cluster_dynamics_colormap_csv(
            self._last_result.dynamics_result,
            path,
            display_mode=(
                "fraction" if display_mode is None else str(display_mode)
            ),
            time_unit="fs" if time_unit is None else str(time_unit),
        )
        dynamics_result = self._last_result.dynamics_result
        row_count = (
            len(dynamics_result.cluster_labels) * dynamics_result.bin_count
        )
        self.run_panel.append_log(
            "Saved clusterdynamicsml colormap data to "
            f"{saved_path}\n"
            f"Rows written: {row_count}"
        )
        self.statusBar().showMessage(f"Saved colormap data to {saved_path}")

    def save_lifetime_table(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved result before exporting."
            )
            return
        default_path = self._default_export_file("lifetime")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save lifetime table",
            str(default_path),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        if (
            self.lifetime_table.rowCount() == 0
            and self._last_result is not None
        ):
            self._populate_lifetime_table(self._last_result)
        saved_path = _write_table_widget_csv(self.lifetime_table, Path(path))
        row_count = self.lifetime_table.rowCount()
        self.run_panel.append_log(
            "Saved clusterdynamicsml lifetime table to "
            f"{saved_path}\n"
            f"Rows written: {row_count}"
        )
        self.statusBar().showMessage(f"Saved lifetime table to {saved_path}")

    def save_powerpoint_report(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved result before exporting."
            )
            return

        self.dynamics_plot_panel.set_result(self._last_result.dynamics_result)
        self.saxs_panel.set_result(self._last_result)
        selection_summary = self.run_panel.selection_box.toPlainText().strip()
        if not selection_summary:
            selection_summary = self._format_preview_text(
                self._last_result.preview
            )
        summary_text = self.summary_box.toPlainText().strip()
        if not summary_text:
            self._populate_summary_box(self._last_result)
            summary_text = self.summary_box.toPlainText().strip()

        default_path = self._default_powerpoint_report_file()
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save clusterdynamicsml PowerPoint report",
            str(default_path),
            "PowerPoint Files (*.pptx);;All Files (*)",
        )
        if not path:
            return

        self.run_panel.progress_label.setText(
            "Progress: generating PowerPoint report"
        )
        self.run_panel.progress_bar.setRange(0, 1)
        self.run_panel.progress_bar.setValue(0)
        self.run_panel.progress_bar.setFormat("%v / %m steps")
        try:
            export_result = export_cluster_dynamicsai_report_pptx(
                result=self._last_result,
                selection_summary=selection_summary,
                result_summary=summary_text,
                dynamics_figure=self.dynamics_plot_panel.figure,
                surrogate_figure=self.saxs_panel.figure,
                output_path=path,
                settings=self._powerpoint_export_settings(),
                project_dir=self.dataset_panel.project_dir(),
                frames_dir=self.trajectory_panel.get_frames_dir(),
                progress_callback=self._on_powerpoint_report_progress,
            )
        except Exception as exc:
            self.run_panel.progress_bar.setRange(0, 1)
            self.run_panel.progress_bar.setValue(0)
            self.run_panel.progress_bar.setFormat("%v / %m steps")
            self.run_panel.progress_label.setText(
                "Progress: PowerPoint export failed"
            )
            self._handle_error(
                "clusterdynamicsml PowerPoint export failed", str(exc)
            )
            return

        self.run_panel.progress_label.setText(
            "Progress: PowerPoint report saved"
        )
        self.run_panel.progress_bar.setValue(
            self.run_panel.progress_bar.maximum()
        )
        self.run_panel.progress_bar.setFormat("%v / %m steps")
        if export_result.appended_to_existing:
            self.run_panel.append_log(
                "Appended clusterdynamicsml report slides to "
                f"{export_result.report_path}\n"
                f"Slides added: {export_result.added_slide_count}"
            )
        else:
            self.run_panel.append_log(
                "Saved clusterdynamicsml PowerPoint report to "
                f"{export_result.report_path}\n"
                f"Slides written: {export_result.added_slide_count}"
            )
        self.statusBar().showMessage(
            f"Saved PowerPoint report to {export_result.report_path}"
        )

    def _inspect_frames_dir(self, frames_dir: Path, *, announce: bool) -> None:
        self._last_summary = None
        try:
            analyzer = ExtractedFrameFolderClusterAnalyzer(
                frames_dir=frames_dir,
                atom_type_definitions={},
                pair_cutoffs_def={},
            )
            self._last_summary = analyzer.inspect()
            self._sync_box_dimensions_from_summary(self._last_summary)
            self._set_frame_format(self._last_summary.get("frame_format"))
            self.trajectory_panel.set_summary(self._last_summary)
            if announce:
                self.run_panel.append_log(
                    "Inspection complete. "
                    f"Detected {self._last_summary['n_frames']} extracted frame(s)."
                )
                self.statusBar().showMessage("Inspection complete")
        except ValueError as exc:
            self._sync_box_dimensions_from_summary(None)
            frame_format, detail = self._detect_frame_format(frames_dir)
            self._set_frame_format(frame_format)
            self.trajectory_panel.set_summary_text(str(exc))
            if detail is not None:
                self.trajectory_panel.set_frame_mode(None, detail=detail)
            if announce:
                self._handle_error("Frames-folder inspection failed", str(exc))
        self._refresh_selection_preview()

    def _on_frames_dir_changed(self, frames_dir: Path | None) -> None:
        if self._suspend_preview_refresh:
            return
        self._last_summary = None
        self._last_result = None
        self.dynamics_plot_panel.set_result(None)
        self.histogram_panel.set_result(None)
        self.saxs_panel.set_result(None)
        self.summary_box.clear()
        self.lifetime_table.setRowCount(0)
        self.time_panel.set_folder_start_time_fs(None, emit_signal=False)
        if frames_dir is None:
            self._sync_box_dimensions_from_summary(None)
            self._set_frame_format(None)
            self.trajectory_panel.set_summary_text("")
            self._refresh_selection_preview()
            return
        self._inspect_frames_dir(frames_dir, announce=False)

    def _on_project_dir_changed(self) -> None:
        if self._restoring_project_dataset:
            return
        self._sync_project_defaults()
        if not self._initializing:
            restored = self._restore_latest_project_result(announce=True)
            if not restored:
                self._refresh_project_history_view()
                self._refresh_selection_preview()

    def _sync_project_defaults(self) -> None:
        if self._suspend_preview_refresh:
            return
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return
        project_file = build_project_paths(project_dir).project_file
        if not project_file.is_file():
            return
        try:
            settings = self._project_manager.load_project(project_dir)
        except Exception:
            return
        if self.prediction_panel.clusters_dir() is None:
            self.prediction_panel.set_clusters_dir(
                settings.resolved_clusters_dir,
                emit_signal=False,
            )
        if self.prediction_panel.experimental_data_file() is None:
            self.prediction_panel.set_experimental_data_file(
                settings.resolved_experimental_data_path,
                emit_signal=False,
            )

    def _refresh_selection_preview(self) -> None:
        if self._suspend_preview_refresh:
            return
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            self.run_panel.set_selection_summary(
                "Select an extracted frames folder to preview the "
                "clusterdynamicsml workflow."
            )
            return
        try:
            preview = self._build_preview_workflow().preview_selection()
            if (
                self.time_panel.folder_start_time_fs() is None
                and preview.dynamics_preview.folder_start_time_fs is not None
                and preview.dynamics_preview.folder_start_time_source
                != "manual field"
            ):
                self.time_panel.set_folder_start_time_fs(
                    preview.dynamics_preview.folder_start_time_fs,
                    emit_signal=False,
                )
            text = self._format_preview_text(preview)
        except Exception as exc:
            text = (
                "Adjust the current settings to preview the extrapolation "
                f"workflow.\nValidation warning: {exc}"
            )
        self.run_panel.set_selection_summary(text)

    def _build_preview_workflow(self) -> ClusterDynamicsMLWorkflow:
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            raise ValueError("No extracted frames folder selected.")
        manual_box_dimensions = self.definitions_panel.box_dimensions()
        resolved_box_dimensions = manual_box_dimensions
        if (
            self.definitions_panel.use_pbc()
            and resolved_box_dimensions is None
        ):
            resolved_box_dimensions = self._detected_box_dimensions()
        return ClusterDynamicsMLWorkflow(
            frames_dir,
            atom_type_definitions=self.definitions_panel.atom_type_definitions(),
            pair_cutoff_definitions=self.definitions_panel.pair_cutoff_definitions(),
            clusters_dir=self.prediction_panel.clusters_dir(),
            project_dir=self.dataset_panel.project_dir(),
            experimental_data_file=self.prediction_panel.experimental_data_file(),
            box_dimensions=resolved_box_dimensions,
            use_pbc=self.definitions_panel.use_pbc(),
            default_cutoff=self.definitions_panel.default_cutoff(),
            shell_levels=self.definitions_panel.shell_growth_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            search_mode=self.definitions_panel.search_mode(),
            folder_start_time_fs=self.time_panel.folder_start_time_fs(),
            first_frame_time_fs=self.time_panel.first_frame_time_fs(),
            frame_timestep_fs=self.time_panel.frame_timestep_fs(),
            frames_per_colormap_timestep=(
                self.time_panel.frames_per_colormap_timestep()
            ),
            analysis_start_fs=self.time_panel.analysis_start_fs(),
            analysis_stop_fs=self.time_panel.analysis_stop_fs(),
            energy_file=self.run_panel.energy_file(),
            target_node_counts=self.prediction_panel.target_node_counts(),
            candidates_per_size=self.prediction_panel.candidates_per_size(),
            prediction_population_share_threshold=(
                self.prediction_panel.prediction_population_share_threshold()
            ),
            q_min=self.prediction_panel.q_min(),
            q_max=self.prediction_panel.q_max(),
            q_points=self.prediction_panel.q_points(),
        )

    def _format_preview_text(self, preview) -> str:
        dynamics_preview = preview.dynamics_preview
        lines = [
            f"Mode: {frame_folder_label(dynamics_preview.frame_format)}",
            f"PBC: {'on' if dynamics_preview.use_pbc else 'off'}",
            "Search mode: "
            f"{format_search_mode_label(self.definitions_panel.search_mode())}",
            f"Frames in folder: {dynamics_preview.total_frames}",
            f"Frames selected: {dynamics_preview.selected_frames}",
            f"Frame timestep: {dynamics_preview.frame_timestep_fs:.3f} fs",
            "Frames per colormap timestep: "
            f"{dynamics_preview.frames_per_colormap_timestep}",
            f"Colormap timestep: {dynamics_preview.colormap_timestep_fs:.3f} fs",
            f"Time bins: {dynamics_preview.bin_count}",
            f"Smaller-cluster labels with structures: {preview.structure_label_count}",
            f"Structure files discovered: {preview.total_structure_files}",
            f"Observed node counts: {preview.observed_node_counts or ('n/a',)}",
            f"Target node counts: {preview.target_node_counts or ('n/a',)}",
            "Stoichiometry bins: "
            + (
                "solute + shell atoms"
                if self.definitions_panel.include_shell_atoms_in_stoichiometry()
                else "solute only"
            ),
            "Resolved box dimensions: "
            f"{format_box_dimensions(dynamics_preview.resolved_box_dimensions)}",
            (
                "Clusters folder: "
                + (
                    "not set"
                    if preview.clusters_dir is None
                    else str(preview.clusters_dir)
                )
            ),
            (
                "Experimental data: "
                + (
                    "not set"
                    if preview.experimental_data_path is None
                    else str(preview.experimental_data_path)
                )
            ),
        ]
        if dynamics_preview.first_selected_frame is not None:
            lines.append(
                "Frame file range: "
                f"{dynamics_preview.first_selected_frame} to "
                f"{dynamics_preview.last_selected_frame}"
            )
        if dynamics_preview.time_warnings:
            lines.extend(
                f"Warning: {message}"
                for message in dynamics_preview.time_warnings
            )
        if preview.warnings:
            lines.extend(f"Warning: {message}" for message in preview.warnings)
        return "\n".join(lines)

    def _populate_summary_box(self, result: ClusterDynamicsMLResult) -> None:
        lines = [
            f"Frames analyzed: {result.dynamics_result.analyzed_frames}",
            f"Time bins: {result.dynamics_result.bin_count}",
            f"Observed node counts: {result.preview.observed_node_counts}",
            f"Target node counts: {result.preview.target_node_counts}",
            f"Predicted candidates: {len(result.predictions)}",
            f"Max observed node count: {result.max_observed_node_count}",
            (
                "Max predicted node count above threshold: "
                f"{result.max_predicted_node_count}"
            ),
            (
                "Prediction-candidate share threshold: "
                f"{result.prediction_population_share_threshold:.3f}"
            ),
            (
                "Store prediction history in project: "
                f"{'on' if self.prediction_panel.store_prediction_history() else 'off'}"
            ),
            (
                "Clusters folder: "
                + (
                    "n/a"
                    if result.preview.clusters_dir is None
                    else str(result.preview.clusters_dir)
                )
            ),
            (
                "Experimental data: "
                + (
                    "n/a"
                    if result.preview.experimental_data_path is None
                    else str(result.preview.experimental_data_path)
                )
            ),
        ]
        if result.saxs_comparison is not None:
            lines.append(
                f"SAXS components in mixture: "
                f"{len(result.saxs_comparison.component_weights)}"
            )
            if result.saxs_comparison.component_output_dir is not None:
                lines.append(
                    "SAXS component files: "
                    f"{result.saxs_comparison.component_output_dir}"
                )
            if result.saxs_comparison.surrogate_structure_dir is not None:
                lines.append(
                    "Surrogate XYZ files: "
                    f"{result.saxs_comparison.surrogate_structure_dir}"
                )
            if result.saxs_comparison.rmse is not None:
                lines.append(
                    f"Cluster-only surrogate SAXS RMSE: "
                    f"{result.saxs_comparison.rmse:.6g}"
                )
        self.summary_box.setPlainText("\n".join(lines))

    def _populate_lifetime_table(
        self, result: ClusterDynamicsMLResult
    ) -> None:
        self.lifetime_table.setSortingEnabled(False)
        lifetime_rows = _combined_model_weight_rows(result)
        self.lifetime_table.setRowCount(len(lifetime_rows))
        for row, entry in enumerate(lifetime_rows):
            values = (
                str(entry["type"]),
                str(entry["nodes"]),
                str(entry["size_rank"]),
                str(entry["candidate_rank"]),
                str(entry["label"]),
                _format_optional_percent(
                    entry["observed_only_normalized_weight"]
                ),
                _format_optional_percent(entry["normalized_weight"]),
                _format_optional_percent(entry["predicted_population_share"]),
                _format_optional_float(entry["mean_lifetime_fs"]),
                _format_optional_float(entry["std_lifetime_fs"]),
                _format_optional_int(entry["completed_lifetime_count"]),
                _format_optional_int(entry["window_truncated_lifetime_count"]),
                f"{float(entry['association_rate_per_ps']):.3f}",
                f"{float(entry['dissociation_rate_per_ps']):.3f}",
                f"{float(entry['occupancy_fraction']) * 100.0:.1f}",
                f"{float(entry['mean_count_per_frame']):.3f}",
                str(entry["reference"]),
                str(entry["notes"]),
            )
            for column, value in enumerate(values):
                self.lifetime_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )
        self.lifetime_table.resizeColumnsToContents()
        self.lifetime_table.setSortingEnabled(True)

    def _refresh_project_history_view(
        self,
        *,
        select_dataset: Path | None = None,
    ) -> None:
        project_dir = self.dataset_panel.project_dir()
        self._project_history_entries = (
            self._project_history_entries_for_project(project_dir)
        )
        selected_dataset = (
            None
            if select_dataset is None
            else Path(select_dataset).expanduser().resolve()
        )
        if selected_dataset is None and self._last_dataset_file is not None:
            selected_dataset = self._last_dataset_file.resolve()

        self.history_table.setSortingEnabled(False)
        self.history_table.setRowCount(len(self._project_history_entries))
        selected_row: int | None = None
        for row, entry in enumerate(self._project_history_entries):
            is_loaded = (
                self._last_dataset_file is not None
                and self._last_dataset_file.resolve() == entry.dataset_file
            )
            values = (
                "Yes" if is_loaded else "",
                entry.saved_label,
                _format_int_sequence(entry.target_node_counts),
                str(entry.candidates_per_size),
                f"{entry.prediction_population_share_threshold:.3f}",
                str(entry.prediction_count),
                _format_optional_int(entry.max_predicted_node_count),
                _format_optional_float(entry.rmse),
                entry.dataset_file.name,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(entry.dataset_file))
                item.setToolTip(str(entry.dataset_file))
                self.history_table.setItem(row, column, item)
            if (
                selected_dataset is not None
                and entry.dataset_file == selected_dataset
            ):
                selected_row = row

        self.history_table.resizeColumnsToContents()
        self.history_table.setSortingEnabled(False)
        self.history_table.clearSelection()
        if self._project_history_entries:
            target_row = 0 if selected_row is None else selected_row
            self.history_table.selectRow(target_row)
        self._update_history_controls()

    def _project_history_entries_for_project(
        self,
        project_dir: Path | None,
    ) -> list[_ProjectPredictionHistoryEntry]:
        saved_results_dir = self._project_saved_results_dir(project_dir)
        if saved_results_dir is None or not saved_results_dir.is_dir():
            return []
        entries: list[_ProjectPredictionHistoryEntry] = []
        for dataset_file in saved_results_dir.rglob(
            "*_clusterdynamicsml.json"
        ):
            if not dataset_file.is_file():
                continue
            entry = _read_project_history_entry(dataset_file)
            if entry is not None:
                entries.append(entry)
        entries.sort(
            key=lambda entry: (
                -float(entry.modified_time),
                entry.dataset_file.name.lower(),
            )
        )
        return entries

    def _selected_history_dataset_file(self) -> Path | None:
        selected_ranges = self.history_table.selectedRanges()
        if not selected_ranges:
            return None
        return self._history_dataset_file_for_row(selected_ranges[0].topRow())

    def _history_dataset_file_for_row(self, row: int) -> Path | None:
        if row < 0 or row >= self.history_table.rowCount():
            return None
        item = self.history_table.item(row, 0)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return None if value in {None, ""} else Path(str(value))

    def _update_history_controls(self) -> None:
        selected_dataset = self._selected_history_dataset_file()
        has_history = bool(self._project_history_entries)
        self.history_table.setEnabled(has_history)
        self.history_load_button.setEnabled(selected_dataset is not None)
        if not has_history:
            project_dir = self.dataset_panel.project_dir()
            if project_dir is None:
                self.history_status_label.setText(
                    "Select a project folder to compare saved prediction runs. "
                    "The most recent run is plotted by default."
                )
            else:
                self.history_status_label.setText(
                    "No cached prediction history was found for the current project."
                )
            return
        if selected_dataset is None:
            self.history_status_label.setText(
                "Select a saved prediction run to compare its parameters and plots."
            )
            return
        if (
            self._last_dataset_file is not None
            and self._last_dataset_file.resolve() == selected_dataset.resolve()
        ):
            self.history_status_label.setText(
                "The selected history entry is currently plotted."
            )
            return
        self.history_status_label.setText(
            "Select a row and click Plot Selected Prediction to compare it."
        )

    def _load_selected_history_entry(self) -> None:
        dataset_file = self._selected_history_dataset_file()
        if dataset_file is None:
            return
        try:
            loaded = load_cluster_dynamicsai_dataset(dataset_file)
        except Exception as exc:
            self._handle_error(
                "clusterdynamicsml history load failed",
                f"Could not load {dataset_file}: {exc}",
            )
            return
        self._apply_loaded_dataset(
            loaded,
            announce=True,
            action_label="Loaded history entry",
        )

    @staticmethod
    def _build_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if headers:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        return table

    def _default_dataset_dir(self) -> Path:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is not None:
            paths = build_project_paths(project_dir)
            target_dir = paths.exported_data_dir / "clusterdynamicsml"
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is not None:
            return frames_dir.parent
        return Path.cwd()

    def _default_dataset_file(self) -> Path:
        if self._last_dataset_file is not None:
            return self._last_dataset_file
        frames_dir = self.trajectory_panel.get_frames_dir()
        folder_label = "cluster_dynamics_ml"
        if frames_dir is not None:
            folder_label = frames_dir.name or folder_label
        return (
            self._default_dataset_dir()
            / f"{folder_label}_clusterdynamicsml.json"
        )

    def _default_export_file(self, suffix_label: str) -> Path:
        dataset_file = self._default_dataset_file()
        return dataset_file.with_name(
            f"{dataset_file.stem}_{suffix_label}.csv"
        )

    def _default_powerpoint_report_file(self) -> Path:
        frames_dir = self.trajectory_panel.get_frames_dir()
        fallback_label = "cluster_dynamics_ml_report"
        if frames_dir is not None:
            fallback_label = (
                f"{frames_dir.name or 'cluster_dynamics_ml'}_report"
            )
        return default_powerpoint_report_path(
            project_dir=self.dataset_panel.project_dir(),
            fallback_dir=self._default_dataset_dir(),
            fallback_stem=fallback_label,
        )

    def _project_saved_results_dir(
        self,
        project_dir: Path | None = None,
    ) -> Path | None:
        resolved_project_dir = (
            self.dataset_panel.project_dir()
            if project_dir is None
            else Path(project_dir).expanduser().resolve()
        )
        if resolved_project_dir is None:
            return None
        saved_results_dir = (
            build_project_paths(resolved_project_dir).exported_data_dir
            / "clusterdynamicsml"
            / "saved_results"
        )
        saved_results_dir.mkdir(parents=True, exist_ok=True)
        return saved_results_dir

    def _latest_project_dataset_file(
        self,
        project_dir: Path | None = None,
    ) -> Path | None:
        saved_results_dir = self._project_saved_results_dir(project_dir)
        if saved_results_dir is None or not saved_results_dir.is_dir():
            return None
        candidates: list[Path] = []
        for candidate in saved_results_dir.rglob("*_clusterdynamicsml.json"):
            if candidate.is_file():
                candidates.append(candidate)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda path: (path.stat().st_mtime, path.name.lower()),
        )

    def _autosave_project_result(
        self,
        result: ClusterDynamicsMLResult,
    ) -> SavedClusterDynamicsMLDataset | None:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return None
        saved_results_dir = self._project_saved_results_dir(project_dir)
        if saved_results_dir is None:
            return None
        frames_dir = self.trajectory_panel.get_frames_dir()
        stem_label = _safe_filename_stem(
            project_dir.name
            if frames_dir is None
            else (frames_dir.name or project_dir.name)
        )
        if self.prediction_panel.store_prediction_history():
            bundle_dir = saved_results_dir / (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{stem_label}"
            )
        else:
            bundle_dir = saved_results_dir / f"latest_{stem_label}"
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = bundle_dir / f"{stem_label}_clusterdynamicsml.json"
        saved = save_cluster_dynamicsai_dataset(
            result,
            dataset_path,
            analysis_settings=self._analysis_settings_payload(),
        )

        selection_summary = self.run_panel.selection_box.toPlainText().strip()
        if not selection_summary:
            selection_summary = self._format_preview_text(result.preview)
        summary_text = self.summary_box.toPlainText().strip()
        if not summary_text:
            self._populate_summary_box(result)
            summary_text = self.summary_box.toPlainText().strip()
        extra_files = [
            (
                saved.dataset_file.with_name(
                    f"{saved.dataset_file.stem}_selection_preview.txt"
                ),
                selection_summary,
            ),
            (
                saved.dataset_file.with_name(
                    f"{saved.dataset_file.stem}_summary.txt"
                ),
                summary_text,
            ),
        ]
        written_files = list(saved.written_files)
        for output_path, content in extra_files:
            output_path.write_text(content.strip() + "\n", encoding="utf-8")
            written_files.append(output_path)

        autosaved = SavedClusterDynamicsMLDataset(
            dataset_file=saved.dataset_file,
            written_files=tuple(written_files),
        )
        self._last_dataset_file = autosaved.dataset_file
        return autosaved

    def _restore_latest_project_result(self, *, announce: bool) -> bool:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return False
        dataset_file = self._latest_project_dataset_file(project_dir)
        if dataset_file is None:
            return False
        try:
            loaded = load_cluster_dynamicsai_dataset(dataset_file)
        except Exception as exc:
            if announce:
                self.run_panel.append_log(
                    "Could not restore cached clusterdynamicsml result from "
                    f"{dataset_file}: {exc}"
                )
            return False
        self._apply_loaded_dataset(
            loaded,
            announce=announce,
            action_label="Restored",
        )
        return True

    def _apply_loaded_dataset(
        self,
        loaded: LoadedClusterDynamicsMLDataset,
        *,
        announce: bool,
        action_label: str,
    ) -> None:
        self._restoring_project_dataset = True
        try:
            self._last_dataset_file = loaded.dataset_file
            self._apply_analysis_settings(loaded.analysis_settings)
            self._last_result = loaded.result
            self.dynamics_plot_panel.set_result(loaded.result.dynamics_result)
            self.histogram_panel.set_result(loaded.result)
            self.saxs_panel.set_result(loaded.result)
            self.run_panel.set_selection_summary(
                self._format_preview_text(loaded.result.preview)
            )
            self._populate_summary_box(loaded.result)
            self._populate_lifetime_table(loaded.result)
            analyzed_frames = max(
                loaded.result.dynamics_result.analyzed_frames, 1
            )
            self.run_panel.progress_bar.setRange(0, analyzed_frames)
            self.run_panel.progress_bar.setValue(analyzed_frames)
            self.run_panel.progress_label.setText(
                f"Progress: loaded saved result ({analyzed_frames} frames)"
            )
        finally:
            self._restoring_project_dataset = False
        self._refresh_project_history_view(select_dataset=loaded.dataset_file)
        if announce:
            self.run_panel.append_log(
                f"{action_label} clusterdynamicsml dataset from "
                f"{loaded.dataset_file}\n"
                f"Predicted candidates: {len(loaded.result.predictions)}"
            )
        self.statusBar().showMessage(
            f"{action_label} clusterdynamicsml dataset from {loaded.dataset_file}"
        )

    def _powerpoint_export_settings(self) -> PowerPointExportSettings:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return PowerPointExportSettings()
        project_file = build_project_paths(project_dir).project_file
        if not project_file.is_file():
            return PowerPointExportSettings()
        try:
            settings = self._project_manager.load_project(project_dir)
        except Exception:
            return PowerPointExportSettings()
        return PowerPointExportSettings.from_dict(
            settings.powerpoint_export_settings.to_dict()
        )

    def _on_powerpoint_report_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        total_steps = max(int(total), 1)
        processed_steps = max(0, min(int(processed), total_steps))
        self.run_panel.progress_label.setText(
            f"Progress: PowerPoint report {processed_steps}/{total_steps}"
        )
        self.run_panel.progress_bar.setRange(0, total_steps)
        self.run_panel.progress_bar.setValue(processed_steps)
        self.run_panel.progress_bar.setFormat("%v / %m steps")
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _analysis_settings_payload(self) -> dict[str, object]:
        frames_dir = self.trajectory_panel.get_frames_dir()
        energy_file = self.run_panel.energy_file()
        project_dir = self.dataset_panel.project_dir()
        return {
            "frames_dir": None if frames_dir is None else str(frames_dir),
            "clusters_dir": (
                None
                if self.prediction_panel.clusters_dir() is None
                else str(self.prediction_panel.clusters_dir())
            ),
            "experimental_data_file": (
                None
                if self.prediction_panel.experimental_data_file() is None
                else str(self.prediction_panel.experimental_data_file())
            ),
            "energy_file": None if energy_file is None else str(energy_file),
            "project_dir": None if project_dir is None else str(project_dir),
            "atom_type_definitions": {
                atom_type: [
                    [element, residue] for element, residue in criteria
                ]
                for atom_type, criteria in self.definitions_panel.atom_type_definitions().items()
            },
            "pair_cutoff_definitions": [
                {
                    "atom1": atom1,
                    "atom2": atom2,
                    "shell_cutoffs": {
                        str(level): float(cutoff)
                        for level, cutoff in shell_cutoffs.items()
                    },
                }
                for (atom1, atom2), shell_cutoffs in sorted(
                    self.definitions_panel.pair_cutoff_definitions().items()
                )
            ],
            "box_dimensions": (
                None
                if self.definitions_panel.box_dimensions() is None
                else list(self.definitions_panel.box_dimensions())
            ),
            "use_pbc": self.definitions_panel.use_pbc(),
            "default_cutoff": self.definitions_panel.default_cutoff(),
            "shell_levels": list(self.definitions_panel.shell_growth_levels()),
            "shared_shells": self.definitions_panel.shared_shells(),
            "include_shell_atoms_in_stoichiometry": (
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            "search_mode": self.definitions_panel.search_mode(),
            "folder_start_time_fs": self.time_panel.folder_start_time_fs(),
            "first_frame_time_fs": self.time_panel.first_frame_time_fs(),
            "frame_timestep_fs": self.time_panel.frame_timestep_fs(),
            "frames_per_colormap_timestep": (
                self.time_panel.frames_per_colormap_timestep()
            ),
            "analysis_start_fs": self.time_panel.analysis_start_fs(),
            "analysis_stop_fs": self.time_panel.analysis_stop_fs(),
            "target_node_counts": list(
                self.prediction_panel.target_node_counts()
            ),
            "candidates_per_size": self.prediction_panel.candidates_per_size(),
            "prediction_population_share_threshold": (
                self.prediction_panel.prediction_population_share_threshold()
            ),
            "store_prediction_history": (
                self.prediction_panel.store_prediction_history()
            ),
            "q_min": self.prediction_panel.q_min(),
            "q_max": self.prediction_panel.q_max(),
            "q_points": self.prediction_panel.q_points(),
        }

    def _apply_analysis_settings(self, payload: dict[str, object]) -> None:
        self._suspend_preview_refresh = True
        try:
            self.trajectory_panel.frames_dir_edit.setText(
                ""
                if _optional_path(payload.get("frames_dir")) is None
                else str(_optional_path(payload.get("frames_dir")))
            )
            self.prediction_panel.set_clusters_dir(
                _optional_path(payload.get("clusters_dir")),
                emit_signal=False,
            )
            self.prediction_panel.set_experimental_data_file(
                _optional_path(payload.get("experimental_data_file")),
                emit_signal=False,
            )
            self.run_panel.energy_path_edit.setText(
                ""
                if _optional_path(payload.get("energy_file")) is None
                else str(_optional_path(payload.get("energy_file")))
            )
            self.dataset_panel.set_project_dir(
                _optional_path(payload.get("project_dir"))
            )
            atom_type_definitions = {
                str(atom_type): [
                    (
                        str(entry[0]),
                        (
                            None
                            if len(entry) < 2 or entry[1] in {None, ""}
                            else str(entry[1])
                        ),
                    )
                    for entry in criteria
                    if isinstance(entry, (list, tuple)) and entry
                ]
                for atom_type, criteria in dict(
                    payload.get("atom_type_definitions", {})
                ).items()
            }
            pair_cutoff_definitions = {
                (str(entry.get("atom1", "")), str(entry.get("atom2", ""))): {
                    int(level): float(cutoff)
                    for level, cutoff in dict(
                        entry.get("shell_cutoffs", {})
                    ).items()
                }
                for entry in payload.get("pair_cutoff_definitions", [])
                if isinstance(entry, dict)
            }
            self.definitions_panel.load_atom_type_definitions(
                atom_type_definitions,
                emit_signal=False,
            )
            self.definitions_panel.load_pair_cutoff_definitions(
                pair_cutoff_definitions,
                emit_signal=False,
            )
            self.definitions_panel.set_box_dimensions(
                _optional_box_dimensions(payload.get("box_dimensions")),
                emit_signal=False,
            )
            self.definitions_panel.set_use_pbc(
                bool(payload.get("use_pbc", False)),
                emit_signal=False,
            )
            self.definitions_panel.set_default_cutoff(
                _optional_float(payload.get("default_cutoff")),
                emit_signal=False,
            )
            self.definitions_panel.set_shell_growth_levels(
                tuple(int(value) for value in payload.get("shell_levels", [])),
                emit_signal=False,
            )
            self.definitions_panel.set_shared_shells(
                bool(payload.get("shared_shells", False)),
                emit_signal=False,
            )
            self.definitions_panel.set_include_shell_atoms_in_stoichiometry(
                bool(
                    payload.get("include_shell_atoms_in_stoichiometry", False)
                ),
                emit_signal=False,
            )
            self.definitions_panel.set_search_mode(
                str(payload.get("search_mode", "kdtree")),
                emit_signal=False,
            )
            self.time_panel.set_folder_start_time_fs(
                _optional_float(payload.get("folder_start_time_fs")),
                emit_signal=False,
            )
            self.time_panel.set_first_frame_time_fs(
                float(payload.get("first_frame_time_fs", 0.0)),
                emit_signal=False,
            )
            self.time_panel.set_frame_timestep_fs(
                float(payload.get("frame_timestep_fs", 0.5)),
                emit_signal=False,
            )
            self.time_panel.set_frames_per_colormap_timestep(
                int(payload.get("frames_per_colormap_timestep", 1)),
                emit_signal=False,
            )
            self.time_panel.set_analysis_start_fs(
                _optional_float(payload.get("analysis_start_fs")),
                emit_signal=False,
            )
            self.time_panel.set_analysis_stop_fs(
                _optional_float(payload.get("analysis_stop_fs")),
                emit_signal=False,
            )
            self.prediction_panel.set_target_node_counts(
                tuple(
                    int(value)
                    for value in payload.get("target_node_counts", [])
                ),
                emit_signal=False,
            )
            self.prediction_panel.set_candidates_per_size(
                int(payload.get("candidates_per_size", 3)),
                emit_signal=False,
            )
            self.prediction_panel.set_prediction_population_share_threshold(
                float(
                    payload.get(
                        "prediction_population_share_threshold",
                        0.02,
                    )
                ),
                emit_signal=False,
            )
            self.prediction_panel.set_store_prediction_history(
                bool(payload.get("store_prediction_history", True)),
                emit_signal=False,
            )
            self.prediction_panel.set_q_settings(
                q_min=float(payload.get("q_min", 0.02)),
                q_max=float(payload.get("q_max", 1.20)),
                q_points=int(payload.get("q_points", 250)),
                emit_signal=False,
            )
        finally:
            self._suspend_preview_refresh = False
        self._refresh_selection_preview()

    def _detected_box_dimensions(self) -> tuple[float, float, float] | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions")
        if value is None:
            value = self._last_summary.get("estimated_box_dimensions")
        if value is None:
            return None
        return tuple(float(component) for component in value)

    def _sync_box_dimensions_from_summary(
        self,
        summary: dict[str, object] | None,
    ) -> None:
        if summary is None:
            self.definitions_panel.set_box_dimensions(None, emit_signal=False)
            return
        if summary.get("box_dimensions_source_kind") == "source_filename":
            value = summary.get("box_dimensions")
            if value is not None:
                self.definitions_panel.set_box_dimensions(
                    tuple(float(component) for component in value),
                    emit_signal=False,
                )
                return
        self.definitions_panel.set_box_dimensions(None, emit_signal=False)

    def _set_frame_format(self, frame_format: object | None) -> None:
        normalized = None if frame_format is None else str(frame_format)
        self._frame_format = normalized
        self.trajectory_panel.set_frame_mode(normalized)
        self.definitions_panel.set_frame_mode(normalized)

    def _detect_frame_format(
        self,
        frames_dir: Path | None,
    ) -> tuple[str | None, str | None]:
        if frames_dir is None:
            return None, None
        try:
            frame_format, _frame_paths = detect_frame_folder_mode(frames_dir)
        except ValueError as exc:
            return None, str(exc)
        return frame_format, None

    def _handle_error(self, title: str, message: str) -> None:
        self.run_panel.append_log(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    @staticmethod
    def _wrap_scroll_area(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_optional_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}"


def _format_optional_int(value: object) -> str:
    return "n/a" if value is None else str(int(value))


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text else Path(text)


def _optional_box_dimensions(
    value: object,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    components = tuple(float(component) for component in value)
    if len(components) != 3:
        return None
    return components


def _safe_filename_stem(value: str) -> str:
    cleaned = "".join(
        (
            character
            if character.isalnum() or character in {".", "_", "-"}
            else "_"
        )
        for character in str(value).strip()
    )
    return cleaned.strip("._") or "clusterdynamicsml"


def _format_int_sequence(values: tuple[int, ...] | list[int]) -> str:
    sequence = tuple(int(value) for value in values)
    if not sequence:
        return "n/a"
    if len(sequence) == 1:
        return str(sequence[0])
    expected = tuple(range(sequence[0], sequence[-1] + 1))
    if sequence == expected:
        return f"{sequence[0]}-{sequence[-1]}"
    return ",".join(str(value) for value in sequence)


def _read_project_history_entry(
    dataset_file: Path,
) -> _ProjectPredictionHistoryEntry | None:
    try:
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    analysis_settings = dict(payload.get("analysis_settings", {}))
    preview_payload = dict(payload.get("preview", {}))
    target_node_counts = tuple(
        int(value)
        for value in (
            analysis_settings.get(
                "target_node_counts",
                preview_payload.get("target_node_counts", []),
            )
            or []
        )
    )
    candidates_per_size = int(
        analysis_settings.get("candidates_per_size", 0) or 0
    )
    prediction_count = len(payload.get("predictions", []))
    max_predicted_node_count = payload.get("max_predicted_node_count")
    saxs_payload = payload.get("saxs_comparison")
    rmse = None
    if isinstance(saxs_payload, dict) and saxs_payload.get("rmse") is not None:
        rmse = float(saxs_payload["rmse"])
    modified_time = float(dataset_file.stat().st_mtime)
    return _ProjectPredictionHistoryEntry(
        dataset_file=dataset_file.expanduser().resolve(),
        modified_time=modified_time,
        saved_label=datetime.fromtimestamp(modified_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        target_node_counts=target_node_counts,
        candidates_per_size=candidates_per_size,
        prediction_population_share_threshold=float(
            analysis_settings.get(
                "prediction_population_share_threshold", 0.02
            )
        ),
        prediction_count=prediction_count,
        max_predicted_node_count=(
            None
            if max_predicted_node_count is None
            else int(max_predicted_node_count)
        ),
        rmse=rmse,
    )


def _write_table_widget_csv(table: QTableWidget, output_path: Path) -> Path:
    resolved_path = output_path.expanduser().resolve()
    if resolved_path.suffix.lower() != ".csv":
        resolved_path = resolved_path.with_suffix(".csv")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        (
            table.horizontalHeaderItem(column).text()
            if table.horizontalHeaderItem(column) is not None
            else f"column_{column}"
        )
        for column in range(table.columnCount())
    ]
    with resolved_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in range(table.rowCount()):
            writer.writerow(
                [
                    (
                        ""
                        if table.item(row, column) is None
                        else table.item(row, column).text()
                    )
                    for column in range(table.columnCount())
                ]
            )
    return resolved_path


def _size_rank_map(node_counts) -> dict[int, int]:
    unique_counts = sorted(
        {int(value) for value in node_counts if int(value) > 0},
        reverse=True,
    )
    return {
        int(node_count): index + 1
        for index, node_count in enumerate(unique_counts)
    }


def _combined_model_weight_rows(
    result: ClusterDynamicsMLResult,
) -> list[dict[str, object]]:
    observed_weights, _predicted_weights = _resolved_population_weights(
        result.training_observations,
        result.predictions,
        frame_timestep_fs=float(
            result.dynamics_result.preview.frame_timestep_fs
        ),
    )
    observed_total_weight = float(np.sum(observed_weights))
    observed_only_weight_by_label = {
        entry.label: (
            float(weight) / observed_total_weight
            if observed_total_weight > 0.0
            else 0.0
        )
        for entry, weight in zip(
            result.training_observations,
            observed_weights,
            strict=False,
        )
    }
    weight_by_observed_label: dict[str, tuple[float, str]] = {}
    weight_by_prediction_label: dict[str, tuple[float, str]] = {}
    if result.saxs_comparison is not None:
        for entry in result.saxs_comparison.component_weights:
            if str(entry.source).startswith("observed"):
                weight_by_observed_label[entry.label] = (
                    float(entry.weight),
                    str(entry.source),
                )
            elif str(entry.source) == "predicted":
                weight_by_prediction_label[entry.label] = (
                    float(entry.weight),
                    str(entry.source),
                )

    size_ranks = _size_rank_map(
        [
            *(entry.node_count for entry in result.training_observations),
            *(entry.target_node_count for entry in result.predictions),
        ]
    )
    rows: list[dict[str, object]] = []
    for entry in result.training_observations:
        normalized_weight, model_source = weight_by_observed_label.get(
            entry.label,
            (0.0, "not_in_model"),
        )
        rows.append(
            {
                "type": "Observed",
                "nodes": int(entry.node_count),
                "size_rank": int(size_ranks.get(int(entry.node_count), 0)),
                "candidate_rank": "",
                "label": entry.label,
                "observed_only_normalized_weight": float(
                    observed_only_weight_by_label.get(entry.label, 0.0)
                ),
                "normalized_weight": float(normalized_weight),
                "predicted_population_share": None,
                "mean_count_per_frame": float(entry.mean_count_per_frame),
                "occupancy_fraction": float(entry.occupancy_fraction),
                "mean_lifetime_fs": entry.mean_lifetime_fs,
                "std_lifetime_fs": entry.std_lifetime_fs,
                "completed_lifetime_count": int(
                    entry.completed_lifetime_count
                ),
                "window_truncated_lifetime_count": int(
                    entry.window_truncated_lifetime_count
                ),
                "association_rate_per_ps": float(
                    entry.association_rate_per_ps
                ),
                "dissociation_rate_per_ps": float(
                    entry.dissociation_rate_per_ps
                ),
                "model_source": model_source,
                "reference": "",
                "notes": "",
            }
        )
    for entry in result.predictions:
        normalized_weight, model_source = weight_by_prediction_label.get(
            entry.label,
            (0.0, "not_in_model"),
        )
        rows.append(
            {
                "type": "Predicted",
                "nodes": int(entry.target_node_count),
                "size_rank": int(
                    size_ranks.get(int(entry.target_node_count), 0)
                ),
                "candidate_rank": int(entry.rank),
                "label": entry.label,
                "observed_only_normalized_weight": None,
                "normalized_weight": float(normalized_weight),
                "predicted_population_share": float(
                    entry.predicted_population_share
                ),
                "mean_count_per_frame": float(
                    entry.predicted_mean_count_per_frame
                ),
                "occupancy_fraction": float(
                    entry.predicted_occupancy_fraction
                ),
                "mean_lifetime_fs": float(entry.predicted_mean_lifetime_fs),
                "std_lifetime_fs": None,
                "completed_lifetime_count": None,
                "window_truncated_lifetime_count": None,
                "association_rate_per_ps": float(
                    entry.predicted_association_rate_per_ps
                ),
                "dissociation_rate_per_ps": float(
                    entry.predicted_dissociation_rate_per_ps
                ),
                "model_source": model_source,
                "reference": (
                    ""
                    if entry.source_label is None
                    else str(entry.source_label)
                ),
                "notes": str(entry.notes),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["normalized_weight"]),
            str(row["type"]),
            -int(row["nodes"]),
            str(row["label"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["weight_rank"] = int(index)
    return rows


def launch_clusterdynamicsml_ui(
    initial_frames_dir: str | Path | None = None,
    *,
    energy_file: str | Path | None = None,
    project_dir: str | Path | None = None,
    clusters_dir: str | Path | None = None,
    experimental_data_file: str | Path | None = None,
) -> int:
    prepare_saxshell_application_identity()
    app = QApplication.instance()
    should_exec = app is None
    if app is None:
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = ClusterDynamicsMLMainWindow(
        initial_frames_dir=(
            None
            if initial_frames_dir is None
            else Path(initial_frames_dir).expanduser().resolve()
        ),
        initial_energy_file=(
            None
            if energy_file is None
            else Path(energy_file).expanduser().resolve()
        ),
        initial_project_dir=(
            None
            if project_dir is None
            else Path(project_dir).expanduser().resolve()
        ),
        initial_clusters_dir=(
            None
            if clusters_dir is None
            else Path(clusters_dir).expanduser().resolve()
        ),
        initial_experimental_data_file=(
            None
            if experimental_data_file is None
            else Path(experimental_data_file).expanduser().resolve()
        ),
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda _obj=None: _OPEN_WINDOWS.remove(window))
    if not should_exec:
        return 0
    return app.exec()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clusterdynamicsml",
        description=(
            "Predict larger-cluster surrogate stoichiometries, representative "
            "structures, and cluster-only SAXS traces from smaller-cluster "
            "cluster-dynamics and structure data."
        ),
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        help="Optional extracted frames directory to prefill in the UI.",
    )
    parser.add_argument(
        "--energy-file",
        help="Optional CP2K .ener file to prefill in the UI.",
    )
    parser.add_argument(
        "--project-dir",
        help="Optional SAXSShell project directory to prefill in the UI.",
    )
    parser.add_argument(
        "--clusters-dir",
        help="Optional smaller-cluster structure directory to prefill in the UI.",
    )
    parser.add_argument(
        "--experimental-data",
        help="Optional experimental SAXS data file to prefill in the UI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return launch_clusterdynamicsml_ui(
        getattr(args, "frames_dir", None),
        energy_file=getattr(args, "energy_file", None),
        project_dir=getattr(args, "project_dir", None),
        clusters_dir=getattr(args, "clusters_dir", None),
        experimental_data_file=getattr(args, "experimental_data", None),
    )
