from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
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
    QProgressBar,
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
    PairCutoffDefinitions,
    detect_frame_folder_mode,
    format_box_dimensions,
    format_search_mode_label,
    frame_folder_label,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.ui.trajectory_panel import ClusterTrajectoryPanel
from saxshell.clusterdynamics import (
    ClusterDynamicsResult,
    ClusterDynamicsSelectionPreview,
    ClusterDynamicsWorkflow,
)
from saxshell.clusterdynamics.dataset import (
    export_cluster_dynamics_colormap_csv,
    export_cluster_dynamics_lifetime_csv,
    load_cluster_dynamics_dataset,
    save_cluster_dynamics_dataset,
)
from saxshell.clusterdynamics.report import (
    default_powerpoint_report_path,
    export_cluster_dynamics_report_pptx,
)
from saxshell.clusterdynamics.ui.plot_panel import ClusterDynamicsPlotPanel
from saxshell.clusterdynamics.workflow import (
    _resolve_colormap_timestep_settings,
)
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
from saxshell.structure import AtomTypeDefinitions

_OPEN_WINDOWS: list["ClusterDynamicsMainWindow"] = []


@dataclass(slots=True)
class ClusterDynamicsJobConfig:
    """Analysis settings assembled from the UI."""

    frames_dir: Path
    energy_file: Path | None
    atom_type_definitions: AtomTypeDefinitions
    pair_cutoff_definitions: PairCutoffDefinitions
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

    @property
    def colormap_timestep_fs(self) -> float:
        return float(self.frame_timestep_fs) * float(
            self.frames_per_colormap_timestep
        )


class ClusterDynamicsWorker(QObject):
    """Background worker for time-binned cluster analysis."""

    progress = Signal(str)
    progress_count = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ClusterDynamicsJobConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            workflow = ClusterDynamicsWorkflow(
                self.config.frames_dir,
                atom_type_definitions=self.config.atom_type_definitions,
                pair_cutoff_definitions=self.config.pair_cutoff_definitions,
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
            )
            preview = workflow.preview_selection()
            self.progress.emit(
                "Preparing time-binned cluster analysis.\n"
                f"Frames selected: {preview.selected_frames}\n"
                f"Time bins: {preview.bin_count}\n"
                f"Frame timestep: {preview.frame_timestep_fs:.3f} fs\n"
                "Frames per colormap timestep: "
                f"{preview.frames_per_colormap_timestep}\n"
                f"Colormap timestep: {preview.colormap_timestep_fs:.3f} fs"
            )
            if preview.energy_file is not None:
                self.progress.emit(
                    f"Will also load CP2K energy data from: {preview.energy_file}"
                )
            total_frames = max(preview.selected_frames, 1)
            self.progress_count.emit(0, total_frames)
            log_interval = (
                1 if total_frames <= 10 else max(total_frames // 8, 25)
            )

            def on_progress(
                processed: int, total: int, frame_name: str
            ) -> None:
                self.progress_count.emit(processed, total)
                should_log = (
                    processed == 1
                    or processed >= total
                    or processed % log_interval == 0
                )
                if should_log:
                    self.progress.emit(
                        f"Analyzed {processed} of {total} frame(s). "
                        f"Last frame: {frame_name}."
                    )

            result = workflow.analyze(progress_callback=on_progress)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class ClusterDynamicsTimePanel(QGroupBox):
    """Time-axis and colormap-binning controls."""

    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Time Axis")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)

        self.folder_start_time_spin = self._make_optional_time_spin(
            "Auto-populated from mdtrajectory export metadata or a folder "
            "name such as splitxyz_f995_t497p5fs. This value is shown as "
            "the folder/start cutoff metadata and is used as a fallback "
            "origin when frame filenames do not expose the original "
            "source-frame indices."
        )
        self.folder_start_time_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        layout.addRow("Folder/start time (fs)", self.folder_start_time_spin)

        self.first_frame_time_spin = QDoubleSpinBox()
        self.first_frame_time_spin.setDecimals(3)
        self.first_frame_time_spin.setRange(0.0, 10**12)
        self.first_frame_time_spin.setSingleStep(1.0)
        self.first_frame_time_spin.setToolTip(
            "Fallback absolute simulation time assigned to the first "
            "extracted frame when the folder does not include mdtrajectory "
            "metadata and the frame filenames do not encode source-frame "
            "indices."
        )
        self.first_frame_time_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        layout.addRow("Fallback start time (fs)", self.first_frame_time_spin)

        self.frame_timestep_spin = QDoubleSpinBox()
        self.frame_timestep_spin.setDecimals(3)
        self.frame_timestep_spin.setRange(0.001, 10**9)
        self.frame_timestep_spin.setValue(0.5)
        self.frame_timestep_spin.setSingleStep(0.5)
        self.frame_timestep_spin.setToolTip(
            "Simulation timestep represented by one source trajectory frame. "
            "When extracted frame filenames preserve their original indices, "
            "the resolved time axis is frame_index x timestep."
        )
        self.frame_timestep_spin.valueChanged.connect(
            self._on_colormap_settings_changed
        )
        layout.addRow("Frame timestep (fs)", self.frame_timestep_spin)

        self.frames_per_colormap_timestep_spin = QSpinBox()
        self.frames_per_colormap_timestep_spin.setRange(1, 10**9)
        self.frames_per_colormap_timestep_spin.setValue(1)
        self.frames_per_colormap_timestep_spin.setToolTip(
            "Number of sampled frames combined into each heatmap timestep."
        )
        self.frames_per_colormap_timestep_spin.valueChanged.connect(
            self._on_colormap_settings_changed
        )
        layout.addRow(
            "Frames / colormap timestep",
            self.frames_per_colormap_timestep_spin,
        )

        self.colormap_timestep_value = QLineEdit()
        self.colormap_timestep_value.setReadOnly(True)
        self.colormap_timestep_value.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.colormap_timestep_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.colormap_timestep_value.setToolTip(
            "Derived heatmap timestep used for the colormap bins."
        )
        layout.addRow(
            "Colormap timestep used (fs)",
            self.colormap_timestep_value,
        )
        self._update_colormap_timestep_display()

        self.analysis_start_spin = self._make_optional_time_spin(
            "Leave at Auto to start from the first extracted frame time."
        )
        self.analysis_start_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        layout.addRow("Analysis start (fs)", self.analysis_start_spin)

        self.analysis_stop_spin = self._make_optional_time_spin(
            "Leave at Auto to use the full selected frame range."
        )
        self.analysis_stop_spin.valueChanged.connect(
            lambda _value: self.settings_changed.emit()
        )
        layout.addRow("Analysis stop (fs)", self.analysis_stop_spin)

    def _make_optional_time_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1.0, 10**12)
        spin.setSingleStep(10.0)
        spin.setSpecialValueText("Auto")
        spin.setValue(-1.0)
        spin.setToolTip(tooltip)
        return spin

    def _update_colormap_timestep_display(self) -> None:
        self.colormap_timestep_value.setText(
            f"{self.colormap_timestep_fs():.3f}"
        )

    def _on_colormap_settings_changed(self, _value: float | int) -> None:
        self._update_colormap_timestep_display()
        self.settings_changed.emit()

    def first_frame_time_fs(self) -> float:
        return float(self.first_frame_time_spin.value())

    def set_first_frame_time_fs(
        self,
        value: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.first_frame_time_spin.blockSignals(True)
        self.first_frame_time_spin.setValue(float(value))
        self.first_frame_time_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def folder_start_time_fs(self) -> float | None:
        value = float(self.folder_start_time_spin.value())
        return (
            None if value <= self.folder_start_time_spin.minimum() else value
        )

    def set_folder_start_time_fs(
        self,
        value: float | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.folder_start_time_spin.blockSignals(True)
        self.folder_start_time_spin.setValue(
            self.folder_start_time_spin.minimum()
            if value is None
            else float(value)
        )
        self.folder_start_time_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def frame_timestep_fs(self) -> float:
        return float(self.frame_timestep_spin.value())

    def set_frame_timestep_fs(
        self,
        value: float,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.frame_timestep_spin.blockSignals(True)
        self.frame_timestep_spin.setValue(float(value))
        self.frame_timestep_spin.blockSignals(False)
        self._update_colormap_timestep_display()
        if emit_signal:
            self.settings_changed.emit()

    def frames_per_colormap_timestep(self) -> int:
        return int(self.frames_per_colormap_timestep_spin.value())

    def set_frames_per_colormap_timestep(
        self,
        value: int,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.frames_per_colormap_timestep_spin.blockSignals(True)
        self.frames_per_colormap_timestep_spin.setValue(max(int(value), 1))
        self.frames_per_colormap_timestep_spin.blockSignals(False)
        self._update_colormap_timestep_display()
        if emit_signal:
            self.settings_changed.emit()

    def colormap_timestep_fs(self) -> float:
        return float(self.frame_timestep_spin.value()) * float(
            self.frames_per_colormap_timestep_spin.value()
        )

    def analysis_start_fs(self) -> float | None:
        value = float(self.analysis_start_spin.value())
        return None if value <= self.analysis_start_spin.minimum() else value

    def set_analysis_start_fs(
        self,
        value: float | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.analysis_start_spin.blockSignals(True)
        self.analysis_start_spin.setValue(
            self.analysis_start_spin.minimum()
            if value is None
            else float(value)
        )
        self.analysis_start_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def analysis_stop_fs(self) -> float | None:
        value = float(self.analysis_stop_spin.value())
        return None if value <= self.analysis_stop_spin.minimum() else value

    def set_analysis_stop_fs(
        self,
        value: float | None,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.analysis_stop_spin.blockSignals(True)
        self.analysis_stop_spin.setValue(
            self.analysis_stop_spin.minimum()
            if value is None
            else float(value)
        )
        self.analysis_stop_spin.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()


class ClusterDynamicsRunPanel(QGroupBox):
    """Panel for preview, optional energy input, and analysis logs."""

    analyze_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Run Analysis")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.energy_path_edit = QLineEdit()
        self.energy_path_edit.setToolTip(
            "Optional CP2K .ener file used for the lower time-series subplot."
        )
        self.energy_path_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "CP2K .ener file", self._make_file_row(self.energy_path_edit)
        )

        layout.addLayout(form)

        layout.addWidget(QLabel("Selection Preview"))
        self.selection_box = QTextEdit()
        self.selection_box.setReadOnly(True)
        self.selection_box.setMinimumHeight(150)
        layout.addWidget(self.selection_box)

        self.analyze_button = QPushButton("Analyze Time-Binned Clusters")
        self.analyze_button.clicked.connect(
            lambda _checked=False: self.analyze_requested.emit()
        )
        analyze_row = QHBoxLayout()
        analyze_row.setContentsMargins(0, 0, 0, 0)
        analyze_row.setSpacing(8)
        analyze_row.addWidget(self.analyze_button)
        self.auto_report_checkbox = QCheckBox("Detailed report")
        self.auto_report_checkbox.setChecked(False)
        self.auto_report_checkbox.setVisible(False)
        self.auto_report_checkbox.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        analyze_row.addWidget(self.auto_report_checkbox)
        analyze_row.addStretch(1)
        layout.addLayout(analyze_row)

        self.progress_label = QLabel("Progress: idle")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Run Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        layout.addWidget(self.log_box)

    def _make_file_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_file(line_edit)
        )
        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_file(self, line_edit: QLineEdit) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select CP2K .ener file",
            "",
            "Energy Files (*.ener);;All Files (*)",
        )
        if path:
            line_edit.setText(path)

    def energy_file(self) -> Path | None:
        text = self.energy_path_edit.text().strip()
        return Path(text) if text else None

    def configure_auto_report_option(
        self,
        *,
        visible: bool,
        text: str | None = None,
        tooltip: str | None = None,
        checked: bool | None = None,
        emit_signal: bool = True,
    ) -> None:
        self.auto_report_checkbox.blockSignals(True)
        if text is not None:
            self.auto_report_checkbox.setText(str(text))
        if tooltip is not None:
            self.auto_report_checkbox.setToolTip(str(tooltip))
        if checked is not None:
            self.auto_report_checkbox.setChecked(bool(checked))
        self.auto_report_checkbox.setVisible(bool(visible))
        self.auto_report_checkbox.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def auto_report_enabled(self) -> bool:
        return bool(
            not self.auto_report_checkbox.isHidden()
            and self.auto_report_checkbox.isChecked()
        )

    def set_auto_report_enabled(
        self,
        enabled: bool,
        *,
        emit_signal: bool = True,
    ) -> None:
        self.auto_report_checkbox.blockSignals(True)
        self.auto_report_checkbox.setChecked(bool(enabled))
        self.auto_report_checkbox.blockSignals(False)
        if emit_signal:
            self.settings_changed.emit()

    def set_selection_summary(self, text: str) -> None:
        self.selection_box.setPlainText(text)

    def set_log(self, text: str) -> None:
        self.log_box.setPlainText(text)

    def append_log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        existing = self.log_box.toPlainText().strip()
        if existing:
            self.log_box.append(message)
        else:
            self.log_box.setPlainText(message)

    def reset_progress(self) -> None:
        self.progress_label.setText("Progress: idle")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")

    def update_progress(self, processed: int, total: int) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_label.setText(
            f"Progress: {processed} processed, {max(total - processed, 0)} remaining"
        )
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m frames")


class ClusterDynamicsDatasetPanel(QGroupBox):
    """Panel for saving and reopening previously computed datasets."""

    save_dataset_requested = Signal()
    load_dataset_requested = Signal()
    save_colormap_requested = Signal()
    save_lifetime_requested = Signal()
    save_powerpoint_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Saved Results")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        helper = QLabel(
            "Save the current analysis result as a reloadable dataset, or "
            "open a previously saved dataset. These actions reuse saved "
            "results and do not rerun the frame analysis. You can also "
            "export the plotted colormap data and lifetime table directly "
            "as CSV files. When a project is set, related tools can also "
            "cache result bundles in the project's exported-results folder "
            "for later reuse."
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        form = QFormLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.setToolTip(
            "Optional active SAXSShell project used only to choose the "
            "default save/load folder in "
            "exported_results/data/clusterdynamics."
        )
        self.project_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Project for defaults",
            self._make_dir_row(self.project_dir_edit),
        )
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.save_dataset_button = QPushButton("Save Current Result")
        self.save_dataset_button.setToolTip(
            "Write the current plotted analysis result to a reloadable "
            "dataset file."
        )
        self.save_dataset_button.clicked.connect(
            lambda _checked=False: self.save_dataset_requested.emit()
        )
        self.load_dataset_button = QPushButton("Open Saved Result")
        self.load_dataset_button.setToolTip(
            "Load a previously saved cluster-dynamics dataset without "
            "rerunning the frame analysis."
        )
        self.load_dataset_button.clicked.connect(
            lambda _checked=False: self.load_dataset_requested.emit()
        )
        button_row.addWidget(self.save_dataset_button)
        button_row.addWidget(self.load_dataset_button)
        layout.addLayout(button_row)

        export_row = QHBoxLayout()
        self.save_colormap_button = QPushButton("Save Colormap Data")
        self.save_colormap_button.setToolTip(
            "Write the currently plotted heatmap data to a CSV file using "
            "the active display mode and time-unit selections."
        )
        self.save_colormap_button.clicked.connect(
            lambda _checked=False: self.save_colormap_requested.emit()
        )
        self.save_lifetime_button = QPushButton("Save Lifetime Table")
        self.save_lifetime_button.setToolTip(
            "Write the observed lifetime summary table to a CSV file."
        )
        self.save_lifetime_button.clicked.connect(
            lambda _checked=False: self.save_lifetime_requested.emit()
        )
        export_row.addWidget(self.save_colormap_button)
        export_row.addWidget(self.save_lifetime_button)
        layout.addLayout(export_row)

        report_row = QHBoxLayout()
        self.save_powerpoint_button = QPushButton("Save PowerPoint Report")
        self.save_powerpoint_button.setToolTip(
            "Generate a PowerPoint summary of the current result and append it "
            "to the existing project report when you save over that file."
        )
        self.save_powerpoint_button.clicked.connect(
            lambda _checked=False: self.save_powerpoint_requested.emit()
        )
        report_row.addWidget(self.save_powerpoint_button)
        report_row.addStretch(1)
        layout.addLayout(report_row)

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_directory(line_edit)
        )
        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_directory(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell project directory",
            line_edit.text().strip(),
        )
        if path:
            line_edit.setText(path)

    def project_dir(self) -> Path | None:
        text = self.project_dir_edit.text().strip()
        return Path(text) if text else None

    def set_project_dir(self, path: Path | None) -> None:
        self.project_dir_edit.setText("" if path is None else str(path))


class ClusterDynamicsMainWindow(QMainWindow):
    """Main Qt window for time-binned cluster-distribution analysis."""

    def __init__(
        self,
        initial_frames_dir: Path | None = None,
        initial_energy_file: Path | None = None,
        initial_project_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._project_manager = SAXSProjectManager()
        self._last_summary: dict[str, object] | None = None
        self._frame_format: str | None = None
        self._run_thread: QThread | None = None
        self._run_worker: ClusterDynamicsWorker | None = None
        self._last_result: ClusterDynamicsResult | None = None
        self._last_dataset_file: Path | None = None
        self._suspend_preview_refresh = False
        self._build_ui()

        if initial_frames_dir is not None:
            self.trajectory_panel.frames_dir_edit.setText(
                str(initial_frames_dir)
            )
        if initial_energy_file is not None:
            self.run_panel.energy_path_edit.setText(str(initial_energy_file))
        if initial_project_dir is not None:
            self.dataset_panel.set_project_dir(initial_project_dir)
        self._sync_project_defaults()

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            QMessageBox.warning(
                self,
                "Cluster Dynamics",
                "Please wait for the current cluster-dynamics analysis to "
                "finish before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (clusterdynamics)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1540, 920)

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
        self.run_panel = ClusterDynamicsRunPanel()
        self.dataset_panel = ClusterDynamicsDatasetPanel()

        left_layout.addWidget(self.trajectory_panel)
        left_layout.addWidget(self.time_panel)
        left_layout.addWidget(self.definitions_panel)
        left_layout.addWidget(self.run_panel)
        left_layout.addWidget(self.dataset_panel)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.plot_panel = ClusterDynamicsPlotPanel()
        right_layout.addWidget(self.plot_panel, stretch=3)

        self.results_tabs = QTabWidget()
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.label_table = self._build_lifetime_table(
            headers=(
                "Label",
                "Size",
                "Mean lifetime (fs)",
                "Std lifetime (fs)",
                "Completed",
                "Window-truncated",
                "Assoc. rate (1/ps)",
                "Dissoc. rate (1/ps)",
                "Occupancy (%)",
                "Mean count/frame",
            )
        )
        self.results_tabs.addTab(self.summary_box, "Summary")
        self.results_tabs.addTab(self.label_table, "Lifetime")
        right_layout.addWidget(self.results_tabs, stretch=2)

        splitter.addWidget(self._wrap_scroll_area(left))
        splitter.addWidget(right)
        splitter.setSizes([500, 1040])

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
        self.trajectory_panel.frames_dir_edit.editingFinished.connect(
            self._register_project_inputs
        )
        self.time_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.definitions_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.run_panel.settings_changed.connect(
            self._refresh_selection_preview
        )
        self.run_panel.energy_path_edit.editingFinished.connect(
            self._register_project_inputs
        )
        self.run_panel.analyze_requested.connect(self.run_analysis)
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
        self.dataset_panel.settings_changed.connect(
            self._on_project_dir_changed
        )

        self.run_panel.set_selection_summary(
            "Select an extracted PDB or XYZ frames folder to preview the "
            "time-binned cluster analysis."
        )
        self.run_panel.set_log(
            "Ready. Load a split frame folder from mdtrajectory, define the "
            "cluster rules, then run the time-binned analysis to build the "
            "cluster-distribution heatmap and lifetime table."
        )
        self._set_frame_format(None)

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
            self.run_panel.reset_progress()
            self.run_panel.set_log(
                "Time-binned cluster analysis request received.\n"
                f"Frames folder: {config.frames_dir}\n"
                f"Mode: {frame_folder_label(self._frame_format or 'pdb')}\n"
                f"PBC: {'on' if config.use_pbc else 'off'}\n"
                "Search mode: "
                f"{format_search_mode_label(config.search_mode)}\n"
                f"Frame timestep: {config.frame_timestep_fs:.3f} fs\n"
                "Frames per colormap timestep: "
                f"{config.frames_per_colormap_timestep}\n"
                f"Colormap timestep: {config.colormap_timestep_fs:.3f} fs"
            )
            self.plot_panel.set_result(None)
            self.summary_box.clear()
            self.label_table.setRowCount(0)
            self.statusBar().showMessage("Analyzing time-binned clusters...")
            self._start_worker(config)
        except Exception as exc:
            self._handle_error("Cluster dynamics analysis failed", str(exc))

    def _build_job_config(self) -> ClusterDynamicsJobConfig:
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
                "Add at least one pair-cutoff definition or specify a default cutoff."
            )

        use_pbc = self.definitions_panel.use_pbc()
        manual_box_dimensions = self.definitions_panel.box_dimensions()
        resolved_box_dimensions = manual_box_dimensions
        if use_pbc and resolved_box_dimensions is None:
            resolved_box_dimensions = self._detected_box_dimensions()
            if resolved_box_dimensions is None:
                raise ValueError(
                    "Periodic boundary conditions are enabled, but no box "
                    "dimensions are available. Enter a manual box or inspect "
                    "a frames folder with a usable coordinate extent."
                )

        return ClusterDynamicsJobConfig(
            frames_dir=frames_dir,
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
        )

    def _start_worker(self, config: ClusterDynamicsJobConfig) -> None:
        self._run_thread = QThread(self)
        self._run_worker = ClusterDynamicsWorker(config)
        self._run_worker.moveToThread(self._run_thread)

        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.progress.connect(self.run_panel.append_log)
        self._run_worker.progress_count.connect(self._on_run_progress)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.failed.connect(self._on_run_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.finished.connect(self._run_worker.deleteLater)
        self._run_thread.start()

    def _on_run_progress(self, processed: int, total: int) -> None:
        self.run_panel.update_progress(processed, total)
        self.statusBar().showMessage(
            f"Analyzing time-binned clusters... {processed}/{max(total, 1)} frames"
        )

    def _on_run_finished(self, result: ClusterDynamicsResult) -> None:
        self._last_result = result
        self.plot_panel.set_result(result)
        self.run_panel.update_progress(
            result.analyzed_frames, result.analyzed_frames
        )
        self.run_panel.append_log(
            "Time-binned analysis complete.\n"
            f"Frames analyzed: {result.analyzed_frames}\n"
            f"Time bins: {result.bin_count}\n"
            f"Unique cluster labels: {len(result.cluster_labels)}"
        )
        self._populate_summary_box(result)
        self._populate_label_table(result)
        registration_message = self._register_project_inputs()
        if registration_message is not None:
            self.run_panel.append_log(registration_message)
        self.statusBar().showMessage(
            "Cluster dynamics analysis complete"
            if registration_message is None
            else "Cluster dynamics analysis complete and project defaults updated"
        )

    def _on_run_failed(self, message: str) -> None:
        self.statusBar().showMessage("Cluster dynamics analysis failed")
        self._handle_error("Cluster dynamics analysis failed", message)

    def _cleanup_run_thread(self) -> None:
        self._run_worker = None
        self._run_thread = None

    def save_dataset(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved dataset before exporting."
            )
            return

        default_path = self._default_dataset_file()
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save cluster dynamics dataset",
            str(default_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        saved = save_cluster_dynamics_dataset(
            self._last_result,
            path,
            analysis_settings=self._analysis_settings_payload(),
        )
        self._last_dataset_file = saved.dataset_file
        self.run_panel.append_log(
            "Saved cluster-dynamics dataset to "
            f"{saved.dataset_file}\n"
            f"Wrote {len(saved.written_files)} file(s)."
        )
        self.statusBar().showMessage(
            f"Saved cluster dynamics dataset to {saved.dataset_file}"
        )

    def load_dataset(self) -> None:
        default_path = (
            self._last_dataset_file
            if self._last_dataset_file is not None
            else self._default_dataset_file()
        )
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load cluster dynamics dataset",
            str(default_path),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return

        loaded = load_cluster_dynamics_dataset(path)
        self._last_dataset_file = loaded.dataset_file
        self._apply_analysis_settings(loaded.analysis_settings)
        self._last_result = loaded.result
        self.plot_panel.set_result(loaded.result)
        self.run_panel.set_selection_summary(
            self._format_preview_text(loaded.result.preview)
        )
        self._populate_summary_box(loaded.result)
        self._populate_label_table(loaded.result)
        self.run_panel.append_log(
            "Loaded cluster-dynamics dataset from "
            f"{loaded.dataset_file}\n"
            f"Frames analyzed: {loaded.result.analyzed_frames}\n"
            f"Time bins: {loaded.result.bin_count}"
        )
        self.statusBar().showMessage(
            f"Loaded cluster dynamics dataset from {loaded.dataset_file}"
        )

    def save_colormap_data(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved dataset before exporting."
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

        display_mode = self.plot_panel.display_mode_combo.currentData()
        time_unit = self.plot_panel.time_unit_combo.currentData()
        saved_path = export_cluster_dynamics_colormap_csv(
            self._last_result,
            path,
            display_mode=(
                "fraction" if display_mode is None else str(display_mode)
            ),
            time_unit="fs" if time_unit is None else str(time_unit),
        )
        row_count = (
            len(self._last_result.cluster_labels) * self._last_result.bin_count
        )
        self.run_panel.append_log(
            "Saved cluster-dynamics colormap data to "
            f"{saved_path}\n"
            f"Rows written: {row_count}"
        )
        self.statusBar().showMessage(f"Saved colormap data to {saved_path}")

    def save_lifetime_table(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved dataset before exporting."
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

        saved_path = export_cluster_dynamics_lifetime_csv(
            self._last_result,
            path,
        )
        self.run_panel.append_log(
            "Saved cluster-dynamics lifetime table to "
            f"{saved_path}\n"
            f"Rows written: {len(self._last_result.lifetime_by_label)}"
        )
        self.statusBar().showMessage(f"Saved lifetime table to {saved_path}")

    def save_powerpoint_report(self) -> None:
        if self._last_result is None:
            self._show_error(
                "Run an analysis or load a saved dataset before exporting."
            )
            return

        self.plot_panel.set_result(self._last_result)
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
            "Save cluster dynamics PowerPoint report",
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
            export_result = export_cluster_dynamics_report_pptx(
                result=self._last_result,
                selection_summary=selection_summary,
                result_summary=summary_text,
                figure=self.plot_panel.figure,
                output_path=path,
                settings=self._powerpoint_export_settings(),
                project_dir=self.dataset_panel.project_dir(),
                frames_dir=self.trajectory_panel.get_frames_dir(),
                progress_callback=self._on_powerpoint_report_progress,
            )
        except Exception as exc:
            self.run_panel.progress_label.setText(
                "Progress: PowerPoint export failed"
            )
            self.run_panel.progress_bar.setRange(0, 1)
            self.run_panel.progress_bar.setValue(0)
            self.run_panel.progress_bar.setFormat("%v / %m steps")
            self._handle_error(
                "Cluster dynamics PowerPoint export failed", str(exc)
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
                "Appended cluster-dynamics report slides to "
                f"{export_result.report_path}\n"
                f"Slides added: {export_result.added_slide_count}"
            )
        else:
            self.run_panel.append_log(
                "Saved cluster-dynamics PowerPoint report to "
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
                    f"Detected {self._last_summary['n_frames']} extracted "
                    "frame(s) in the selected folder."
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
        self.plot_panel.set_result(None)
        self.summary_box.clear()
        self.label_table.setRowCount(0)
        self.time_panel.set_folder_start_time_fs(None, emit_signal=False)
        if frames_dir is None:
            self._sync_box_dimensions_from_summary(None)
            self._set_frame_format(None)
            self.trajectory_panel.set_summary_text("")
            self._refresh_selection_preview()
            return
        self._inspect_frames_dir(frames_dir, announce=False)

    def _refresh_selection_preview(self) -> None:
        if self._suspend_preview_refresh:
            return
        frames_dir = self.trajectory_panel.get_frames_dir()
        if frames_dir is None:
            self.run_panel.set_selection_summary(
                "Select an extracted PDB or XYZ frames folder to preview the "
                "time-binned cluster analysis."
            )
            return

        warning: str | None = None
        try:
            workflow = self._build_preview_workflow()
            preview = workflow.preview_selection()
            if (
                self.time_panel.folder_start_time_fs() is None
                and preview.folder_start_time_fs is not None
                and preview.folder_start_time_source != "manual field"
            ):
                self.time_panel.set_folder_start_time_fs(
                    preview.folder_start_time_fs,
                    emit_signal=False,
                )
            text = self._format_preview_text(preview)
        except Exception as exc:
            warning = str(exc)
            text = (
                "Adjust the current settings to preview the time-binned "
                f"analysis.\nValidation warning: {warning}"
            )
        self.run_panel.set_selection_summary(text)

    def _build_preview_workflow(self) -> ClusterDynamicsWorkflow:
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

        return ClusterDynamicsWorkflow(
            frames_dir,
            atom_type_definitions=self.definitions_panel.atom_type_definitions(),
            pair_cutoff_definitions=self.definitions_panel.pair_cutoff_definitions(),
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
        )

    def _format_preview_text(
        self,
        preview: ClusterDynamicsSelectionPreview,
    ) -> str:
        box_label = self._box_dimensions_label()
        resolved_box_text = format_box_dimensions(
            preview.resolved_box_dimensions
        )
        lines = [
            f"Mode: {frame_folder_label(preview.frame_format)}",
            f"PBC: {'on' if preview.use_pbc else 'off'}",
            "Search mode: "
            f"{format_search_mode_label(self.definitions_panel.search_mode())}",
            f"Frames in folder: {preview.total_frames}",
            f"Frames selected: {preview.selected_frames}",
            f"First frame time: {preview.first_frame_time_fs:.3f} fs",
            "Time source: " f"{preview.time_source_label}",
            f"Frame timestep: {preview.frame_timestep_fs:.3f} fs",
            f"Colormap timestep: {preview.colormap_timestep_fs:.3f} fs",
            f"Time window: {preview.analysis_start_fs:.3f} to "
            f"{preview.analysis_stop_fs:.3f} fs",
            f"Time bins: {preview.bin_count}",
            f"Shell growth: {self._shell_growth_text()}",
            "Stoichiometry bins: "
            + (
                "solute + shell atoms"
                if self.definitions_panel.include_shell_atoms_in_stoichiometry()
                else "solute only"
            ),
            f"{box_label}: {resolved_box_text}",
        ]
        if preview.frames_per_colormap_timestep is not None:
            lines.insert(
                8,
                "Frames per colormap timestep: "
                f"{preview.frames_per_colormap_timestep}",
            )
        box_source = self._box_dimensions_source()
        if box_source is not None:
            lines.append(f"Box source: {box_source}")
        if preview.folder_start_time_fs is not None:
            source_label = (
                f" ({preview.folder_start_time_source})"
                if preview.folder_start_time_source
                else ""
            )
            lines.append(
                "Folder/start time: "
                f"{preview.folder_start_time_fs:.3f} fs{source_label}"
            )
        if preview.first_selected_frame is not None:
            lines.append(
                "Frame file range: "
                f"{preview.first_selected_frame} to {preview.last_selected_frame}"
            )
        if preview.first_selected_source_frame_index is not None:
            lines.append(
                "Source frame index range: "
                f"{preview.first_selected_source_frame_index} to "
                f"{preview.last_selected_source_frame_index}"
            )
        if preview.first_selected_time_fs is not None:
            lines.append(
                "Selected frame times: "
                f"{preview.first_selected_time_fs:.3f} to "
                f"{preview.last_selected_time_fs:.3f} fs"
            )
        if preview.energy_file is not None:
            lines.append(f"Energy overlay: {preview.energy_file}")
        if preview.time_warnings:
            lines.extend(
                f"Warning: {message}" for message in preview.time_warnings
            )
        return "\n".join(lines)

    def _populate_summary_box(self, result: ClusterDynamicsResult) -> None:
        preview = result.preview
        lines = [
            f"Mode: {frame_folder_label(preview.frame_format)}",
            f"Frames analyzed: {result.analyzed_frames}",
            f"Time bins: {result.bin_count}",
            f"Unique cluster labels: {len(result.cluster_labels)}",
            f"Frame timestep: {preview.frame_timestep_fs:.3f} fs",
            f"Colormap timestep: {preview.colormap_timestep_fs:.3f} fs",
            f"Time source: {preview.time_source_label}",
            f"Time window: {preview.analysis_start_fs:.3f} to "
            f"{preview.analysis_stop_fs:.3f} fs",
            "Resolved box dimensions: "
            f"{format_box_dimensions(preview.resolved_box_dimensions)}",
            f"Total clusters sampled: {int(result.total_clusters_per_frame.sum())}",
        ]
        if preview.frames_per_colormap_timestep is not None:
            lines.insert(
                5,
                "Frames per colormap timestep: "
                f"{preview.frames_per_colormap_timestep}",
            )
        if preview.folder_start_time_fs is not None:
            lines.append(
                "Folder/start time: " f"{preview.folder_start_time_fs:.3f} fs"
            )
        if result.energy_data is not None:
            lines.append(
                f"Energy points in view: {len(result.energy_series('temperature')[0])}"
            )
        if preview.time_warnings:
            lines.extend(
                f"Warning: {message}" for message in preview.time_warnings
            )
        self.summary_box.setPlainText("\n".join(lines))

    def _populate_label_table(self, result: ClusterDynamicsResult) -> None:
        self.label_table.setSortingEnabled(False)
        self.label_table.setRowCount(len(result.lifetime_by_label))
        for row, entry in enumerate(result.lifetime_by_label):
            values = (
                entry.label,
                str(entry.cluster_size),
                _format_optional_float(entry.mean_lifetime_fs),
                _format_optional_float(entry.std_lifetime_fs),
                str(entry.completed_lifetime_count),
                str(entry.window_truncated_lifetime_count),
                f"{entry.association_rate_per_ps:.3f}",
                f"{entry.dissociation_rate_per_ps:.3f}",
                f"{entry.occupancy_fraction * 100.0:.1f}",
                f"{entry.mean_count_per_frame:.3f}",
            )
            for column, value in enumerate(values):
                self.label_table.setItem(row, column, QTableWidgetItem(value))
        self.label_table.resizeColumnsToContents()
        self.label_table.setSortingEnabled(True)

    @staticmethod
    def _build_lifetime_table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if len(headers) > 0:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        return table

    def _default_dataset_dir(self) -> Path:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is not None:
            paths = build_project_paths(project_dir)
            target_dir = paths.exported_data_dir / "clusterdynamics"
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
        folder_label = "cluster_dynamics"
        if frames_dir is not None:
            folder_label = frames_dir.name or folder_label
        return (
            self._default_dataset_dir()
            / f"{folder_label}_cluster_dynamics.json"
        )

    def _default_export_file(self, suffix_label: str) -> Path:
        dataset_file = self._default_dataset_file()
        return dataset_file.with_name(
            f"{dataset_file.stem}_{suffix_label}.csv"
        )

    def _default_powerpoint_report_file(self) -> Path:
        frames_dir = self.trajectory_panel.get_frames_dir()
        fallback_label = "cluster_dynamics_report"
        if frames_dir is not None:
            fallback_label = f"{frames_dir.name or 'cluster_dynamics'}_report"
        return default_powerpoint_report_path(
            project_dir=self.dataset_panel.project_dir(),
            fallback_dir=self._default_dataset_dir(),
            fallback_stem=fallback_label,
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
            "colormap_timestep_fs": self.time_panel.colormap_timestep_fs(),
            "analysis_start_fs": self.time_panel.analysis_start_fs(),
            "analysis_stop_fs": self.time_panel.analysis_stop_fs(),
        }

    def _apply_analysis_settings(self, payload: dict[str, object]) -> None:
        self._suspend_preview_refresh = True
        try:
            frames_dir = _optional_path(payload.get("frames_dir"))
            energy_file = _optional_path(payload.get("energy_file"))
            project_dir = _optional_path(payload.get("project_dir"))

            self.trajectory_panel.frames_dir_edit.setText(
                "" if frames_dir is None else str(frames_dir)
            )
            self.run_panel.energy_path_edit.setText(
                "" if energy_file is None else str(energy_file)
            )
            self.dataset_panel.set_project_dir(project_dir)

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
                    payload.get(
                        "include_shell_atoms_in_stoichiometry",
                        False,
                    )
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
            frame_timestep_fs = float(payload.get("frame_timestep_fs", 0.5))
            self.time_panel.set_frame_timestep_fs(
                frame_timestep_fs,
                emit_signal=False,
            )
            (
                frames_per_colormap_timestep,
                colormap_timestep_fs,
            ) = _resolve_colormap_timestep_settings(
                frame_timestep_fs=frame_timestep_fs,
                frames_per_colormap_timestep=payload.get(
                    "frames_per_colormap_timestep"
                ),
                colormap_timestep_fs=_optional_float(
                    payload.get("colormap_timestep_fs")
                ),
                legacy_bin_size_fs=_optional_float(payload.get("bin_size_fs")),
                require_integral_ratio=False,
            )
            if frames_per_colormap_timestep is None:
                frames_per_colormap_timestep = max(
                    int(
                        (
                            float(colormap_timestep_fs)
                            / float(frame_timestep_fs)
                        )
                        + 0.5
                    ),
                    1,
                )
            self.time_panel.set_frames_per_colormap_timestep(
                frames_per_colormap_timestep,
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
        finally:
            self._suspend_preview_refresh = False
        self._refresh_selection_preview()

    def _detected_box_dimensions(
        self,
    ) -> tuple[float, float, float] | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions")
        if value is None:
            value = self._last_summary.get("estimated_box_dimensions")
        if value is None:
            return None
        return tuple(float(component) for component in value)

    def _box_dimensions_source_kind(self) -> str | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions_source_kind")
        return None if value is None else str(value)

    def _box_dimensions_source(self) -> str | None:
        if self._last_summary is None:
            return None
        value = self._last_summary.get("box_dimensions_source")
        return None if value is None else str(value)

    def _box_dimensions_label(self) -> str:
        if self._box_dimensions_source_kind() == "source_filename":
            return "Source box dimensions"
        return "Estimated box dimensions"

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

    def _shell_growth_text(self) -> str:
        levels = self.definitions_panel.shell_growth_levels()
        if not levels:
            return "core only"
        return ", ".join(str(level) for level in levels)

    def _on_project_dir_changed(self) -> None:
        changed = self._sync_project_defaults()
        if changed:
            self._refresh_selection_preview()

    def _sync_project_defaults(self) -> bool:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return False
        project_file = build_project_paths(project_dir).project_file
        if not project_file.is_file():
            return False
        try:
            settings = self._project_manager.load_project(project_dir)
        except Exception:
            return False
        changed = False
        if (
            self.trajectory_panel.get_frames_dir() is None
            and settings.resolved_frames_dir is not None
        ):
            self.trajectory_panel.frames_dir_edit.setText(
                str(settings.resolved_frames_dir)
            )
            changed = True
        if (
            self.run_panel.energy_file() is None
            and settings.resolved_energy_file is not None
        ):
            self.run_panel.energy_path_edit.setText(
                str(settings.resolved_energy_file)
            )
            changed = True
        return changed

    def _register_project_inputs(self) -> str | None:
        project_dir = self.dataset_panel.project_dir()
        if project_dir is None:
            return None
        project_file = build_project_paths(project_dir).project_file
        if not project_file.is_file():
            return None
        try:
            settings = self._project_manager.load_project(project_dir)
            frames_dir = self.trajectory_panel.get_frames_dir()
            energy_file = self.run_panel.energy_file()
            settings.frames_dir = (
                None
                if frames_dir is None
                else str(Path(frames_dir).expanduser().resolve())
            )
            settings.energy_file = (
                None
                if energy_file is None
                else str(Path(energy_file).expanduser().resolve())
            )
            self._project_manager.save_project(settings)
        except Exception as exc:
            return (
                "Analysis finished, but the project frames/energy "
                f"references could not be updated: {exc}"
            )
        updates: list[str] = []
        if frames_dir is not None:
            updates.append(f"frames={Path(frames_dir).expanduser().resolve()}")
        if energy_file is not None:
            updates.append(
                f"energy={Path(energy_file).expanduser().resolve()}"
            )
        if not updates:
            return None
        return "Updated project references: " + ", ".join(updates)

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


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


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
        raise ValueError("Saved box dimensions must contain three values.")
    return components


def launch_clusterdynamics_ui(
    frames_dir: str | Path | None = None,
    *,
    energy_file: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> int:
    """Launch the Qt6 cluster-dynamics UI."""
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = ClusterDynamicsMainWindow(
        initial_frames_dir=(None if frames_dir is None else Path(frames_dir)),
        initial_energy_file=(
            None if energy_file is None else Path(energy_file)
        ),
        initial_project_dir=(
            None if project_dir is None else Path(project_dir)
        ),
    )
    _OPEN_WINDOWS.append(window)
    window.show()
    if owns_app:
        return app.exec()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for launching the cluster-dynamics UI."""
    parser = argparse.ArgumentParser(
        prog="clusterdynamics-ui",
        description=(
            "Launch the SAXSShell clusterdynamics UI for time-binned "
            "cluster-distribution analysis on extracted frame folders."
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
    args = parser.parse_args(argv)
    return launch_clusterdynamics_ui(
        args.frames_dir,
        energy_file=args.energy_file,
        project_dir=args.project_dir,
    )


__all__ = [
    "ClusterDynamicsJobConfig",
    "ClusterDynamicsMainWindow",
    "ClusterDynamicsWorker",
    "launch_clusterdynamics_ui",
    "main",
]
