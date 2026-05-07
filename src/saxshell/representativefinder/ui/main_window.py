from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisPreset,
    BondPairDefinition,
    load_presets,
    ordered_preset_names,
    save_custom_preset,
)
from saxshell.representativefinder.workflow import (
    RepresentativeFinderCandidate,
    RepresentativeFinderFolderInspection,
    RepresentativeFinderInputInspection,
    RepresentativeFinderOperationCancelled,
    RepresentativeFinderPlotSeries,
    RepresentativeFinderResult,
    RepresentativeFinderSettings,
    analyze_representative_structure_folder,
    estimate_representativefinder_total_work,
    inspect_representative_structure_input,
    load_representativefinder_result,
    persist_representativefinder_result_to_project,
    suggest_representativefinder_output_dir,
    suggest_representativefinder_target_output_dir,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityStructureViewer,
)
from saxshell.saxs.electron_density_mapping.workflow import (
    ElectronDensityMeshGeometry,
    ElectronDensityStructure,
    build_electron_density_mesh,
    legacy_born_average_default_mesh_settings,
    load_electron_density_structure,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

_ALGORITHM_ITEMS = [
    (
        "Quantile Distance (Recommended)",
        "target_distribution_quantile_distance",
    ),
    ("Mean/Std Distance", "target_distribution_moment_distance"),
]
_ANALYSIS_MODE_ITEMS = [
    ("Selected Stoichiometry Only", "single"),
    ("All Discovered Stoichiometries", "all"),
]
_DISPLAY_MODE_ITEMS = [
    ("Selected Candidate", "selected_candidate"),
    ("Observed Representative", "observed_representative"),
    (
        "Predicted Optimized Representative",
        "predicted_optimized_representative",
    ),
    (
        "Solvent-completed Predicted Representative",
        "solvent_completed_predicted_representative",
    ),
]


@dataclass(slots=True)
class RepresentativeFinderAnalysisTarget:
    inspection: RepresentativeFinderFolderInspection
    output_dir: Path
    estimated_total_work: int


@dataclass(slots=True)
class RepresentativeFinderJobConfig:
    analysis_mode: str
    targets: tuple[RepresentativeFinderAnalysisTarget, ...]
    settings: RepresentativeFinderSettings
    project_dir: Path | None = None


@dataclass(slots=True, frozen=True)
class RepresentativeFinderTargetFailure:
    input_dir: Path
    structure_label: str
    message: str


@dataclass(slots=True, frozen=True)
class RepresentativeFinderRunSummary:
    analysis_mode: str
    targets: tuple[RepresentativeFinderAnalysisTarget, ...]
    results: tuple[RepresentativeFinderResult, ...]
    failures: tuple[RepresentativeFinderTargetFailure, ...]


@dataclass(slots=True)
class RepresentativeFinderSessionState:
    input_dir_text: str
    output_dir_text: str
    analysis_mode: str
    settings: RepresentativeFinderSettings | None
    results_by_input_dir: dict[str, RepresentativeFinderResult]
    failures_by_input_dir: dict[str, str]
    selected_stoichiometry_key: str | None
    display_mode: str | None
    console_text: str


_PROJECT_SESSION_STATES: dict[str, RepresentativeFinderSessionState] = {}


class RepresentativeFinderWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    target_started = Signal(str)
    result_ready = Signal(object)
    target_failed = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, config: RepresentativeFinderJobConfig) -> None:
        super().__init__()
        self.config = config
        self._cancel_requested = False

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return (
            self._cancel_requested
            or QThread.currentThread().isInterruptionRequested()
        )

    @Slot()
    def run(self) -> None:
        try:
            results: list[RepresentativeFinderResult] = []
            failures: list[RepresentativeFinderTargetFailure] = []
            target_count = len(self.config.targets)
            global_total_work = max(
                sum(
                    target.estimated_total_work
                    for target in self.config.targets
                ),
                1,
            )
            completed_work = 0

            for index, target in enumerate(self.config.targets, start=1):
                if self.is_cancel_requested():
                    raise RepresentativeFinderOperationCancelled(
                        "Representative-structure analysis canceled."
                    )
                target_key = str(target.inspection.input_dir)
                target_label = target.inspection.structure_label
                target_prefix = (
                    f"[{index}/{target_count}] {target_label}: "
                    if target_count > 1
                    else ""
                )
                log_prefix = f"[{target_label}] " if target_count > 1 else ""
                self.target_started.emit(target_key)
                self.log.emit(
                    f"{log_prefix}Starting representative selection for "
                    f"{target_label}."
                )

                def on_progress(
                    processed: int,
                    total: int,
                    message: str,
                ) -> None:
                    del total
                    bounded = min(
                        max(int(processed), 0),
                        max(target.estimated_total_work, 1),
                    )
                    self.progress.emit(
                        min(completed_work + bounded, global_total_work),
                        global_total_work,
                        f"{target_prefix}{message}",
                    )

                def on_log(message: str) -> None:
                    self.log.emit(f"{log_prefix}{message}")

                try:
                    result = analyze_representative_structure_folder(
                        target.inspection.input_dir,
                        settings=self.config.settings,
                        output_dir=target.output_dir,
                        project_dir=self.config.project_dir,
                        progress_callback=on_progress,
                        log_callback=on_log,
                        cancel_callback=self.is_cancel_requested,
                    )
                except RepresentativeFinderOperationCancelled:
                    raise
                except Exception as exc:
                    failure = RepresentativeFinderTargetFailure(
                        input_dir=target.inspection.input_dir,
                        structure_label=target_label,
                        message=str(exc),
                    )
                    failures.append(failure)
                    self.target_failed.emit(failure)
                    completed_work += target.estimated_total_work
                    self.progress.emit(
                        min(completed_work, global_total_work),
                        global_total_work,
                        f"{target_prefix}failed",
                    )
                    if target_count == 1:
                        self.failed.emit(str(exc))
                        return
                    continue

                results.append(result)
                self.result_ready.emit(result)
                completed_work += target.estimated_total_work

            completion_message = (
                "Representative selection complete."
                if not failures
                else (
                    "Representative selection complete with "
                    f"{len(failures)} failed stoichiometry run(s)."
                )
            )
            self.progress.emit(
                global_total_work, global_total_work, completion_message
            )
            self.finished.emit(
                RepresentativeFinderRunSummary(
                    analysis_mode=self.config.analysis_mode,
                    targets=self.config.targets,
                    results=tuple(results),
                    failures=tuple(failures),
                )
            )
        except RepresentativeFinderOperationCancelled:
            self.canceled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class RepresentativeDistributionPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: RepresentativeFinderResult | None = None
        self._candidate: RepresentativeFinderCandidate | None = None
        self._plot_series: tuple[RepresentativeFinderPlotSeries, ...] = ()
        self._selected_series_index = 0
        self.figure = Figure(figsize=(9.0, 7.4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._build_ui()
        self.draw_placeholder()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Distribution"))
        self.previous_distribution_button = QPushButton("<")
        self.previous_distribution_button.clicked.connect(
            self._select_previous_distribution
        )
        selector_row.addWidget(self.previous_distribution_button)
        self.distribution_selector_combo = QComboBox()
        self.distribution_selector_combo.currentIndexChanged.connect(
            self._on_distribution_changed
        )
        selector_row.addWidget(self.distribution_selector_combo, stretch=1)
        self.next_distribution_button = QPushButton(">")
        self.next_distribution_button.clicked.connect(
            self._select_next_distribution
        )
        selector_row.addWidget(self.next_distribution_button)
        layout.addLayout(selector_row)
        layout.addWidget(self.toolbar)
        self.canvas.setMinimumHeight(340)
        layout.addWidget(self.canvas, stretch=1)

    def draw_placeholder(self) -> None:
        self._result = None
        self._candidate = None
        self._set_plot_series(())
        self._draw_message(
            "Run representative selection to compare the folder-wide bond and angle distributions with one candidate structure.",
            secondary=(
                "Use the distribution selector or Previous/Next buttons to inspect one computed distribution at a time."
            ),
        )

    def set_result(
        self,
        result: RepresentativeFinderResult | None,
        *,
        candidate: RepresentativeFinderCandidate | None = None,
    ) -> None:
        previous_label = self._current_distribution_label()
        self._result = result
        self._candidate = candidate or (
            None if result is None else result.selected_candidate
        )
        if self._result is None or self._candidate is None:
            self._set_plot_series(())
        else:
            filtered_series = tuple(
                series
                for series in self._result.plot_series_for_candidate(
                    self._candidate
                )
                if series.distribution_values.size > 0
                or bool(series.candidate_values)
            )
            self._set_plot_series(
                filtered_series,
                selected_label=previous_label,
            )
        self.refresh_plot()

    def refresh_plot(self) -> None:
        self.figure.clear()
        if self._result is None or self._candidate is None:
            self._draw_message(
                "Run representative selection to compare the folder-wide bond and angle distributions with one candidate structure.",
                secondary=(
                    "Use the distribution selector or Previous/Next buttons to inspect one computed distribution at a time."
                ),
            )
            return
        if not self._plot_series:
            self._draw_message(
                "No computed bond or angle distributions are available for the active stoichiometry and candidate.",
                secondary=(
                    "Only distributions with measured values for the active stoichiometry are listed in the selector."
                ),
            )
            return
        self._selected_series_index = min(
            max(self._selected_series_index, 0),
            len(self._plot_series) - 1,
        )
        series = self._plot_series[self._selected_series_index]
        axis = self.figure.add_subplot(111)
        self._draw_series(axis, series)
        self.figure.suptitle(
            f"Distribution Comparison • {self._candidate.file_name}",
            y=0.995,
        )
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
        self.canvas.draw_idle()

    def _set_plot_series(
        self,
        plot_series: tuple[RepresentativeFinderPlotSeries, ...],
        *,
        selected_label: str | None = None,
    ) -> None:
        self._plot_series = tuple(plot_series)
        target_index = 0
        if self._plot_series and selected_label:
            for index, series in enumerate(self._plot_series):
                if self._series_selector_label(series) == selected_label:
                    target_index = index
                    break
        self._selected_series_index = min(
            max(target_index, 0),
            max(len(self._plot_series) - 1, 0),
        )
        self.distribution_selector_combo.blockSignals(True)
        self.distribution_selector_combo.clear()
        for series in self._plot_series:
            self.distribution_selector_combo.addItem(
                self._series_selector_label(series)
            )
        if self._plot_series:
            self.distribution_selector_combo.setCurrentIndex(
                self._selected_series_index
            )
        self.distribution_selector_combo.blockSignals(False)
        self._update_distribution_controls()

    def _update_distribution_controls(self) -> None:
        series_count = len(self._plot_series)
        has_series = series_count > 0
        self.distribution_selector_combo.setEnabled(has_series)
        self.previous_distribution_button.setEnabled(series_count > 1)
        self.next_distribution_button.setEnabled(series_count > 1)

    def _current_distribution_label(self) -> str | None:
        if not self._plot_series:
            return None
        return self._series_selector_label(
            self._plot_series[
                min(
                    max(self._selected_series_index, 0),
                    len(self._plot_series) - 1,
                )
            ]
        )

    @staticmethod
    def _series_selector_label(series: RepresentativeFinderPlotSeries) -> str:
        prefix = "Bond" if series.category == "bond" else "Angle"
        return f"{prefix}: {series.display_label}"

    def _draw_message(
        self,
        message: str,
        *,
        secondary: str | None = None,
    ) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(
            0.5,
            0.56 if secondary else 0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=axis.transAxes,
        )
        if secondary:
            axis.text(
                0.5,
                0.40,
                secondary,
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
                alpha=0.8,
            )
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _on_distribution_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._plot_series):
            return
        self._selected_series_index = index
        self.refresh_plot()

    def _select_previous_distribution(self) -> None:
        if len(self._plot_series) <= 1:
            return
        new_index = (self._selected_series_index - 1) % len(self._plot_series)
        self.distribution_selector_combo.setCurrentIndex(new_index)

    def _select_next_distribution(self) -> None:
        if len(self._plot_series) <= 1:
            return
        new_index = (self._selected_series_index + 1) % len(self._plot_series)
        self.distribution_selector_combo.setCurrentIndex(new_index)

    def _draw_series(
        self, axis, series: RepresentativeFinderPlotSeries
    ) -> None:
        if series.distribution_values.size > 0:
            axis.hist(
                series.distribution_values,
                bins=60,
                color="#355070" if series.category == "bond" else "#bc6c25",
                edgecolor="white",
                alpha=0.88,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No distribution values were available for this definition.",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
        for index, value in enumerate(series.candidate_values):
            axis.axvline(
                value,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=("Candidate value" if index == 0 else None),
            )
        axis.set_title(series.display_label)
        axis.set_xlabel(series.xlabel)
        axis.set_ylabel("Count")
        if series.candidate_values:
            axis.legend(frameon=False, loc="upper right")


class RepresentativeStructureFinderMainWindow(QMainWindow):
    project_results_changed = Signal(str)

    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_input_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._browse_start_dir = (
            self._initial_project_dir
            if self._initial_project_dir is not None
            else Path.home()
        )
        self._last_suggested_output_dir: str | None = None
        self._presets: dict[str, BondAnalysisPreset] = {}
        self._input_inspection: RepresentativeFinderInputInspection | None = (
            None
        )
        self._analysis_results_by_input_dir: dict[
            str, RepresentativeFinderResult
        ] = {}
        self._analysis_failures_by_input_dir: dict[str, str] = {}
        self._stoichiometry_row_by_input_dir: dict[str, int] = {}
        self._active_stoichiometry_key: str | None = None
        self._run_summary: RepresentativeFinderRunSummary | None = None
        self._viewer_scene_payload_by_path: dict[
            str,
            tuple[
                ElectronDensityStructure, ElectronDensityMeshGeometry | None
            ],
        ] = {}
        self._shared_project_representative_path_by_input_dir: dict[
            str, Path
        ] = {}
        self._shared_project_representative_entry_by_input_dir: dict[
            str,
            object,
        ] = {}
        self._analysis_thread: QThread | None = None
        self._analysis_worker: RepresentativeFinderWorker | None = None
        self._closing_after_analysis_cancel = False

        self.setWindowTitle("Representative Structures")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1380, 900)
        self._build_ui()
        self._refresh_analysis_mode_ui()
        self._reload_presets()

        if initial_input_path is not None:
            resolved_input_path = (
                Path(initial_input_path).expanduser().resolve()
            )
            if resolved_input_path.is_dir():
                self._browse_start_dir = resolved_input_path
                self.input_dir_edit.setText(str(resolved_input_path))
        self._refresh_input_preview()
        restored_from_session = self._restore_project_session_state()
        if not restored_from_session:
            self._restore_project_cached_results_with_startup_progress()

    def closeEvent(self, event) -> None:
        if (
            self._analysis_thread is not None
            and self._analysis_thread.isRunning()
        ):
            self._cancel_analysis_for_close()
            event.ignore()
            return
        self._save_project_session_state()
        super().closeEvent(event)

    def _cancel_analysis(self) -> None:
        worker = self._analysis_worker
        if worker is not None:
            worker.cancel()
        thread = self._analysis_thread
        if thread is not None and thread.isRunning():
            thread.requestInterruption()
            thread.quit()

    def _cancel_analysis_for_close(self) -> None:
        if self._closing_after_analysis_cancel:
            return
        self._closing_after_analysis_cancel = True
        self.run_status_label.setText(
            "Representative selection: canceling so the window can close..."
        )
        self.statusBar().showMessage("Stopping representative selection...")
        self._append_console(
            "Close requested. Canceling representative-structure analysis."
        )
        self._cancel_analysis()
        self.setEnabled(False)
        self.hide()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        intro_label = QLabel(
            "Select one representative structure per stoichiometry using the shared bondanalysis preset workflow and solvent-aware scoring from the contrast descriptor backend. Saved project representatives can then be reused by compatible SAXS and RMCSetup tools."
        )
        intro_label.setWordWrap(True)
        root.addWidget(intro_label)

        self._pane_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._pane_splitter.setChildrenCollapsible(False)
        self._pane_splitter.setStretchFactor(0, 0)
        self._pane_splitter.setStretchFactor(1, 1)
        root.addWidget(self._pane_splitter, stretch=1)

        self._left_scroll = QScrollArea(self)
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._right_scroll = QScrollArea(self)
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._left_panel = QWidget()
        self._right_panel = QWidget()
        self._left_layout = QVBoxLayout(self._left_panel)
        self._left_layout.setContentsMargins(10, 10, 10, 10)
        self._left_layout.setSpacing(10)
        self._right_layout = QVBoxLayout(self._right_panel)
        self._right_layout.setContentsMargins(10, 10, 10, 10)
        self._right_layout.setSpacing(10)
        self._left_scroll.setWidget(self._left_panel)
        self._right_scroll.setWidget(self._right_panel)
        self._pane_splitter.addWidget(self._left_scroll)
        self._pane_splitter.addWidget(self._right_scroll)
        self._pane_splitter.setSizes([430, 900])

        self._build_left_panel()
        self._build_right_panel()

        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> None:
        self._left_layout.addWidget(self._build_input_group())
        self._left_layout.addWidget(self._build_preset_group())
        self._left_layout.addWidget(self._build_bond_pairs_group())
        self._left_layout.addWidget(self._build_angle_triplets_group())
        self._left_layout.addWidget(self._build_advanced_group())
        self._left_layout.addWidget(self._build_solvent_shell_group())
        self._left_layout.addWidget(self._build_run_group())
        self._left_layout.addStretch(1)

    def _build_right_panel(self) -> None:
        self._right_splitter = QSplitter(
            Qt.Orientation.Vertical,
            self._right_panel,
        )
        self._right_splitter.setChildrenCollapsible(False)
        self._right_layout.addWidget(self._right_splitter, stretch=1)

        self._stoichiometry_group = self._build_stoichiometry_group()
        self._result_summary_group = self._build_result_summary_group()
        self._candidate_scores_group = self._build_candidate_scores_group()
        self._plot_group = self._build_plot_group()
        self._viewer_group = self._build_viewer_group()

        for widget in (
            self._stoichiometry_group,
            self._result_summary_group,
            self._candidate_scores_group,
            self._plot_group,
            self._viewer_group,
        ):
            self._right_splitter.addWidget(widget)

        self._right_splitter.setStretchFactor(0, 3)
        self._right_splitter.setStretchFactor(1, 2)
        self._right_splitter.setStretchFactor(2, 2)
        self._right_splitter.setStretchFactor(3, 3)
        self._right_splitter.setStretchFactor(4, 3)
        self._apply_initial_right_splitter_sizes()
        QTimer.singleShot(0, self._apply_initial_right_splitter_sizes)

    def _apply_initial_right_splitter_sizes(self) -> None:
        if not hasattr(self, "_right_splitter"):
            return
        self._right_splitter.setSizes([420, 210, 240, 320, 300])

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input Settings")
        layout = QVBoxLayout(group)
        form = QFormLayout()

        if self._initial_project_dir is not None:
            self.project_dir_edit = QLineEdit(str(self._initial_project_dir))
            self.project_dir_edit.setReadOnly(True)
            form.addRow("Project folder", self.project_dir_edit)
        else:
            self.project_dir_edit = None

        input_row = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText(
            "Choose one stoichiometry folder, or a parent folder whose immediate subfolders are stoichiometries..."
        )
        self.input_dir_edit.editingFinished.connect(
            self._refresh_input_preview
        )
        input_row.addWidget(self.input_dir_edit, stretch=1)
        self.browse_input_button = QPushButton("Browse...")
        self.browse_input_button.clicked.connect(self._browse_input_dir)
        input_row.addWidget(self.browse_input_button)
        input_widget = QWidget()
        input_widget.setLayout(input_row)
        form.addRow("Input folder", input_widget)

        self.analysis_mode_combo = QComboBox()
        for label, value in _ANALYSIS_MODE_ITEMS:
            self.analysis_mode_combo.addItem(label, value)
        self.analysis_mode_combo.currentIndexChanged.connect(
            self._refresh_analysis_mode_ui
        )
        form.addRow("Analysis mode", self.analysis_mode_combo)

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Representative output folder or batch output root"
        )
        output_row.addWidget(self.output_dir_edit, stretch=1)
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(self.browse_output_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)
        form.addRow("Output folder", output_widget)

        layout.addLayout(form)
        self.input_preview_box = QPlainTextEdit()
        self.input_preview_box.setReadOnly(True)
        self.input_preview_box.setMinimumHeight(130)
        layout.addWidget(self.input_preview_box)
        return group

    def _build_preset_group(self) -> QGroupBox:
        group = QGroupBox("Bondanalysis Presets")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.preset_combo = QComboBox()
        row.addWidget(self.preset_combo, stretch=1)
        self.load_preset_button = QPushButton("Load")
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        row.addWidget(self.load_preset_button)
        self.save_preset_button = QPushButton("Save Current As...")
        self.save_preset_button.clicked.connect(self._save_current_preset)
        row.addWidget(self.save_preset_button)
        layout.addLayout(row)
        hint = QLabel(
            "Uses the same preset file as bondanalysis so the same bond-pair and angle-triplet definitions can drive representative selection here."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _build_bond_pairs_group(self) -> QGroupBox:
        group = QGroupBox("Bond Pairs")
        layout = QVBoxLayout(group)
        controls = QHBoxLayout()
        self.add_bond_pair_button = QPushButton("Add Bond Pair")
        self.add_bond_pair_button.clicked.connect(self._add_bond_pair_row)
        controls.addWidget(self.add_bond_pair_button)
        self.remove_bond_pair_button = QPushButton("Remove Selected")
        self.remove_bond_pair_button.clicked.connect(
            self._remove_selected_bond_pair_rows
        )
        controls.addWidget(self.remove_bond_pair_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.bond_pair_table = QTableWidget(0, 3)
        self.bond_pair_table.setHorizontalHeaderLabels(
            ["Atom 1", "Atom 2", "Cutoff (A)"]
        )
        self.bond_pair_table.horizontalHeader().setStretchLastSection(True)
        self.bond_pair_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.bond_pair_table)
        self._add_empty_bond_pair_row(blocked=True)
        return group

    def _build_angle_triplets_group(self) -> QGroupBox:
        group = QGroupBox("Angle Triplets")
        layout = QVBoxLayout(group)
        controls = QHBoxLayout()
        self.add_angle_triplet_button = QPushButton("Add Angle Triplet")
        self.add_angle_triplet_button.clicked.connect(
            self._add_angle_triplet_row
        )
        controls.addWidget(self.add_angle_triplet_button)
        self.remove_angle_triplet_button = QPushButton("Remove Selected")
        self.remove_angle_triplet_button.clicked.connect(
            self._remove_selected_angle_triplet_rows
        )
        controls.addWidget(self.remove_angle_triplet_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.angle_triplet_table = QTableWidget(0, 5)
        self.angle_triplet_table.setHorizontalHeaderLabels(
            [
                "Vertex",
                "Arm 1",
                "Arm 2",
                "Vertex-Arm 1 Cutoff (A)",
                "Vertex-Arm 2 Cutoff (A)",
            ]
        )
        self.angle_triplet_table.horizontalHeader().setStretchLastSection(True)
        self.angle_triplet_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self.angle_triplet_table)
        self._add_empty_angle_triplet_row(blocked=True)
        return group

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced Scoring")
        layout = QFormLayout(group)
        self.algorithm_combo = QComboBox()
        for label, value in _ALGORITHM_ITEMS:
            self.algorithm_combo.addItem(label, value)
        layout.addRow("Bond/angle distance", self.algorithm_combo)
        self.bond_weight_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=1.0,
        )
        layout.addRow("Bond weight", self.bond_weight_spin)
        self.angle_weight_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=1.0,
        )
        layout.addRow("Angle weight", self.angle_weight_spin)
        self.solvent_weight_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=3,
            value=1.0,
        )
        layout.addRow("Solvent weight", self.solvent_weight_spin)
        hint = QLabel(
            "Solvent scoring is inferred automatically from the selected stoichiometry folder name and the solvent-shell metrics measured from each candidate structure."
        )
        hint.setWordWrap(True)
        layout.addRow("", hint)
        self.generate_predicted_checkbox = QCheckBox(
            "Generate Predicted Optimized Representative"
        )
        self.generate_predicted_checkbox.setChecked(False)
        layout.addRow("", self.generate_predicted_checkbox)
        predicted_hint = QLabel(
            "Optional: generate a synthetic optimized representative alongside the observed representative. When project solvent settings are available, the tool will also attempt a solvent-completed predicted output."
        )
        predicted_hint.setWordWrap(True)
        layout.addRow("", predicted_hint)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        self.overwrite_existing_checkbox = QCheckBox(
            "Overwrite Existing Representative Structures"
        )
        self.overwrite_existing_checkbox.setChecked(False)
        self.overwrite_existing_checkbox.setToolTip(
            "When unchecked, stoichiometries with saved project "
            "representatives are skipped instead of recalculated."
        )
        layout.addWidget(self.overwrite_existing_checkbox)
        button_row = QHBoxLayout()
        self.run_button = QPushButton("Analyze Representative Structure")
        self.run_button.clicked.connect(self._run_analysis)
        button_row.addWidget(self.run_button)
        self.open_output_button = QPushButton("Show Output Path")
        self.open_output_button.clicked.connect(self._show_output_folder)
        self.open_output_button.setEnabled(False)
        button_row.addWidget(self.open_output_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        self.run_status_label = QLabel("Representative selection: idle")
        self.run_status_label.setWordWrap(True)
        layout.addWidget(self.run_status_label)
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        layout.addWidget(self.run_progress_bar)
        self.console_box = QPlainTextEdit()
        self.console_box.setReadOnly(True)
        self.console_box.setMinimumHeight(220)
        layout.addWidget(self.console_box)
        return group

    def _build_solvent_shell_group(self) -> QGroupBox:
        group = QGroupBox("Build Solvent Shell")
        layout = QVBoxLayout(group)
        self.solvent_shell_toggle_button = QPushButton(
            "Show Solvent Shell Builder Options"
        )
        self.solvent_shell_toggle_button.setCheckable(True)
        self.solvent_shell_toggle_button.setChecked(False)
        self.solvent_shell_toggle_button.toggled.connect(
            self._toggle_solvent_shell_options
        )
        layout.addWidget(self.solvent_shell_toggle_button)

        self.solvent_shell_body = QWidget(group)
        body_layout = QVBoxLayout(self.solvent_shell_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        hint = QLabel(
            "Open the shared solvent-shell builder preloaded with the active representative structure. Use this after representative selection when the stored source representative does not yet contain the solvent shell you want to preserve."
        )
        hint.setWordWrap(True)
        body_layout.addWidget(hint)
        self.solvent_shell_status_label = QLabel(
            "Select or compute a representative structure to enable this handoff."
        )
        self.solvent_shell_status_label.setWordWrap(True)
        body_layout.addWidget(self.solvent_shell_status_label)
        button_row = QHBoxLayout()
        self.open_selected_solvent_shell_button = QPushButton(
            "Open for Selected Representative"
        )
        self.open_selected_solvent_shell_button.clicked.connect(
            self._open_solvent_shell_builder_for_selected_representative
        )
        button_row.addWidget(self.open_selected_solvent_shell_button)
        button_row.addStretch(1)
        body_layout.addLayout(button_row)
        self.solvent_shell_body.setVisible(False)
        layout.addWidget(self.solvent_shell_body)
        self._refresh_solvent_shell_controls()
        return group

    def _build_result_summary_group(self) -> QGroupBox:
        group = QGroupBox("Selected Stoichiometry Summary")
        layout = QVBoxLayout(group)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Displayed structure"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setEnabled(False)
        self.display_mode_combo.currentIndexChanged.connect(
            self._update_selected_candidate_view
        )
        selector_row.addWidget(self.display_mode_combo, stretch=1)
        layout.addLayout(selector_row)
        self.result_summary_box = QPlainTextEdit()
        self.result_summary_box.setReadOnly(True)
        self.result_summary_box.setMinimumHeight(180)
        self.result_summary_box.setPlainText(
            "Select a stoichiometry row to inspect it. Completed runs will populate the summary, candidate scores, plots, and representative viewer."
        )
        layout.addWidget(self.result_summary_box)
        return group

    def _build_stoichiometry_group(self) -> QGroupBox:
        group = QGroupBox("Stoichiometry Results")
        layout = QVBoxLayout(group)
        self.stoichiometry_table = QTableWidget(0, 8)
        self.stoichiometry_table.setMinimumHeight(280)
        self.stoichiometry_table.setHorizontalHeaderLabels(
            [
                "Stoichiometry",
                "Candidates",
                "Motifs",
                "Status",
                "Representative",
                "Score",
                "Output",
                "Open",
            ]
        )
        self.stoichiometry_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.stoichiometry_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.stoichiometry_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.stoichiometry_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.stoichiometry_table.horizontalHeader().setStretchLastSection(True)
        self.stoichiometry_table.itemSelectionChanged.connect(
            self._update_selected_stoichiometry_view
        )
        layout.addWidget(self.stoichiometry_table)
        return group

    def _build_candidate_scores_group(self) -> QGroupBox:
        group = QGroupBox("Candidate Scores")
        layout = QVBoxLayout(group)
        self.candidate_table = QTableWidget(0, 8)
        self.candidate_table.setHorizontalHeaderLabels(
            [
                "File",
                "Source",
                "Score",
                "Bond",
                "Angle",
                "Solvent",
                "Atoms",
                "Solvent Atoms",
            ]
        )
        self.candidate_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.candidate_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.candidate_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.candidate_table.horizontalHeader().setStretchLastSection(False)
        self.candidate_table.itemSelectionChanged.connect(
            self._update_selected_candidate_view
        )
        layout.addWidget(self.candidate_table)
        return group

    def _build_plot_group(self) -> QGroupBox:
        group = QGroupBox("Distribution Plots")
        layout = QVBoxLayout(group)
        self.plot_widget = RepresentativeDistributionPlotWidget(self)
        layout.addWidget(self.plot_widget)
        return group

    def _build_viewer_group(self) -> QGroupBox:
        group = QGroupBox("Structure Viewer")
        layout = QVBoxLayout(group)
        self.viewer_widget = ElectronDensityStructureViewer(self)
        self.viewer_widget.mesh_contrast_spin.setValue(90.0)
        self.viewer_widget.mesh_linewidth_spin.setValue(1.6)
        layout.addWidget(self.viewer_widget)
        return group

    def _browse_input_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select stoichiometry folder or stoichiometry parent folder",
            str(self._browse_start_dir),
        )
        if not selected:
            return
        self.input_dir_edit.setText(selected)
        self._browse_start_dir = Path(selected).expanduser().resolve()
        self._refresh_input_preview()

    def _browse_output_dir(self) -> None:
        start_dir = self.output_dir_edit.text().strip() or str(
            self._browse_start_dir
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select representative output folder",
            start_dir,
        )
        if selected:
            self.output_dir_edit.setText(selected)

    def _current_analysis_mode(self) -> str:
        payload = self.analysis_mode_combo.currentData()
        if payload is None:
            return "single"
        return str(payload)

    def _refresh_analysis_mode_ui(self) -> None:
        mode = self._current_analysis_mode()
        if hasattr(self, "run_button"):
            self.run_button.setText(
                "Analyze All Stoichiometries"
                if mode == "all"
                else "Analyze Selected Stoichiometry"
            )
        if hasattr(self, "output_dir_edit"):
            self._refresh_suggested_output_dir()

    def _refresh_input_preview(self) -> None:
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            self._input_inspection = None
            self._reset_analysis_results()
            self.input_preview_box.setPlainText(
                "Choose a stoichiometry folder, or a parent folder whose immediate subfolders are stoichiometries, to inspect representative-selection inputs."
            )
            self._populate_stoichiometry_table(None)
            self._refresh_project_representative_path_cache()
            self._refresh_solvent_shell_controls()
            return
        try:
            inspection = inspect_representative_structure_input(input_dir)
        except Exception as exc:
            self._input_inspection = None
            self._reset_analysis_results()
            self.input_preview_box.setPlainText(str(exc))
            self._populate_stoichiometry_table(None)
            self._refresh_project_representative_path_cache()
            self._refresh_solvent_shell_controls()
            self.statusBar().showMessage(
                "Representative folder inspection failed"
            )
            return
        self._input_inspection = inspection
        self._reset_analysis_results()
        self.input_preview_box.setPlainText(inspection.summary_text())
        self._populate_stoichiometry_table(inspection)
        self._refresh_project_representative_path_cache()
        self._refresh_solvent_shell_controls()
        self.statusBar().showMessage(
            f"Discovered {inspection.stoichiometry_count} stoichiometry folder(s)"
        )
        self._refresh_suggested_output_dir()

    def _toggle_solvent_shell_options(self, checked: bool) -> None:
        expanded = bool(checked)
        self.solvent_shell_body.setVisible(expanded)
        self.solvent_shell_toggle_button.setText(
            "Hide Solvent Shell Builder Options"
            if expanded
            else "Show Solvent Shell Builder Options"
        )

    def _refresh_project_representative_path_cache(self) -> None:
        self._shared_project_representative_path_by_input_dir = {}
        self._shared_project_representative_entry_by_input_dir = {}
        if self._initial_project_dir is None or self._input_inspection is None:
            return
        try:
            from saxshell.fullrmc.project_model import (
                ensure_rmcsetup_structure,
            )
            from saxshell.fullrmc.representatives import (
                load_representative_selection_metadata,
            )
        except Exception:
            return
        rmcsetup_paths = ensure_rmcsetup_structure(self._initial_project_dir)
        metadata = load_representative_selection_metadata(
            rmcsetup_paths.representative_selection_path
        )
        if metadata is None:
            return
        shared_entry_by_structure = {}
        for entry in metadata.representative_entries:
            structure = str(entry.structure).strip()
            source_file = str(entry.source_file).strip()
            if structure and source_file:
                shared_entry_by_structure[structure] = entry
        for stoich in self._input_inspection.stoichiometry_folders:
            representative_entry = shared_entry_by_structure.get(
                stoich.structure_label
            )
            if representative_entry is None:
                continue
            shared_path = (
                Path(representative_entry.source_file).expanduser().resolve()
            )
            key = str(stoich.input_dir)
            self._shared_project_representative_path_by_input_dir[key] = (
                shared_path
            )
            self._shared_project_representative_entry_by_input_dir[key] = (
                representative_entry
            )
            row = self._stoichiometry_row_by_input_dir.get(key)
            if row is not None:
                output_text, output_tooltip = (
                    self._project_representative_output_display(
                        representative_entry,
                        shared_path,
                    )
                )
                self._update_stoichiometry_row(
                    key,
                    status="Complete",
                    representative=(
                        str(representative_entry.source_file_name).strip()
                        or shared_path.name
                    ),
                    score=_format_score(
                        getattr(representative_entry, "score_total", None)
                    ),
                    output_text=output_text,
                    output_tooltip=output_tooltip,
                    representative_path=shared_path,
                )

    def _project_representative_output_display(
        self,
        representative_entry: object,
        shared_path: Path,
    ) -> tuple[str, str]:
        cached_results_path = str(
            getattr(representative_entry, "cached_results_path", "") or ""
        ).strip()
        if cached_results_path:
            output_dir = (
                Path(cached_results_path).expanduser().resolve().parent
            )
            return output_dir.name, str(output_dir)
        source_dir = shared_path.parent
        return source_dir.name, str(source_dir)

    def _project_representative_path_for_key(
        self, key: str | None
    ) -> Path | None:
        if key is None:
            return None
        return self._shared_project_representative_path_by_input_dir.get(key)

    def _project_representative_entry_for_key(
        self, key: str | None
    ) -> object | None:
        if key is None:
            return None
        return self._shared_project_representative_entry_by_input_dir.get(key)

    def _current_display_mode(self) -> str | None:
        if not hasattr(self, "display_mode_combo"):
            return None
        payload = self.display_mode_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _set_display_mode(self, mode: str) -> bool:
        if not hasattr(self, "display_mode_combo"):
            return False
        for index in range(self.display_mode_combo.count()):
            if self.display_mode_combo.itemData(index) == mode:
                self.display_mode_combo.setCurrentIndex(index)
                return True
        return False

    def _refresh_display_mode_options(
        self,
        result: RepresentativeFinderResult | None,
    ) -> None:
        previous_mode = self._current_display_mode()
        self.display_mode_combo.blockSignals(True)
        self.display_mode_combo.clear()
        self.display_mode_combo.setEnabled(False)
        if result is not None:
            available_modes = {"selected_candidate", "observed_representative"}
            if (
                result.predicted_candidate is not None
                and result.predicted_output_path is not None
            ):
                available_modes.add("predicted_optimized_representative")
            if (
                result.solvent_completed_predicted_candidate is not None
                and result.solvent_completed_predicted_output_path is not None
            ):
                available_modes.add(
                    "solvent_completed_predicted_representative"
                )
            for label, value in _DISPLAY_MODE_ITEMS:
                if value in available_modes:
                    self.display_mode_combo.addItem(label, value)
            self.display_mode_combo.setEnabled(
                self.display_mode_combo.count() > 0
            )
            if previous_mode is not None and self._set_display_mode(
                previous_mode
            ):
                pass
            elif self.display_mode_combo.count() > 0:
                self.display_mode_combo.setCurrentIndex(0)
        self.display_mode_combo.blockSignals(False)

    def _active_display_payload(
        self,
    ) -> tuple[RepresentativeFinderCandidate, Path, str] | None:
        result = self._selected_result()
        if result is None:
            return None
        mode = self._current_display_mode() or "selected_candidate"
        shared_project_path = self._project_representative_path_for_key(
            str(result.input_dir)
        )
        if mode == "observed_representative":
            return (
                result.selected_candidate,
                shared_project_path or result.representative_output_path,
                "Observed Representative",
            )
        if (
            mode == "predicted_optimized_representative"
            and result.predicted_candidate is not None
            and result.predicted_output_path is not None
        ):
            return (
                result.predicted_candidate,
                result.predicted_output_path,
                "Predicted Optimized Representative",
            )
        if (
            mode == "solvent_completed_predicted_representative"
            and result.solvent_completed_predicted_candidate is not None
            and result.solvent_completed_predicted_output_path is not None
        ):
            return (
                result.solvent_completed_predicted_candidate,
                result.solvent_completed_predicted_output_path,
                "Solvent-completed Predicted Representative",
            )
        candidate = self._selected_candidate()
        if candidate is None:
            candidate = result.selected_candidate
        return candidate, candidate.file_path, "Selected Candidate"

    def _active_representative_input_path(self) -> Path | None:
        payload = self._active_display_payload()
        if payload is not None:
            _candidate, display_path, _display_label = payload
            if display_path.is_file():
                return display_path
        key = self._selected_stoichiometry_key()
        shared_path = self._project_representative_path_for_key(key)
        if shared_path is not None and shared_path.is_file():
            return shared_path
        return None

    def _refresh_solvent_shell_controls(self) -> None:
        representative_path = self._active_representative_input_path()
        self.open_selected_solvent_shell_button.setEnabled(
            representative_path is not None
        )
        project_root_text = (
            ""
            if self._initial_project_dir is None
            else f"Project root: {self._initial_project_dir}\n"
        )
        if representative_path is not None:
            self.solvent_shell_status_label.setText(
                project_root_text
                + "Selected representative source:\n"
                + str(representative_path)
            )
            return
        self.solvent_shell_status_label.setText(
            project_root_text
            + "Select or compute a representative structure to enable this handoff."
        )

    def _open_solvent_shell_builder_for_selected_representative(self) -> None:
        from saxshell.fullrmc.ui.solvent_shell_builder_window import (
            launch_solvent_shell_builder_ui,
        )

        representative_path = self._active_representative_input_path()
        if representative_path is None:
            QMessageBox.information(
                self,
                "Solvent Shell Builder",
                "Select or compute a representative structure first.",
            )
            return
        window = launch_solvent_shell_builder_ui(
            initial_project_dir=self._initial_project_dir,
            initial_input_path=representative_path,
        )
        window.raise_()
        self.statusBar().showMessage(
            f"Opened solvent shell builder for {representative_path.name}"
        )
        self._append_console(
            "Opened solvent shell builder for representative structure: "
            f"{representative_path}"
        )

    def _refresh_suggested_output_dir(self) -> None:
        inspection = self._input_inspection
        if inspection is None:
            return
        batch_output = (
            self._current_analysis_mode() == "all"
            or inspection.stoichiometry_count > 1
            or not inspection.input_is_stoichiometry_folder
        )
        suggestion_source = (
            inspection.input_dir
            if batch_output
            else inspection.stoichiometry_folders[0].input_dir
        )
        suggested_output = suggest_representativefinder_output_dir(
            suggestion_source,
            project_dir=self._initial_project_dir,
            batch=batch_output,
        )
        current_output = self.output_dir_edit.text().strip()
        if (
            not current_output
            or current_output == self._last_suggested_output_dir
        ):
            self.output_dir_edit.setText(str(suggested_output))
        self._last_suggested_output_dir = str(suggested_output)

    def _project_session_cache_key(self) -> str | None:
        if self._initial_project_dir is not None:
            return str(self._initial_project_dir)
        input_dir_text = self.input_dir_edit.text().strip()
        if input_dir_text:
            try:
                return str(Path(input_dir_text).expanduser().resolve())
            except Exception:
                return input_dir_text
        return None

    def _capture_project_session_state(
        self,
    ) -> RepresentativeFinderSessionState | None:
        cache_key = self._project_session_cache_key()
        if cache_key is None or not self._analysis_results_by_input_dir:
            return None
        settings: RepresentativeFinderSettings | None = None
        try:
            settings = self._current_settings()
        except Exception:
            result = next(
                iter(self._analysis_results_by_input_dir.values()), None
            )
            settings = None if result is None else result.settings
        return RepresentativeFinderSessionState(
            input_dir_text=self.input_dir_edit.text().strip(),
            output_dir_text=self.output_dir_edit.text().strip(),
            analysis_mode=self._current_analysis_mode(),
            settings=settings,
            results_by_input_dir=dict(self._analysis_results_by_input_dir),
            failures_by_input_dir=dict(self._analysis_failures_by_input_dir),
            selected_stoichiometry_key=self._selected_stoichiometry_key(),
            display_mode=self._current_display_mode(),
            console_text=self.console_box.toPlainText(),
        )

    def _save_project_session_state(self) -> None:
        cache_key = self._project_session_cache_key()
        if cache_key is None:
            return
        state = self._capture_project_session_state()
        if state is None:
            return
        _PROJECT_SESSION_STATES[cache_key] = state

    def _restore_project_session_state(self) -> bool:
        cache_key = self._project_session_cache_key()
        if cache_key is None:
            return False
        state = _PROJECT_SESSION_STATES.get(cache_key)
        if state is None:
            return False
        restored_input_dir = str(state.input_dir_text).strip()
        if (
            restored_input_dir
            and restored_input_dir != self.input_dir_edit.text().strip()
        ):
            restored_input_path = (
                Path(restored_input_dir).expanduser().resolve()
            )
            if restored_input_path.is_dir():
                self._browse_start_dir = restored_input_path
                self.input_dir_edit.setText(str(restored_input_path))
                self._refresh_input_preview()
        self._set_analysis_mode(state.analysis_mode)
        if state.output_dir_text:
            self.output_dir_edit.setText(state.output_dir_text)
            self._last_suggested_output_dir = state.output_dir_text
        if state.settings is not None:
            self._apply_settings_to_controls(state.settings)
        restored_any = False
        valid_keys = set(self._stoichiometry_row_by_input_dir)
        self._analysis_results_by_input_dir = {}
        self._analysis_failures_by_input_dir = {}
        for key, result in state.results_by_input_dir.items():
            if key not in valid_keys:
                continue
            self._analysis_results_by_input_dir[key] = result
            self._update_stoichiometry_row(
                key,
                status="Complete",
                representative=result.selected_candidate.file_name,
                score=_format_score(result.selected_candidate.score_total),
                output_text=result.output_dir.name,
                output_tooltip=str(result.output_dir),
                representative_path=result.representative_output_path,
            )
            restored_any = True
        for key, message in state.failures_by_input_dir.items():
            if (
                key not in valid_keys
                or key in self._analysis_results_by_input_dir
            ):
                continue
            self._analysis_failures_by_input_dir[key] = message
            self._update_stoichiometry_row(
                key,
                status="Failed",
                representative="",
                score="",
                output_text="",
                output_tooltip="",
                representative_path=False,
            )
            restored_any = True
        if not restored_any:
            return False
        self.console_box.setPlainText(state.console_text)
        self.run_status_label.setText(
            "Representative selection: restored from project session"
        )
        self.open_output_button.setEnabled(
            bool(self._analysis_results_by_input_dir)
            or bool(self.output_dir_edit.text().strip())
        )
        selected_key = state.selected_stoichiometry_key
        if (
            selected_key is not None
            and selected_key in self._analysis_results_by_input_dir
        ):
            self._select_stoichiometry_row_by_key(selected_key)
        else:
            first_key = next(iter(self._analysis_results_by_input_dir), None)
            if first_key is not None:
                self._select_stoichiometry_row_by_key(first_key)
        if state.display_mode:
            self._set_display_mode(state.display_mode)
        self._refresh_solvent_shell_controls()
        self.statusBar().showMessage(
            "Restored representative-structure results from the current project session"
        )
        return True

    def _restore_project_cached_results_with_startup_progress(self) -> bool:
        if self._initial_project_dir is None or self._input_inspection is None:
            return False
        load_items: list[tuple[str, str, object, Path]] = []
        for stoich in self._input_inspection.stoichiometry_folders:
            key = str(stoich.input_dir)
            representative_entry = self._project_representative_entry_for_key(
                key
            )
            if representative_entry is None:
                continue
            cached_result_path = self._project_cached_result_path_for_entry(
                representative_entry
            )
            if cached_result_path is None:
                continue
            load_items.append(
                (
                    key,
                    stoich.structure_label,
                    representative_entry,
                    cached_result_path,
                )
            )
        if not load_items:
            return False

        progress_dialog = QProgressDialog(
            "Loading saved representative-structure analysis...",
            "Cancel",
            0,
            len(load_items),
            self,
        )
        progress_dialog.setWindowTitle("Loading Representative Structures")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setCancelButton(None)
        progress_dialog.setValue(0)
        progress_dialog.show()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        restored_keys: list[str] = []
        first_settings: RepresentativeFinderSettings | None = None
        for (
            index,
            (key, label, representative_entry, cached_result_path),
        ) in enumerate(load_items, start=1):
            progress_dialog.setLabelText(
                f"Loading saved representative analysis for {label}..."
            )
            if app is not None:
                app.processEvents()
            try:
                result = load_representativefinder_result(cached_result_path)
            except Exception as exc:
                self._append_console(
                    "Unable to restore saved representative analysis for "
                    f"{label}: {exc}"
                )
                progress_dialog.setValue(index)
                continue

            result.input_dir = Path(key).expanduser().resolve()
            self._analysis_results_by_input_dir[key] = result
            self._analysis_failures_by_input_dir.pop(key, None)
            if first_settings is None:
                first_settings = result.settings
            shared_path = self._project_representative_path_for_key(key)
            output_text, output_tooltip = (
                self._project_representative_output_display(
                    representative_entry,
                    shared_path or result.representative_output_path,
                )
            )
            if result.output_dir:
                output_text = result.output_dir.name
                output_tooltip = str(result.output_dir)
            self._update_stoichiometry_row(
                key,
                status="Complete",
                representative=(
                    str(
                        getattr(
                            representative_entry,
                            "source_file_name",
                            "",
                        )
                    ).strip()
                    or result.selected_candidate.file_name
                ),
                score=_format_score(result.selected_candidate.score_total),
                output_text=output_text,
                output_tooltip=output_tooltip,
                representative_path=(
                    shared_path or result.representative_output_path
                ),
            )
            restored_keys.append(key)
            progress_dialog.setValue(index)
            if app is not None:
                app.processEvents()

        progress_dialog.close()
        if not restored_keys:
            return False

        if first_settings is not None:
            self._apply_settings_to_controls(first_settings)
        self.run_status_label.setText(
            "Representative selection: restored from saved project analysis"
        )
        self.open_output_button.setEnabled(True)
        self._append_console(
            "Loaded saved representative-structure analysis for "
            f"{len(restored_keys)} stoichiometry row(s) from the active project."
        )
        selected_key = self._selected_stoichiometry_key()
        if selected_key not in self._analysis_results_by_input_dir:
            selected_key = restored_keys[0]
        self._select_stoichiometry_row_by_key(selected_key)
        self._refresh_solvent_shell_controls()
        self.statusBar().showMessage(
            "Loaded saved representative-structure analysis from the active project"
        )
        return True

    def _project_cached_result_path_for_entry(
        self,
        representative_entry: object,
    ) -> Path | None:
        for attribute_name in (
            "project_cached_results_path",
            "cached_results_path",
        ):
            path_text = str(
                getattr(representative_entry, attribute_name, "") or ""
            ).strip()
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            if (
                not path.is_absolute()
                and self._initial_project_dir is not None
            ):
                path = self._initial_project_dir / path
            resolved_path = path.resolve()
            if resolved_path.is_file():
                return resolved_path
        return None

    def _set_analysis_mode(self, mode: str) -> None:
        for index in range(self.analysis_mode_combo.count()):
            if self.analysis_mode_combo.itemData(index) == mode:
                self.analysis_mode_combo.setCurrentIndex(index)
                return

    def _apply_settings_to_controls(
        self,
        settings: RepresentativeFinderSettings,
    ) -> None:
        self._set_bond_pair_rows(settings.bond_pairs)
        self._set_angle_triplet_rows(settings.angle_triplets)
        self.bond_weight_spin.setValue(float(settings.bond_weight))
        self.angle_weight_spin.setValue(float(settings.angle_weight))
        self.solvent_weight_spin.setValue(float(settings.solvent_weight))
        self.generate_predicted_checkbox.setChecked(
            bool(settings.generate_predicted_optimized_representative)
        )
        for index in range(self.algorithm_combo.count()):
            if (
                self.algorithm_combo.itemData(index)
                == settings.selection_algorithm
            ):
                self.algorithm_combo.setCurrentIndex(index)
                break

    def _reset_analysis_results(self) -> None:
        self._analysis_results_by_input_dir = {}
        self._analysis_failures_by_input_dir = {}
        self._active_stoichiometry_key = None
        self._run_summary = None
        self._viewer_scene_payload_by_path = {}
        self.open_output_button.setEnabled(False)
        self._clear_selected_result_view(
            "Select a stoichiometry row to inspect it. Completed runs will populate the summary, candidate scores, plots, and representative viewer."
        )
        self._refresh_solvent_shell_controls()

    def _clear_selected_result_view(self, message: str) -> None:
        self.result_summary_box.setPlainText(message)
        self.candidate_table.setRowCount(0)
        self.display_mode_combo.blockSignals(True)
        self.display_mode_combo.clear()
        self.display_mode_combo.setEnabled(False)
        self.display_mode_combo.blockSignals(False)
        self.plot_widget.set_result(None)
        self.viewer_widget.draw_placeholder()
        self._refresh_solvent_shell_controls()

    def _populate_stoichiometry_table(
        self,
        inspection: RepresentativeFinderInputInspection | None,
    ) -> None:
        self.stoichiometry_table.blockSignals(True)
        self.stoichiometry_table.setRowCount(0)
        self._stoichiometry_row_by_input_dir = {}
        if inspection is not None:
            for row, stoich in enumerate(inspection.stoichiometry_folders):
                self.stoichiometry_table.insertRow(row)
                key = str(stoich.input_dir)
                self._stoichiometry_row_by_input_dir[key] = row
                self._set_stoichiometry_table_item(
                    row, 0, stoich.structure_label, key
                )
                self._set_stoichiometry_table_item(
                    row,
                    1,
                    str(stoich.candidate_count),
                    key,
                )
                self._set_stoichiometry_table_item(
                    row,
                    2,
                    (
                        ", ".join(stoich.motif_labels)
                        if stoich.motif_labels
                        else "none"
                    ),
                    key,
                )
                self._set_stoichiometry_table_item(row, 3, "Pending", key)
                self._set_stoichiometry_table_item(row, 4, "", key)
                self._set_stoichiometry_table_item(row, 5, "", key)
                self._set_stoichiometry_table_item(row, 6, "", key)
                self._set_stoichiometry_open_button(row, key, None)
        self.stoichiometry_table.blockSignals(False)
        self.stoichiometry_table.resizeColumnsToContents()
        if self.stoichiometry_table.rowCount() > 0:
            self.stoichiometry_table.selectRow(0)
        self._update_selected_stoichiometry_view()

    def _set_stoichiometry_table_item(
        self,
        row: int,
        column: int,
        text: str,
        key: str,
    ) -> None:
        item = QTableWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, key)
        self.stoichiometry_table.setItem(row, column, item)

    def _set_stoichiometry_open_button(
        self,
        row: int,
        key: str,
        representative_path: Path | None,
    ) -> None:
        button = self.stoichiometry_table.cellWidget(row, 7)
        if not isinstance(button, QPushButton):
            button = QPushButton("Open in Finder", self.stoichiometry_table)
            button.clicked.connect(
                lambda _checked=False, row_key=key: (
                    self._select_stoichiometry_row_by_key(row_key),
                    self._open_stoichiometry_representative_path(row_key),
                )
            )
            self.stoichiometry_table.setCellWidget(row, 7, button)
        resolved_path = (
            None
            if representative_path is None
            else representative_path.expanduser().resolve()
        )
        if resolved_path is None or not resolved_path.is_file():
            button.setEnabled(False)
            button.setToolTip(
                "Representative output file is not available for this stoichiometry yet."
            )
            return
        button.setEnabled(True)
        button.setToolTip(str(resolved_path))

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        previous_name = selected_name or self._selected_preset_name()
        self._presets = load_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Select preset...", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_preset_names(self._presets),
            start=1,
        ):
            preset = self._presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.preset_combo.setCurrentIndex(selected_index)
        self.preset_combo.blockSignals(False)

    def _selected_preset_name(self) -> str | None:
        payload = self.preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def load_preset(self, preset_name: str) -> None:
        preset = self._presets.get(preset_name)
        if preset is None:
            raise KeyError(f"Unknown preset: {preset_name}")
        self._set_bond_pair_rows(preset.bond_pairs)
        self._set_angle_triplet_rows(preset.angle_triplets)
        self._select_preset_name(preset.name)

    def _load_selected_preset(self) -> None:
        preset_name = self._selected_preset_name()
        if not preset_name:
            QMessageBox.information(
                self,
                "Representative Presets",
                "Choose a preset first.",
            )
            return
        try:
            self.load_preset(preset_name)
        except KeyError:
            QMessageBox.warning(
                self,
                "Representative Presets",
                f"The selected preset is no longer available: {preset_name}",
            )
            return
        self._append_console(f"Loaded representative preset: {preset_name}")

    def _save_current_preset(self) -> None:
        try:
            bond_pairs = self._read_bond_pairs()
            angle_triplets = self._read_angle_triplets()
        except ValueError as exc:
            QMessageBox.warning(self, "Representative Presets", str(exc))
            return
        suggested_name = self._selected_preset_name() or ""
        name, accepted = QInputDialog.getText(
            self,
            "Save Representative Preset",
            "Preset name",
            text=suggested_name,
        )
        if not accepted:
            return
        normalized_name = name.strip()
        if not normalized_name:
            QMessageBox.warning(
                self,
                "Representative Presets",
                "Preset names cannot be empty.",
            )
            return
        preset = BondAnalysisPreset(
            name=normalized_name,
            bond_pairs=bond_pairs,
            angle_triplets=angle_triplets,
            builtin=False,
        )
        save_custom_preset(preset)
        self._reload_presets(selected_name=normalized_name)
        self._append_console(f"Saved representative preset: {normalized_name}")

    def _select_preset_name(self, preset_name: str) -> None:
        for index in range(self.preset_combo.count()):
            if self.preset_combo.itemData(index) == preset_name:
                self.preset_combo.setCurrentIndex(index)
                return

    def _set_bond_pair_rows(
        self,
        definitions: tuple[BondPairDefinition, ...],
    ) -> None:
        self.bond_pair_table.blockSignals(True)
        self.bond_pair_table.setRowCount(0)
        if not definitions:
            self._add_empty_bond_pair_row(blocked=True)
        else:
            for definition in definitions:
                row = self.bond_pair_table.rowCount()
                self.bond_pair_table.insertRow(row)
                self.bond_pair_table.setItem(
                    row, 0, QTableWidgetItem(definition.atom1)
                )
                self.bond_pair_table.setItem(
                    row, 1, QTableWidgetItem(definition.atom2)
                )
                self.bond_pair_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(f"{definition.cutoff_angstrom:g}"),
                )
        self.bond_pair_table.blockSignals(False)

    def _set_angle_triplet_rows(
        self,
        definitions: tuple[AngleTripletDefinition, ...],
    ) -> None:
        self.angle_triplet_table.blockSignals(True)
        self.angle_triplet_table.setRowCount(0)
        if not definitions:
            self._add_empty_angle_triplet_row(blocked=True)
        else:
            for definition in definitions:
                row = self.angle_triplet_table.rowCount()
                self.angle_triplet_table.insertRow(row)
                self.angle_triplet_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(definition.vertex),
                )
                self.angle_triplet_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(definition.arm1),
                )
                self.angle_triplet_table.setItem(
                    row,
                    2,
                    QTableWidgetItem(definition.arm2),
                )
                self.angle_triplet_table.setItem(
                    row,
                    3,
                    QTableWidgetItem(f"{definition.cutoff1_angstrom:g}"),
                )
                self.angle_triplet_table.setItem(
                    row,
                    4,
                    QTableWidgetItem(f"{definition.cutoff2_angstrom:g}"),
                )
        self.angle_triplet_table.blockSignals(False)

    def _add_empty_bond_pair_row(self, *, blocked: bool = False) -> None:
        previous = self.bond_pair_table.blockSignals(blocked)
        row = self.bond_pair_table.rowCount()
        self.bond_pair_table.insertRow(row)
        for column in range(self.bond_pair_table.columnCount()):
            self.bond_pair_table.setItem(row, column, QTableWidgetItem(""))
        self.bond_pair_table.blockSignals(previous)

    def _add_empty_angle_triplet_row(self, *, blocked: bool = False) -> None:
        previous = self.angle_triplet_table.blockSignals(blocked)
        row = self.angle_triplet_table.rowCount()
        self.angle_triplet_table.insertRow(row)
        for column in range(self.angle_triplet_table.columnCount()):
            self.angle_triplet_table.setItem(row, column, QTableWidgetItem(""))
        self.angle_triplet_table.blockSignals(previous)

    def _add_bond_pair_row(self) -> None:
        self._add_empty_bond_pair_row(blocked=True)

    def _remove_selected_bond_pair_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.bond_pair_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.bond_pair_table.removeRow(row)
        if self.bond_pair_table.rowCount() == 0:
            self._add_empty_bond_pair_row(blocked=True)

    def _add_angle_triplet_row(self) -> None:
        self._add_empty_angle_triplet_row(blocked=True)

    def _remove_selected_angle_triplet_rows(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.angle_triplet_table.selectedIndexes()
            },
            reverse=True,
        )
        for row in rows:
            self.angle_triplet_table.removeRow(row)
        if self.angle_triplet_table.rowCount() == 0:
            self._add_empty_angle_triplet_row(blocked=True)

    def _read_bond_pairs(self) -> tuple[BondPairDefinition, ...]:
        definitions: list[BondPairDefinition] = []
        for row in range(self.bond_pair_table.rowCount()):
            atom1 = self._table_text(self.bond_pair_table, row, 0)
            atom2 = self._table_text(self.bond_pair_table, row, 1)
            cutoff_text = self._table_text(self.bond_pair_table, row, 2)
            if not atom1 and not atom2 and not cutoff_text:
                continue
            if not atom1 or not atom2 or not cutoff_text:
                raise ValueError(
                    "Every non-empty bond-pair row must include Atom 1, Atom 2, and a cutoff."
                )
            definitions.append(
                BondPairDefinition(atom1, atom2, float(cutoff_text))
            )
        return tuple(definitions)

    def _read_angle_triplets(self) -> tuple[AngleTripletDefinition, ...]:
        definitions: list[AngleTripletDefinition] = []
        for row in range(self.angle_triplet_table.rowCount()):
            vertex = self._table_text(self.angle_triplet_table, row, 0)
            arm1 = self._table_text(self.angle_triplet_table, row, 1)
            arm2 = self._table_text(self.angle_triplet_table, row, 2)
            cutoff1_text = self._table_text(self.angle_triplet_table, row, 3)
            cutoff2_text = self._table_text(self.angle_triplet_table, row, 4)
            if (
                not vertex
                and not arm1
                and not arm2
                and not cutoff1_text
                and not cutoff2_text
            ):
                continue
            if not all((vertex, arm1, arm2, cutoff1_text, cutoff2_text)):
                raise ValueError(
                    "Every non-empty angle-triplet row must include all five values."
                )
            definitions.append(
                AngleTripletDefinition(
                    vertex=vertex,
                    arm1=arm1,
                    arm2=arm2,
                    cutoff1_angstrom=float(cutoff1_text),
                    cutoff2_angstrom=float(cutoff2_text),
                )
            )
        return tuple(definitions)

    def _current_settings(self) -> RepresentativeFinderSettings:
        return RepresentativeFinderSettings(
            selection_algorithm=str(
                self.algorithm_combo.currentData()
                or "target_distribution_quantile_distance"
            ),
            bond_weight=float(self.bond_weight_spin.value()),
            angle_weight=float(self.angle_weight_spin.value()),
            solvent_weight=float(self.solvent_weight_spin.value()),
            generate_predicted_optimized_representative=bool(
                self.generate_predicted_checkbox.isChecked()
            ),
            bond_pairs=self._read_bond_pairs(),
            angle_triplets=self._read_angle_triplets(),
        )

    def _stoichiometry_inspection_for_key(
        self,
        key: str | None,
    ) -> RepresentativeFinderFolderInspection | None:
        inspection = self._input_inspection
        if inspection is None or key is None:
            return None
        for stoich in inspection.stoichiometry_folders:
            if str(stoich.input_dir) == key:
                return stoich
        return None

    def _selected_stoichiometry_key(self) -> str | None:
        if not hasattr(self, "stoichiometry_table"):
            return None
        selected_items = self.stoichiometry_table.selectedItems()
        if selected_items:
            return str(
                selected_items[0].data(Qt.ItemDataRole.UserRole) or ""
            ).strip()
        if self.stoichiometry_table.rowCount() <= 0:
            return None
        item = self.stoichiometry_table.item(0, 0)
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole) or "").strip()

    def _selected_stoichiometry_inspection(
        self,
    ) -> RepresentativeFinderFolderInspection | None:
        return self._stoichiometry_inspection_for_key(
            self._selected_stoichiometry_key()
        )

    def _selected_result(self) -> RepresentativeFinderResult | None:
        key = self._selected_stoichiometry_key()
        if key is None:
            return None
        return self._analysis_results_by_input_dir.get(key)

    def _overwrite_existing_representatives(self) -> bool:
        checkbox = getattr(self, "overwrite_existing_checkbox", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _stoichiometry_has_saved_project_representative(
        self,
        stoich: RepresentativeFinderFolderInspection,
    ) -> bool:
        key = str(stoich.input_dir)
        shared_path = self._project_representative_path_for_key(key)
        return (
            self._project_representative_entry_for_key(key) is not None
            and shared_path is not None
            and shared_path.is_file()
        )

    def _selected_stoichiometries_for_current_mode(
        self,
    ) -> tuple[RepresentativeFinderFolderInspection, ...]:
        inspection = self._input_inspection
        if inspection is None or not inspection.stoichiometry_folders:
            raise ValueError(
                "Choose a valid stoichiometry folder or stoichiometry parent folder before running the analysis."
            )
        if self._current_analysis_mode() == "all":
            return tuple(inspection.stoichiometry_folders)
        selected_stoich = (
            self._selected_stoichiometry_inspection()
            or inspection.stoichiometry_folders[0]
        )
        return (selected_stoich,)

    def _analysis_targets_from_inputs(
        self,
        *,
        output_root: Path,
        settings: RepresentativeFinderSettings,
    ) -> tuple[RepresentativeFinderAnalysisTarget, ...]:
        inspection = self._input_inspection
        selected_stoichiometries = (
            self._selected_stoichiometries_for_current_mode()
        )
        if not self._overwrite_existing_representatives():
            selected_stoichiometries = tuple(
                stoich
                for stoich in selected_stoichiometries
                if not self._stoichiometry_has_saved_project_representative(
                    stoich
                )
            )

        use_direct_output_dir = (
            inspection.input_is_stoichiometry_folder
            and inspection.stoichiometry_count == 1
        )
        targets: list[RepresentativeFinderAnalysisTarget] = []
        for stoich in selected_stoichiometries:
            target_output_dir = (
                output_root
                if use_direct_output_dir
                else suggest_representativefinder_target_output_dir(
                    output_root,
                    stoich.structure_label,
                )
            )
            targets.append(
                RepresentativeFinderAnalysisTarget(
                    inspection=stoich,
                    output_dir=target_output_dir,
                    estimated_total_work=estimate_representativefinder_total_work(
                        stoich.candidate_count,
                        solvent_phase_enabled=settings.solvent_weight > 0.0,
                        predicted_phase_enabled=bool(
                            settings.generate_predicted_optimized_representative
                        ),
                        predicted_solvent_phase_enabled=bool(
                            settings.generate_predicted_optimized_representative
                            and self._initial_project_dir is not None
                        ),
                    ),
                )
            )
        return tuple(targets)

    def _reset_stoichiometry_run_state(self, target_keys: set[str]) -> None:
        self._analysis_results_by_input_dir = {}
        self._analysis_failures_by_input_dir = {}
        self._active_stoichiometry_key = None
        self._run_summary = None
        self._viewer_scene_payload_by_path = {}
        for key in self._stoichiometry_row_by_input_dir:
            representative_entry = self._project_representative_entry_for_key(
                key
            )
            shared_path = self._project_representative_path_for_key(key)
            if (
                key not in target_keys
                and representative_entry is not None
                and shared_path is not None
                and shared_path.is_file()
            ):
                output_text, output_tooltip = (
                    self._project_representative_output_display(
                        representative_entry,
                        shared_path,
                    )
                )
                self._update_stoichiometry_row(
                    key,
                    status="Complete",
                    representative=(
                        str(
                            getattr(
                                representative_entry,
                                "source_file_name",
                                "",
                            )
                        ).strip()
                        or shared_path.name
                    ),
                    score=_format_score(
                        getattr(representative_entry, "score_total", None)
                    ),
                    output_text=output_text,
                    output_tooltip=output_tooltip,
                    representative_path=shared_path,
                )
                continue
            self._update_stoichiometry_row(
                key,
                status=("Queued" if key in target_keys else "Not selected"),
                representative="",
                score="",
                output_text="",
                output_tooltip="",
                representative_path=False,
            )

    def _run_analysis(self) -> None:
        if self._analysis_thread is not None:
            QMessageBox.information(
                self,
                "Representative selection running",
                "A representative-selection run is already in progress.",
            )
            return
        input_dir_text = self.input_dir_edit.text().strip()
        output_dir_text = self.output_dir_edit.text().strip()
        if not input_dir_text:
            QMessageBox.warning(
                self,
                "Representative Structure Finder",
                "Choose a stoichiometry folder or stoichiometry parent folder before running the analysis.",
            )
            return
        if not output_dir_text:
            QMessageBox.warning(
                self,
                "Representative Structure Finder",
                "Choose an output folder before running the analysis.",
            )
            return
        try:
            settings = self._current_settings()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Representative Structure Finder",
                str(exc),
            )
            return

        output_root = Path(output_dir_text).expanduser().resolve()
        try:
            self._refresh_project_representative_path_cache()
            requested_stoichiometries = (
                self._selected_stoichiometries_for_current_mode()
            )
            targets = self._analysis_targets_from_inputs(
                output_root=output_root,
                settings=settings,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Representative Structure Finder",
                str(exc),
            )
            return
        skipped_existing_count = (
            len(requested_stoichiometries) - len(targets)
            if not self._overwrite_existing_representatives()
            else 0
        )
        if not targets:
            self._reset_stoichiometry_run_state(set())
            message = (
                "All selected stoichiometries already have saved "
                "representative structures. Enable overwrite to recalculate "
                "them."
            )
            self.run_status_label.setText(
                f"Representative selection: {message}"
            )
            self.statusBar().showMessage(message)
            self._append_console(message)
            self.open_output_button.setEnabled(bool(output_dir_text))
            return

        self._reset_stoichiometry_run_state(
            {str(target.inspection.input_dir) for target in targets}
        )
        self.run_button.setEnabled(False)
        self.open_output_button.setEnabled(bool(output_dir_text))
        self.run_status_label.setText(
            "Representative selection: starting background task..."
        )
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.result_summary_box.setPlainText(
            "Representative selection is running. Click a stoichiometry row to follow its status. Completed rows will populate the score table, plots, and representative viewer."
        )
        self.candidate_table.setRowCount(0)
        self.display_mode_combo.blockSignals(True)
        self.display_mode_combo.clear()
        self.display_mode_combo.setEnabled(False)
        self.display_mode_combo.blockSignals(False)
        self.plot_widget.set_result(None)
        self.viewer_widget.draw_placeholder()
        self._append_console("Starting representative-structure analysis.")
        if skipped_existing_count > 0:
            self._append_console(
                "Skipping "
                f"{skipped_existing_count} stoichiometr"
                f"{'y' if skipped_existing_count == 1 else 'ies'} "
                "with saved project representative structures. Enable "
                "overwrite to recalculate them."
            )

        config = RepresentativeFinderJobConfig(
            analysis_mode=self._current_analysis_mode(),
            targets=targets,
            settings=settings,
            project_dir=self._initial_project_dir,
        )
        self._analysis_thread = QThread(self)
        self._analysis_worker = RepresentativeFinderWorker(config)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.log.connect(self._append_console)
        self._analysis_worker.progress.connect(self._update_progress)
        self._analysis_worker.target_started.connect(self._on_target_started)
        self._analysis_worker.result_ready.connect(
            self._on_target_result_ready
        )
        self._analysis_worker.target_failed.connect(self._on_target_failed)
        self._analysis_worker.finished.connect(self._finish_analysis_run)
        self._analysis_worker.failed.connect(self._fail_analysis)
        self._analysis_worker.canceled.connect(self._cancel_analysis_complete)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_worker.canceled.connect(self._analysis_thread.quit)
        self._analysis_thread.finished.connect(self._cleanup_thread)
        self._analysis_thread.finished.connect(
            self._analysis_thread.deleteLater
        )
        self._analysis_worker.finished.connect(
            self._analysis_worker.deleteLater
        )
        self._analysis_worker.failed.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.canceled.connect(
            self._analysis_worker.deleteLater
        )
        self._analysis_thread.start()

    def _update_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.run_progress_bar.setRange(0, max(total, 1))
        self.run_progress_bar.setValue(min(max(processed, 0), max(total, 1)))
        self.run_status_label.setText(f"Representative selection: {message}")
        self.statusBar().showMessage(message)

    def _on_target_started(self, key: str) -> None:
        self._active_stoichiometry_key = key
        self._update_stoichiometry_row(
            key,
            status="Running",
        )
        if self._selected_stoichiometry_key() == key:
            self._update_selected_stoichiometry_view()

    def _on_target_result_ready(
        self, result: RepresentativeFinderResult
    ) -> None:
        key = str(result.input_dir)
        had_results = bool(self._analysis_results_by_input_dir)
        shared_project_path: Path | None = None
        if self._initial_project_dir is not None:
            try:
                shared_project_path = (
                    persist_representativefinder_result_to_project(
                        self._initial_project_dir,
                        result,
                    )
                )
            except Exception as exc:
                self._append_console(
                    "Unable to publish representative structure into the project "
                    f"folder: {exc}"
                )
            else:
                self._shared_project_representative_path_by_input_dir[key] = (
                    shared_project_path
                )
                self.project_results_changed.emit(
                    str(Path(self._initial_project_dir).resolve())
                )
        self._analysis_results_by_input_dir[key] = result
        self._analysis_failures_by_input_dir.pop(key, None)
        self._update_stoichiometry_row(
            key,
            status="Complete",
            representative=result.selected_candidate.file_name,
            score=_format_score(result.selected_candidate.score_total),
            output_text=result.output_dir.name,
            output_tooltip=str(result.output_dir),
            representative_path=(
                shared_project_path or result.representative_output_path
            ),
        )
        if not had_results or self._selected_stoichiometry_key() == key:
            self._select_stoichiometry_row_by_key(key)

    def _on_target_failed(
        self, failure: RepresentativeFinderTargetFailure
    ) -> None:
        key = str(failure.input_dir)
        self._analysis_failures_by_input_dir[key] = failure.message
        self._update_stoichiometry_row(
            key,
            status="Failed",
            representative="",
            score="",
            output_text="",
            output_tooltip="",
            representative_path=False,
        )
        self._append_console(
            f"[{failure.structure_label}] Representative selection failed: {failure.message}"
        )
        if self._selected_stoichiometry_key() == key:
            self._update_selected_stoichiometry_view()

    def _finish_analysis_run(
        self, summary: RepresentativeFinderRunSummary
    ) -> None:
        self._run_summary = summary
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(
            bool(self._analysis_results_by_input_dir)
            or bool(self.output_dir_edit.text().strip())
        )
        if summary.failures:
            self.run_status_label.setText(
                "Representative selection: complete with failures"
            )
        else:
            self.run_status_label.setText("Representative selection: complete")
        self.run_progress_bar.setValue(self.run_progress_bar.maximum())
        self.statusBar().showMessage("Representative selection complete")
        selected_key = self._selected_stoichiometry_key()
        if summary.results and (
            selected_key not in self._analysis_results_by_input_dir
        ):
            self._select_stoichiometry_row_by_key(
                str(summary.results[0].input_dir)
            )
        elif summary.failures:
            failed_keys = {
                str(failure.input_dir) for failure in summary.failures
            }
            if selected_key not in failed_keys and not summary.results:
                self._select_stoichiometry_row_by_key(
                    str(summary.failures[0].input_dir)
                )
            else:
                self._update_selected_stoichiometry_view()
        else:
            self._update_selected_stoichiometry_view()
        self._append_console(
            "Representative selection complete: "
            f"{len(summary.results)} stoichiometry run(s) completed, "
            f"{len(summary.failures)} failed."
        )

    def _fail_analysis(self, message: str) -> None:
        if self._closing_after_analysis_cancel:
            self.statusBar().showMessage("Representative selection stopped")
            self._append_console(
                f"Representative selection stopped while closing: {message}"
            )
            return
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(
            bool(self._analysis_results_by_input_dir)
            or bool(self.output_dir_edit.text().strip())
        )
        self.run_status_label.setText("Representative selection: failed")
        self._update_selected_stoichiometry_view()
        self.statusBar().showMessage("Representative selection failed")
        QMessageBox.warning(
            self,
            "Representative selection failed",
            message,
        )

    def _cancel_analysis_complete(self) -> None:
        if self._closing_after_analysis_cancel:
            self.statusBar().showMessage("Representative selection canceled")
            return
        self.run_button.setEnabled(True)
        self.open_output_button.setEnabled(
            bool(self._analysis_results_by_input_dir)
            or bool(self.output_dir_edit.text().strip())
        )
        self.run_status_label.setText("Representative selection: canceled")
        self._update_selected_stoichiometry_view()
        self.statusBar().showMessage("Representative selection canceled")
        self._append_console(
            "Representative selection canceled. Any completed stoichiometry rows remain available for review."
        )

    def _cleanup_thread(self) -> None:
        self._analysis_worker = None
        self._analysis_thread = None
        if self._closing_after_analysis_cancel:
            self._closing_after_analysis_cancel = False
            QTimer.singleShot(0, self.close)

    def _update_stoichiometry_row(
        self,
        key: str,
        *,
        status: str | None = None,
        representative: str | None = None,
        score: str | None = None,
        output_text: str | None = None,
        output_tooltip: str | None = None,
        representative_path: Path | None | bool = None,
    ) -> None:
        row = self._stoichiometry_row_by_input_dir.get(key)
        if row is None:
            return

        def set_column(
            column: int,
            value: str | None,
            *,
            tooltip: str | None = None,
        ) -> None:
            if value is None:
                return
            item = self.stoichiometry_table.item(row, column)
            if item is None:
                self._set_stoichiometry_table_item(row, column, value, key)
                item = self.stoichiometry_table.item(row, column)
            else:
                item.setText(value)
            if item is not None:
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(tooltip if tooltip is not None else value)

        set_column(3, status)
        set_column(4, representative)
        set_column(5, score)
        set_column(6, output_text, tooltip=output_tooltip)
        if representative_path is not None:
            self._set_stoichiometry_open_button(
                row,
                key,
                (
                    representative_path
                    if isinstance(representative_path, Path)
                    else None
                ),
            )
        self.stoichiometry_table.resizeColumnsToContents()

    def _select_stoichiometry_row_by_key(self, key: str) -> None:
        row = self._stoichiometry_row_by_input_dir.get(key)
        if row is None:
            return
        self.stoichiometry_table.selectRow(row)
        self._update_selected_stoichiometry_view()

    def _populate_candidate_table(
        self,
        result: RepresentativeFinderResult,
    ) -> None:
        self.candidate_table.setRowCount(0)
        for row, candidate in enumerate(result.candidates):
            self.candidate_table.insertRow(row)
            self._set_candidate_table_item(
                row, 0, candidate.file_name, candidate
            )
            self._set_candidate_table_item(
                row, 1, candidate.relative_label, candidate
            )
            self._set_candidate_table_item(
                row, 2, _format_score(candidate.score_total), candidate
            )
            self._set_candidate_table_item(
                row, 3, _format_score(candidate.score_bond), candidate
            )
            self._set_candidate_table_item(
                row, 4, _format_score(candidate.score_angle), candidate
            )
            self._set_candidate_table_item(
                row, 5, _format_score(candidate.score_solvent), candidate
            )
            self._set_candidate_table_item(
                row, 6, str(candidate.atom_count), candidate
            )
            self._set_candidate_table_item(
                row, 7, str(candidate.solvent_atom_count), candidate
            )
        self.candidate_table.resizeColumnsToContents()

    def _set_candidate_table_item(
        self,
        row: int,
        column: int,
        text: str,
        candidate: RepresentativeFinderCandidate,
    ) -> None:
        item = QTableWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, str(candidate.file_path))
        self.candidate_table.setItem(row, column, item)

    def _select_candidate_row(self, row: int) -> None:
        if row < 0 or row >= self.candidate_table.rowCount():
            return
        self.candidate_table.selectRow(row)
        self._update_selected_candidate_view()

    def _selected_candidate(self) -> RepresentativeFinderCandidate | None:
        result = self._selected_result()
        if result is None:
            return None
        selected_items = self.candidate_table.selectedItems()
        if not selected_items:
            return result.selected_candidate
        target_path = str(
            selected_items[0].data(Qt.ItemDataRole.UserRole) or ""
        ).strip()
        for candidate in result.candidates:
            if str(candidate.file_path) == target_path:
                return candidate
        return result.selected_candidate

    def _update_selected_stoichiometry_view(self) -> None:
        key = self._selected_stoichiometry_key()
        self._active_stoichiometry_key = key
        if key is None:
            self._clear_selected_result_view(
                "Select a stoichiometry row to inspect it."
            )
            return
        result = self._analysis_results_by_input_dir.get(key)
        if result is not None:
            self._populate_candidate_table(result)
            self._refresh_display_mode_options(result)
            self.plot_widget.set_result(result)
            self._select_candidate_row(0)
            return
        representative_entry = self._project_representative_entry_for_key(key)
        if representative_entry is not None:
            shared_path = self._project_representative_path_for_key(key)
            if shared_path is not None:
                self._clear_selected_result_view(
                    self._project_representative_summary_text(
                        representative_entry,
                        shared_path,
                    )
                )
                return

        inspection = self._stoichiometry_inspection_for_key(key)
        if inspection is None:
            self._clear_selected_result_view(
                "Select a stoichiometry row to inspect it."
            )
            return
        row = self._stoichiometry_row_by_input_dir.get(key, -1)
        status_item = (
            self.stoichiometry_table.item(row, 3) if row >= 0 else None
        )
        status_text = (
            status_item.text().strip()
            if status_item is not None
            else "Pending"
        )
        failure_message = self._analysis_failures_by_input_dir.get(key)
        lines = [
            f"Stoichiometry: {inspection.structure_label}",
            f"Input folder: {inspection.input_dir}",
            f"Status: {status_text}",
            f"Candidate files: {inspection.candidate_count}",
            f"Direct files: {inspection.direct_file_count}",
            "Motif folders: "
            + (
                ", ".join(inspection.motif_labels)
                if inspection.motif_labels
                else "none"
            ),
        ]
        if failure_message:
            lines.extend(["", "Failure", failure_message])
        else:
            lines.extend(
                [
                    "",
                    "Run representative selection to populate this stoichiometry with a representative structure and candidate ranking.",
                ]
            )
        self._clear_selected_result_view("\n".join(lines))

    def _project_representative_summary_text(
        self,
        representative_entry: object,
        shared_path: Path,
    ) -> str:
        element_counts = (
            getattr(representative_entry, "element_counts", {}) or {}
        )
        element_text = (
            ", ".join(
                f"{element} x{count}"
                for element, count in sorted(dict(element_counts).items())
            )
            if element_counts
            else "none"
        )
        output_text, output_tooltip = (
            self._project_representative_output_display(
                representative_entry,
                shared_path,
            )
        )
        cached_results_path = str(
            getattr(representative_entry, "cached_results_path", "") or ""
        ).strip()
        lines = [
            f"Stoichiometry: {getattr(representative_entry, 'structure', '')}",
            "Status: Complete",
            "Representative: "
            + (
                str(
                    getattr(representative_entry, "source_file_name", "")
                ).strip()
                or shared_path.name
            ),
            "Score: "
            + _format_score(
                getattr(representative_entry, "score_total", None)
            ),
            f"Atoms: {int(getattr(representative_entry, 'atom_count', 0))}",
            f"Elements: {element_text}",
            f"Output: {output_text}",
            f"Output folder: {output_tooltip}",
            f"Project representative file: {shared_path}",
        ]
        analysis_source = str(
            getattr(representative_entry, "analysis_source", "") or ""
        ).strip()
        if analysis_source:
            lines.append(f"Analysis source: {analysis_source}")
        source_mode = str(
            getattr(representative_entry, "source_solvent_mode", "") or ""
        ).strip()
        if source_mode:
            lines.append(f"Source solvent mode: {source_mode}")
        if cached_results_path:
            lines.append(f"Cached analysis metadata: {cached_results_path}")
        return "\n".join(lines)

    def _update_selected_candidate_view(self) -> None:
        result = self._selected_result()
        payload = self._active_display_payload()
        if result is None or payload is None:
            return
        candidate, display_path, display_mode_label = payload
        self.result_summary_box.setPlainText(
            self._candidate_summary_text(
                result,
                candidate,
                display_mode_label=display_mode_label,
                display_path=display_path,
            )
        )
        self.plot_widget.set_result(result, candidate=candidate)
        try:
            structure, mesh_geometry = self._load_viewer_scene(
                candidate,
                structure_path=display_path,
                display_label=(
                    candidate.relative_label
                    if display_mode_label == "Selected Candidate"
                    else display_mode_label
                ),
            )
            self.viewer_widget.set_structure(
                structure,
                mesh_geometry=mesh_geometry,
                scene_key=str(display_path),
            )
        except Exception:
            self.viewer_widget.draw_placeholder()
            self._refresh_solvent_shell_controls()
            return
        self._refresh_solvent_shell_controls()

    def _load_viewer_scene(
        self,
        candidate: RepresentativeFinderCandidate,
        *,
        structure_path: Path | None = None,
        display_label: str | None = None,
    ) -> tuple[ElectronDensityStructure, ElectronDensityMeshGeometry | None]:
        resolved_path = (
            candidate.file_path.expanduser().resolve()
            if structure_path is None
            else structure_path.expanduser().resolve()
        )
        cache_key = str(resolved_path)
        cached = self._viewer_scene_payload_by_path.get(cache_key)
        if cached is not None:
            structure, mesh_geometry = cached
            resolved_label = str(
                display_label or candidate.relative_label
            ).strip()
            if resolved_label and resolved_label != structure.display_label:
                structure = replace(structure, display_label=resolved_label)
            return structure, mesh_geometry

        structure = load_electron_density_structure(resolved_path)
        resolved_label = str(display_label or candidate.relative_label).strip()
        if resolved_label and resolved_label != structure.display_label:
            structure = replace(structure, display_label=resolved_label)
        mesh_geometry = build_electron_density_mesh(
            structure,
            legacy_born_average_default_mesh_settings(structure),
        )
        payload = (structure, mesh_geometry)
        self._viewer_scene_payload_by_path[cache_key] = payload
        return payload

    def _candidate_summary_text(
        self,
        result: RepresentativeFinderResult,
        candidate: RepresentativeFinderCandidate,
        *,
        display_mode_label: str,
        display_path: Path,
    ) -> str:
        shared_project_path = self._project_representative_path_for_key(
            str(result.input_dir)
        )
        representative_entry = self._project_representative_entry_for_key(
            str(result.input_dir)
        )
        lines = [
            f"Stoichiometry: {result.structure_label}",
            "Status: Complete",
            f"Displayed structure: {display_mode_label}",
            "Representative: "
            + (
                shared_project_path.name
                if shared_project_path is not None
                else result.representative_output_path.name
            ),
            f"Candidate: {candidate.file_name}",
            f"Source label: {candidate.relative_label}",
            f"Displayed file: {display_path}",
            (
                "Scores: "
                f"total={_format_score(candidate.score_total)}, "
                f"bond={_format_score(candidate.score_bond)}, "
                f"angle={_format_score(candidate.score_angle)}, "
                f"solvent={_format_score(candidate.score_solvent)}"
            ),
            f"Atoms: {candidate.atom_count}",
            "Elements: "
            + ", ".join(
                f"{element} x{count}"
                for element, count in sorted(candidate.element_counts.items())
            ),
            (
                "Solvent shell: "
                f"total={candidate.solvent_atom_count}, "
                f"direct={candidate.direct_solvent_atom_count}, "
                f"outer={candidate.outer_solvent_atom_count}"
            ),
            (
                "Mean direct solvent coordination: "
                f"{candidate.mean_direct_solvent_coordination:.6g}"
            ),
            "",
            "Observed representative output: "
            + str(shared_project_path or result.representative_output_path),
        ]
        if shared_project_path is not None:
            lines.append(f"Project representative file: {shared_project_path}")
        source_mode = (
            ""
            if representative_entry is None
            else str(
                getattr(representative_entry, "source_solvent_mode", "") or ""
            ).strip()
        )
        if source_mode:
            lines.append(f"Source solvent mode: {source_mode}")
        if result.predicted_output_path is not None:
            lines.append(
                "Predicted optimized representative output: "
                + str(result.predicted_output_path)
            )
        if result.solvent_completed_predicted_output_path is not None:
            lines.append(
                "Solvent-completed predicted output: "
                + str(result.solvent_completed_predicted_output_path)
            )
        if result.predicted_generation_notes:
            lines.extend(["", "Predicted-output notes"])
            lines.extend(
                f"  {note}" for note in result.predicted_generation_notes
            )
        if candidate.descriptor_notes:
            lines.extend(["", "Notes"])
            lines.extend(f"  {note}" for note in candidate.descriptor_notes)
        return "\n".join(lines)

    def _show_output_folder(self) -> None:
        result = self._selected_result()
        if result is not None:
            output_path: Path | None = result.output_dir
        else:
            output_text = self.output_dir_edit.text().strip()
            output_path = (
                Path(output_text).expanduser().resolve()
                if output_text
                else None
            )
        if output_path is None:
            return
        QMessageBox.information(
            self,
            "Representative Output Folder",
            str(output_path),
        )

    def _open_stoichiometry_representative_path(self, key: str) -> None:
        shared_project_path = self._project_representative_path_for_key(key)
        if shared_project_path is not None and shared_project_path.is_file():
            representative_path = shared_project_path
        else:
            result = self._analysis_results_by_input_dir.get(key)
            representative_path = (
                None
                if result is None
                else result.representative_output_path.expanduser().resolve()
            )
        if representative_path is None or not representative_path.is_file():
            QMessageBox.information(
                self,
                "Representative Structure Path",
                "This stoichiometry does not have a saved representative structure file yet.",
            )
            return
        try:
            self._reveal_path_in_file_manager(representative_path)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Representative Structure Path",
                f"Could not open the representative structure path:\n{exc}",
            )
            return
        self.statusBar().showMessage(
            f"Opened representative structure path for {representative_path.name}"
        )
        self._append_console(
            f"Opened representative structure path in Finder: {representative_path}"
        )

    @staticmethod
    def _reveal_path_in_file_manager(path: Path) -> None:
        resolved_path = path.expanduser().resolve()
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(resolved_path)])
            return
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", f"/select,{resolved_path}"])
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved_path.parent)))

    def _append_console(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        existing = self.console_box.toPlainText().strip()
        if existing:
            self.console_box.setPlainText(existing + "\n" + text)
        else:
            self.console_box.setPlainText(text)
        cursor = self.console_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.console_box.setTextCursor(cursor)

    @staticmethod
    def _table_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        if item is None:
            return ""
        return item.text().strip()

    @staticmethod
    def _new_float_spin(
        *,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def launch_representativefinder_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
) -> RepresentativeStructureFinderMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)
    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=initial_project_dir,
        initial_input_path=initial_input_path,
    )
    window.show()
    window.raise_()
    return window


__all__ = [
    "RepresentativeStructureFinderMainWindow",
    "launch_representativefinder_ui",
]
