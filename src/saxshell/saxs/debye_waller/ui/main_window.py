from __future__ import annotations

import sys
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.debye_waller.workflow import (
    DebyeWallerAggregatedPairSummary,
    DebyeWallerAnalysisResult,
    DebyeWallerInputInspection,
    DebyeWallerStoichiometryInfoSummary,
    DebyeWallerStoichiometryResult,
    DebyeWallerWorkflow,
    build_debye_waller_aggregated_pair_summaries,
    find_saved_project_debye_waller_analysis,
    inspect_debye_waller_input,
    load_debye_waller_analysis_result,
    save_debye_waller_analysis_to_project,
    suggest_output_dir,
)
from saxshell.saxs.project_manager import (
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.saxs.ui.project_status_label import CompactProjectStatusLabel

_OPEN_WINDOWS: list["DebyeWallerAnalysisMainWindow"] = []
DEBYE_WALLER_WINDOW_LOAD_TOTAL_STEPS = 6
DebyeWallerStartupProgressCallback = Callable[[int, int, str], None]


class DebyeWallerWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    status = Signal(str)
    stoichiometry_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, workflow: DebyeWallerWorkflow) -> None:
        super().__init__()
        self.workflow = workflow

    @Slot()
    def run(self) -> None:
        try:
            result = self.workflow.run(
                progress_callback=self._emit_progress,
                log_callback=self.log.emit,
                stoichiometry_callback=self.stoichiometry_ready.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.emit(processed, total)
        self.status.emit(message)


def _scope_display_name(scope: str) -> str:
    if scope == "intra_molecular":
        return "Intra-molecular"
    if scope == "inter_molecular":
        return "Inter-molecular"
    return str(scope)


class DebyeWallerAnalysisMainWindow(QMainWindow):
    """Standalone UI for trajectory-based Debye-Waller estimation."""

    project_paths_registered = Signal(object)
    project_analysis_saved = Signal(object)

    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_clusters_dir: str | Path | None = None,
        initial_output_dir: str | Path | None = None,
        startup_progress_callback: (
            DebyeWallerStartupProgressCallback | None
        ) = None,
        startup_log_callback: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_manager = SAXSProjectManager()
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._project_settings_cache: ProjectSettings | None = None
        self._project_settings_loaded = False
        self._run_thread: QThread | None = None
        self._run_worker: DebyeWallerWorker | None = None
        self._last_result: DebyeWallerAnalysisResult | None = None
        self._project_had_saved_analysis_before_run = False
        self._partial_stoichiometry_results: list[
            DebyeWallerStoichiometryResult
        ] = []
        self._build_ui()
        self._initialize_startup_state(
            initial_clusters_dir=initial_clusters_dir,
            initial_output_dir=initial_output_dir,
            startup_progress_callback=startup_progress_callback,
            startup_log_callback=startup_log_callback,
        )

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            QMessageBox.warning(
                self,
                "Debye-Waller Analysis",
                "Please wait for the active Debye-Waller run to finish "
                "before closing this window.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell (Debye-Waller Analysis)")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1380, 880)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([470, 910])

        root.addWidget(splitter)
        self.setCentralWidget(central)

        self.project_status_label = self._build_project_status_label()
        if self.project_status_label is not None:
            self.statusBar().addPermanentWidget(self.project_status_label)
        self.statusBar().showMessage("Ready")

    def _initialize_startup_state(
        self,
        *,
        initial_clusters_dir: str | Path | None,
        initial_output_dir: str | Path | None,
        startup_progress_callback: DebyeWallerStartupProgressCallback | None,
        startup_log_callback: Callable[[str], None] | None,
    ) -> None:
        resolved_initial_clusters_dir = (
            None
            if initial_clusters_dir is None
            else Path(initial_clusters_dir).expanduser().resolve()
        )
        resolved_initial_output_dir = (
            None
            if initial_output_dir is None
            else Path(initial_output_dir).expanduser().resolve()
        )
        processed_steps = 0

        def advance(
            message: str,
            *,
            log_message: str | None = None,
        ) -> None:
            nonlocal processed_steps
            processed_steps += 1
            if startup_progress_callback is not None:
                startup_progress_callback(
                    processed_steps,
                    DEBYE_WALLER_WINDOW_LOAD_TOTAL_STEPS,
                    message,
                )
            if (
                startup_log_callback is not None
                and log_message is not None
                and log_message.strip()
            ):
                startup_log_callback(log_message.strip())

        advance(
            "Preparing Debye-Waller analysis window...",
            log_message="Preparing Debye-Waller analysis window.",
        )
        if (
            self._project_dir is not None
            and resolved_initial_clusters_dir is None
        ):
            advance(
                "Loading active project settings...",
                log_message=(
                    "Loading active project settings from "
                    f"{self._project_dir}."
                ),
            )
            self._load_project_settings()
        else:
            advance(
                "Resolving Debye-Waller launch settings...",
                log_message=(
                    "Resolving Debye-Waller launch settings from the main UI."
                    if self._project_dir is not None
                    else "Resolving standalone Debye-Waller launch settings."
                ),
            )
        advance(
            "Configuring startup folders...",
            log_message="Configuring Debye-Waller startup folders.",
        )
        if resolved_initial_clusters_dir is not None:
            self.set_clusters_dir(
                resolved_initial_clusters_dir,
                register_project=False,
                refresh_summary=False,
            )
        else:
            self._adopt_project_clusters_reference()
            self._apply_default_output_dir()
        if resolved_initial_output_dir is not None:
            self.set_output_dir(resolved_initial_output_dir)

        advance(
            "Checking for saved project analysis...",
            log_message="Checking the active project for saved Debye-Waller results.",
        )
        restored_saved_result = False
        summary_path = (
            None
            if self._project_dir is None
            else find_saved_project_debye_waller_analysis(self._project_dir)
        )
        if summary_path is not None and summary_path.is_file():
            advance(
                "Loading saved project analysis...",
                log_message=(
                    "Loading any saved Debye-Waller analysis linked to the "
                    "active project."
                ),
            )
            restored_saved_result = (
                self._restore_saved_project_analysis_if_available(
                    register_project=False,
                    summary_path=summary_path,
                )
            )
        else:
            advance(
                "Inspecting selected clusters folder...",
                log_message="Inspecting the selected Debye-Waller clusters folder.",
            )
        if not restored_saved_result:
            self._refresh_summary()

        advance(
            "Finalizing Debye-Waller window...",
            log_message="Debye-Waller analysis window is ready.",
        )
        self._refresh_project_save_controls()
        self._refresh_project_status_label()

    def _build_left_panel(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_group = QGroupBox("Debye-Waller Analysis (beta)")
        header_layout = QVBoxLayout(header_group)
        header_notice = QLabel(
            "Estimate contiguous-segment intra- and inter-molecular "
            "thermal-displacement coefficients from sorted PDB cluster files "
            "using PDB-based molecule grouping."
        )
        header_notice.setWordWrap(True)
        header_layout.addWidget(header_notice)
        layout.addWidget(header_group)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch(1)
        return left

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        summary_group = QGroupBox("Selection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(180)
        summary_layout.addWidget(self.summary_box)
        layout.addWidget(summary_group)

        self.results_tabs = QTabWidget()
        self.stoichiometry_info_table = self._build_table(
            (
                "Stoichiometry",
                "Frames",
                "Frame Sets",
                "Avg Frames/Set",
                "Atoms/Frame Avg",
                "Mols/Frame Avg",
                "Solvent-like/Frame Avg",
                "Shared Sites/Set Avg",
                "Residue Names",
                "Elements",
                "Solvent-like Mol",
                "Most Common Mol",
            )
        )
        self.aggregated_pair_table = self._build_table(
            (
                "Scope",
                "Type Def.",
                "Pair",
                "Stoichs",
                "Segments",
                "Mean Sigma (A)",
                "Std Sigma (A)",
                "Mean B (A^2)",
                "Std B (A^2)",
                "Mean Pair Count",
            )
        )
        self.pair_summary_table = self._build_table(
            (
                "Stoichiometry",
                "Scope",
                "Type Def.",
                "Pair",
                "Segments",
                "Mean Sigma (A)",
                "Std Sigma (A)",
                "Mean B (A^2)",
                "Std B (A^2)",
                "Mean Pair Count",
            )
        )
        self.scope_summary_table = self._build_table(
            (
                "Stoichiometry",
                "Scope",
                "Rows",
                "Mean Sigma (A)",
                "Std Sigma (A)",
                "Mean B (A^2)",
                "Std B (A^2)",
                "Mean Pair Count",
            )
        )
        self.segment_table = self._build_table(
            (
                "Stoichiometry",
                "Segment",
                "Frames",
                "Scope",
                "Type Def.",
                "Pair",
                "Sigma (A)",
                "B (A^2)",
                "Pair Count",
            )
        )
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.results_tabs.addTab(
            self.stoichiometry_info_table, "Stoichiometries"
        )
        self.results_tabs.addTab(
            self.aggregated_pair_table, "Aggregated Pairs"
        )
        self.results_tabs.addTab(self.pair_summary_table, "Pair Types")
        self.results_tabs.addTab(self.scope_summary_table, "Scopes")
        self.results_tabs.addTab(self.segment_table, "Segments")
        self.results_tabs.addTab(self.log_box, "Log")
        layout.addWidget(self.results_tabs)

        return right

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Paths")
        layout = QFormLayout(group)

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.editingFinished.connect(
            self._on_clusters_dir_edited
        )
        browse_clusters_button = QPushButton("Browse...")
        browse_clusters_button.clicked.connect(self._browse_clusters_dir)
        clusters_row = QHBoxLayout()
        clusters_row.addWidget(self.clusters_dir_edit, 1)
        clusters_row.addWidget(browse_clusters_button)
        layout.addRow("Sorted clusters folder", self._wrap_row(clusters_row))

        self.output_dir_edit = QLineEdit()
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(browse_output_button)
        layout.addRow("Output folder", self._wrap_row(output_row))

        self.output_basename_edit = QLineEdit("debye_waller_analysis")
        layout.addRow("Output basename", self.output_basename_edit)

        self.paths_hint_label = QLabel(
            "PDB-only input. XYZ cluster files are rejected so the tool can "
            "use PDB residue/sequence/segment metadata for intra/inter "
            "molecule classification."
        )
        self.paths_hint_label.setWordWrap(True)
        layout.addRow("", self.paths_hint_label)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)

        self.progress_label = QLabel("Progress: idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.run_button = QPushButton("Run Debye-Waller Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.save_project_button = QPushButton(
            "Save Current Analysis to Project"
        )
        self.save_project_button.clicked.connect(
            self._save_current_analysis_to_project
        )
        self.save_project_button.setEnabled(False)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.run_button)
        layout.addWidget(self.save_project_button)
        return group

    def _build_table(self, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.setSortingEnabled(False)
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            header.setStretchLastSection(True)
        return table

    def _wrap_row(self, layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _load_project_settings(
        self,
        *,
        force_reload: bool = False,
    ) -> ProjectSettings | None:
        if self._project_dir is None:
            return None
        if self._project_settings_loaded and not force_reload:
            return self._project_settings_cache
        project_file = build_project_paths(self._project_dir).project_file
        if not project_file.is_file():
            self._project_settings_cache = None
            self._project_settings_loaded = True
            return None
        try:
            self._project_settings_cache = self._project_manager.load_project(
                self._project_dir
            )
        except Exception:
            self._project_settings_cache = None
            self._project_settings_loaded = False
            return None
        self._project_settings_loaded = True
        return self._project_settings_cache

    def _refresh_project_status_label(self) -> None:
        if self.project_status_label is None:
            return
        status_text = self._project_status_text()
        if status_text is not None:
            self.project_status_label.set_full_text(status_text)
        self.project_status_label.setToolTip(
            self._project_status_tooltip() or ""
        )

    def _project_status_text(self) -> str | None:
        if self._project_dir is None:
            return None
        return f"Active project: {self._project_dir}"

    def _project_status_tooltip(self) -> str | None:
        if self._project_dir is None:
            return None
        project_name = (
            self._project_dir.name
            if self._project_settings_cache is None
            else self._project_settings_cache.project_name.strip()
            or self._project_dir.name
        )
        return (
            f"Active project: {project_name}\n"
            f"{self._project_dir}\n\n"
            "This window inherits the active clusters folder when launched "
            "from the main SAXS UI."
        )

    def _build_project_status_label(
        self,
    ) -> CompactProjectStatusLabel | None:
        status_text = self._project_status_text()
        if status_text is None:
            return None
        label = CompactProjectStatusLabel(self.statusBar())
        label.setToolTip(self._project_status_tooltip() or "")
        label.set_full_text(status_text)
        return label

    def _clusters_dir_path(self) -> Path | None:
        text = self.clusters_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _project_clusters_dir(self) -> Path | None:
        settings = self._load_project_settings()
        if settings is None:
            return None
        return settings.resolved_clusters_dir

    def _output_dir_path(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _output_basename(self) -> str:
        text = self.output_basename_edit.text().strip()
        return text or "debye_waller_analysis"

    def _set_clusters_dir_text(self, path: str | Path | None) -> None:
        self.clusters_dir_edit.setText(
            "" if path is None else str(Path(path).expanduser().resolve())
        )

    def _set_output_dir_text(self, path: str | Path | None) -> None:
        self.output_dir_edit.setText(
            "" if path is None else str(Path(path).expanduser().resolve())
        )

    @staticmethod
    def _normalized_registered_path_value(
        value: str | Path | None,
    ) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return str(Path(text).expanduser().resolve())

    def _save_project_without_registered_path_refresh(
        self,
        settings: ProjectSettings,
    ) -> None:
        save_project = self._project_manager.save_project
        try:
            save_signature = signature(save_project)
        except (TypeError, ValueError):
            save_signature = None
        if save_signature is not None:
            supports_refresh_flag = (
                "refresh_registered_paths" in save_signature.parameters
                or any(
                    parameter.kind == Parameter.VAR_KEYWORD
                    for parameter in save_signature.parameters.values()
                )
            )
            if not supports_refresh_flag:
                save_project(settings)
                return
        save_project(
            settings,
            refresh_registered_paths=False,
        )

    def set_clusters_dir(
        self,
        path: str | Path | None,
        *,
        register_project: bool = True,
        refresh_summary: bool = True,
    ) -> None:
        self._set_clusters_dir_text(path)
        self._apply_default_output_dir()
        if register_project:
            self._register_project_clusters_dir()
        if refresh_summary:
            self._refresh_summary()

    def set_output_dir(self, path: str | Path | None) -> None:
        self._set_output_dir_text(path)

    def _adopt_project_clusters_reference(self) -> None:
        if self.clusters_dir_edit.text().strip():
            return
        clusters_dir = self._project_clusters_dir()
        if clusters_dir is None:
            return
        self._set_clusters_dir_text(clusters_dir)

    def _apply_default_output_dir(self) -> None:
        if self.output_dir_edit.text().strip():
            return
        clusters_dir = self._clusters_dir_path()
        if clusters_dir is None:
            if self._project_dir is None:
                return
            suggested = suggest_output_dir(
                self._project_dir, project_dir=self._project_dir
            )
        else:
            suggested = suggest_output_dir(
                clusters_dir,
                project_dir=self._project_dir,
            )
        self.output_dir_edit.setText(str(suggested))

    def _browse_clusters_dir(self) -> None:
        start_dir = self.clusters_dir_edit.text().strip() or (
            "" if self._project_dir is None else str(self._project_dir)
        )
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Sorted Clusters Folder",
            start_dir,
        )
        if chosen_dir:
            self.set_clusters_dir(chosen_dir)

    def _browse_output_dir(self) -> None:
        start_dir = self.output_dir_edit.text().strip() or (
            "" if self._project_dir is None else str(self._project_dir)
        )
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Debye-Waller Output Folder",
            start_dir,
        )
        if chosen_dir:
            self.set_output_dir(chosen_dir)

    def _on_clusters_dir_edited(self) -> None:
        self._apply_default_output_dir()
        self._register_project_clusters_dir()
        self._refresh_summary()

    def _register_project_clusters_dir(self) -> None:
        settings = self._load_project_settings()
        clusters_dir = self._clusters_dir_path()
        changed = False
        if settings is not None:
            try:
                resolved_clusters_dir = (
                    None
                    if clusters_dir is None
                    else str(clusters_dir.expanduser().resolve())
                )
                previous_clusters_dir = self._normalized_registered_path_value(
                    settings.clusters_dir
                )
                if previous_clusters_dir != resolved_clusters_dir:
                    changed = True
                    settings.clusters_dir = resolved_clusters_dir
                    settings.clusters_dir_snapshot = None
                    self._save_project_without_registered_path_refresh(
                        settings
                    )
                self._project_settings_cache = settings
                self._project_settings_loaded = True
            except Exception:
                pass
        if self._project_dir is not None and changed:
            self.project_paths_registered.emit(
                {
                    "project_dir": str(self._project_dir),
                    "clusters_dir": (
                        None
                        if clusters_dir is None
                        else str(clusters_dir.expanduser().resolve())
                    ),
                }
            )

    def _refresh_project_save_controls(self) -> None:
        self.save_project_button.setEnabled(
            self._project_dir is not None and self._last_result is not None
        )

    def _emit_project_analysis_saved(
        self,
        result: DebyeWallerAnalysisResult,
    ) -> None:
        if self._project_dir is None or result.artifacts is None:
            return
        self.project_analysis_saved.emit(
            {
                "project_dir": str(self._project_dir),
                "clusters_dir": str(result.clusters_dir),
                "summary_path": str(result.artifacts.summary_json_path),
            }
        )

    def _populate_result_tables(
        self,
        result: DebyeWallerAnalysisResult,
    ) -> None:
        self._populate_aggregated_pair_table(result.aggregated_pair_summaries)
        self._populate_stoichiometry_info_table_from_stoichiometries(
            result.stoichiometry_results
        )
        self._populate_pair_summary_table_from_stoichiometries(
            result.stoichiometry_results
        )
        self._populate_scope_summary_table_from_stoichiometries(
            result.stoichiometry_results
        )
        self._populate_segment_table_from_stoichiometries(
            result.stoichiometry_results
        )

    def _restore_saved_project_analysis_if_available(
        self,
        *,
        register_project: bool = True,
        summary_path: Path | None = None,
    ) -> bool:
        if self._project_dir is None:
            return False
        if summary_path is None:
            summary_path = find_saved_project_debye_waller_analysis(
                self._project_dir
            )
        if summary_path is None or not summary_path.is_file():
            return False
        try:
            result = load_debye_waller_analysis_result(summary_path)
        except Exception:
            return False

        self._last_result = result
        self._partial_stoichiometry_results = list(
            result.stoichiometry_results
        )
        self.set_clusters_dir(
            result.clusters_dir,
            register_project=register_project,
            refresh_summary=False,
        )
        self.set_output_dir(result.output_dir)
        if result.artifacts is not None:
            self.output_basename_edit.setText(
                result.artifacts.summary_json_path.stem
            )
        self._populate_result_tables(result)
        self.summary_box.setPlainText(self._describe_result(result))
        self.log_box.setPlainText(
            "\n".join(
                [
                    "Loaded saved Debye-Waller analysis from the active project.",
                    f"Summary JSON: {summary_path}",
                ]
            )
        )
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_label.setText("Progress: loaded saved project analysis")
        self.statusBar().showMessage(
            "Loaded saved Debye-Waller analysis from project"
        )
        self._refresh_project_save_controls()
        return True

    @Slot()
    def _save_current_analysis_to_project(self) -> None:
        if self._project_dir is None or self._last_result is None:
            return
        saved_result = save_debye_waller_analysis_to_project(
            self._last_result,
            self._project_dir,
        )
        self._last_result = saved_result
        self._partial_stoichiometry_results = list(
            saved_result.stoichiometry_results
        )
        self.set_output_dir(saved_result.output_dir)
        if saved_result.artifacts is not None:
            self.output_basename_edit.setText(
                saved_result.artifacts.summary_json_path.stem
            )
            self._append_log(
                "Saved the current Debye-Waller analysis to the active project."
            )
            self._append_log(
                f"Project summary JSON: {saved_result.artifacts.summary_json_path}"
            )
        self.summary_box.setPlainText(self._describe_result(saved_result))
        self.statusBar().showMessage(
            "Saved Debye-Waller analysis to active project"
        )
        self._project_had_saved_analysis_before_run = True
        self._emit_project_analysis_saved(saved_result)
        self._refresh_project_save_controls()

    def _describe_inspection(
        self,
        inspection: DebyeWallerInputInspection | None,
    ) -> str:
        lines = [
            "Debye-Waller Analysis",
            "",
            "This tool expects sorted cluster frames in PDB format and uses "
            "PDB residue/sequence/segment grouping to separate intra- and "
            "inter-molecular pair statistics.",
        ]
        if self._project_dir is not None:
            lines.extend(
                [
                    "",
                    f"Active project: {self._project_dir}",
                ]
            )
            project_clusters_dir = self._project_clusters_dir()
            if (
                project_clusters_dir is not None
                and self.clusters_dir_edit.text().strip()
                and project_clusters_dir == self._clusters_dir_path()
            ):
                lines.append(
                    "Using the clusters folder reference inherited from the "
                    "active project."
                )
        clusters_dir = self._clusters_dir_path()
        if clusters_dir is None:
            lines.extend(
                [
                    "",
                    "No clusters folder is currently selected.",
                ]
            )
            return "\n".join(lines)

        lines.extend(
            [
                "",
                f"Sorted clusters folder: {clusters_dir}",
            ]
        )
        if inspection is None:
            lines.append("Selection has not been inspected yet.")
            return "\n".join(lines)

        lines.extend(
            [
                f"Detected stoichiometry bins: {inspection.stoichiometry_count}",
                f"Detected structure files: {inspection.total_structure_files}",
            ]
        )
        if inspection.stoichiometry_labels:
            preview_labels = "\n".join(
                f"- {label}" for label in inspection.stoichiometry_labels[:10]
            )
            lines.extend(
                [
                    "",
                    "Stoichiometry labels:",
                    preview_labels,
                ]
            )
            remaining = len(inspection.stoichiometry_labels) - min(
                len(inspection.stoichiometry_labels), 10
            )
            if remaining > 0:
                lines.append(f"... and {remaining} more.")
        if inspection.invalid_xyz_files:
            lines.extend(
                [
                    "",
                    "XYZ input detected. This tool will refuse to run until "
                    "those files are converted to PDB:",
                    *[
                        f"- {path}"
                        for path in inspection.invalid_xyz_files[:10]
                    ],
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Input validation: PDB-only input detected.",
                ]
            )
        output_dir = self._output_dir_path()
        if output_dir is not None:
            lines.extend(
                [
                    "",
                    f"Output folder: {output_dir}",
                    f"Output basename: {self._output_basename()}",
                ]
            )
        return "\n".join(lines)

    def _refresh_summary(self) -> None:
        inspection = None
        clusters_dir = self._clusters_dir_path()
        if clusters_dir is not None and clusters_dir.exists():
            try:
                inspection = inspect_debye_waller_input(clusters_dir)
            except Exception:
                inspection = None
        self.summary_box.setPlainText(self._describe_inspection(inspection))

    def _append_log(self, message: str) -> None:
        current = self.log_box.toPlainText()
        if current:
            self.log_box.appendPlainText(message)
        else:
            self.log_box.setPlainText(message)

    @Slot()
    def run_analysis(self) -> None:
        clusters_dir = self._clusters_dir_path()
        output_dir = self._output_dir_path()
        if clusters_dir is None:
            QMessageBox.warning(
                self,
                "Debye-Waller Analysis",
                "Select a sorted clusters folder before running the analysis.",
            )
            return
        if output_dir is None:
            QMessageBox.warning(
                self,
                "Debye-Waller Analysis",
                "Select an output folder before running the analysis.",
            )
            return

        workflow = DebyeWallerWorkflow(
            clusters_dir,
            project_dir=self._project_dir,
            output_dir=output_dir,
            output_basename=self._output_basename(),
        )
        self._project_had_saved_analysis_before_run = (
            self._project_dir is not None
            and find_saved_project_debye_waller_analysis(self._project_dir)
            is not None
        )
        self._partial_stoichiometry_results = []
        self._last_result = None
        self.stoichiometry_info_table.setRowCount(0)
        self.aggregated_pair_table.setRowCount(0)
        self.pair_summary_table.setRowCount(0)
        self.scope_summary_table.setRowCount(0)
        self.segment_table.setRowCount(0)
        self.log_box.setPlainText("")
        self.results_tabs.setCurrentWidget(self.log_box)
        self._append_log(f"Starting Debye-Waller analysis for {clusters_dir}.")
        self.run_button.setEnabled(False)
        self._refresh_project_save_controls()
        self.progress_label.setText(
            "Progress: validating input and preparing the run"
        )
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.summary_box.setPlainText(
            "\n".join(
                [
                    "Debye-Waller Analysis",
                    "",
                    "Run in progress.",
                    f"Sorted clusters folder: {clusters_dir}",
                    f"Output folder: {output_dir}",
                    "",
                    "Waiting for contiguous frame sets and coefficient rows...",
                ]
            )
        )

        thread = QThread(self)
        worker = DebyeWallerWorker(workflow)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.progress.connect(self._update_progress)
        worker.status.connect(self._update_status)
        worker.stoichiometry_ready.connect(self._update_partial_results)
        worker.finished.connect(self._finish_run)
        worker.finished.connect(self._cleanup_run_thread)
        worker.failed.connect(self._fail_run)
        worker.failed.connect(self._cleanup_run_thread)
        self._run_thread = thread
        self._run_worker = worker
        thread.start()

    @Slot(int, int)
    def _update_progress(self, processed: int, total: int) -> None:
        safe_total = max(int(total), 1)
        self.progress_bar.setRange(0, safe_total)
        self.progress_bar.setValue(min(max(int(processed), 0), safe_total))

    @Slot(str)
    def _update_status(self, message: str) -> None:
        self.progress_label.setText(f"Progress: {message}")
        self.statusBar().showMessage(message)

    @Slot(object)
    def _update_partial_results(self, payload: object) -> None:
        if not isinstance(payload, DebyeWallerStoichiometryResult):
            return
        for index, existing in enumerate(self._partial_stoichiometry_results):
            if existing.label == payload.label:
                self._partial_stoichiometry_results[index] = payload
                break
        else:
            self._partial_stoichiometry_results.append(payload)
        self._populate_aggregated_pair_table(
            build_debye_waller_aggregated_pair_summaries(
                self._partial_stoichiometry_results
            )
        )
        self._populate_stoichiometry_info_table_from_stoichiometries(
            self._partial_stoichiometry_results
        )
        self._populate_pair_summary_table_from_stoichiometries(
            self._partial_stoichiometry_results
        )
        self._populate_scope_summary_table_from_stoichiometries(
            self._partial_stoichiometry_results
        )
        self._populate_segment_table_from_stoichiometries(
            self._partial_stoichiometry_results
        )
        self.summary_box.setPlainText(
            self._describe_partial_progress(
                self._partial_stoichiometry_results
            )
        )

    @Slot(object)
    def _finish_run(self, payload: object) -> None:
        if not isinstance(payload, DebyeWallerAnalysisResult):
            self._fail_run(
                "The Debye-Waller workflow returned an invalid payload."
            )
            return
        self._last_result = payload
        self.run_button.setEnabled(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_label.setText("Progress: complete")
        self.statusBar().showMessage("Debye-Waller analysis complete")
        self._append_log("Debye-Waller analysis finished successfully.")
        if payload.artifacts is not None:
            self._append_log(
                f"Summary JSON: {payload.artifacts.summary_json_path}"
            )
        self._populate_result_tables(payload)
        self.summary_box.setPlainText(self._describe_result(payload))
        if (
            self._project_dir is not None
            and not self._project_had_saved_analysis_before_run
        ):
            saved_result = save_debye_waller_analysis_to_project(
                payload,
                self._project_dir,
            )
            self._last_result = saved_result
            self.set_output_dir(saved_result.output_dir)
            if saved_result.artifacts is not None:
                self.output_basename_edit.setText(
                    saved_result.artifacts.summary_json_path.stem
                )
                self._append_log(
                    "Auto-saved the first Debye-Waller analysis for this "
                    "project."
                )
                self._append_log(
                    f"Project summary JSON: {saved_result.artifacts.summary_json_path}"
                )
            self.summary_box.setPlainText(self._describe_result(saved_result))
            self._project_had_saved_analysis_before_run = True
            self._emit_project_analysis_saved(saved_result)
        self._refresh_project_save_controls()

    @Slot(str)
    def _fail_run(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: failed")
        self.statusBar().showMessage("Debye-Waller analysis failed")
        self._append_log(f"Run failed: {message}")
        self._refresh_project_save_controls()
        QMessageBox.critical(self, "Debye-Waller Analysis", message)

    @Slot(object)
    def _cleanup_run_thread(self, _payload: object) -> None:
        if self._run_thread is None:
            return
        self._run_thread.quit()
        self._run_thread.wait()
        if self._run_worker is not None:
            self._run_worker.deleteLater()
        self._run_thread.deleteLater()
        self._run_worker = None
        self._run_thread = None

    def _describe_result(self, result: DebyeWallerAnalysisResult) -> str:
        lines = [
            "Debye-Waller Analysis",
            "",
            f"Created: {result.created_at}",
            f"Clusters folder: {result.clusters_dir}",
            f"Output folder: {result.output_dir}",
            "",
            f"Stoichiometry results: {len(result.stoichiometry_results)}",
            f"Aggregated pair rows: {len(result.aggregated_pair_summaries)}",
            f"Pair-type summary rows: "
            f"{sum(len(entry.pair_summaries) for entry in result.stoichiometry_results)}",
            f"Scope summary rows: "
            f"{sum(len(entry.scope_summaries) for entry in result.stoichiometry_results)}",
            f"Segment rows: "
            f"{sum(len(entry.segment_statistics) for entry in result.stoichiometry_results)}",
        ]
        if result.artifacts is not None:
            lines.extend(
                [
                    "",
                    "Artifacts:",
                    f"- {result.artifacts.summary_json_path}",
                    f"- {result.artifacts.aggregated_pair_summary_csv_path}",
                    f"- {result.artifacts.pair_summary_csv_path}",
                    f"- {result.artifacts.scope_summary_csv_path}",
                    f"- {result.artifacts.segment_csv_path}",
                ]
            )
        for stoichiometry in result.stoichiometry_results:
            lines.extend(
                [
                    "",
                    f"{stoichiometry.label}: "
                    f"{len(stoichiometry.contiguous_frame_sets)} frame set(s), "
                    f"{len(stoichiometry.pair_summaries)} pair-type row(s), "
                    f"{len(stoichiometry.scope_summaries)} scope row(s).",
                ]
            )
            for note in stoichiometry.notes[:4]:
                lines.append(f"  note: {note}")
            remaining_notes = max(len(stoichiometry.notes) - 4, 0)
            if remaining_notes:
                lines.append(f"  ... and {remaining_notes} more note(s).")
        return "\n".join(lines)

    def _describe_partial_progress(
        self,
        stoichiometry_results: (
            list[DebyeWallerStoichiometryResult]
            | tuple[DebyeWallerStoichiometryResult, ...]
        ),
    ) -> str:
        completed_labels = len(stoichiometry_results)
        lines = [
            "Debye-Waller Analysis",
            "",
            "Run in progress.",
            f"Completed stoichiometry labels: {completed_labels}",
            "Tables update as each contiguous frame set finishes.",
            "",
            f"Aggregated pair rows: "
            f"{len(build_debye_waller_aggregated_pair_summaries(stoichiometry_results))}",
            f"Pair-type summary rows: "
            f"{sum(len(entry.pair_summaries) for entry in stoichiometry_results)}",
            f"Scope summary rows: "
            f"{sum(len(entry.scope_summaries) for entry in stoichiometry_results)}",
            f"Segment rows: "
            f"{sum(len(entry.segment_statistics) for entry in stoichiometry_results)}",
        ]
        for stoichiometry in stoichiometry_results[-5:]:
            info_summary = stoichiometry.info_summary
            lines.extend(
                [
                    "",
                    f"{stoichiometry.label}: "
                    f"{len(stoichiometry.contiguous_frame_sets)} frame set(s), "
                    f"{len(stoichiometry.segment_statistics)} segment row(s), "
                    f"{0.0 if info_summary is None else info_summary.average_atoms_per_frame:.1f} "
                    "atom(s)/frame.",
                ]
            )
        return "\n".join(lines)

    def _populate_stoichiometry_info_table_from_stoichiometries(
        self,
        stoichiometry_results: (
            list[DebyeWallerStoichiometryResult]
            | tuple[DebyeWallerStoichiometryResult, ...]
        ),
    ) -> None:
        rows: list[DebyeWallerStoichiometryInfoSummary] = []
        for stoichiometry in stoichiometry_results:
            if stoichiometry.info_summary is not None:
                rows.append(stoichiometry.info_summary)
        self.stoichiometry_info_table.setSortingEnabled(False)
        self.stoichiometry_info_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.stoichiometry_label,
                f"{row.processed_frame_count}/{row.total_frame_count}",
                f"{row.processed_frame_set_count}/{row.total_frame_set_count}",
                f"{float(row.average_frames_per_set):.2f}",
                f"{float(row.average_atoms_per_frame):.2f}",
                f"{float(row.average_molecules_per_frame):.2f}",
                f"{float(row.average_solvent_like_molecules_per_frame):.2f}",
                f"{float(row.average_shared_atom_sites_per_set):.2f}",
                ", ".join(row.unique_residue_names) or "n/a",
                ", ".join(row.unique_elements) or "n/a",
                row.solvent_like_molecule_signature or "n/a",
                row.most_common_molecule_signature or "n/a",
            )
            for column_index, value in enumerate(values):
                self.stoichiometry_info_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.stoichiometry_info_table.resizeColumnsToContents()
        self.stoichiometry_info_table.setSortingEnabled(True)

    def _populate_aggregated_pair_table(
        self,
        rows: (
            list[DebyeWallerAggregatedPairSummary]
            | tuple[DebyeWallerAggregatedPairSummary, ...]
        ),
    ) -> None:
        self.aggregated_pair_table.setSortingEnabled(False)
        self.aggregated_pair_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                _scope_display_name(row.scope),
                row.type_definition,
                row.pair_label,
                str(row.stoichiometry_count),
                str(row.segment_count),
                f"{float(row.sigma_mean):.6f}",
                f"{float(row.sigma_std):.6f}",
                f"{float(row.b_factor_mean):.6f}",
                f"{float(row.b_factor_std):.6f}",
                f"{float(row.mean_pair_count):.2f}",
            )
            for column_index, value in enumerate(values):
                self.aggregated_pair_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.aggregated_pair_table.resizeColumnsToContents()
        self.aggregated_pair_table.setSortingEnabled(True)

    def _populate_pair_summary_table_from_stoichiometries(
        self,
        stoichiometry_results: (
            list[DebyeWallerStoichiometryResult]
            | tuple[DebyeWallerStoichiometryResult, ...]
        ),
    ) -> None:
        rows = [
            summary
            for stoichiometry in stoichiometry_results
            for summary in stoichiometry.pair_summaries
        ]
        self.pair_summary_table.setSortingEnabled(False)
        self.pair_summary_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.stoichiometry_label,
                _scope_display_name(row.scope),
                row.type_definition,
                row.pair_label,
                str(row.segment_count),
                f"{float(row.sigma_mean):.6f}",
                f"{float(row.sigma_std):.6f}",
                f"{float(row.b_factor_mean):.6f}",
                f"{float(row.b_factor_std):.6f}",
                f"{float(row.mean_pair_count):.2f}",
            )
            for column_index, value in enumerate(values):
                self.pair_summary_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.pair_summary_table.resizeColumnsToContents()
        self.pair_summary_table.setSortingEnabled(True)

    def _populate_scope_summary_table_from_stoichiometries(
        self,
        stoichiometry_results: (
            list[DebyeWallerStoichiometryResult]
            | tuple[DebyeWallerStoichiometryResult, ...]
        ),
    ) -> None:
        rows = [
            summary
            for stoichiometry in stoichiometry_results
            for summary in stoichiometry.scope_summaries
        ]
        self.scope_summary_table.setSortingEnabled(False)
        self.scope_summary_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.stoichiometry_label,
                _scope_display_name(row.scope),
                str(row.segment_count),
                f"{float(row.sigma_mean):.6f}",
                f"{float(row.sigma_std):.6f}",
                f"{float(row.b_factor_mean):.6f}",
                f"{float(row.b_factor_std):.6f}",
                f"{float(row.mean_pair_count):.2f}",
            )
            for column_index, value in enumerate(values):
                self.scope_summary_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.scope_summary_table.resizeColumnsToContents()
        self.scope_summary_table.setSortingEnabled(True)

    def _populate_segment_table_from_stoichiometries(
        self,
        stoichiometry_results: (
            list[DebyeWallerStoichiometryResult]
            | tuple[DebyeWallerStoichiometryResult, ...]
        ),
    ) -> None:
        rows = [
            summary
            for stoichiometry in stoichiometry_results
            for summary in stoichiometry.segment_statistics
        ]
        self.segment_table.setSortingEnabled(False)
        self.segment_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            frame_text = (
                f"{row.frame_start}-{row.frame_end} "
                f"({row.frame_count} frame(s))"
            )
            values = (
                row.stoichiometry_label,
                f"{row.segment_index}: {row.series_label}",
                frame_text,
                _scope_display_name(row.scope),
                row.type_definition,
                row.pair_label,
                f"{float(row.sigma):.6f}",
                f"{float(row.b_factor):.6f}",
                str(row.pair_count),
            )
            for column_index, value in enumerate(values):
                self.segment_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )
        self.segment_table.resizeColumnsToContents()
        self.segment_table.setSortingEnabled(True)


def _release_window(window: DebyeWallerAnalysisMainWindow) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def launch_debye_waller_analysis_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_clusters_dir: str | Path | None = None,
    initial_output_dir: str | Path | None = None,
    startup_progress_callback: (
        DebyeWallerStartupProgressCallback | None
    ) = None,
    startup_log_callback: Callable[[str], None] | None = None,
) -> DebyeWallerAnalysisMainWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = DebyeWallerAnalysisMainWindow(
        initial_project_dir=initial_project_dir,
        initial_clusters_dir=initial_clusters_dir,
        initial_output_dir=initial_output_dir,
        startup_progress_callback=startup_progress_callback,
        startup_log_callback=startup_log_callback,
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda _obj=None: _release_window(window))
    return window


__all__ = [
    "DEBYE_WALLER_WINDOW_LOAD_TOTAL_STEPS",
    "DebyeWallerAnalysisMainWindow",
    "launch_debye_waller_analysis_ui",
]
