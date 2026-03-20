from __future__ import annotations

import csv
import pickle
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QObject,
    QSettings,
    QSize,
    Qt,
    QThread,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs._model_templates import list_template_specs
from saxshell.saxs.dream import (
    DreamRunBundle,
    DreamRunSettings,
    SAXSDreamResultsLoader,
    SAXSDreamWorkflow,
)
from saxshell.saxs.prefit import PrefitScaleRecommendation, SAXSPrefitWorkflow
from saxshell.saxs.project_manager import (
    ClusterImportResult,
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.distribution_window import DistributionSetupWindow
from saxshell.saxs.ui.dream_tab import DreamTab
from saxshell.saxs.ui.dream_violin_export_dialog import DreamViolinExportDialog
from saxshell.saxs.ui.prefit_tab import PrefitTab
from saxshell.saxs.ui.prior_histogram_window import PriorHistogramWindow
from saxshell.saxs.ui.progress_dialog import SAXSProgressDialog
from saxshell.saxs.ui.project_setup_tab import ProjectSetupTab
from saxshell.version import __version__

GITHUB_REPOSITORY_URL = "https://github.com/kewh5868/SAXShell"
CONTACT_EMAIL = "keith.white@colorado.edu"
RECENT_PROJECTS_KEY = "recent_project_dirs"
MAX_RECENT_PROJECTS = 10
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class RuntimeBundleOpener:
    label: str
    stored_value: str
    launch_target: str
    launch_mode: str


class SAXSProjectTaskWorker(QObject):
    """Background worker for cluster import and project build tasks."""

    progress = Signal(int, int, str)
    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(
        self,
        task_name: str,
        task_fn: Callable[[Callable[[int, int, str], None]], object],
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self.task_fn = task_fn

    @Slot()
    def run(self) -> None:
        try:
            result = self.task_fn(self._emit_progress)
        except Exception as exc:
            self.failed.emit(self.task_name, str(exc))
            return
        self.finished.emit(self.task_name, result)

    def _emit_progress(self, processed: int, total: int, message: str) -> None:
        self.progress.emit(int(processed), int(total), str(message))


class SAXSDreamRunWorker(QObject):
    """Background worker for DREAM runtime execution."""

    status = Signal(str)
    output = Signal(str)
    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(
        self,
        project_dir: str | Path,
        bundle: DreamRunBundle,
        *,
        verbose_output_interval_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self.project_dir = str(Path(project_dir).expanduser().resolve())
        self.bundle = bundle
        self.verbose_output_interval_seconds = max(
            float(verbose_output_interval_seconds),
            0.1,
        )

    @Slot()
    def run(self) -> None:
        try:
            self.status.emit("Executing DREAM runtime bundle...")
            workflow = SAXSDreamWorkflow(self.project_dir)
            result = workflow.run_bundle(
                self.bundle,
                output_callback=self.output.emit,
                output_interval_seconds=self.verbose_output_interval_seconds,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(self.bundle.run_dir), result)


class SAXSMainWindow(QMainWindow):
    """Main Qt window for SAXS project setup, prefit, and DREAM
    refinement."""

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.project_manager = SAXSProjectManager()
        self.current_settings: ProjectSettings | None = None
        self.prefit_workflow: SAXSPrefitWorkflow | None = None
        self.dream_workflow: SAXSDreamWorkflow | None = None
        self.distribution_window: DistributionSetupWindow | None = None
        self._last_results_loader: SAXSDreamResultsLoader | None = None
        self._last_written_dream_bundle: DreamRunBundle | None = None
        self._dream_workflow_project_dir: str | None = None
        self._dream_parameter_map_saved_in_session = False
        self._active_dream_settings_snapshot: DreamRunSettings | None = None
        self._current_dream_preset_name: str | None = None
        self._prior_histogram_windows: list[PriorHistogramWindow] = []
        self._task_thread: QThread | None = None
        self._task_worker: SAXSProjectTaskWorker | None = None
        self._progress_dialog: SAXSProgressDialog | None = None
        self._active_task_name: str | None = None
        self._active_task_settings: ProjectSettings | None = None
        self._dream_task_thread: QThread | None = None
        self._dream_task_worker: SAXSDreamRunWorker | None = None
        self._dream_progress_dialog: SAXSProgressDialog | None = None
        self._active_dream_run_settings: DreamRunSettings | None = None
        self._warn_on_prefit_template_change = True
        self._restoring_prefit_template = False
        self._ui_scale = 1.0
        self._base_font_point_size = self._resolve_base_font_point_size()
        self._scale_shortcuts: list[QShortcut] = []
        self._child_tool_windows: list[QWidget] = []
        self._build_ui()
        self._capture_scale_baselines(self)
        self._register_scale_shortcuts()
        self._apply_ui_scale(announce=False)
        template_specs = list_template_specs()
        self.project_setup_tab.set_available_templates(template_specs, None)
        self.prefit_tab.set_templates(template_specs, None)
        self.dream_tab.set_available_settings_presets([], None)
        if initial_project_dir is not None:
            self.load_project(initial_project_dir)
        else:
            self.project_setup_tab.set_project_selected(False)
            self.project_setup_tab.draw_component_plot(None)
            self.project_setup_tab.draw_prior_plot(None)
            self.prefit_tab.plot_evaluation(None)
            self.dream_tab.clear_plots()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXShell (saxs)")
        self.resize(self._default_window_size())

        self._build_menu_bar()
        self.tabs = QTabWidget()
        self.project_setup_tab = ProjectSetupTab()
        self.prefit_tab = PrefitTab()
        self.dream_tab = DreamTab()
        self.tabs.addTab(self.project_setup_tab, "Project Setup")
        self.tabs.addTab(self.prefit_tab, "SAXS Prefit")
        self.tabs.addTab(self.dream_tab, "SAXS DREAM Fit")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Ready")

        self.project_setup_tab.create_project_requested.connect(
            self.create_project_from_tab
        )
        self.project_setup_tab.open_project_requested.connect(
            self.open_project_from_dialog
        )
        self.project_setup_tab.save_project_requested.connect(
            self.save_project_state
        )
        self.project_setup_tab.autosave_project_requested.connect(
            self._autosave_project_from_tab
        )
        self.project_setup_tab.scan_clusters_requested.connect(
            self.scan_clusters_from_tab
        )
        self.project_setup_tab.build_components_requested.connect(
            self.build_project_components
        )
        self.project_setup_tab.build_prior_weights_requested.connect(
            self.build_prior_weights
        )
        self.project_setup_tab.prior_mode_combo.currentTextChanged.connect(
            lambda _text: self._refresh_prior_plot()
        )
        self.project_setup_tab.generate_prior_plot_requested.connect(
            self.show_prior_histogram_window
        )
        self.project_setup_tab.save_prior_png_requested.connect(
            self.save_prior_plot_png
        )

        self.prefit_tab.template_changed.connect(
            self._on_prefit_template_changed
        )
        self.prefit_tab.autosave_toggled.connect(self._on_autosave_changed)
        self.prefit_tab.update_model_requested.connect(
            self.update_prefit_model
        )
        self.prefit_tab.run_fit_requested.connect(self.run_prefit)
        self.prefit_tab.apply_recommended_scale_requested.connect(
            self.apply_recommended_scale_settings
        )
        self.prefit_tab.set_best_prefit_requested.connect(
            self.set_best_prefit_parameters
        )
        self.prefit_tab.reset_best_prefit_requested.connect(
            self.reset_parameters_to_best_prefit
        )
        self.prefit_tab.save_fit_requested.connect(self.save_prefit)
        self.prefit_tab.restore_state_requested.connect(
            self.restore_prefit_state
        )
        self.prefit_tab.reset_requested.connect(self.reset_prefit_entries)

        self.dream_tab.edit_parameter_map_requested.connect(
            self.open_distribution_editor
        )
        self.dream_tab.save_settings_requested.connect(
            self.save_dream_settings
        )
        self.dream_tab.write_runtime_requested.connect(self.write_dream_bundle)
        self.dream_tab.preview_runtime_requested.connect(
            self.preview_dream_runtime_bundle
        )
        self.dream_tab.run_dream_requested.connect(self.run_dream_bundle)
        self.dream_tab.load_results_requested.connect(self.load_latest_results)
        self.dream_tab.save_report_requested.connect(self.save_dream_report)
        self.dream_tab.save_model_fit_requested.connect(
            self.save_dream_model_fit
        )
        self.dream_tab.save_violin_data_requested.connect(
            self.save_dream_violin_data
        )
        self.dream_tab.settings_preset_changed.connect(
            self._on_dream_settings_preset_changed
        )
        self.dream_tab.visualization_settings_changed.connect(
            self._refresh_loaded_dream_results
        )
        self._refresh_recent_projects_menu()
        self._update_file_menu_state()

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("File")
        self.create_project_action = QAction("Create Project", self)
        self.create_project_action.triggered.connect(
            self._create_project_from_menu
        )
        self.file_menu.addAction(self.create_project_action)

        self.open_project_action = QAction("Open Existing Project...", self)
        self.open_project_action.triggered.connect(
            self._open_project_from_menu
        )
        self.file_menu.addAction(self.open_project_action)

        self.open_recent_menu = self.file_menu.addMenu("Open Recent Project")

        self.save_project_action = QAction("Save Project", self)
        save_shortcuts = QKeySequence.keyBindings(
            QKeySequence.StandardKey.Save
        )
        if save_shortcuts:
            self.save_project_action.setShortcuts(save_shortcuts)
        self.save_project_action.triggered.connect(self.save_project_state)
        self.file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction("Save Project As...", self)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.file_menu.addAction(self.save_project_as_action)

        self.tools_menu = menu_bar.addMenu("Tools")
        self.mdtrajectory_action = QAction("Open mdtrajectory", self)
        self.mdtrajectory_action.triggered.connect(
            self._open_mdtrajectory_tool
        )
        self.tools_menu.addAction(self.mdtrajectory_action)

        self.cluster_action = QAction("Open Cluster Extraction", self)
        self.cluster_action.triggered.connect(self._open_cluster_tool)
        self.tools_menu.addAction(self.cluster_action)

        self.xyz2pdb_action = QAction("Open xyz2pdb Conversion", self)
        self.xyz2pdb_action.triggered.connect(self._open_xyz2pdb_tool)
        self.tools_menu.addAction(self.xyz2pdb_action)

        self.bondanalysis_action = QAction("Open Bond Analysis", self)
        self.bondanalysis_action.triggered.connect(
            self._open_bondanalysis_tool
        )
        self.tools_menu.addAction(self.bondanalysis_action)

        self.tools_menu.addSeparator()
        placeholder_specs = [
            ("PDF Calculation (Coming Soon)", "PDF Calculation"),
            (
                "Number Density Estimate (Coming Soon)",
                "Number Density Estimate",
            ),
            (
                "Volume Fraction Estimate (Coming Soon)",
                "Volume Fraction Estimate",
            ),
            (
                "Bond Association/Dissociation Analysis (Coming Soon)",
                "Bond Association/Dissociation Analysis",
            ),
            ("fullrmc Setup (Coming Soon)", "fullrmc Setup"),
        ]
        self._placeholder_tool_actions: list[QAction] = []
        for label, tool_name in placeholder_specs:
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked=False, name=tool_name: self._show_placeholder_tool_message(
                    name
                )
            )
            self.tools_menu.addAction(action)
            self._placeholder_tool_actions.append(action)

        self.settings_menu = menu_bar.addMenu("Settings")
        self.dream_output_settings_action = QAction(
            "DREAM Output Settings...",
            self,
        )
        self.dream_output_settings_action.triggered.connect(
            self._open_dream_output_settings_dialog
        )
        self.settings_menu.addAction(self.dream_output_settings_action)

        self.help_menu = menu_bar.addMenu("Help")
        self.version_info_action = QAction("Version Information", self)
        self.version_info_action.triggered.connect(
            self._show_version_information
        )
        self.help_menu.addAction(self.version_info_action)

        self.github_action = QAction("Open GitHub Repository", self)
        self.github_action.triggered.connect(self._open_github_repository)
        self.help_menu.addAction(self.github_action)

        self.contact_action = QAction("Contact Developer", self)
        self.contact_action.triggered.connect(self._show_contact_information)
        self.help_menu.addAction(self.contact_action)

    def _resolve_base_font_point_size(self) -> float:
        font = self.font()
        point_size = font.pointSizeF()
        if point_size <= 0:
            app = QApplication.instance()
            if app is not None:
                point_size = app.font().pointSizeF()
        return point_size if point_size > 0 else 12.0

    def _register_scale_shortcuts(self) -> None:
        shortcut_map = [
            ("Meta+=", lambda: self._adjust_ui_scale(0.1)),
            ("Meta++", lambda: self._adjust_ui_scale(0.1)),
            ("Ctrl+=", lambda: self._adjust_ui_scale(0.1)),
            ("Ctrl++", lambda: self._adjust_ui_scale(0.1)),
            ("Meta+-", lambda: self._adjust_ui_scale(-0.1)),
            ("Ctrl+-", lambda: self._adjust_ui_scale(-0.1)),
            ("Meta+0", self._reset_ui_scale),
            ("Ctrl+0", self._reset_ui_scale),
        ]
        for sequence, handler in shortcut_map:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self._scale_shortcuts.append(shortcut)

    def _adjust_ui_scale(self, delta: float) -> None:
        self._set_ui_scale(self._ui_scale + delta)

    def _reset_ui_scale(self) -> None:
        self._set_ui_scale(1.0)

    def _set_ui_scale(self, scale: float) -> None:
        bounded = max(0.7, min(1.6, round(float(scale), 2)))
        if abs(bounded - self._ui_scale) < 1e-9:
            return
        self._ui_scale = bounded
        self._apply_ui_scale(announce=True)

    def _apply_ui_scale(self, *, announce: bool) -> None:
        scaled_font = self.font()
        scaled_font.setPointSizeF(self._base_font_point_size * self._ui_scale)
        self.setFont(scaled_font)
        self._apply_scale_to_widget_tree(self)
        if announce:
            self.statusBar().showMessage(
                f"Interface scale: {int(round(self._ui_scale * 100))}%"
            )

    def _capture_scale_baselines(self, widget: QWidget) -> None:
        if widget.property("_saxs_base_min_width") is None:
            widget.setProperty("_saxs_base_min_width", widget.minimumWidth())
            widget.setProperty("_saxs_base_min_height", widget.minimumHeight())
        if isinstance(widget, QSplitter):
            if widget.property("_saxs_base_handle_width") is None:
                widget.setProperty(
                    "_saxs_base_handle_width", widget.handleWidth()
                )
        for child in widget.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            self._capture_scale_baselines(child)

    def _apply_scale_to_widget_tree(self, widget: QWidget) -> None:
        base_min_width = widget.property("_saxs_base_min_width")
        base_min_height = widget.property("_saxs_base_min_height")
        if isinstance(base_min_width, int) and base_min_width > 0:
            widget.setMinimumWidth(
                max(1, round(base_min_width * self._ui_scale))
            )
        if isinstance(base_min_height, int) and base_min_height > 0:
            widget.setMinimumHeight(
                max(1, round(base_min_height * self._ui_scale))
            )
        if isinstance(widget, QSplitter):
            base_handle_width = widget.property("_saxs_base_handle_width")
            if isinstance(base_handle_width, int) and base_handle_width > 0:
                widget.setHandleWidth(
                    max(2, round(base_handle_width * self._ui_scale))
                )
        for child in widget.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            self._apply_scale_to_widget_tree(child)

    def _default_window_size(self) -> QSize:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QSize(1360, 860)

        available = screen.availableGeometry()
        target_width = min(1360, max(1180, available.width() - 120))
        target_height = min(860, max(760, available.height() - 140))
        return QSize(target_width, target_height)

    def create_project_from_tab(self) -> None:
        try:
            project_dir = self.project_setup_tab.project_dir()
            if project_dir is None:
                self._show_error(
                    "Create project failed",
                    "Select a project directory and enter a project folder name first.",
                )
                return
            if project_dir.exists():
                if not project_dir.is_dir():
                    self._show_error(
                        "Create project failed",
                        "The selected project path already exists and is not a directory.",
                    )
                    return
                response = QMessageBox.warning(
                    self,
                    "Project folder already exists",
                    (
                        f"The folder\n{project_dir}\n\nalready exists. "
                        "Creating this project can overwrite the existing "
                        "SAXS project files in that folder. Do you want to continue?"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if response != QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage("Project creation canceled")
                    return
            project_name = project_dir.name
            settings = self.project_manager.create_project(
                project_dir,
                project_name=project_name,
            )
            self.current_settings = settings
            self._apply_project_settings(settings)
            self._remember_recent_project(settings.project_dir)
            self.project_setup_tab.append_summary(
                f"Created project {settings.project_name} at {settings.project_dir}"
            )
            self.statusBar().showMessage("Project created")
        except Exception as exc:
            self._show_error("Create project failed", str(exc))

    def _create_project_from_menu(self) -> None:
        self.tabs.setCurrentWidget(self.project_setup_tab)
        self.create_project_from_tab()

    def open_project_from_dialog(self) -> None:
        try:
            selected_path = self.project_setup_tab.open_project_dir()
            if selected_path is not None:
                self.load_project(self._validated_project_dir(selected_path))
                return
            selected = QFileDialog.getExistingDirectory(
                self,
                "Open SAXS project folder",
                str(Path.home()),
            )
            if selected:
                self.load_project(self._validated_project_dir(selected))
        except Exception as exc:
            self._show_error("Open project failed", str(exc))

    def _open_project_from_menu(self) -> None:
        try:
            start_dir = (
                self.current_settings.project_dir
                if self.current_settings is not None
                else str(Path.home())
            )
            selected = QFileDialog.getExistingDirectory(
                self,
                "Open SAXS project folder",
                start_dir,
            )
            if selected:
                self.tabs.setCurrentWidget(self.project_setup_tab)
                self.load_project(self._validated_project_dir(selected))
        except Exception as exc:
            self._show_error("Open project failed", str(exc))

    def load_project(self, project_dir: str | Path) -> None:
        settings = self.project_manager.load_project(project_dir)
        self.current_settings = settings
        self._apply_project_settings(settings)
        self._remember_recent_project(settings.project_dir)
        if settings.clusters_dir:
            self.project_setup_tab.request_cluster_scan()
        self.project_setup_tab.append_summary(
            f"Loaded project {settings.project_name}"
        )
        self.statusBar().showMessage("Project loaded")

    def scan_clusters_from_tab(self) -> None:
        clusters_dir = self.project_setup_tab.clusters_dir()
        if clusters_dir is None:
            self._show_error(
                "Cluster import failed",
                "Select a clusters directory first.",
            )
            return
        self._start_project_task(
            "scan_clusters",
            lambda progress: self.project_manager.scan_cluster_inventory(
                clusters_dir,
                progress_callback=progress,
            ),
            start_message="Importing cluster files...",
        )

    def build_project_components(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            self._save_settings(
                settings,
                status_message="Project auto-saved before building SAXS components",
            )
            self.current_settings = settings
            self._start_project_task(
                "build_components",
                lambda progress: (
                    settings,
                    self.project_manager.build_scattering_components(
                        settings,
                        progress_callback=progress,
                    ),
                ),
                start_message="Building SAXS components...",
                settings=settings,
            )
        except Exception as exc:
            self._show_error("Build failed", str(exc))

    def build_prior_weights(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            self._save_settings(
                settings,
                status_message="Project auto-saved before generating prior weights",
            )
            self.current_settings = settings
            self._start_project_task(
                "build_prior_weights",
                lambda progress: (
                    settings,
                    self.project_manager.generate_prior_weights(
                        settings,
                        progress_callback=progress,
                    ),
                ),
                start_message="Generating prior weights...",
                settings=settings,
            )
        except Exception as exc:
            self._show_error("Generate prior weights failed", str(exc))

    def save_project_state(self) -> None:
        try:
            settings = self._settings_from_project_tab()
            saved_path = self._save_settings(settings)
            self.project_setup_tab.append_summary(
                f"Saved project state to {saved_path}"
            )
            self.statusBar().showMessage("Project state saved")
        except Exception as exc:
            self._show_error("Save project state failed", str(exc))

    @Slot(str)
    def _autosave_project_from_tab(self, reason: str) -> None:
        if self.project_setup_tab.project_dir() is None:
            return
        try:
            settings = self._settings_from_project_tab()
            self._save_settings(
                settings,
                status_message=f"Project auto-saved after {reason}",
            )
        except Exception:
            # Autosave should never interrupt the user's file-selection flow.
            return

    def _save_settings(
        self,
        settings: ProjectSettings,
        *,
        status_message: str | None = None,
    ) -> Path:
        self.current_settings = settings
        saved_path = self.project_manager.save_project(settings)
        self._remember_recent_project(settings.project_dir)
        if status_message is not None:
            self.statusBar().showMessage(status_message)
        self._update_file_menu_state()
        return saved_path

    def save_project_as(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save Project As failed",
                "Load or create a project first.",
            )
            return
        try:
            current_settings = self._settings_from_project_tab()
            source_dir = (
                Path(current_settings.project_dir).expanduser().resolve()
            )
            if not source_dir.is_dir():
                raise ValueError(
                    "The active project folder could not be found on disk."
                )
            self._save_settings(
                current_settings,
                status_message="Project saved before Save Project As",
            )
            parent_dir = QFileDialog.getExistingDirectory(
                self,
                "Select destination parent folder",
                str(source_dir.parent),
            )
            if not parent_dir:
                self.statusBar().showMessage("Save Project As canceled")
                return
            project_name, accepted = QInputDialog.getText(
                self,
                "Save Project As",
                "New project folder name:",
                text=f"{source_dir.name}_copy",
            )
            project_name = str(project_name).strip()
            if not accepted or not project_name:
                self.statusBar().showMessage("Save Project As canceled")
                return
            destination_dir = (
                Path(parent_dir).expanduser().resolve() / project_name
            )
            if destination_dir.exists():
                raise ValueError(
                    "The selected Save Project As destination already exists. "
                    "Choose a new folder name."
                )
            shutil.copytree(source_dir, destination_dir)
            new_settings = ProjectSettings.from_dict(
                current_settings.to_dict()
            )
            new_settings.project_name = destination_dir.name
            new_settings.project_dir = str(destination_dir)
            self._remap_copied_project_paths(
                new_settings,
                old_project_dir=source_dir,
                new_project_dir=destination_dir,
            )
            saved_path = self._save_settings(
                new_settings,
                status_message="Project saved to new folder",
            )
            self.current_settings = new_settings
            self._apply_project_settings(new_settings)
            self.tabs.setCurrentWidget(self.project_setup_tab)
            self.project_setup_tab.append_summary(
                f"Saved project as {destination_dir}\nProject file: {saved_path}"
            )
            self.statusBar().showMessage("Project saved to new folder")
        except Exception as exc:
            self._show_error("Save Project As failed", str(exc))

    def _start_project_task(
        self,
        task_name: str,
        task_fn: Callable[[Callable[[int, int, str], None]], object],
        *,
        start_message: str,
        settings: ProjectSettings | None = None,
    ) -> None:
        if self._task_thread is not None:
            self.statusBar().showMessage(
                "A SAXS project task is already running."
            )
            return

        self._active_task_name = task_name
        self._active_task_settings = settings
        self._task_thread = QThread(self)
        self._task_worker = SAXSProjectTaskWorker(task_name, task_fn)
        self._task_worker.moveToThread(self._task_thread)
        self._task_thread.started.connect(self._task_worker.run)
        self._task_worker.progress.connect(self._on_task_progress)
        self._task_worker.finished.connect(self._on_task_finished)
        self._task_worker.failed.connect(self._on_task_failed)
        self._task_worker.finished.connect(self._task_thread.quit)
        self._task_worker.failed.connect(self._task_thread.quit)
        self._task_thread.finished.connect(self._cleanup_task_thread)

        self.project_setup_tab.start_activity_progress(1, start_message)
        self._show_progress_dialog(1, start_message)
        self.statusBar().showMessage(start_message)
        self._task_thread.start()

    @Slot(int, int, str)
    def _on_task_progress(
        self,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        self.project_setup_tab.update_activity_progress(
            processed,
            total,
            message,
        )
        if self._progress_dialog is not None:
            self._progress_dialog.update_progress(
                processed,
                total,
                message,
            )
        self.statusBar().showMessage(message)

    @Slot(str, object)
    def _on_task_finished(self, task_name: str, result: object) -> None:
        if task_name == "scan_clusters":
            cluster_result = result
            if isinstance(cluster_result, ClusterImportResult):
                self.project_setup_tab.apply_cluster_import_data(
                    cluster_result.available_elements,
                    cluster_result.cluster_rows,
                )
                if self.current_settings is not None:
                    self.current_settings.available_elements = (
                        cluster_result.available_elements
                    )
                self.project_setup_tab.append_summary(
                    "Imported cluster files for project setup.\n"
                    f"Files scanned: {cluster_result.total_files}\n"
                    f"Recognized cluster bins: {len(cluster_result.cluster_rows)}"
                )
                self.project_setup_tab.finish_activity_progress(
                    "Cluster import complete."
                )
                self.statusBar().showMessage("Cluster import complete")
        elif task_name == "build_components":
            settings, build_result = result
            self.current_settings = settings
            self._apply_project_settings(settings)
            self.project_setup_tab.apply_cluster_import_data(
                settings.available_elements,
                build_result.cluster_rows,
            )
            self.project_setup_tab.append_summary(
                "Built SAXS components for "
                f"{len(build_result.component_entries)} cluster bins.\n"
                f"Saved component map: {build_result.model_map_path}\n"
                "Generate prior weights separately when you are ready."
            )
            self._refresh_component_plot()
            try:
                self._load_prefit_workflow()
                self._load_dream_workflow()
            except Exception:
                pass
            self.project_setup_tab.finish_activity_progress(
                "SAXS component build complete."
            )
            self.statusBar().showMessage("Project components built")
        elif task_name == "build_prior_weights":
            settings, build_result = result
            self.current_settings = settings
            self._apply_project_settings(settings)
            self.project_setup_tab.apply_cluster_import_data(
                settings.available_elements,
                build_result.cluster_rows,
            )
            self.project_setup_tab.append_summary(
                "Generated prior weights for "
                f"{len(build_result.component_entries)} cluster bins.\n"
                f"Saved prior weights: {build_result.md_prior_weights_path}\n"
                f"Saved prior plot data: {build_result.prior_plot_data_path}"
            )
            self._refresh_prior_plot()
            self.project_setup_tab.finish_activity_progress(
                "Prior-weight generation complete."
            )
            self.statusBar().showMessage("Prior weights generated")
        self._close_progress_dialog()

    @Slot(str, str)
    def _on_task_failed(self, task_name: str, message: str) -> None:
        del task_name
        self.project_setup_tab.finish_activity_progress("Progress: failed")
        self._close_progress_dialog()
        self._show_error("SAXS project task failed", message)

    def _cleanup_task_thread(self) -> None:
        if self._task_worker is not None:
            self._task_worker.deleteLater()
            self._task_worker = None
        if self._task_thread is not None:
            self._task_thread.deleteLater()
            self._task_thread = None
        self._active_task_name = None
        self._active_task_settings = None

    def _ensure_progress_dialog(self) -> SAXSProgressDialog:
        if self._progress_dialog is None:
            self._progress_dialog = SAXSProgressDialog(self)
        return self._progress_dialog

    def _show_progress_dialog(self, total: int, message: str) -> None:
        dialog = self._ensure_progress_dialog()
        dialog.begin(total, message)

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()

    def _ensure_dream_progress_dialog(self) -> SAXSProgressDialog:
        if self._dream_progress_dialog is None:
            self._dream_progress_dialog = SAXSProgressDialog(self)
        return self._dream_progress_dialog

    def _show_dream_progress_dialog(self, message: str) -> None:
        dialog = self._ensure_dream_progress_dialog()
        dialog.begin_busy(message, title="SAXS DREAM Progress")

    def _close_dream_progress_dialog(self) -> None:
        if self._dream_progress_dialog is not None:
            self._dream_progress_dialog.close()

    def _start_dream_run_task(
        self,
        bundle: DreamRunBundle,
        settings: DreamRunSettings,
    ) -> None:
        if self._dream_task_thread is not None:
            self.statusBar().showMessage(
                "A DREAM refinement is already running."
            )
            return
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")

        self._active_dream_run_settings = self._copy_dream_settings(settings)
        self._dream_task_thread = QThread(self)
        self._dream_task_worker = SAXSDreamRunWorker(
            self.current_settings.project_dir,
            bundle,
            verbose_output_interval_seconds=(
                settings.verbose_output_interval_seconds
            ),
        )
        self._dream_task_worker.moveToThread(self._dream_task_thread)
        self._dream_task_thread.started.connect(self._dream_task_worker.run)
        self._dream_task_worker.status.connect(self._on_dream_run_status)
        self._dream_task_worker.output.connect(self._on_dream_run_output)
        self._dream_task_worker.finished.connect(self._on_dream_run_finished)
        self._dream_task_worker.failed.connect(self._on_dream_run_failed)
        self._dream_task_worker.finished.connect(self._dream_task_thread.quit)
        self._dream_task_worker.failed.connect(self._dream_task_thread.quit)
        self._dream_task_thread.finished.connect(
            self._cleanup_dream_task_thread
        )

        start_message = "DREAM refinement in progress..."
        self.dream_tab.start_progress(start_message)
        self._show_dream_progress_dialog(start_message)
        self.dream_tab.run_button.setEnabled(False)
        self.statusBar().showMessage(start_message)
        self._dream_task_thread.start()

    @Slot(str)
    def _on_dream_run_status(self, message: str) -> None:
        self.dream_tab.append_log(message)
        self.dream_tab.start_progress(message)
        if (
            self._dream_progress_dialog is not None
            and self._dream_progress_dialog.isVisible()
        ):
            self._dream_progress_dialog.message_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot(str, object)
    def _on_dream_run_finished(
        self,
        run_dir: str,
        result: object,
    ) -> None:
        settings = (
            self._active_dream_run_settings
            or self.dream_tab.settings_payload()
        )
        self._last_results_loader = SAXSDreamResultsLoader(
            run_dir,
            burnin_percent=settings.burnin_percent,
        )
        self._refresh_loaded_dream_results()
        result_payload = dict(result) if isinstance(result, dict) else {}
        self.dream_tab.append_log(
            "DREAM run complete.\n"
            f"Run directory: {run_dir}\n"
            f"Samples: {result_payload.get('sampled_params_path', 'unknown')}\n"
            f"Log-posteriors: {result_payload.get('log_ps_path', 'unknown')}"
        )
        self.dream_tab.finish_runtime_output()
        self.dream_tab.finish_progress("DREAM refinement complete.")
        self._close_dream_progress_dialog()
        self.dream_tab.run_button.setEnabled(True)
        self.statusBar().showMessage("DREAM run complete")

    @Slot(str)
    def _on_dream_run_failed(self, message: str) -> None:
        self.dream_tab.finish_runtime_output()
        self.dream_tab.append_log("DREAM run failed.\n" + message)
        self.dream_tab.finish_progress("DREAM refinement failed.")
        self._close_dream_progress_dialog()
        self.dream_tab.run_button.setEnabled(True)
        self._show_error("Run DREAM failed", message)

    @Slot(str)
    def _on_dream_run_output(self, message: str) -> None:
        stripped = message.strip()
        if not stripped:
            return
        self.dream_tab.append_runtime_output(stripped)
        latest_line = next(
            (line for line in reversed(stripped.splitlines()) if line.strip()),
            stripped,
        )
        self.dream_tab.progress_label.setText(latest_line)
        if (
            self._dream_progress_dialog is not None
            and self._dream_progress_dialog.isVisible()
        ):
            self._dream_progress_dialog.message_label.setText(latest_line)
        self.statusBar().showMessage(latest_line)

    def _cleanup_dream_task_thread(self) -> None:
        if self._dream_task_worker is not None:
            self._dream_task_worker.deleteLater()
            self._dream_task_worker = None
        if self._dream_task_thread is not None:
            self._dream_task_thread.deleteLater()
            self._dream_task_thread = None
        self._active_dream_run_settings = None

    def update_prefit_model(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
            evaluation = self.prefit_workflow.evaluate(entries)
            self.prefit_tab.plot_evaluation(evaluation)
            run_config = self.prefit_tab.run_config()
            self.prefit_tab.append_log(
                "Updated model preview.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {run_config.method}\n"
                f"Max nfev: {run_config.max_nfev}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(evaluation)
            )
            self._maybe_append_scale_recommendation(entries)
        except Exception as exc:
            self._show_error("Update model failed", str(exc))

    def run_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            config = self.prefit_tab.run_config()
            self.prefit_tab.append_log(
                "Running prefit.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {config.method}\n"
                f"Max nfev: {config.max_nfev}\n"
                "Autosave fit results: "
                + (
                    "enabled"
                    if self.prefit_workflow.settings.autosave_prefits
                    else "disabled"
                )
            )
            result = self.prefit_workflow.run_fit(
                self.prefit_tab.parameter_entries(),
                method=config.method,
                max_nfev=config.max_nfev,
            )
            self.prefit_tab.populate_parameter_table(result.parameter_entries)
            self.prefit_tab.plot_evaluation(result.evaluation)
            self.prefit_tab.append_log(
                "Fit complete.\n"
                f"Minimizer: {result.method}\n"
                f"Max nfev request: {config.max_nfev}\n"
                f"R^2: {result.r_squared:.6g}\n"
                f"Reduced chi^2: {result.reduced_chi_square:.6g}\n"
                f"Function evals: {result.nfev}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(
                    result.evaluation,
                    fit_result=result,
                    report_path=result.report_path,
                )
            )
            self._refresh_saved_prefit_states(
                selected_name=(
                    result.report_path.parent.name
                    if result.report_path is not None
                    else None
                )
            )
            self._load_dream_workflow()
            self.statusBar().showMessage("Prefit complete")
        except Exception as exc:
            self._show_error("Prefit failed", str(exc))

    def save_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
            evaluation = self.prefit_workflow.evaluate(entries)
            config = self.prefit_tab.run_config()
            report_path = self.prefit_workflow.save_fit(
                entries,
                evaluation=evaluation,
                method=config.method,
                max_nfev=config.max_nfev,
                autosave_prefits=self.prefit_tab.autosave_checkbox.isChecked(),
            )
            self.prefit_tab.plot_evaluation(evaluation)
            self.prefit_tab.append_log(
                "Saved prefit state.\n"
                f"Template: {self.prefit_workflow.template_spec.name}\n"
                f"Minimizer: {config.method}\n"
                f"Max nfev: {config.max_nfev}\n"
                f"Saved report: {report_path}"
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(
                    evaluation,
                    report_path=report_path,
                )
            )
            self._refresh_saved_prefit_states(
                selected_name=report_path.parent.name
            )
            self._load_dream_workflow()
            self.statusBar().showMessage("Prefit saved")
        except Exception as exc:
            self._show_error("Save fit failed", str(exc))

    def reset_prefit_entries(self) -> None:
        if self.prefit_workflow is None:
            return
        self.prefit_workflow.parameter_entries = (
            self.prefit_workflow.load_template_reset_entries()
        )
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        self.prefit_tab.append_log(
            "Reset parameter table to the template-default prefit preset "
            "saved in the project."
        )
        self.update_prefit_model()

    def set_best_prefit_parameters(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
            self.prefit_workflow.parameter_entries = entries
            self.prefit_workflow.save_best_prefit_entries(entries)
            if self.current_settings is not None:
                self.current_settings = self.prefit_workflow.settings
            self.prefit_tab.append_log(
                "Saved the current parameter table as the Best Prefit preset "
                "in the project file."
            )
            self.statusBar().showMessage("Best prefit preset saved")
        except Exception as exc:
            self._show_error("Save Best Prefit failed", str(exc))

    def reset_parameters_to_best_prefit(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            entries = self.prefit_workflow.load_best_prefit_entries()
            if entries is None:
                raise ValueError(
                    "No Best Prefit preset is saved for the active template."
                )
            self.prefit_workflow.parameter_entries = entries
            self.prefit_tab.populate_parameter_table(entries)
            self.prefit_tab.append_log(
                "Reset parameter table to the Best Prefit preset saved in "
                "the project."
            )
            self.update_prefit_model()
            self.statusBar().showMessage("Best prefit preset applied")
        except Exception as exc:
            self._show_error("Reset to Best Prefit failed", str(exc))

    def restore_prefit_state(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        state_name = self.prefit_tab.selected_saved_state_name()
        if not state_name:
            self._show_error(
                "Restore Prefit State failed",
                "Select a saved prefit snapshot folder first.",
            )
            return
        try:
            saved_state = self.prefit_workflow.load_saved_state(state_name)
            if (
                saved_state.template_name
                != self.prefit_workflow.template_spec.name
            ):
                raise ValueError(
                    "The selected prefit snapshot was saved with template "
                    f"{saved_state.template_name}, but the active prefit "
                    f"template is {self.prefit_workflow.template_spec.name}. "
                    "Switch to the matching template before restoring this "
                    "snapshot so the Best Prefit preset remains untouched."
                )
            self.prefit_workflow.parameter_entries = (
                saved_state.parameter_entries
            )
            self.prefit_tab.populate_parameter_table(
                saved_state.parameter_entries
            )
            if saved_state.method and saved_state.max_nfev is not None:
                self.prefit_tab.set_run_config(
                    method=saved_state.method,
                    max_nfev=saved_state.max_nfev,
                )
            if saved_state.autosave_prefits is not None:
                self.prefit_workflow.settings.autosave_prefits = bool(
                    saved_state.autosave_prefits
                )
                if self.current_settings is not None:
                    self.current_settings.autosave_prefits = bool(
                        saved_state.autosave_prefits
                    )
                self.prefit_tab.set_autosave(saved_state.autosave_prefits)
            self.prefit_tab.append_log(
                "Restored prefit snapshot.\n"
                f"Snapshot: {saved_state.name}\n"
                f"Template: {saved_state.template_name}\n"
                f"Minimizer: {saved_state.method or self.prefit_tab.run_config().method}\n"
                "Max nfev: "
                f"{saved_state.max_nfev or self.prefit_tab.run_config().max_nfev}\n"
                "Best Prefit preset was not modified."
            )
            self.update_prefit_model()
            self.statusBar().showMessage("Prefit state restored")
        except Exception as exc:
            self._show_error("Restore Prefit State failed", str(exc))

    def save_dream_settings(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            active_path = workflow.save_settings(settings)
            self._active_dream_settings_snapshot = self._copy_dream_settings(
                settings
            )
            preset_name = self._prompt_dream_settings_preset_name(
                suggested_name=(
                    self.dream_tab.selected_settings_preset_name()
                    or settings.run_label
                    or f"dream_settings_{datetime.now():%Y%m%d_%H%M%S}"
                )
            )
            preset_path = None
            if preset_name:
                preset_path = workflow.save_settings_preset(
                    settings,
                    preset_name,
                )
            self._invalidate_written_dream_bundle()
            preset_names = workflow.list_settings_presets()
            self.dream_tab.set_available_settings_presets(
                preset_names,
                DreamTab.ACTIVE_SETTINGS_LABEL,
            )
            self.dream_tab.set_settings(settings, preset_name=None)
            self._current_dream_preset_name = None
            self.dream_tab.append_log(
                "Saved DREAM settings.\n"
                f"Active settings: {active_path}\n"
                "Preset: "
                + (
                    f"{preset_name} ({preset_path})"
                    if preset_name and preset_path is not None
                    else "no named preset created"
                )
            )
            self.statusBar().showMessage("DREAM settings saved")
        except Exception as exc:
            self._show_error("Save DREAM settings failed", str(exc))

    def open_distribution_editor(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            has_saved_parameter_map = workflow.parameter_map_path.is_file()
            entries = workflow.load_parameter_map(persist_if_missing=False)
            if self.distribution_window is None:
                self.distribution_window = DistributionSetupWindow(
                    entries, self
                )
                self.distribution_window.saved.connect(
                    self._save_distribution_entries
                )
            self.distribution_window.load_entries(
                entries,
                has_existing_parameter_map=has_saved_parameter_map,
            )
            self.distribution_window.show()
            self.distribution_window.raise_()
            self.distribution_window.activateWindow()
        except Exception as exc:
            self._show_error("Open priors failed", str(exc))

    def write_dream_bundle(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            entries = workflow.load_parameter_map()
            self._append_dream_vary_recommendation(entries)
            bundle = workflow.create_runtime_bundle(
                settings=settings,
                entries=entries,
            )
            self._last_written_dream_bundle = bundle
            self.dream_tab.append_log(
                "Wrote DREAM runtime bundle.\n"
                f"Runtime script: {bundle.runtime_script_path}\n"
                f"Best-fit method: {settings.bestfit_method}\n"
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}\n"
                f"Violin data mode: {settings.violin_parameter_mode}\n"
                f"Violin sample source: {settings.violin_sample_source}"
            )
            self.statusBar().showMessage("DREAM runtime bundle written")
        except Exception as exc:
            self._show_error("Write DREAM bundle failed", str(exc))

    def preview_dream_runtime_bundle(self) -> None:
        try:
            script_path = self._resolve_runtime_script_preview_path()
            opener = self._get_or_prompt_runtime_bundle_opener()
            if opener is None:
                self.statusBar().showMessage("Runtime bundle preview canceled")
                return
            self._launch_runtime_bundle_with_opener(script_path, opener)
            self.dream_tab.append_log(
                "Opened DREAM runtime bundle preview: "
                f"{script_path}\nApplication: {opener.label}"
            )
            self.statusBar().showMessage("Opened DREAM runtime bundle")
        except Exception as exc:
            self._show_error("Preview Runtime Bundle failed", str(exc))

    def _get_or_prompt_runtime_bundle_opener(
        self,
    ) -> RuntimeBundleOpener | None:
        if self.current_settings is None:
            raise ValueError(
                "Load a project before previewing a runtime bundle."
            )
        stored_value = (
            self.current_settings.runtime_bundle_opener or ""
        ).strip()
        if stored_value:
            opener = self._runtime_bundle_opener_from_stored_value(
                stored_value
            )
            if opener is not None:
                return opener
        opener = self._prompt_runtime_bundle_opener()
        if opener is None:
            return None
        self.current_settings.runtime_bundle_opener = opener.stored_value
        self.project_manager.save_project(self.current_settings)
        return opener

    def _prompt_runtime_bundle_opener(self) -> RuntimeBundleOpener | None:
        openers = self._available_runtime_bundle_openers()
        labels = [opener.label for opener in openers]
        custom_label = "Choose another application..."
        labels.append(custom_label)
        current_index = 0
        if self.current_settings is not None:
            current_value = (
                self.current_settings.runtime_bundle_opener or ""
            ).strip()
            for index, opener in enumerate(openers):
                if opener.stored_value == current_value:
                    current_index = index
                    break
        selected_label, accepted = QInputDialog.getItem(
            self,
            "Choose Runtime Bundle Opener",
            (
                "Select the application to use when previewing the DREAM "
                "runtime bundle for this project."
            ),
            labels,
            current=current_index,
            editable=False,
        )
        if not accepted or not str(selected_label).strip():
            return None
        if str(selected_label) == custom_label:
            return self._prompt_custom_runtime_bundle_opener()
        for opener in openers:
            if opener.label == str(selected_label):
                return opener
        return None

    def _prompt_custom_runtime_bundle_opener(
        self,
    ) -> RuntimeBundleOpener | None:
        if sys.platform == "darwin":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select application to open the DREAM runtime bundle",
                "/Applications",
            )
            if not selected:
                return None
            selected_path = Path(selected).expanduser().resolve()
            if selected_path.suffix != ".app":
                raise ValueError(
                    "Choose a macOS .app bundle when selecting a custom "
                    "runtime bundle opener."
                )
            return RuntimeBundleOpener(
                label=selected_path.stem,
                stored_value=str(selected_path),
                launch_target=str(selected_path),
                launch_mode="mac_app",
            )
        selected, _file_filter = QFileDialog.getOpenFileName(
            self,
            "Select application to open the DREAM runtime bundle",
            str(Path.home()),
            "Applications (*)",
        )
        if not selected:
            return None
        selected_path = Path(selected).expanduser().resolve()
        return RuntimeBundleOpener(
            label=selected_path.stem or selected_path.name,
            stored_value=str(selected_path),
            launch_target=str(selected_path),
            launch_mode="executable",
        )

    def _available_runtime_bundle_openers(self) -> list[RuntimeBundleOpener]:
        openers: list[RuntimeBundleOpener] = []
        if sys.platform == "darwin":
            candidate_paths = [
                ("TextEdit", "/System/Applications/TextEdit.app"),
                ("Visual Studio Code", "/Applications/Visual Studio Code.app"),
                ("Cursor", "/Applications/Cursor.app"),
                ("Sublime Text", "/Applications/Sublime Text.app"),
                ("BBEdit", "/Applications/BBEdit.app"),
                ("CotEditor", "/Applications/CotEditor.app"),
                ("Xcode", "/Applications/Xcode.app"),
                ("PyCharm CE", "/Applications/PyCharm CE.app"),
                ("PyCharm", "/Applications/PyCharm.app"),
                (
                    "PyCharm Community Edition",
                    "/Applications/PyCharm Community Edition.app",
                ),
            ]
            for label, raw_path in candidate_paths:
                app_path = Path(raw_path).expanduser()
                if not app_path.exists():
                    continue
                openers.append(
                    RuntimeBundleOpener(
                        label=label,
                        stored_value=str(app_path.resolve()),
                        launch_target=str(app_path.resolve()),
                        launch_mode="mac_app",
                    )
                )
            return openers

        command_candidates = [
            ("Visual Studio Code", "code"),
            ("Cursor", "cursor"),
            ("Sublime Text", "subl"),
            ("Kate", "kate"),
            ("Gedit", "gedit"),
            ("Mousepad", "mousepad"),
            ("Geany", "geany"),
            ("Xed", "xed"),
            ("Pluma", "pluma"),
            ("Notepad", "notepad"),
            ("Notepad++", "notepad++"),
        ]
        for label, command in command_candidates:
            executable = shutil.which(command)
            if executable is None:
                continue
            openers.append(
                RuntimeBundleOpener(
                    label=label,
                    stored_value=str(Path(executable).resolve()),
                    launch_target=str(Path(executable).resolve()),
                    launch_mode="executable",
                )
            )
        return openers

    def _runtime_bundle_opener_from_stored_value(
        self,
        stored_value: str,
    ) -> RuntimeBundleOpener | None:
        normalized = str(stored_value).strip()
        if not normalized:
            return None
        for opener in self._available_runtime_bundle_openers():
            if opener.stored_value == normalized:
                return opener
        opener_path = Path(normalized).expanduser()
        if opener_path.exists():
            launch_mode = (
                "mac_app"
                if sys.platform == "darwin" and opener_path.suffix == ".app"
                else "executable"
            )
            return RuntimeBundleOpener(
                label=opener_path.stem or opener_path.name,
                stored_value=str(opener_path.resolve()),
                launch_target=str(opener_path.resolve()),
                launch_mode=launch_mode,
            )
        executable = shutil.which(normalized)
        if executable is None:
            return None
        executable_path = Path(executable).resolve()
        return RuntimeBundleOpener(
            label=executable_path.stem or executable_path.name,
            stored_value=str(executable_path),
            launch_target=str(executable_path),
            launch_mode="executable",
        )

    def _launch_runtime_bundle_with_opener(
        self,
        script_path: Path,
        opener: RuntimeBundleOpener,
    ) -> None:
        if opener.launch_mode == "mac_app":
            subprocess.Popen(
                ["open", "-a", opener.launch_target, str(script_path)]
            )
            return
        subprocess.Popen([opener.launch_target, str(script_path)])

    def run_dream_bundle(self) -> None:
        try:
            if self._dream_task_thread is not None:
                self.statusBar().showMessage(
                    "A DREAM refinement is already running."
                )
                return
            workflow = self._load_dream_workflow()
            settings = self.dream_tab.settings_payload()
            if not settings.model_name:
                settings.model_name = self.prefit_workflow.template_spec.name
            entries = workflow.load_parameter_map()
            if not self._dream_parameter_map_saved_in_session:
                self.dream_tab.blink_edit_priors_button()
                self.dream_tab.append_log(
                    "Run DREAM blocked.\n"
                    "Review the priors in Edit Priors and click Save "
                    "Parameter Map before starting a DREAM refinement."
                )
                self.statusBar().showMessage(
                    "Edit and save the DREAM parameter map first"
                )
                self._append_dream_vary_recommendation(entries)
                return
            if (
                self._last_written_dream_bundle is None
                or not self._last_written_dream_bundle.run_dir.exists()
            ):
                self.dream_tab.blink_write_bundle_button()
                self.dream_tab.append_log(
                    "Run DREAM blocked.\n"
                    "Runtime Bundle not generated. Click Write Runtime "
                    "Bundle before running DREAM."
                )
                self._show_error(
                    "Runtime Bundle not generated",
                    "Runtime Bundle not generated. Click Write Runtime Bundle before running DREAM.",
                )
                return
            self._append_dream_vary_recommendation(entries)
            bundle = self._last_written_dream_bundle
            self.dream_tab.append_log(
                "Running DREAM.\n"
                f"Model name: {settings.model_name}\n"
                f"Chains: {settings.nchains}\n"
                f"Iterations: {settings.niterations}\n"
                f"Burn-in: {settings.burnin_percent}%\n"
                f"Best-fit method: {settings.bestfit_method}\n"
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}\n"
                f"Violin data mode: {settings.violin_parameter_mode}\n"
                f"Violin sample source: {settings.violin_sample_source}"
            )
            self._start_dream_run_task(bundle, settings)
        except Exception as exc:
            self._show_error("Run DREAM failed", str(exc))

    def load_latest_results(self) -> None:
        try:
            workflow = self._load_dream_workflow()
            run_dirs = sorted(
                workflow.paths.dream_runtime_dir.glob("dream_*"),
                key=lambda path: path.name,
            )
            latest_dir = next(
                reversed(
                    [
                        path
                        for path in run_dirs
                        if (path / "dream_sampled_params.npy").is_file()
                        and (path / "dream_log_ps.npy").is_file()
                    ]
                )
            )
            self._last_results_loader = SAXSDreamResultsLoader(
                latest_dir,
                burnin_percent=self.dream_tab.settings_payload().burnin_percent,
            )
            self._refresh_loaded_dream_results()
            self.dream_tab.append_log(
                f"Loaded DREAM results from {latest_dir}"
            )
            self.statusBar().showMessage("DREAM results loaded")
        except StopIteration:
            self._show_error(
                "Load results failed",
                "No DREAM sample files were found in the runtime bundle folder.",
            )
        except Exception as exc:
            self._show_error("Load results failed", str(exc))

    def _resolve_runtime_script_preview_path(self) -> Path:
        if (
            self._last_written_dream_bundle is not None
            and self._last_written_dream_bundle.runtime_script_path.is_file()
        ):
            return self._last_written_dream_bundle.runtime_script_path
        workflow = self._load_dream_workflow()
        run_dirs = sorted(
            workflow.paths.dream_runtime_dir.glob("dream_*"),
            key=lambda path: path.name,
        )
        for run_dir in reversed(run_dirs):
            scripts = sorted(run_dir.glob("*.py"))
            if scripts:
                return scripts[0]
        raise FileNotFoundError(
            "No DREAM runtime bundle script was found. Write a runtime "
            "bundle before previewing it."
        )

    def save_dream_report(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save report failed",
                "Load DREAM results first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            reports_dir = build_project_paths(
                self.current_settings.project_dir
            ).reports_dir
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / (
                f"dream_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            self._last_results_loader.save_statistics_report(
                output_path,
                bestfit_method=settings.bestfit_method,
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
                violin_parameter_mode=settings.violin_parameter_mode,
                violin_sample_source=settings.violin_sample_source,
            )
            self.dream_tab.append_log(
                f"Saved DREAM statistics to {output_path}"
            )
            self.statusBar().showMessage("DREAM statistics saved")
        except Exception as exc:
            self._show_error("Save report failed", str(exc))

    def save_dream_model_fit(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save model fit failed",
                "Load DREAM results first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Save model fit failed",
                "Load or build a project first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            paths = build_project_paths(self.current_settings.project_dir)
            paths.plots_dir.mkdir(parents=True, exist_ok=True)
            output_path = paths.plots_dir / (
                "dream_model_fit_"
                f"{settings.bestfit_method}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            model_plot = self._last_results_loader.build_model_fit_data(
                bestfit_method=settings.bestfit_method,
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
            )
            np.savetxt(
                output_path,
                np.column_stack(
                    [
                        model_plot.q_values,
                        model_plot.experimental_intensities,
                        model_plot.model_intensities,
                    ]
                ),
                delimiter=",",
                header="q,experimental_intensity,model_intensity",
                comments="",
            )
            self.dream_tab.append_log(
                f"Saved DREAM model fit data to {output_path}"
            )
            self.statusBar().showMessage("DREAM model fit data saved")
        except Exception as exc:
            self._show_error("Save model fit failed", str(exc))

    def save_dream_violin_data(self) -> None:
        if self._last_results_loader is None:
            self._show_error(
                "Save violin data failed",
                "Load DREAM results first.",
            )
            return
        if self.current_settings is None:
            self._show_error(
                "Save violin data failed",
                "Load or build a project first.",
            )
            return
        try:
            settings = self.dream_tab.settings_payload()
            paths = build_project_paths(self.current_settings.project_dir)
            paths.plots_dir.mkdir(parents=True, exist_ok=True)
            base_name = (
                "dream_violin_"
                f"{self._effective_dream_violin_mode(settings)}_"
                f"{settings.violin_weight_order}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            dialog = DreamViolinExportDialog(
                default_output_dir=paths.plots_dir,
                default_base_name=base_name,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_options = dialog.selected_options
            if export_options is None:
                return
            export_options.output_dir.mkdir(parents=True, exist_ok=True)
            violin_plot = self._last_results_loader.build_violin_data(
                mode=self._effective_dream_violin_mode(settings),
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
                sample_source=settings.violin_sample_source,
                weight_order=settings.violin_weight_order,
            )
            summary = self._last_results_loader.get_summary(
                bestfit_method=settings.bestfit_method,
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
            )
            plot_payload = self.dream_tab.prepare_violin_plot_payload(
                summary,
                violin_plot,
            )
            saved_paths: list[Path] = []
            if export_options.save_csv:
                csv_output_path = (
                    export_options.output_dir
                    / f"{export_options.base_name}.csv"
                )
                with csv_output_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(plot_payload["display_names"])
                    writer.writerows(
                        np.asarray(plot_payload["samples"], dtype=float)
                    )
                saved_paths.append(csv_output_path)
            if export_options.save_pkl:
                pkl_output_path = (
                    export_options.output_dir
                    / f"{export_options.base_name}.pkl"
                )
                payload = {
                    "exported_at": datetime.now().isoformat(),
                    "project_dir": str(self.current_settings.project_dir),
                    "settings": settings.to_dict(),
                    "summary": {
                        "bestfit_method": summary.bestfit_method,
                        "posterior_filter_mode": summary.posterior_filter_mode,
                        "posterior_sample_count": summary.posterior_sample_count,
                        "credible_interval_low": summary.credible_interval_low,
                        "credible_interval_high": summary.credible_interval_high,
                        "full_parameter_names": list(
                            summary.full_parameter_names
                        ),
                        "active_parameter_names": list(
                            summary.active_parameter_names
                        ),
                        "map_chain": summary.map_chain,
                        "map_step": summary.map_step,
                        "run_dir": str(summary.run_dir),
                    },
                    "violin_plot": {
                        "parameter_names": list(violin_plot.parameter_names),
                        "display_names": list(violin_plot.display_names),
                        "mode": violin_plot.mode,
                        "sample_source": violin_plot.sample_source,
                        "sample_count": violin_plot.sample_count,
                        "weight_order": violin_plot.weight_order,
                    },
                    "plot_payload": {
                        "display_names": list(plot_payload["display_names"]),
                        "samples": np.asarray(
                            plot_payload["samples"],
                            dtype=float,
                        ),
                        "selected_values": np.asarray(
                            plot_payload["selected_values"],
                            dtype=float,
                        ),
                        "interval_low_values": np.asarray(
                            plot_payload["interval_low_values"],
                            dtype=float,
                        ),
                        "interval_high_values": np.asarray(
                            plot_payload["interval_high_values"],
                            dtype=float,
                        ),
                        "ylabel": str(plot_payload["ylabel"]),
                        "title": str(plot_payload["title"]),
                        "y_limits": plot_payload["y_limits"],
                    },
                }
                with pkl_output_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                saved_paths.append(pkl_output_path)
            self.dream_tab.append_log(
                "Saved DREAM violin plot data to:\n"
                + "\n".join(str(path) for path in saved_paths)
            )
            self.statusBar().showMessage("DREAM violin data saved")
        except Exception as exc:
            self._show_error("Save violin data failed", str(exc))

    @staticmethod
    def _effective_dream_violin_mode(settings: DreamRunSettings) -> str:
        if settings.violin_value_scale_mode == "weights_unit_interval":
            return "weights_only"
        if settings.violin_value_scale_mode == "normalized_all":
            return "all_parameters"
        return settings.violin_parameter_mode

    def save_prior_plot_data_as(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save prior plot data failed",
                "Load or build a project first.",
            )
            return
        try:
            paths = build_project_paths(self.current_settings.project_dir)
            source = paths.plots_dir / "prior_histogram_data.json"
            if not source.is_file():
                raise FileNotFoundError(
                    "No prior_histogram_data.json file was found. Generate "
                    "prior weights first."
                )
            destination, _ = QFileDialog.getSaveFileName(
                self,
                "Save prior histogram data",
                str(source),
                "JSON files (*.json)",
            )
            if destination:
                Path(destination).write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                self.project_setup_tab.append_summary(
                    f"Saved prior histogram data to {destination}"
                )
        except Exception as exc:
            self._show_error("Save prior data failed", str(exc))

    def _apply_project_settings(self, settings: ProjectSettings) -> None:
        template_specs = list_template_specs()
        self._last_results_loader = None
        self._dream_workflow_project_dir = None
        self._dream_parameter_map_saved_in_session = False
        self._active_dream_settings_snapshot = None
        self._current_dream_preset_name = None
        self.dream_tab.reset_progress()
        self.project_setup_tab.set_project_selected(True)
        self.project_setup_tab.set_project_settings(settings, template_specs)
        self._refresh_component_plot()
        self._refresh_prior_plot()
        try:
            self._load_prefit_workflow()
        except Exception as exc:
            self.prefit_workflow = None
            self.prefit_tab.set_templates(template_specs, None)
            self.prefit_tab.plot_evaluation(None)
            self.prefit_tab.set_log_text(
                "Prefit workflow is not ready yet.\n" f"{exc}"
            )
            self.prefit_tab.set_summary_text(
                "Prefit summary is not available yet.\n" f"{exc}"
            )
        try:
            self._load_dream_workflow()
        except Exception as exc:
            self.dream_workflow = None
            self.dream_tab.set_log_text(
                "DREAM workflow is not ready yet.\n" f"{exc}"
            )
            self.dream_tab.set_summary_text(
                "DREAM summary is not available yet.\n" f"{exc}"
            )
            self.dream_tab.clear_plots()
        self._update_file_menu_state()

    def _settings_from_project_tab(self) -> ProjectSettings:
        project_dir = self.project_setup_tab.project_dir()
        if project_dir is None:
            raise ValueError("Select a project directory.")
        base = (
            self.current_settings
            if self.current_settings is not None
            else ProjectSettings(
                project_name=project_dir.name,
                project_dir=str(project_dir),
            )
        )
        base.project_name = project_dir.name
        base.project_dir = str(project_dir)
        base.clusters_dir = (
            str(self.project_setup_tab.clusters_dir())
            if self.project_setup_tab.clusters_dir() is not None
            else None
        )
        base.experimental_data_path = (
            str(self.project_setup_tab.experimental_data_path())
            if self.project_setup_tab.experimental_data_path() is not None
            else None
        )
        base.copied_experimental_data_file = None
        base.solvent_data_path = (
            str(self.project_setup_tab.solvent_data_path())
            if self.project_setup_tab.solvent_data_path() is not None
            else None
        )
        base.copied_solvent_data_file = None
        base.experimental_header_rows = (
            self.project_setup_tab.experimental_header_rows()
        )
        base.experimental_q_column = (
            self.project_setup_tab.experimental_q_column()
        )
        base.experimental_intensity_column = (
            self.project_setup_tab.experimental_intensity_column()
        )
        base.experimental_error_column = (
            self.project_setup_tab.experimental_error_column()
        )
        base.solvent_header_rows = self.project_setup_tab.solvent_header_rows()
        base.solvent_q_column = self.project_setup_tab.solvent_q_column()
        base.solvent_intensity_column = (
            self.project_setup_tab.solvent_intensity_column()
        )
        base.solvent_error_column = (
            self.project_setup_tab.solvent_error_column()
        )
        base.q_min = self.project_setup_tab.q_min()
        base.q_max = self.project_setup_tab.q_max()
        base.use_experimental_grid = (
            self.project_setup_tab.use_experimental_grid()
        )
        base.q_points = self.project_setup_tab.q_points()
        base.available_elements = self.project_setup_tab.available_elements()
        base.include_elements = []
        base.exclude_elements = self.project_setup_tab.exclude_elements()
        base.component_trace_colors = (
            self.project_setup_tab.component_trace_colors()
        )
        base.experimental_trace_visible = (
            self.project_setup_tab.experimental_trace_visible()
        )
        base.experimental_trace_color = (
            self.project_setup_tab.experimental_trace_color()
        )
        base.solvent_trace_visible = (
            self.project_setup_tab.solvent_trace_visible()
        )
        base.solvent_trace_color = self.project_setup_tab.solvent_trace_color()
        base.selected_model_template = (
            self.project_setup_tab.selected_template_name()
        )
        return base

    def _load_prefit_workflow(self) -> SAXSPrefitWorkflow:
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")
        self.prefit_workflow = SAXSPrefitWorkflow(
            self.current_settings.project_dir
        )
        self.current_settings = self.prefit_workflow.settings
        template_specs = list_template_specs()
        self.prefit_tab.set_templates(
            template_specs,
            self.prefit_workflow.template_spec.name,
        )
        self.prefit_tab.set_autosave(
            self.prefit_workflow.settings.autosave_prefits
        )
        self._restoring_prefit_template = True
        self.prefit_tab.set_selected_template(
            self.prefit_workflow.template_spec.name
        )
        self._restoring_prefit_template = False
        self.prefit_tab.populate_parameter_table(
            self.prefit_workflow.parameter_entries
        )
        evaluation = self.prefit_workflow.evaluate()
        self.prefit_tab.plot_evaluation(evaluation)
        self.prefit_tab.set_log_text(self._format_prefit_console_intro())
        self.prefit_tab.set_summary_text(
            self._format_prefit_summary(evaluation)
        )
        if self.prefit_workflow.has_best_prefit_entries():
            self.prefit_tab.append_log(
                "Loaded the Best Prefit preset from the project file."
            )
        self._refresh_saved_prefit_states()
        self._maybe_append_scale_recommendation(
            self.prefit_workflow.parameter_entries
        )
        return self.prefit_workflow

    def _load_dream_workflow(self) -> SAXSDreamWorkflow:
        if self.current_settings is None:
            raise ValueError("No project is currently loaded.")
        project_dir = str(Path(self.current_settings.project_dir).resolve())
        is_new_project = self._dream_workflow_project_dir != project_dir
        if self.dream_workflow is None or is_new_project:
            self.dream_workflow = SAXSDreamWorkflow(project_dir)
        self._dream_workflow_project_dir = project_dir
        selected_preset = self.dream_tab.selected_settings_preset_name()
        preset_names = self.dream_workflow.list_settings_presets()
        if selected_preset not in preset_names:
            selected_preset = None
        self.dream_tab.set_available_settings_presets(
            preset_names,
            selected_preset,
        )
        if not is_new_project:
            self._current_dream_preset_name = selected_preset
        if is_new_project:
            self._invalidate_written_dream_bundle()
            settings = self.dream_workflow.load_settings_preset(
                selected_preset
            )
            self.dream_tab.set_settings(settings, preset_name=selected_preset)
            self._active_dream_settings_snapshot = self._copy_dream_settings(
                settings
            )
            self._current_dream_preset_name = selected_preset
            try:
                parameter_map_entries = self.dream_workflow.load_parameter_map(
                    persist_if_missing=False
                )
            except Exception:
                parameter_map_entries = []
            self.dream_tab.set_parameter_map_entries(parameter_map_entries)
            self._dream_parameter_map_saved_in_session = False
            self.dream_tab.set_log_text(self._format_dream_console_intro())
            self.dream_tab.set_summary_text(
                "DREAM results are not loaded yet."
            )
            self.dream_tab.reset_progress()
            self.dream_tab.clear_plots()
        return self.dream_workflow

    def _refresh_prior_plot(self) -> None:
        if self.current_settings is None:
            self.project_setup_tab.draw_prior_plot(None)
            return
        paths = build_project_paths(self.current_settings.project_dir)
        prior_json = paths.project_dir / "md_prior_weights.json"
        self.project_setup_tab.draw_prior_plot(
            prior_json if prior_json.is_file() else None
        )

    def _refresh_component_plot(self) -> None:
        if self.current_settings is None:
            self.project_setup_tab.draw_component_plot(None)
            return
        paths = build_project_paths(self.current_settings.project_dir)
        component_paths = sorted(paths.scattering_components_dir.glob("*.txt"))
        self.project_setup_tab.draw_component_plot(component_paths or None)

    def _refresh_saved_prefit_states(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        if self.prefit_workflow is None:
            self.prefit_tab.set_saved_states([], None)
            return
        self.prefit_tab.set_saved_states(
            self.prefit_workflow.list_saved_states(),
            selected_name=selected_name,
        )

    def _save_distribution_entries(self, entries: list) -> None:
        try:
            workflow = self._load_dream_workflow()
            workflow.save_parameter_map(entries)
            self._dream_parameter_map_saved_in_session = True
            self._invalidate_written_dream_bundle()
            self.dream_tab.set_parameter_map_entries(entries)
            self.dream_tab.append_log(
                "Updated DREAM parameter map.\n"
                "The current DREAM session is now allowed to write or run a "
                "bundle with these priors."
            )
            self._append_dream_vary_recommendation(entries)
        except Exception as exc:
            self._show_error("Save priors failed", str(exc))

    def _on_prefit_template_changed(self, template_name: str) -> None:
        if (
            self.prefit_workflow is None
            or not template_name
            or self._restoring_prefit_template
        ):
            return
        current_name = self.prefit_workflow.template_spec.name
        if template_name == current_name:
            return
        should_continue, disable_future_warnings = (
            self._confirm_prefit_template_change(current_name, template_name)
        )
        if not should_continue:
            self._restore_prefit_template_selection(current_name)
            return
        if disable_future_warnings:
            self._warn_on_prefit_template_change = False
        try:
            self.prefit_workflow.set_template(template_name)
            if self.current_settings is not None:
                self.current_settings = self.prefit_workflow.settings
            self.project_setup_tab.set_available_templates(
                list_template_specs(),
                template_name,
            )
            self.prefit_tab.populate_parameter_table(
                self.prefit_workflow.parameter_entries
            )
            evaluation = self.prefit_workflow.evaluate()
            self.prefit_tab.plot_evaluation(evaluation)
            self.prefit_tab.append_log(
                "Prefit template changed.\n"
                f"Previous template: {current_name}\n"
                f"Current template: {template_name}\n"
                "The parameter table was reset to the selected template "
                "defaults, the template reset preset was updated, and any "
                "saved Best Prefit preset from the previous template was "
                "cleared."
            )
            self.prefit_tab.set_summary_text(
                self._format_prefit_summary(evaluation)
            )
            self._maybe_append_scale_recommendation()
            self.statusBar().showMessage("Prefit template updated")
        except Exception as exc:
            self._restore_prefit_template_selection(current_name)
            self._show_error("Template change failed", str(exc))

    def _on_autosave_changed(self, enabled: bool) -> None:
        if self.prefit_workflow is None:
            return
        self.prefit_workflow.set_autosave(enabled)
        if self.current_settings is not None:
            self.current_settings = self.prefit_workflow.settings
        self.prefit_tab.append_log(
            "Autosave fit results " + ("enabled." if enabled else "disabled.")
        )

    def _on_dream_settings_preset_changed(self, _text: str) -> None:
        if self.dream_workflow is None:
            return
        try:
            preset_name = self.dream_tab.selected_settings_preset_name()
            previous_preset_name = self._current_dream_preset_name
            if previous_preset_name is None and preset_name is not None:
                self._active_dream_settings_snapshot = (
                    self._copy_dream_settings(
                        self.dream_tab.settings_payload()
                    )
                )
            if preset_name is None:
                settings = (
                    self._copy_dream_settings(
                        self._active_dream_settings_snapshot
                    )
                    if self._active_dream_settings_snapshot is not None
                    else self.dream_workflow.load_settings()
                )
            else:
                settings = self.dream_workflow.load_settings_preset(
                    preset_name
                )
            self.dream_tab.set_settings(settings, preset_name=preset_name)
            self._current_dream_preset_name = preset_name
            self._invalidate_written_dream_bundle()
            self.dream_tab.append_log(
                "Loaded DREAM settings preset: "
                f"{preset_name or 'active project settings'}"
            )
            self._refresh_loaded_dream_results()
        except Exception as exc:
            self._show_error("Load DREAM settings failed", str(exc))

    @staticmethod
    def _copy_dream_settings(
        settings: DreamRunSettings,
    ) -> DreamRunSettings:
        return DreamRunSettings.from_dict(settings.to_dict())

    def _recent_projects_settings(self) -> QSettings:
        return QSettings("SAXShell", "SAXS")

    def _recent_project_paths(self) -> list[str]:
        raw_value = self._recent_projects_settings().value(
            RECENT_PROJECTS_KEY,
            [],
        )
        if isinstance(raw_value, str):
            candidates = [raw_value]
        elif isinstance(raw_value, (list, tuple)):
            candidates = [str(item) for item in raw_value]
        else:
            candidates = []
        existing_paths: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized:
                continue
            if Path(normalized).expanduser().exists():
                existing_paths.append(normalized)
        if existing_paths != candidates:
            self._recent_projects_settings().setValue(
                RECENT_PROJECTS_KEY,
                existing_paths,
            )
        return existing_paths[:MAX_RECENT_PROJECTS]

    def _remember_recent_project(self, project_dir: str | Path) -> None:
        normalized = str(Path(project_dir).expanduser().resolve())
        recent = [
            path for path in self._recent_project_paths() if path != normalized
        ]
        recent.insert(0, normalized)
        self._recent_projects_settings().setValue(
            RECENT_PROJECTS_KEY,
            recent[:MAX_RECENT_PROJECTS],
        )
        self._refresh_recent_projects_menu()

    def _refresh_recent_projects_menu(self) -> None:
        self.open_recent_menu.clear()
        recent_paths = self._recent_project_paths()
        if not recent_paths:
            empty_action = self.open_recent_menu.addAction(
                "No recent projects"
            )
            empty_action.setEnabled(False)
            return
        for project_path in recent_paths:
            action = self.open_recent_menu.addAction(project_path)
            action.triggered.connect(
                lambda checked=False, path=project_path: self._open_recent_project(
                    path
                )
            )

    def _open_recent_project(self, project_path: str | Path) -> None:
        try:
            self.tabs.setCurrentWidget(self.project_setup_tab)
            self.load_project(self._validated_project_dir(project_path))
        except Exception as exc:
            self._show_error("Open recent project failed", str(exc))

    def _update_file_menu_state(self) -> None:
        has_project = self.current_settings is not None
        self.save_project_action.setEnabled(has_project)
        self.save_project_as_action.setEnabled(has_project)

    def _remap_copied_project_paths(
        self,
        settings: ProjectSettings,
        *,
        old_project_dir: Path,
        new_project_dir: Path,
    ) -> None:
        for attribute in (
            "experimental_data_path",
            "copied_experimental_data_file",
            "solvent_data_path",
            "copied_solvent_data_file",
            "clusters_dir",
        ):
            current_value = getattr(settings, attribute)
            if not current_value:
                continue
            remapped = self._remap_if_within_project(
                current_value,
                old_project_dir=old_project_dir,
                new_project_dir=new_project_dir,
            )
            setattr(settings, attribute, remapped)
        self._restore_internal_staged_paths(settings, new_project_dir)

    @staticmethod
    def _restore_internal_staged_paths(
        settings: ProjectSettings,
        project_dir: Path,
    ) -> None:
        experimental_dir = (project_dir / "experimental_data").resolve()
        if (
            not settings.copied_experimental_data_file
            and settings.experimental_data_path
        ):
            experimental_path = Path(settings.experimental_data_path).resolve()
            if experimental_dir in experimental_path.parents:
                settings.copied_experimental_data_file = str(experimental_path)
        if (
            not settings.copied_solvent_data_file
            and settings.solvent_data_path
        ):
            solvent_path = Path(settings.solvent_data_path).resolve()
            if experimental_dir in solvent_path.parents:
                settings.copied_solvent_data_file = str(solvent_path)

    @staticmethod
    def _remap_if_within_project(
        path_text: str,
        *,
        old_project_dir: Path,
        new_project_dir: Path,
    ) -> str:
        try:
            resolved_path = Path(path_text).expanduser().resolve()
            relative = resolved_path.relative_to(old_project_dir.resolve())
        except Exception:
            return path_text
        return str((new_project_dir / relative).resolve())

    def _open_mdtrajectory_tool(self) -> None:
        from saxshell.mdtrajectory.ui.main_window import (
            launch_mdtrajectory_app,
        )

        window = launch_mdtrajectory_app()
        self._child_tool_windows.append(window)
        self.statusBar().showMessage("Opened mdtrajectory")

    def _open_cluster_tool(self) -> None:
        from saxshell.cluster.ui.main_window import launch_cluster_ui

        launch_cluster_ui()
        self.statusBar().showMessage("Opened cluster extraction")

    def _open_xyz2pdb_tool(self) -> None:
        from saxshell.xyz2pdb.ui.main_window import launch_xyz2pdb_ui

        window = launch_xyz2pdb_ui()
        self._child_tool_windows.append(window)
        self.statusBar().showMessage("Opened xyz2pdb conversion")

    def _open_bondanalysis_tool(self) -> None:
        from saxshell.bondanalysis.ui.main_window import BondAnalysisMainWindow

        clusters_dir = None
        if (
            self.current_settings is not None
            and self.current_settings.clusters_dir
        ):
            clusters_dir = self.current_settings.clusters_dir
        window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
        window.show()
        window.raise_()
        self._child_tool_windows.append(window)
        if clusters_dir:
            self.statusBar().showMessage(
                f"Opened bond analysis for {clusters_dir}"
            )
        else:
            self.statusBar().showMessage("Opened bond analysis")

    def _show_placeholder_tool_message(self, tool_name: str) -> None:
        QMessageBox.information(
            self,
            "Coming soon",
            (
                f"{tool_name} is listed in the Tools menu as a placeholder "
                "for future SAXShell integration."
            ),
        )
        self.statusBar().showMessage(f"{tool_name} is not available yet", 5000)

    def _open_dream_output_settings_dialog(self) -> None:
        settings = self.dream_tab.settings_payload()
        dialog = QDialog(self)
        dialog.setWindowTitle("DREAM Output Settings")
        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        verbose_checkbox = QCheckBox("Verbose sampler output")
        verbose_checkbox.setChecked(settings.verbose)
        verbose_checkbox.setToolTip(
            "Enable or disable verbose DREAM sampler progress output."
        )
        interval_spin = QDoubleSpinBox()
        interval_spin.setRange(0.1, 30.0)
        interval_spin.setDecimals(1)
        interval_spin.setSingleStep(0.1)
        interval_spin.setValue(settings.verbose_output_interval_seconds)
        interval_spin.setToolTip(
            "Minimum number of seconds between DREAM runtime output "
            "updates shown in the UI while verbose output is enabled."
        )
        interval_spin.setEnabled(verbose_checkbox.isChecked())
        verbose_checkbox.toggled.connect(interval_spin.setEnabled)
        form_layout.addRow(verbose_checkbox)
        form_layout.addRow("Output interval (s)", interval_spin)
        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_dream_output_settings(
            verbose=verbose_checkbox.isChecked(),
            interval_seconds=interval_spin.value(),
        )

    def _apply_dream_output_settings(
        self,
        *,
        verbose: bool,
        interval_seconds: float,
    ) -> None:
        settings = self.dream_tab.settings_payload()
        settings.verbose = bool(verbose)
        settings.verbose_output_interval_seconds = max(
            float(interval_seconds),
            0.1,
        )
        self.dream_tab.set_settings(
            settings,
            preset_name=self.dream_tab.selected_settings_preset_name(),
        )
        self.dream_tab.append_log(
            "Updated DREAM output settings.\n"
            f"Verbose sampler output: {'on' if settings.verbose else 'off'}\n"
            "Runtime output interval: "
            f"{settings.verbose_output_interval_seconds:.1f} s\n"
            "Save DREAM settings if you want to persist this change."
        )
        self.statusBar().showMessage("DREAM output settings updated")

    def _show_version_information(self) -> None:
        QMessageBox.information(
            self,
            "Version Information",
            self._version_information_text(),
        )

    def _version_information_text(self) -> str:
        branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")
        commit = self._git_output("rev-parse", "--short", "HEAD")
        origin_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "origin")
            or GITHUB_REPOSITORY_URL
        )
        upstream_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "upstream") or ""
        )
        lines = [
            "SAXShell Version Information",
            "",
            f"Package version: {__version__}",
            f"Git branch: {branch or 'unavailable'}",
            f"Git commit: {commit or 'unavailable'}",
            f"GitHub repository: {origin_url or GITHUB_REPOSITORY_URL}",
        ]
        if upstream_url:
            lines.append(f"Upstream repository: {upstream_url}")
        lines.extend(
            [
                "",
                "This information is read from the local Git checkout so it "
                "stays aligned with the GitHub-backed repository state for "
                "the branch you have open.",
                f"Developer contact: {CONTACT_EMAIL}",
            ]
        )
        return "\n".join(lines)

    def _open_github_repository(self) -> None:
        repository_url = self._normalize_repository_url(
            self._git_output("remote", "get-url", "origin")
            or GITHUB_REPOSITORY_URL
        )
        QDesktopServices.openUrl(QUrl(repository_url))
        self.statusBar().showMessage("Opened SAXShell GitHub repository")

    def _show_contact_information(self) -> None:
        QMessageBox.information(
            self,
            "Developer Contact",
            (
                "For SAXShell questions, template requests, or bug reports, "
                "contact the developer at:\n\n"
                f"{CONTACT_EMAIL}"
            ),
        )
        self.statusBar().showMessage("Opened developer contact information")

    @staticmethod
    def _normalize_repository_url(url: str) -> str:
        normalized = str(url or "").strip()
        if not normalized:
            return ""
        if normalized.startswith("git@github.com:"):
            normalized = normalized.replace(
                "git@github.com:",
                "https://github.com/",
                1,
            )
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        return normalized

    @staticmethod
    def _git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:
            return None
        output = result.stdout.strip()
        return output or None

    def _prompt_dream_settings_preset_name(
        self,
        *,
        suggested_name: str,
    ) -> str | None:
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save DREAM Settings",
            (
                "Enter a name for this DREAM settings preset.\n"
                "Leave it blank to save only the active project settings."
            ),
            text=suggested_name.strip(),
        )
        if not accepted:
            return None
        return preset_name.strip() or None

    def show_prior_histogram_window(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Prior histogram unavailable",
                "Load or build a project first.",
            )
            return
        paths = build_project_paths(self.current_settings.project_dir)
        prior_json = paths.project_dir / "md_prior_weights.json"
        if not prior_json.is_file():
            self._show_error(
                "Prior histogram unavailable",
                "Generate prior weights before opening prior histograms.",
            )
            return
        if (
            self.project_setup_tab.prior_mode().startswith("solvent_sort")
            and self.project_setup_tab.prior_secondary_element() is None
        ):
            self._show_error(
                "Prior histogram unavailable",
                "Select a secondary atom filter before opening a solvent-sort prior histogram.",
            )
            return
        window = PriorHistogramWindow(
            prior_json,
            mode=self.project_setup_tab.prior_mode(),
            secondary_element=self.project_setup_tab.prior_secondary_element(),
            cmap=self.project_setup_tab.prior_cmap(),
            parent=None,
        )
        self._prior_histogram_windows.append(window)
        window.destroyed.connect(self._on_prior_histogram_window_destroyed)
        window.show()
        window.raise_()
        window.activateWindow()

    def save_prior_plot_png(self) -> None:
        if self.current_settings is None:
            self._show_error(
                "Save prior histogram failed",
                "Load or build a project first.",
            )
            return
        paths = build_project_paths(self.current_settings.project_dir)
        prior_json = paths.project_dir / "md_prior_weights.json"
        if not prior_json.is_file():
            self._show_error(
                "Save prior histogram failed",
                "Generate prior weights before saving a prior histogram image.",
            )
            return
        if (mode := self.project_setup_tab.prior_mode()).startswith(
            "solvent_sort"
        ) and (self.project_setup_tab.prior_secondary_element() is None):
            self._show_error(
                "Save prior histogram failed",
                "Select a secondary atom filter before saving a solvent-sort prior histogram image.",
            )
            return
        paths.plots_dir.mkdir(parents=True, exist_ok=True)
        secondary = self.project_setup_tab.prior_secondary_element()
        suffix = f"_{secondary}" if secondary else ""
        output_path = paths.plots_dir / (
            f"prior_histogram_{mode}{suffix}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        try:
            self.project_setup_tab.draw_prior_plot(prior_json)
            self.project_setup_tab.prior_figure.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )
        except Exception as exc:
            self._show_error("Save prior histogram failed", str(exc))
            return
        self.project_setup_tab.append_summary(
            f"Saved prior histogram image to {output_path}"
        )
        self.statusBar().showMessage("Prior histogram image saved")

    @Slot(QObject)
    def _on_prior_histogram_window_destroyed(
        self,
        window: QObject | None,
    ) -> None:
        self._prior_histogram_windows = [
            open_window
            for open_window in self._prior_histogram_windows
            if open_window is not window
        ]

    def _refresh_loaded_dream_results(self) -> None:
        if self._last_results_loader is None:
            return
        try:
            settings = self.dream_tab.settings_payload()
            summary = self._last_results_loader.get_summary(
                bestfit_method=settings.bestfit_method,
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
            )
            model_plot = self._last_results_loader.build_model_fit_data(
                bestfit_method=settings.bestfit_method,
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
            )
            violin_plot = self._last_results_loader.build_violin_data(
                mode=self._effective_dream_violin_mode(settings),
                posterior_filter_mode=settings.posterior_filter_mode,
                posterior_top_percent=settings.posterior_top_percent,
                posterior_top_n=settings.posterior_top_n,
                credible_interval_low=settings.credible_interval_low,
                credible_interval_high=settings.credible_interval_high,
                sample_source=settings.violin_sample_source,
                weight_order=settings.violin_weight_order,
            )
            self.dream_tab.set_summary_text(
                self._format_dream_summary(summary, settings=settings)
            )
            self.dream_tab.plot_model_fit(model_plot)
            self.dream_tab.plot_violin_plot(summary, violin_plot)
        except Exception as exc:
            self._show_error("Render DREAM results failed", str(exc))

    def _format_dream_summary(
        self,
        summary,
        *,
        settings: DreamRunSettings,
    ) -> str:
        lines = [
            f"Run directory: {summary.run_dir}",
            f"Best-fit method: {summary.bestfit_method}",
            (
                "Posterior filter: "
                f"{self._describe_posterior_filter(settings)}"
            ),
            f"Posterior samples kept: {summary.posterior_sample_count}",
            (
                "Credible interval (%): "
                f"{summary.credible_interval_low:g} - "
                f"{summary.credible_interval_high:g}"
            ),
            f"Violin data mode: {settings.violin_parameter_mode}",
            f"Violin sample source: {settings.violin_sample_source}",
            f"Weight order: {settings.violin_weight_order}",
            f"Y-axis scale: {settings.violin_value_scale_mode}",
            f"Violin palette: {settings.violin_palette}",
            f"Point color: {settings.violin_point_color}",
            (
                "MAP location: "
                f"chain {summary.map_chain + 1}, step {summary.map_step + 1}"
            ),
            "",
            "Posterior summary:",
        ]
        for index, name in enumerate(summary.full_parameter_names):
            lines.append(
                f"{name}: selected={summary.bestfit_params[index]:.6g}, "
                f"MAP={summary.map_params[index]:.6g}, "
                f"chain_mean={summary.chain_mean_params[index]:.6g}, "
                f"median={summary.median_params[index]:.6g}, "
                f"p{summary.credible_interval_low:g}="
                f"{summary.interval_low_values[index]:.6g}, "
                f"p{summary.credible_interval_high:g}="
                f"{summary.interval_high_values[index]:.6g}"
            )
        return "\n".join(lines)

    def _format_dream_console_intro(self) -> str:
        if self.prefit_workflow is None:
            return "DREAM workflow is not loaded."
        settings = self.dream_tab.settings_payload()
        return (
            "DREAM workflow loaded.\n"
            f"Template: {self.prefit_workflow.template_spec.name}\n"
            "Review the priors with Edit Priors and click Save Parameter "
            "Map before running DREAM.\n"
            f"Best-fit method: {settings.bestfit_method}\n"
            f"Posterior filter: {self._describe_posterior_filter(settings)}\n"
            f"Violin data mode: {settings.violin_parameter_mode}\n"
            f"Violin sample source: {settings.violin_sample_source}\n"
            f"Weight order: {settings.violin_weight_order}\n"
            f"Y-axis scale: {settings.violin_value_scale_mode}\n"
            f"Violin palette: {settings.violin_palette}\n"
            f"Point color: {settings.violin_point_color}\n"
            "Recommendation: all refinable parameters are usually allowed "
            "to vary during DREAM refinement."
        )

    @staticmethod
    def _describe_posterior_filter(settings: DreamRunSettings) -> str:
        if settings.posterior_filter_mode == "top_percent_logp":
            return (
                f"top_percent_logp "
                f"(top {settings.posterior_top_percent:g}% by log-posterior)"
            )
        if settings.posterior_filter_mode == "top_n_logp":
            return (
                f"top_n_logp "
                f"(top {settings.posterior_top_n} samples by log-posterior)"
            )
        return "all_post_burnin"

    def _append_dream_vary_recommendation(self, entries: list) -> None:
        fixed = [entry.param for entry in entries if not entry.vary]
        if not fixed:
            return
        names = ", ".join(fixed[:10])
        if len(fixed) > 10:
            names += ", ..."
        self.dream_tab.append_log(
            "Recommendation: allow all refinable parameters to vary during "
            "the DREAM refinement. The current parameter map has vary=off "
            f"for: {names}"
        )

    def _invalidate_written_dream_bundle(self) -> None:
        self._last_written_dream_bundle = None

    def _format_prefit_console_intro(self) -> str:
        if self.prefit_workflow is None:
            return "Prefit workflow is not loaded."
        settings = self.prefit_workflow.settings
        q_values = self.prefit_workflow.evaluate().q_values
        run_config = self.prefit_tab.run_config()
        if settings.use_experimental_grid:
            grid_text = (
                "Using the experimental q-grid cropped to the nearest "
                "available q-points inside the requested range "
                f"({len(q_values)} points from {float(q_values.min()):.6g} "
                f"to {float(q_values.max()):.6g})."
            )
        else:
            grid_text = (
                "Resampling the experimental data onto "
                f"{len(q_values)} evenly spaced q-points between "
                f"{float(q_values.min()):.6g} and {float(q_values.max()):.6g}."
            )
        excluded = ", ".join(settings.exclude_elements) or "None"
        return (
            "Prefit workflow loaded.\n"
            f"Template: {self.prefit_workflow.template_spec.name}\n"
            f"{grid_text}\n"
            f"Excluded elements: {excluded}\n"
            "Project presets: template-default reset is available"
            + (
                "; Best Prefit preset will be applied on project reload."
                if self.prefit_workflow.has_best_prefit_entries()
                else "; no Best Prefit preset is saved yet."
            )
            + "\n"
            "Recommended order: refine scale first, then scale + offset. "
            "Component weights w<##> are not recommended for prefit refinement.\n"
            f"Default minimizer: {run_config.method}\n"
            f"Default max nfev: {run_config.max_nfev}\n"
            "Autosave fit results: "
            + ("enabled" if settings.autosave_prefits else "disabled")
        )

    def _format_prefit_summary(
        self,
        evaluation,
        *,
        fit_result=None,
        report_path: Path | None = None,
    ) -> str:
        if self.prefit_workflow is None:
            return "Prefit summary is not available."
        residuals = np.asarray(evaluation.residuals, dtype=float)
        q_values = np.asarray(evaluation.q_values, dtype=float)
        rms_residual = float(np.sqrt(np.mean(residuals**2)))
        mean_abs_residual = float(np.mean(np.abs(residuals)))
        lines = [
            "Prefit summary:",
            f"Template: {self.prefit_workflow.template_spec.name}",
            f"Points: {len(q_values)}",
            (
                f"q-range: {float(q_values.min()):.6g} to "
                f"{float(q_values.max()):.6g}"
            ),
            f"Residual RMS: {rms_residual:.6g}",
            f"Mean |residual|: {mean_abs_residual:.6g}",
            f"Configured minimizer: {self.prefit_tab.run_config().method}",
            f"Configured max nfev: {self.prefit_tab.run_config().max_nfev}",
            (
                "Autosave fits: "
                + (
                    "enabled"
                    if self.prefit_workflow.settings.autosave_prefits
                    else "disabled"
                )
            ),
        ]
        if fit_result is not None:
            lines.extend(
                [
                    f"Method: {fit_result.method}",
                    f"Function evals: {fit_result.nfev}",
                    f"Chi^2: {fit_result.chi_square:.6g}",
                    f"Reduced chi^2: {fit_result.reduced_chi_square:.6g}",
                    f"R^2: {fit_result.r_squared:.6g}",
                ]
            )
        if report_path is not None:
            lines.append(f"Saved report: {report_path}")
        return "\n".join(lines)

    def _confirm_prefit_template_change(
        self,
        current_name: str,
        new_name: str,
    ) -> tuple[bool, bool]:
        if not self._warn_on_prefit_template_change:
            return True, False
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Change prefit template?")
        dialog.setText(
            "Changing the SAXS prefit template will reset the parameter table "
            "to the new template defaults and immediately update the model."
        )
        dialog.setInformativeText(
            f"Current template: {current_name}\n" f"New template: {new_name}"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        suppress_checkbox = QCheckBox(
            "Don't warn again for template changes during this session"
        )
        dialog.setCheckBox(suppress_checkbox)
        response = dialog.exec()
        return (
            response == QMessageBox.StandardButton.Yes,
            suppress_checkbox.isChecked(),
        )

    def _restore_prefit_template_selection(self, template_name: str) -> None:
        self._restoring_prefit_template = True
        self.prefit_tab.set_selected_template(template_name)
        self._restoring_prefit_template = False

    def apply_recommended_scale_settings(self) -> None:
        if self.prefit_workflow is None:
            self._show_error(
                "Prefit unavailable",
                "Build a project and load its SAXS components first.",
            )
            return
        try:
            entries = self.prefit_tab.parameter_entries()
            recommendation = self.prefit_workflow.recommend_scale_settings(
                entries
            )
            self.prefit_tab.set_parameter_row(
                "scale",
                value=recommendation.recommended_scale,
                minimum=recommendation.recommended_minimum,
                maximum=recommendation.recommended_maximum,
                vary=True,
            )
            self.prefit_tab.append_log(
                "Applied recommended scale settings.\n"
                f"Current scale: {recommendation.current_scale:.6g}\n"
                f"Recommended scale: {recommendation.recommended_scale:.6g}\n"
                f"Scale min: {recommendation.recommended_minimum:.6g}\n"
                f"Scale max: {recommendation.recommended_maximum:.6g}\n"
                f"Adjustment factor: {recommendation.adjustment_factor:.6g}\n"
                f"Points used: {recommendation.points_used}"
            )
            self.update_prefit_model()
            self.statusBar().showMessage("Recommended scale settings applied")
        except Exception as exc:
            self._show_error("Scale recommendation failed", str(exc))

    def _append_scale_recommendation_log(
        self,
        recommendation: PrefitScaleRecommendation,
    ) -> None:
        self.prefit_tab.append_log(
            "Recommended scale estimate available.\n"
            f"Current scale: {recommendation.current_scale:.6g}\n"
            f"Recommended scale: {recommendation.recommended_scale:.6g}\n"
            f"Suggested range: {recommendation.recommended_minimum:.6g} "
            f"to {recommendation.recommended_maximum:.6g}\n"
            f"Adjustment factor: {recommendation.adjustment_factor:.6g}\n"
            f"Points used: {recommendation.points_used}"
        )

    def _maybe_append_scale_recommendation(
        self,
        entries=None,
    ) -> None:
        if self.prefit_workflow is None:
            return
        try:
            recommendation = self.prefit_workflow.recommend_scale_settings(
                entries
            )
        except Exception:
            return
        self._append_scale_recommendation_log(recommendation)

    def _validated_project_dir(self, project_dir: str | Path) -> Path:
        resolved_dir = Path(project_dir).expanduser().resolve()
        project_file = build_project_paths(resolved_dir).project_file
        if not project_file.is_file():
            raise ValueError(
                "Select a complete SAXS project folder that contains "
                "saxs_project.json, not a parent directory of multiple projects."
            )
        return resolved_dir

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)


def launch_saxs_ui(
    initial_project_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    window = SAXSMainWindow(initial_project_dir=initial_project_dir)
    window.show()
    if owns_app:
        return int(app.exec())
    return 0
