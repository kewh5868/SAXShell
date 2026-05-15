from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    BondAnalysisPreset,
    load_presets,
    ordered_preset_names,
)
from saxshell.representativefinder.run_config import (
    RepresentativeFinderRunConfig,
    build_representativefinder_run_config,
    run_representativefinder_run_config,
)
from saxshell.representativefinder.workflow import (
    RepresentativeFinderSettings,
    suggest_representativefinder_output_dir,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)


def _new_item_id() -> str:
    return uuid.uuid4().hex


def _optional_path(text: str) -> Path | None:
    stripped = text.strip()
    if not stripped:
        return None
    return Path(stripped).expanduser().resolve()


def _required_path(text: str, field_name: str) -> Path:
    path = _optional_path(text)
    if path is None:
        raise ValueError(f"{field_name} is required.")
    return path


def _required_project_dir(text: str) -> Path:
    project_dir = _required_path(text, "Project folder")
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        raise ValueError(f"Project file does not exist: {project_file}")
    return project_dir


def _required_clusters_dir(text: str) -> Path:
    clusters_dir = _required_path(text, "Project clusters folder")
    if not clusters_dir.is_dir():
        raise ValueError(
            f"Project clusters folder does not exist: {clusters_dir}"
        )
    return clusters_dir


def _dialog_start_dir(*candidates: str | Path | None) -> str:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.parent)
        if path.is_dir():
            return str(path)
    return str(Path.home())


def _choose_existing_directories(
    parent: QWidget,
    *,
    title: str,
    start_dir: str | Path,
) -> tuple[Path, ...]:
    dialog = QFileDialog(parent, title, str(start_dir))
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    for view in dialog.findChildren(QListView) + dialog.findChildren(
        QTreeView
    ):
        view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
    if dialog.exec() != int(QFileDialog.DialogCode.Accepted):
        return ()
    return tuple(
        Path(path).expanduser().resolve() for path in dialog.selectedFiles()
    )


def _project_reference_text(project_dir: Path | None) -> str:
    if project_dir is None:
        return "Project reference: choose a SAXSShell project folder."
    project_file = build_project_paths(project_dir).project_file
    if project_file.is_file():
        return f"Project reference: {project_file}"
    return f"Project reference: no project file found at {project_file}"


def _suggest_batch_output_dir(project_dir: Path, clusters_dir: Path) -> Path:
    return suggest_representativefinder_output_dir(
        clusters_dir,
        project_dir=project_dir,
        batch=True,
    )


def _settings_from_preset(
    preset: BondAnalysisPreset,
) -> RepresentativeFinderSettings:
    return RepresentativeFinderSettings(
        bond_pairs=tuple(preset.bond_pairs),
        angle_triplets=tuple(preset.angle_triplets),
    )


@dataclass(slots=True, frozen=True)
class RepresentativeFinderBatchJob:
    project_dir: Path
    clusters_dir: Path
    output_dir: Path
    config: RepresentativeFinderRunConfig


@dataclass(slots=True)
class RepresentativeFinderBatchResult:
    project_dir: Path
    clusters_dir: Path
    output_dir: Path
    completed_count: int
    failed_count: int
    skipped_count: int


@dataclass(slots=True)
class RepresentativeFinderBatchItem:
    item_id: str
    project_dir: Path | None = None
    clusters_dir: Path | None = None
    output_dir: Path | None = None

    def display_name(self) -> str:
        if self.project_dir is not None:
            return self.project_dir.name
        if self.clusters_dir is not None:
            return self.clusters_dir.name
        return "New representative analysis"


def _queue_item_from_project_defaults(
    project_dir: str | Path,
    *,
    item_id: str | None = None,
) -> RepresentativeFinderBatchItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = RepresentativeFinderBatchItem(
        item_id=item_id or _new_item_id(),
        project_dir=resolved_project_dir,
    )
    try:
        settings = SAXSProjectManager().load_project(resolved_project_dir)
    except Exception:
        return item
    clusters_dir = settings.resolved_clusters_dir
    output_dir = None
    if clusters_dir is not None and clusters_dir.is_dir():
        try:
            output_dir = _suggest_batch_output_dir(
                resolved_project_dir,
                clusters_dir,
            )
        except Exception:
            output_dir = None
    return replace(item, clusters_dir=clusters_dir, output_dir=output_dir)


class RepresentativeFinderBatchItemWidget(QFrame):
    settings_changed = Signal(str)
    remove_requested = Signal(str)
    duplicate_requested = Signal(str)

    def __init__(
        self,
        item: RepresentativeFinderBatchItem,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._loading = False
        self._selected = False
        self._last_suggested_output_dir: Path | None = None
        self._build_ui()
        self._load_item(item)
        self._set_settings_visible(False)

    @property
    def item_id(self) -> str:
        return self._item.item_id

    def item(self) -> RepresentativeFinderBatchItem:
        return self._item

    def collect_item(self) -> RepresentativeFinderBatchItem:
        self._item = RepresentativeFinderBatchItem(
            item_id=self._item.item_id,
            project_dir=_optional_path(self.project_dir_edit.text()),
            clusters_dir=_optional_path(self.clusters_dir_edit.text()),
            output_dir=_optional_path(self.output_dir_edit.text()),
        )
        self._refresh_header()
        self._refresh_project_reference()
        return self._item

    def job(
        self,
        *,
        settings: RepresentativeFinderSettings,
    ) -> RepresentativeFinderBatchJob:
        self.collect_item()
        project_dir = _required_project_dir(self.project_dir_edit.text())
        clusters_dir = _required_clusters_dir(self.clusters_dir_edit.text())
        output_dir = _optional_path(
            self.output_dir_edit.text()
        ) or _suggest_batch_output_dir(project_dir, clusters_dir)
        config = build_representativefinder_run_config(
            project_dir=project_dir,
            input_dir=clusters_dir,
            output_dir=output_dir,
            analysis_mode="all",
            settings=settings,
            overwrite_existing=False,
        )
        return RepresentativeFinderBatchJob(
            project_dir=project_dir,
            clusters_dir=clusters_dir,
            output_dir=output_dir,
            config=config,
        )

    def set_locked(self, locked: bool) -> None:
        self.settings_group.setEnabled(not locked)
        self.validate_button.setEnabled(not locked)
        self.duplicate_button.setEnabled(not locked)
        self.remove_button.setEnabled(not locked)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_progress(self, processed: int, total: int) -> None:
        self.progress_bar.setRange(0, max(int(total), 1))
        self.progress_bar.setValue(max(int(processed), 0))

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.header_frame.setProperty("selected", self._selected)
        self.header_frame.setStyleSheet(
            "QFrame#RepresentativeFinderBatchItemHeader {"
            + (
                "background-color: #dce8f7; " "border: 1px solid #8fb0d7;"
                if self._selected
                else "background-color: #f6f8fb; " "border: 1px solid #cfd7e3;"
            )
            + "border-radius: 5px;}"
        )

    def validate_paths(self) -> None:
        project_dir = _required_project_dir(self.project_dir_edit.text())
        clusters_dir = _required_clusters_dir(self.clusters_dir_edit.text())
        suggested = _suggest_batch_output_dir(project_dir, clusters_dir)
        current = _optional_path(self.output_dir_edit.text())
        if current is None or current == self._last_suggested_output_dir:
            self.output_dir_edit.setText(str(suggested))
        self._last_suggested_output_dir = suggested
        self.set_status("Ready")

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.header_frame = QFrame()
        self.header_frame.setObjectName("RepresentativeFinderBatchItemHeader")
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(8, 6, 8, 6)
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._set_settings_visible)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("New representative analysis")
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(190)
        header.addWidget(self.status_label)
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self._validate_from_button)
        header.addWidget(self.validate_button)
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(
            lambda: self.duplicate_requested.emit(self.item_id)
        )
        header.addWidget(self.duplicate_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(
            lambda: self.remove_requested.emit(self.item_id)
        )
        header.addWidget(self.remove_button)
        root.addWidget(self.header_frame)
        self.set_selected(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m steps")
        root.addWidget(self.progress_bar)

        self.settings_group = QGroupBox(
            "Representative Structure Batch Settings"
        )
        root.addWidget(self.settings_group)
        settings_layout = QVBoxLayout(self.settings_group)

        form = QFormLayout()
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(self._on_project_changed)
        form.addRow(
            "Project folder",
            self._path_row(self.project_dir_edit, self._choose_project_dir),
        )
        self.project_reference_label = QLabel()
        self.project_reference_label.setWordWrap(True)
        self.project_reference_label.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", self.project_reference_label)

        self.clusters_dir_edit = QLineEdit()
        self.clusters_dir_edit.editingFinished.connect(
            self._on_clusters_changed
        )
        form.addRow(
            "Project clusters folder",
            self._path_row(self.clusters_dir_edit, self._choose_clusters_dir),
        )

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.editingFinished.connect(self._on_editor_changed)
        form.addRow(
            "Output root",
            self._path_row(self.output_dir_edit, self._choose_output_dir),
        )
        self.analysis_mode_label = QLabel("All Discovered Stoichiometries")
        form.addRow("Analysis mode", self.analysis_mode_label)
        settings_layout.addLayout(form)

    def _path_row(self, edit: QLineEdit, slot) -> QWidget:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)
        button = QPushButton("Browse...")
        button.clicked.connect(slot)
        row.addWidget(button)
        return row_widget

    def _load_item(self, item: RepresentativeFinderBatchItem) -> None:
        self._loading = True
        self.project_dir_edit.setText(
            "" if item.project_dir is None else str(item.project_dir)
        )
        self.clusters_dir_edit.setText(
            "" if item.clusters_dir is None else str(item.clusters_dir)
        )
        self.output_dir_edit.setText(
            "" if item.output_dir is None else str(item.output_dir)
        )
        if item.output_dir is not None:
            self._last_suggested_output_dir = item.output_dir
        self._loading = False
        self._refresh_header()
        self._refresh_project_reference()
        self._validate_quietly()

    def _set_settings_visible(self, visible: bool) -> None:
        self.settings_group.setVisible(bool(visible))
        self.toggle_button.setChecked(bool(visible))
        self.toggle_button.setText("Hide Settings" if visible else "Settings")
        parent_item = self._list_item()
        if parent_item is not None:
            parent_item.setSizeHint(self.sizeHint())

    def _list_item(self) -> QListWidgetItem | None:
        parent = self.parent()
        while parent is not None and not isinstance(parent, QListWidget):
            parent = parent.parent()
        if not isinstance(parent, QListWidget):
            return None
        for row in range(parent.count()):
            list_item = parent.item(row)
            if parent.itemWidget(list_item) is self:
                return list_item
        return None

    def _choose_project_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAXSShell project folder",
            _dialog_start_dir(self.project_dir_edit.text()),
        )
        if not selected:
            return
        self._load_item(
            _queue_item_from_project_defaults(
                selected,
                item_id=self.item_id,
            )
        )
        self._on_editor_changed()

    def _choose_clusters_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select project clusters folder",
            _dialog_start_dir(
                self.clusters_dir_edit.text(),
                self.project_dir_edit.text(),
            ),
        )
        if not selected:
            return
        self.clusters_dir_edit.setText(selected)
        self._on_clusters_changed()

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select representative output root",
            _dialog_start_dir(
                self.output_dir_edit.text(),
                self.clusters_dir_edit.text(),
            ),
        )
        if not selected:
            return
        self.output_dir_edit.setText(selected)
        self._on_editor_changed()

    def _on_project_changed(self) -> None:
        project_dir = _optional_path(self.project_dir_edit.text())
        if project_dir is None:
            self._on_editor_changed()
            return
        item = _queue_item_from_project_defaults(
            project_dir,
            item_id=self.item_id,
        )
        if item.clusters_dir is not None:
            self.clusters_dir_edit.setText(str(item.clusters_dir))
        if item.output_dir is not None:
            self.output_dir_edit.setText(str(item.output_dir))
            self._last_suggested_output_dir = item.output_dir
        self._validate_quietly()
        self._on_editor_changed()

    def _on_clusters_changed(self) -> None:
        clusters_dir = _optional_path(self.clusters_dir_edit.text())
        project_dir = _optional_path(self.project_dir_edit.text())
        if clusters_dir is not None and project_dir is not None:
            try:
                suggested = _suggest_batch_output_dir(
                    project_dir, clusters_dir
                )
                current = _optional_path(self.output_dir_edit.text())
                if (
                    current is None
                    or current == self._last_suggested_output_dir
                ):
                    self.output_dir_edit.setText(str(suggested))
                self._last_suggested_output_dir = suggested
            except Exception:
                pass
        self._validate_quietly()
        self._on_editor_changed()

    def _validate_from_button(self) -> None:
        try:
            self.validate_paths()
            self._on_editor_changed()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to validate representative batch item",
                str(exc),
            )
            self.set_status("Validation failed")
            self._on_editor_changed()

    def _validate_quietly(self) -> None:
        if not self.clusters_dir_edit.text().strip():
            return
        try:
            self.validate_paths()
        except Exception:
            self.set_status("Validation failed")

    def _on_editor_changed(self, *_args) -> None:
        if self._loading:
            return
        try:
            self.collect_item()
            if self.status_label.text() in {"Validation failed", "Failed"}:
                self.set_status("Ready")
        except Exception:
            self._refresh_header()
            self._refresh_project_reference()
        self.settings_changed.emit(self.item_id)

    def _refresh_header(self) -> None:
        self.title_label.setText(self._item.display_name())

    def _refresh_project_reference(self) -> None:
        self.project_reference_label.setText(
            _project_reference_text(
                _optional_path(self.project_dir_edit.text())
            )
        )


class RepresentativeFinderBatchWorker(QObject):
    item_started = Signal(str, int, int)
    item_progress = Signal(str, int, int, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    log = Signal(str)
    status = Signal(str)
    project_results_changed = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        queue_entries: list[tuple[str, RepresentativeFinderBatchJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[RepresentativeFinderBatchResult] = []
        total_items = len(self.queue_entries)
        for index, (item_id, job) in enumerate(
            self.queue_entries,
            start=1,
        ):
            if self._cancel_requested.is_set():
                self.log.emit("Batch queue stopped before the next project.")
                break
            self.item_started.emit(item_id, index, total_items)
            self.status.emit(
                f"Running {index}/{total_items}: {job.project_dir.name}"
            )
            try:
                result = self._run_job(item_id, job)
            except Exception as exc:
                message = str(exc)
                self.item_failed.emit(item_id, message)
                self.failed.emit(item_id, message)
                return
            results.append(result)
            self.item_finished.emit(item_id, result)
            self.project_results_changed.emit(str(result.project_dir))
        self.status.emit("Representative structure batch queue finished")
        self.finished.emit(results)

    def _run_job(
        self,
        item_id: str,
        job: RepresentativeFinderBatchJob,
    ) -> RepresentativeFinderBatchResult:
        self.log.emit(
            f"[{job.project_dir.name}] Starting representative analysis."
        )
        last_progress_emit = 0.0

        def on_progress(processed: int, total: int, message: str) -> None:
            nonlocal last_progress_emit
            now = time.monotonic()
            is_terminal_update = total > 0 and processed >= total
            if not is_terminal_update and now - last_progress_emit < 0.15:
                return
            last_progress_emit = now
            self.item_progress.emit(item_id, processed, total, message)

        def on_log(message: str) -> None:
            self.log.emit(f"[{job.project_dir.name}] {message}")

        summary = run_representativefinder_run_config(
            job.project_dir,
            job.config,
            log_callback=on_log,
            progress_callback=on_progress,
        )
        result = RepresentativeFinderBatchResult(
            project_dir=job.project_dir,
            clusters_dir=job.clusters_dir,
            output_dir=job.output_dir,
            completed_count=summary.completed_count,
            failed_count=summary.failed_count,
            skipped_count=len(summary.skipped_existing),
        )
        if result.failed_count:
            self.log.emit(
                f"[{job.project_dir.name}] Completed with "
                f"{result.failed_count} failed stoichiometry run(s)."
            )
        else:
            self.log.emit(
                f"[{job.project_dir.name}] Completed "
                f"{result.completed_count} representative selection(s)."
            )
        return result


class RepresentativeFinderBatchQueueWindow(QMainWindow):
    """Queue representative-structure analysis for multiple projects."""

    project_results_changed = Signal(str)

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_clusters_dir: str | Path | None = None,
        initial_input_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if initial_clusters_dir is None:
            initial_clusters_dir = initial_input_path
        self._widgets_by_id: dict[str, RepresentativeFinderBatchItemWidget] = (
            {}
        )
        self._run_thread: QThread | None = None
        self._run_worker: RepresentativeFinderBatchWorker | None = None
        self._presets: dict[str, BondAnalysisPreset] = {}
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_clusters_dir = (
            None
            if initial_clusters_dir is None
            else Path(initial_clusters_dir).expanduser().resolve()
        )
        self._build_ui()
        self._reload_presets()
        if (
            self._initial_project_dir is not None
            or self._initial_clusters_dir is not None
        ):
            self._add_current_project()

    def closeEvent(self, event) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            self._request_cancel()
            self.hide()
            while (
                self._run_thread is not None and self._run_thread.isRunning()
            ):
                QApplication.processEvents()
                if self._run_thread is not None:
                    self._run_thread.wait(50)
            event.accept()
            return
        super().closeEvent(event)

    def add_queue_item(
        self,
        item: RepresentativeFinderBatchItem | None = None,
    ) -> RepresentativeFinderBatchItemWidget:
        resolved_item = item or RepresentativeFinderBatchItem(
            item_id=_new_item_id()
        )
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_list.addItem(list_item)
        widget = RepresentativeFinderBatchItemWidget(
            resolved_item,
            parent=self.queue_list,
        )
        widget.settings_changed.connect(self._on_item_settings_changed)
        widget.remove_requested.connect(self._remove_item)
        widget.duplicate_requested.connect(self._duplicate_item)
        self._widgets_by_id[resolved_item.item_id] = widget
        list_item.setSizeHint(widget.sizeHint())
        self.queue_list.setItemWidget(list_item, widget)
        self.queue_list.setCurrentItem(list_item)
        self._refresh_order_labels()
        return widget

    def queue_jobs_in_order(
        self,
    ) -> list[tuple[str, RepresentativeFinderBatchJob]]:
        preset = self._selected_preset()
        if preset is None:
            raise ValueError("Choose a bondanalysis preset before running.")
        settings = _settings_from_preset(preset)
        entries: list[tuple[str, RepresentativeFinderBatchJob]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.job(settings=settings)))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell Representative Structures Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1120, 840)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls = QHBoxLayout()
        self.add_current_button = QPushButton("Add Current Project")
        self.add_current_button.clicked.connect(self._add_current_project)
        controls.addWidget(self.add_current_button)
        self.add_project_button = QPushButton("Add Projects...")
        self.add_project_button.clicked.connect(self._choose_projects_to_add)
        controls.addWidget(self.add_project_button)
        controls.addStretch(1)
        root.addLayout(controls)

        preset_group = QGroupBox("Batch Settings")
        preset_layout = QFormLayout(preset_group)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_preset_summary()
        )
        preset_row.addWidget(self.preset_combo, stretch=1)
        self.reload_presets_button = QPushButton("Reload Presets")
        self.reload_presets_button.clicked.connect(self._reload_presets)
        preset_row.addWidget(self.reload_presets_button)
        preset_widget = QWidget()
        preset_widget.setLayout(preset_row)
        preset_layout.addRow("Bondanalysis preset", preset_widget)
        preset_layout.addRow(
            "Analysis mode",
            QLabel("All Discovered Stoichiometries"),
        )
        self.preset_summary_label = QLabel()
        self.preset_summary_label.setWordWrap(True)
        preset_layout.addRow("", self.preset_summary_label)
        root.addWidget(preset_group)

        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.queue_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.queue_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.queue_list.setAlternatingRowColors(True)
        self.queue_list.setStyleSheet(
            "QListWidget::item:selected { background: transparent; }"
            "QListWidget::item:hover { background: transparent; }"
            "QListWidget::item { margin: 3px; }"
        )
        self.queue_list.model().rowsMoved.connect(self._refresh_order_labels)
        self.queue_list.itemSelectionChanged.connect(
            self._refresh_item_selection_styles
        )
        root.addWidget(self.queue_list, stretch=1)

        run_group = QGroupBox("Execute Queue")
        run_layout = QVBoxLayout(run_group)
        run_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run Complete Queue")
        self.run_button.clicked.connect(self._start_queue)
        run_buttons.addWidget(self.run_button)
        self.cancel_button = QPushButton("Stop Queue")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        run_buttons.addWidget(self.cancel_button)
        run_buttons.addStretch(1)
        run_layout.addLayout(run_buttons)
        self.queue_status_label = QLabel("Queue idle")
        run_layout.addWidget(self.queue_status_label)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(150)
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _reload_presets(self) -> None:
        current_name = self._selected_preset_name()
        self._presets = load_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        selected_index = 0
        for index, name in enumerate(ordered_preset_names(self._presets)):
            preset = self._presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.preset_combo.addItem(label, name)
            if name == current_name:
                selected_index = index
        if self.preset_combo.count() > 0:
            self.preset_combo.setCurrentIndex(selected_index)
        self.preset_combo.blockSignals(False)
        self._refresh_preset_summary()

    def _selected_preset_name(self) -> str | None:
        if not hasattr(self, "preset_combo"):
            return None
        payload = self.preset_combo.currentData()
        return None if payload is None else str(payload)

    def _selected_preset(self) -> BondAnalysisPreset | None:
        preset_name = self._selected_preset_name()
        if preset_name is None:
            return None
        return self._presets.get(preset_name)

    def _refresh_preset_summary(self) -> None:
        preset = self._selected_preset()
        if preset is None:
            self.preset_summary_label.setText(
                "No bondanalysis preset is selected."
            )
            return
        self.preset_summary_label.setText(
            f"Using {preset.name}: {len(preset.bond_pairs)} bond pair(s), "
            f"{len(preset.angle_triplets)} angle triplet(s). Advanced "
            "representative scoring and solvent shell builder settings use "
            "their defaults."
        )

    def _add_current_project(self) -> None:
        if (
            self._initial_project_dir is None
            and self._initial_clusters_dir is None
        ):
            QMessageBox.information(
                self,
                "No active project",
                "The main UI did not provide an active project reference.",
            )
            return
        item = (
            _queue_item_from_project_defaults(self._initial_project_dir)
            if self._initial_project_dir is not None
            else RepresentativeFinderBatchItem(item_id=_new_item_id())
        )
        if self._initial_clusters_dir is not None:
            output_dir = None
            if self._initial_project_dir is not None:
                try:
                    output_dir = _suggest_batch_output_dir(
                        self._initial_project_dir,
                        self._initial_clusters_dir,
                    )
                except Exception:
                    output_dir = item.output_dir
            item = replace(
                item,
                clusters_dir=self._initial_clusters_dir,
                output_dir=output_dir or item.output_dir,
            )
        self.add_queue_item(item)

    def _choose_projects_to_add(self) -> None:
        selected_dirs = _choose_existing_directories(
            self,
            title="Select SAXSShell project folders",
            start_dir=self._initial_project_dir or Path.home(),
        )
        if not selected_dirs:
            return
        for project_dir in selected_dirs:
            self.add_queue_item(_queue_item_from_project_defaults(project_dir))

    def _on_item_settings_changed(self, _item_id: str) -> None:
        self._refresh_order_labels()

    def _refresh_order_labels(self, *_args) -> None:
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id.get(item_id)
            if widget is None:
                continue
            widget.title_label.setText(
                f"{row + 1}. {widget.item().display_name()}"
            )
            list_item.setSizeHint(widget.sizeHint())
        self._refresh_item_selection_styles()

    def _refresh_item_selection_styles(self) -> None:
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id.get(item_id)
            if widget is not None:
                widget.set_selected(list_item.isSelected())

    def _remove_item(self, item_id: str) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            return
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            if str(list_item.data(Qt.ItemDataRole.UserRole)) == item_id:
                self.queue_list.takeItem(row)
                break
        self._widgets_by_id.pop(item_id, None)
        self._refresh_order_labels()

    def _duplicate_item(self, item_id: str) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        try:
            item = widget.collect_item()
        except Exception:
            item = widget.item()
        self.add_queue_item(replace(item, item_id=_new_item_id()))

    def _set_running(self, running: bool) -> None:
        self.add_current_button.setEnabled(not running)
        self.add_project_button.setEnabled(not running)
        self.reload_presets_button.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.queue_list.setDragEnabled(not running)
        self.queue_list.setAcceptDrops(not running)
        for widget in self._widgets_by_id.values():
            widget.set_locked(running)

    def _start_queue(self) -> None:
        if self.queue_list.count() == 0:
            QMessageBox.information(
                self,
                "Representative structure batch queue",
                "Add at least one project before running the queue.",
            )
            return
        try:
            entries = self.queue_jobs_in_order()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid representative structure batch settings",
                str(exc),
            )
            return

        self.console.clear()
        self._set_running(True)
        self.queue_status_label.setText(
            f"Running 0/{len(entries)} queued project(s)"
        )
        for widget in self._widgets_by_id.values():
            widget.set_progress(0, 1)
            widget.set_status("Queued")

        self._run_thread = QThread(self)
        self._run_worker = RepresentativeFinderBatchWorker(entries)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.item_started.connect(self._on_item_started)
        self._run_worker.item_progress.connect(self._on_item_progress)
        self._run_worker.item_finished.connect(self._on_item_finished)
        self._run_worker.item_failed.connect(self._on_item_failed)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.status.connect(self._on_status)
        self._run_worker.project_results_changed.connect(
            self.project_results_changed.emit
        )
        self._run_worker.finished.connect(self._on_queue_finished)
        self._run_worker.failed.connect(self._on_queue_failed)
        self._run_worker.finished.connect(self._run_thread.quit)
        self._run_worker.failed.connect(self._run_thread.quit)
        self._run_thread.finished.connect(self._cleanup_run_thread)
        self._run_thread.finished.connect(self._run_thread.deleteLater)
        self._run_thread.start()

    def _request_cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.queue_status_label.setText(
            "Stopping queue after the active project finishes"
        )
        self._append_log(
            "Stop requested; the current project will finish before the "
            "queue exits."
        )
        if self._run_worker is not None:
            self._run_worker.request_cancel()

    def _append_log(self, message: str) -> None:
        self.console.append(message)

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.queue_status_label.setText(message)

    def _on_item_started(
        self,
        item_id: str,
        index: int,
        total: int,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_status(f"Running {index}/{total}")
            widget.set_progress(0, 1)
        self.queue_status_label.setText(
            f"Running {index}/{total} queued project(s)"
        )

    def _on_item_progress(
        self,
        item_id: str,
        processed: int,
        total: int,
        message: str,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_progress(processed, total)
            widget.set_status(message)

    def _on_item_finished(
        self,
        item_id: str,
        result: RepresentativeFinderBatchResult,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        widget.set_progress(
            result.completed_count, max(result.completed_count, 1)
        )
        widget.set_status(
            f"Complete: {result.completed_count} selected"
            + (
                ""
                if result.failed_count == 0
                else f", {result.failed_count} failed"
            )
        )

    def _on_item_failed(self, item_id: str, message: str) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_status("Failed")
        self._append_log(message)

    def _on_queue_finished(self, results: object) -> None:
        self._set_running(False)
        result_count = len(results) if isinstance(results, list) else 0
        self.queue_status_label.setText(
            f"Queue finished: {result_count} project(s) processed"
        )
        self.statusBar().showMessage(
            "Representative structure batch queue finished"
        )

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage(
            "Representative structure batch queue failed",
            5000,
        )
        QMessageBox.warning(
            self,
            "Representative structure batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_representativefinder_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_clusters_dir: str | Path | None = None,
    initial_input_path: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = RepresentativeFinderBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_clusters_dir=initial_clusters_dir,
        initial_input_path=initial_input_path,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "RepresentativeFinderBatchItem",
    "RepresentativeFinderBatchItemWidget",
    "RepresentativeFinderBatchJob",
    "RepresentativeFinderBatchQueueWindow",
    "RepresentativeFinderBatchResult",
    "RepresentativeFinderBatchWorker",
    "launch_representativefinder_batch_queue_ui",
]
