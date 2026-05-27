from __future__ import annotations

import gc
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis import (
    BondAnalysisPreset,
    BondAnalysisWorkflow,
    load_presets,
    ordered_preset_names,
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
from saxshell.structure_distributions import (
    application_structure_distribution_store_dir,
)

_CONSOLE_MAX_BLOCK_COUNT = 1200
_AUTOMATIC_PRESET_NAMES = ("DMSO", "DMF")


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
    clusters_dir = _required_path(text, "Clusters folder")
    if not clusters_dir.is_dir():
        raise ValueError(f"Clusters folder does not exist: {clusters_dir}")
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


def _safe_path_label(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() else "_" for char in str(value).strip()
    ).strip("_")
    return cleaned or "clusters"


def _suggest_batch_output_dir(project_dir: Path, clusters_dir: Path) -> Path:
    return (
        project_dir
        / "analysis"
        / "bondanalysis"
        / _safe_path_label(clusters_dir.name)
    )


def _preset_display_label(name: str, preset: BondAnalysisPreset) -> str:
    return f"{name} (Built-in)" if preset.builtin else name


def _bondanalysis_preset_name_for_project(
    project_dir: Path,
    *,
    project_name: str | None = None,
) -> str | None:
    search_text = " ".join(
        part
        for part in (project_name, project_dir.name, str(project_dir))
        if part
    ).casefold()
    for preset_name in _AUTOMATIC_PRESET_NAMES:
        if preset_name.casefold() in search_text:
            return preset_name
    return None


@dataclass(slots=True)
class BondAnalysisBatchQueueItem:
    item_id: str
    project_dir: Path | None = None
    clusters_dir: Path | None = None
    output_dir: Path | None = None
    preset_name: str | None = None


@dataclass(slots=True, frozen=True)
class BondAnalysisBatchJob:
    project_dir: Path
    clusters_dir: Path
    output_dir: Path
    preset_name: str
    preset: BondAnalysisPreset


@dataclass(slots=True, frozen=True)
class BondAnalysisBatchQueueResult:
    project_dir: Path
    clusters_dir: Path
    output_dir: Path
    results_index_path: Path
    preset_name: str
    csv_files: tuple[Path, ...]


@dataclass(slots=True, frozen=True)
class BondAnalysisBatchQueueSummary:
    results: tuple[BondAnalysisBatchQueueResult, ...]

    @property
    def csv_count(self) -> int:
        return sum(len(result.csv_files) for result in self.results)

    @property
    def output_dirs(self) -> tuple[Path, ...]:
        return tuple(result.output_dir for result in self.results)


def _queue_item_from_project_defaults(
    project_dir: str | Path,
    *,
    item_id: str | None = None,
) -> BondAnalysisBatchQueueItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = BondAnalysisBatchQueueItem(
        item_id=item_id or _new_item_id(),
        project_dir=resolved_project_dir,
        output_dir=None,
        preset_name=_bondanalysis_preset_name_for_project(
            resolved_project_dir
        ),
    )
    try:
        settings = SAXSProjectManager().load_project(resolved_project_dir)
    except Exception:
        return item
    clusters_dir = settings.resolved_clusters_dir
    output_dir = (
        None
        if clusters_dir is None
        else _suggest_batch_output_dir(resolved_project_dir, clusters_dir)
    )
    return replace(
        item,
        clusters_dir=clusters_dir,
        output_dir=output_dir,
        preset_name=(
            _bondanalysis_preset_name_for_project(
                resolved_project_dir,
                project_name=settings.project_name,
            )
            or item.preset_name
        ),
    )


class BondAnalysisBatchWorker(QObject):
    item_started = Signal(str, int, int)
    item_progress = Signal(str, int, int, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        queue_entries: list[tuple[str, BondAnalysisBatchJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[BondAnalysisBatchQueueResult] = []
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
            gc.collect()
        summary = BondAnalysisBatchQueueSummary(
            results=tuple(results),
        )
        self.status.emit("Bond analysis batch queue finished")
        self.finished.emit(summary)

    def _run_job(
        self,
        item_id: str,
        job: BondAnalysisBatchJob,
    ) -> BondAnalysisBatchQueueResult:
        self.log.emit(
            f"[{job.project_dir.name}] Starting bond analysis with preset "
            f"{job.preset_name}."
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

        workflow = BondAnalysisWorkflow(
            job.clusters_dir,
            bond_pairs=job.preset.bond_pairs,
            angle_triplets=job.preset.angle_triplets,
            coordination_numbers=job.preset.coordination_numbers,
            output_dir=job.output_dir,
            structure_distribution_store_dir=(
                application_structure_distribution_store_dir(
                    project_dir=job.project_dir,
                    application="bondanalysis",
                )
            ),
            generate_preview_plots=False,
        )
        batch_result = workflow.run(
            progress_callback=on_progress,
            log_callback=on_log,
        )
        csv_files = tuple(sorted(batch_result.output_dir.rglob("*.csv")))
        self.log.emit(
            f"[{job.project_dir.name}] Wrote {len(csv_files)} CSV file(s) "
            f"to project output folders."
        )
        self.log.emit(
            f"[{job.project_dir.name}] Output directory: "
            f"{batch_result.output_dir}"
        )
        self.log.emit(
            f"[{job.project_dir.name}] All-cluster tables: "
            f"{batch_result.output_dir / 'all_clusters'}"
        )
        self.log.emit(
            f"[{job.project_dir.name}] Per-cluster-type tables: "
            f"{batch_result.output_dir / 'cluster_types'}"
        )
        self.log.emit(
            f"[{job.project_dir.name}] Comparison tables: "
            f"{batch_result.output_dir / 'comparisons'}"
        )
        self.log.emit(
            f"[{job.project_dir.name}] Results index: "
            f"{batch_result.results_index_path}"
        )
        return BondAnalysisBatchQueueResult(
            project_dir=job.project_dir,
            clusters_dir=job.clusters_dir,
            output_dir=batch_result.output_dir,
            results_index_path=batch_result.results_index_path,
            preset_name=job.preset_name,
            csv_files=csv_files,
        )


class BondAnalysisBatchQueueWindow(QMainWindow):
    """Queue bond-analysis runs across multiple SAXSShell projects."""

    COL_PROJECT = 0
    COL_CLUSTERS = 1
    COL_OUTPUT = 2
    COL_PRESET = 3
    COL_STATUS = 4

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_clusters_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._items_by_id: dict[str, BondAnalysisBatchQueueItem] = {}
        self._run_thread: QThread | None = None
        self._run_worker: BondAnalysisBatchWorker | None = None
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
        item: BondAnalysisBatchQueueItem | None = None,
    ) -> BondAnalysisBatchQueueItem:
        resolved_item = item or BondAnalysisBatchQueueItem(
            item_id=_new_item_id()
        )
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        self._items_by_id[resolved_item.item_id] = resolved_item
        project_text = (
            ""
            if resolved_item.project_dir is None
            else str(resolved_item.project_dir)
        )
        project_item = QTableWidgetItem(project_text)
        project_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_table.setItem(row, self.COL_PROJECT, project_item)
        self.queue_table.setItem(
            row,
            self.COL_CLUSTERS,
            QTableWidgetItem(
                ""
                if resolved_item.clusters_dir is None
                else str(resolved_item.clusters_dir)
            ),
        )
        self.queue_table.setItem(
            row,
            self.COL_OUTPUT,
            QTableWidgetItem(
                ""
                if resolved_item.output_dir is None
                else str(resolved_item.output_dir)
            ),
        )
        preset_combo = self._preset_combo(resolved_item.preset_name)
        self.queue_table.setCellWidget(row, self.COL_PRESET, preset_combo)
        self.queue_table.setItem(
            row,
            self.COL_STATUS,
            QTableWidgetItem("Ready"),
        )
        self._resize_queue_table()
        return resolved_item

    def queue_jobs_in_order(self) -> list[tuple[str, BondAnalysisBatchJob]]:
        entries: list[tuple[str, BondAnalysisBatchJob]] = []
        for row in range(self.queue_table.rowCount()):
            item_id = self._row_item_id(row)
            if item_id is None:
                continue
            project_dir = _required_project_dir(
                self._table_text(row, self.COL_PROJECT)
            )
            clusters_dir = _required_clusters_dir(
                self._table_text(row, self.COL_CLUSTERS)
            )
            output_dir = _optional_path(
                self._table_text(row, self.COL_OUTPUT)
            ) or _suggest_batch_output_dir(project_dir, clusters_dir)
            preset_name = self._row_preset_name(row)
            if preset_name is None:
                raise ValueError(
                    f"Choose a bondanalysis preset for row {row + 1}."
                )
            preset = self._presets.get(preset_name)
            if preset is None:
                raise ValueError(f"Unknown preset: {preset_name}")
            entries.append(
                (
                    item_id,
                    BondAnalysisBatchJob(
                        project_dir=project_dir,
                        clusters_dir=clusters_dir,
                        output_dir=output_dir,
                        preset_name=preset_name,
                        preset=preset,
                    ),
                )
            )
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell Bond Analysis Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1180, 780)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls = QHBoxLayout()
        self.add_current_button = QPushButton("Add Current Project")
        self.add_current_button.clicked.connect(self._add_current_project)
        controls.addWidget(self.add_current_button)
        self.add_projects_button = QPushButton("Add Projects...")
        self.add_projects_button.clicked.connect(self._choose_projects_to_add)
        controls.addWidget(self.add_projects_button)
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self._remove_selected_rows)
        controls.addWidget(self.remove_selected_button)
        controls.addStretch(1)
        root.addLayout(controls)

        preset_group = QGroupBox("Batch Presets")
        preset_layout = QFormLayout(preset_group)
        preset_row = QHBoxLayout()
        self.global_preset_combo = QComboBox()
        preset_row.addWidget(self.global_preset_combo, stretch=1)
        self.apply_preset_all_button = QPushButton("Apply Preset to All")
        self.apply_preset_all_button.clicked.connect(self._apply_preset_to_all)
        preset_row.addWidget(self.apply_preset_all_button)
        self.reload_presets_button = QPushButton("Reload Presets")
        self.reload_presets_button.clicked.connect(self._reload_presets)
        preset_row.addWidget(self.reload_presets_button)
        preset_widget = QWidget()
        preset_widget.setLayout(preset_row)
        preset_layout.addRow("Bondanalysis preset", preset_widget)
        self.preset_summary_label = QLabel()
        self.preset_summary_label.setWordWrap(True)
        preset_layout.addRow("", self.preset_summary_label)
        root.addWidget(preset_group)

        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["Project", "Clusters", "Output", "Preset", "Status"]
        )
        self.queue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.queue_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.verticalHeader().setVisible(False)
        root.addWidget(self.queue_table, stretch=1)

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
        self.queue_progress_bar = QProgressBar()
        self.queue_progress_bar.setRange(0, 1)
        self.queue_progress_bar.setValue(0)
        self.queue_progress_bar.setFormat("%v / %m projects")
        run_layout.addWidget(self.queue_progress_bar)
        self.queue_status_label = QLabel("Queue idle")
        run_layout.addWidget(self.queue_status_label)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(150)
        self.console.document().setMaximumBlockCount(_CONSOLE_MAX_BLOCK_COUNT)
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _preset_combo(self, selected_name: str | None = None) -> QComboBox:
        combo = QComboBox()
        self._populate_preset_combo(combo, selected_name=selected_name)
        return combo

    def _populate_preset_combo(
        self,
        combo: QComboBox,
        *,
        selected_name: str | None = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        selected_index = -1
        for index, name in enumerate(ordered_preset_names(self._presets)):
            combo.addItem(
                _preset_display_label(name, self._presets[name]),
                name,
            )
            if name == selected_name:
                selected_index = index
        if selected_index < 0 and combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _reload_presets(self) -> None:
        self._presets = load_presets()
        current_global = self.global_preset_combo.currentData()
        selected_global = (
            None if current_global is None else str(current_global)
        )
        self._populate_preset_combo(
            self.global_preset_combo,
            selected_name=selected_global,
        )
        for row in range(self.queue_table.rowCount()):
            selected_name = self._row_preset_name(row)
            combo = self.queue_table.cellWidget(row, self.COL_PRESET)
            if isinstance(combo, QComboBox):
                self._populate_preset_combo(combo, selected_name=selected_name)
        self._refresh_preset_summary()

    def _refresh_preset_summary(self) -> None:
        preset_names = ordered_preset_names(self._presets)
        if not preset_names:
            self.preset_summary_label.setText(
                "No Bondanalysis presets are available. Reload presets after "
                "creating or restoring the preset file."
            )
            return
        labels = [
            _preset_display_label(name, self._presets[name])
            for name in preset_names
        ]
        self.preset_summary_label.setText(
            "Available presets: "
            + ", ".join(labels)
            + ". Use Apply Preset to All when every queued project should "
            "use the same bond, angle, and coordination definitions."
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
            else BondAnalysisBatchQueueItem(item_id=_new_item_id())
        )
        if self._initial_clusters_dir is not None:
            output_dir = item.output_dir
            if self._initial_project_dir is not None:
                output_dir = _suggest_batch_output_dir(
                    self._initial_project_dir,
                    self._initial_clusters_dir,
                )
            item = replace(
                item,
                clusters_dir=self._initial_clusters_dir,
                output_dir=output_dir,
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
        progress_dialog = self._project_load_progress_dialog(
            len(selected_dirs)
        )
        try:
            for index, project_dir in enumerate(selected_dirs, start=1):
                if progress_dialog is not None:
                    progress_dialog.setLabelText(
                        "Loading bond-analysis project "
                        f"{index}/{len(selected_dirs)}:\n{project_dir}"
                    )
                    progress_dialog.setValue(index - 1)
                    QApplication.processEvents()
                self.add_queue_item(
                    _queue_item_from_project_defaults(project_dir)
                )
                if progress_dialog is not None:
                    progress_dialog.setValue(index)
                    QApplication.processEvents()
        finally:
            if progress_dialog is not None:
                progress_dialog.setValue(len(selected_dirs))
                progress_dialog.close()

    def _project_load_progress_dialog(
        self,
        project_count: int,
    ) -> QProgressDialog | None:
        if project_count <= 1:
            return None
        dialog = QProgressDialog(
            "Loading selected bond-analysis projects...",
            None,
            0,
            project_count,
            self,
        )
        dialog.setWindowTitle("Loading Bond Analysis Projects")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

    def _apply_preset_to_all(self) -> None:
        preset_name = self.global_preset_combo.currentData()
        if preset_name is None:
            return
        for row in range(self.queue_table.rowCount()):
            combo = self.queue_table.cellWidget(row, self.COL_PRESET)
            if isinstance(combo, QComboBox):
                index = combo.findData(str(preset_name))
                if index >= 0:
                    combo.setCurrentIndex(index)

    def _remove_selected_rows(self) -> None:
        if self._run_thread is not None and self._run_thread.isRunning():
            return
        rows = sorted(
            {index.row() for index in self.queue_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            item_id = self._row_item_id(row)
            if item_id is not None:
                self._items_by_id.pop(item_id, None)
            self.queue_table.removeRow(row)
        self._resize_queue_table()

    def _row_item_id(self, row: int) -> str | None:
        item = self.queue_table.item(row, self.COL_PROJECT)
        if item is None:
            return None
        payload = item.data(Qt.ItemDataRole.UserRole)
        return None if payload is None else str(payload)

    def _row_preset_name(self, row: int) -> str | None:
        combo = self.queue_table.cellWidget(row, self.COL_PRESET)
        if not isinstance(combo, QComboBox):
            return None
        payload = combo.currentData()
        return None if payload is None else str(payload)

    def _table_text(self, row: int, column: int) -> str:
        item = self.queue_table.item(row, column)
        return "" if item is None else item.text().strip()

    def _resize_queue_table(self) -> None:
        self.queue_table.resizeColumnsToContents()
        self.queue_table.horizontalHeader().setStretchLastSection(True)

    def _set_running(self, running: bool) -> None:
        self.add_current_button.setEnabled(not running)
        self.add_projects_button.setEnabled(not running)
        self.remove_selected_button.setEnabled(not running)
        self.reload_presets_button.setEnabled(not running)
        self.apply_preset_all_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.queue_table.setEnabled(not running)

    def _start_queue(self) -> None:
        if self.queue_table.rowCount() == 0:
            QMessageBox.information(
                self,
                "Bond analysis batch queue",
                "Add at least one project before running the queue.",
            )
            return
        try:
            entries = self.queue_jobs_in_order()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid bond analysis batch settings",
                str(exc),
            )
            return

        self.console.clear()
        self._set_running(True)
        self.queue_progress_bar.setRange(0, max(len(entries), 1))
        self.queue_progress_bar.setValue(0)
        self.queue_status_label.setText(
            f"Running 0/{len(entries)} queued project(s)"
        )
        for row in range(self.queue_table.rowCount()):
            self.queue_table.item(row, self.COL_STATUS).setText("Queued")

        self._run_thread = QThread(self)
        self._run_worker = BondAnalysisBatchWorker(entries)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.item_started.connect(self._on_item_started)
        self._run_worker.item_progress.connect(self._on_item_progress)
        self._run_worker.item_finished.connect(self._on_item_finished)
        self._run_worker.item_failed.connect(self._on_item_failed)
        self._run_worker.log.connect(self._append_log)
        self._run_worker.status.connect(self._on_status)
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

    def _row_for_item_id(self, item_id: str) -> int | None:
        for row in range(self.queue_table.rowCount()):
            if self._row_item_id(row) == item_id:
                return row
        return None

    def _on_item_started(
        self,
        item_id: str,
        index: int,
        total: int,
    ) -> None:
        row = self._row_for_item_id(item_id)
        if row is not None:
            self.queue_table.item(row, self.COL_STATUS).setText(
                f"Running {index}/{total}"
            )
        self.queue_progress_bar.setRange(0, total)
        self.queue_progress_bar.setValue(index - 1)
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
        row = self._row_for_item_id(item_id)
        if row is not None:
            self.queue_table.item(row, self.COL_STATUS).setText(message)
        del processed, total

    def _on_item_finished(
        self,
        item_id: str,
        result: BondAnalysisBatchQueueResult,
    ) -> None:
        row = self._row_for_item_id(item_id)
        if row is not None:
            self.queue_table.item(row, self.COL_STATUS).setText(
                f"Complete: {len(result.csv_files)} CSV files"
            )
        self.queue_progress_bar.setValue(self.queue_progress_bar.value() + 1)

    def _on_item_failed(self, item_id: str, message: str) -> None:
        row = self._row_for_item_id(item_id)
        if row is not None:
            self.queue_table.item(row, self.COL_STATUS).setText("Failed")
        self._append_log(message)

    def _on_queue_finished(self, payload: object) -> None:
        self._set_running(False)
        if isinstance(payload, BondAnalysisBatchQueueSummary):
            self.queue_status_label.setText(
                "Queue finished: "
                f"{len(payload.results)} project(s), "
                f"{payload.csv_count} CSV file(s)."
            )
            self.statusBar().showMessage("Bond analysis batch queue finished")
            if payload.output_dirs:
                self._append_log("Batch output locations:")
                for output_dir in payload.output_dirs:
                    self._append_log(f"  {output_dir}")
        else:
            self.queue_status_label.setText("Queue finished")

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage("Bond analysis batch queue failed", 5000)
        QMessageBox.warning(
            self,
            "Bond analysis batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_bondanalysis_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_clusters_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = BondAnalysisBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_clusters_dir=initial_clusters_dir,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "BondAnalysisBatchJob",
    "BondAnalysisBatchQueueItem",
    "BondAnalysisBatchQueueResult",
    "BondAnalysisBatchQueueSummary",
    "BondAnalysisBatchQueueWindow",
    "BondAnalysisBatchWorker",
    "launch_bondanalysis_batch_queue_ui",
]
