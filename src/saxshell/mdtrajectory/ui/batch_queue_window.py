from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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

from saxshell.mdtrajectory.frame.manager import DEFAULT_FRAME_TIMESTEP_FS
from saxshell.mdtrajectory.workflow import MDTrajectoryWorkflow
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)

DEFAULT_TIME_CUTOFF_FS = 1000.0


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


def _required_existing_file(text: str, field_name: str) -> Path:
    path = _required_path(text, field_name)
    if not path.is_file():
        raise ValueError(f"{field_name} does not exist: {path}")
    return path


def _required_project_dir(text: str) -> Path:
    project_dir = _required_path(text, "Project folder")
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        raise ValueError(f"Project file does not exist: {project_file}")
    return project_dir


def _project_reference_text(project_dir: Path | None) -> str:
    if project_dir is None:
        return "Project reference: choose a SAXSShell project folder."
    project_file = build_project_paths(project_dir).project_file
    if project_file.is_file():
        return f"Project reference: {project_file}"
    return f"Project reference: no project file found at {project_file}"


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


@dataclass(slots=True, frozen=True)
class MDTrajectoryBatchJob:
    project_dir: Path
    trajectory_file: Path
    topology_file: Path | None
    energy_file: Path
    output_dir: Path | None = None
    cutoff_fs: float = DEFAULT_TIME_CUTOFF_FS
    frame_timestep_fs: float = DEFAULT_FRAME_TIMESTEP_FS
    use_manual_frame_timestep: bool = False
    include_restart_duplicates: bool = False


@dataclass(slots=True)
class MDTrajectoryBatchResult:
    project_dir: Path
    output_dir: Path
    written_count: int
    selected_frames: int
    cutoff_fs: float
    frame_timestep_fs: float = DEFAULT_FRAME_TIMESTEP_FS
    use_manual_frame_timestep: bool = False
    metadata_file: Path | None = None
    include_restart_duplicates: bool = False


@dataclass(slots=True)
class MDTrajectoryBatchItem:
    item_id: str
    project_dir: Path | None = None
    trajectory_file: Path | None = None
    topology_file: Path | None = None
    energy_file: Path | None = None
    output_dir: Path | None = None
    cutoff_fs: float = DEFAULT_TIME_CUTOFF_FS
    frame_timestep_fs: float = DEFAULT_FRAME_TIMESTEP_FS
    use_manual_frame_timestep: bool = False
    include_restart_duplicates: bool = False

    def display_name(self) -> str:
        if self.project_dir is not None:
            return self.project_dir.name
        if self.trajectory_file is not None:
            return self.trajectory_file.name
        return "New MD trajectory extraction"

    def to_job(self) -> MDTrajectoryBatchJob:
        project_dir = _required_project_dir(
            "" if self.project_dir is None else str(self.project_dir)
        )
        trajectory_file = _required_existing_file(
            "" if self.trajectory_file is None else str(self.trajectory_file),
            "Trajectory file",
        )
        topology_file = None
        if self.topology_file is not None:
            topology_file = _required_existing_file(
                str(self.topology_file),
                "Topology file",
            )
        energy_file = _required_existing_file(
            "" if self.energy_file is None else str(self.energy_file),
            "Energy file",
        )
        output_dir = None
        if self.output_dir is not None:
            output_dir = self.output_dir.expanduser().resolve()
            if output_dir.exists() and not output_dir.is_dir():
                raise ValueError(
                    f"Output folder exists but is not a directory: {output_dir}"
                )
        cutoff_fs = float(self.cutoff_fs)
        if cutoff_fs < 0.0:
            raise ValueError("Time cutoff must be zero or greater.")
        frame_timestep_fs = float(self.frame_timestep_fs)
        if frame_timestep_fs <= 0.0:
            raise ValueError("Frame timestep must be greater than zero.")
        return MDTrajectoryBatchJob(
            project_dir=project_dir,
            trajectory_file=trajectory_file,
            topology_file=topology_file,
            energy_file=energy_file,
            output_dir=output_dir,
            cutoff_fs=cutoff_fs,
            frame_timestep_fs=frame_timestep_fs,
            use_manual_frame_timestep=self.use_manual_frame_timestep,
            include_restart_duplicates=self.include_restart_duplicates,
        )


def _queue_item_from_project_defaults(
    project_dir: str | Path,
    *,
    item_id: str | None = None,
) -> MDTrajectoryBatchItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = MDTrajectoryBatchItem(
        item_id=item_id or _new_item_id(),
        project_dir=resolved_project_dir,
    )
    try:
        settings = SAXSProjectManager().load_project(resolved_project_dir)
    except Exception:
        return item
    return replace(
        item,
        trajectory_file=settings.resolved_trajectory_file,
        topology_file=settings.resolved_topology_file,
        energy_file=settings.resolved_energy_file,
    )


class MDTrajectoryBatchItemWidget(QFrame):
    settings_changed = Signal(str)
    remove_requested = Signal(str)
    duplicate_requested = Signal(str)

    def __init__(
        self,
        item: MDTrajectoryBatchItem,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._loading = False
        self._locked = False
        self._selected = False
        self._last_suggested_output_dir: Path | None = None
        self._build_ui()
        self._load_item(item)
        self._set_settings_visible(False)

    @property
    def item_id(self) -> str:
        return self._item.item_id

    def item(self) -> MDTrajectoryBatchItem:
        return self._item

    def collect_item(self) -> MDTrajectoryBatchItem:
        self._item = MDTrajectoryBatchItem(
            item_id=self._item.item_id,
            project_dir=_optional_path(self.project_dir_edit.text()),
            trajectory_file=_optional_path(self.trajectory_file_edit.text()),
            topology_file=_optional_path(self.topology_file_edit.text()),
            energy_file=_optional_path(self.energy_file_edit.text()),
            output_dir=_optional_path(self.output_dir_edit.text()),
            cutoff_fs=float(self.cutoff_spin.value()),
            frame_timestep_fs=float(self.timestep_spin.value()),
            use_manual_frame_timestep=(not self.auto_timestep_box.isChecked()),
            include_restart_duplicates=(
                self.include_restart_duplicates_box.isChecked()
            ),
        )
        self._refresh_header()
        self._refresh_project_reference()
        return self._item

    def job(self) -> MDTrajectoryBatchJob:
        return self.collect_item().to_job()

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        self.settings_group.setEnabled(not locked)
        self.preview_button.setEnabled(not locked)
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
            "QFrame#MDTrajectoryBatchItemHeader {"
            + (
                "background-color: #dce8f7; " "border: 1px solid #8fb0d7;"
                if self._selected
                else "background-color: #f6f8fb; " "border: 1px solid #cfd7e3;"
            )
            + "border-radius: 5px;}"
        )

    def preview_selection(self) -> None:
        job = self.job()
        workflow = MDTrajectoryWorkflow(
            trajectory_file=job.trajectory_file,
            topology_file=job.topology_file,
            energy_file=job.energy_file,
            include_restart_duplicates=job.include_restart_duplicates,
            frame_timestep_fs=job.frame_timestep_fs,
            use_inferred_frame_times=job.use_manual_frame_timestep,
        )
        current_output_dir = _optional_path(self.output_dir_edit.text())
        use_suggested_output_dir = (
            current_output_dir is None
            or current_output_dir == self._last_suggested_output_dir
        )
        selection = workflow.preview_selection(
            use_cutoff=True,
            cutoff_fs=job.cutoff_fs,
            output_dir=(
                None if use_suggested_output_dir else current_output_dir
            ),
        )
        detected_timestep = selection.preview.detected_frame_timestep_fs
        if detected_timestep is not None and not job.use_manual_frame_timestep:
            self._set_timestep_value(float(detected_timestep))
            job = replace(job, frame_timestep_fs=float(detected_timestep))
        if use_suggested_output_dir:
            self._last_suggested_output_dir = selection.output_dir.resolve()
            self.output_dir_edit.setText(str(selection.output_dir))
            job = replace(job, output_dir=selection.output_dir.resolve())
        preview = selection.preview
        lines = [
            f"Frames selected: {preview.selected_frames} / "
            f"{preview.total_frames}",
            f"Output folder: {selection.output_dir}",
            f"Applied cutoff: {job.cutoff_fs:g} fs",
            "Frame timestep: "
            f"{job.frame_timestep_fs:g} fs "
            f"({'manual' if job.use_manual_frame_timestep else 'auto'})",
            "Restart duplicate frames: "
            f"{'included' if job.include_restart_duplicates else 'skipped'}",
        ]
        if preview.first_frame_index is not None:
            lines.append(
                "Frame index range: "
                f"{preview.first_frame_index} to {preview.last_frame_index}"
            )
        if preview.first_time_fs is not None:
            lines.append(
                "Time range: "
                f"{preview.first_time_fs:.3f} fs to "
                f"{preview.last_time_fs:.3f} fs"
            )
        self.preview_summary_label.setText("\n".join(lines))
        self.set_progress(0, max(preview.selected_frames, 1))
        self.set_status("Preview ready")

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
        self.header_frame.setObjectName("MDTrajectoryBatchItemHeader")
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(8, 6, 8, 6)
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._set_settings_visible)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("New MD trajectory extraction")
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(180)
        header.addWidget(self.status_label)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self._preview_from_button)
        header.addWidget(self.preview_button)
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
        self.progress_bar.setFormat("%v / %m frames")
        root.addWidget(self.progress_bar)

        self.settings_group = QFrame()
        self.settings_group.setFrameShape(QFrame.Shape.StyledPanel)
        root.addWidget(self.settings_group)
        form = QFormLayout(self.settings_group)

        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_dir_edit = QLineEdit()
        self.project_dir_edit.editingFinished.connect(self._on_editor_changed)
        project_layout.addWidget(self.project_dir_edit, stretch=1)
        project_button = QPushButton("Browse...")
        project_button.clicked.connect(self._choose_project_dir)
        project_layout.addWidget(project_button)
        form.addRow("Project folder", project_row)

        self.project_reference_label = QLabel()
        self.project_reference_label.setWordWrap(True)
        self.project_reference_label.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", self.project_reference_label)

        trajectory_row = QWidget()
        trajectory_layout = QHBoxLayout(trajectory_row)
        trajectory_layout.setContentsMargins(0, 0, 0, 0)
        self.trajectory_file_edit = QLineEdit()
        self.trajectory_file_edit.editingFinished.connect(
            self._on_editor_changed
        )
        trajectory_layout.addWidget(self.trajectory_file_edit, stretch=1)
        trajectory_button = QPushButton("Browse...")
        trajectory_button.clicked.connect(self._choose_trajectory_file)
        trajectory_layout.addWidget(trajectory_button)
        form.addRow("Trajectory file", trajectory_row)

        topology_row = QWidget()
        topology_layout = QHBoxLayout(topology_row)
        topology_layout.setContentsMargins(0, 0, 0, 0)
        self.topology_file_edit = QLineEdit()
        self.topology_file_edit.editingFinished.connect(
            self._on_editor_changed
        )
        topology_layout.addWidget(self.topology_file_edit, stretch=1)
        topology_button = QPushButton("Browse...")
        topology_button.clicked.connect(self._choose_topology_file)
        topology_layout.addWidget(topology_button)
        form.addRow("Topology file", topology_row)

        energy_row = QWidget()
        energy_layout = QHBoxLayout(energy_row)
        energy_layout.setContentsMargins(0, 0, 0, 0)
        self.energy_file_edit = QLineEdit()
        self.energy_file_edit.editingFinished.connect(self._on_editor_changed)
        energy_layout.addWidget(self.energy_file_edit, stretch=1)
        energy_button = QPushButton("Browse...")
        energy_button.clicked.connect(self._choose_energy_file)
        energy_layout.addWidget(energy_button)
        form.addRow("Energy file", energy_row)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setToolTip(
            "Target folder for this project's extracted XYZ frames. Leave "
            "blank to use the preview-generated default."
        )
        self.output_dir_edit.editingFinished.connect(self._on_editor_changed)
        output_layout.addWidget(self.output_dir_edit, stretch=1)
        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self._choose_output_dir)
        output_layout.addWidget(output_button)
        form.addRow("Output folder", output_row)

        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.0, 1.0e12)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setSingleStep(100.0)
        self.cutoff_spin.setSuffix(" fs")
        self.cutoff_spin.setValue(DEFAULT_TIME_CUTOFF_FS)
        self.cutoff_spin.valueChanged.connect(self._on_editor_changed)
        form.addRow("Time cutoff", self.cutoff_spin)

        self.auto_timestep_box = QCheckBox(
            "Auto-calculate frame timestep from trajectory"
        )
        self.auto_timestep_box.setChecked(True)
        self.auto_timestep_box.setToolTip(
            "Use trajectory time metadata when available. The frame "
            "timestep field is updated during preview and used as a "
            "fallback when the trajectory has no frame times."
        )
        self.auto_timestep_box.toggled.connect(self._on_editor_changed)
        form.addRow("", self.auto_timestep_box)

        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(1.0e-9, 1.0e12)
        self.timestep_spin.setDecimals(6)
        self.timestep_spin.setSingleStep(0.5)
        self.timestep_spin.setSuffix(" fs")
        self.timestep_spin.setValue(DEFAULT_FRAME_TIMESTEP_FS)
        self.timestep_spin.valueChanged.connect(self._handle_timestep_changed)
        form.addRow("Frame timestep", self.timestep_spin)

        self.include_restart_duplicates_box = QCheckBox(
            "Include duplicate restart frames"
        )
        self.include_restart_duplicates_box.setToolTip(
            "Export duplicate frames from overlapping simulation restarts. "
            "Leave this off for the cleaned continuation trajectory."
        )
        self.include_restart_duplicates_box.toggled.connect(
            self._on_editor_changed
        )
        form.addRow("", self.include_restart_duplicates_box)

        self.preview_summary_label = QLabel(
            "Preview the trajectory to verify the generated output folder."
        )
        self.preview_summary_label.setWordWrap(True)
        self.preview_summary_label.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", self.preview_summary_label)

    def _load_item(self, item: MDTrajectoryBatchItem) -> None:
        self._loading = True
        self.project_dir_edit.setText(
            "" if item.project_dir is None else str(item.project_dir)
        )
        self.trajectory_file_edit.setText(
            "" if item.trajectory_file is None else str(item.trajectory_file)
        )
        self.topology_file_edit.setText(
            "" if item.topology_file is None else str(item.topology_file)
        )
        self.energy_file_edit.setText(
            "" if item.energy_file is None else str(item.energy_file)
        )
        self.output_dir_edit.setText(
            "" if item.output_dir is None else str(item.output_dir)
        )
        if item.output_dir is not None:
            self._last_suggested_output_dir = item.output_dir.resolve()
        self.cutoff_spin.setValue(float(item.cutoff_fs))
        self._set_timestep_value(float(item.frame_timestep_fs))
        self.auto_timestep_box.setChecked(not item.use_manual_frame_timestep)
        self.include_restart_duplicates_box.setChecked(
            item.include_restart_duplicates
        )
        self._loading = False
        self._refresh_header()
        self._refresh_project_reference()

    def _set_timestep_value(self, value: float) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            self.timestep_spin.setValue(float(value))
        finally:
            self._loading = was_loading

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
        project_dir = Path(selected).expanduser().resolve()
        self._load_item(
            replace(
                _queue_item_from_project_defaults(
                    project_dir,
                    item_id=self.item_id,
                ),
                cutoff_fs=float(self.cutoff_spin.value()),
                frame_timestep_fs=float(self.timestep_spin.value()),
                use_manual_frame_timestep=(
                    not self.auto_timestep_box.isChecked()
                ),
                include_restart_duplicates=(
                    self.include_restart_duplicates_box.isChecked()
                ),
            )
        )
        self._on_editor_changed()

    def _choose_trajectory_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select trajectory file",
            _dialog_start_dir(
                self.trajectory_file_edit.text(),
                self.project_dir_edit.text(),
            ),
            "Trajectory files (*.xyz *.pdb);;All files (*)",
        )
        if not selected:
            return
        self.trajectory_file_edit.setText(selected)
        self._on_editor_changed()

    def _choose_topology_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select topology file",
            _dialog_start_dir(
                self.topology_file_edit.text(),
                self.project_dir_edit.text(),
            ),
            "Topology files (*.pdb *.gro *.top *.psf);;All files (*)",
        )
        if not selected:
            return
        self.topology_file_edit.setText(selected)
        self._on_editor_changed()

    def _choose_energy_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select CP2K energy file",
            _dialog_start_dir(
                self.energy_file_edit.text(),
                self.project_dir_edit.text(),
            ),
            "Energy files (*.ener *.out *.txt);;All files (*)",
        )
        if not selected:
            return
        self.energy_file_edit.setText(selected)
        self._on_editor_changed()

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for extracted XYZ frames",
            _dialog_start_dir(
                self.output_dir_edit.text(),
                self.trajectory_file_edit.text(),
                self.project_dir_edit.text(),
            ),
        )
        if not selected:
            return
        self.output_dir_edit.setText(selected)
        self._on_editor_changed()

    def _preview_from_button(self) -> None:
        try:
            self.preview_selection()
        except Exception as exc:
            QMessageBox.warning(self, "Unable to preview trajectory", str(exc))
            self.preview_summary_label.setText(str(exc))
            self.set_status("Preview failed")
            self._on_editor_changed()

    def _handle_timestep_changed(self, _value: float) -> None:
        if self._loading:
            return
        if self.auto_timestep_box.isChecked():
            self.auto_timestep_box.setChecked(False)
        self._on_editor_changed()

    def _on_editor_changed(self, *_args) -> None:
        if self._loading:
            return
        try:
            self.collect_item()
            self.set_status("Ready")
        except Exception:
            self._refresh_header()
            self._refresh_project_reference()
        self.settings_changed.emit(self.item_id)

    def _refresh_header(self) -> None:
        self.title_label.setText(self._item.display_name())

    def _refresh_project_reference(self) -> None:
        project_dir = _optional_path(self.project_dir_edit.text())
        self.project_reference_label.setText(
            _project_reference_text(project_dir)
        )


class MDTrajectoryBatchWorker(QObject):
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
        queue_entries: list[tuple[str, MDTrajectoryBatchJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()
        self._project_manager = SAXSProjectManager()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[MDTrajectoryBatchResult] = []
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
            self.log.emit(f"Starting {index}/{total_items}: {job.project_dir}")
            try:
                result = self._run_job(item_id, job)
            except Exception as exc:
                message = str(exc)
                self.item_failed.emit(item_id, message)
                self.failed.emit(item_id, message)
                return
            results.append(result)
            self.item_finished.emit(item_id, result)
        self.status.emit("MD trajectory batch queue finished")
        self.finished.emit(results)

    def _run_job(
        self,
        item_id: str,
        job: MDTrajectoryBatchJob,
    ) -> MDTrajectoryBatchResult:
        settings = self._project_manager.load_project(job.project_dir)
        workflow = MDTrajectoryWorkflow(
            trajectory_file=job.trajectory_file,
            topology_file=job.topology_file,
            energy_file=job.energy_file,
            include_restart_duplicates=job.include_restart_duplicates,
            frame_timestep_fs=job.frame_timestep_fs,
            use_inferred_frame_times=job.use_manual_frame_timestep,
        )
        self.item_progress.emit(
            item_id,
            0,
            1,
            "Inspecting trajectory",
        )
        selection = workflow.preview_selection(
            use_cutoff=True,
            cutoff_fs=job.cutoff_fs,
            output_dir=job.output_dir,
        )
        summary = workflow.inspect()
        self.log.emit(
            f"[{job.project_dir.name}] Selected "
            f"{selection.preview.selected_frames} of "
            f"{selection.preview.total_frames} frame(s); output "
            f"{selection.output_dir}"
        )
        duplicate_source_frames = int(
            summary.get("duplicate_source_frames", 0)
        )
        if duplicate_source_frames and job.include_restart_duplicates:
            self.log.emit(
                f"[{job.project_dir.name}] Included "
                f"{duplicate_source_frames} duplicate source frame(s) from "
                "overlapping trajectory chunks."
            )
        elif duplicate_source_frames:
            self.log.emit(
                f"[{job.project_dir.name}] Skipped "
                f"{duplicate_source_frames} duplicate source frame(s) from "
                "overlapping trajectory chunks."
            )
        export_result = workflow.export_frames(
            use_cutoff=True,
            cutoff_fs=job.cutoff_fs,
            output_dir=job.output_dir,
            progress_callback=(
                lambda processed, total, message: self.item_progress.emit(
                    item_id,
                    processed,
                    total,
                    message,
                )
            ),
        )
        settings.trajectory_file = str(job.trajectory_file)
        settings.topology_file = (
            None if job.topology_file is None else str(job.topology_file)
        )
        settings.energy_file = str(job.energy_file)
        settings.frames_dir = str(export_result.output_dir)
        self._project_manager.save_project(settings)
        self.log.emit(
            f"[{job.project_dir.name}] Registered XYZ frames folder: "
            f"{export_result.output_dir}"
        )
        return MDTrajectoryBatchResult(
            project_dir=job.project_dir,
            output_dir=export_result.output_dir,
            written_count=len(export_result.written_files),
            selected_frames=export_result.selection.preview.selected_frames,
            cutoff_fs=job.cutoff_fs,
            frame_timestep_fs=job.frame_timestep_fs,
            use_manual_frame_timestep=job.use_manual_frame_timestep,
            include_restart_duplicates=job.include_restart_duplicates,
            metadata_file=export_result.metadata_file,
        )


class MDTrajectoryBatchQueueWindow(QMainWindow):
    """Queue MD trajectory frame extraction for multiple projects."""

    project_paths_registered = Signal(object)

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_trajectory_file: str | Path | None = None,
        initial_topology_file: str | Path | None = None,
        initial_energy_file: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets_by_id: dict[str, MDTrajectoryBatchItemWidget] = {}
        self._run_thread: QThread | None = None
        self._run_worker: MDTrajectoryBatchWorker | None = None
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_trajectory_file = (
            None
            if initial_trajectory_file is None
            else Path(initial_trajectory_file).expanduser().resolve()
        )
        self._initial_topology_file = (
            None
            if initial_topology_file is None
            else Path(initial_topology_file).expanduser().resolve()
        )
        self._initial_energy_file = (
            None
            if initial_energy_file is None
            else Path(initial_energy_file).expanduser().resolve()
        )
        self._build_ui()
        if (
            self._initial_project_dir is not None
            or self._initial_trajectory_file is not None
            or self._initial_energy_file is not None
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
        item: MDTrajectoryBatchItem | None = None,
    ) -> MDTrajectoryBatchItemWidget:
        resolved_item = item or MDTrajectoryBatchItem(item_id=_new_item_id())
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_list.addItem(list_item)
        widget = MDTrajectoryBatchItemWidget(
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

    def queue_jobs_in_order(self) -> list[tuple[str, MDTrajectoryBatchJob]]:
        entries: list[tuple[str, MDTrajectoryBatchJob]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.job()))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell MD Trajectory Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1080, 820)

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

        run_group = QFrame()
        run_group.setFrameShape(QFrame.Shape.StyledPanel)
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
        self.console.setMinimumHeight(160)
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _add_current_project(self) -> None:
        if (
            self._initial_project_dir is None
            and self._initial_trajectory_file is None
            and self._initial_energy_file is None
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
            else MDTrajectoryBatchItem(item_id=_new_item_id())
        )
        self.add_queue_item(
            replace(
                item,
                trajectory_file=(
                    self._initial_trajectory_file or item.trajectory_file
                ),
                topology_file=(
                    self._initial_topology_file or item.topology_file
                ),
                energy_file=self._initial_energy_file or item.energy_file,
            )
        )

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
                "MD trajectory batch queue",
                "Add at least one project before running the queue.",
            )
            return
        try:
            entries = self.queue_jobs_in_order()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid MD trajectory batch settings",
                str(exc),
            )
            return

        self.console.clear()
        self._set_running(True)
        self.queue_status_label.setText(
            f"Running 0/{len(entries)} queued extraction(s)"
        )
        for widget in self._widgets_by_id.values():
            widget.set_progress(0, 1)
            widget.set_status("Queued")

        self._run_thread = QThread(self)
        self._run_worker = MDTrajectoryBatchWorker(entries)
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
            f"Running {index}/{total} queued extraction(s)"
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
        result: MDTrajectoryBatchResult,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        widget.set_progress(
            result.written_count,
            max(result.selected_frames, 1),
        )
        widget.set_status("Complete")
        self.project_paths_registered.emit(
            {
                "project_dir": result.project_dir,
                "frames_dir": result.output_dir,
            }
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
            f"Queue finished: {result_count} extraction(s) saved"
        )
        self.statusBar().showMessage("MD trajectory batch queue finished")

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage(
            "MD trajectory batch queue failed",
            5000,
        )
        QMessageBox.warning(
            self,
            "MD trajectory batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_mdtrajectory_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_trajectory_file: str | Path | None = None,
    initial_topology_file: str | Path | None = None,
    initial_energy_file: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = MDTrajectoryBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_trajectory_file=initial_trajectory_file,
        initial_topology_file=initial_topology_file,
        initial_energy_file=initial_energy_file,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "DEFAULT_FRAME_TIMESTEP_FS",
    "DEFAULT_TIME_CUTOFF_FS",
    "MDTrajectoryBatchItem",
    "MDTrajectoryBatchItemWidget",
    "MDTrajectoryBatchJob",
    "MDTrajectoryBatchQueueWindow",
    "MDTrajectoryBatchResult",
    "MDTrajectoryBatchWorker",
    "launch_mdtrajectory_batch_queue_ui",
]
