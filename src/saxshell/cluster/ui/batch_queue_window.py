from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from saxshell.cluster import (
    DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME,
    DEFAULT_SAVE_STATE_FREQUENCY,
    ExtractedFrameFolderClusterAnalyzer,
    PairCutoffDefinitions,
    format_box_dimensions,
    frame_folder_label,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.ui.main_window import (
    ClusterExportResult,
    ClusterExportWorker,
    ClusterJobConfig,
    suggest_cluster_output_dir,
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
from saxshell.structure import AtomTypeDefinitions

_CONSOLE_MAX_BLOCK_COUNT = 2000


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


def _required_frames_dir(text: str) -> Path:
    frames_dir = _required_path(text, "Frames folder")
    if not frames_dir.is_dir():
        raise ValueError(f"Frames folder does not exist: {frames_dir}")
    return frames_dir


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


def _copy_atom_type_definitions(
    definitions: AtomTypeDefinitions,
) -> AtomTypeDefinitions:
    return {
        atom_type: list(criteria)
        for atom_type, criteria in definitions.items()
    }


def _copy_pair_cutoff_definitions(
    definitions: PairCutoffDefinitions,
) -> PairCutoffDefinitions:
    return {pair: dict(levels) for pair, levels in definitions.items()}


def _summary_box_dimensions(
    summary: dict[str, object] | None,
) -> tuple[float, float, float] | None:
    if summary is None:
        return None
    value = summary.get("box_dimensions")
    if value is None:
        value = summary.get("estimated_box_dimensions")
    if value is None:
        return None
    return tuple(float(component) for component in value)


def _summary_text(summary: dict[str, object]) -> str:
    source_kind = summary.get("box_dimensions_source_kind")
    box_label = (
        "Source box dimensions"
        if source_kind == "source_filename"
        else "Estimated box dimensions"
    )
    lines = [
        f"Frames folder: {summary.get('input_dir')}",
        f"Mode: {summary.get('mode_label')}",
        f"Frames: {summary.get('n_frames')}",
        f"Output format: {summary.get('output_file_extension')}",
        f"{box_label}: {format_box_dimensions(_summary_box_dimensions(summary))}",
    ]
    if summary.get("box_dimensions_source") is not None:
        lines.append(f"Box source: {summary.get('box_dimensions_source')}")
    return "\n".join(lines)


def _source_kind_for_project_settings(settings: object) -> str:
    pdb_frames_dir = getattr(settings, "resolved_pdb_frames_dir", None)
    if pdb_frames_dir is not None:
        return "pdb"
    return "xyz"


def _frames_dir_for_project_settings(settings: object) -> Path | None:
    return getattr(settings, "resolved_pdb_frames_dir", None) or getattr(
        settings, "resolved_frames_dir", None
    )


@dataclass(slots=True)
class ClusterBatchJob:
    project_dir: Path
    frames_dir: Path
    frames_source_kind: str
    config: ClusterJobConfig


@dataclass(slots=True)
class ClusterBatchResult:
    project_dir: Path
    frames_dir: Path
    frames_source_kind: str
    output_dir: Path
    analyzed_frames: int
    total_clusters: int
    written_count: int


@dataclass(slots=True)
class ClusterBatchItem:
    item_id: str
    project_dir: Path | None = None
    frames_dir: Path | None = None
    frames_source_kind: str = "pdb"
    output_dir: Path | None = None
    atom_type_definitions: AtomTypeDefinitions = field(default_factory=dict)
    pair_cutoff_definitions: PairCutoffDefinitions = field(
        default_factory=dict
    )
    box_dimensions: tuple[float, float, float] | None = None
    use_pbc: bool = False
    search_mode: str = "kdtree"
    save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY
    default_cutoff: float | None = None
    shell_levels: tuple[int, ...] = ()
    include_shell_levels: tuple[int, ...] = (0,)
    shared_shells: bool = True
    smart_solvation_shells: bool = True
    include_shell_atoms_in_stoichiometry: bool = False

    def display_name(self) -> str:
        if self.project_dir is not None:
            return self.project_dir.name
        if self.frames_dir is not None:
            return self.frames_dir.name
        return "New cluster extraction"


def _queue_item_from_project_defaults(
    project_dir: str | Path,
    *,
    item_id: str | None = None,
) -> ClusterBatchItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = ClusterBatchItem(
        item_id=item_id or _new_item_id(),
        project_dir=resolved_project_dir,
    )
    try:
        settings = SAXSProjectManager().load_project(resolved_project_dir)
    except Exception:
        return item
    frames_dir = _frames_dir_for_project_settings(settings)
    return replace(
        item,
        frames_dir=frames_dir,
        frames_source_kind=_source_kind_for_project_settings(settings),
        output_dir=(
            None
            if frames_dir is None
            else suggest_cluster_output_dir(frames_dir)
        ),
    )


class ClusterBatchItemWidget(QFrame):
    settings_changed = Signal(str)
    remove_requested = Signal(str)
    duplicate_requested = Signal(str)

    def __init__(
        self,
        item: ClusterBatchItem,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._loading = False
        self._selected = False
        self._last_summary: dict[str, object] | None = None
        self._last_suggested_output_dir: Path | None = None
        self._build_ui()
        self._load_item(item)
        self._set_settings_visible(False)

    @property
    def item_id(self) -> str:
        return self._item.item_id

    def item(self) -> ClusterBatchItem:
        return self._item

    def collect_item(self) -> ClusterBatchItem:
        self._item = ClusterBatchItem(
            item_id=self._item.item_id,
            project_dir=_optional_path(self.project_dir_edit.text()),
            frames_dir=_optional_path(self.frames_dir_edit.text()),
            frames_source_kind=self._item.frames_source_kind,
            output_dir=_optional_path(self.output_dir_edit.text()),
            atom_type_definitions=_copy_atom_type_definitions(
                self.definitions_panel.atom_type_definitions()
            ),
            pair_cutoff_definitions=_copy_pair_cutoff_definitions(
                self.definitions_panel.pair_cutoff_definitions()
            ),
            box_dimensions=self.definitions_panel.box_dimensions(),
            use_pbc=self.definitions_panel.use_pbc(),
            search_mode=self.definitions_panel.search_mode(),
            save_state_frequency=self.definitions_panel.save_state_frequency(),
            default_cutoff=self.definitions_panel.default_cutoff(),
            shell_levels=self.definitions_panel.shell_growth_levels(),
            include_shell_levels=self.definitions_panel.include_shell_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            smart_solvation_shells=(
                self.definitions_panel.smart_solvation_shells()
            ),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
        )
        self._refresh_header()
        self._refresh_project_reference()
        return self._item

    def job(self) -> ClusterBatchJob:
        self.collect_item()
        project_dir = _required_project_dir(self.project_dir_edit.text())
        frames_dir = _required_frames_dir(self.frames_dir_edit.text())
        output_dir = _optional_path(
            self.output_dir_edit.text()
        ) or suggest_cluster_output_dir(frames_dir)
        atom_type_definitions = self.definitions_panel.atom_type_definitions()
        if not atom_type_definitions:
            raise ValueError(
                "Add at least one atom-type definition before exporting."
            )
        if not (
            atom_type_definitions.get("node")
            or atom_type_definitions.get("linker")
        ):
            raise ValueError("Define at least one node or linker atom type.")
        pair_cutoffs = self.definitions_panel.pair_cutoff_definitions()
        default_cutoff = self.definitions_panel.default_cutoff()
        if not pair_cutoffs and default_cutoff is None:
            raise ValueError(
                "Add at least one pair-cutoff definition or specify a "
                "default cutoff."
            )

        summary = self._last_summary
        if summary is None:
            summary = self._inspect_frames(frames_dir)
        frame_format = str(summary.get("frame_format", ""))
        box_dimensions = self.definitions_panel.box_dimensions()
        if self.definitions_panel.use_pbc() and box_dimensions is None:
            box_dimensions = _summary_box_dimensions(summary)
            if box_dimensions is None:
                raise ValueError(
                    "Periodic boundary conditions are enabled, but no box "
                    "dimensions are available."
                )

        config = ClusterJobConfig(
            frames_dir=frames_dir,
            atom_type_definitions=atom_type_definitions,
            pair_cutoff_definitions=pair_cutoffs,
            box_dimensions=box_dimensions,
            use_pbc=self.definitions_panel.use_pbc(),
            search_mode=self.definitions_panel.search_mode(),
            save_state_frequency=self.definitions_panel.save_state_frequency(),
            default_cutoff=default_cutoff,
            shell_levels=self.definitions_panel.shell_growth_levels(),
            include_shell_levels=self.definitions_panel.include_shell_levels(),
            shared_shells=self.definitions_panel.shared_shells(),
            smart_solvation_shells=(
                frame_format == "pdb"
                and self.definitions_panel.smart_solvation_shells()
            ),
            include_shell_atoms_in_stoichiometry=(
                self.definitions_panel.include_shell_atoms_in_stoichiometry()
            ),
            output_dir=output_dir,
        )
        return ClusterBatchJob(
            project_dir=project_dir,
            frames_dir=frames_dir,
            frames_source_kind=self._item.frames_source_kind,
            config=config,
        )

    def set_locked(self, locked: bool) -> None:
        self.settings_group.setEnabled(not locked)
        self.inspect_button.setEnabled(not locked)
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
            "QFrame#ClusterBatchItemHeader {"
            + (
                "background-color: #dce8f7; " "border: 1px solid #8fb0d7;"
                if self._selected
                else "background-color: #f6f8fb; " "border: 1px solid #cfd7e3;"
            )
            + "border-radius: 5px;}"
        )

    def analyze_input(self) -> None:
        frames_dir = _required_frames_dir(self.frames_dir_edit.text())
        summary = self._inspect_frames(frames_dir)
        self._apply_summary(summary)
        self.set_progress(0, max(int(summary.get("n_frames", 1)), 1))
        self.set_status("Input inspected")

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
        self.header_frame.setObjectName("ClusterBatchItemHeader")
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(8, 6, 8, 6)
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._set_settings_visible)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("New cluster extraction")
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(180)
        header.addWidget(self.status_label)
        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.clicked.connect(self._inspect_from_button)
        header.addWidget(self.inspect_button)
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

        self.settings_group = QGroupBox("Cluster Extraction Settings")
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

        self.frames_dir_edit = QLineEdit()
        self.frames_dir_edit.editingFinished.connect(self._on_frames_changed)
        form.addRow(
            "Frames folder",
            self._path_row(self.frames_dir_edit, self._choose_frames_dir),
        )

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.editingFinished.connect(self._on_editor_changed)
        form.addRow(
            "Output folder",
            self._path_row(self.output_dir_edit, self._choose_output_dir),
        )
        settings_layout.addLayout(form)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(120)
        self.summary_box.setPlainText(
            "Inspect the frames folder to detect PDB/XYZ mode and box "
            "settings."
        )
        settings_layout.addWidget(self.summary_box)

        self.definitions_panel = ClusterDefinitionsPanel()
        self.definitions_panel.load_preset(
            DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME
        )
        self.definitions_panel.settings_changed.connect(
            self._on_editor_changed
        )
        settings_layout.addWidget(self.definitions_panel)

    def _path_row(self, edit: QLineEdit, slot) -> QWidget:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)
        button = QPushButton("Browse...")
        button.clicked.connect(slot)
        row.addWidget(button)
        return row_widget

    def _load_item(self, item: ClusterBatchItem) -> None:
        self._loading = True
        self.project_dir_edit.setText(
            "" if item.project_dir is None else str(item.project_dir)
        )
        self.frames_dir_edit.setText(
            "" if item.frames_dir is None else str(item.frames_dir)
        )
        self.output_dir_edit.setText(
            "" if item.output_dir is None else str(item.output_dir)
        )
        if item.atom_type_definitions or item.pair_cutoff_definitions:
            self.definitions_panel.load_atom_type_definitions(
                item.atom_type_definitions,
                emit_signal=False,
            )
            self.definitions_panel.load_pair_cutoff_definitions(
                item.pair_cutoff_definitions,
                emit_signal=False,
            )
            self.definitions_panel.set_use_pbc(item.use_pbc, emit_signal=False)
            self.definitions_panel.set_search_mode(
                item.search_mode,
                emit_signal=False,
            )
            self.definitions_panel.set_save_state_frequency(
                item.save_state_frequency,
                emit_signal=False,
            )
            self.definitions_panel.set_default_cutoff(
                item.default_cutoff,
                emit_signal=False,
            )
            self.definitions_panel.set_shell_growth_levels(
                item.shell_levels,
                emit_signal=False,
            )
            self.definitions_panel.set_shared_shells(
                item.shared_shells,
                emit_signal=False,
            )
            self.definitions_panel.set_smart_solvation_shells(
                item.smart_solvation_shells,
                emit_signal=False,
            )
            self.definitions_panel.set_include_shell_atoms_in_stoichiometry(
                item.include_shell_atoms_in_stoichiometry,
                emit_signal=False,
            )
            self.definitions_panel.set_box_dimensions(
                item.box_dimensions,
                emit_signal=False,
            )
        self._loading = False
        self._refresh_header()
        self._refresh_project_reference()
        self._analyze_quietly()

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
            replace(
                _queue_item_from_project_defaults(
                    selected,
                    item_id=self.item_id,
                ),
                atom_type_definitions=self.definitions_panel.atom_type_definitions(),
                pair_cutoff_definitions=(
                    self.definitions_panel.pair_cutoff_definitions()
                ),
                box_dimensions=self.definitions_panel.box_dimensions(),
                use_pbc=self.definitions_panel.use_pbc(),
                search_mode=self.definitions_panel.search_mode(),
                save_state_frequency=self.definitions_panel.save_state_frequency(),
                default_cutoff=self.definitions_panel.default_cutoff(),
                shell_levels=self.definitions_panel.shell_growth_levels(),
                include_shell_levels=self.definitions_panel.include_shell_levels(),
                shared_shells=self.definitions_panel.shared_shells(),
                smart_solvation_shells=(
                    self.definitions_panel.smart_solvation_shells()
                ),
                include_shell_atoms_in_stoichiometry=(
                    self.definitions_panel.include_shell_atoms_in_stoichiometry()
                ),
            )
        )
        self._on_editor_changed()

    def _choose_frames_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select extracted PDB or XYZ frames folder",
            _dialog_start_dir(
                self.frames_dir_edit.text(),
                self.project_dir_edit.text(),
            ),
        )
        if not selected:
            return
        self.frames_dir_edit.setText(selected)
        self._on_frames_changed()

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select cluster output folder",
            _dialog_start_dir(
                self.output_dir_edit.text(),
                self.frames_dir_edit.text(),
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
        try:
            item = _queue_item_from_project_defaults(
                project_dir,
                item_id=self.item_id,
            )
        except Exception:
            self._on_editor_changed()
            return
        if item.frames_dir is not None:
            self.frames_dir_edit.setText(str(item.frames_dir))
        if item.output_dir is not None:
            self.output_dir_edit.setText(str(item.output_dir))
        self._item = replace(
            self._item,
            project_dir=project_dir,
            frames_dir=item.frames_dir,
            frames_source_kind=item.frames_source_kind,
            output_dir=item.output_dir,
        )
        self._analyze_quietly()
        self._on_editor_changed()

    def _on_frames_changed(self) -> None:
        frames_dir = _optional_path(self.frames_dir_edit.text())
        if frames_dir is not None:
            suggested = suggest_cluster_output_dir(frames_dir)
            current = _optional_path(self.output_dir_edit.text())
            if current is None or current == self._last_suggested_output_dir:
                self.output_dir_edit.setText(str(suggested))
            self._last_suggested_output_dir = suggested
        self._analyze_quietly()
        self._on_editor_changed()

    def _inspect_from_button(self) -> None:
        try:
            self.analyze_input()
            self._on_editor_changed()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Unable to inspect frames folder",
                str(exc),
            )
            self.summary_box.setPlainText(str(exc))
            self.set_status("Inspection failed")
            self._on_editor_changed()

    def _analyze_quietly(self) -> None:
        if not self.frames_dir_edit.text().strip():
            return
        try:
            self.analyze_input()
        except Exception as exc:
            self.summary_box.setPlainText(str(exc))
            self.set_status("Inspection failed")

    def _inspect_frames(self, frames_dir: Path) -> dict[str, object]:
        analyzer = ExtractedFrameFolderClusterAnalyzer(
            frames_dir=frames_dir,
            atom_type_definitions={},
            pair_cutoffs_def={},
        )
        return analyzer.inspect()

    def _apply_summary(self, summary: dict[str, object]) -> None:
        self._last_summary = summary
        frame_format = str(summary.get("frame_format", "") or "")
        self.definitions_panel.set_frame_mode(frame_format)
        if summary.get("box_dimensions_source_kind") == "source_filename":
            box_dimensions = _summary_box_dimensions(summary)
            if box_dimensions is not None:
                self.definitions_panel.set_box_dimensions(
                    box_dimensions,
                    emit_signal=False,
                )
        self.summary_box.setPlainText(_summary_text(summary))
        self.set_status(
            f"{frame_folder_label(frame_format)} mode, "
            f"{int(summary.get('n_frames', 0))} frame(s)"
        )

    def _on_editor_changed(self, *_args) -> None:
        if self._loading:
            return
        try:
            self.collect_item()
            if self.status_label.text() in {"Inspection failed", "Failed"}:
                self.set_status("Ready")
        except Exception:
            self._refresh_header()
            self._refresh_project_reference()
        self.settings_changed.emit(self.item_id)

    def _refresh_header(self) -> None:
        self.title_label.setText(self._item.display_name())

    def _refresh_project_reference(self) -> None:
        project_dir = _optional_path(self.project_dir_edit.text())
        if project_dir is None:
            text = "Project reference: choose a SAXSShell project folder."
        else:
            project_file = build_project_paths(project_dir).project_file
            if project_file.is_file():
                text = f"Project reference: {project_file}"
            else:
                text = f"Project reference: no project file found at {project_file}"
        self.project_reference_label.setText(text)


class ClusterBatchWorker(QObject):
    item_started = Signal(str, int, int)
    item_progress = Signal(str, int, int, str)
    item_phase_changed = Signal(str, str)
    item_finished = Signal(str, object)
    item_failed = Signal(str, str)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        queue_entries: list[tuple[str, ClusterBatchJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()
        self._project_manager = SAXSProjectManager()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[ClusterBatchResult] = []
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
        self.status.emit("Cluster extraction batch queue finished")
        self.finished.emit(results)

    def _run_job(
        self,
        item_id: str,
        job: ClusterBatchJob,
    ) -> ClusterBatchResult:
        worker = ClusterExportWorker(job.config)
        results: list[ClusterExportResult] = []
        failures: list[str] = []
        worker.progress.connect(
            lambda message: self.log.emit(
                f"[{job.project_dir.name}] {message}"
            )
        )
        worker.phase_changed.connect(
            lambda phase: self.item_phase_changed.emit(item_id, phase)
        )
        worker.progress_count.connect(
            lambda processed, total: self.item_progress.emit(
                item_id,
                processed,
                total,
                f"{processed}/{max(total, 1)} frame(s)",
            )
        )
        worker.finished.connect(results.append)
        worker.failed.connect(failures.append)
        worker.run()
        if failures:
            raise RuntimeError(failures[0])
        if not results:
            raise RuntimeError("Cluster extraction did not return a result.")
        export_result = results[0]
        settings = self._project_manager.load_project(job.project_dir)
        settings.clusters_dir = str(
            export_result.output_dir.expanduser().resolve()
        )
        self._project_manager.save_project(settings)
        self.log.emit(
            f"[{job.project_dir.name}] Registered clusters folder: "
            f"{settings.clusters_dir}"
        )
        return ClusterBatchResult(
            project_dir=job.project_dir,
            frames_dir=job.frames_dir,
            frames_source_kind=job.frames_source_kind,
            output_dir=export_result.output_dir.expanduser().resolve(),
            analyzed_frames=export_result.analyzed_frames,
            total_clusters=export_result.total_clusters,
            written_count=len(export_result.written_files),
        )


class ClusterBatchQueueWindow(QMainWindow):
    """Queue cluster extractions for multiple projects."""

    project_paths_registered = Signal(object)

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_frames_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets_by_id: dict[str, ClusterBatchItemWidget] = {}
        self._run_thread: QThread | None = None
        self._run_worker: ClusterBatchWorker | None = None
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_frames_dir = (
            None
            if initial_frames_dir is None
            else Path(initial_frames_dir).expanduser().resolve()
        )
        self._build_ui()
        if (
            self._initial_project_dir is not None
            or self._initial_frames_dir is not None
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
        item: ClusterBatchItem | None = None,
    ) -> ClusterBatchItemWidget:
        resolved_item = item or ClusterBatchItem(item_id=_new_item_id())
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_list.addItem(list_item)
        widget = ClusterBatchItemWidget(
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

    def queue_jobs_in_order(self) -> list[tuple[str, ClusterBatchJob]]:
        entries: list[tuple[str, ClusterBatchJob]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.job()))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell Cluster Extraction Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1180, 880)

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
        self.console.setMinimumHeight(160)
        self.console.document().setMaximumBlockCount(_CONSOLE_MAX_BLOCK_COUNT)
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _add_current_project(self) -> None:
        if (
            self._initial_project_dir is None
            and self._initial_frames_dir is None
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
            else ClusterBatchItem(item_id=_new_item_id())
        )
        if self._initial_frames_dir is not None:
            item = replace(
                item,
                frames_dir=self._initial_frames_dir,
                output_dir=suggest_cluster_output_dir(
                    self._initial_frames_dir
                ),
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
                        "Loading cluster extraction project "
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
            "Loading selected cluster extraction projects...",
            None,
            0,
            project_count,
            self,
        )
        dialog.setWindowTitle("Loading Cluster Extraction Projects")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

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
                "Cluster extraction batch queue",
                "Add at least one project before running the queue.",
            )
            return
        try:
            entries = self.queue_jobs_in_order()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid cluster extraction batch settings",
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
        self._run_worker = ClusterBatchWorker(entries)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.item_started.connect(self._on_item_started)
        self._run_worker.item_progress.connect(self._on_item_progress)
        self._run_worker.item_phase_changed.connect(
            self._on_item_phase_changed
        )
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
        text = message.strip()
        if not text:
            return

        scroll_bar = self.console.verticalScrollBar()
        previous_value = scroll_bar.value()
        was_at_bottom = previous_value >= max(scroll_bar.maximum() - 4, 0)

        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.console.document().characterCount() > 1:
            cursor.insertBlock()
        cursor.insertText(text)
        self.console.setTextCursor(cursor)

        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())
        else:
            scroll_bar.setValue(previous_value)

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

    def _on_item_phase_changed(self, item_id: str, phase: str) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is not None:
            widget.set_status(
                "Sorting clusters" if phase == "sorting" else "Extracting"
            )

    def _on_item_finished(
        self,
        item_id: str,
        result: ClusterBatchResult,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        widget.set_progress(
            result.analyzed_frames,
            max(result.analyzed_frames, 1),
        )
        widget.set_status("Complete")
        self.project_paths_registered.emit(
            {
                "project_dir": result.project_dir,
                "clusters_dir": result.output_dir,
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
        self.statusBar().showMessage("Cluster extraction batch queue finished")

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage(
            "Cluster extraction batch queue failed",
            5000,
        )
        QMessageBox.warning(
            self,
            "Cluster extraction batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_cluster_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_frames_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = ClusterBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_frames_dir=initial_frames_dir,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "ClusterBatchItem",
    "ClusterBatchItemWidget",
    "ClusterBatchJob",
    "ClusterBatchQueueWindow",
    "ClusterBatchResult",
    "ClusterBatchWorker",
    "launch_cluster_batch_queue_ui",
]
