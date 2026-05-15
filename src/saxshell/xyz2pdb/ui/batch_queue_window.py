from __future__ import annotations

import re
import threading
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
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
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    XYZToPDBMappingWorkflow,
)
from saxshell.xyz2pdb.workflow import (
    ReferenceLibraryEntry,
    XYZToPDBExportResult,
    default_reference_library_dir,
    list_reference_library,
    suggest_output_dir,
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


def _required_existing_input_path(text: str) -> Path:
    path = _required_path(text, "XYZ input path")
    if not path.exists():
        raise ValueError(f"XYZ input path does not exist: {path}")
    return path


def _required_project_dir(text: str) -> Path:
    project_dir = _required_path(text, "Project folder")
    project_file = build_project_paths(project_dir).project_file
    if not project_file.is_file():
        raise ValueError(f"Project file does not exist: {project_file}")
    return project_dir


def _validated_residue_code(value: str, field_name: str) -> str:
    residue = value.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", residue):
        raise ValueError(
            f"{field_name} must be exactly three capital letters."
        )
    return residue


def _default_free_atom_residue(element: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", element).upper()
    return (letters + "XX")[:3]


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
class XYZToPDBBatchJob:
    project_dir: Path
    input_path: Path
    reference_library_dir: Path
    molecule_inputs: tuple[MoleculeMappingInput, ...]
    free_atom_inputs: tuple[FreeAtomMappingInput, ...]
    hydrogen_mode: str = "leave_unassigned"


@dataclass(slots=True)
class XYZToPDBBatchResult:
    project_dir: Path
    input_path: Path
    output_dir: Path
    written_count: int


@dataclass(slots=True)
class XYZToPDBBatchItem:
    item_id: str
    project_dir: Path | None = None
    input_path: Path | None = None
    reference_library_dir: Path = default_reference_library_dir()
    molecule_inputs: tuple[MoleculeMappingInput, ...] = ()
    free_atom_inputs: tuple[FreeAtomMappingInput, ...] = ()
    hydrogen_mode: str = "leave_unassigned"

    def display_name(self) -> str:
        if self.project_dir is not None:
            return self.project_dir.name
        if self.input_path is not None:
            return self.input_path.name
        return "New XYZ -> PDB conversion"

    def to_job(self) -> XYZToPDBBatchJob:
        project_dir = _required_project_dir(
            "" if self.project_dir is None else str(self.project_dir)
        )
        input_path = _required_existing_input_path(
            "" if self.input_path is None else str(self.input_path)
        )
        library_dir = Path(self.reference_library_dir).expanduser().resolve()
        if not library_dir.is_dir():
            raise ValueError(
                f"Reference library folder does not exist: {library_dir}"
            )
        if not self.molecule_inputs and not self.free_atom_inputs:
            raise ValueError(
                "Add at least one reference molecule or free atom mapping."
            )
        return XYZToPDBBatchJob(
            project_dir=project_dir,
            input_path=input_path,
            reference_library_dir=library_dir,
            molecule_inputs=tuple(self.molecule_inputs),
            free_atom_inputs=tuple(self.free_atom_inputs),
            hydrogen_mode=self.hydrogen_mode or "leave_unassigned",
        )


def _queue_item_from_project_defaults(
    project_dir: str | Path,
    *,
    item_id: str | None = None,
    reference_library_dir: str | Path | None = None,
) -> XYZToPDBBatchItem:
    resolved_project_dir = Path(project_dir).expanduser().resolve()
    item = XYZToPDBBatchItem(
        item_id=item_id or _new_item_id(),
        project_dir=resolved_project_dir,
        reference_library_dir=(
            default_reference_library_dir()
            if reference_library_dir is None
            else Path(reference_library_dir).expanduser().resolve()
        ),
    )
    try:
        settings = SAXSProjectManager().load_project(resolved_project_dir)
    except Exception:
        return item
    return replace(item, input_path=settings.resolved_frames_dir)


class XYZToPDBBatchItemWidget(QFrame):
    settings_changed = Signal(str)
    remove_requested = Signal(str)
    duplicate_requested = Signal(str)

    def __init__(
        self,
        item: XYZToPDBBatchItem,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._loading = False
        self._selected = False
        self._available_elements: tuple[str, ...] = ()
        self._reference_entries: tuple[ReferenceLibraryEntry, ...] = ()
        self._build_ui()
        self._load_item(item)
        self._refresh_reference_entries()
        self._set_settings_visible(False)

    @property
    def item_id(self) -> str:
        return self._item.item_id

    def item(self) -> XYZToPDBBatchItem:
        return self._item

    def collect_item(self) -> XYZToPDBBatchItem:
        self._item = XYZToPDBBatchItem(
            item_id=self._item.item_id,
            project_dir=_optional_path(self.project_dir_edit.text()),
            input_path=_optional_path(self.input_path_edit.text()),
            reference_library_dir=(
                _optional_path(self.reference_library_edit.text())
                or default_reference_library_dir()
            ),
            molecule_inputs=tuple(self._molecule_inputs_from_table()),
            free_atom_inputs=tuple(self._free_atom_inputs_from_table()),
            hydrogen_mode=str(
                self.hydrogen_mode_combo.currentData() or "leave_unassigned"
            ),
        )
        self._refresh_header()
        self._refresh_project_reference()
        return self._item

    def job(self) -> XYZToPDBBatchJob:
        return self.collect_item().to_job()

    def set_locked(self, locked: bool) -> None:
        self.settings_group.setEnabled(not locked)
        self.analyze_button.setEnabled(not locked)
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
            "QFrame#XYZToPDBBatchItemHeader {"
            + (
                "background-color: #dce8f7; " "border: 1px solid #8fb0d7;"
                if self._selected
                else "background-color: #f6f8fb; " "border: 1px solid #cfd7e3;"
            )
            + "border-radius: 5px;}"
        )

    def analyze_input(self) -> None:
        input_path = _required_existing_input_path(self.input_path_edit.text())
        library_dir = (
            _optional_path(self.reference_library_edit.text())
            or default_reference_library_dir()
        )
        workflow = XYZToPDBMappingWorkflow(
            input_path,
            reference_library_dir=library_dir,
        )
        analysis = workflow.analyze_input()
        self._available_elements = tuple(sorted(analysis.element_counts))
        self._refresh_free_element_combo()
        self._refresh_reference_entries()
        lines = [
            f"XYZ files: {analysis.inspection.total_files}",
            f"Sample frame: {analysis.sample_file.name}",
            "Elements: "
            + ", ".join(
                f"{element} x{count}"
                for element, count in sorted(analysis.element_counts.items())
            ),
            f"Suggested PDB folder: {suggest_output_dir(input_path)}",
        ]
        self.analysis_summary_label.setText("\n".join(lines))
        self.set_progress(0, max(analysis.inspection.total_files, 1))
        self.set_status("Input analyzed")

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
        self.header_frame.setObjectName("XYZToPDBBatchItemHeader")
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(8, 6, 8, 6)
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._set_settings_visible)
        header.addWidget(self.toggle_button)
        self.title_label = QLabel("New XYZ -> PDB conversion")
        self.title_label.setStyleSheet("font-weight: 600;")
        header.addWidget(self.title_label, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(180)
        header.addWidget(self.status_label)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self._analyze_from_button)
        header.addWidget(self.analyze_button)
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

        self.settings_group = QGroupBox("XYZ -> PDB Conversion Settings")
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

        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.editingFinished.connect(self._on_editor_changed)
        input_layout.addWidget(self.input_path_edit, stretch=1)
        input_folder_button = QPushButton("Folder...")
        input_folder_button.clicked.connect(self._choose_input_dir)
        input_layout.addWidget(input_folder_button)
        input_file_button = QPushButton("File...")
        input_file_button.clicked.connect(self._choose_input_file)
        input_layout.addWidget(input_file_button)
        form.addRow("XYZ input", input_row)

        library_row = QWidget()
        library_layout = QHBoxLayout(library_row)
        library_layout.setContentsMargins(0, 0, 0, 0)
        self.reference_library_edit = QLineEdit()
        self.reference_library_edit.editingFinished.connect(
            self._on_reference_library_changed
        )
        library_layout.addWidget(self.reference_library_edit, stretch=1)
        library_button = QPushButton("Browse...")
        library_button.clicked.connect(self._choose_reference_library_dir)
        library_layout.addWidget(library_button)
        form.addRow("Reference library", library_row)

        self.analysis_summary_label = QLabel(
            "Analyze the XYZ input to populate the free-atom element list."
        )
        self.analysis_summary_label.setWordWrap(True)
        self.analysis_summary_label.setFrameShape(QFrame.Shape.StyledPanel)
        form.addRow("", self.analysis_summary_label)

        form.addRow("", self._build_free_atoms_group())
        form.addRow("", self._build_reference_molecules_group())

        self.hydrogen_mode_combo = QComboBox()
        self.hydrogen_mode_combo.addItem(
            "Leave unassigned",
            "leave_unassigned",
        )
        self.hydrogen_mode_combo.addItem(
            "Assign orphaned hydrogen",
            "assign_orphaned",
        )
        self.hydrogen_mode_combo.addItem(
            "Restore missing hydrogen",
            "restore_missing",
        )
        self.hydrogen_mode_combo.currentIndexChanged.connect(
            self._on_editor_changed
        )
        form.addRow("Hydrogen handling", self.hydrogen_mode_combo)

    def _build_free_atoms_group(self) -> QGroupBox:
        group = QGroupBox("Free Atoms")
        layout = QVBoxLayout(group)
        controls = QGridLayout()
        self.free_element_combo = QComboBox()
        self.free_element_combo.currentIndexChanged.connect(
            self._on_free_element_changed
        )
        self.free_residue_edit = QLineEdit()
        self.free_residue_edit.setPlaceholderText("SOL")
        add_button = QPushButton("Add Free Atom")
        add_button.clicked.connect(self._add_free_atom)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_free_atom)
        controls.addWidget(QLabel("Element"), 0, 0)
        controls.addWidget(self.free_element_combo, 0, 1)
        controls.addWidget(QLabel("Residue"), 0, 2)
        controls.addWidget(self.free_residue_edit, 0, 3)
        controls.addWidget(add_button, 0, 4)
        controls.addWidget(remove_button, 0, 5)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        layout.addLayout(controls)

        self.free_atom_table = QTableWidget(0, 2)
        self.free_atom_table.setHorizontalHeaderLabels(["Element", "Residue"])
        self.free_atom_table.verticalHeader().setVisible(False)
        self.free_atom_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.free_atom_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.free_atom_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.free_atom_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.free_atom_table.setMinimumHeight(120)
        layout.addWidget(self.free_atom_table)
        return group

    def _build_reference_molecules_group(self) -> QGroupBox:
        group = QGroupBox("Reference Molecules")
        layout = QVBoxLayout(group)
        controls = QGridLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(
            self._on_reference_selection_changed
        )
        self.molecule_residue_edit = QLineEdit()
        self.molecule_residue_edit.setPlaceholderText("DMF")
        add_button = QPushButton("Add Molecule")
        add_button.clicked.connect(self._add_molecule)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_molecule)
        controls.addWidget(QLabel("Reference"), 0, 0)
        controls.addWidget(self.reference_combo, 0, 1)
        controls.addWidget(QLabel("Residue"), 0, 2)
        controls.addWidget(self.molecule_residue_edit, 0, 3)
        controls.addWidget(add_button, 0, 4)
        controls.addWidget(remove_button, 0, 5)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        layout.addLayout(controls)

        self.molecule_table = QTableWidget(0, 2)
        self.molecule_table.setHorizontalHeaderLabels(["Reference", "Residue"])
        self.molecule_table.verticalHeader().setVisible(False)
        self.molecule_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.molecule_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.molecule_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.molecule_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.molecule_table.setMinimumHeight(140)
        layout.addWidget(self.molecule_table)
        return group

    def _load_item(self, item: XYZToPDBBatchItem) -> None:
        self._loading = True
        self.project_dir_edit.setText(
            "" if item.project_dir is None else str(item.project_dir)
        )
        self.input_path_edit.setText(
            "" if item.input_path is None else str(item.input_path)
        )
        self.reference_library_edit.setText(str(item.reference_library_dir))
        self._set_free_atom_inputs(item.free_atom_inputs)
        self._set_molecule_inputs(item.molecule_inputs)
        self._set_combo_value(self.hydrogen_mode_combo, item.hydrogen_mode)
        self._loading = False
        self._refresh_header()
        self._refresh_project_reference()

    def _set_free_atom_inputs(
        self,
        inputs: tuple[FreeAtomMappingInput, ...],
    ) -> None:
        self.free_atom_table.setRowCount(0)
        for item in inputs:
            row = self.free_atom_table.rowCount()
            self.free_atom_table.insertRow(row)
            self.free_atom_table.setItem(
                row,
                0,
                self._readonly_table_item(item.element),
            )
            self.free_atom_table.setItem(
                row,
                1,
                self._readonly_table_item(item.residue_name),
            )

    def _set_molecule_inputs(
        self,
        inputs: tuple[MoleculeMappingInput, ...],
    ) -> None:
        self.molecule_table.setRowCount(0)
        for item in inputs:
            row = self.molecule_table.rowCount()
            self.molecule_table.insertRow(row)
            self.molecule_table.setItem(
                row,
                0,
                self._readonly_table_item(item.reference_name),
            )
            self.molecule_table.setItem(
                row,
                1,
                self._readonly_table_item(item.residue_name),
            )

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
        current_library = (
            _optional_path(self.reference_library_edit.text())
            or default_reference_library_dir()
        )
        self._load_item(
            replace(
                _queue_item_from_project_defaults(
                    selected,
                    item_id=self.item_id,
                    reference_library_dir=current_library,
                ),
                molecule_inputs=tuple(self._molecule_inputs_from_table()),
                free_atom_inputs=tuple(self._free_atom_inputs_from_table()),
                hydrogen_mode=str(
                    self.hydrogen_mode_combo.currentData()
                    or "leave_unassigned"
                ),
            )
        )
        self._on_editor_changed()
        self._analyze_quietly()

    def _choose_input_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select XYZ frames folder",
            _dialog_start_dir(
                self.input_path_edit.text(),
                self.project_dir_edit.text(),
            ),
        )
        if not selected:
            return
        self.input_path_edit.setText(selected)
        self._on_editor_changed()
        self._analyze_quietly()

    def _choose_input_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select XYZ frame file",
            _dialog_start_dir(
                self.input_path_edit.text(),
                self.project_dir_edit.text(),
            ),
            "XYZ files (*.xyz);;All files (*)",
        )
        if not selected:
            return
        self.input_path_edit.setText(selected)
        self._on_editor_changed()
        self._analyze_quietly()

    def _choose_reference_library_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select reference library folder",
            _dialog_start_dir(self.reference_library_edit.text()),
        )
        if not selected:
            return
        self.reference_library_edit.setText(selected)
        self._on_reference_library_changed()

    def _analyze_from_button(self) -> None:
        try:
            self.analyze_input()
            self._on_editor_changed()
        except Exception as exc:
            QMessageBox.warning(self, "Unable to analyze XYZ input", str(exc))
            self.analysis_summary_label.setText(str(exc))
            self.set_status("Analysis failed")
            self._on_editor_changed()

    def _analyze_quietly(self) -> None:
        if not self.input_path_edit.text().strip():
            return
        try:
            self.analyze_input()
        except Exception as exc:
            self.analysis_summary_label.setText(str(exc))
            self.set_status("Analysis failed")

    def _refresh_reference_entries(self) -> None:
        library_dir = (
            _optional_path(self.reference_library_edit.text())
            or default_reference_library_dir()
        )
        try:
            entries = tuple(list_reference_library(library_dir))
        except Exception:
            entries = ()
        self._reference_entries = entries
        current = self.reference_combo.currentData()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        for entry in entries:
            self.reference_combo.addItem(entry.name, entry.name)
        if current is not None:
            index = self.reference_combo.findData(current)
            if index >= 0:
                self.reference_combo.setCurrentIndex(index)
        self.reference_combo.blockSignals(False)
        self._apply_selected_reference_default_residue()

    def _refresh_free_element_combo(self) -> None:
        current = self.free_element_combo.currentData()
        self.free_element_combo.blockSignals(True)
        self.free_element_combo.clear()
        for element in self._available_elements:
            self.free_element_combo.addItem(element, element)
        if current is not None:
            index = self.free_element_combo.findData(current)
            if index >= 0:
                self.free_element_combo.setCurrentIndex(index)
        self.free_element_combo.blockSignals(False)
        self._on_free_element_changed()

    def _on_reference_library_changed(self) -> None:
        self._refresh_reference_entries()
        self._on_editor_changed()

    def _on_free_element_changed(self, *_args) -> None:
        element = str(self.free_element_combo.currentData() or "").strip()
        if element and not self.free_residue_edit.text().strip():
            self.free_residue_edit.setText(_default_free_atom_residue(element))

    def _on_reference_selection_changed(self, *_args) -> None:
        self._apply_selected_reference_default_residue()

    def _apply_selected_reference_default_residue(self) -> None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        if not reference_name:
            return
        entry = next(
            (
                entry
                for entry in self._reference_entries
                if entry.name == reference_name
            ),
            None,
        )
        if entry is None:
            return
        if not self.molecule_residue_edit.text().strip():
            self.molecule_residue_edit.setText(entry.residue_name)

    def _add_free_atom(self) -> None:
        try:
            element = str(self.free_element_combo.currentData() or "").strip()
            if not element:
                raise ValueError("Choose an element to add as a free atom.")
            residue = _validated_residue_code(
                self.free_residue_edit.text(),
                "Free-atom residue",
            )
            for row in range(self.free_atom_table.rowCount()):
                item = self.free_atom_table.item(row, 0)
                if item is not None and item.text().strip() == element:
                    raise ValueError(
                        f"{element} is already listed as a free atom."
                    )
        except Exception as exc:
            QMessageBox.warning(self, "Unable to add free atom", str(exc))
            return
        row = self.free_atom_table.rowCount()
        self.free_atom_table.insertRow(row)
        self.free_atom_table.setItem(
            row,
            0,
            self._readonly_table_item(element),
        )
        self.free_atom_table.setItem(
            row,
            1,
            self._readonly_table_item(residue),
        )
        self._on_editor_changed()

    def _remove_selected_free_atom(self) -> None:
        row = self.free_atom_table.currentRow()
        if row < 0:
            return
        self.free_atom_table.removeRow(row)
        self._on_editor_changed()

    def _add_molecule(self) -> None:
        try:
            reference_name = str(
                self.reference_combo.currentData() or ""
            ).strip()
            if not reference_name:
                raise ValueError("Choose a reference molecule first.")
            residue = _validated_residue_code(
                self.molecule_residue_edit.text(),
                "Reference-molecule residue",
            )
            for row in range(self.molecule_table.rowCount()):
                item = self.molecule_table.item(row, 1)
                if item is not None and item.text().strip() == residue:
                    raise ValueError(f"Residue {residue} is already listed.")
        except Exception as exc:
            QMessageBox.warning(self, "Unable to add molecule", str(exc))
            return
        row = self.molecule_table.rowCount()
        self.molecule_table.insertRow(row)
        self.molecule_table.setItem(
            row,
            0,
            self._readonly_table_item(reference_name),
        )
        self.molecule_table.setItem(
            row,
            1,
            self._readonly_table_item(residue),
        )
        self._on_editor_changed()

    def _remove_selected_molecule(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0:
            return
        self.molecule_table.removeRow(row)
        self._on_editor_changed()

    def _free_atom_inputs_from_table(self) -> list[FreeAtomMappingInput]:
        inputs: list[FreeAtomMappingInput] = []
        for row in range(self.free_atom_table.rowCount()):
            element_item = self.free_atom_table.item(row, 0)
            residue_item = self.free_atom_table.item(row, 1)
            if element_item is None or residue_item is None:
                continue
            inputs.append(
                FreeAtomMappingInput(
                    element=element_item.text().strip(),
                    residue_name=residue_item.text().strip(),
                )
            )
        return inputs

    def _molecule_inputs_from_table(self) -> list[MoleculeMappingInput]:
        inputs: list[MoleculeMappingInput] = []
        for row in range(self.molecule_table.rowCount()):
            reference_item = self.molecule_table.item(row, 0)
            residue_item = self.molecule_table.item(row, 1)
            if reference_item is None or residue_item is None:
                continue
            inputs.append(
                MoleculeMappingInput(
                    reference_name=reference_item.text().strip(),
                    residue_name=residue_item.text().strip(),
                )
            )
        return inputs

    def _on_editor_changed(self, *_args) -> None:
        if self._loading:
            return
        try:
            self.collect_item()
            if self.status_label.text() in {"Analysis failed", "Failed"}:
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

    @staticmethod
    def _readonly_table_item(value: object) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)


class XYZToPDBBatchWorker(QObject):
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
        queue_entries: list[tuple[str, XYZToPDBBatchJob]],
    ) -> None:
        super().__init__()
        self.queue_entries = list(queue_entries)
        self._cancel_requested = threading.Event()
        self._project_manager = SAXSProjectManager()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    @Slot()
    def run(self) -> None:
        results: list[XYZToPDBBatchResult] = []
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
        self.status.emit("XYZ -> PDB batch queue finished")
        self.finished.emit(results)

    def _run_job(
        self,
        item_id: str,
        job: XYZToPDBBatchJob,
    ) -> XYZToPDBBatchResult:
        settings = self._project_manager.load_project(job.project_dir)
        workflow = XYZToPDBMappingWorkflow(
            job.input_path,
            reference_library_dir=job.reference_library_dir,
        )
        self.item_progress.emit(
            item_id,
            0,
            1,
            "Preparing mapping",
        )

        def on_progress(
            processed: int,
            total: int,
            message: str,
        ) -> None:
            self.item_progress.emit(item_id, processed, total, message)

        result: XYZToPDBExportResult = workflow.export_with_mapping(
            molecule_inputs=job.molecule_inputs,
            free_atom_inputs=job.free_atom_inputs,
            hydrogen_mode=job.hydrogen_mode,
            progress_callback=on_progress,
            log_callback=(
                lambda message: self.log.emit(
                    f"[{job.project_dir.name}] {message}"
                )
            ),
            cancel_callback=self._cancel_requested.is_set,
        )
        settings.pdb_frames_dir = str(result.output_dir.expanduser().resolve())
        self._project_manager.save_project(settings)
        self.log.emit(
            f"[{job.project_dir.name}] Registered PDB frames folder: "
            f"{settings.pdb_frames_dir}"
        )
        return XYZToPDBBatchResult(
            project_dir=job.project_dir,
            input_path=job.input_path,
            output_dir=result.output_dir.expanduser().resolve(),
            written_count=len(result.written_files),
        )


class XYZToPDBBatchQueueWindow(QMainWindow):
    """Queue XYZ-to-PDB conversions for multiple projects."""

    project_paths_registered = Signal(object)

    def __init__(
        self,
        initial_project_dir: str | Path | None = None,
        *,
        initial_input_path: str | Path | None = None,
        reference_library_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._widgets_by_id: dict[str, XYZToPDBBatchItemWidget] = {}
        self._run_thread: QThread | None = None
        self._run_worker: XYZToPDBBatchWorker | None = None
        self._initial_project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._initial_input_path = (
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        )
        self._reference_library_dir = (
            default_reference_library_dir()
            if reference_library_dir is None
            else Path(reference_library_dir).expanduser().resolve()
        )
        self._build_ui()
        if (
            self._initial_project_dir is not None
            or self._initial_input_path is not None
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
        item: XYZToPDBBatchItem | None = None,
        *,
        auto_analyze: bool = False,
    ) -> XYZToPDBBatchItemWidget:
        resolved_item = item or XYZToPDBBatchItem(item_id=_new_item_id())
        list_item = QListWidgetItem()
        list_item.setData(Qt.ItemDataRole.UserRole, resolved_item.item_id)
        self.queue_list.addItem(list_item)
        widget = XYZToPDBBatchItemWidget(
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
        if auto_analyze:
            widget._analyze_quietly()
        return widget

    def queue_jobs_in_order(self) -> list[tuple[str, XYZToPDBBatchJob]]:
        entries: list[tuple[str, XYZToPDBBatchJob]] = []
        for row in range(self.queue_list.count()):
            list_item = self.queue_list.item(row)
            item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
            widget = self._widgets_by_id[item_id]
            entries.append((item_id, widget.job()))
        return entries

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell XYZ -> PDB Batch Queue")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1120, 860)

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
        run_layout.addWidget(self.console)
        root.addWidget(run_group)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _add_current_project(self) -> None:
        if (
            self._initial_project_dir is None
            and self._initial_input_path is None
        ):
            QMessageBox.information(
                self,
                "No active project",
                "The main UI did not provide an active project reference.",
            )
            return
        item = (
            _queue_item_from_project_defaults(
                self._initial_project_dir,
                reference_library_dir=self._reference_library_dir,
            )
            if self._initial_project_dir is not None
            else XYZToPDBBatchItem(
                item_id=_new_item_id(),
                reference_library_dir=self._reference_library_dir,
            )
        )
        item = replace(
            item,
            input_path=self._initial_input_path or item.input_path,
        )
        self.add_queue_item(item, auto_analyze=item.input_path is not None)

    def _choose_projects_to_add(self) -> None:
        selected_dirs = _choose_existing_directories(
            self,
            title="Select SAXSShell project folders",
            start_dir=self._initial_project_dir or Path.home(),
        )
        if not selected_dirs:
            return
        for project_dir in selected_dirs:
            item = _queue_item_from_project_defaults(
                project_dir,
                reference_library_dir=self._reference_library_dir,
            )
            self.add_queue_item(item, auto_analyze=item.input_path is not None)

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
                "XYZ -> PDB batch queue",
                "Add at least one project before running the queue.",
            )
            return
        try:
            entries = self.queue_jobs_in_order()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid XYZ -> PDB batch settings",
                str(exc),
            )
            return

        self.console.clear()
        self._set_running(True)
        self.queue_status_label.setText(
            f"Running 0/{len(entries)} queued conversion(s)"
        )
        for widget in self._widgets_by_id.values():
            widget.set_progress(0, 1)
            widget.set_status("Queued")

        self._run_thread = QThread(self)
        self._run_worker = XYZToPDBBatchWorker(entries)
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
            f"Running {index}/{total} queued conversion(s)"
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
        result: XYZToPDBBatchResult,
    ) -> None:
        widget = self._widgets_by_id.get(item_id)
        if widget is None:
            return
        widget.set_progress(result.written_count, max(result.written_count, 1))
        widget.set_status("Complete")
        self.project_paths_registered.emit(
            {
                "project_dir": result.project_dir,
                "pdb_frames_dir": result.output_dir,
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
            f"Queue finished: {result_count} conversion(s) saved"
        )
        self.statusBar().showMessage("XYZ -> PDB batch queue finished")

    def _on_queue_failed(self, item_id: str, message: str) -> None:
        self._set_running(False)
        self.queue_status_label.setText("Queue stopped after a failure")
        self.statusBar().showMessage("XYZ -> PDB batch queue failed", 5000)
        QMessageBox.warning(
            self,
            "XYZ -> PDB batch queue failed",
            f"Queue item {item_id} failed:\n{message}",
        )

    def _cleanup_run_thread(self) -> None:
        self._run_thread = None
        self._run_worker = None


def launch_xyz2pdb_batch_queue_ui(
    initial_project_dir: str | Path | None = None,
    *,
    initial_input_path: str | Path | None = None,
    reference_library_dir: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication([])
    configure_saxshell_application(app)
    window = XYZToPDBBatchQueueWindow(
        initial_project_dir=initial_project_dir,
        initial_input_path=initial_input_path,
        reference_library_dir=reference_library_dir,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "XYZToPDBBatchItem",
    "XYZToPDBBatchItemWidget",
    "XYZToPDBBatchJob",
    "XYZToPDBBatchQueueWindow",
    "XYZToPDBBatchResult",
    "XYZToPDBBatchWorker",
    "launch_xyz2pdb_batch_queue_ui",
]
