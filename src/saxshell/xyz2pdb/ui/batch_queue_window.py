from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, replace
from math import acos, degrees, dist
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
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
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSpinBox,
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
from saxshell.structure import PDBStructure
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    ReferenceBondToleranceInput,
    XYZToPDBMappingWorkflow,
    reference_bond_tolerances,
)
from saxshell.xyz2pdb.ui.input_panel import (
    xyz_input_convention_warning_message,
)
from saxshell.xyz2pdb.workflow import (
    ReferenceLibraryEntry,
    XYZToPDBExportResult,
    default_reference_library_dir,
    list_reference_library,
    suggest_output_dir,
)

_FALLBACK_BOND_TOLERANCE_PERCENT = 12.0
_BATCH_CONSOLE_MAX_BLOCKS = 1200
_BATCH_PROGRESS_LOG_MILESTONES = 20


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


def _normalized_table_atom_name(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).strip().upper()
    if not text:
        text = fallback.upper()
    return text[:4]


def _is_progress_milestone(processed: int, total: int) -> bool:
    total = max(int(total), 1)
    processed = max(int(processed), 0)
    if processed <= 0 or processed == 1 or processed >= total:
        return True
    stride = max(
        (total + _BATCH_PROGRESS_LOG_MILESTONES - 1)
        // _BATCH_PROGRESS_LOG_MILESTONES,
        1,
    )
    return processed % stride == 0


def _should_emit_batch_log_message(message: str) -> bool:
    match = re.match(r"^\[(\d+)/(\d+)\]", str(message).strip())
    if match is None:
        return True
    return _is_progress_milestone(int(match.group(1)), int(match.group(2)))


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
        self._molecule_inputs: list[MoleculeMappingInput] = []
        self._reference_entries: tuple[ReferenceLibraryEntry, ...] = ()
        self._reference_bond_defaults: dict[
            str,
            tuple[ReferenceBondToleranceInput, ...],
        ] = {}
        self._reference_paths: dict[str, str] = {}
        self._reference_atom_coordinates: dict[
            str,
            dict[str, tuple[float, float, float]],
        ] = {}
        self._last_autofilled_residue_name: str | None = None
        self._populating_bond_table = False
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
        target_output_dir = suggest_output_dir(input_path)
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
            f"Target PDB folder: {target_output_dir.name}",
            f"Target PDB path: {target_output_dir}",
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
        layout = QHBoxLayout(group)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        controls = QGridLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(
            self._on_reference_selection_changed
        )
        self.molecule_residue_edit = QLineEdit()
        self.molecule_residue_edit.setPlaceholderText("DMF")
        self.tight_scale_spin = QDoubleSpinBox()
        self.tight_scale_spin.setDecimals(1)
        self.tight_scale_spin.setRange(1.0, 500.0)
        self.tight_scale_spin.setSingleStep(5.0)
        self.tight_scale_spin.setSuffix(" %")
        self.tight_scale_spin.setValue(85.0)
        self.tight_scale_spin.valueChanged.connect(
            self._on_bond_range_controls_changed
        )
        self.relaxed_scale_spin = QDoubleSpinBox()
        self.relaxed_scale_spin.setDecimals(1)
        self.relaxed_scale_spin.setRange(1.0, 500.0)
        self.relaxed_scale_spin.setSingleStep(5.0)
        self.relaxed_scale_spin.setSuffix(" %")
        self.relaxed_scale_spin.setValue(135.0)
        self.relaxed_scale_spin.valueChanged.connect(
            self._on_bond_range_controls_changed
        )
        self.max_missing_h_spin = QSpinBox()
        self.max_missing_h_spin.setRange(0, 8)
        self.max_missing_h_spin.setValue(0)
        add_button = QPushButton("Add Molecule")
        add_button.clicked.connect(self._add_molecule)
        update_button = QPushButton("Update Selected")
        update_button.clicked.connect(self._update_selected_molecule)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_molecule)
        controls.addWidget(QLabel("Reference"), 0, 0)
        controls.addWidget(self.reference_combo, 0, 1)
        controls.addWidget(QLabel("Residue"), 0, 2)
        controls.addWidget(self.molecule_residue_edit, 0, 3)
        controls.addWidget(QLabel("Missing H"), 1, 0)
        controls.addWidget(self.max_missing_h_spin, 1, 1)
        controls.addWidget(add_button, 1, 2)
        controls.addWidget(update_button, 1, 3)
        controls.addWidget(remove_button, 1, 4)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)
        left_layout.addLayout(controls)

        self.molecule_table = QTableWidget(0, 6)
        self.molecule_table.setHorizontalHeaderLabels(
            [
                "Reference",
                "Residue",
                "Bonds",
                "Tight %",
                "Relaxed %",
                "Missing H",
            ]
        )
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
        self.molecule_table.itemSelectionChanged.connect(
            self._on_selected_molecule_changed
        )
        header = self.molecule_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.molecule_table.setMinimumHeight(140)
        left_layout.addWidget(self.molecule_table)

        range_controls = QGridLayout()
        range_controls.addWidget(QLabel("Tight"), 0, 0)
        range_controls.addWidget(self.tight_scale_spin, 0, 1)
        range_controls.addWidget(QLabel("Relaxed"), 0, 2)
        range_controls.addWidget(self.relaxed_scale_spin, 0, 3)
        range_controls.setColumnStretch(1, 1)
        range_controls.setColumnStretch(3, 1)
        right_layout.addLayout(range_controls)

        bond_label = QLabel("Direct Bond Tolerances")
        right_layout.addWidget(bond_label)
        self.bond_table = QTableWidget(0, 8)
        self.bond_table.setHorizontalHeaderLabels(
            [
                "Atom 1",
                "Atom 2",
                "Ref (A)",
                "Tolerance (%)",
                "Tight Min (A)",
                "Tight Max (A)",
                "Relaxed Min (A)",
                "Relaxed Max (A)",
            ]
        )
        self.bond_table.verticalHeader().setVisible(False)
        self.bond_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.bond_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.bond_table.itemChanged.connect(self._on_bond_table_changed)
        bond_header = self.bond_table.horizontalHeader()
        for column in range(self.bond_table.columnCount()):
            bond_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.bond_table.setMinimumHeight(140)
        right_layout.addWidget(self.bond_table)

        angle_label = QLabel("Reference Bond Angles")
        right_layout.addWidget(angle_label)
        self.angle_table = QTableWidget(0, 4)
        self.angle_table.setHorizontalHeaderLabels(
            [
                "Atom 1",
                "Center",
                "Atom 3",
                "Angle (deg)",
            ]
        )
        self.angle_table.verticalHeader().setVisible(False)
        self.angle_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.angle_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.angle_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        angle_header = self.angle_table.horizontalHeader()
        for column in range(self.angle_table.columnCount()):
            angle_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        self.angle_table.setMinimumHeight(100)
        right_layout.addWidget(self.angle_table)

        layout.addWidget(left_panel, stretch=2)
        layout.addWidget(right_panel, stretch=3)
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
        self._molecule_inputs = list(inputs)
        self._refresh_molecule_table()
        if self._molecule_inputs:
            self.molecule_table.selectRow(0)
        else:
            self._populate_bond_table(())

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
            input_warning = xyz_input_convention_warning_message(
                _optional_path(self.input_path_edit.text())
            )
            if input_warning is not None:
                QMessageBox.warning(
                    self,
                    "Check XYZ input",
                    input_warning,
                )
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
        self._reference_paths = {
            entry.name: str(entry.path) for entry in entries
        }
        self._reference_atom_coordinates.clear()
        self._reference_bond_defaults = {}
        for entry in entries:
            try:
                self._reference_bond_defaults[entry.name] = (
                    reference_bond_tolerances(
                        entry.name,
                        library_dir=library_dir,
                    )
                )
            except Exception:
                self._reference_bond_defaults[entry.name] = ()
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
        self._apply_selected_reference_default_residue(force=False)
        if 0 <= self.molecule_table.currentRow() < len(self._molecule_inputs):
            self._on_selected_molecule_changed()
        else:
            self._load_default_bonds_for_reference()

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
        self._apply_selected_reference_default_residue(force=True)
        self._load_default_bonds_for_reference()

    def _apply_selected_reference_default_residue(
        self,
        *,
        force: bool,
    ) -> None:
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
        current_residue = self.molecule_residue_edit.text().strip()
        should_replace = force or not current_residue
        if (
            not should_replace
            and self._last_autofilled_residue_name is not None
            and current_residue == self._last_autofilled_residue_name
        ):
            should_replace = True
        if not should_replace:
            return
        self.molecule_residue_edit.setText(entry.residue_name)
        self._last_autofilled_residue_name = entry.residue_name

    def _load_default_bonds_for_reference(self) -> None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        self._populate_bond_table(
            self._reference_bond_defaults.get(reference_name, ())
        )

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
            molecule = self._molecule_from_controls()
            for item in self._molecule_inputs:
                if item.residue_name == molecule.residue_name:
                    raise ValueError(
                        f"Residue {molecule.residue_name} is already listed."
                    )
        except Exception as exc:
            QMessageBox.warning(self, "Unable to add molecule", str(exc))
            return
        self._molecule_inputs.append(molecule)
        self._refresh_molecule_table()
        self.molecule_table.selectRow(
            max(self.molecule_table.rowCount() - 1, 0)
        )
        self._on_editor_changed()

    def _update_selected_molecule(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0:
            return
        try:
            molecule = self._molecule_from_controls()
            for index, item in enumerate(self._molecule_inputs):
                if index != row and item.residue_name == molecule.residue_name:
                    raise ValueError(
                        f"Residue {molecule.residue_name} is already listed."
                    )
        except Exception as exc:
            QMessageBox.warning(self, "Unable to update molecule", str(exc))
            return
        self._molecule_inputs[row] = molecule
        self._refresh_molecule_table()
        self.molecule_table.selectRow(row)
        self._on_editor_changed()

    def _remove_selected_molecule(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0:
            return
        if row < len(self._molecule_inputs):
            del self._molecule_inputs[row]
        self._refresh_molecule_table()
        if self.molecule_table.rowCount():
            self.molecule_table.selectRow(
                min(row, self.molecule_table.rowCount() - 1)
            )
        else:
            self._populate_bond_table(())
        self._on_editor_changed()

    def _molecule_from_controls(self) -> MoleculeMappingInput:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        if not reference_name:
            raise ValueError("Choose a reference molecule first.")
        residue = _validated_residue_code(
            self.molecule_residue_edit.text(),
            "Reference-molecule residue",
        )
        bond_tolerances = (
            self._bond_inputs_from_table()
            if self.bond_table.rowCount()
            else tuple(self._reference_bond_defaults.get(reference_name, ()))
        )
        return MoleculeMappingInput(
            reference_name=reference_name,
            residue_name=residue,
            bond_tolerances=bond_tolerances,
            tight_pass_scale=float(self.tight_scale_spin.value()) / 100.0,
            relaxed_pass_scale=float(self.relaxed_scale_spin.value()) / 100.0,
            max_missing_hydrogens=int(self.max_missing_h_spin.value()),
        )

    def _refresh_molecule_table(self) -> None:
        self.molecule_table.setRowCount(0)
        for row, molecule in enumerate(self._molecule_inputs):
            self.molecule_table.insertRow(row)
            values = (
                molecule.reference_name,
                molecule.residue_name,
                str(len(molecule.bond_tolerances)),
                f"{molecule.tight_pass_scale * 100.0:.1f}%",
                f"{molecule.relaxed_pass_scale * 100.0:.1f}%",
                str(int(molecule.max_missing_hydrogens)),
            )
            for column, value in enumerate(values):
                item = self._readonly_table_item(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.molecule_table.setItem(row, column, item)

    def _on_selected_molecule_changed(self) -> None:
        row = self.molecule_table.currentRow()
        if row < 0 or row >= len(self._molecule_inputs):
            self._populate_bond_table(())
            return
        molecule = self._molecule_inputs[row]
        self._set_combo_value(self.reference_combo, molecule.reference_name)
        self.molecule_residue_edit.setText(molecule.residue_name)
        self.tight_scale_spin.setValue(
            float(molecule.tight_pass_scale) * 100.0
        )
        self.relaxed_scale_spin.setValue(
            float(molecule.relaxed_pass_scale) * 100.0
        )
        self.max_missing_h_spin.setValue(int(molecule.max_missing_hydrogens))
        self._populate_bond_table(molecule.bond_tolerances)

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
        return list(self._molecule_inputs)

    def _populate_bond_table(
        self,
        bond_inputs: tuple[ReferenceBondToleranceInput, ...],
    ) -> None:
        reference_name = str(self.reference_combo.currentData() or "").strip()
        self._populating_bond_table = True
        self.bond_table.setRowCount(0)
        for row, bond in enumerate(bond_inputs):
            reference_length = self._reference_bond_length(
                reference_name,
                bond.atom1_name,
                bond.atom2_name,
            )
            self.bond_table.insertRow(row)
            self.bond_table.setItem(
                row,
                0,
                self._readonly_table_item(bond.atom1_name),
            )
            self.bond_table.setItem(
                row,
                1,
                self._readonly_table_item(bond.atom2_name),
            )
            reference_item = self._readonly_table_item(
                ""
                if reference_length is None
                else f"{float(reference_length):.3f}"
            )
            reference_item.setData(
                Qt.ItemDataRole.UserRole,
                reference_length,
            )
            self.bond_table.setItem(row, 2, reference_item)
            tolerance_item = QTableWidgetItem(f"{float(bond.tolerance):.2f}")
            tolerance_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bond_table.setItem(row, 3, tolerance_item)
            for column in range(4, self.bond_table.columnCount()):
                self.bond_table.setItem(
                    row,
                    column,
                    self._readonly_table_item(""),
                )
        self._populating_bond_table = False
        self._refresh_bond_table_range_columns()
        self._populate_angle_table(reference_name, bond_inputs)

    def _bond_inputs_from_table(
        self,
    ) -> tuple[ReferenceBondToleranceInput, ...]:
        result: list[ReferenceBondToleranceInput] = []
        for row in range(self.bond_table.rowCount()):
            atom1_item = self.bond_table.item(row, 0)
            atom2_item = self.bond_table.item(row, 1)
            tolerance_item = self.bond_table.item(row, 3)
            if (
                atom1_item is None
                or atom2_item is None
                or tolerance_item is None
            ):
                continue
            try:
                tolerance = float(tolerance_item.text().strip())
            except ValueError:
                tolerance = _FALLBACK_BOND_TOLERANCE_PERCENT
            result.append(
                ReferenceBondToleranceInput(
                    atom1_name=atom1_item.text().strip(),
                    atom2_name=atom2_item.text().strip(),
                    tolerance=tolerance,
                )
            )
        return tuple(result)

    def _on_bond_table_changed(self, _item: QTableWidgetItem) -> None:
        if self._populating_bond_table:
            return
        self._refresh_bond_table_range_columns()
        row = self.molecule_table.currentRow()
        if row < 0 or row >= len(self._molecule_inputs):
            return
        molecule = self._molecule_inputs[row]
        self._molecule_inputs[row] = MoleculeMappingInput(
            reference_name=molecule.reference_name,
            residue_name=molecule.residue_name,
            bond_tolerances=self._bond_inputs_from_table(),
            tight_pass_scale=molecule.tight_pass_scale,
            relaxed_pass_scale=molecule.relaxed_pass_scale,
            max_assignment_distance=molecule.max_assignment_distance,
            max_missing_hydrogens=molecule.max_missing_hydrogens,
        )
        self._refresh_molecule_table()
        self.molecule_table.selectRow(row)
        self._on_editor_changed()

    def _on_bond_range_controls_changed(self, _value: float) -> None:
        self._refresh_bond_table_range_columns()

    def _refresh_bond_table_range_columns(self) -> None:
        if not self.bond_table.rowCount():
            return
        tight_scale = float(self.tight_scale_spin.value()) / 100.0
        relaxed_scale = float(self.relaxed_scale_spin.value()) / 100.0
        self._populating_bond_table = True
        for row in range(self.bond_table.rowCount()):
            reference_item = self.bond_table.item(row, 2)
            tolerance_item = self.bond_table.item(row, 3)
            if reference_item is None or tolerance_item is None:
                continue
            reference_length = reference_item.data(Qt.ItemDataRole.UserRole)
            try:
                resolved_reference_length = float(reference_length)
            except (TypeError, ValueError):
                resolved_reference_length = None
            try:
                tolerance_percent = float(tolerance_item.text().strip())
            except ValueError:
                tolerance_percent = _FALLBACK_BOND_TOLERANCE_PERCENT
            if resolved_reference_length is None:
                values = ("", "", "", "")
            else:
                tight_min, tight_max = self._bond_search_bounds(
                    resolved_reference_length,
                    tolerance_percent,
                    tight_scale,
                )
                relaxed_min, relaxed_max = self._bond_search_bounds(
                    resolved_reference_length,
                    tolerance_percent,
                    relaxed_scale,
                )
                values = (
                    f"{tight_min:.3f}",
                    f"{tight_max:.3f}",
                    f"{relaxed_min:.3f}",
                    f"{relaxed_max:.3f}",
                )
            for column, value in enumerate(values, start=4):
                item = self.bond_table.item(row, column)
                if item is None:
                    item = self._readonly_table_item(value)
                    self.bond_table.setItem(row, column, item)
                else:
                    item.setText(value)
        self._populating_bond_table = False

    def _reference_bond_length(
        self,
        reference_name: str,
        atom1_name: str,
        atom2_name: str,
    ) -> float | None:
        atom_coordinates = self._reference_coordinates(reference_name)
        coord1 = atom_coordinates.get(
            _normalized_table_atom_name(atom1_name, fallback="A1")
        )
        coord2 = atom_coordinates.get(
            _normalized_table_atom_name(atom2_name, fallback="A2")
        )
        if coord1 is None or coord2 is None:
            return None
        return float(dist(coord1, coord2))

    def _reference_coordinates(
        self,
        reference_name: str,
    ) -> dict[str, tuple[float, float, float]]:
        atom_coordinates = self._reference_atom_coordinates.get(reference_name)
        if atom_coordinates is not None:
            return atom_coordinates
        reference_path = self._reference_paths.get(reference_name)
        if not reference_path:
            return {}
        structure = PDBStructure.from_file(reference_path)
        atom_coordinates = {}
        for index, atom in enumerate(structure.atoms, start=1):
            fallback = f"{atom.element}{index}"
            atom_coordinates[
                _normalized_table_atom_name(
                    atom.atom_name,
                    fallback=fallback,
                )
            ] = tuple(float(value) for value in atom.coordinates)
        self._reference_atom_coordinates[reference_name] = atom_coordinates
        return atom_coordinates

    def _populate_angle_table(
        self,
        reference_name: str,
        bond_inputs: tuple[ReferenceBondToleranceInput, ...],
    ) -> None:
        self.angle_table.setRowCount(0)
        atom_coordinates = self._reference_coordinates(reference_name)
        if not atom_coordinates:
            return
        adjacency: dict[str, set[str]] = {}
        for bond in bond_inputs:
            atom1 = _normalized_table_atom_name(
                bond.atom1_name,
                fallback="A1",
            )
            atom2 = _normalized_table_atom_name(
                bond.atom2_name,
                fallback="A2",
            )
            adjacency.setdefault(atom1, set()).add(atom2)
            adjacency.setdefault(atom2, set()).add(atom1)
        rows: list[tuple[str, str, str, float]] = []
        for center, neighbors in sorted(adjacency.items()):
            sorted_neighbors = sorted(neighbors)
            for first_index, atom1 in enumerate(sorted_neighbors):
                for atom3 in sorted_neighbors[first_index + 1 :]:
                    angle = self._reference_angle_degrees(
                        atom_coordinates,
                        atom1,
                        center,
                        atom3,
                    )
                    if angle is not None:
                        rows.append((atom1, center, atom3, angle))
        for row, (atom1, center, atom3, angle) in enumerate(rows):
            self.angle_table.insertRow(row)
            for column, value in enumerate(
                (atom1, center, atom3, f"{angle:.3f}")
            ):
                self.angle_table.setItem(
                    row,
                    column,
                    self._readonly_table_item(value),
                )

    def _reference_angle_degrees(
        self,
        atom_coordinates: dict[str, tuple[float, float, float]],
        atom1: str,
        center: str,
        atom3: str,
    ) -> float | None:
        coord1 = atom_coordinates.get(atom1)
        center_coord = atom_coordinates.get(center)
        coord3 = atom_coordinates.get(atom3)
        if coord1 is None or center_coord is None or coord3 is None:
            return None
        vector1 = [float(a) - float(b) for a, b in zip(coord1, center_coord)]
        vector3 = [float(a) - float(b) for a, b in zip(coord3, center_coord)]
        norm1 = sum(value * value for value in vector1) ** 0.5
        norm3 = sum(value * value for value in vector3) ** 0.5
        if norm1 <= 0.0 or norm3 <= 0.0:
            return None
        dot_product = sum(a * b for a, b in zip(vector1, vector3))
        cosine = max(min(dot_product / (norm1 * norm3), 1.0), -1.0)
        return float(degrees(acos(cosine)))

    def _bond_search_bounds(
        self,
        reference_length: float,
        tolerance_percent: float,
        pass_scale: float,
    ) -> tuple[float, float]:
        absolute_tolerance = (
            max(float(reference_length), 0.0)
            * max(float(tolerance_percent), 0.0)
            / 100.0
        )
        scaled_tolerance = absolute_tolerance * max(float(pass_scale), 0.0)
        minimum = max(float(reference_length) - scaled_tolerance, 0.0)
        maximum = float(reference_length) + scaled_tolerance
        return minimum, maximum

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
        target_output_dir = suggest_output_dir(job.input_path)
        workflow = XYZToPDBMappingWorkflow(
            job.input_path,
            reference_library_dir=job.reference_library_dir,
        )
        self.log.emit(
            f"[{job.project_dir.name}] Target PDB folder: "
            f"{target_output_dir.name} ({target_output_dir})"
        )
        self.item_progress.emit(
            item_id,
            0,
            1,
            f"Target PDB folder: {target_output_dir.name}",
        )

        def on_progress(
            processed: int,
            total: int,
            message: str,
        ) -> None:
            self.item_progress.emit(item_id, processed, total, message)

        def on_log(message: str) -> None:
            if _should_emit_batch_log_message(message):
                self.log.emit(f"[{job.project_dir.name}] {message}")

        result: XYZToPDBExportResult = workflow.export_with_mapping(
            molecule_inputs=job.molecule_inputs,
            free_atom_inputs=job.free_atom_inputs,
            hydrogen_mode=job.hydrogen_mode,
            output_dir=target_output_dir,
            progress_callback=on_progress,
            log_callback=on_log,
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
        self._last_progress_log_step_by_item: dict[str, tuple[int, int]] = {}
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
        self.console.document().setMaximumBlockCount(_BATCH_CONSOLE_MAX_BLOCKS)
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
        progress_dialog = self._project_load_progress_dialog(
            len(selected_dirs)
        )
        try:
            for index, project_dir in enumerate(selected_dirs, start=1):
                if progress_dialog is not None:
                    progress_dialog.setLabelText(
                        "Loading XYZ -> PDB project "
                        f"{index}/{len(selected_dirs)}:\n{project_dir}"
                    )
                    progress_dialog.setValue(index - 1)
                    QApplication.processEvents()
                item = _queue_item_from_project_defaults(
                    project_dir,
                    reference_library_dir=self._reference_library_dir,
                )
                self.add_queue_item(
                    item,
                    auto_analyze=item.input_path is not None,
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
            "Loading selected XYZ -> PDB projects...",
            None,
            0,
            project_count,
            self,
        )
        dialog.setWindowTitle("Loading XYZ -> PDB Projects")
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
        self._last_progress_log_step_by_item.clear()
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

    def _should_log_progress(
        self,
        item_id: str,
        processed: int,
        total: int,
    ) -> bool:
        if not _is_progress_milestone(processed, total):
            return False
        key = (max(int(processed), 0), max(int(total), 1))
        if (
            key[0] < key[1]
            and self._last_progress_log_step_by_item.get(item_id) == key
        ):
            return False
        self._last_progress_log_step_by_item[item_id] = key
        return True

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
        self._last_progress_log_step_by_item.pop(item_id, None)
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
        display_name = item_id
        if widget is not None:
            widget.set_progress(processed, total)
            widget.set_status(message)
            display_name = widget.item().display_name()
        progress_text = f"[{display_name}] {processed}/{total}: {message}"
        self.queue_status_label.setText(progress_text)
        self.statusBar().showMessage(message)
        if self._should_log_progress(item_id, processed, total):
            self._append_log(progress_text)

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
