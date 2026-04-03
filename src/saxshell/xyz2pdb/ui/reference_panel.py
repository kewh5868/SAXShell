from __future__ import annotations

from collections import Counter
import re
from pathlib import Path

from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.structure import PDBStructure
from saxshell.xyz2pdb import ReferenceLibraryEntry


class ReferenceLibraryPanel(QGroupBox):
    """Panel for browsing and creating reference-molecule PDBs."""

    refresh_requested = Signal()
    create_requested = Signal()
    selection_changed = Signal()
    library_dir_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__("Reference Molecules")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(12)

        form_column = QWidget()
        form_layout = QVBoxLayout(form_column)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(10)

        form_layout.addWidget(QLabel("Reference library"))
        self.library_dir_edit = QLineEdit()
        self.library_dir_edit.setToolTip(
            "Folder that stores the reference molecule PDB files used by "
            "xyz2pdb."
        )
        self.library_dir_edit.textChanged.connect(
            lambda _text: self.library_dir_changed.emit(self.get_library_dir())
        )
        form_layout.addWidget(
            self._make_dir_row(
                self.library_dir_edit,
                title="Select reference library folder",
            )
        )

        create_group = QGroupBox("Add Reference Molecule")
        create_layout = QFormLayout(create_group)
        create_layout.setContentsMargins(10, 12, 10, 10)
        create_layout.setHorizontalSpacing(8)
        create_layout.setVerticalSpacing(8)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Choose a source XYZ or PDB file")
        self.source_edit.textChanged.connect(
            lambda _text: self._refresh_backbone_pair_controls()
        )
        create_layout.addRow(
            "Source PDB/XYZ",
            self._make_file_row(self.source_edit),
        )

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("dmf")
        create_layout.addRow("Reference name", self.name_edit)

        self.residue_edit = QLineEdit()
        self.residue_edit.setPlaceholderText("DMF")
        self.residue_edit.setToolTip(
            "Residue code written into the saved reference PDB. "
            "When provided, it must be exactly three capital letters."
        )
        _configure_residue_line_edit(self.residue_edit)
        create_layout.addRow("Residue name", self.residue_edit)

        self.backbone_atom1_combo = QComboBox()
        self.backbone_atom2_combo = QComboBox()
        for combo in (self.backbone_atom1_combo, self.backbone_atom2_combo):
            combo.setToolTip(
                "Preferred backbone pair stored with the reference. "
                "Manual selection is only available for PDB sources."
            )
        create_layout.addRow(
            "Preferred backbone pair",
            self._make_backbone_pair_row(),
        )

        self.backbone_help_label = QLabel()
        self.backbone_help_label.setWordWrap(True)
        self.backbone_help_label.setStyleSheet("color: #555;")
        create_layout.addRow("", self.backbone_help_label)

        add_button = QPushButton("Create Reference")
        add_button.clicked.connect(
            lambda _checked=False: self.create_requested.emit()
        )
        create_layout.addRow("", add_button)

        self._refresh_backbone_pair_controls()

        form_layout.addWidget(create_group)
        form_layout.addStretch(1)

        browser_column = QWidget()
        browser_layout = QVBoxLayout(browser_column)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Available references"))
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(
            lambda _index: self.selection_changed.emit()
        )
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(
            lambda _checked=False: self.refresh_requested.emit()
        )
        header_row.addWidget(self.reference_combo, stretch=1)
        header_row.addWidget(refresh_button)
        browser_layout.addLayout(header_row)

        browser_layout.addWidget(QLabel("Reference information"))
        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(220)
        browser_layout.addWidget(self.details_box, stretch=1)

        content_row.addWidget(form_column, stretch=5)
        content_row.addWidget(browser_column, stretch=6)
        layout.addLayout(content_row)
        self.setLayout(layout)

    def _make_file_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_source_file(line_edit)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _make_backbone_pair_row(self) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.backbone_atom1_combo, stretch=1)
        row.addWidget(QLabel("to"))
        row.addWidget(self.backbone_atom2_combo, stretch=1)
        return widget

    def _make_dir_row(self, line_edit: QLineEdit, *, title: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit, title=title)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_source_file(self, line_edit: QLineEdit) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select source structure file",
            filter="Structure files (*.pdb *.xyz);;All files (*)",
        )
        if path:
            line_edit.setText(path)

    def _choose_dir(self, line_edit: QLineEdit, *, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def _refresh_backbone_pair_controls(self) -> None:
        source_path = self.get_source_path()
        atom_names: tuple[str, ...] = ()
        if source_path is not None and source_path.suffix.lower() == ".pdb":
            try:
                atom_names = _stored_reference_source_atom_names(source_path)
            except Exception:
                atom_names = ()

        for combo in (self.backbone_atom1_combo, self.backbone_atom2_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Auto", None)
            for atom_name in atom_names:
                combo.addItem(atom_name, atom_name)
            combo.setCurrentIndex(0)
            combo.setEnabled(len(atom_names) >= 2)
            combo.blockSignals(False)

        if source_path is None or not str(source_path).strip():
            self.backbone_help_label.setText(
                "Choose a source file first. PDB sources let you select a "
                "preferred backbone pair. XYZ sources are auto-formatted into "
                "a stored PDB and use inferred backbone pairs."
            )
        elif source_path.suffix.lower() == ".pdb":
            if len(atom_names) >= 2:
                preview_text = ", ".join(atom_names[:8])
                if len(atom_names) > 8:
                    preview_text += ", ..."
                self.backbone_help_label.setText(
                    "PDB source detected. You can leave the pair on Auto to "
                    "infer backbone pairs, or choose one preferred pair from "
                    "the stored atom names: "
                    f"{preview_text}"
                )
            else:
                self.backbone_help_label.setText(
                    "This PDB source did not provide at least two usable atom "
                    "names, so backbone pairs will be inferred automatically."
                )
        else:
            self.backbone_help_label.setText(
                "XYZ source detected. xyz2pdb will assign atom names and save "
                "a reference PDB automatically, then infer backbone pairs. "
                "Manual backbone selection is only available for PDB sources."
            )

    def set_reference_entries(
        self,
        entries: list[ReferenceLibraryEntry],
        *,
        preferred_name: str | None = None,
    ) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        for entry in entries:
            self.reference_combo.addItem(entry.name, entry)
        self.reference_combo.blockSignals(False)

        if not entries:
            self.details_box.setPlainText(
                "No reference PDB files were found in the current library folder."
            )
            return

        selected_index = 0
        if preferred_name:
            for index, entry in enumerate(entries):
                if entry.name == preferred_name:
                    selected_index = index
                    break
        self.reference_combo.setCurrentIndex(selected_index)
        self.update_details()

    def update_details(self) -> None:
        entry = self.current_entry()
        if entry is None:
            self.details_box.setPlainText(
                "Select a reference molecule to inspect it."
            )
            return
        preview_names = ", ".join(entry.atom_names[:12])
        if len(entry.atom_names) > 12:
            preview_names += ", ..."
        self.details_box.setPlainText(
            "\n".join(
                [
                    f"Name: {entry.name}",
                    f"Path: {entry.path}",
                    f"Residue name: {entry.residue_name}",
                    f"Atom count: {entry.atom_count}",
                    "Backbone pairs: "
                    + (
                        ", ".join(
                            f"{atom1_name}-{atom2_name}"
                            for atom1_name, atom2_name in entry.backbone_pairs
                        )
                        if entry.backbone_pairs
                        else "auto"
                    ),
                    f"Atom names: {preview_names}",
                ]
            )
        )

    def current_entry(self) -> ReferenceLibraryEntry | None:
        data = self.reference_combo.currentData()
        return data if isinstance(data, ReferenceLibraryEntry) else None

    def get_source_path(self) -> Path | None:
        text = self.source_edit.text().strip()
        return Path(text) if text else None

    def get_library_dir(self) -> Path | None:
        text = self.library_dir_edit.text().strip()
        return Path(text) if text else None

    def get_reference_name(self) -> str:
        return self.name_edit.text().strip()

    def get_residue_name(self) -> str | None:
        text = self.residue_edit.text().strip()
        if text and not _is_valid_residue_code(text):
            raise ValueError(
                "Reference residues must be exactly three capital letters."
            )
        return text or None

    def get_backbone_pairs(self) -> tuple[tuple[str, str], ...] | None:
        source_path = self.get_source_path()
        if source_path is None or source_path.suffix.lower() != ".pdb":
            return None
        atom1_name = self.backbone_atom1_combo.currentData()
        atom2_name = self.backbone_atom2_combo.currentData()
        if atom1_name is None and atom2_name is None:
            return None
        if atom1_name is None or atom2_name is None:
            raise ValueError(
                "Choose both backbone atoms or leave both backbone selectors on Auto."
            )
        if atom1_name == atom2_name:
            raise ValueError(
                "Backbone atom 1 and atom 2 must be different."
            )
        return ((str(atom1_name), str(atom2_name)),)


def _configure_residue_line_edit(line_edit: QLineEdit) -> None:
    line_edit.setMaxLength(3)
    line_edit.setValidator(
        QRegularExpressionValidator(
            QRegularExpression(r"[A-Z]{0,3}"),
            line_edit,
        )
    )
    line_edit.textChanged.connect(
        lambda text, edit=line_edit: _normalize_residue_line_edit(edit, text)
    )


def _normalize_residue_line_edit(line_edit: QLineEdit, text: str) -> None:
    normalized = re.sub(r"[^A-Za-z]", "", str(text or "")).upper()[:3]
    if normalized == text:
        return
    cursor_position = min(line_edit.cursorPosition(), len(normalized))
    line_edit.blockSignals(True)
    line_edit.setText(normalized)
    line_edit.setCursorPosition(cursor_position)
    line_edit.blockSignals(False)


def _is_valid_residue_code(value: str) -> bool:
    return re.fullmatch(r"[A-Z]{3}", str(value or "").strip()) is not None


def _stored_reference_source_atom_names(source_path: Path) -> tuple[str, ...]:
    structure = PDBStructure.from_file(source_path)
    atoms = [atom.copy() for atom in structure.atoms]
    if not atoms:
        return ()
    unique_names = {atom.atom_name for atom in atoms if atom.atom_name}
    should_rename = len(unique_names) != len(atoms)
    if should_rename:
        counters: Counter[str] = Counter()
        for atom in atoms:
            counters[atom.element] += 1
            atom.atom_name = _normalized_reference_atom_name(
                atom.atom_name,
                fallback=f"{atom.element}{counters[atom.element]}",
            )
    normalized_names: list[str] = []
    for index, atom in enumerate(atoms, start=1):
        normalized_names.append(
            _normalized_reference_atom_name(
                atom.atom_name,
                fallback=f"{atom.element}{index}",
            )
        )
    return tuple(normalized_names)


def _normalized_reference_atom_name(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).strip().upper()
    if not text:
        text = fallback.upper()
    return text[:4]
