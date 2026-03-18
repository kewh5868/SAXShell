from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from saxshell.xyz2pdb import ReferenceLibraryEntry


class ReferenceLibraryPanel(QGroupBox):
    """Panel for browsing and creating reference-molecule PDBs."""

    refresh_requested = Signal()
    create_requested = Signal()
    selection_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Reference Molecules")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

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
        layout.addLayout(header_row)

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(160)
        layout.addWidget(self.details_box)

        create_group = QGroupBox("Add Reference Molecule")
        create_layout = QFormLayout(create_group)

        self.source_edit = QLineEdit()
        create_layout.addRow(
            "Source PDB/XYZ",
            self._make_file_row(self.source_edit),
        )

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("dmf")
        create_layout.addRow("Reference name", self.name_edit)

        self.residue_edit = QLineEdit()
        self.residue_edit.setPlaceholderText("DMF")
        create_layout.addRow("Residue name", self.residue_edit)

        add_button = QPushButton("Create Reference")
        add_button.clicked.connect(
            lambda _checked=False: self.create_requested.emit()
        )
        create_layout.addRow("", add_button)

        layout.addWidget(create_group)
        self.setLayout(layout)

    def _make_file_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_source_file(line_edit)
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

    def get_reference_name(self) -> str:
        return self.name_edit.text().strip()

    def get_residue_name(self) -> str | None:
        text = self.residue_edit.text().strip()
        return text or None

