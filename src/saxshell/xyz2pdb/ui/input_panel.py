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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class XYZToPDBInputPanel(QGroupBox):
    """Panel for selecting XYZ input, config, and reference library."""

    inspect_requested = Signal()
    preview_requested = Signal()
    settings_changed = Signal()
    input_path_changed = Signal(object)
    library_dir_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__("XYZ Input")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        self.mode_label = QLabel("Input: Auto-detect")
        layout.addWidget(self.mode_label)

        form = QFormLayout()

        self.input_edit = QLineEdit()
        self.input_edit.setToolTip(
            "Path to one XYZ file or a folder containing XYZ files."
        )
        self.input_edit.textChanged.connect(
            lambda _text: self.input_path_changed.emit(self.get_input_path())
        )
        self.input_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow("XYZ input", self._make_input_row())

        self.config_edit = QLineEdit()
        self.config_edit.setToolTip(
            "Residue-assignment JSON file describing reference molecules, "
            "anchor pairs, and free-atom residue names."
        )
        self.config_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Config JSON", self._make_file_row(self.config_edit, "json")
        )

        self.library_dir_edit = QLineEdit()
        self.library_dir_edit.setToolTip(
            "Folder that stores the reference molecule PDB files used by "
            "xyz2pdb."
        )
        self.library_dir_edit.textChanged.connect(
            lambda _text: self.library_dir_changed.emit(self.get_library_dir())
        )
        self.library_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Reference library",
            self._make_dir_row(
                self.library_dir_edit,
                title="Select reference library folder",
            ),
        )

        layout.addLayout(form)

        actions_row = QHBoxLayout()
        inspect_button = QPushButton("Inspect")
        inspect_button.clicked.connect(
            lambda _checked=False: self.inspect_requested.emit()
        )
        preview_button = QPushButton("Preview")
        preview_button.clicked.connect(
            lambda _checked=False: self.preview_requested.emit()
        )
        actions_row.addWidget(inspect_button)
        actions_row.addWidget(preview_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        layout.addWidget(QLabel("Input Summary"))
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(160)
        layout.addWidget(self.summary_box)

        self.setLayout(layout)

    def _make_input_row(self) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        file_button = QPushButton("Browse File")
        file_button.clicked.connect(
            lambda _checked=False: self._choose_input_file()
        )
        folder_button = QPushButton("Browse Folder")
        folder_button.clicked.connect(
            lambda _checked=False: self._choose_input_dir()
        )

        row.addWidget(self.input_edit)
        row.addWidget(file_button)
        row.addWidget(folder_button)
        return widget

    def _make_file_row(self, line_edit: QLineEdit, suffix: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_file(line_edit, suffix=suffix)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _make_dir_row(self, line_edit: QLineEdit, *, title: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit, title=title)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_file(self, line_edit: QLineEdit, *, suffix: str) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            f"Select {suffix.upper()} file",
            filter=f"{suffix.upper()} files (*.{suffix});;All files (*)",
        )
        if path:
            line_edit.setText(path)

    def _choose_dir(self, line_edit: QLineEdit, *, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def _choose_input_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select XYZ file",
            filter="XYZ files (*.xyz);;All files (*)",
        )
        if path:
            self.input_edit.setText(path)

    def _choose_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder of XYZ files",
        )
        if path:
            self.input_edit.setText(path)

    def get_input_path(self) -> Path | None:
        text = self.input_edit.text().strip()
        return Path(text) if text else None

    def get_config_path(self) -> Path | None:
        text = self.config_edit.text().strip()
        return Path(text) if text else None

    def get_library_dir(self) -> Path | None:
        text = self.library_dir_edit.text().strip()
        return Path(text) if text else None

    def set_input_mode(self, input_mode: str | None) -> None:
        if input_mode == "single_xyz":
            self.mode_label.setText("Input: Single XYZ file")
        elif input_mode == "xyz_folder":
            self.mode_label.setText("Input: XYZ folder")
        else:
            self.mode_label.setText("Input: Auto-detect")

    def set_summary_text(self, text: str) -> None:
        self.summary_box.setPlainText(text)
