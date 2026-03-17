from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QFormLayout
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QTextEdit
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget


class ExportPanel(QGroupBox):
    """Panel for choosing export settings and writing frames."""

    export_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Export")
        self._suggested_output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        form = QFormLayout()

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setToolTip(
            "Directory where the selected frames will be written. By default "
            "this is a new subfolder inside the input trajectory directory "
            "unless you override it."
        )
        self.output_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Output directory",
            self._make_dir_row(self.output_dir_edit),
        )

        self.use_cutoff_box = QCheckBox("Apply selected cutoff")
        self.use_cutoff_box.setToolTip(
            "Only export frames at or after the currently selected cutoff "
            "time."
        )
        self.use_cutoff_box.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        form.addRow("", self.use_cutoff_box)

        layout.addLayout(form)

        layout.addWidget(QLabel("Selection Preview"))
        self.selection_box = QTextEdit()
        self.selection_box.setReadOnly(True)
        self.selection_box.setMinimumHeight(120)
        self.selection_box.setToolTip(
            "Preview of how many frames match the current export settings."
        )
        layout.addWidget(self.selection_box)

        self.export_button = QPushButton("Export Frames")
        self.export_button.setToolTip(
            "Write the currently selected frames to the output directory."
        )
        self.export_button.clicked.connect(
            lambda _checked=False: self.export_requested.emit()
        )
        layout.addWidget(self.export_button)

        layout.addWidget(QLabel("Export Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        self.log_box.setToolTip(
            "Summary of the last frame export operation."
        )
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.setToolTip("Browse for the output directory.")
        button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
        )
        if path:
            line_edit.setText(path)

    def get_output_dir(self) -> Path | None:
        text = self.output_dir_edit.text().strip()
        return Path(text) if text else None

    def suggest_output_dir(self, output_dir: Path) -> None:
        current = self.get_output_dir()
        should_update = (
            current is None or current == self._suggested_output_dir
        )
        self._suggested_output_dir = output_dir
        if should_update and self.output_dir_edit.text() != str(output_dir):
            self.output_dir_edit.setText(str(output_dir))

    def use_cutoff(self) -> bool:
        return self.use_cutoff_box.isChecked()

    def set_selection_summary(self, text: str) -> None:
        self.selection_box.setPlainText(text)

    def set_log(self, text: str) -> None:
        self.log_box.setPlainText(text)

    def append_log(self, text: str) -> None:
        current = self.log_box.toPlainText().strip()
        if not current:
            self.log_box.setPlainText(text)
            return
        self.log_box.setPlainText(f"{current}\n{text}")
