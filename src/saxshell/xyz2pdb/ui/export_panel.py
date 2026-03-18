from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class XYZToPDBExportPanel(QGroupBox):
    """Panel for output settings, preview, and export progress."""

    export_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Convert")
        self._suggested_output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        form = QFormLayout()

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Output directory",
            self._make_dir_row(self.output_dir_edit),
        )
        layout.addLayout(form)

        layout.addWidget(QLabel("Conversion Preview"))
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(140)
        layout.addWidget(self.preview_box)

        self.export_button = QPushButton("Convert XYZ to PDB")
        self.export_button.clicked.connect(
            lambda _checked=False: self.export_requested.emit()
        )
        layout.addWidget(self.export_button)

        self.progress_label = QLabel("Progress: idle")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Run Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit)
        )

        row.addWidget(line_edit)
        row.addWidget(button)
        return widget

    def _choose_dir(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select xyz2pdb output directory",
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

    def set_preview_text(self, text: str) -> None:
        self.preview_box.setPlainText(text)

    def set_log_text(self, text: str) -> None:
        self.log_box.setPlainText(text)

    def append_log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        scroll_bar = self.log_box.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= max(scroll_bar.maximum() - 4, 0)
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.log_box.document().characterCount() > 1:
            cursor.insertBlock()
        cursor.insertText(message)
        self.log_box.setTextCursor(cursor)
        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())

    def reset_progress(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        self.progress_label.setText("Progress: idle")

    def update_progress(self, processed: int, total: int) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m files")
        self.progress_label.setText(
            f"Progress: {processed} processed, {max(total - processed, 0)} remaining"
        )

