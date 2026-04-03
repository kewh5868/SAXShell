from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

    estimate_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Convert")
        self._suggested_output_dir: Path | None = None
        self._last_log_message = ""
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

        layout.addWidget(QLabel("Mapping Summary"))
        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(140)
        layout.addWidget(self.preview_box)

        actions_row = QHBoxLayout()
        self.estimate_button = QPushButton("Estimate PDB Mapping")
        self.estimate_button.clicked.connect(
            lambda _checked=False: self.estimate_requested.emit()
        )
        self.export_button = QPushButton("Convert XYZ Frames to PDB")
        self.export_button.clicked.connect(
            lambda _checked=False: self.export_requested.emit()
        )
        self.cancel_button = QPushButton("Cancel Mapping")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip(
            "Stop the current xyz2pdb mapping/conversion run so you can "
            "adjust parameters and retry."
        )
        self.cancel_button.clicked.connect(
            lambda _checked=False: self.cancel_requested.emit()
        )
        self.assertion_mode_checkbox = QCheckBox("Assertion Mode")
        self.assertion_mode_checkbox.setChecked(False)
        self.assertion_mode_checkbox.setToolTip(
            "When enabled, xyz2pdb writes per-molecule PDB files into an "
            "assertion folder and compares each molecule's internal "
            "distance distribution against the reference molecule and the "
            "rest of the exported set. Use it to spot skewed or unusually "
            "distorted molecules after conversion."
        )
        self.assertion_mode_checkbox.toggled.connect(
            lambda _checked: self.settings_changed.emit()
        )
        actions_row.addWidget(self.estimate_button)
        actions_row.addWidget(self.assertion_mode_checkbox)
        actions_row.addWidget(self.export_button)
        actions_row.addWidget(self.cancel_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        solution_row = QHBoxLayout()
        solution_row.addWidget(QLabel("Estimate solution"))
        self.solution_combo = QComboBox()
        self.solution_combo.setEnabled(False)
        self.solution_combo.currentIndexChanged.connect(
            lambda _index: self.settings_changed.emit()
        )
        solution_row.addWidget(self.solution_combo)
        layout.addLayout(solution_row)

        self.progress_label = QLabel("Progress: idle")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m steps")
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Run Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        self.log_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        self.output_dir_button = QPushButton("Browse")
        self.output_dir_button.clicked.connect(
            lambda _checked=False: self._choose_dir(line_edit)
        )

        row.addWidget(line_edit)
        row.addWidget(self.output_dir_button)
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
        self._last_log_message = ""
        self.log_box.setPlainText(self._format_initial_log_text(text))

    def append_log(self, text: str) -> None:
        message = self._normalize_log_message(text)
        if not message:
            return
        if message == self._last_log_message:
            return
        self._last_log_message = message
        scroll_bar = self.log_box.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= max(scroll_bar.maximum() - 4, 0)
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.log_box.document().characterCount() > 1:
            cursor.insertBlock()
        cursor.insertText(self._format_log_entry(message))
        self.log_box.setTextCursor(cursor)
        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())

    def set_controls_enabled(self, enabled: bool) -> None:
        self.output_dir_edit.setEnabled(enabled)
        self.output_dir_button.setEnabled(enabled)
        self.estimate_button.setEnabled(enabled)
        self.assertion_mode_checkbox.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)
        self.solution_combo.setEnabled(
            enabled and self.solution_combo.count() > 0
        )

    def reset_progress(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m steps")
        self.progress_label.setText("Progress: idle")

    def set_busy_progress(self, text: str) -> None:
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Working...")
        self.progress_label.setText(text)

    def update_progress(
        self,
        processed: int,
        total: int,
        text: str | None = None,
    ) -> None:
        if total <= 0:
            self.set_busy_progress(text or "Working...")
            return
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m steps")
        if text:
            self.progress_label.setText(text)
        else:
            self.progress_label.setText(
                f"Progress: {processed} processed, {max(total - processed, 0)} remaining"
            )

    def set_progress_complete(
        self,
        text: str,
        *,
        total: int | None = None,
    ) -> None:
        maximum = max(1, int(total) if total is not None else 1)
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(maximum)
        self.progress_bar.setFormat("%v / %m steps")
        self.progress_label.setText(text)

    def set_progress_failed(self, text: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m steps")
        self.progress_label.setText(text)

    def set_solution_options(
        self,
        options: list[str],
        *,
        selected_index: int = 0,
    ) -> None:
        self.solution_combo.blockSignals(True)
        self.solution_combo.clear()
        for index, option in enumerate(options):
            self.solution_combo.addItem(option, index)
        self.solution_combo.setEnabled(bool(options))
        if options:
            self.solution_combo.setCurrentIndex(
                max(0, min(int(selected_index), len(options) - 1))
            )
        self.solution_combo.blockSignals(False)

    def selected_solution_index(self) -> int:
        data = self.solution_combo.currentData()
        if isinstance(data, int):
            return data
        return max(self.solution_combo.currentIndex(), 0)

    def assertion_mode_enabled(self) -> bool:
        return bool(self.assertion_mode_checkbox.isChecked())

    def _normalize_log_message(self, text: str) -> str:
        lines = [line.rstrip() for line in str(text).splitlines()]
        normalized_lines = [line.strip() for line in lines if line.strip()]
        return "\n".join(normalized_lines)

    def _format_initial_log_text(self, text: str) -> str:
        message = self._normalize_log_message(text)
        if not message:
            return ""
        lines = message.splitlines()
        if len(lines) <= 1:
            return lines[0]
        return lines[0] + "\n\n" + "\n".join(lines[1:])

    def _format_log_entry(self, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        lines = message.splitlines()
        if len(lines) == 1:
            return f"[{timestamp}] {lines[0]}"
        formatted = [f"[{timestamp}] {lines[0]}"]
        formatted.extend(f"          {line}" for line in lines[1:])
        return "\n".join(formatted)
