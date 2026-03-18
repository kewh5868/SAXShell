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


class ClusterExportPanel(QGroupBox):
    """Panel for output settings, preview, and cluster export."""

    export_requested = Signal()
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Run")
        self._suggested_output_dir: Path | None = None
        self._progress_phase = "idle"
        self._progress_processed = 0
        self._progress_total = 1
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        form = QFormLayout()

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setToolTip(
            "Root output directory for the extracted cluster files. "
            "Finished exports are sorted into stoichiometry folders such "
            "as Pb2I or Pb."
        )
        self.output_dir_edit.textChanged.connect(
            lambda _text: self.settings_changed.emit()
        )
        form.addRow(
            "Output directory",
            self._make_dir_row(self.output_dir_edit),
        )
        layout.addLayout(form)

        layout.addWidget(QLabel("Selection Preview"))
        self.selection_box = QTextEdit()
        self.selection_box.setReadOnly(True)
        self.selection_box.setMinimumHeight(140)
        self.selection_box.setToolTip(
            "Preview of the frames that will be analyzed and the current "
            "cluster export destination."
        )
        layout.addWidget(self.selection_box)

        self.export_button = QPushButton("Analyze and Export Clusters")
        self.export_button.setToolTip(
            "Run cluster analysis on the selected frames and write one PDB "
            "file per cluster."
        )
        self.export_button.clicked.connect(
            lambda _checked=False: self.export_requested.emit()
        )
        layout.addWidget(self.export_button)

        self.progress_label = QLabel("Progress: idle")
        self.progress_label.setToolTip(
            "How many extracted frames have been processed during the "
            "current cluster-analysis run."
        )
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setToolTip(
            "Progress through the extracted frames selected for cluster "
            "analysis."
        )
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Run Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.log_box.setToolTip(
            "Status updates and results from the last cluster-analysis run."
        )
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.set_frame_mode(None)
        self.reset_progress()

    def _make_dir_row(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)

        button = QPushButton("Browse")
        button.setToolTip("Browse for the cluster output directory.")
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

    def set_selection_summary(self, text: str) -> None:
        self.selection_box.setPlainText(text)

    def set_log(self, text: str) -> None:
        self.log_box.setPlainText(text)

    def append_log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return

        scroll_bar = self.log_box.verticalScrollBar()
        previous_value = scroll_bar.value()
        was_at_bottom = previous_value >= max(scroll_bar.maximum() - 4, 0)

        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.log_box.document().characterCount() > 1:
            cursor.insertBlock()
        cursor.insertText(message)
        self.log_box.setTextCursor(cursor)

        if was_at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())
        else:
            scroll_bar.setValue(previous_value)

    def reset_progress(self) -> None:
        self._progress_phase = "idle"
        self._progress_processed = 0
        self._progress_total = 1
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m frames")
        self.progress_label.setText("Progress: idle")

    def set_progress_phase(self, phase: str) -> None:
        self._progress_phase = str(phase)
        if self._progress_phase == "idle":
            self.progress_label.setText("Progress: idle")
            return
        self._update_progress_label(
            self._progress_processed,
            self._progress_total,
        )

    def update_progress(self, processed: int, total: int) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self._progress_processed = processed
        self._progress_total = total
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat("%v / %m frames")
        self._update_progress_label(processed, total)

    def _update_progress_label(self, processed: int, total: int) -> None:
        remaining = max(total - processed, 0)
        if self._progress_phase == "sorting":
            label = "Sorting"
        elif self._progress_phase == "extracting":
            label = "Extracting"
        else:
            label = "Progress"
        self.progress_label.setText(
            f"{label}: {processed} processed, {remaining} remaining"
        )

    def set_frame_mode(self, frame_format: str | None) -> None:
        if frame_format == "pdb":
            button_text = "Analyze and Export Cluster PDBs"
            button_tooltip = (
                "Run cluster analysis on the selected frames and write one "
                "PDB file per cluster."
            )
        elif frame_format == "xyz":
            button_text = "Analyze and Export Cluster XYZs"
            button_tooltip = (
                "Run cluster analysis on the selected frames and write one "
                "XYZ file per cluster."
            )
        else:
            button_text = "Analyze and Export Clusters"
            button_tooltip = (
                "Run cluster analysis on the selected frames and write one "
                "cluster file per detected cluster."
            )

        self.export_button.setText(button_text)
        self.export_button.setToolTip(button_tooltip)
