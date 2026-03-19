from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from saxshell.saxs.project_manager import (
    ExperimentalDataSummary,
    guess_experimental_header_rows,
    infer_experimental_columns,
    load_experimental_data_file,
    read_experimental_column_names,
)


class ExperimentalDataHeaderDialog(QDialog):
    """Recover experimental-data loading by choosing header rows and
    columns."""

    def __init__(
        self,
        file_path: str | Path,
        parent: QWidget | None = None,
        *,
        initial_header_rows: int | None = None,
        initial_q_column: int | None = None,
        initial_intensity_column: int | None = None,
        initial_error_column: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_path = Path(file_path).expanduser().resolve()
        self._accepted_summary: ExperimentalDataSummary | None = None
        self._preview_lines = self._read_preview_lines()
        self._current_column_names: list[str] = []
        self._initial_q_column = initial_q_column
        self._initial_intensity_column = initial_intensity_column
        self._initial_error_column = initial_error_column
        self._build_ui()
        detected_header_rows = (
            initial_header_rows
            if initial_header_rows is not None
            else guess_experimental_header_rows(self.file_path)
        )
        self.header_rows_spin.setValue(detected_header_rows)
        self._refresh_state()

    @property
    def accepted_summary(self) -> ExperimentalDataSummary | None:
        return self._accepted_summary

    def header_rows(self) -> int:
        return int(self.header_rows_spin.value())

    def q_column(self) -> int:
        return int(self.q_column_combo.currentData())

    def intensity_column(self) -> int:
        return int(self.intensity_column_combo.currentData())

    def error_column(self) -> int | None:
        data = self.error_column_combo.currentData()
        return None if data is None else int(data)

    def _build_ui(self) -> None:
        self.setWindowTitle("Check Experimental Data File")
        self.resize(900, 720)

        root = QVBoxLayout(self)
        intro_label = QLabel(
            "The selected file could not be parsed directly. Adjust the "
            "number of header rows to skip, confirm which columns correspond "
            "to q, intensity, and error, and then load the file again."
        )
        intro_label.setWordWrap(True)
        root.addWidget(intro_label)

        form = QFormLayout()
        self.file_label = QLabel(str(self.file_path))
        self.file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("File", self.file_label)

        self.header_rows_spin = QSpinBox()
        self.header_rows_spin.setRange(0, max(len(self._preview_lines), 0))
        self.header_rows_spin.valueChanged.connect(self._refresh_state)
        form.addRow("Header rows", self.header_rows_spin)

        self.q_column_combo = QComboBox()
        form.addRow("q column", self.q_column_combo)

        self.intensity_column_combo = QComboBox()
        form.addRow("Intensity column", self.intensity_column_combo)

        self.error_column_combo = QComboBox()
        form.addRow("Error column", self.error_column_combo)
        root.addLayout(form)

        self.preview_box = QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.preview_box.setMinimumHeight(420)
        root.addWidget(self.preview_box)

        self.status_label = QLabel(
            "Adjust the header length and selected columns, then click Load File."
        )
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Open
        )
        self.load_button = button_box.button(
            QDialogButtonBox.StandardButton.Open
        )
        self.load_button.setText("Load File")
        button_box.accepted.connect(self._try_accept)
        button_box.rejected.connect(self.reject)
        root.addWidget(button_box)

    def _read_preview_lines(self, max_lines: int = 80) -> list[str]:
        lines: list[str] = []
        with self.file_path.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for index, line in enumerate(handle):
                if index >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return lines

    def _refresh_state(self) -> None:
        self._refresh_preview()
        self._refresh_column_choices()

    def _refresh_preview(self) -> None:
        highlight_count = self.header_rows()
        rendered_lines = []
        for index, line in enumerate(self._preview_lines):
            prefix = ">> " if index < highlight_count else "   "
            rendered_lines.append(f"{prefix}{index + 1:>3}: {line}")
        self.preview_box.setPlainText("\n".join(rendered_lines))

    def _refresh_column_choices(self) -> None:
        self._current_column_names = read_experimental_column_names(
            self.file_path,
            skiprows=self.header_rows(),
        )
        if not self._current_column_names:
            self._current_column_names = ["Column 1", "Column 2"]
        self._populate_column_combo(
            self.q_column_combo,
            self._current_column_names,
            selected_index=self._initial_q_column,
        )
        inferred_q, inferred_i, inferred_e = infer_experimental_columns(
            self._current_column_names
        )
        self._populate_column_combo(
            self.intensity_column_combo,
            self._current_column_names,
            selected_index=(
                self._initial_intensity_column
                if self._initial_intensity_column is not None
                else inferred_i
            ),
            fallback_index=1 if len(self._current_column_names) > 1 else 0,
        )
        self._populate_error_combo(
            selected_index=(
                self._initial_error_column
                if self._initial_error_column is not None
                else inferred_e
            )
        )

        if self._initial_q_column is None:
            self.q_column_combo.setCurrentIndex(
                inferred_q if inferred_q is not None else 0
            )

        self.status_label.setText(
            "Detected columns: " + ", ".join(self._current_column_names)
        )

    def _populate_column_combo(
        self,
        combo: QComboBox,
        column_names: list[str],
        *,
        selected_index: int | None,
        fallback_index: int = 0,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for index, name in enumerate(column_names):
            combo.addItem(name, index)
        target_index = selected_index
        if target_index is None or not (0 <= target_index < len(column_names)):
            target_index = fallback_index
        combo.setCurrentIndex(max(0, min(target_index, combo.count() - 1)))
        combo.blockSignals(False)

    def _populate_error_combo(self, *, selected_index: int | None) -> None:
        self.error_column_combo.blockSignals(True)
        self.error_column_combo.clear()
        self.error_column_combo.addItem("None", None)
        for index, name in enumerate(self._current_column_names):
            self.error_column_combo.addItem(name, index)
        if selected_index is None:
            self.error_column_combo.setCurrentIndex(0)
        else:
            combo_index = self.error_column_combo.findData(selected_index)
            self.error_column_combo.setCurrentIndex(
                combo_index if combo_index >= 0 else 0
            )
        self.error_column_combo.blockSignals(False)

    def _try_accept(self) -> None:
        try:
            summary = load_experimental_data_file(
                self.file_path,
                skiprows=self.header_rows(),
                q_column=self.q_column(),
                intensity_column=self.intensity_column(),
                error_column=self.error_column(),
            )
        except Exception as exc:
            self.status_label.setText(
                "Parsing still failed. Adjust the header length or column "
                "selection and try again.\n"
                f"{html.escape(str(exc))}"
            )
            return
        self._accepted_summary = summary
        self.accept()


__all__ = ["ExperimentalDataHeaderDialog"]
