from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(slots=True)
class DreamViolinExportOptions:
    output_dir: Path
    base_name: str
    save_csv: bool
    save_pkl: bool


class DreamViolinExportDialog(QDialog):
    def __init__(
        self,
        *,
        default_output_dir: str | Path,
        default_base_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._result: DreamViolinExportOptions | None = None
        self._default_output_dir = Path(default_output_dir).expanduser()
        self._default_base_name = default_base_name
        self._build_ui()

    @property
    def selected_options(self) -> DreamViolinExportOptions | None:
        return self._result

    def _build_ui(self) -> None:
        self.setWindowTitle("Save DREAM Violin Plot Data")
        self.resize(540, 260)

        root = QVBoxLayout(self)

        description = QLabel(
            "Choose where to save the DREAM violin-plot data and which "
            "formats to write. The PKL export stores the plotted arrays and "
            "metadata so the plot can be reconstructed later in Python or a "
            "Jupyter notebook."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        path_group = QGroupBox("Destination")
        path_layout = QFormLayout(path_group)
        self.output_dir_edit = QLineEdit(str(self._default_output_dir))
        path_layout.addRow("Output directory", self._path_row())
        self.base_name_edit = QLineEdit(self._default_base_name)
        path_layout.addRow("Base filename", self.base_name_edit)
        root.addWidget(path_group)

        format_group = QGroupBox("Formats")
        format_layout = QVBoxLayout(format_group)
        self.csv_checkbox = QCheckBox("CSV")
        self.csv_checkbox.setChecked(True)
        self.pkl_checkbox = QCheckBox("PKL")
        self.pkl_checkbox.setChecked(False)
        format_layout.addWidget(self.csv_checkbox)
        format_layout.addWidget(self.pkl_checkbox)
        root.addWidget(format_group)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._accept_if_valid)
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)
        root.addLayout(buttons)

    def _path_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.output_dir_edit, stretch=1)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_output_dir)
        layout.addWidget(browse_button)
        return row

    def _browse_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select export directory",
            self.output_dir_edit.text().strip()
            or str(self._default_output_dir),
        )
        if selected:
            self.output_dir_edit.setText(selected)

    def _accept_if_valid(self) -> None:
        output_dir_text = self.output_dir_edit.text().strip()
        base_name = self.base_name_edit.text().strip()
        if not output_dir_text:
            QMessageBox.warning(
                self,
                "Export options incomplete",
                "Choose an output directory first.",
            )
            return
        if not base_name:
            QMessageBox.warning(
                self,
                "Export options incomplete",
                "Enter a base filename first.",
            )
            return
        if not (
            self.csv_checkbox.isChecked() or self.pkl_checkbox.isChecked()
        ):
            QMessageBox.warning(
                self,
                "Export options incomplete",
                "Select at least one output format.",
            )
            return

        self._result = DreamViolinExportOptions(
            output_dir=Path(output_dir_text).expanduser(),
            base_name=base_name,
            save_csv=self.csv_checkbox.isChecked(),
            save_pkl=self.pkl_checkbox.isChecked(),
        )
        self.accept()


__all__ = ["DreamViolinExportDialog", "DreamViolinExportOptions"]
