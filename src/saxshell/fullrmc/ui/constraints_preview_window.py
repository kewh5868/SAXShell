from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ConstraintsPreviewWindow(QMainWindow):
    def __init__(
        self,
        constraints_path: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.constraints_path = Path(constraints_path).expanduser().resolve()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(
            f"Merged Constraints Preview - {self.constraints_path.name}"
        )
        self.resize(980, 760)
        self._build_ui()
        self._load_text()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.path_label = QLabel(str(self.constraints_path))
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.path_label.setWordWrap(True)
        root.addWidget(self.path_label)

        controls_row = QHBoxLayout()
        self.copy_all_button = QPushButton("Copy All")
        self.copy_all_button.clicked.connect(self._copy_all_text)
        controls_row.addWidget(self.copy_all_button)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self._load_text)
        controls_row.addWidget(self.reload_button)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self.text_box, stretch=1)

        self.setCentralWidget(central)

    def _load_text(self) -> None:
        self.text_box.setPlainText(
            self.constraints_path.read_text(encoding="utf-8")
        )

    def _copy_all_text(self) -> None:
        QApplication.clipboard().setText(self.text_box.toPlainText())
        self.statusBar().showMessage("Copied merged constraints to clipboard.")
