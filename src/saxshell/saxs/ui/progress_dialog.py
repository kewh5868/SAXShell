from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SAXSProgressDialog(QDialog):
    """Non-modal progress popup for long-running SAXS project tasks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXS Project Progress")
        self.setModal(False)
        self.resize(560, 120)

        layout = QVBoxLayout(self)
        self.message_label = QLabel("Preparing task...")
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m items")
        layout.addWidget(self.progress_bar)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(140)
        self.output_box.hide()
        layout.addWidget(self.output_box)

    def begin(
        self,
        total: int,
        message: str,
        *,
        unit_label: str = "items",
        title: str | None = None,
    ) -> None:
        total = max(int(total), 1)
        if title:
            self.setWindowTitle(title)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"%v / %m {unit_label}")
        self.message_label.setText(message)
        self.clear_output()
        self.show()
        self.raise_()

    def begin_busy(
        self,
        message: str,
        *,
        title: str | None = None,
    ) -> None:
        if title:
            self.setWindowTitle(title)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.message_label.setText(message)
        self.clear_output()
        self.show()
        self.raise_()

    def update_progress(
        self,
        processed: int,
        total: int,
        message: str,
        *,
        unit_label: str = "items",
    ) -> None:
        total = max(int(total), 1)
        processed = max(0, min(int(processed), total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"%v / %m {unit_label}")
        self.message_label.setText(message)

    def clear_output(self) -> None:
        self.output_box.clear()
        self.output_box.hide()
        self.resize(max(self.width(), 560), 120)

    def append_output(self, message: str) -> None:
        stripped = str(message).strip()
        if not stripped:
            return
        if self.output_box.toPlainText().strip():
            self.output_box.append(stripped)
        else:
            self.output_box.setPlainText(stripped)
        if not self.output_box.isVisible():
            self.output_box.show()
            self.resize(max(self.width(), 560), max(self.height(), 280))
        scrollbar = self.output_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


__all__ = ["SAXSProgressDialog"]
