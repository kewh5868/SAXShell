from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
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


__all__ = ["SAXSProgressDialog"]
