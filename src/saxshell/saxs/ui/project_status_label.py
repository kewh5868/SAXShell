from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class CompactProjectStatusLabel(QLabel):
    """Single-line status-bar label that elides long project paths."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self.setMinimumWidth(220)
        self.setMaximumWidth(420)

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text).strip()
        self._update_display_text()

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display_text()

    def _update_display_text(self) -> None:
        if not self._full_text:
            super().setText("")
            return
        available_width = max(self.contentsRect().width(), 1)
        elided = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        super().setText(elided)


__all__ = ["CompactProjectStatusLabel"]
