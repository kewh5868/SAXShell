from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter, QWidget


class PaneSnapFilter(QObject):
    """Global event filter that focuses a horizontal QSplitter on
    whichever pane the user clicks in.

    Install one instance per splitter. Call ``set_enabled(True)`` to activate
    and ``set_enabled(False)`` to restore normal splitter behaviour. The filter
    returns False for every event so it never swallows normal interactions.
    """

    def __init__(
        self,
        splitter: QSplitter,
        left_widget: QWidget,
        right_widget: QWidget,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._splitter = splitter
        self._left_widget = left_widget
        self._right_widget = right_widget
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        app = QApplication.instance()
        if app is None:
            return
        if enabled:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)

    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # QObject interface
    # ------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            not self._enabled
            or self._splitter.orientation() != Qt.Orientation.Horizontal
            or not self._splitter.isVisible()
        ):
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if self._is_descendant(watched, self._left_widget):
                self._snap_to(0)
            elif self._is_descendant(watched, self._right_widget):
                self._snap_to(1)
        return False  # never consume events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_descendant(widget: QObject, ancestor: QWidget) -> bool:
        w: QObject | None = widget
        while w is not None:
            if w is ancestor:
                return True
            w = w.parent()
        return False

    def _snap_to(self, index: int) -> None:
        if self._splitter.count() < 2:
            return
        sizes = self._splitter.sizes()
        if len(sizes) < 2:
            return
        handle_w = self._splitter.handleWidth()
        total = self._splitter.width() - handle_w
        if total <= 0:
            return
        left_w = self._left_widget
        right_w = self._right_widget
        left_min = self._minimum_width(left_w)
        right_min = self._minimum_width(right_w)
        left_max = self._clamp_width(total - right_min, left_min, total)
        right_max = self._clamp_width(total - left_min, right_min, total)
        if index == 0:
            right = self._unfocused_width(
                widget=right_w,
                current_width=sizes[1],
                minimum_width=right_min,
                maximum_width=right_max,
            )
            left = self._clamp_width(total - right, left_min, left_max)
            self._splitter.setSizes(
                [left, self._clamp_width(total - left, right_min, right_max)]
            )
        else:
            left = self._unfocused_width(
                widget=left_w,
                current_width=sizes[0],
                minimum_width=left_min,
                maximum_width=left_max,
            )
            right = self._clamp_width(total - left, right_min, right_max)
            self._splitter.setSizes(
                [self._clamp_width(total - right, left_min, left_max), right]
            )

    def _unfocused_width(
        self,
        *,
        widget: QWidget,
        current_width: int,
        minimum_width: int,
        maximum_width: int,
    ) -> int:
        preferred_width = self._clamp_width(
            self._preferred_width(widget),
            minimum_width,
            maximum_width,
        )
        current_width = self._clamp_width(
            current_width,
            minimum_width,
            maximum_width,
        )
        # A pane click should only shrink the opposite pane, never expand it.
        return min(current_width, preferred_width)

    @staticmethod
    def _clamp_width(
        width: int, minimum_width: int, maximum_width: int
    ) -> int:
        maximum_width = max(int(maximum_width), int(minimum_width))
        return max(int(minimum_width), min(int(width), maximum_width))

    @staticmethod
    def _minimum_width(widget: QWidget) -> int:
        return max(widget.minimumSizeHint().width(), widget.minimumWidth(), 1)

    @staticmethod
    def _preferred_width(widget: QWidget) -> int:
        if isinstance(widget, QScrollArea):
            inner = widget.widget()
            if inner is not None:
                w = inner.sizeHint().width()
                # account for the vertical scrollbar and frame border
                sb = widget.verticalScrollBar()
                if sb is not None:
                    w += sb.sizeHint().width()
                w += widget.frameWidth() * 2
                return max(w, widget.minimumSizeHint().width())
        hint = widget.sizeHint().width()
        return hint if hint > 0 else widget.width()
