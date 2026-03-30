from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplashScreen,
    QVBoxLayout,
    QWidget,
)

SAXSHELL_APPLICATION_NAME = "SAXSShell"
BRAND_PRIMARY_HEX = "#0f4aa6"
BRAND_SECONDARY_HEX = "#4f6074"
BRAND_ICON_MIN_SIZE = 34
BRAND_ICON_MAX_SIZE = 34
BRAND_TITLE_MAX_POINT_SIZE = 13.5


def saxshell_icon_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "_ui_assets"
        / ("saxshell_icon.svg")
    )


@lru_cache(maxsize=1)
def load_saxshell_icon() -> QIcon:
    return QIcon(str(saxshell_icon_path()))


class SAXShellBrandWidget(QWidget):
    """Top-left application branding that tracks UI font scaling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("saxshellBrandWidget")
        self.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Fixed,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(14, 2, 16, 2)
        self._layout.setSpacing(10)

        self._icon_label = QLabel(self)
        self._icon_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._layout.addWidget(
            self._icon_label,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(0)

        self._title_label = QLabel("SAXSShell", self)
        self._title_label.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Fixed,
        )
        self._title_label.setStyleSheet(f"color: {BRAND_PRIMARY_HEX};")
        text_column.addWidget(self._title_label)

        self._layout.addLayout(text_column)
        self._sync_brand_metrics()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
        ):
            self._sync_brand_metrics()

    def _sync_brand_metrics(self) -> None:
        base_font = QFont(self.font())
        base_point_size = base_font.pointSizeF()
        if base_point_size <= 0:
            app_font = QApplication.font(self)
            base_point_size = app_font.pointSizeF()
            base_font = QFont(app_font)
        if base_point_size <= 0:
            base_point_size = 10.0

        title_font = QFont(base_font)
        title_font.setBold(True)
        title_font.setPointSizeF(
            min(
                max(base_point_size * 1.18, base_point_size + 1.5),
                BRAND_TITLE_MAX_POINT_SIZE,
            )
        )
        self._title_label.setFont(title_font)

        icon_size = round(title_font.pointSizeF() * 2.35)
        icon_size = max(
            BRAND_ICON_MIN_SIZE, min(BRAND_ICON_MAX_SIZE, icon_size)
        )
        self._icon_label.setPixmap(
            load_saxshell_icon().pixmap(icon_size, icon_size)
        )
        self._icon_label.setFixedSize(icon_size, icon_size)

        layout_size = self._layout.sizeHint()
        self.setMinimumWidth(layout_size.width())
        self.setFixedHeight(layout_size.height())
        self.updateGeometry()


def build_saxshell_brand_widget(parent: QWidget | None = None) -> QWidget:
    return SAXShellBrandWidget(parent)


def _configure_macos_application_identity() -> None:
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle, NSProcessInfo
    except Exception:
        return

    NSProcessInfo.processInfo().setProcessName_(SAXSHELL_APPLICATION_NAME)
    info = NSBundle.mainBundle().infoDictionary()
    if info is not None:
        info["CFBundleName"] = SAXSHELL_APPLICATION_NAME
        info["CFBundleDisplayName"] = SAXSHELL_APPLICATION_NAME


def prepare_saxshell_application_identity() -> None:
    QCoreApplication.setApplicationName(SAXSHELL_APPLICATION_NAME)
    QApplication.setApplicationDisplayName(SAXSHELL_APPLICATION_NAME)
    QApplication.setDesktopFileName(SAXSHELL_APPLICATION_NAME)
    _configure_macos_application_identity()


def configure_saxshell_application(app: QApplication) -> None:
    prepare_saxshell_application_identity()
    app.setApplicationName(SAXSHELL_APPLICATION_NAME)
    app.setApplicationDisplayName(SAXSHELL_APPLICATION_NAME)
    app.setDesktopFileName(SAXSHELL_APPLICATION_NAME)
    app.setWindowIcon(load_saxshell_icon())


def create_saxshell_startup_splash() -> QSplashScreen:
    pixmap = QPixmap(420, 210)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#f7f9fc"))
    painter.drawRoundedRect(10, 10, 400, 190, 18, 18)

    border_pen = QPen(QColor("#d7e1f0"))
    border_pen.setWidth(2)
    painter.setPen(border_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(10, 10, 400, 190, 18, 18)

    icon_pixmap = load_saxshell_icon().pixmap(92, 92)
    painter.drawPixmap(26, 48, icon_pixmap)

    title_font = QFont()
    title_font.setBold(True)
    title_font.setPointSize(19)
    painter.setFont(title_font)
    painter.setPen(QColor(BRAND_PRIMARY_HEX))
    painter.drawText(136, 84, "SAXSShell")

    subtitle_font = QFont()
    subtitle_font.setPointSize(10)
    painter.setFont(subtitle_font)
    painter.setPen(QColor(BRAND_SECONDARY_HEX))
    painter.drawText(138, 113, "Loading SAXS workflow...")
    painter.drawText(138, 136, "Initializing interface and project state")
    painter.end()

    splash = QSplashScreen(
        pixmap,
        Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint,
    )
    splash.setWindowIcon(load_saxshell_icon())
    splash.showMessage(
        "Starting SAXSShell",
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
        QColor(BRAND_PRIMARY_HEX),
    )
    return splash
