from __future__ import annotations

import base64
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote_to_bytes
from xml.etree import ElementTree

from PySide6.QtCore import QCoreApplication, QEvent, QSize, Qt
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
BRAND_ICON_MAX_ASPECT_RATIO = 3.0
_SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_NON_DRAWING_SVG_TAGS = {
    f"{_SVG_NAMESPACE}defs",
    f"{_SVG_NAMESPACE}desc",
    f"{_SVG_NAMESPACE}metadata",
    f"{_SVG_NAMESPACE}style",
    f"{_SVG_NAMESPACE}title",
}


def saxshell_icon_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "_ui_assets"
        / ("saxshell_icon.svg")
    )


def _decode_data_uri_image(data_uri: str) -> bytes | None:
    metadata, separator, data = data_uri.partition(",")
    if separator == "":
        return None
    if ";base64" in metadata:
        try:
            return base64.b64decode("".join(data.split()))
        except ValueError:
            return None
    return unquote_to_bytes(data)


def _trim_transparent_padding(pixmap: QPixmap) -> QPixmap:
    image = pixmap.toImage()
    min_x = image.width()
    min_y = image.height()
    max_x = -1
    max_y = -1

    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() <= 0:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if max_x < min_x or max_y < min_y:
        return pixmap
    if (
        min_x == 0
        and min_y == 0
        and max_x == image.width() - 1
        and max_y == image.height() - 1
    ):
        return pixmap

    return pixmap.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def _load_embedded_svg_raster_pixmap(path: Path) -> QPixmap | None:
    try:
        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ElementTree.ParseError):
        return None

    image_element: ElementTree.Element | None = None
    for element in root.iter():
        if element is root:
            continue
        if element.tag == f"{_SVG_NAMESPACE}image":
            if image_element is not None:
                return None
            image_element = element
            continue
        if element.tag not in _NON_DRAWING_SVG_TAGS:
            return None

    if image_element is None:
        return None

    href = image_element.get("href") or image_element.get(_XLINK_HREF)
    if not href or not href.startswith("data:image/"):
        return None

    image_bytes = _decode_data_uri_image(href)
    if image_bytes is None:
        return None

    pixmap = QPixmap()
    if not pixmap.loadFromData(image_bytes):
        return None
    return _trim_transparent_padding(pixmap)


def _load_embedded_svg_raster_icon(path: Path) -> QIcon | None:
    pixmap = _load_embedded_svg_raster_pixmap(path)
    if pixmap is None:
        return None
    return QIcon(pixmap)


@lru_cache(maxsize=1)
def load_saxshell_icon() -> QIcon:
    path = saxshell_icon_path()
    if path.suffix.lower() == ".svg":
        # Qt's SVG handler warns on embedded raster data URIs even though it
        # still renders them, so decode that case ourselves first.
        embedded_icon = _load_embedded_svg_raster_icon(path)
        if embedded_icon is not None:
            return embedded_icon
    return QIcon(str(path))


def load_saxshell_brand_pixmap(target_height: int) -> QPixmap:
    target_height = max(1, int(target_height))
    path = saxshell_icon_path()
    if path.suffix.lower() == ".svg":
        embedded_pixmap = _load_embedded_svg_raster_pixmap(path)
        if embedded_pixmap is not None:
            return embedded_pixmap.scaledToHeight(
                target_height,
                Qt.TransformationMode.SmoothTransformation,
            )

    max_width = max(
        target_height,
        round(target_height * BRAND_ICON_MAX_ASPECT_RATIO),
    )
    return load_saxshell_icon().pixmap(QSize(max_width, target_height))


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
        icon_pixmap = load_saxshell_brand_pixmap(icon_size)
        icon_width = round(
            icon_pixmap.width() / icon_pixmap.devicePixelRatio()
        )
        icon_height = round(
            icon_pixmap.height() / icon_pixmap.devicePixelRatio()
        )
        self._icon_label.setPixmap(icon_pixmap)
        self._icon_label.setFixedSize(icon_width, icon_height)

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
