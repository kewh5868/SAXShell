from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication

from saxshell.saxs.ui.branding import (
    load_saxshell_brand_pixmap,
    load_saxshell_icon,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_load_saxshell_icon_handles_embedded_raster_svg_without_qt_warning(
    qapp,
    capfd,
):
    del qapp
    load_saxshell_icon.cache_clear()

    icon = load_saxshell_icon()
    pixmap = icon.pixmap(64, 64)
    captured = capfd.readouterr()

    assert not icon.isNull()
    assert not pixmap.isNull()
    assert "Image filename is empty" not in captured.err


def test_load_saxshell_brand_pixmap_fills_requested_height(qapp):
    del qapp

    pixmap = load_saxshell_brand_pixmap(34)
    logical_width = pixmap.width() / pixmap.devicePixelRatio()
    logical_height = pixmap.height() / pixmap.devicePixelRatio()

    assert not pixmap.isNull()
    assert logical_height == pytest.approx(34.0)
    assert logical_width > logical_height
