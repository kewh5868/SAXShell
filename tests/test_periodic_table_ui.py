from __future__ import annotations

import os

from PySide6.QtWidgets import QApplication

from saxshell.ui.periodic_table import (
    PERIODIC_TABLE_ELEMENTS,
    PeriodicTableWidget,
    element_by_symbol,
)


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_periodic_table_widget_selects_element_symbol():
    qapp()
    widget = PeriodicTableWidget()
    selected: list[str] = []
    widget.element_selected.connect(selected.append)

    widget.select_element("cs")

    assert widget.selected_symbol() == "Cs"
    assert selected == ["Cs"]
    assert element_by_symbol("pb").name == "Lead"
    assert len(PERIODIC_TABLE_ELEMENTS) == 118
