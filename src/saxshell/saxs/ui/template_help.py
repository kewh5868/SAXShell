from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget

TEMPLATE_HELP_TEXT = (
    "A SAXS template defines the model form used by the prefit and DREAM "
    "workflows. Each template file provides the SAXS model equations, the "
    "parameter list, and the callable model functions used by the app.\n\n"
    "To add a template manually, place a new Python file in "
    "`src/saxshell/saxs/_model_templates/` and follow the structure used by "
    "the existing `template_*.py` files in that folder.\n\n"
    "If you would like a new SAXS model added to the repository, open a "
    "GitHub issue and include the form factor, structure factor, parameter "
    "names, equations or literature reference, and any example data that "
    "helps define the model behavior.\n\n"
    "Contact: keith.white@colorado.edu"
)


def show_template_help(parent: QWidget | None) -> None:
    QMessageBox.information(
        parent,
        "SAXS Template Help",
        TEMPLATE_HELP_TEXT,
    )


__all__ = ["TEMPLATE_HELP_TEXT", "show_template_help"]
