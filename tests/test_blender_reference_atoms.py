from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from saxshell.toolbox.blender.common import (
    ATOM_STYLE_LABELS,
    ATOM_STYLE_CHOICES,
    LIGHTING_LEVEL_LABELS,
    LIGHTING_LEVEL_CHOICES,
    OrientationSpec,
)
from saxshell.toolbox.blender.ui.main_window import OrientationPreviewWidget
from saxshell.toolbox.blender.ui.reference_atoms import (
    REFERENCE_ATOM_BACKGROUND,
    REFERENCE_ATOM_ELEMENTS,
    REFERENCE_ATOM_LABELS,
    iter_reference_atom_matrix,
    reference_atom_asset_dir,
    reference_atom_filename,
    reference_atom_key,
    reference_atom_path,
)
from saxshell.toolbox.blender.workflow import (
    BlenderPreviewStructure,
    PreviewAtomRecord,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_reference_atom_matrix_covers_all_styles_and_lighting_levels():
    matrix = iter_reference_atom_matrix()

    assert len(matrix) == (
        len(REFERENCE_ATOM_ELEMENTS)
        * len(ATOM_STYLE_CHOICES)
        * len(LIGHTING_LEVEL_CHOICES)
    )
    assert len(set(matrix)) == len(matrix)


def test_reference_atom_paths_are_stable_and_live_under_ui_assets():
    key = reference_atom_key("toon_matte", 4, "C")
    filename = reference_atom_filename("toon_matte", 4, "C")
    path = reference_atom_path("toon_matte", 4, "C")

    assert key == "c_reference_toon_matte_lighting_4"
    assert filename == "c_reference_toon_matte_lighting_4.png"
    assert path.parent == reference_atom_asset_dir()
    assert path.name == filename
    assert REFERENCE_ATOM_BACKGROUND == "#ffffff"


@pytest.mark.filterwarnings(
    "ignore:This figure includes Axes that are not compatible with tight_layout.*:UserWarning"
)
def test_orientation_preview_widget_loads_reference_atom_swatch(qapp):
    del qapp
    widget = OrientationPreviewWidget()
    widget.resize(960, 720)
    widget.show()

    structure = BlenderPreviewStructure(
        input_path=Path("reference_atoms.xyz"),
        structure_comment="Reference Atoms",
        atoms=(PreviewAtomRecord("S", (0.0, 0.0, 0.0)),),
        bonds=(),
    )
    orientation = OrientationSpec(
        key="reference_toon_matte_lighting_3",
        label="Reference",
        source="reference",
        x_degrees=0.0,
        y_degrees=0.0,
        z_degrees=0.0,
        enabled=True,
        atom_style="toon_matte",
        lighting_level=3,
    )

    try:
        widget.set_reference_background_color("#fff4cf")
        widget.set_preview(
            structure,
            orientation,
            row_index=0,
            row_count=1,
            atom_style="toon_matte",
        )
        QApplication.processEvents()

        dark_pixmap = widget.reference_dark_image_label.pixmap()
        light_pixmap = widget.reference_light_image_label.pixmap()
        assert reference_atom_path("toon_matte", 3, "C").is_file()
        assert reference_atom_path("toon_matte", 3, "S").is_file()
        assert widget.reference_style_label.text() == (
            f"Aesthetic: {ATOM_STYLE_LABELS['toon_matte']}"
        )
        assert widget.reference_lighting_label.text() == (
            f"Lighting: {LIGHTING_LEVEL_LABELS[3]}"
        )
        assert widget.reference_dark_title_label.text() == REFERENCE_ATOM_LABELS["C"]
        assert widget.reference_light_title_label.text() == REFERENCE_ATOM_LABELS["S"]
        assert widget.reference_background_color() == "#fff4cf"
        assert "#fff4cf" in widget.reference_dark_image_label.styleSheet()
        assert "#fff4cf" in widget.reference_light_image_label.styleSheet()
        assert widget.reference_status_label.text() == ""
        assert dark_pixmap is not None
        assert not dark_pixmap.isNull()
        assert light_pixmap is not None
        assert not light_pixmap.isNull()
    finally:
        widget.close()
