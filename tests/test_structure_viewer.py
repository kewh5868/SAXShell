from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from saxshell.saxs.structure_viewer.ui.main_window import (
    StructureViewerMainWindow,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_xyz(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_structure_viewer_loads_single_structure(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "methane.xyz",
        [
            "5",
            "methane",
            "C 0.0 0.0 0.0",
            "H 0.8 0.8 0.8",
            "H -0.8 -0.8 0.8",
            "H -0.8 0.8 -0.8",
            "H 0.8 -0.8 -0.8",
        ],
    )

    window = StructureViewerMainWindow(initial_input_path=structure_path)

    assert window.input_mode_value.text() == "Single structure file"
    assert window.reference_file_value.text() == str(structure_path.resolve())
    assert "5 atoms" in window.structure_summary_value.text()
    assert "Previewing methane" in window.viewer_status_label.text()
    assert window.structure_viewer.current_structure is not None
    assert (
        window.structure_viewer.current_structure.file_path
        == structure_path.resolve()
    )
    assert window.structure_viewer.current_mesh_geometry is not None
    assert "shells=" in window.active_mesh_value.text()
    window.close()


def test_structure_viewer_center_updates_preserve_viewer_display(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "snap_preserve.xyz",
        [
            "4",
            "snap preserve",
            "C 0.0 0.0 0.0",
            "H 1.1 0.2 0.0",
            "H -0.2 0.9 0.3",
            "O 2.3 0.0 0.0",
        ],
    )
    window = StructureViewerMainWindow(initial_input_path=structure_path)
    viewer = window.structure_viewer

    viewer.atom_contrast_spin.lineEdit().setText("30")
    viewer.mesh_contrast_spin.lineEdit().setText("45")
    viewer.mesh_linewidth_spin.lineEdit().setText("2.70")
    viewer.atom_contrast_spin.interpretText()
    viewer.mesh_contrast_spin.interpretText()
    viewer.mesh_linewidth_spin.interpretText()
    viewer.point_atoms_checkbox.setChecked(True)

    monkeypatch.setattr(
        "saxshell.saxs.structure_viewer.ui.widget.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#ff6600"),
    )
    viewer.mesh_color_button.click()

    viewer._view_radius = 7.5
    viewer._view_center = np.asarray([0.2, -0.4, 0.6], dtype=float)
    viewer._scene_rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    previous_centered = np.asarray(
        viewer.current_structure.centered_coordinates,
        dtype=float,
    ).copy()

    window.snap_center_button.click()

    assert viewer._atom_contrast == pytest.approx(0.30)
    assert viewer._mesh_contrast == pytest.approx(0.45)
    assert viewer._mesh_linewidth == pytest.approx(2.7)
    assert viewer._mesh_color == "#ff6600"
    assert viewer._atom_render_mode == "points"
    assert viewer._view_radius == pytest.approx(7.5)
    assert np.allclose(viewer._view_center, [0.2, -0.4, 0.6])
    assert np.allclose(
        viewer._scene_rotation,
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert viewer.current_structure.center_mode == "nearest_atom"
    assert not np.allclose(
        viewer.current_structure.centered_coordinates,
        previous_centered,
    )
    overlay_texts = [
        text
        for text in viewer._axis.texts
        if getattr(text, "get_gid", lambda: None)() == "active-visual-settings"
    ]
    assert len(overlay_texts) == 1
    overlay_text = overlay_texts[0].get_text()
    assert f"ZOOM {viewer._current_zoom_percentage():05.1f}%" in overlay_text
    assert "ATOM 030.0%" in overlay_text
    assert "MESH 045.0%" in overlay_text
    assert "LINE 2.70px" in overlay_text
    assert "#FF6600" in overlay_text
    window.close()


def test_structure_viewer_mesh_notice_tracks_pending_fields(qapp, tmp_path):
    del qapp
    structure_path = _write_xyz(
        tmp_path / "mesh_update.xyz",
        [
            "3",
            "mesh update",
            "O -1.16 0.0 0.0",
            "C 0.0 0.0 0.0",
            "O 1.16 0.0 0.0",
        ],
    )
    window = StructureViewerMainWindow(initial_input_path=structure_path)

    window.rstep_spin.setValue(0.2)

    assert "differ from the rendered mesh" in window.pending_mesh_value.text()

    window.update_mesh_button.click()

    assert window.structure_viewer.current_mesh_geometry is not None
    assert (
        window.structure_viewer.current_mesh_geometry.settings.rstep
        == pytest.approx(0.2)
    )
    assert "match the rendered mesh" in window.pending_mesh_value.text()
    assert "rstep=0.200 A" in window.active_mesh_value.text()
    window.close()
