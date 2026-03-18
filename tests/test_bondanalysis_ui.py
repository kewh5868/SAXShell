import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from saxshell.bondanalysis.ui.main_window import BondAnalysisMainWindow


def _write_xyz_cluster(path, atoms):
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_bondanalysis_main_window_prefills_cluster_types_and_output_dir(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)

    assert window.cluster_type_list.count() == 2
    assert not window.use_checked_cluster_types_box.isChecked()
    assert (
        window.cluster_type_list.item(0).checkState() == Qt.CheckState.Checked
    )
    assert (
        window.cluster_type_list.item(1).checkState() == Qt.CheckState.Checked
    )
    assert window.output_dir_edit.text().endswith(
        "bondanalysis_clusters_splitxyz0001"
    )
    preset_names = [
        window.preset_combo.itemText(index)
        for index in range(window.preset_combo.count())
    ]
    assert "DMSO (Built-in)" in preset_names
    assert "DMF (Built-in)" in preset_names
    preview_text = window.selection_box.toPlainText()
    assert "Cluster types detected: 2" in preview_text
    assert "Checked cluster types: 2" in preview_text
    assert "Displacement analysis: deprecated" in preview_text

    window.load_preset("DMSO")

    assert window.bond_pair_table.rowCount() == 7
    assert window.bond_pair_table.item(0, 0).text() == "Pb"
    assert window.bond_pair_table.item(0, 1).text() == "I"
    assert window.bond_pair_table.item(0, 2).text() == "4"
    assert window.angle_triplet_table.rowCount() == 5
    assert window.angle_triplet_table.item(2, 0).text() == "O"
    assert window.angle_triplet_table.item(2, 1).text() == "Pb"
    assert window.angle_triplet_table.item(2, 2).text() == "S"


def test_bondanalysis_main_window_saves_custom_presets_across_sessions(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    preset_file = tmp_path / "bondanalysis_presets.json"
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(preset_file),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir(parents=True)
    _write_xyz_cluster(
        clusters_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.load_preset("DMF")
    window.bond_pair_table.item(0, 2).setText("4.5")
    window.save_current_preset("DMF Custom")

    assert preset_file.exists()

    reloaded_window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    preset_names = [
        reloaded_window.preset_combo.itemText(index)
        for index in range(reloaded_window.preset_combo.count())
    ]
    assert "DMF Custom" in preset_names

    reloaded_window.load_preset("DMF Custom")

    assert reloaded_window.bond_pair_table.item(0, 2).text() == "4.5"


def test_bondanalysis_main_window_uses_checked_cluster_types_filter(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )

    window = BondAnalysisMainWindow(initial_clusters_dir=clusters_dir)
    window.use_checked_cluster_types_box.setChecked(True)
    window.cluster_type_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert window._selected_cluster_types() == ["PbI2"]
    preview_text = window.selection_box.toPlainText()
    assert "Checked cluster types: 1" in preview_text
    assert "Analyzing checked cluster types: PbI2" in preview_text
