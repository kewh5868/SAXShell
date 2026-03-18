from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb.ui.main_window import XYZToPDBMainWindow


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _write_reference_pdb(path: Path) -> None:
    structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="PB1",
                residue_name="PBI",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="Pb",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="I1",
                residue_name="PBI",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="I",
            ),
        ]
    )
    structure.write_pdb_file(path)


def _write_xyz(path: Path, *, i_x: float, oxygen_x: float) -> None:
    path.write_text(
        "3\n"
        f"{path.stem}\n"
        "Pb 0.0 0.0 0.0\n"
        f"I {i_x:.3f} 0.0 0.0\n"
        f"O {oxygen_x:.3f} 0.0 0.0\n"
    )


def _write_config(path: Path, *, reference_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "molecules": [
                    {
                        "name": "PBI",
                        "reference": reference_name,
                        "residue_name": "PBI",
                        "anchors": [{"pair": ["PB1", "I1"], "tol": 0.25}],
                    }
                ],
                "free_atoms": {"O": "SOL"},
            },
            indent=2,
        )
        + "\n"
    )


def test_reference_library_dropdown_populates_from_selected_folder(
    qapp,
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.input_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()

    assert window.reference_panel.reference_combo.count() == 1
    assert window.reference_panel.reference_combo.currentText() == "pbi"
    assert (
        "Residue name: PBI" in window.reference_panel.details_box.toPlainText()
    )


def test_main_window_creates_reference_and_suggests_output_dir(
    qapp,
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()

    source_xyz = tmp_path / "reference_source.xyz"
    _write_xyz(source_xyz, i_x=1.0, oxygen_x=2.0)
    input_xyz = tmp_path / "input.xyz"
    _write_xyz(input_xyz, i_x=1.1, oxygen_x=2.1)

    config_file = tmp_path / "assignments.json"
    _write_config(config_file, reference_name="pbi")

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.input_panel.library_dir_edit.setText(str(refs_dir))
    window.reference_panel.source_edit.setText(str(source_xyz))
    window.reference_panel.name_edit.setText("pbi")
    window.reference_panel.residue_edit.setText("PBI")

    window.create_reference_molecule()

    assert (refs_dir / "pbi.pdb").exists()
    assert window.reference_panel.reference_combo.currentText() == "pbi"

    window.input_panel.input_edit.setText(str(input_xyz))
    window.input_panel.config_edit.setText(str(config_file))
    window._suggest_output_dir_from_input(input_xyz)

    assert window.export_panel.get_output_dir() == (tmp_path / "xyz2pdb_input")
