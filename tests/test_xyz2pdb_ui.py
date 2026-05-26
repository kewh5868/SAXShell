from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMessageBox,
    QScrollArea,
    QSplitter,
)

import saxshell.xyz2pdb.ui.batch_queue_window as batch_queue_module
from saxshell.saxs.project_manager import SAXSProjectManager
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
)
from saxshell.xyz2pdb.ui.batch_queue_window import (
    XYZToPDBBatchItem,
    XYZToPDBBatchQueueWindow,
    XYZToPDBBatchResult,
    XYZToPDBBatchWorker,
)
from saxshell.xyz2pdb.ui.main_window import XYZToPDBMainWindow
from saxshell.xyz2pdb.ui.run_file_window import XYZToPDBRunFileWindow
from saxshell.xyz2pdb.workflow import (
    XYZToPDBAssertionResidueSummary,
    XYZToPDBAssertionResult,
    XYZToPDBExportResult,
    XYZToPDBReferenceUpdateCandidate,
)


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


def _write_ocn_source_pdb(path: Path) -> None:
    PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="C1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="N1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[2.4, 0.0, 0.0],
                element="N",
            ),
        ]
    ).write_pdb_file(path)


def test_reference_library_dropdown_populates_from_selected_folder(
    qapp,
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()

    assert window.reference_panel.reference_combo.count() == 1
    assert window.reference_panel.reference_combo.currentText() == "pbi"
    assert (
        "Residue name: PBI" in window.reference_panel.details_box.toPlainText()
    )
    assert window.mapping_panel.molecule_residue_edit.text() == "PBI"
    assert window.mapping_panel.max_missing_h_spin.value() == 0
    assert window.mapping_panel.tight_scale_spin.suffix().strip() == "%"
    assert window.mapping_panel.relaxed_scale_spin.suffix().strip() == "%"
    assert window.mapping_panel.tight_scale_spin.value() == pytest.approx(85.0)
    assert window.mapping_panel.relaxed_scale_spin.value() == pytest.approx(
        135.0
    )
    assert "bond-table percent tolerances" in (
        window.mapping_panel.relaxed_scale_spin.toolTip()
    )
    assert "avoid assuming deprotonation" in (
        window.mapping_panel.max_missing_h_spin.toolTip()
    )
    header = window.mapping_panel.molecule_table.horizontalHeader()
    assert (
        header.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    )
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    bond_header = window.mapping_panel.bond_table.horizontalHeader()
    assert (
        bond_header.sectionResizeMode(0)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    assert (
        bond_header.sectionResizeMode(1)
        == QHeaderView.ResizeMode.ResizeToContents
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(2).text() == (
        "Ref (A)"
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(3).text() == (
        "Tolerance (%)"
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(4).text() == (
        "Tight Min (A)"
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(5).text() == (
        "Tight Max (A)"
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(6).text() == (
        "Relaxed Min (A)"
    )
    assert window.mapping_panel.bond_table.horizontalHeaderItem(7).text() == (
        "Relaxed Max (A)"
    )
    assert window.mapping_panel.bond_table.item(0, 2).text() == "1.000"
    assert window.mapping_panel.bond_table.item(0, 3).text() == "18.00"
    assert window.mapping_panel.bond_table.item(0, 4).text() == "0.847"
    assert window.mapping_panel.bond_table.item(0, 5).text() == "1.153"
    assert window.mapping_panel.bond_table.item(0, 6).text() == "0.757"
    assert window.mapping_panel.bond_table.item(0, 7).text() == "1.243"

    window.mapping_panel.tight_scale_spin.setValue(100.0)
    window.mapping_panel.relaxed_scale_spin.setValue(200.0)
    assert window.mapping_panel.bond_table.item(0, 4).text() == "0.820"
    assert window.mapping_panel.bond_table.item(0, 5).text() == "1.180"
    assert window.mapping_panel.bond_table.item(0, 6).text() == "0.640"
    assert window.mapping_panel.bond_table.item(0, 7).text() == "1.360"
    window.close()


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

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.reference_panel.source_edit.setText(str(source_xyz))
    window.reference_panel.name_edit.setText("pbi")
    window.reference_panel.residue_edit.setText("PBI")
    assert not window.reference_panel.backbone_atom1_combo.isEnabled()
    assert not window.reference_panel.backbone_atom2_combo.isEnabled()
    assert (
        "XYZ source detected"
        in window.reference_panel.backbone_help_label.text()
    )

    window.create_reference_molecule()

    assert (refs_dir / "pbi.pdb").exists()
    assert (refs_dir / "pbi.json").exists()
    assert window.reference_panel.reference_combo.currentText() == "pbi"
    details_text = window.reference_panel.details_box.toPlainText()
    assert "Backbone pairs:" in details_text
    assert "PB1-I1" in details_text

    window.input_panel.input_edit.setText(str(input_xyz))
    window._suggest_output_dir_from_input(input_xyz)

    assert window.export_panel.get_output_dir() == (tmp_path / "xyz2pdb_input")
    window.close()


def test_main_window_creates_reference_with_selected_pdb_backbone_pair(
    qapp,
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()

    source_pdb = tmp_path / "ocn_source.pdb"
    _write_ocn_source_pdb(source_pdb)

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.reference_panel.source_edit.setText(str(source_pdb))
    window.reference_panel.name_edit.setText("ocn")
    window.reference_panel.residue_edit.setText("OCN")

    atom_names = [
        window.reference_panel.backbone_atom1_combo.itemText(index)
        for index in range(window.reference_panel.backbone_atom1_combo.count())
    ]
    assert atom_names == ["Auto", "O1", "C1", "N1"]
    assert window.reference_panel.backbone_atom1_combo.isEnabled()
    assert window.reference_panel.backbone_atom2_combo.isEnabled()
    assert (
        "PDB source detected"
        in window.reference_panel.backbone_help_label.text()
    )

    window.reference_panel.backbone_atom1_combo.setCurrentText("O1")
    window.reference_panel.backbone_atom2_combo.setCurrentText("N1")
    window.create_reference_molecule()

    metadata = json.loads((refs_dir / "ocn.json").read_text(encoding="utf-8"))
    assert metadata["backbone_pairs"] == [["O1", "N1"]]
    assert window.reference_panel.reference_combo.currentText() == "ocn"
    assert "O1-N1" in window.reference_panel.details_box.toPlainText()
    window.close()


def test_xyz2pdb_export_registers_pdb_folder_with_project(qapp, tmp_path):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    output_dir = tmp_path / "xyz2pdb_splitxyz"
    output_dir.mkdir()

    window = XYZToPDBMainWindow(initial_project_dir=project_dir)
    updates = []
    window.project_paths_registered.connect(updates.append)

    preview = type("Preview", (), {"output_dir": output_dir})()
    result = XYZToPDBExportResult(
        output_dir=output_dir,
        written_files=(output_dir / "frame_0000.pdb",),
        preview=preview,
    )

    window._on_export_finished(result)

    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_pdb_frames_dir == output_dir.resolve()
    assert saved_settings.pdb_frames_dir_snapshot is not None
    assert updates == [
        {
            "project_dir": project_dir.resolve(),
            "pdb_frames_dir": output_dir.resolve(),
        }
    ]
    assert "Registered the converted PDB structure folder with project" in (
        window.export_panel.log_box.toPlainText()
    )
    window.close()


def test_xyz2pdb_batch_queue_prefills_project_xyz_reference_and_elements(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    frames_dir = tmp_path / "splitxyz_f0fs"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)
    settings.frames_dir = str(frames_dir)
    manager.save_project(settings)

    window = XYZToPDBBatchQueueWindow(
        initial_project_dir=project_dir,
        reference_library_dir=refs_dir,
    )

    assert window.queue_list.count() == 1
    list_item = window.queue_list.item(0)
    item_id = str(list_item.data(Qt.ItemDataRole.UserRole))
    widget = window._widgets_by_id[item_id]
    assert widget.input_path_edit.text() == str(frames_dir.resolve())
    assert {
        widget.free_element_combo.itemData(index)
        for index in range(widget.free_element_combo.count())
    } == {"I", "O", "Pb"}
    assert {
        widget.reference_combo.itemData(index)
        for index in range(widget.reference_combo.count())
    } == {"pbi"}
    assert widget.molecule_residue_edit.text() == "PBI"
    assert widget.tight_scale_spin.value() == pytest.approx(85.0)
    assert widget.relaxed_scale_spin.value() == pytest.approx(135.0)
    assert widget.max_missing_h_spin.value() == 0
    assert widget.molecule_table.horizontalHeaderItem(2).text() == "Bonds"
    assert widget.bond_table.horizontalHeaderItem(2).text() == "Ref (A)"
    assert widget.bond_table.horizontalHeaderItem(3).text() == "Tolerance (%)"
    assert widget.bond_table.item(0, 2).text() == "1.000"
    assert widget.bond_table.item(0, 3).text() == "18.00"
    assert widget.bond_table.item(0, 4).text() == "0.847"
    assert widget.bond_table.item(0, 5).text() == "1.153"
    assert widget.bond_table.item(0, 6).text() == "0.757"
    assert widget.bond_table.item(0, 7).text() == "1.243"
    assert "Elements: I x1, O x1, Pb x1" in (
        widget.analysis_summary_label.text()
    )
    assert "Target PDB folder: xyz2pdb_splitxyz_f0fs" in (
        widget.analysis_summary_label.text()
    )
    assert str(tmp_path / "xyz2pdb_splitxyz_f0fs") in (
        widget.analysis_summary_label.text()
    )
    window.close()


def test_xyz2pdb_batch_queue_reference_selection_updates_residue(
    qapp,
    tmp_path,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    _write_ocn_source_pdb(refs_dir / "ocn.pdb")
    window = XYZToPDBBatchQueueWindow(reference_library_dir=refs_dir)
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="batch-item",
            reference_library_dir=refs_dir,
        )
    )

    widget.reference_combo.setCurrentIndex(
        widget.reference_combo.findData("pbi")
    )
    assert widget.molecule_residue_edit.text() == "PBI"

    widget.molecule_residue_edit.setText("ALT")
    widget.reference_combo.setCurrentIndex(
        widget.reference_combo.findData("ocn")
    )

    assert widget.molecule_residue_edit.text() == "OCN"
    assert widget.bond_table.rowCount() == 2
    assert widget.bond_table.item(0, 2).text() == "1.200"
    assert widget.angle_table.rowCount() == 1
    assert widget.angle_table.item(0, 1).text() == "C1"
    assert widget.angle_table.item(0, 3).text() == "180.000"
    window.close()


def test_xyz2pdb_batch_queue_preserves_reference_bond_tolerances(
    qapp,
    tmp_path,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    window = XYZToPDBBatchQueueWindow(reference_library_dir=refs_dir)
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="batch-item",
            reference_library_dir=refs_dir,
        )
    )

    widget.bond_table.item(0, 3).setText("20.00")
    widget.tight_scale_spin.setValue(100.0)
    widget.relaxed_scale_spin.setValue(200.0)
    widget.max_missing_h_spin.setValue(1)
    widget._add_molecule()
    molecule = widget.item().molecule_inputs[0]

    assert widget.molecule_table.item(0, 2).text() == "1"
    assert widget.molecule_table.item(0, 3).text() == "100.0%"
    assert widget.molecule_table.item(0, 4).text() == "200.0%"
    assert widget.molecule_table.item(0, 5).text() == "1"
    assert molecule.reference_name == "pbi"
    assert molecule.residue_name == "PBI"
    assert molecule.bond_tolerances[0].tolerance == pytest.approx(20.0)
    assert molecule.tight_pass_scale == pytest.approx(1.0)
    assert molecule.relaxed_pass_scale == pytest.approx(2.0)
    assert molecule.max_missing_hydrogens == 1
    window.close()


def test_xyz2pdb_batch_queue_shows_progress_dialog_for_multiple_project_load(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()
    window = XYZToPDBBatchQueueWindow()
    monkeypatch.setattr(
        batch_queue_module,
        "_choose_existing_directories",
        lambda *_args, **_kwargs: (project_a.resolve(), project_b.resolve()),
    )
    dialogs: list[object] = []

    class FakeProgressDialog:
        def __init__(self, label, cancel_text, minimum, maximum, parent):
            del cancel_text, parent
            self.labels = [label]
            self.values: list[int] = []
            self.minimum = minimum
            self.maximum = maximum
            self.closed = False
            self.shown = False
            dialogs.append(self)

        def setWindowTitle(self, title):
            self.title = title

        def setWindowModality(self, modality):
            self.modality = modality

        def setMinimumDuration(self, duration):
            self.minimum_duration = duration

        def setAutoClose(self, enabled):
            self.auto_close = enabled

        def setAutoReset(self, enabled):
            self.auto_reset = enabled

        def setValue(self, value):
            self.values.append(value)

        def show(self):
            self.shown = True

        def setLabelText(self, label):
            self.labels.append(label)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        batch_queue_module,
        "QProgressDialog",
        FakeProgressDialog,
    )

    window._choose_projects_to_add()

    assert len(dialogs) == 1
    dialog = dialogs[0]
    assert dialog.maximum == 2
    assert dialog.shown is True
    assert dialog.closed is True
    assert dialog.values[-1] == 2
    assert any("project_a" in label for label in dialog.labels)
    assert any("project_b" in label for label in dialog.labels)
    assert window.queue_list.count() == 2
    window.close()


def test_xyz2pdb_batch_worker_exports_and_registers_pdb_folder(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    frames_dir = tmp_path / "splitxyz_f0fs"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)
    settings.frames_dir = str(frames_dir)
    manager.save_project(settings)

    worker = XYZToPDBBatchWorker(
        [
            (
                "job-1",
                XYZToPDBBatchItem(
                    item_id="job-1",
                    project_dir=project_dir,
                    input_path=frames_dir,
                    reference_library_dir=refs_dir,
                    molecule_inputs=(
                        MoleculeMappingInput(
                            reference_name="pbi",
                            residue_name="PBI",
                        ),
                    ),
                    free_atom_inputs=(
                        FreeAtomMappingInput(
                            element="O",
                            residue_name="SOL",
                        ),
                    ),
                ).to_job(),
            )
        ]
    )
    failures: list[tuple[str, str]] = []
    finished_items: list[tuple[str, XYZToPDBBatchResult]] = []
    finished_batches: list[list[XYZToPDBBatchResult]] = []
    logs: list[str] = []
    progress_messages: list[tuple[str, int, int, str]] = []
    worker.failed.connect(
        lambda item_id, message: failures.append((item_id, message))
    )
    worker.item_finished.connect(
        lambda item_id, result: finished_items.append((item_id, result))
    )
    worker.finished.connect(finished_batches.append)
    worker.log.connect(logs.append)
    worker.item_progress.connect(
        lambda item_id, processed, total, message: progress_messages.append(
            (item_id, processed, total, message)
        )
    )

    worker.run()

    assert failures == []
    assert len(finished_items) == 1
    item_id, result = finished_items[0]
    assert item_id == "job-1"
    assert result.output_dir.name == "xyz2pdb_splitxyz_f0fs"
    assert result.written_count == 1
    assert (result.output_dir / "frame_0000.pdb").is_file()
    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_frames_dir == frames_dir.resolve()
    assert saved_settings.resolved_pdb_frames_dir == result.output_dir
    assert saved_settings.pdb_frames_dir_snapshot is not None
    assert finished_batches == [[result]]
    assert any(
        "Target PDB folder: xyz2pdb_splitxyz_f0fs" in message
        for message in logs
    )
    assert any("Registered PDB frames folder" in message for message in logs)
    assert progress_messages[0] == (
        "job-1",
        0,
        1,
        "Target PDB folder: xyz2pdb_splitxyz_f0fs",
    )


def test_xyz2pdb_batch_queue_emits_registered_pdb_folder(qapp, tmp_path):
    del qapp
    project_dir = tmp_path / "saxs_project"
    SAXSProjectManager().create_project(project_dir)
    output_dir = tmp_path / "xyz2pdb_splitxyz_f0fs"
    output_dir.mkdir()
    input_dir = tmp_path / "splitxyz_f0fs"
    input_dir.mkdir()

    window = XYZToPDBBatchQueueWindow()
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="job-1",
            project_dir=project_dir,
            input_path=input_dir,
        )
    )
    updates = []
    window.project_paths_registered.connect(updates.append)

    window._on_item_finished(
        widget.item_id,
        XYZToPDBBatchResult(
            project_dir=project_dir.resolve(),
            input_path=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            written_count=1,
        ),
    )

    assert updates == [
        {
            "project_dir": project_dir.resolve(),
            "pdb_frames_dir": output_dir.resolve(),
        }
    ]
    assert widget.status_label.text() == "Complete"
    window.close()


def test_xyz2pdb_batch_queue_logs_active_project_progress(qapp, tmp_path):
    del qapp
    project_dir = tmp_path / "saxs_project"
    SAXSProjectManager().create_project(project_dir)
    input_dir = tmp_path / "splitxyz_f0fs"
    input_dir.mkdir()

    window = XYZToPDBBatchQueueWindow()
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="job-1",
            project_dir=project_dir,
            input_path=input_dir,
        )
    )

    window._on_item_progress(
        widget.item_id,
        1,
        3,
        "Mapping template from frame_0000.xyz...",
    )

    assert widget.status_label.text() == (
        "Mapping template from frame_0000.xyz..."
    )
    assert widget.progress_bar.maximum() == 3
    assert widget.progress_bar.value() == 1
    assert window.queue_status_label.text() == (
        "[saxs_project] 1/3: Mapping template from frame_0000.xyz..."
    )
    assert "Mapping template from frame_0000.xyz..." in (
        window.console.toPlainText()
    )
    assert window.statusBar().currentMessage() == (
        "Mapping template from frame_0000.xyz..."
    )
    window.close()


def test_xyz2pdb_batch_queue_throttles_large_progress_logs(qapp, tmp_path):
    del qapp
    project_dir = tmp_path / "saxs_project"
    SAXSProjectManager().create_project(project_dir)
    input_dir = tmp_path / "splitxyz_f0fs"
    input_dir.mkdir()

    window = XYZToPDBBatchQueueWindow()
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="job-1",
            project_dir=project_dir,
            input_path=input_dir,
        )
    )

    for index in range(1, 101):
        window._on_item_progress(
            widget.item_id,
            index,
            100,
            f"[{index}/100] Wrote frame_{index:04d}.xyz",
        )

    text = window.console.toPlainText()
    assert window.console.document().maximumBlockCount() == (
        batch_queue_module._BATCH_CONSOLE_MAX_BLOCKS
    )
    assert "[saxs_project] 1/100" in text
    assert "[saxs_project] 2/100" not in text
    assert "[saxs_project] 5/100" in text
    assert "[saxs_project] 100/100" in text
    assert window.queue_status_label.text() == (
        "[saxs_project] 100/100: [100/100] Wrote frame_0100.xyz"
    )
    assert widget.progress_bar.value() == 100
    window.close()


def test_xyz2pdb_batch_queue_console_log_is_bounded(qapp):
    window = XYZToPDBBatchQueueWindow()
    window.show()
    qapp.processEvents()

    for index in range(batch_queue_module._BATCH_CONSOLE_MAX_BLOCKS + 25):
        window._append_log(f"line {index}")
    qapp.processEvents()

    assert window.console.document().blockCount() == (
        batch_queue_module._BATCH_CONSOLE_MAX_BLOCKS
    )
    text = window.console.toPlainText()
    assert "line 0" not in text
    assert f"line {batch_queue_module._BATCH_CONSOLE_MAX_BLOCKS + 24}" in text
    window.close()


def test_xyz2pdb_batch_worker_filters_dense_progress_logs():
    assert batch_queue_module._should_emit_batch_log_message(
        "[1/100] frame_0001.xyz"
    )
    assert not batch_queue_module._should_emit_batch_log_message(
        "[2/100] frame_0002.xyz"
    )
    assert batch_queue_module._should_emit_batch_log_message(
        "[5/100] frame_0005.xyz"
    )
    assert batch_queue_module._should_emit_batch_log_message(
        "Template mapping counts: PBI 1"
    )


def test_xyz2pdb_run_file_window_builds_project_config(qapp, tmp_path):
    del qapp
    project_dir = tmp_path / "project"
    SAXSProjectManager().create_project(project_dir)

    refs_dir = project_dir / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    frames_dir = project_dir / "frames"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)

    window = XYZToPDBRunFileWindow(
        initial_project_dir=project_dir,
        initial_input_path=frames_dir,
    )
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()
    window.mapping_panel._molecule_inputs = [
        MoleculeMappingInput(reference_name="pbi", residue_name="PBI")
    ]
    window.mapping_panel._refresh_molecule_table()
    window.mapping_panel.set_available_elements(("O", "Pb", "I"))
    window.mapping_panel.free_element_combo.setCurrentText("O")
    window.mapping_panel.free_residue_edit.setText("SOL")
    window.mapping_panel._add_free_atom()
    window.output_dir_edit.setText(str(project_dir / "pdb_frames"))
    window.pbc_params_edit.setPlainText('{"a": 20.0, "b": 21.0, "c": 22.0}')

    config = window._current_config(project_dir)

    assert config.input_path == "frames"
    assert config.output_dir == "pdb_frames"
    assert config.reference_library_dir == "references"
    assert config.molecule_inputs[0].reference_name == "pbi"
    assert config.free_atom_inputs[0].element == "O"
    assert config.pbc_params["a"] == 20.0
    assert "xyz2pdb run" in window.command_box.toPlainText()
    window.close()


def test_main_window_native_mapping_flow_estimates_and_reports_export_progress_without_json(
    qapp,
    tmp_path,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    splitxyz_dir = tmp_path / "splitxyz_f0fs"
    splitxyz_dir.mkdir()
    input_xyz = splitxyz_dir / "frame_0000.xyz"
    _write_xyz(input_xyz, i_x=1.0, oxygen_x=2.0)

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.input_panel.input_edit.setText(str(input_xyz))

    window.inspect_input()
    assert "Sample atoms: 3" in window.input_panel.summary_box.toPlainText()
    assert window.mapping_panel.free_element_combo.count() >= 1

    window.mapping_panel.free_element_combo.setCurrentText("O")
    window.mapping_panel.free_residue_edit.setText("SOL")
    window.mapping_panel._add_free_atom()

    window.mapping_panel.reference_combo.setCurrentText("pbi")
    window.mapping_panel.molecule_residue_edit.setText("PBI")
    window.mapping_panel._add_molecule()

    window.estimate_mapping()
    assert (
        "Estimated molecules: PBI x1"
        in window.export_panel.preview_box.toPlainText()
    )
    assert window.export_panel.solution_combo.count() == 1

    assert not hasattr(window.export_panel, "test_button")
    assert (
        window.export_panel.export_button.text() == "Convert XYZ Frames to PDB"
    )
    assert window.export_panel.cancel_button.text() == "Cancel Mapping"
    assert window.export_panel.cancel_button.isEnabled() is False
    assert window.export_panel.assertion_mode_enabled() is False
    assert "per-molecule PDB files" in (
        window.export_panel.assertion_mode_checkbox.toolTip()
    )
    assert "distance distribution" in (
        window.export_panel.assertion_mode_checkbox.toolTip()
    )

    window._handle_export_progress(
        1,
        3,
        "Mapping template from input.xyz...",
    )
    assert window.export_panel.progress_bar.maximum() == 3
    assert window.export_panel.progress_bar.value() == 1
    assert window.export_panel.progress_bar.format() == "%v / %m steps"
    assert (
        window.export_panel.progress_label.text()
        == "Mapping template from input.xyz..."
    )
    assert "Mapping template from input.xyz..." in (
        window.export_panel.log_box.toPlainText()
    )
    assert (
        window.statusBar().currentMessage()
        == "Mapping template from input.xyz..."
    )
    window._set_export_running(True)
    assert window.export_panel.export_button.isEnabled() is False
    assert window.export_panel.cancel_button.isEnabled() is True
    window.close()


def test_xyz2pdb_analyze_input_warns_for_raw_or_unsplit_xyz(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    input_xyz = tmp_path / "calc-pos-1.xyz"
    _write_xyz(input_xyz, i_x=1.0, oxygen_x=2.0)
    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.input_panel.input_edit.setText(str(input_xyz))
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    window.inspect_input()

    assert len(warnings) == 1
    assert warnings[0][0] == "Check XYZ input"
    assert "*pos-1.xyz" in warnings[0][1]
    assert "folder named splitxyz*" in warnings[0][1]
    assert "Sample atoms: 3" in window.input_panel.summary_box.toPlainText()
    window.close()


def test_xyz2pdb_batch_analyze_input_warns_for_unsplit_xyz(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    input_xyz = tmp_path / "frame_0000.xyz"
    _write_xyz(input_xyz, i_x=1.0, oxygen_x=2.0)
    window = XYZToPDBBatchQueueWindow(reference_library_dir=tmp_path)
    widget = window.add_queue_item(
        XYZToPDBBatchItem(
            item_id="batch-item",
            input_path=input_xyz,
            reference_library_dir=tmp_path,
        )
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    widget._analyze_from_button()

    assert len(warnings) == 1
    assert warnings[0][0] == "Check XYZ input"
    assert "folder named splitxyz*" in warnings[0][1]
    assert "Target PDB folder" in widget.analysis_summary_label.text()
    window.close()


def test_main_window_cancel_export_requests_stop_without_closing(
    qapp,
    tmp_path,
):
    del qapp

    class FakeThread:
        def __init__(self) -> None:
            self.request_interruption_called = False
            self._running = True

        def isRunning(self) -> bool:
            return self._running

        def requestInterruption(self) -> None:
            self.request_interruption_called = True

    class FakeWorker:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    window = XYZToPDBMainWindow(reference_library_dir=tmp_path)
    fake_thread = FakeThread()
    fake_worker = FakeWorker()
    window._export_thread = fake_thread
    window._export_worker = fake_worker
    window._set_export_running(True)

    window.cancel_export()

    assert fake_worker.cancel_called is True
    assert fake_thread.request_interruption_called is True
    assert window.export_panel.cancel_button.isEnabled() is False
    assert window.export_panel.progress_label.text() == (
        "Canceling current conversion..."
    )
    assert (
        "Cancellation requested." in window.export_panel.log_box.toPlainText()
    )
    assert window.isVisible() is False


def test_xyz2pdb_window_closes_and_cancels_active_export(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp

    class FakeThread:
        def __init__(self) -> None:
            self.quit_called = False
            self.request_interruption_called = False
            self.terminate_called = False
            self.wait_calls: list[int] = []
            self._running = True

        def isRunning(self) -> bool:
            return self._running

        def requestInterruption(self) -> None:
            self.request_interruption_called = True

        def quit(self) -> None:
            self.quit_called = True

        def wait(self, timeout: int | None = None) -> bool:
            self.wait_calls.append(0 if timeout is None else int(timeout))
            if len(self.wait_calls) >= 2:
                self._running = False
                return True
            return False

        def terminate(self) -> None:
            self.terminate_called = True
            self._running = False

        def deleteLater(self) -> None:
            return

    class FakeWorker:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

        def deleteLater(self) -> None:
            return

    warnings: list[str] = []

    def fake_warning(parent, title, message):
        del parent, title
        warnings.append(message)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    window = XYZToPDBMainWindow(reference_library_dir=tmp_path)
    fake_thread = FakeThread()
    fake_worker = FakeWorker()
    window._export_thread = fake_thread
    window._export_worker = fake_worker
    window.show()
    QApplication.processEvents()

    assert window.close()
    assert warnings == []
    assert fake_worker.cancel_called is True
    assert fake_thread.request_interruption_called is True
    assert fake_thread.quit_called is True
    assert fake_thread.terminate_called is True
    assert fake_thread.wait_calls == [25, 25]


def test_main_window_reports_assertion_mode_results_after_export(
    qapp,
    tmp_path,
):
    del qapp
    output_dir = tmp_path / "xyz2pdb_output"
    output_dir.mkdir()
    report_file = output_dir / "assertion_molecules" / "assertion_report.txt"
    report_file.parent.mkdir(parents=True)
    report_file.write_text("assertion report\n", encoding="utf-8")

    window = XYZToPDBMainWindow()
    assertion_result = XYZToPDBAssertionResult(
        molecule_dir=report_file.parent,
        report_file=report_file,
        total_molecules=2,
        passed=True,
        residue_summaries=(
            XYZToPDBAssertionResidueSummary(
                residue_name="PBI",
                molecule_count=2,
                common_atom_count=2,
                distance_pair_count=1,
                median_distribution_rmsd=0.0,
                max_distribution_rmsd=0.0,
                median_max_distance_delta=0.0,
                max_max_distance_delta=0.0,
                outlier_count=0,
                passed=True,
            ),
        ),
        warnings=(),
    )
    preview = type("Preview", (), {"output_dir": output_dir})()
    result = XYZToPDBExportResult(
        output_dir=output_dir,
        written_files=(output_dir / "frame_0000.pdb",),
        preview=preview,
        progress_total_steps=5,
        assertion_result=assertion_result,
    )

    window._on_export_finished(result)

    assert window.export_panel.progress_bar.maximum() == 5
    assert window.export_panel.progress_bar.value() == 5
    assert "Assertion mode passed. Molecule folder:" in (
        window.export_panel.log_box.toPlainText()
    )
    assert f"Assertion report: {report_file}" in (
        window.export_panel.log_box.toPlainText()
    )
    window.close()


def test_main_window_assertion_candidates_can_save_new_version_one_at_a_time(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")
    _write_reference_pdb(refs_dir / "dmf.pdb")

    candidate_dir = tmp_path / "assertion_candidates"
    candidate_dir.mkdir()
    averaged_pbi = candidate_dir / "pbi_average.pdb"
    averaged_dmf = candidate_dir / "dmf_average.pdb"
    _write_reference_pdb(averaged_pbi)
    _write_reference_pdb(averaged_dmf)

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()

    prompted_names: list[str] = []
    decisions = iter(("save_new_version", "skip"))

    monkeypatch.setattr(
        window,
        "_prompt_reference_update_candidate",
        lambda candidate, *, versioned_reference_name: (
            prompted_names.append(
                f"{candidate.reference_name}:{versioned_reference_name}"
            )
            or next(decisions)
        ),
    )
    monkeypatch.setattr(
        window,
        "_versioned_reference_name",
        lambda base_name: f"{base_name}_20260401_120000",
    )

    window._offer_assertion_reference_updates(
        (
            XYZToPDBReferenceUpdateCandidate(
                residue_name="PBI",
                reference_name="pbi",
                reference_path=refs_dir / "pbi.pdb",
                reference_residue_name="PBI",
                average_structure_file=averaged_pbi,
                molecule_count=2,
                median_distribution_rmsd=0.0,
                max_distribution_rmsd=0.0,
                backbone_pairs=(("PB1", "I1"),),
            ),
            XYZToPDBReferenceUpdateCandidate(
                residue_name="DMF",
                reference_name="dmf",
                reference_path=refs_dir / "dmf.pdb",
                reference_residue_name="PBI",
                average_structure_file=averaged_dmf,
                molecule_count=2,
                median_distribution_rmsd=0.0,
                max_distribution_rmsd=0.0,
                backbone_pairs=(("PB1", "I1"),),
            ),
        )
    )

    assert prompted_names == [
        "pbi:pbi_20260401_120000",
        "dmf:dmf_20260401_120000",
    ]
    assert (refs_dir / "pbi_20260401_120000.pdb").exists()
    assert (refs_dir / "pbi_20260401_120000.json").exists()
    assert not (refs_dir / "dmf_20260401_120000.pdb").exists()
    assert (
        "Saved assertion-derived reference version pbi_20260401_120000."
        in (window.export_panel.log_box.toPlainText())
    )
    assert "Skipped assertion-derived reference update for dmf." in (
        window.export_panel.log_box.toPlainText()
    )
    window.close()


def test_main_window_assertion_candidates_can_replace_existing_reference(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    averaged_pbi = tmp_path / "pbi_average.pdb"
    PDBStructure(
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
                coordinates=[1.2, 0.0, 0.0],
                element="I",
            ),
        ]
    ).write_pdb_file(averaged_pbi)

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()

    monkeypatch.setattr(
        window,
        "_prompt_reference_update_candidate",
        lambda candidate, *, versioned_reference_name: "replace_existing",
    )
    monkeypatch.setattr(
        window,
        "_versioned_reference_name",
        lambda base_name: f"{base_name}_20260401_120000",
    )

    window._offer_assertion_reference_updates(
        (
            XYZToPDBReferenceUpdateCandidate(
                residue_name="PBI",
                reference_name="pbi",
                reference_path=refs_dir / "pbi.pdb",
                reference_residue_name="PBI",
                average_structure_file=averaged_pbi,
                molecule_count=2,
                median_distribution_rmsd=0.0,
                max_distribution_rmsd=0.0,
                backbone_pairs=(("PB1", "I1"),),
            ),
        )
    )

    replaced_structure = PDBStructure.from_file(refs_dir / "pbi.pdb")
    bond_distance = float(
        np.linalg.norm(
            replaced_structure.atoms[1].coordinates
            - replaced_structure.atoms[0].coordinates
        )
    )
    assert bond_distance == pytest.approx(1.2, abs=1.0e-6)
    assert "Updated reference pbi with assertion-averaged coordinates." in (
        window.export_panel.log_box.toPlainText()
    )
    window.close()


def test_xyz2pdb_residue_fields_enforce_three_capital_letters(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    source_xyz = tmp_path / "reference_source.xyz"
    _write_reference_pdb(refs_dir / "pbi.pdb")
    _write_xyz(source_xyz, i_x=1.0, oxygen_x=2.0)

    window = XYZToPDBMainWindow(reference_library_dir=refs_dir)
    window.reference_panel.library_dir_edit.setText(str(refs_dir))
    window.refresh_reference_library()

    window.mapping_panel.free_residue_edit.setText("sol")
    window.mapping_panel.molecule_residue_edit.setText("pbi")
    window.reference_panel.residue_edit.setText("dmso")

    assert window.mapping_panel.free_residue_edit.text() == "SOL"
    assert window.mapping_panel.molecule_residue_edit.text() == "PBI"
    assert window.reference_panel.residue_edit.text() == "DMS"

    warnings: list[str] = []
    window.mapping_panel._warn = warnings.append
    window.mapping_panel.set_available_elements(("O",))
    window.mapping_panel.free_element_combo.setCurrentText("O")
    window.mapping_panel.free_residue_edit.setText("so")
    window.mapping_panel._add_free_atom()
    assert warnings[-1] == (
        "Free-atom residues must be exactly three capital letters."
    )
    assert window.mapping_panel.free_atom_table.rowCount() == 0

    window.mapping_panel.reference_combo.setCurrentText("pbi")
    window.mapping_panel.molecule_residue_edit.setText("pb")
    window.mapping_panel._add_molecule()
    assert warnings[-1] == (
        "Reference-molecule residues must be exactly three capital letters."
    )
    assert window.mapping_panel.molecule_table.rowCount() == 0

    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        window,
        "_show_error",
        lambda title, message: errors.append((title, message)),
    )
    window.reference_panel.source_edit.setText(str(source_xyz))
    window.reference_panel.name_edit.setText("dmso_test")
    window.reference_panel.residue_edit.setText("dm")
    window.create_reference_molecule()
    assert errors == [
        (
            "Create reference failed",
            "Reference residues must be exactly three capital letters.",
        )
    ]
    window.close()


def test_xyz2pdb_ui_uses_resizable_left_panes_and_removes_legacy_json(
    qapp,
    tmp_path,
):
    del tmp_path
    window = XYZToPDBMainWindow()
    window.show()
    qapp.processEvents()

    assert isinstance(window._main_splitter, QSplitter)
    assert window._main_splitter.orientation() == Qt.Orientation.Horizontal
    assert window._main_splitter.count() == 2
    assert window._main_splitter.widget(0) is window._left_splitter
    assert window._main_splitter.widget(1) is window._export_scroll_area
    assert isinstance(window._export_scroll_area, QScrollArea)

    assert isinstance(window._left_splitter, QSplitter)
    assert window._left_splitter.orientation() == Qt.Orientation.Vertical
    assert window._left_splitter.count() == 3
    assert window._left_splitter.widget(0) is window._input_scroll_area
    assert window._left_splitter.widget(1) is window._reference_scroll_area
    assert window._left_splitter.widget(2) is window._mapping_scroll_area
    assert all(size > 0 for size in window._left_splitter.sizes())

    assert not hasattr(window.input_panel, "config_edit")
    assert hasattr(window.reference_panel, "library_dir_edit")
    assert window.reference_panel.get_library_dir() is not None
    assert window.input_panel.title() == "XYZ Input"

    window.close()
