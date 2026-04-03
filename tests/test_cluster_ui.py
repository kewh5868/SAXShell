import json
import os

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QTableWidgetItem

import saxshell.cluster.cli as cluster_cli_module
import saxshell.clusters as clusters_launcher
from saxshell import saxshell as saxshell_module
from saxshell.cluster import (
    DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME,
    DEFAULT_SAVE_STATE_FREQUENCY,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
)
from saxshell.cluster.ui.definitions_panel import ClusterDefinitionsPanel
from saxshell.cluster.ui.export_panel import ClusterExportPanel
from saxshell.cluster.ui.main_window import (
    ClusterExportResult,
    ClusterExportWorker,
    ClusterJobConfig,
    ClusterMainWindow,
    _format_tqdm_meter,
    estimate_selection,
    suggest_cluster_output_dir,
)
from saxshell.saxs.project_manager import SAXSProjectManager


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def cluster_preset_path(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SAXSHELL_CLUSTER_EXTRACTION_PRESETS_PATH",
        str(tmp_path / "cluster_extraction_presets.json"),
    )


def test_cluster_output_dir_suggestion_uses_source_directory(tmp_path):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitpdb0001"
    frames_dir.mkdir()

    suggested = suggest_cluster_output_dir(frames_dir)

    assert suggested == source_dir / "clusters_splitpdb0001"


def test_cluster_output_dir_suggestion_appends_unique_suffix(tmp_path):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitpdb0001"
    frames_dir.mkdir()
    (source_dir / "clusters_splitpdb0001").mkdir()

    suggested = suggest_cluster_output_dir(frames_dir)

    assert suggested == source_dir / "clusters_splitpdb00010001"


def test_definitions_panel_builds_atom_and_pair_rules(qapp):
    panel = ClusterDefinitionsPanel()
    panel.atom_table.setRowCount(0)
    panel.pair_table.setRowCount(0)

    panel._add_atom_row("node")
    panel._add_atom_row("linker")
    panel._add_atom_row("shell")
    atom_type_widget = panel.atom_table.cellWidget(0, 0)
    assert isinstance(atom_type_widget, QComboBox)
    atom_type_widget.setCurrentText("node")
    panel.atom_table.setItem(0, 1, QTableWidgetItem("Pb"))
    panel.atom_table.setItem(0, 2, QTableWidgetItem("SOL"))

    panel.atom_table.setItem(1, 1, QTableWidgetItem("I"))
    panel.atom_table.setItem(1, 2, QTableWidgetItem("SOL"))

    panel.atom_table.setItem(2, 1, QTableWidgetItem("O"))
    panel.atom_table.setItem(2, 2, QTableWidgetItem("WAT"))
    panel._sync_pair_element_choices()

    panel._add_pair_row()
    atom1_combo = panel.pair_table.cellWidget(0, 0)
    atom2_combo = panel.pair_table.cellWidget(0, 1)
    assert isinstance(atom1_combo, QComboBox)
    assert isinstance(atom2_combo, QComboBox)
    atom1_combo.setCurrentText("Pb")
    atom2_combo.setCurrentText("I")
    panel.pair_table.setItem(0, 2, QTableWidgetItem("1.7"))
    panel.pair_table.setItem(0, 3, QTableWidgetItem("1.9"))

    assert panel.atom_type_definitions() == {
        "node": [("Pb", "SOL")],
        "linker": [("I", "SOL")],
        "shell": [("O", "WAT")],
    }
    assert panel.pair_cutoff_definitions() == {("Pb", "I"): {0: 1.7, 1: 1.9}}


def test_definitions_panel_switches_to_xyz_mode(qapp):
    panel = ClusterDefinitionsPanel()

    panel.set_frame_mode("xyz")

    assert panel.title() == "Cluster Definitions (XYZ mode)"
    assert panel.atom_table.isColumnHidden(2)
    assert "element-only" in panel.mode_hint_label.text()
    assert not panel.use_pbc()
    assert panel.search_mode() == "kdtree"
    assert panel.save_state_frequency() == DEFAULT_SAVE_STATE_FREQUENCY
    assert not panel.include_shell_atoms_in_stoichiometry()


def test_definitions_panel_starts_blank_with_builtin_preset_available(qapp):
    panel = ClusterDefinitionsPanel()

    preset_names = [
        panel.preset_combo.itemText(index)
        for index in range(panel.preset_combo.count())
    ]

    assert panel.atom_type_definitions() == {}
    assert panel.pair_cutoff_definitions() == {}
    assert f"{DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME} (Built-in)" in preset_names

    panel.load_preset(DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME)

    assert panel.atom_type_definitions() == {
        "node": [("Pb", None)],
        "linker": [("I", None)],
        "shell": [("O", None)],
    }
    assert panel.pair_cutoff_definitions() == {
        ("Pb", "I"): {0: 3.36},
        ("Pb", "O"): {0: 3.36},
    }
    assert not panel.use_pbc()
    assert panel.search_mode() == "kdtree"
    assert panel.save_state_frequency() == DEFAULT_SAVE_STATE_FREQUENCY
    assert not panel.include_shell_atoms_in_stoichiometry()


def test_definitions_panel_saves_and_recalls_custom_presets_without_box(
    qapp,
    tmp_path,
):
    del qapp
    preset_file = tmp_path / "cluster_extraction_presets.json"
    panel = ClusterDefinitionsPanel()
    panel.load_preset(DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME)
    panel.set_use_pbc(True)
    panel.set_search_mode("vectorized")
    panel.set_save_state_frequency(250)
    panel.set_default_cutoff(4.25)
    panel.set_shell_growth_levels((1, 2))
    panel.set_shared_shells(True)
    panel.set_include_shell_atoms_in_stoichiometry(True)
    panel.set_box_dimensions((10.0, 11.0, 12.0))

    panel.save_current_preset("Custom solvent shell")

    assert preset_file.exists()
    payload = json.loads(preset_file.read_text())
    preset_payload = payload["presets"]["Custom solvent shell"]
    assert "box_dimensions" not in json.dumps(preset_payload)

    panel.set_box_dimensions((21.0, 22.0, 23.0), emit_signal=False)
    panel.load_atom_type_definitions({}, emit_signal=False)
    panel.load_pair_cutoff_definitions({}, emit_signal=False)
    panel.set_use_pbc(False, emit_signal=False)
    panel.set_search_mode("kdtree", emit_signal=False)
    panel.set_save_state_frequency(
        DEFAULT_SAVE_STATE_FREQUENCY,
        emit_signal=False,
    )
    panel.set_default_cutoff(None, emit_signal=False)
    panel.set_shell_growth_levels((), emit_signal=False)
    panel.set_shared_shells(False, emit_signal=False)
    panel.set_include_shell_atoms_in_stoichiometry(False, emit_signal=False)

    panel.load_preset("Custom solvent shell")

    assert panel.atom_type_definitions() == example_atom_type_definitions()
    assert panel.pair_cutoff_definitions() == example_pair_cutoff_definitions()
    assert panel.use_pbc()
    assert panel.search_mode() == "vectorized"
    assert panel.save_state_frequency() == 250
    assert panel.default_cutoff() == 4.25
    assert panel.shell_growth_levels() == (1, 2)
    assert panel.shared_shells()
    assert panel.include_shell_atoms_in_stoichiometry()
    assert panel.box_dimensions() == (21.0, 22.0, 23.0)


def test_cluster_main_window_preview_includes_output_details(
    qapp,
    tmp_path,
):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitpdb0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.pdb").write_text("MODEL        1\nENDMDL\n")

    window = ClusterMainWindow()
    window.trajectory_panel.frames_dir_edit.setText(str(frames_dir))
    window._last_summary = {
        "input_dir": str(frames_dir),
        "file_type": "pdb_frames",
        "frame_format": "pdb",
        "n_frames": 10,
        "first_frame": "frame_0000.pdb",
        "last_frame": "frame_0009.pdb",
        "estimated_box_dimensions": (10.0, 11.0, 12.0),
    }
    window._update_suggested_output_dir(frames_dir)

    preview = estimate_selection(
        total_frames=10,
        first_frame_name="frame_0000.pdb",
        last_frame_name="frame_0009.pdb",
    )
    text = window._format_selection_summary(
        preview,
        window.export_panel.get_output_dir(),
    )

    assert "Mode: PDB frames" in text
    assert "PBC: off" in text
    assert "Search mode: KDTree" in text
    assert (
        "Save-state frequency: every "
        f"{DEFAULT_SAVE_STATE_FREQUENCY} frames" in text
    )
    assert "Stoichiometry bins: solute only" in text
    assert "Frames selected: 10" in text
    assert "Estimated box dimensions: 10.000 x 11.000 x 12.000 A" in text
    assert "Output format: .pdb" in text
    assert "Frame file range: frame_0000.pdb to frame_0009.pdb" in text
    assert "Output folder: clusters_splitpdb0001" in text


def test_cluster_main_window_shows_compact_project_status_and_registers_paths(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text(
        "1\nframe\nPb 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    clusters_dir.mkdir()

    window = ClusterMainWindow(
        initial_frames_dir=frames_dir,
        initial_project_dir=project_dir,
    )
    updates = []
    window.project_paths_registered.connect(updates.append)
    message = window._register_project_paths(
        frames_dir=frames_dir,
        clusters_dir=clusters_dir,
    )
    saved_settings = manager.load_project(project_dir)

    assert window.project_banner is None
    assert window.project_status_label is not None
    assert project_dir.name in window.project_status_label.toolTip()
    assert str(project_dir) in window.project_status_label.full_text()
    assert window.project_status_label.parent() is window.statusBar()
    assert saved_settings.resolved_frames_dir == frames_dir.resolve()
    assert saved_settings.resolved_clusters_dir == clusters_dir.resolve()
    assert updates == [
        {
            "project_dir": project_dir.resolve(),
            "frames_dir": frames_dir.resolve(),
            "clusters_dir": clusters_dir.resolve(),
        }
    ]
    assert "Updated project folder references:" in (message or "")
    window.close()


def test_cluster_main_window_can_toggle_between_project_xyz_and_pdb_folders(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    xyz_frames_dir = tmp_path / "splitxyz0001"
    xyz_frames_dir.mkdir()
    (xyz_frames_dir / "frame_0000.xyz").write_text(
        "1\nframe\nPb 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    pdb_frames_dir = tmp_path / "xyz2pdb_splitxyz0001"
    pdb_frames_dir.mkdir()
    (pdb_frames_dir / "frame_0000.pdb").write_text(
        "MODEL        1\nENDMDL\n",
        encoding="utf-8",
    )
    settings.frames_dir = str(xyz_frames_dir)
    settings.pdb_frames_dir = str(pdb_frames_dir)
    manager.save_project(settings)

    window = ClusterMainWindow(initial_project_dir=project_dir)

    assert not window.trajectory_panel.project_source_widget.isHidden()
    assert window.trajectory_panel.project_source_combo.count() == 2
    assert window.trajectory_panel.get_frames_dir() == xyz_frames_dir

    window.trajectory_panel.project_source_combo.setCurrentIndex(1)

    assert window.trajectory_panel.selected_project_source_kind() == "pdb"
    assert window.trajectory_panel.get_frames_dir() == pdb_frames_dir
    window.close()


def test_cluster_main_window_switches_to_xyz_mode(qapp, tmp_path):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text(
        "2\nframe_0000\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\n"
    )

    window = ClusterMainWindow()
    window.trajectory_panel.frames_dir_edit.setText(str(frames_dir))

    assert window.trajectory_panel.mode_label.text() == "Mode: XYZ frames"
    assert window.definitions_panel.title() == "Cluster Definitions (XYZ mode)"
    assert window.definitions_panel.atom_table.isColumnHidden(2)
    assert window.export_panel.export_button.text().endswith("Cluster XYZs")
    assert "Estimated box dimensions:" in (
        window.export_panel.selection_box.toPlainText()
    )


def test_cluster_main_window_prefills_box_from_xyz_source_filename(
    qapp,
    tmp_path,
):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text(
        "2\nframe_0000\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\n"
    )
    (
        source_dir / "xyz_pbi2_dmso_1M_RT_den1p47_pbc_17p07x_041-pos-1.xyz"
    ).write_text("2\nsource\n")

    window = ClusterMainWindow()
    window.trajectory_panel.frames_dir_edit.setText(str(frames_dir))

    assert window.definitions_panel.box_dimensions() == (
        17.07,
        17.07,
        17.07,
    )
    preview_text = window.export_panel.selection_box.toPlainText()
    assert "Source box dimensions: 17.070 x 17.070 x 17.070 A" in preview_text
    assert (
        "Box source: "
        "xyz_pbi2_dmso_1M_RT_den1p47_pbc_17p07x_041-pos-1.xyz" in preview_text
    )


def test_cluster_main_window_uses_estimated_box_when_pbc_is_auto(
    qapp,
    tmp_path,
):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitpdb0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.pdb").write_text(
        "MODEL        1\n"
        "ATOM      1 PB1 SOL X   1       0.000   0.000   0.000"
        "  1.00  0.00          PB\n"
        "ATOM      2 I1  SOL X   1      10.000  12.000  14.000"
        "  1.00  0.00           I\n"
        "ENDMDL\n"
    )

    window = ClusterMainWindow()
    window.trajectory_panel.frames_dir_edit.setText(str(frames_dir))
    window.definitions_panel.use_pbc_box.setChecked(True)
    window._refresh_selection_preview()

    preview_text = window.export_panel.selection_box.toPlainText()

    assert "PBC: on" in preview_text
    assert (
        "Resolved box dimensions: 10.000 x 12.000 x 14.000 A (auto)"
        in preview_text
    )


def test_cluster_export_worker_emits_frame_progress(qapp, tmp_path):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text(
        "2\nframe_0000\nPb 0.0 0.0 0.0\nI 1.0 0.0 0.0\n"
    )
    (frames_dir / "frame_0001.xyz").write_text(
        "2\nframe_0001\nPb 0.0 0.0 0.0\nI 5.0 0.0 0.0\n"
    )

    config = ClusterJobConfig(
        frames_dir=frames_dir,
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
        box_dimensions=None,
        use_pbc=False,
        search_mode="kdtree",
        save_state_frequency=250,
        default_cutoff=None,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
        shared_shells=False,
        include_shell_atoms_in_stoichiometry=False,
        output_dir=source_dir / "clusters_splitxyz0001",
    )
    worker = ClusterExportWorker(config)
    counts = []
    messages = []
    phases = []
    results = []

    worker.progress_count.connect(
        lambda processed, total: counts.append((processed, total))
    )
    worker.progress.connect(messages.append)
    worker.phase_changed.connect(phases.append)
    worker.finished.connect(results.append)

    worker.run()

    assert counts[0] == (0, 2)
    assert counts[-1] == (2, 2)
    assert phases == ["extracting", "sorting"]
    assert results[0].analyzed_frames == 2
    assert any(
        message == "Save-state frequency: every 250 frame(s)."
        for message in messages
    )
    assert any(
        message == "Frame extraction complete. Sorting cluster files into "
        "stoichiometry folders..."
        for message in messages
    )
    assert any(
        message.startswith("Processed 1 of 2 frame(s); 1 remaining.")
        for message in messages
    )
    assert any(
        message.startswith("Sorted 1 of 2 frame(s); 1 remaining.")
        for message in messages
    )


def test_cluster_export_panel_append_log_preserves_manual_scroll(qapp):
    panel = ClusterExportPanel()
    panel.resize(420, 420)
    panel.show()
    qapp.processEvents()

    panel.set_log("\n".join(f"line {index}" for index in range(200)))
    qapp.processEvents()

    scroll_bar = panel.log_box.verticalScrollBar()
    scroll_bar.setValue(max(scroll_bar.maximum() // 2, 0))
    previous_value = scroll_bar.value()

    panel.append_log("new line")
    qapp.processEvents()

    assert scroll_bar.value() < scroll_bar.maximum()
    assert abs(scroll_bar.value() - previous_value) <= 2


def test_tqdm_meter_format_includes_counts():
    meter = _format_tqdm_meter(1, 4, 2.0)

    assert "1/4" in meter
    assert "25%" in meter
    assert "frame/s" in meter


def test_cluster_main_window_progress_popup_closes_on_completion(
    qapp,
    tmp_path,
):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()

    window = ClusterMainWindow()
    window._show_progress_dialog(total_frames=3)

    assert window._progress_dialog is not None
    assert window._progress_dialog.isVisible()

    result = ClusterExportResult(
        summary={
            "input_dir": str(frames_dir),
            "frame_format": "xyz",
            "output_file_extension": ".xyz",
            "n_frames": 3,
            "first_frame": "frame_0000.xyz",
            "last_frame": "frame_0002.xyz",
        },
        preview=estimate_selection(
            total_frames=3,
            first_frame_name="frame_0000.xyz",
            last_frame_name="frame_0002.xyz",
        ),
        written_files=[],
        analyzed_frames=3,
        total_clusters=2,
        output_dir=tmp_path / "clusters_splitxyz0001",
    )

    window._on_export_finished(result)

    assert not window._progress_dialog.isVisible()
    assert window.statusBar().currentMessage() == "Extraction Complete!"


def test_cluster_main_window_status_reflects_sorting_phase(qapp):
    window = ClusterMainWindow()
    window._show_progress_dialog(total_frames=4)

    window._on_export_phase_changed("sorting")
    window._on_export_progress(2, 4)

    assert window.statusBar().currentMessage() == (
        "Sorting cluster files... 2/4 frames"
    )
    assert window.export_panel.progress_label.text() == (
        "Sorting: 2 processed, 2 remaining"
    )
    assert window._progress_dialog is not None
    assert window._progress_dialog.phase_label.text() == (
        "Sorting cluster files into stoichiometry folders..."
    )


def test_saxshell_cli_forwards_to_cluster_subcommand(monkeypatch):
    captured = {}

    def fake_cluster_main(argv=None):
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(cluster_cli_module, "main", fake_cluster_main)

    exit_code = saxshell_module.main(["cluster", "--", "ui", "traj.pdb"])

    assert exit_code == 7
    assert captured["argv"] == ["ui", "traj.pdb"]


def test_clusters_launcher_forwards_to_cli_module(monkeypatch):
    captured = {}

    def fake_cluster_main(argv=None):
        captured["argv"] = argv
        return 9

    monkeypatch.setattr(clusters_launcher, "cluster_main", fake_cluster_main)

    exit_code = clusters_launcher.main(["ui", "traj.pdb"])

    assert exit_code == 9
    assert captured["argv"] == ["ui", "traj.pdb"]
