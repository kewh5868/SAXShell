import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.manager import FrameSelectionPreview
from saxshell.mdtrajectory.ui.cutoff_panel import CutoffPanel
from saxshell.mdtrajectory.ui.export_panel import ExportPanel
from saxshell.mdtrajectory.ui.main_window import MDTrajectoryMainWindow


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_export_panel_suggest_output_dir_keeps_manual_override(
    qapp,
    tmp_path,
):
    panel = ExportPanel()
    suggested_dir = tmp_path / "trajectory_dir"
    manual_dir = tmp_path / "manual_dir"
    new_suggested_dir = tmp_path / "new_trajectory_dir"

    panel.suggest_output_dir(suggested_dir)
    assert panel.get_output_dir() == suggested_dir

    panel.output_dir_edit.setText(str(manual_dir))
    panel.suggest_output_dir(new_suggested_dir)

    assert panel.get_output_dir() == manual_dir


def test_cutoff_panel_load_energy_draws_target_temperature_line(
    qapp,
    tmp_path,
):
    panel = CutoffPanel()
    target_temp = 315.0
    panel.temp_target_spin.setValue(target_temp)

    energy_data = CP2KEnergyData(
        filepath=Path(tmp_path / "sample.ener"),
        step=np.array([0.0, 1.0, 2.0]),
        time_fs=np.array([0.0, 50.0, 100.0]),
        kinetic=np.array([1.0, 1.1, 1.2]),
        temperature=np.array([290.0, 300.0, 310.0]),
        potential=np.array([-10.0, -10.2, -10.1]),
    )

    panel.load_energy_data(energy_data)

    horizontal_lines = [
        line
        for line in panel.canvas.ax2.lines
        if all(abs(float(y) - target_temp) < 1.0e-12 for y in line.get_ydata())
    ]

    assert horizontal_lines
    assert panel.temp_target_spin.toolTip()
    assert panel.cutoff_spin.toolTip()


def test_main_window_reuses_loaded_trajectory_for_new_energy_file(
    qapp,
    tmp_path,
    monkeypatch,
):
    window = MDTrajectoryMainWindow()
    trajectory_file = tmp_path / "traj.xyz"
    energy_file = tmp_path / "traj.ener"
    dummy_manager = object()
    summary = {"n_frames": 5, "file_type": "xyz"}

    window.manager = dummy_manager
    window._last_summary = summary
    window.state.trajectory_file = trajectory_file
    window.state.topology_file = None
    window.state.energy_file = None

    window.trajectory_panel.trajectory_edit.setText(str(trajectory_file))
    window.trajectory_panel.energy_edit.setText(str(energy_file))

    captured = {}

    def fake_start_inspection_worker(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        window, "_start_inspection_worker", fake_start_inspection_worker
    )

    window.inspect_trajectory()

    assert captured["reload_trajectory"] is False
    assert captured["manager"] is dummy_manager
    assert captured["summary"] == summary


def test_selection_preview_includes_output_folder_details(qapp, tmp_path):
    window = MDTrajectoryMainWindow()
    output_dir = tmp_path / "selected_frames"
    window.export_panel.suggest_output_dir(output_dir)

    preview = FrameSelectionPreview(
        total_frames=10,
        selected_frames=4,
        start=0,
        stop=None,
        stride=2,
        min_time_fs=50.0,
        first_frame_index=2,
        last_frame_index=8,
        first_time_fs=50.0,
        last_time_fs=200.0,
        time_metadata_frames=10,
    )

    text = window._format_selection_summary(preview)

    assert "Output folder: selected_frames" in text
    assert f"Output path: {output_dir}" in text


def test_main_window_suggests_new_output_subfolder_in_source_dir(
    qapp,
    tmp_path,
):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    trajectory_file = source_dir / "traj.xyz"

    window = MDTrajectoryMainWindow()
    window.trajectory_panel.trajectory_edit.setText(str(trajectory_file))
    window._update_suggested_output_dir(trajectory_path=trajectory_file)

    assert window.export_panel.get_output_dir() == source_dir / "splitxyz"


def test_main_window_suggests_cutoff_timestamp_and_unique_suffix(
    qapp,
    tmp_path,
):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    trajectory_file = source_dir / "traj.xyz"
    existing_dir = source_dir / "splitxyz_t50fs"
    existing_dir.mkdir()

    window = MDTrajectoryMainWindow()
    window.trajectory_panel.trajectory_edit.setText(str(trajectory_file))
    window.export_panel.use_cutoff_box.setChecked(True)
    window.cutoff_panel.cutoff_spin.setValue(50.0)
    window._update_suggested_output_dir(trajectory_path=trajectory_file)

    assert window.export_panel.get_output_dir() == (
        source_dir / "splitxyz_t50fs0001"
    )
