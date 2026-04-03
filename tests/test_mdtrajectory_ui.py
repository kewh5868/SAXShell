import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtWidgets import QApplication

from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.manager import FrameSelectionPreview
from saxshell.mdtrajectory.ui.cutoff_panel import CutoffPanel
from saxshell.mdtrajectory.ui.export_panel import ExportPanel
from saxshell.mdtrajectory.ui.main_window import (
    ExportResult,
    MDTrajectoryMainWindow,
)
from saxshell.saxs.project_manager import SAXSProjectManager


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


def test_export_panel_post_cutoff_stride_controls_follow_cutoff_toggle(qapp):
    del qapp
    panel = ExportPanel()

    assert panel.use_cutoff()
    assert panel.post_cutoff_stride_box.isEnabled()
    assert not panel.post_cutoff_stride_spin.isEnabled()

    panel.use_cutoff_box.setChecked(False)

    assert not panel.post_cutoff_stride_box.isEnabled()
    assert not panel.post_cutoff_stride_spin.isEnabled()

    panel.use_cutoff_box.setChecked(True)
    panel.post_cutoff_stride_box.setChecked(True)
    panel.post_cutoff_stride_spin.setValue(3)

    assert panel.use_post_cutoff_stride()
    assert panel.get_post_cutoff_stride() == 3


def test_export_panel_progress_methods_update_ui(qapp):
    del qapp
    panel = ExportPanel()

    panel.set_busy_progress("Inspection progress: loading trajectory")

    assert panel.progress_bar.minimum() == 0
    assert panel.progress_bar.maximum() == 0
    assert panel.progress_label.text() == (
        "Inspection progress: loading trajectory"
    )

    panel.update_progress(2, 5, "Export progress: writing frames")

    assert panel.progress_bar.maximum() == 5
    assert panel.progress_bar.value() == 2
    assert panel.progress_label.text() == "Export progress: writing frames"

    panel.set_progress_complete("Export progress: complete.", total=5)

    assert panel.progress_bar.maximum() == 5
    assert panel.progress_bar.value() == 5
    assert panel.progress_label.text() == "Export progress: complete."


def test_cutoff_panel_load_energy_draws_target_temperature_line(
    qapp,
    tmp_path,
):
    del qapp
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


def test_cutoff_panel_uses_matplotlib_navigation_toolbar(
    qapp,
    tmp_path,
):
    del qapp
    panel = CutoffPanel()
    energy_data = CP2KEnergyData(
        filepath=Path(tmp_path / "sample.ener"),
        step=np.array([0.0, 1.0, 2.0, 3.0]),
        time_fs=np.array([0.0, 50.0, 100.0, 150.0]),
        kinetic=np.array([1.0, 1.1, 1.2, 1.3]),
        temperature=np.array([290.0, 300.0, 310.0, 320.0]),
        potential=np.array([-10.0, -10.2, -10.1, -10.4]),
    )

    panel.load_energy_data(energy_data)

    assert isinstance(panel.plot_toolbar, NavigationToolbar2QT)
    assert panel.canvas._cutoff_lines

    panel.canvas.set_cutoff(75.0)
    assert panel.canvas.cutoff_x_fs == pytest.approx(75.0)


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
        post_cutoff_stride=3,
        first_frame_index=2,
        last_frame_index=8,
        first_time_fs=50.0,
        last_time_fs=200.0,
        time_metadata_frames=10,
    )

    text = window._format_selection_summary(preview)

    assert "Output folder: selected_frames" in text
    assert f"Output path: {output_dir}" in text
    assert "Post-cutoff frame interval: 3" in text


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


def test_main_window_uses_first_selected_frame_and_time_in_output_name(
    qapp,
    tmp_path,
):
    del qapp
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    trajectory_file = source_dir / "traj.xyz"

    window = MDTrajectoryMainWindow()
    window.state.trajectory_file = trajectory_file
    window.export_panel.use_cutoff_box.setChecked(True)
    window.cutoff_panel.cutoff_spin.setValue(497.5)

    preview = FrameSelectionPreview(
        total_frames=1005,
        selected_frames=10,
        start=0,
        stop=None,
        stride=1,
        min_time_fs=497.5,
        post_cutoff_stride=1,
        first_frame_index=995,
        last_frame_index=1004,
        first_time_fs=497.5,
        last_time_fs=502.0,
        time_metadata_frames=1005,
    )

    class FakeManager:
        def preview_selection(self, **_kwargs):
            return preview

    window.manager = FakeManager()

    window._refresh_selection_preview()

    assert window.export_panel.get_output_dir() == (
        source_dir / "splitxyz_f995_t497p5fs"
    )


def test_mdtrajectory_window_shows_compact_project_status_and_inherits_file_refs(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    trajectory_file = tmp_path / "traj.xyz"
    topology_file = tmp_path / "topology.pdb"
    energy_file = tmp_path / "traj.ener"
    trajectory_file.write_text("1\nframe\nPb 0.0 0.0 0.0\n", encoding="utf-8")
    topology_file.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
    energy_file.write_text(
        "# step time kinetic temperature potential\n"
        "1 0.0 1.0 300.0 -10.0\n",
        encoding="utf-8",
    )
    settings.trajectory_file = str(trajectory_file)
    settings.topology_file = str(topology_file)
    settings.energy_file = str(energy_file)
    manager.save_project(settings)

    window = MDTrajectoryMainWindow(initial_project_dir=project_dir)

    assert window.project_banner is None
    assert window.project_status_label is not None
    assert project_dir.name in window.project_status_label.toolTip()
    assert str(project_dir) in window.project_status_label.full_text()
    assert window.project_status_label.parent() is window.statusBar()
    assert window.trajectory_panel.get_trajectory_path() == trajectory_file
    assert window.trajectory_panel.get_topology_path() == topology_file
    assert window.trajectory_panel.get_energy_path() == energy_file
    window.close()


def test_mdtrajectory_inspect_registers_project_file_references(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    trajectory_file = tmp_path / "traj.xyz"
    topology_file = tmp_path / "topology.pdb"
    energy_file = tmp_path / "traj.ener"
    trajectory_file.write_text("1\nframe\nPb 0.0 0.0 0.0\n", encoding="utf-8")
    topology_file.write_text("MODEL        1\nENDMDL\n", encoding="utf-8")
    energy_file.write_text(
        "# step time kinetic temperature potential\n"
        "1 0.0 1.0 300.0 -10.0\n",
        encoding="utf-8",
    )

    window = MDTrajectoryMainWindow(initial_project_dir=project_dir)
    window.trajectory_panel.trajectory_edit.setText(str(trajectory_file))
    window.trajectory_panel.topology_edit.setText(str(topology_file))
    window.trajectory_panel.energy_edit.setText(str(energy_file))

    monkeypatch.setattr(
        window,
        "_start_inspection_worker",
        lambda **_kwargs: None,
    )

    window.inspect_trajectory()

    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_trajectory_file == trajectory_file.resolve()
    assert saved_settings.resolved_topology_file == topology_file.resolve()
    assert saved_settings.resolved_energy_file == energy_file.resolve()
    assert saved_settings.trajectory_file_snapshot is not None
    assert saved_settings.topology_file_snapshot is not None
    assert saved_settings.energy_file_snapshot is not None
    window.close()


def test_mdtrajectory_export_registers_frames_dir_with_project(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    manager.create_project(project_dir)
    output_dir = tmp_path / "splitxyz_registered"

    class FakeManager:
        def preview_selection(self, **_kwargs):
            return FrameSelectionPreview(
                total_frames=2,
                selected_frames=2,
                start=0,
                stop=None,
                stride=1,
                min_time_fs=None,
                post_cutoff_stride=1,
                first_frame_index=0,
                last_frame_index=1,
                first_time_fs=0.0,
                last_time_fs=5.0,
                time_metadata_frames=2,
            )

    window = MDTrajectoryMainWindow(initial_project_dir=project_dir)
    window.manager = FakeManager()
    window.export_panel.output_dir_edit.setText(str(output_dir))
    updates = []
    window.project_paths_registered.connect(updates.append)

    def fake_start_export_worker(**kwargs):
        resolved = kwargs["output_dir"].expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        frame_path = resolved / "frame_0000.xyz"
        frame_path.write_text(
            "1\nframe\nPb 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        window._handle_export_finished(
            ExportResult(
                output_dir=resolved,
                written_files=[frame_path],
                preview=kwargs["preview"],
                applied_cutoff_fs=kwargs["min_time_fs"],
            )
        )

    monkeypatch.setattr(
        window, "_start_export_worker", fake_start_export_worker
    )

    window.export_frames()

    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_frames_dir == output_dir.resolve()
    assert saved_settings.frames_dir_snapshot is not None
    assert updates == [
        {
            "project_dir": project_dir.resolve(),
            "frames_dir": output_dir.resolve(),
        }
    ]
    assert "Registered the exported frames folder with project" in (
        window.export_panel.log_box.toPlainText()
    )
    window.close()
