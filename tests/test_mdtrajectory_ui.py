import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import saxshell.mdtrajectory.ui.batch_queue_window as md_batch_queue_module
from saxshell.mdtrajectory.frame.cp2k_ener import CP2KEnergyData
from saxshell.mdtrajectory.frame.manager import (
    DEFAULT_FRAME_TIMESTEP_FS,
    FrameSelectionPreview,
    TrajectoryManager,
)
from saxshell.mdtrajectory.ui.batch_queue_window import (
    DEFAULT_TIME_CUTOFF_FS,
    MDTrajectoryBatchJob,
    MDTrajectoryBatchQueueWindow,
    MDTrajectoryBatchResult,
    MDTrajectoryBatchWorker,
)
from saxshell.mdtrajectory.ui.cutoff_panel import CutoffPanel
from saxshell.mdtrajectory.ui.export_panel import ExportPanel
from saxshell.mdtrajectory.ui.main_window import (
    ExportResult,
    InspectionResult,
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


def _write_batch_xyz(path: Path) -> None:
    path.write_text(
        "1\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "Pb 0.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 500.0, E = -1.0\n"
        "Pb 0.1 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1500.0, E = -1.0\n"
        "Pb 0.2 0.0 0.0\n",
        encoding="utf-8",
    )


def _write_uniform_timestep_xyz(
    path: Path,
    *,
    timestep_fs: float,
    n_frames: int = 4,
) -> None:
    lines: list[str] = []
    for frame_index in range(n_frames):
        lines.extend(
            [
                "1\n",
                (
                    "i = "
                    f"{frame_index}, time = "
                    f"{frame_index * timestep_fs:.3f}, E = -1.0\n"
                ),
                f"Pb {frame_index:.1f} 0.0 0.0\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_batch_restart_overlap_xyz(path: Path) -> None:
    path.write_text(
        "1\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "Pb 0.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 500.0, E = -1.0\n"
        "Pb 0.1 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1500.0, E = -1.0\n"
        "Pb 0.2 0.0 0.0\n"
        "1\n"
        "i = 1, time = 500.0, E = -1.0\n"
        "Pb 9.1 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1500.0, E = -1.0\n"
        "Pb 9.2 0.0 0.0\n"
        "1\n"
        "i = 3, time = 2000.0, E = -1.0\n"
        "Pb 0.3 0.0 0.0\n",
        encoding="utf-8",
    )


def _write_batch_energy(path: Path) -> None:
    path.write_text(
        "# step time kinetic temperature potential\n"
        "1 0.0 1.0 300.0 -10.0\n"
        "2 500.0 1.0 301.0 -10.0\n"
        "3 1500.0 1.0 300.5 -10.0\n",
        encoding="utf-8",
    )


def _create_mdtrajectory_batch_project(
    tmp_path: Path,
    name: str,
) -> tuple[Path, Path, Path]:
    manager = SAXSProjectManager()
    project_dir = tmp_path / name
    settings = manager.create_project(project_dir)
    trajectory_file = project_dir / "traj.xyz"
    energy_file = project_dir / "traj.ener"
    _write_batch_xyz(trajectory_file)
    _write_batch_energy(energy_file)
    settings.trajectory_file = str(trajectory_file)
    settings.energy_file = str(energy_file)
    manager.save_project(settings)
    return project_dir, trajectory_file, energy_file


def _create_mdtrajectory_overlap_batch_project(
    tmp_path: Path,
    name: str,
) -> tuple[Path, Path, Path]:
    project_dir, trajectory_file, energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, name)
    )
    _write_batch_restart_overlap_xyz(trajectory_file)
    return project_dir, trajectory_file, energy_file


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


def test_export_panel_restart_duplicate_option_defaults_off(qapp):
    del qapp
    panel = ExportPanel()

    assert not panel.include_restart_duplicates()

    panel.include_restart_duplicates_box.setChecked(True)

    assert panel.include_restart_duplicates()


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
    assert panel.window_spin.value() == 2


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


def test_mdtrajectory_batch_queue_prefills_current_project_defaults(
    qapp,
    tmp_path,
):
    del qapp
    project_dir, trajectory_file, energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, "project_a")
    )

    window = MDTrajectoryBatchQueueWindow(initial_project_dir=project_dir)

    assert window.queue_list.count() == 1
    widget = next(iter(window._widgets_by_id.values()))
    assert widget.project_dir_edit.text() == str(project_dir.resolve())
    assert widget.trajectory_file_edit.text() == str(trajectory_file.resolve())
    assert widget.energy_file_edit.text() == str(energy_file.resolve())
    assert widget.output_dir_edit.text() == ""
    assert widget.cutoff_spin.value() == pytest.approx(DEFAULT_TIME_CUTOFF_FS)
    assert widget.timestep_spin.value() == pytest.approx(
        DEFAULT_FRAME_TIMESTEP_FS
    )
    assert widget.auto_timestep_box.isChecked()
    assert not widget.include_restart_duplicates_box.isChecked()
    window.close()


def test_mdtrajectory_window_updates_timestep_field_after_inspection(
    qapp,
    tmp_path,
):
    del qapp
    trajectory_file = tmp_path / "traj.xyz"
    _write_uniform_timestep_xyz(trajectory_file, timestep_fs=25.0)
    window = MDTrajectoryMainWindow()
    manager = TrajectoryManager(
        input_file=trajectory_file,
        frame_timestep_fs=DEFAULT_FRAME_TIMESTEP_FS,
    )
    summary = manager.inspect()
    window.state.trajectory_file = trajectory_file

    window._handle_inspection_metadata(
        InspectionResult(
            manager=manager,
            summary=summary,
            energy_data=None,
        )
    )

    assert window.trajectory_panel.auto_timestep_box.isChecked()
    assert window.trajectory_panel.timestep_spin.value() == pytest.approx(25.0)
    window.close()


def test_mdtrajectory_batch_preview_updates_timestep_field_from_metadata(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir = tmp_path / "project_timestep"
    settings = manager.create_project(project_dir)
    trajectory_file = project_dir / "traj.xyz"
    energy_file = project_dir / "traj.ener"
    _write_uniform_timestep_xyz(trajectory_file, timestep_fs=25.0)
    _write_batch_energy(energy_file)
    settings.trajectory_file = str(trajectory_file)
    settings.energy_file = str(energy_file)
    manager.save_project(settings)
    window = MDTrajectoryBatchQueueWindow(initial_project_dir=project_dir)
    widget = next(iter(window._widgets_by_id.values()))

    widget.preview_selection()

    assert widget.auto_timestep_box.isChecked()
    assert widget.timestep_spin.value() == pytest.approx(25.0)
    assert "Frame timestep: 25 fs (auto)" in (
        widget.preview_summary_label.text()
    )
    window.close()


def test_mdtrajectory_batch_queue_exposes_and_uses_editable_output_folder(
    qapp,
    tmp_path,
):
    del qapp
    project_dir, _trajectory_file, _energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, "project_a")
    )
    custom_output_dir = tmp_path / "custom_splitxyz"
    window = MDTrajectoryBatchQueueWindow(initial_project_dir=project_dir)
    widget = next(iter(window._widgets_by_id.values()))

    widget.preview_selection()

    assert widget.output_dir_edit.text().endswith("splitxyz_f2_t1500fs")

    widget.output_dir_edit.setText(str(custom_output_dir))
    jobs = [job for _item_id, job in window.queue_jobs_in_order()]

    assert jobs[0].output_dir == custom_output_dir.resolve()
    window.close()


def test_mdtrajectory_batch_queue_exposes_restart_duplicate_option(
    qapp,
    tmp_path,
):
    del qapp
    project_dir, _trajectory_file, _energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, "project_a")
    )
    window = MDTrajectoryBatchQueueWindow(initial_project_dir=project_dir)
    widget = next(iter(window._widgets_by_id.values()))

    widget.include_restart_duplicates_box.setChecked(True)
    jobs = [job for _item_id, job in window.queue_jobs_in_order()]

    assert jobs[0].include_restart_duplicates
    window.close()


def test_mdtrajectory_batch_queue_adds_multiple_selected_projects(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    project_a, trajectory_a, energy_a = _create_mdtrajectory_batch_project(
        tmp_path,
        "project_a",
    )
    project_b, trajectory_b, energy_b = _create_mdtrajectory_batch_project(
        tmp_path,
        "project_b",
    )
    monkeypatch.setattr(
        md_batch_queue_module,
        "_choose_existing_directories",
        lambda *_args, **_kwargs: (project_a, project_b),
    )
    window = MDTrajectoryBatchQueueWindow()

    window._choose_projects_to_add()

    assert window.queue_list.count() == 2
    jobs = [job for _item_id, job in window.queue_jobs_in_order()]
    assert [job.project_dir for job in jobs] == [
        project_a.resolve(),
        project_b.resolve(),
    ]
    assert [job.trajectory_file for job in jobs] == [
        trajectory_a.resolve(),
        trajectory_b.resolve(),
    ]
    assert [job.energy_file for job in jobs] == [
        energy_a.resolve(),
        energy_b.resolve(),
    ]
    assert [job.cutoff_fs for job in jobs] == [
        DEFAULT_TIME_CUTOFF_FS,
        DEFAULT_TIME_CUTOFF_FS,
    ]
    window.close()


def test_mdtrajectory_batch_worker_exports_and_registers_each_project(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_a, trajectory_a, energy_a = _create_mdtrajectory_batch_project(
        tmp_path,
        "project_a",
    )
    project_b, trajectory_b, energy_b = _create_mdtrajectory_batch_project(
        tmp_path,
        "project_b",
    )
    worker = MDTrajectoryBatchWorker(
        [
            (
                "item-a",
                MDTrajectoryBatchJob(
                    project_dir=project_a,
                    trajectory_file=trajectory_a,
                    topology_file=None,
                    energy_file=energy_a,
                    cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
                ),
            ),
            (
                "item-b",
                MDTrajectoryBatchJob(
                    project_dir=project_b,
                    trajectory_file=trajectory_b,
                    topology_file=None,
                    energy_file=energy_b,
                    cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
                ),
            ),
        ]
    )
    finished: list[list[object]] = []
    failures: list[tuple[str, str]] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(
        lambda item_id, message: failures.append((item_id, message))
    )

    worker.run()

    assert failures == []
    assert len(finished) == 1
    results = finished[0]
    assert len(results) == 2
    for result, project_dir, trajectory_file, energy_file in zip(
        results,
        (project_a, project_b),
        (trajectory_a, trajectory_b),
        (energy_a, energy_b),
        strict=True,
    ):
        saved_settings = manager.load_project(project_dir)
        assert saved_settings.resolved_frames_dir == (
            result.output_dir.resolve()
        )
        assert saved_settings.resolved_trajectory_file == (
            trajectory_file.resolve()
        )
        assert saved_settings.resolved_energy_file == energy_file.resolve()
        assert saved_settings.frames_dir_snapshot is not None
        assert result.written_count == 1
        assert result.selected_frames == 1
        assert result.output_dir.name == "splitxyz_f2_t1500fs"
        assert result.metadata_file is not None
        assert result.metadata_file.is_file()
        assert (result.output_dir / "frame_0002.xyz").is_file()


def test_mdtrajectory_batch_worker_honors_custom_output_folder(
    qapp,
    tmp_path,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir, trajectory_file, energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, "project_custom_output")
    )
    custom_output_dir = tmp_path / "queued_xyz_output"
    worker = MDTrajectoryBatchWorker(
        [
            (
                "item-custom",
                MDTrajectoryBatchJob(
                    project_dir=project_dir,
                    trajectory_file=trajectory_file,
                    topology_file=None,
                    energy_file=energy_file,
                    output_dir=custom_output_dir,
                    cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
                ),
            )
        ]
    )
    finished: list[list[object]] = []
    failures: list[tuple[str, str]] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(
        lambda item_id, message: failures.append((item_id, message))
    )

    worker.run()

    assert failures == []
    result = finished[0][0]
    assert result.output_dir == custom_output_dir
    assert (custom_output_dir / "frame_0002.xyz").is_file()
    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_frames_dir == custom_output_dir.resolve()


def test_mdtrajectory_batch_worker_uses_source_indices_without_validation_pass(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    manager = SAXSProjectManager()
    project_dir, trajectory_file, energy_file = (
        _create_mdtrajectory_overlap_batch_project(tmp_path, "project_overlap")
    )

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("Batch queue must not run export assertions")

    monkeypatch.setattr(
        md_batch_queue_module.MDTrajectoryWorkflow,
        "validate_export",
        fail_validation,
    )
    worker = MDTrajectoryBatchWorker(
        [
            (
                "item-overlap",
                MDTrajectoryBatchJob(
                    project_dir=project_dir,
                    trajectory_file=trajectory_file,
                    topology_file=None,
                    energy_file=energy_file,
                    cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
                ),
            )
        ]
    )
    finished: list[list[object]] = []
    failures: list[tuple[str, str]] = []
    log_messages: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(
        lambda item_id, message: failures.append((item_id, message))
    )
    worker.log.connect(log_messages.append)

    worker.run()

    assert failures == []
    assert len(finished) == 1
    result = finished[0][0]
    assert result.written_count == 2
    assert result.selected_frames == 2
    assert result.output_dir.name == "splitxyz_f2_t1500fs"
    assert (result.output_dir / "frame_0002.xyz").is_file()
    assert (result.output_dir / "frame_0003.xyz").is_file()
    assert not (result.output_dir / "frame_0004.xyz").exists()
    assert "i = 2" in (result.output_dir / "frame_0002.xyz").read_text()
    assert "i = 3" in (result.output_dir / "frame_0003.xyz").read_text()
    assert "9.2" in (result.output_dir / "frame_0002.xyz").read_text()
    assert any(
        "Skipped 2 duplicate source frame(s)" in message
        for message in log_messages
    )
    saved_settings = manager.load_project(project_dir)
    assert saved_settings.resolved_frames_dir == result.output_dir.resolve()


def test_mdtrajectory_batch_worker_can_include_restart_duplicates(
    qapp,
    tmp_path,
):
    del qapp
    project_dir, trajectory_file, energy_file = (
        _create_mdtrajectory_overlap_batch_project(
            tmp_path,
            "project_overlap_include",
        )
    )
    worker = MDTrajectoryBatchWorker(
        [
            (
                "item-overlap",
                MDTrajectoryBatchJob(
                    project_dir=project_dir,
                    trajectory_file=trajectory_file,
                    topology_file=None,
                    energy_file=energy_file,
                    cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
                    include_restart_duplicates=True,
                ),
            )
        ]
    )
    finished: list[list[object]] = []
    failures: list[tuple[str, str]] = []
    log_messages: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(
        lambda item_id, message: failures.append((item_id, message))
    )
    worker.log.connect(log_messages.append)

    worker.run()

    assert failures == []
    result = finished[0][0]
    assert result.include_restart_duplicates
    assert result.written_count == 3
    assert result.selected_frames == 3
    assert (result.output_dir / "frame_0002_duplicate0001.xyz").is_file()
    assert (result.output_dir / "frame_0002.xyz").is_file()
    assert (result.output_dir / "frame_0003.xyz").is_file()
    assert (
        "0.2"
        in (result.output_dir / "frame_0002_duplicate0001.xyz").read_text()
    )
    assert "9.2" in (result.output_dir / "frame_0002.xyz").read_text()
    assert any(
        "Included 2 duplicate source frame(s)" in message
        for message in log_messages
    )


def test_mdtrajectory_batch_queue_emits_registered_frames_folder(
    qapp,
    tmp_path,
):
    del qapp
    project_dir, _trajectory_file, _energy_file = (
        _create_mdtrajectory_batch_project(tmp_path, "project_a")
    )
    output_dir = tmp_path / "splitxyz_f2_t1500fs"
    output_dir.mkdir()
    window = MDTrajectoryBatchQueueWindow(initial_project_dir=project_dir)
    updates: list[dict[str, object]] = []
    window.project_paths_registered.connect(updates.append)
    item_id = str(window.queue_list.item(0).data(Qt.ItemDataRole.UserRole))

    window._on_item_finished(
        item_id,
        MDTrajectoryBatchResult(
            project_dir=project_dir.resolve(),
            output_dir=output_dir.resolve(),
            written_count=1,
            selected_frames=1,
            cutoff_fs=DEFAULT_TIME_CUTOFF_FS,
            metadata_file=None,
        ),
    )

    assert updates == [
        {
            "project_dir": project_dir.resolve(),
            "frames_dir": output_dir.resolve(),
        }
    ]
    window.close()
