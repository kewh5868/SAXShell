import pytest

from saxshell.mdtrajectory.frame.manager import TrajectoryManager


def _write_uniform_time_xyz(
    path,
    *,
    n_frames: int,
    timestep_fs: float,
) -> None:
    lines: list[str] = []
    for frame_index in range(n_frames):
        lines.extend(
            [
                "1\n",
                (
                    "i = "
                    f"{frame_index}, time = {frame_index * timestep_fs:.3f}, "
                    "E = -1.0\n"
                ),
                "H 0.0 0.0 0.0\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def test_preview_selection_respects_time_cutoff(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
        "2\n"
        "i = 2, time = 100.0, E = -1.0\n"
        "H 0.0 0.2 0.0\n"
        "O 1.0 0.2 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    preview = manager.preview_selection(start=1, min_time_fs=60.0)

    assert preview.total_frames == 3
    assert preview.selected_frames == 1
    assert preview.first_frame_index == 2
    assert preview.last_frame_index == 2
    assert preview.first_time_fs == pytest.approx(100.0)
    assert preview.last_time_fs == pytest.approx(100.0)


def test_export_frames_returns_written_files(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
        "2\n"
        "i = 2, time = 100.0, E = -1.0\n"
        "H 0.0 0.2 0.0\n"
        "O 1.0 0.2 0.0\n"
    )
    output_dir = tmp_path / "frames"

    manager = TrajectoryManager(input_file=trajectory_file)
    written_files = manager.export_frames(
        output_dir=output_dir,
        min_time_fs=50.0,
    )

    assert [path.name for path in written_files] == [
        "frame_0001.xyz",
        "frame_0002.xyz",
    ]
    assert all(path.exists() for path in written_files)
    assert written_files[0].read_text().startswith("2\n")


def test_preview_selection_can_keep_every_nth_frame_after_cutoff(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
        "2\n"
        "i = 2, time = 100.0, E = -1.0\n"
        "H 0.0 0.2 0.0\n"
        "O 1.0 0.2 0.0\n"
        "2\n"
        "i = 3, time = 150.0, E = -1.0\n"
        "H 0.0 0.3 0.0\n"
        "O 1.0 0.3 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    preview = manager.preview_selection(
        min_time_fs=50.0,
        post_cutoff_stride=2,
    )

    assert preview.selected_frames == 2
    assert preview.post_cutoff_stride == 2
    assert preview.first_frame_index == 1
    assert preview.last_frame_index == 3
    assert preview.first_time_fs == pytest.approx(50.0)
    assert preview.last_time_fs == pytest.approx(150.0)


def test_time_cutoff_requires_time_metadata(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "frame without time metadata\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "2\n"
        "still no time metadata\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)

    with pytest.raises(
        ValueError,
        match="does not include frame times",
    ):
        manager.preview_selection(min_time_fs=10.0)


def test_prefix_style_headers_can_supply_frame_times(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "NSTEP=1 TIME[fs]= 0.0\n"
        "2\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "NSTEP=2 TIME[fs]= 100.0\n"
        "2\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    preview = manager.preview_selection(min_time_fs=50.0)

    assert preview.selected_frames == 1
    assert preview.first_frame_index == 1
    assert preview.first_time_fs == pytest.approx(100.0)


def test_inspect_counts_frames_without_loading_entire_selection(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "frame 0 comment\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "NSTEP=2 TIME[fs]= 100.0\n"
        "2\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    summary = manager.inspect()

    assert summary["file_type"] == "xyz"
    assert summary["n_frames"] == 2
    assert manager.frames is None


def test_preview_selection_uses_frame_metadata_without_loading_frames(
    tmp_path,
    monkeypatch,
):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "2\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "O 1.0 0.0 0.0\n"
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n"
        "2\n"
        "i = 2, time = 100.0, E = -1.0\n"
        "H 0.0 0.2 0.0\n"
        "O 1.0 0.2 0.0\n"
    )

    manager = TrajectoryManager(input_file=trajectory_file)

    def fail_load_frames():
        raise AssertionError("preview_selection should not load full frames")

    monkeypatch.setattr(manager, "load_frames", fail_load_frames)

    preview = manager.preview_selection(min_time_fs=50.0)

    assert preview.selected_frames == 2
    assert preview.first_frame_index == 1
    assert preview.last_frame_index == 2


def test_export_frames_keeps_exact_half_fs_cutoff_boundary(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    _write_uniform_time_xyz(
        trajectory_file,
        n_frames=1005,
        timestep_fs=0.5,
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    preview = manager.preview_selection(min_time_fs=497.5)
    written_files = manager.export_frames(
        output_dir=tmp_path / "frames",
        min_time_fs=497.5,
    )

    assert preview.selected_frames == 10
    assert preview.first_frame_index == 995
    assert preview.first_time_fs == pytest.approx(497.5)
    assert written_files[0].name == "frame_0995.xyz"
    assert written_files[-1].name == "frame_1004.xyz"
