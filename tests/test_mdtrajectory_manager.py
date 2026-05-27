import pytest

from saxshell.mdtrajectory.frame.base import FrameRecord
from saxshell.mdtrajectory.frame.exporters import export_xyz_frames
from saxshell.mdtrajectory.frame.manager import (
    DEFAULT_FRAME_TIMESTEP_FS,
    TrajectoryManager,
)


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


def test_inspect_detects_frame_timestep_from_trajectory_times(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    _write_uniform_time_xyz(
        trajectory_file,
        n_frames=4,
        timestep_fs=25.0,
    )

    manager = TrajectoryManager(
        input_file=trajectory_file,
        frame_timestep_fs=DEFAULT_FRAME_TIMESTEP_FS,
    )

    summary = manager.inspect()
    preview = manager.preview_selection()

    assert summary["detected_frame_timestep_fs"] == pytest.approx(25.0)
    assert summary["source_time_metadata_frames"] == 4
    assert summary["inferred_time_frames"] == 0
    assert preview.detected_frame_timestep_fs == pytest.approx(25.0)
    assert preview.frame_timestep_fs == pytest.approx(
        DEFAULT_FRAME_TIMESTEP_FS
    )


def test_manual_frame_timestep_can_supply_missing_times(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "1\n"
        "no source time\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        "still no source time\n"
        "H 1.0 0.0 0.0\n"
        "1\n"
        "still no source time\n"
        "H 2.0 0.0 0.0\n",
        encoding="utf-8",
    )

    manager = TrajectoryManager(
        input_file=trajectory_file,
        frame_timestep_fs=2.5,
    )

    summary = manager.inspect()
    preview = manager.preview_selection(min_time_fs=2.5)

    assert summary["source_time_metadata_frames"] == 0
    assert summary["inferred_time_frames"] == 3
    assert preview.first_frame_index == 1
    assert preview.first_time_fs == pytest.approx(2.5)


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


def test_cp2k_restart_overlap_frames_keep_later_source_index_occurrence(
    tmp_path,
):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "1\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 1.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 9.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 9.0 0.0 0.0\n"
        "1\n"
        "i = 3, time = 1.5, E = -1.0\n"
        "H 3.0 0.0 0.0\n",
        encoding="utf-8",
    )

    manager = TrajectoryManager(input_file=trajectory_file)
    summary = manager.inspect()
    preview = manager.preview_selection(min_time_fs=0.5)
    written_files = manager.export_frames(
        output_dir=tmp_path / "frames",
        min_time_fs=0.5,
    )

    assert summary["n_frames"] == 4
    assert summary["raw_frames"] == 6
    assert summary["duplicate_source_frames"] == 2
    assert preview.total_frames == 4
    assert preview.selected_frames == 3
    assert preview.first_frame_index == 1
    assert preview.last_frame_index == 3
    assert [path.name for path in written_files] == [
        "frame_0001.xyz",
        "frame_0002.xyz",
        "frame_0003.xyz",
    ]
    assert "H        9.0" in written_files[0].read_text()
    assert "H        9.0" in written_files[1].read_text()
    assert "H        3.0" in written_files[2].read_text()


def test_cp2k_restart_overlap_frames_can_include_duplicate_occurrences(
    tmp_path,
):
    trajectory_file = tmp_path / "traj.xyz"
    trajectory_file.write_text(
        "1\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 1.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 9.1 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 9.2 0.0 0.0\n"
        "1\n"
        "i = 3, time = 1.5, E = -1.0\n"
        "H 3.0 0.0 0.0\n",
        encoding="utf-8",
    )

    manager = TrajectoryManager(
        input_file=trajectory_file,
        include_restart_duplicates=True,
    )
    summary = manager.inspect()
    preview = manager.preview_selection(min_time_fs=0.5)
    written_files = manager.export_frames(
        output_dir=tmp_path / "frames",
        min_time_fs=0.5,
    )

    assert summary["n_frames"] == 6
    assert summary["raw_frames"] == 6
    assert summary["duplicate_source_frames"] == 2
    assert summary["include_restart_duplicates"] is True
    assert preview.total_frames == 6
    assert preview.selected_frames == 5
    assert [path.name for path in written_files] == [
        "frame_0001_duplicate0001.xyz",
        "frame_0002_duplicate0001.xyz",
        "frame_0001.xyz",
        "frame_0002.xyz",
        "frame_0003.xyz",
    ]
    assert (
        "H        1.0"
        in (tmp_path / "frames" / "frame_0001_duplicate0001.xyz").read_text()
    )
    assert (
        "H        9.1" in (tmp_path / "frames" / "frame_0001.xyz").read_text()
    )
    assert (
        "H        2.0"
        in (tmp_path / "frames" / "frame_0002_duplicate0001.xyz").read_text()
    )
    assert (
        "H        9.2" in (tmp_path / "frames" / "frame_0002.xyz").read_text()
    )


def test_export_rejects_xyz_header_index_mismatch(tmp_path):
    frame = FrameRecord(
        frame_index=2559,
        file_type="xyz",
        atom_count=1,
        lines=[
            "i = 2501, time = 1250.5, E = -1.0\n",
            "H 0.0 0.0 0.0\n",
        ],
        time_fs=1250.5,
    )

    with pytest.raises(
        ValueError,
        match="header reports i = 2501",
    ):
        export_xyz_frames([frame], tmp_path / "frames")


def test_export_rejects_duplicate_xyz_output_names(tmp_path):
    frames = [
        FrameRecord(
            frame_index=7,
            file_type="xyz",
            atom_count=1,
            lines=[
                "i = 7, time = 3.5, E = -1.0\n",
                "H 0.0 0.0 0.0\n",
            ],
            time_fs=3.5,
        ),
        FrameRecord(
            frame_index=7,
            file_type="xyz",
            atom_count=1,
            lines=[
                "i = 7, time = 3.5, E = -1.0\n",
                "H 1.0 0.0 0.0\n",
            ],
            time_fs=3.5,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="same output file",
    ):
        export_xyz_frames(frames, tmp_path / "frames")
