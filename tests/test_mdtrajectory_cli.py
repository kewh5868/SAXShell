from __future__ import annotations

import json
from pathlib import Path

from saxshell.mdtrajectory import MDTrajectoryWorkflow
from saxshell.mdtrajectory.cli import main as mdtrajectory_main
from saxshell.saxshell import main as saxshell_main


def _write_uniform_time_xyz(
    path: Path,
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


def _write_sample_xyz(path: Path) -> None:
    path.write_text(
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


def _write_sample_ener(path: Path) -> None:
    path.write_text(
        "# step time kinetic temperature potential\n"
        "0 0.0 1.0 290.0 -10.0\n"
        "1 50.0 1.0 300.0 -10.0\n"
        "2 100.0 1.0 300.5 -10.0\n"
        "3 150.0 1.0 299.8 -10.0\n"
    )


def _write_restart_overlap_xyz(path: Path) -> None:
    path.write_text(
        "1\n"
        "i = 0, time = 0.0, E = -1.0\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 1.0 0.0 0.0\n"
        "1\n"
        "i = 1, time = 0.5, E = -1.0\n"
        "H 9.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n",
        encoding="utf-8",
    )


def test_workflow_supports_notebook_style_end_to_end_usage(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    energy_file = tmp_path / "traj.ener"
    _write_sample_xyz(trajectory_file)
    _write_sample_ener(energy_file)

    workflow = MDTrajectoryWorkflow(
        trajectory_file=trajectory_file,
        energy_file=energy_file,
    )

    summary = workflow.inspect()
    suggested = workflow.suggest_cutoff(
        temp_target_k=300.0,
        temp_tol_k=1.0,
        window=2,
    )
    workflow.set_selected_cutoff(suggested.cutoff_time_fs)
    selection = workflow.preview_selection(use_cutoff=True)
    export = workflow.export_frames(use_cutoff=True)

    assert summary["n_frames"] == 4
    assert suggested.cutoff_time_fs == 50.0
    assert selection.output_dir == tmp_path / "splitxyz_f1_t50fs"
    assert selection.preview.selected_frames == 3
    assert export.output_dir == selection.output_dir
    assert (
        export.metadata_file == export.output_dir / "mdtrajectory_export.json"
    )
    metadata_payload = json.loads(export.metadata_file.read_text())
    assert metadata_payload["selection"]["applied_cutoff_fs"] == 50.0
    assert (
        metadata_payload["written_frames"][0]["filename"] == "frame_0001.xyz"
    )
    assert metadata_payload["written_frames"][0]["time_fs"] == 50.0
    assert [path.name for path in export.written_files] == [
        "frame_0001.xyz",
        "frame_0002.xyz",
        "frame_0003.xyz",
    ]


def test_workflow_can_keep_every_nth_frame_after_cutoff(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    energy_file = tmp_path / "traj.ener"
    _write_sample_xyz(trajectory_file)
    _write_sample_ener(energy_file)

    workflow = MDTrajectoryWorkflow(
        trajectory_file=trajectory_file,
        energy_file=energy_file,
    )

    suggested = workflow.suggest_cutoff(
        temp_target_k=300.0,
        temp_tol_k=1.0,
        window=2,
    )
    workflow.set_selected_cutoff(suggested.cutoff_time_fs)
    selection = workflow.preview_selection(
        use_cutoff=True,
        post_cutoff_stride=2,
    )
    export = workflow.export_frames(
        use_cutoff=True,
        post_cutoff_stride=2,
    )

    assert selection.preview.selected_frames == 2
    assert selection.preview.post_cutoff_stride == 2
    assert [path.name for path in export.written_files] == [
        "frame_0001.xyz",
        "frame_0003.xyz",
    ]


def test_workflow_names_cutoff_exports_from_first_selected_frame_and_time(
    tmp_path,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_uniform_time_xyz(
        trajectory_file,
        n_frames=1005,
        timestep_fs=0.5,
    )

    workflow = MDTrajectoryWorkflow(trajectory_file=trajectory_file)

    selection = workflow.preview_selection(
        use_cutoff=True,
        cutoff_fs=497.5,
    )

    assert selection.preview.first_frame_index == 995
    assert selection.preview.first_time_fs == 497.5
    assert selection.output_dir == tmp_path / "splitxyz_f995_t497p5fs"


def test_mdtrajectory_cli_export_runs_complete_headless_workflow(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    energy_file = tmp_path / "traj.ener"
    _write_sample_xyz(trajectory_file)
    _write_sample_ener(energy_file)

    exit_code = mdtrajectory_main(
        [
            "export",
            str(trajectory_file),
            "--energy-file",
            str(energy_file),
            "--use-suggested-cutoff",
            "--temp-target-k",
            "300.0",
            "--temp-tol-k",
            "1.0",
            "--window",
            "2",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "splitxyz_f1_t50fs"

    assert exit_code == 0
    assert "Frame export complete." in captured.out
    assert f"Output directory: {output_dir}" in captured.out
    assert output_dir.exists()
    assert sorted(path.name for path in output_dir.glob("*.xyz")) == [
        "frame_0001.xyz",
        "frame_0002.xyz",
        "frame_0003.xyz",
    ]


def test_mdtrajectory_cli_reports_detected_timestep_and_frame_interval(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_uniform_time_xyz(
        trajectory_file,
        n_frames=4,
        timestep_fs=25.0,
    )

    inspect_code = mdtrajectory_main(["inspect", str(trajectory_file)])
    preview_code = mdtrajectory_main(
        [
            "preview",
            str(trajectory_file),
            "--frame-interval",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert inspect_code == 0
    assert preview_code == 0
    assert "Detected frame timestep: 25 fs" in captured.out
    assert "Frame interval: 2" in captured.out


def test_mdtrajectory_cli_export_can_include_restart_duplicates(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_restart_overlap_xyz(trajectory_file)

    exit_code = mdtrajectory_main(
        [
            "export",
            str(trajectory_file),
            "--include-restart-duplicates",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "splitxyz_f0_t0fs"
    metadata_payload = json.loads(
        (output_dir / "mdtrajectory_export.json").read_text()
    )

    assert exit_code == 0
    assert "Restart duplicate frames: included" in captured.out
    assert sorted(path.name for path in output_dir.glob("*.xyz")) == [
        "frame_0000.xyz",
        "frame_0001.xyz",
        "frame_0001_duplicate0001.xyz",
        "frame_0002.xyz",
    ]
    assert (
        "H        1.0"
        in (output_dir / "frame_0001_duplicate0001.xyz").read_text()
    )
    assert "H        9.0" in (output_dir / "frame_0001.xyz").read_text()
    assert metadata_payload["selection"]["include_restart_duplicates"] is True


def test_workflow_can_validate_exported_xyz_frame_mapping(tmp_path):
    trajectory_file = tmp_path / "traj.xyz"
    _write_sample_xyz(trajectory_file)

    workflow = MDTrajectoryWorkflow(trajectory_file=trajectory_file)
    export = workflow.export_frames(use_cutoff=True, cutoff_fs=50.0)

    result = workflow.validate_export(
        export.output_dir,
        expect_contiguous=True,
    )

    assert result.passed
    assert result.exported_files == 3
    assert result.validated_files == 3
    assert result.filename_index_min == 1
    assert result.filename_index_max == 3
    assert result.header_index_min == 1
    assert result.header_index_max == 3
    assert result.filename_header_offsets == {0: 3}
    assert result.issue_counts == {}


def test_workflow_validation_accepts_purged_source_duplicate_conflicts(
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
        "i = 1, time = 0.5, E = -1.0\n"
        "H 9.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = MDTrajectoryWorkflow(trajectory_file=trajectory_file)
    export = workflow.export_frames()

    result = workflow.validate_export(export.output_dir)
    strict_result = workflow.validate_export(
        export.output_dir,
        strict_source_duplicates=True,
    )

    assert result.passed
    assert result.source_duplicate_indices == 1
    assert result.source_duplicate_conflicts == 1
    assert result.issue_counts == {}
    assert strict_result.failure_count == 1
    assert not strict_result.passed
    assert "H        9.0" in (export.output_dir / "frame_0001.xyz").read_text()


def test_workflow_validation_allows_identical_source_duplicates_by_default(
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
        "i = 1, time = 0.5, E = -1.0\n"
        "H 1.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = MDTrajectoryWorkflow(trajectory_file=trajectory_file)
    export = workflow.export_frames()

    result = workflow.validate_export(export.output_dir)
    strict_result = workflow.validate_export(
        export.output_dir,
        strict_source_duplicates=True,
    )

    assert result.passed
    assert result.source_duplicate_indices == 1
    assert result.source_duplicate_conflicts == 0
    assert result.issue_counts == {}
    assert strict_result.failure_count == 1
    assert not strict_result.passed


def test_workflow_validation_rejects_export_that_keeps_earlier_overlap(
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
        "i = 1, time = 0.5, E = -1.0\n"
        "H 9.0 0.0 0.0\n"
        "1\n"
        "i = 2, time = 1.0, E = -1.0\n"
        "H 2.0 0.0 0.0\n",
        encoding="utf-8",
    )
    bad_export_dir = tmp_path / "bad_frames"
    bad_export_dir.mkdir()
    (bad_export_dir / "frame_0001.xyz").write_text(
        "1\n" "i = 1, time = 0.5, E = -1.0\n" "H 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = MDTrajectoryWorkflow(trajectory_file=trajectory_file)
    result = workflow.validate_export(bad_export_dir)

    assert not result.passed
    assert result.source_duplicate_conflicts == 1
    assert result.issue_counts == {"coordinate_mismatch": 1}


def test_mdtrajectory_cli_validate_export_reports_mapping_failures(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_sample_xyz(trajectory_file)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_0001.xyz").write_text(
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n",
        encoding="utf-8",
    )
    (frame_dir / "frame_0002.xyz").write_text(
        "2\n"
        "i = 1, time = 50.0, E = -1.0\n"
        "H 0.0 0.1 0.0\n"
        "O 1.0 0.1 0.0\n",
        encoding="utf-8",
    )

    exit_code = mdtrajectory_main(
        [
            "validate-export",
            str(trajectory_file),
            str(frame_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Export validation failed." in captured.out
    assert "- filename_header_offset: 1" in captured.out
    assert "- coordinate_mismatch: 1" in captured.out
    assert "- duplicate_export_header_index: 1" in captured.out


def test_mdtrajectory_cli_validate_export_fails_empty_frame_directory(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_sample_xyz(trajectory_file)
    frame_dir = tmp_path / "empty_frames"
    frame_dir.mkdir()

    exit_code = mdtrajectory_main(
        [
            "validate-export",
            str(trajectory_file),
            str(frame_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "- no_exported_xyz_files: 1" in captured.out


def test_mdtrajectory_cli_suggest_cutoff_defaults_to_window_two(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    energy_file = tmp_path / "traj.ener"
    _write_sample_xyz(trajectory_file)
    _write_sample_ener(energy_file)

    exit_code = mdtrajectory_main(
        [
            "suggest-cutoff",
            str(trajectory_file),
            "--energy-file",
            str(energy_file),
            "--temp-target-k",
            "300.0",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Suggested cutoff: 50.000 fs" in captured.out
    assert "Window: 2" in captured.out


def test_saxshell_cli_forwards_to_mdtrajectory_subcommand(
    tmp_path,
    capsys,
):
    trajectory_file = tmp_path / "traj.xyz"
    _write_sample_xyz(trajectory_file)

    exit_code = saxshell_main(
        [
            "mdtrajectory",
            "inspect",
            str(trajectory_file),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Trajectory file: {trajectory_file}" in captured.out
    assert "Frames: 4" in captured.out
