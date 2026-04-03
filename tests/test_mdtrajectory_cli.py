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
