from __future__ import annotations

from pathlib import Path

from saxshell.cluster import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    ClusterWorkflow,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
)
from saxshell.cluster.cli import main as cluster_main


def _write_xyz_frame(
    path: Path,
    *,
    pb1_x: float,
    i_x: float,
    pb2_x: float,
) -> None:
    path.write_text(
        "5\n"
        f"{path.stem}\n"
        f"Pb {pb1_x:.1f} 0.0 0.0\n"
        f"I {i_x:.1f} 0.0 0.0\n"
        f"Pb {pb2_x:.1f} 0.0 0.0\n"
        "O 0.2 1.0 0.0\n"
        "H 0.2 1.7 0.0\n"
    )


def test_cluster_workflow_supports_notebook_style_end_to_end_usage(tmp_path):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    _write_xyz_frame(
        frames_dir / "frame_0000.xyz", pb1_x=0.0, i_x=1.0, pb2_x=2.0
    )
    _write_xyz_frame(
        frames_dir / "frame_0001.xyz",
        pb1_x=0.0,
        i_x=5.0,
        pb2_x=10.0,
    )

    workflow = ClusterWorkflow(
        frames_dir=frames_dir,
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
        use_pbc=True,
    )

    summary = workflow.inspect()
    selection = workflow.preview_selection()
    export = workflow.export_clusters()

    assert summary["frame_format"] == "xyz"
    assert summary["estimated_box_dimensions"] == (2.0, 1.7, 0.0)
    assert selection.search_mode == "kdtree"
    assert selection.save_state_frequency == DEFAULT_SAVE_STATE_FREQUENCY
    assert not selection.smart_solvation_shells
    assert selection.resolved_box_dimensions == (2.0, 1.7, 0.0)
    assert selection.output_dir == tmp_path / "clusters_splitxyz0001"
    assert export.output_dir == selection.output_dir
    assert export.metadata_path == (
        selection.output_dir / "cluster_extraction_metadata.json"
    )
    assert not export.resumed
    assert not export.already_complete
    assert sorted(path.name for path in export.written_files) == [
        "frame_0000_AAA.xyz",
        "frame_0001_AAA.xyz",
        "frame_0001_AAB.xyz",
    ]


def test_clusters_cli_export_runs_complete_headless_workflow(
    tmp_path,
    capsys,
):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    _write_xyz_frame(
        frames_dir / "frame_0000.xyz", pb1_x=0.0, i_x=1.0, pb2_x=2.0
    )
    _write_xyz_frame(
        frames_dir / "frame_0001.xyz",
        pb1_x=0.0,
        i_x=5.0,
        pb2_x=10.0,
    )

    exit_code = cluster_main(
        [
            "export",
            str(frames_dir),
            "--use-pbc",
            "--node",
            "Pb",
            "--linker",
            "I",
            "--shell",
            "O",
            "--pair-cutoff",
            "Pb:I:3.36",
            "--pair-cutoff",
            "Pb:O:3.36",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "clusters_splitxyz0001"

    assert exit_code == 0
    assert "Mode: XYZ frames" in captured.out
    assert "Smart solvation shells: off" in captured.out
    assert "Search mode: KDTree" in captured.out
    assert (
        "Save-state frequency: every "
        f"{DEFAULT_SAVE_STATE_FREQUENCY} frames" in captured.out
    )
    assert f"Output directory: {output_dir}" in captured.out
    assert "Files written: 3" in captured.out
    assert "Resume status: new extraction" in captured.out
    assert (
        f"Metadata file: {output_dir / 'cluster_extraction_metadata.json'}"
        in captured.out
    )
    assert output_dir.exists()
    assert sorted(path.name for path in output_dir.rglob("*.xyz")) == [
        "frame_0000_AAA.xyz",
        "frame_0001_AAA.xyz",
        "frame_0001_AAB.xyz",
    ]
