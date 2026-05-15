from __future__ import annotations

import json
from pathlib import Path

from saxshell.cluster import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    ClusterWorkflow,
    build_cluster_run_config,
    default_cluster_run_file_path,
    example_atom_type_definitions,
    example_pair_cutoff_definitions,
    load_cluster_run_config,
    resolve_run_config_path,
    run_cluster_run_config,
    save_cluster_run_config,
)
from saxshell.cluster.cli import main as cluster_main


def _create_project(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "saxs_project.json").write_text(
        json.dumps(
            {
                "project_name": project_dir.name,
                "project_dir": str(project_dir.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_project(project_dir: Path) -> dict[str, object]:
    return json.loads(
        (project_dir / "saxs_project.json").read_text(encoding="utf-8")
    )


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


def test_cluster_run_config_round_trips_project_relative_paths(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    frames_dir = project_dir / "frames" / "splitxyz0001"
    frames_dir.mkdir(parents=True)
    output_dir = project_dir / "clusters" / "splitxyz0001"

    config = build_cluster_run_config(
        project_dir=project_dir,
        frames_dir=frames_dir,
        output_dir=output_dir,
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
        use_pbc=True,
        search_mode="vectorized",
        save_state_frequency=250,
    )
    run_file = default_cluster_run_file_path(project_dir)
    save_cluster_run_config(run_file, config)

    loaded = load_cluster_run_config(run_file)

    assert loaded.frames_dir == "frames/splitxyz0001"
    assert loaded.output_dir == "clusters/splitxyz0001"
    assert loaded.use_pbc is True
    assert loaded.search_mode == "vectorized"
    assert loaded.save_state_frequency == 250
    assert (
        resolve_run_config_path(loaded.frames_dir, project_dir=project_dir)
        == frames_dir.resolve()
    )


def test_cluster_project_backed_run_updates_project_clusters_dir(
    tmp_path,
):
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    frames_dir = project_dir / "frames" / "splitxyz0001"
    frames_dir.mkdir(parents=True)
    _write_xyz_frame(
        frames_dir / "frame_0000.xyz", pb1_x=0.0, i_x=1.0, pb2_x=2.0
    )
    project_payload = _read_project(project_dir)
    project_payload["frames_dir"] = str(frames_dir)
    (project_dir / "saxs_project.json").write_text(
        json.dumps(project_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    output_dir = project_dir / "clusters_splitxyz0001"
    config = build_cluster_run_config(
        project_dir=project_dir,
        frames_dir=frames_dir,
        output_dir=output_dir,
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
        smart_solvation_shells=False,
    )
    save_cluster_run_config(default_cluster_run_file_path(project_dir), config)

    exit_code = cluster_main(["run", str(project_dir)])
    saved_settings = _read_project(project_dir)

    assert exit_code == 0
    assert (
        Path(str(saved_settings["frames_dir"])).resolve()
        == frames_dir.resolve()
    )
    assert (
        Path(str(saved_settings["clusters_dir"])).resolve()
        == output_dir.resolve()
    )
    assert saved_settings["clusters_dir_snapshot"] is not None
    assert output_dir.exists()
    assert sorted(path.name for path in output_dir.rglob("*.xyz")) == [
        "frame_0000_AAA.xyz",
    ]


def test_run_cluster_run_config_preserves_existing_pdb_frames_field(tmp_path):
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    pdb_frames_dir = tmp_path / "pdb_frames"
    pdb_frames_dir.mkdir()
    frames_dir = project_dir / "frames" / "splitxyz0001"
    frames_dir.mkdir(parents=True)
    _write_xyz_frame(
        frames_dir / "frame_0000.xyz", pb1_x=0.0, i_x=1.0, pb2_x=2.0
    )
    project_payload = _read_project(project_dir)
    project_payload["pdb_frames_dir"] = str(pdb_frames_dir)
    (project_dir / "saxs_project.json").write_text(
        json.dumps(project_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    config = build_cluster_run_config(
        project_dir=project_dir,
        frames_dir=frames_dir,
        output_dir=project_dir / "clusters_splitxyz0001",
        atom_type_definitions=example_atom_type_definitions(),
        pair_cutoff_definitions=example_pair_cutoff_definitions(),
        smart_solvation_shells=False,
    )

    summary = run_cluster_run_config(project_dir, config)
    saved_settings = _read_project(project_dir)

    assert summary.written_count == 1
    assert (
        Path(str(saved_settings["pdb_frames_dir"])).resolve()
        == pdb_frames_dir.resolve()
    )
    assert (
        Path(str(saved_settings["clusters_dir"])).resolve()
        == summary.output_dir.resolve()
    )
