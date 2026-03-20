from __future__ import annotations

from pathlib import Path

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisWorkflow,
    BondPairDefinition,
)
from saxshell.bondanalysis.cli import main as bondanalysis_main
from saxshell.saxshell import main as saxshell_main


def _write_xyz_cluster(
    path: Path,
    *,
    atoms: list[tuple[str, float, float, float]],
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n")


def _build_sample_clusters_dir(base_dir: Path) -> Path:
    clusters_dir = base_dir / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    pbo_dir = clusters_dir / "PbO"
    pbi2_dir.mkdir(parents=True)
    pbo_dir.mkdir(parents=True)

    _write_xyz_cluster(
        pbi2_dir / "frame_0000_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_cluster(
        pbo_dir / "frame_0001_AAA.xyz",
        atoms=[
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 1.8, 0.0, 0.0),
        ],
    )
    return clusters_dir


def test_bondanalysis_workflow_supports_notebook_style_usage(tmp_path):
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    workflow = BondAnalysisWorkflow(
        clusters_dir,
        bond_pairs=[
            BondPairDefinition("Pb", "I", 2.5),
            BondPairDefinition("Pb", "O", 2.5),
        ],
        angle_triplets=[
            AngleTripletDefinition("Pb", "I", "I", 2.5, 2.5),
        ],
    )

    summary = workflow.inspect()
    result = workflow.run()

    assert summary["cluster_type_count"] == 2
    assert summary["total_structure_files"] == 2
    assert result.output_dir == tmp_path / "bondanalysis_clusters_splitxyz0001"
    assert result.total_structure_files == 2
    assert result.results_index_path.exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.npy"
    ).exists()
    assert (
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.png"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.npy"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.png"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.csv"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.npy"
    ).exists()


def test_bondanalysis_cli_run_writes_expected_outputs(tmp_path, capsys):
    clusters_dir = _build_sample_clusters_dir(tmp_path)

    exit_code = bondanalysis_main(
        [
            "run",
            str(clusters_dir),
            "--cluster-type",
            "PbI2",
            "--bond-pair",
            "Pb:I:2.5",
            "--angle-triplet",
            "Pb:I:I:2.5:2.5",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "bondanalysis_clusters_splitxyz0001"

    assert exit_code == 0
    assert f"Output directory: {output_dir}" in captured.out
    assert "Selected cluster types: PbI2" in captured.out
    assert "Structure files processed: 1" in captured.out
    assert "Results index file:" in captured.out
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_distribution.npy"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_angles.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_angles.npy"
    ).exists()


def test_saxshell_cli_forwards_to_bondanalysis_subcommand(
    tmp_path,
    capsys,
):
    clusters_dir = _build_sample_clusters_dir(tmp_path)

    exit_code = saxshell_main(
        [
            "bondanalysis",
            "inspect",
            str(clusters_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Clusters directory: {clusters_dir}" in captured.out
    assert "Cluster types detected: 2" in captured.out
    assert "Cluster types: PbI2, PbO" in captured.out
