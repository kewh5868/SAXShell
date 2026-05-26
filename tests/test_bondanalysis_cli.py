from __future__ import annotations

import csv
from pathlib import Path

import pytest

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisWorkflow,
    BondPairDefinition,
    CoordinationNumberDefinition,
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


def _read_histogram_csv(
    path: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                if len(row) > 1:
                    metadata[row[0].removeprefix("# ").strip()] = row[1]
                continue
            header = row
            break
        if header is None:
            return metadata, rows
        for row in reader:
            rows.append(dict(zip(header, row)))
    return metadata, rows


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
        coordination_numbers=[
            CoordinationNumberDefinition("Pb", "I", 2.5),
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
        result.output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "Pb_I_distribution.npy"
    ).exists()
    all_histogram_csv = (
        result.output_dir / "all_clusters" / "Pb_I_histogram.csv"
    )
    assert all_histogram_csv.exists()
    angle_histogram_csv = (
        result.output_dir / "all_clusters" / "Pb_I_I_histogram.csv"
    )
    assert angle_histogram_csv.exists()
    coordination_histogram_csv = (
        result.output_dir / "all_clusters" / "CN_Pb_I_histogram.csv"
    )
    assert coordination_histogram_csv.exists()
    metadata, histogram_rows = _read_histogram_csv(all_histogram_csv)
    assert metadata["distribution_type"] == "bond"
    assert metadata["distribution_label"] == "Pb-I"
    assert metadata["scope"] == "All selected clusters"
    assert metadata["value_label"] == "Distance (A)"
    assert int(metadata["point_count"]) == 2
    assert float(metadata["mean"]) == 2.0
    assert float(metadata["median"]) == 2.0
    assert float(metadata["sigma"]) == 0.0
    assert float(metadata["sample_sigma"]) == 0.0
    angle_metadata, _angle_rows = _read_histogram_csv(angle_histogram_csv)
    assert angle_metadata["distribution_type"] == "angle"
    assert "sigma" in angle_metadata
    coordination_metadata, coordination_rows = _read_histogram_csv(
        coordination_histogram_csv
    )
    assert coordination_metadata["distribution_type"] == "coordination"
    assert coordination_metadata["distribution_label"] == "CN Pb-I"
    assert coordination_metadata["scope"] == "All selected clusters"
    assert coordination_metadata["value_label"] == "Coordination Number"
    assert coordination_metadata["center_atom"] == "Pb"
    assert coordination_metadata["atom_of_interest"] == "I"
    assert float(coordination_metadata["cutoff_angstrom"]) == 2.5
    assert int(coordination_metadata["point_count"]) == 2
    assert float(coordination_metadata["mean"]) == 1.0
    assert float(coordination_metadata["median"]) == 1.0
    assert float(coordination_metadata["mode"]) == 0.0
    assert float(coordination_metadata["sigma"]) == 1.0
    assert sum(int(row["count"]) for row in coordination_rows) == 2
    assert sum(int(row["count"]) for row in histogram_rows) == 2
    assert {"bin_left", "bin_right", "bin_center", "count", "density"} <= set(
        histogram_rows[0]
    )
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.png"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.csv"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "Pb_I_cluster_type_overlay.npy"
    ).exists()
    assert (
        result.output_dir
        / "cluster_types"
        / "PbI2"
        / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        result.output_dir
        / "cluster_types"
        / "PbI2"
        / "CN_Pb_I_coordination.npy"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        result.output_dir / "all_clusters" / "CN_Pb_I_coordination.npy"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "CN_Pb_I_cluster_type_overlay.csv"
    ).exists()
    assert (
        result.output_dir / "comparisons" / "CN_Pb_I_cluster_type_overlay.npy"
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
            "--coordination-number",
            "Pb:I:2.5",
        ]
    )

    captured = capsys.readouterr()
    output_dir = tmp_path / "bondanalysis_clusters_splitxyz0001"

    assert exit_code == 0
    assert f"Output directory: {output_dir}" in captured.out
    assert "Selected cluster types: PbI2" in captured.out
    assert "Structure files processed: 1" in captured.out
    assert "Results index file:" in captured.out
    assert "1 coordination values" in captured.out
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
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_histogram.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "Pb_I_I_histogram.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "CN_Pb_I_coordination.csv"
    ).exists()
    assert (
        output_dir / "cluster_types" / "PbI2" / "CN_Pb_I_histogram.csv"
    ).exists()


def test_histogram_csv_metadata_includes_nonzero_sigma(tmp_path):
    histogram_path = tmp_path / "distribution_histogram.csv"

    BondAnalysisWorkflow._write_histogram_csv(
        histogram_path,
        [1.0, 3.0],
        distribution_type="bond",
        distribution_label="Pb-I",
        scope_label="All selected clusters",
        value_label="Distance (A)",
        bins=2,
    )

    metadata, _histogram_rows = _read_histogram_csv(histogram_path)

    assert float(metadata["sigma"]) == 1.0
    assert float(metadata["standard_deviation"]) == 1.0
    assert float(metadata["sample_sigma"]) == pytest.approx(2**0.5)


def test_bondanalysis_cli_batch_ui_delegates_to_launcher(
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "project"
    clusters_dir = tmp_path / "clusters"
    launched: dict[str, object] = {}

    def fake_launch_batch_ui(
        initial_project_dir=None,
        *,
        initial_clusters_dir=None,
    ):
        launched["initial_project_dir"] = initial_project_dir
        launched["initial_clusters_dir"] = initial_clusters_dir
        return 7

    monkeypatch.setattr(
        "saxshell.bondanalysis.ui.batch_queue_window."
        "launch_bondanalysis_batch_queue_ui",
        fake_launch_batch_ui,
    )

    exit_code = bondanalysis_main(
        [
            "batch-ui",
            str(project_dir),
            "--clusters-dir",
            str(clusters_dir),
        ]
    )

    assert exit_code == 7
    assert launched["initial_project_dir"] == project_dir
    assert launched["initial_clusters_dir"] == clusters_dir


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
