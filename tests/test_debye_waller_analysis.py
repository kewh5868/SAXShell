from __future__ import annotations

import json
from pathlib import Path

import pytest

from saxshell.saxs.debye_waller.workflow import (
    DebyeWallerWorkflow,
    find_saved_project_debye_waller_analysis,
    load_debye_waller_analysis_result,
    save_debye_waller_analysis_to_project,
)


def _pdb_atom_line(
    atom_id: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    element: str,
) -> str:
    return (
        f"ATOM  {atom_id:5d} {atom_name:<4} {residue_name:>3}  "
        f"{residue_number:4d}    "
        f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
        f"  1.00  0.00          {element:>2}\n"
    )


def _write_water_frame(
    path: Path,
    *,
    mol1_oxygen_x: float,
    mol1_hydrogen_x: float,
    mol2_oxygen_x: float,
    mol2_hydrogen_x: float,
) -> None:
    lines = [
        _pdb_atom_line(1, "O", "HOH", 1, mol1_oxygen_x, 0.0, 0.0, "O"),
        _pdb_atom_line(2, "H1", "HOH", 1, mol1_hydrogen_x, 0.0, 0.0, "H"),
        _pdb_atom_line(3, "O", "HOH", 2, mol2_oxygen_x, 0.0, 0.0, "O"),
        _pdb_atom_line(4, "H1", "HOH", 2, mol2_hydrogen_x, 0.0, 0.0, "H"),
        "END\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_debye_waller_workflow_rejects_xyz_cluster_files(tmp_path):
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "H2O2"
    structure_dir.mkdir(parents=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "2\nframe\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = DebyeWallerWorkflow(clusters_dir)

    with pytest.raises(ValueError, match="PDB cluster files only"):
        workflow.run()


def test_debye_waller_workflow_computes_segmented_intra_and_inter_results(
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "H2O2"
    structure_dir.mkdir(parents=True)

    _write_water_frame(
        structure_dir / "water_frame_0001.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.00,
        mol2_oxygen_x=5.0,
        mol2_hydrogen_x=6.00,
    )
    _write_water_frame(
        structure_dir / "water_frame_0002.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.10,
        mol2_oxygen_x=5.2,
        mol2_hydrogen_x=6.35,
    )
    _write_water_frame(
        structure_dir / "water_frame_0004.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=0.95,
        mol2_oxygen_x=4.9,
        mol2_hydrogen_x=6.05,
    )
    _write_water_frame(
        structure_dir / "water_frame_0005.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.05,
        mol2_oxygen_x=5.1,
        mol2_hydrogen_x=6.30,
    )

    output_dir = tmp_path / "out"
    workflow = DebyeWallerWorkflow(
        clusters_dir,
        output_dir=output_dir,
        output_basename="water_dw",
    )
    logs: list[str] = []
    partial_updates = []
    result = workflow.run(
        log_callback=logs.append,
        stoichiometry_callback=partial_updates.append,
    )

    assert len(result.stoichiometry_results) == 1
    stoichiometry = result.stoichiometry_results[0]
    assert stoichiometry.label == "H2O2"
    assert len(stoichiometry.contiguous_frame_sets) == 2
    assert [
        entry.frame_ids for entry in stoichiometry.contiguous_frame_sets
    ] == [
        (1, 2),
        (4, 5),
    ]

    pair_labels = {
        (entry.scope, entry.pair_label): entry
        for entry in stoichiometry.pair_summaries
    }
    assert ("intra_molecular", "HOH:H1-HOH:O") in pair_labels
    assert ("inter_molecular", "H-O") in pair_labels
    assert ("inter_molecular", "H-H") in pair_labels
    assert ("inter_molecular", "O-O") in pair_labels

    intra_row = pair_labels[("intra_molecular", "HOH:H1-HOH:O")]
    assert intra_row.segment_count == 2
    assert intra_row.mean_pair_count == pytest.approx(2.0)
    assert intra_row.sigma_mean > 0.0
    assert intra_row.b_factor_mean > 0.0

    scope_rows = {
        entry.scope: entry for entry in stoichiometry.scope_summaries
    }
    assert set(scope_rows) == {"intra_molecular", "inter_molecular"}
    assert scope_rows["intra_molecular"].sigma_mean > 0.0
    assert scope_rows["inter_molecular"].sigma_mean > 0.0

    assert result.artifacts is not None
    assert result.artifacts.summary_json_path.exists()
    assert result.artifacts.aggregated_pair_summary_csv_path.exists()
    assert result.artifacts.pair_summary_csv_path.exists()
    assert result.artifacts.scope_summary_csv_path.exists()
    assert result.artifacts.segment_csv_path.exists()
    assert stoichiometry.info_summary is not None
    assert stoichiometry.info_summary.total_frame_count == 4
    assert stoichiometry.info_summary.total_frame_set_count == 2
    assert stoichiometry.info_summary.average_frames_per_set == pytest.approx(
        2.0
    )
    assert stoichiometry.info_summary.average_atoms_per_frame == pytest.approx(
        4.0
    )
    assert (
        stoichiometry.info_summary.average_molecules_per_frame
        == pytest.approx(2.0)
    )
    assert (
        stoichiometry.info_summary.average_solvent_like_molecules_per_frame
        == pytest.approx(2.0)
    )
    assert stoichiometry.info_summary.unique_residue_names == ("HOH",)
    assert stoichiometry.info_summary.unique_elements == ("H", "O")
    assert (
        stoichiometry.info_summary.solvent_like_molecule_signature == "HOH:HO"
    )
    aggregated_rows = {
        (entry.scope, entry.pair_label): entry
        for entry in result.aggregated_pair_summaries
    }
    assert ("intra_molecular", "HOH:H1-HOH:O") in aggregated_rows
    assert ("inter_molecular", "H-O") in aggregated_rows
    aggregated_intra = aggregated_rows[("intra_molecular", "HOH:H1-HOH:O")]
    assert aggregated_intra.stoichiometry_count == 1
    assert aggregated_intra.segment_count == 2
    assert aggregated_intra.sigma_mean == pytest.approx(intra_row.sigma_mean)
    assert aggregated_intra.b_factor_mean == pytest.approx(
        intra_row.b_factor_mean
    )

    assert "H2O2: found 2 contiguous frame set(s)." in logs
    assert (
        "H2O2: frame set 1/2 spans frames 1-2 (2 frame(s)) in series 'water'."
    ) in logs
    assert (
        "H2O2: frame set 2/2 spans frames 4-5 (2 frame(s)) in series 'water'."
    ) in logs
    assert any(
        "accumulated" in entry and "segment row(s) so far" in entry
        for entry in logs
    )

    assert len(partial_updates) == 2
    assert all(update.label == "H2O2" for update in partial_updates)
    assert len(partial_updates[0].segment_statistics) < len(
        partial_updates[-1].segment_statistics
    )
    assert (
        partial_updates[-1].segment_statistics
        == stoichiometry.segment_statistics
    )
    assert partial_updates[-1].pair_summaries == stoichiometry.pair_summaries
    assert partial_updates[-1].scope_summaries == stoichiometry.scope_summaries

    payload = json.loads(
        result.artifacts.summary_json_path.read_text(encoding="utf-8")
    )
    assert payload["inspection"]["is_pdb_only"] is True
    assert payload["stoichiometry_results"][0]["label"] == "H2O2"
    assert payload["aggregated_pair_summaries"][0]["segment_count"] >= 1
    assert payload["stoichiometry_results"][0]["info_summary"][
        "average_atoms_per_frame"
    ] == pytest.approx(4.0)
    assert (
        payload["stoichiometry_results"][0]["pair_summaries"][0][
            "segment_count"
        ]
        >= 1
    )


def test_debye_waller_analysis_can_be_saved_and_reloaded_from_project(
    tmp_path,
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "H2O2"
    structure_dir.mkdir(parents=True)

    _write_water_frame(
        structure_dir / "water_frame_0001.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.00,
        mol2_oxygen_x=5.0,
        mol2_hydrogen_x=6.00,
    )
    _write_water_frame(
        structure_dir / "water_frame_0002.pdb",
        mol1_oxygen_x=0.0,
        mol1_hydrogen_x=1.10,
        mol2_oxygen_x=5.2,
        mol2_hydrogen_x=6.35,
    )

    result = DebyeWallerWorkflow(
        clusters_dir,
        output_dir=tmp_path / "external_out",
        output_basename="external_run",
    ).run()

    saved = save_debye_waller_analysis_to_project(result, project_dir)
    summary_path = find_saved_project_debye_waller_analysis(project_dir)

    assert summary_path is not None
    assert summary_path.exists()
    assert saved.artifacts is not None
    assert saved.artifacts.summary_json_path == summary_path
    assert saved.artifacts.aggregated_pair_summary_csv_path.exists()

    reloaded = load_debye_waller_analysis_result(summary_path)

    assert reloaded.project_dir == project_dir.resolve()
    assert reloaded.output_dir == saved.output_dir
    assert len(reloaded.stoichiometry_results) == 1
    assert len(reloaded.aggregated_pair_summaries) >= 1
    assert reloaded.stoichiometry_results[0].label == "H2O2"
    assert reloaded.stoichiometry_results[0].info_summary is not None
