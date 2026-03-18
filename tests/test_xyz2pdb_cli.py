from __future__ import annotations

import json
from pathlib import Path

from saxshell.saxshell import main as saxshell_main
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import XYZToPDBWorkflow
from saxshell.xyz2pdb.cli import main as xyz2pdb_main


def _write_reference_pdb(path: Path) -> None:
    structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="PB1",
                residue_name="PBI",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="Pb",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="I1",
                residue_name="PBI",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="I",
            ),
        ]
    )
    structure.write_pdb_file(path)


def _write_xyz(path: Path, *, i_x: float, oxygen_x: float) -> None:
    path.write_text(
        "3\n"
        f"{path.stem}\n"
        "Pb 0.0 0.0 0.0\n"
        f"I {i_x:.3f} 0.0 0.0\n"
        f"O {oxygen_x:.3f} 0.0 0.0\n"
    )


def _write_config(path: Path, *, reference_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "molecules": [
                    {
                        "name": "PBI",
                        "reference": reference_name,
                        "residue_name": "PBI",
                        "anchors": [{"pair": ["PB1", "I1"], "tol": 0.25}],
                    }
                ],
                "free_atoms": {
                    "O": {
                        "residue_name": "SOL",
                        "atom_name": "O1",
                    }
                },
            },
            indent=2,
        )
        + "\n"
    )


def test_xyz2pdb_workflow_supports_notebook_style_end_to_end_usage(tmp_path):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)
    _write_xyz(frames_dir / "frame_0001.xyz", i_x=1.1, oxygen_x=2.1)

    config_file = tmp_path / "assignments.json"
    _write_config(config_file, reference_name="pbi")

    workflow = XYZToPDBWorkflow(
        frames_dir,
        config_file=config_file,
        reference_library_dir=refs_dir,
    )

    inspection = workflow.inspect()
    preview = workflow.preview_conversion()
    export = workflow.export_pdbs()

    assert inspection.input_mode == "xyz_folder"
    assert inspection.total_files == 2
    assert inspection.configured_reference_names == ("pbi",)
    assert preview.molecule_counts["PBI"] == 1
    assert preview.residue_counts["PBI"] == 1
    assert preview.residue_counts["SOL"] == 1
    assert export.output_dir == tmp_path / "xyz2pdb_frames"
    assert [path.name for path in export.written_files] == [
        "frame_0000.pdb",
        "frame_0001.pdb",
    ]
    first_output = export.written_files[0].read_text()
    assert "PBI" in first_output
    assert "SOL" in first_output


def test_xyz2pdb_cli_export_runs_complete_headless_workflow(
    tmp_path,
    capsys,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference_pdb(refs_dir / "pbi.pdb")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)
    _write_xyz(frames_dir / "frame_0001.xyz", i_x=1.1, oxygen_x=2.1)

    config_file = tmp_path / "assignments.json"
    _write_config(config_file, reference_name="pbi")

    exit_code = xyz2pdb_main(
        [
            "export",
            str(frames_dir),
            "--config",
            str(config_file),
            "--library-dir",
            str(refs_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "XYZ to PDB conversion complete." in captured.out
    assert f"Output directory: {tmp_path / 'xyz2pdb_frames'}" in captured.out
    assert "Files written: 2" in captured.out


def test_xyz2pdb_reference_cli_and_saxshell_forwarding(tmp_path, capsys):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()

    source_xyz = tmp_path / "reference_source.xyz"
    _write_xyz(source_xyz, i_x=1.0, oxygen_x=2.0)

    add_exit_code = xyz2pdb_main(
        [
            "references",
            "add",
            str(source_xyz),
            "--name",
            "pbi",
            "--residue-name",
            "PBI",
            "--library-dir",
            str(refs_dir),
        ]
    )
    add_output = capsys.readouterr().out

    assert add_exit_code == 0
    assert (refs_dir / "pbi.pdb").exists()
    assert "Reference created: pbi" in add_output

    list_exit_code = xyz2pdb_main(
        ["references", "list", "--library-dir", str(refs_dir)]
    )
    list_output = capsys.readouterr().out

    assert list_exit_code == 0
    assert "pbi: residue PBI" in list_output

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(frames_dir / "frame_0000.xyz", i_x=1.0, oxygen_x=2.0)
    config_file = tmp_path / "assignments.json"
    _write_config(config_file, reference_name="pbi")

    inspect_exit_code = saxshell_main(
        [
            "xyz2pdb",
            "inspect",
            str(frames_dir),
            "--config",
            str(config_file),
            "--library-dir",
            str(refs_dir),
        ]
    )
    inspect_output = capsys.readouterr().out

    assert inspect_exit_code == 0
    assert f"Input path: {frames_dir}" in inspect_output
    assert "Configured molecules: PBI" in inspect_output
