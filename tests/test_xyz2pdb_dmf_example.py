from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import (
    XYZToPDBWorkflow,
    default_reference_library_dir,
    resolve_reference_path,
)


def _rotation_x(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


def _rotation_y(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=float,
    )


def _rotation_z(angle_degrees: float) -> np.ndarray:
    radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(radians))
    sine = float(np.sin(radians))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _load_reference_atoms() -> list[PDBAtom]:
    reference_path = resolve_reference_path(
        "dmf",
        library_dir=default_reference_library_dir(),
    )
    structure = PDBStructure.from_file(reference_path)
    return [atom.copy() for atom in structure.atoms]


def _build_transformed_molecule(
    reference_atoms: list[PDBAtom],
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[tuple[str, str, np.ndarray]]:
    reference_coordinates = np.array(
        [atom.coordinates for atom in reference_atoms],
        dtype=float,
    )
    centroid = np.mean(reference_coordinates, axis=0)
    transformed_atoms: list[tuple[str, str, np.ndarray]] = []
    for atom in reference_atoms:
        rotated = (atom.coordinates - centroid) @ rotation.T + translation
        transformed_atoms.append(
            (atom.atom_name, atom.element, rotated.astype(float))
        )
    return transformed_atoms


def _write_xyz_cell(
    path: Path,
    molecules: list[list[tuple[str, str, np.ndarray]]],
) -> None:
    atom_records: list[tuple[int, str, np.ndarray]] = []
    for molecule_index, molecule_atoms in enumerate(molecules):
        for atom_name, element, coordinates in molecule_atoms:
            atom_records.append(
                (molecule_index, atom_name, element, coordinates)
            )

    permutation = np.random.default_rng(20260318).permutation(
        len(atom_records)
    )
    lines = [str(len(atom_records)), "Synthetic DMF cell"]
    for atom_index in permutation:
        _molecule_index, _atom_name, element, coordinates = atom_records[
            int(atom_index)
        ]
        lines.append(
            f"{element} {coordinates[0]:.6f} {coordinates[1]:.6f} {coordinates[2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_dmf_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pbc": {
                    "a": 24.0,
                    "b": 24.0,
                    "c": 24.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                    "space_group": "P 1",
                },
                "molecules": [
                    {
                        "name": "DMF",
                        "reference": "dmf",
                        "residue_name": "DMF",
                        "anchors": [
                            {"pair": ["O1", "N1"], "tol": 0.15},
                            {"pair": ["N1", "C1"], "tol": 0.15},
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )


def _signature_from_expected_molecule(
    molecule_atoms: list[tuple[str, str, np.ndarray]],
) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        sorted(
            (
                atom_name,
                round(float(coordinates[0]), 3),
                round(float(coordinates[1]), 3),
                round(float(coordinates[2]), 3),
            )
            for atom_name, _element, coordinates in molecule_atoms
        )
    )


def _signature_from_residue(
    residue_atoms: list[PDBAtom],
) -> tuple[tuple[str, float, float, float], ...]:
    return tuple(
        sorted(
            (
                atom.atom_name,
                round(float(atom.coordinates[0]), 3),
                round(float(atom.coordinates[1]), 3),
                round(float(atom.coordinates[2]), 3),
            )
            for atom in residue_atoms
        )
    )


def test_xyz2pdb_preserves_positions_for_synthetic_dmf_cell(tmp_path):
    reference_atoms = _load_reference_atoms()
    transforms = [
        (_rotation_z(0.0), np.array([5.0, 5.0, 5.0], dtype=float)),
        (
            _rotation_x(35.0) @ _rotation_z(20.0),
            np.array([15.0, 6.0, 9.0], dtype=float),
        ),
        (
            _rotation_y(60.0) @ _rotation_x(25.0),
            np.array([7.0, 15.0, 13.0], dtype=float),
        ),
        (
            _rotation_z(110.0) @ _rotation_y(45.0),
            np.array([17.0, 17.0, 7.0], dtype=float),
        ),
    ]
    expected_molecules = [
        _build_transformed_molecule(
            reference_atoms,
            rotation=rotation,
            translation=translation,
        )
        for rotation, translation in transforms
    ]

    xyz_path = tmp_path / "dmf_cell.xyz"
    _write_xyz_cell(xyz_path, expected_molecules)

    config_path = tmp_path / "dmf_assignments.json"
    _write_dmf_config(config_path)

    workflow = XYZToPDBWorkflow(
        xyz_path,
        config_file=config_path,
        reference_library_dir=default_reference_library_dir(),
    )

    preview = workflow.preview_conversion()
    export = workflow.export_pdbs()
    output_path = export.written_files[0]
    output_structure = PDBStructure.from_file(output_path)

    residues_by_number: dict[int, list[PDBAtom]] = defaultdict(list)
    for atom in output_structure.atoms:
        residues_by_number[atom.residue_number].append(atom)

    expected_signatures = sorted(
        _signature_from_expected_molecule(molecule)
        for molecule in expected_molecules
    )
    actual_signatures = sorted(
        _signature_from_residue(residue_atoms)
        for residue_atoms in residues_by_number.values()
    )

    assert preview.molecule_counts["DMF"] == len(expected_molecules)
    assert preview.residue_counts["DMF"] == len(expected_molecules)
    assert len(output_structure.atoms) == len(reference_atoms) * len(
        expected_molecules
    )
    assert len(residues_by_number) == len(expected_molecules)
    assert expected_signatures == actual_signatures
    assert output_path.read_text().splitlines()[0].startswith("CRYST1")
