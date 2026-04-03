from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from saxshell.xyz2pdb import (
    create_reference_molecule,
    default_reference_library_dir,
    list_reference_library,
)
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import mapping_workflow as mapping_workflow_module
from saxshell.xyz2pdb import workflow as workflow_module
from saxshell.xyz2pdb.mapping_workflow import (
    FreeAtomMappingInput,
    MoleculeMappingInput,
    ReferenceBondToleranceInput,
    XYZToPDBMappingWorkflow,
    reference_bond_tolerances,
)
from saxshell.xyz2pdb.workflow import (
    AnchorPairDefinition,
    MoleculeDefinition,
    XYZAtomRecord,
    XYZFrame,
)


def _write_reference(path: Path, atoms: list[PDBAtom]) -> None:
    PDBStructure(atoms=atoms).write_pdb_file(path)


def _write_xyz(path: Path, atoms: list[tuple[str, float, float, float]]) -> None:
    lines = [str(len(atoms)), path.stem]
    lines.extend(
        f"{element} {x:.3f} {y:.3f} {z:.3f}"
        for element, x, y, z in atoms
    )
    path.write_text("\n".join(lines) + "\n")


def _write_xyz_from_pdb(xyz_path: Path, pdb_path: Path) -> None:
    structure = PDBStructure.from_file(pdb_path)
    _write_xyz(
        xyz_path,
        [
            (
                atom.element,
                float(atom.coordinates[0]),
                float(atom.coordinates[1]),
                float(atom.coordinates[2]),
            )
            for atom in structure.atoms
        ],
    )


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _build_dmso_box_xyz(
    path: Path,
    *,
    molecule_count: int = 4,
    seed: int = 20260401,
) -> tuple[list[tuple[int, ...]], list[np.ndarray], Counter[str]]:
    reference_path = default_reference_library_dir() / "dmso.pdb"
    structure = PDBStructure.from_file(reference_path)
    reference_atoms = list(structure.atoms)
    reference_coordinates = np.array(
        [atom.coordinates for atom in reference_atoms],
        dtype=float,
    )
    sulfur_index = next(
        index
        for index, atom in enumerate(reference_atoms)
        if atom.element == "S"
    )
    sulfur_origin = reference_coordinates[sulfur_index]

    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    while len(centers) < molecule_count:
        candidate = rng.uniform(3.0, 18.0, size=3)
        if all(
            float(np.linalg.norm(candidate - other)) >= 6.5
            for other in centers
        ):
            centers.append(candidate)

    xyz_atoms: list[tuple[str, float, float, float]] = []
    expected_source_blocks: list[tuple[int, ...]] = []
    expected_centroids: list[np.ndarray] = []
    molecule_scales = [1.05]
    molecule_scales.extend(
        1.0 + float(rng.uniform(-0.015, 0.015))
        for _ in range(max(molecule_count - 1, 0))
    )

    for center, scale in zip(centers, molecule_scales):
        rotation = _random_rotation_matrix(rng)
        transformed = (
            ((reference_coordinates - sulfur_origin) * scale) @ rotation.T
        ) + center
        rounded_coordinates = np.array(
            [[round(float(value), 3) for value in row] for row in transformed],
            dtype=float,
        )
        start_index = len(xyz_atoms)
        for atom, coordinates in zip(reference_atoms, rounded_coordinates):
            xyz_atoms.append(
                (
                    atom.element,
                    float(coordinates[0]),
                    float(coordinates[1]),
                    float(coordinates[2]),
                )
            )
        expected_source_blocks.append(
            tuple(range(start_index, start_index + len(reference_atoms)))
        )
        expected_centroids.append(np.mean(rounded_coordinates, axis=0))

    free_atom_records = [
        ("Na", 20.500, 2.500, 2.500),
        ("Na", 2.500, 20.500, 2.500),
        ("Cl", 2.500, 2.500, 20.500),
    ]
    for record in free_atom_records:
        xyz_atoms.append(record)

    _write_xyz(path, xyz_atoms)
    return (
        expected_source_blocks,
        expected_centroids,
        Counter(element for element, *_coords in free_atom_records),
    )


def _residue_centroid(atoms: list[PDBAtom]) -> np.ndarray:
    return np.mean(np.array([atom.coordinates for atom in atoms], dtype=float), axis=0)


def test_native_mapping_workflow_estimates_tests_and_exports_without_json(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0000.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_0001.xyz",
        [("Pb", 0.0, 0.1, 0.0), ("I", 1.1, 0.1, 0.0), ("O", 2.1, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    molecule_inputs = [MoleculeMappingInput(reference_name="pbi", residue_name="PBI")]
    free_atom_inputs = [FreeAtomMappingInput(element="O", residue_name="SOL")]

    estimate = workflow.estimate_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
    )
    assert len(estimate.solutions) == 1
    solution = estimate.solutions[0]
    assert solution.molecule_count_by_residue(estimate.plan) == {"PBI": 1}
    assert solution.free_atom_counts == {"O": 1}
    assert solution.unassigned_counts == {}

    test_result = workflow.test_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
    )
    assert test_result.molecule_counts == {"PBI": 1}
    assert test_result.residue_counts == {"PBI": 1, "SOL": 1}
    assert test_result.unassigned_counts == {}
    assert any("tight pass" in line for line in test_result.console_messages)
    assert any(
        "Backbone PB1-I1: found 1 candidate pair(s)." in line
        for line in test_result.console_messages
    )

    export = workflow.export_with_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
    )
    assert export.output_dir == tmp_path / "xyz2pdb_frames"
    assert [path.name for path in export.written_files] == [
        "frame_0000.pdb",
        "frame_0001.pdb",
    ]
    first_output = export.written_files[0].read_text()
    assert "PBI" in first_output
    assert "SOL" in first_output


def test_export_reuses_first_frame_mapping_template_for_later_frames(
    tmp_path,
    monkeypatch,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0000.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_0001.xyz",
        [("Pb", 0.1, 0.0, 0.0), ("I", 1.1, 0.0, 0.0), ("O", 2.1, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_0002.xyz",
        [("Pb", 0.2, 0.0, 0.0), ("I", 1.2, 0.0, 0.0), ("O", 2.2, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    molecule_inputs = [MoleculeMappingInput(reference_name="pbi", residue_name="PBI")]
    free_atom_inputs = [FreeAtomMappingInput(element="O", residue_name="SOL")]
    progress_events: list[tuple[int, int, str]] = []
    log_messages: list[str] = []
    map_frame_calls = 0
    original_map_frame = workflow._map_frame

    def counting_map_frame(
        frame,
        plan,
        solution,
        mapping_status_callback=None,
        log_callback=None,
        cancel_callback=None,
    ):
        nonlocal map_frame_calls
        map_frame_calls += 1
        return original_map_frame(
            frame,
            plan,
            solution,
            mapping_status_callback=mapping_status_callback,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )

    monkeypatch.setattr(workflow, "_map_frame", counting_map_frame)

    export = workflow.export_with_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
        progress_callback=lambda processed, total, message: progress_events.append(
            (processed, total, message)
        ),
        log_callback=log_messages.append,
    )

    assert map_frame_calls == 1
    assert [path.name for path in export.written_files] == [
        "frame_0000.pdb",
        "frame_0001.pdb",
        "frame_0002.pdb",
    ]
    assert progress_events[0] == (0, 7, "Estimating first-frame mapping...")
    assert progress_events[1] == (
        1,
        7,
        "Validating XYZ atom order [1/3]: frame_0000.xyz (reference)",
    )
    assert progress_events[2] == (
        2,
        7,
        "Validating XYZ atom order [2/3]: frame_0001.xyz",
    )
    assert progress_events[3] == (
        3,
        7,
        "Validating XYZ atom order [3/3]: frame_0002.xyz",
    )
    assert progress_events[4] == (4, 7, "Mapping template from frame_0000.xyz...")
    assert any(
        message == "Template mapping progress in frame_0000.xyz: PBI 1/1"
        for _processed, _total, message in progress_events
    )
    assert progress_events[-1] == (7, 7, "[3/3] Wrote frame_0002.xyz")
    assert any(
        "Validated XYZ atom order across 3 frame(s) against frame_0000.xyz "
        "before template mapping."
        == message
        for message in log_messages
    )
    assert any(
        "First-frame mapping succeeded. Reusing its atom-index template"
        in message
        for message in log_messages
    )
    assert any(
        message == "Template mapping counts: PBI 1/1"
        for message in log_messages
    )
    assert any(
        "PBI: tight pass [pbi]: Backbone PB1-I1: found 1 candidate pair(s)."
        == message
        for message in log_messages
    )
    assert any(
        "Reused the first-frame atom-order mapping for frame_0001.xyz."
        == message
        for message in log_messages
    )
    assert any(
        "Reused the first-frame atom-order mapping for frame_0002.xyz."
        == message
        for message in log_messages
    )


def test_export_uses_natural_frame_order_for_non_padded_indices(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_1002.xyz",
        [("Pb", 0.2, 0.0, 0.0), ("I", 1.2, 0.0, 0.0), ("O", 2.2, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_10000.xyz",
        [("Pb", 0.3, 0.0, 0.0), ("I", 1.3, 0.0, 0.0), ("O", 2.3, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_10001.xyz",
        [("Pb", 0.4, 0.0, 0.0), ("I", 1.4, 0.0, 0.0), ("O", 2.4, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    inspection = workflow.inspect()
    assert [path.name for path in inspection.xyz_files] == [
        "frame_1002.xyz",
        "frame_10000.xyz",
        "frame_10001.xyz",
    ]

    analysis = workflow.analyze_input()
    assert analysis.sample_file.name == "frame_1002.xyz"

    progress_events: list[tuple[int, int, str]] = []
    export = workflow.export_with_mapping(
        molecule_inputs=[
            MoleculeMappingInput(reference_name="pbi", residue_name="PBI")
        ],
        free_atom_inputs=[FreeAtomMappingInput(element="O", residue_name="SOL")],
        progress_callback=lambda processed, total, message: progress_events.append(
            (processed, total, message)
        ),
    )

    assert [path.name for path in export.written_files] == [
        "frame_1002.pdb",
        "frame_10000.pdb",
        "frame_10001.pdb",
    ]
    assert progress_events[1] == (
        1,
        7,
        "Validating XYZ atom order [1/3]: frame_1002.xyz (reference)",
    )
    assert progress_events[2] == (
        2,
        7,
        "Validating XYZ atom order [2/3]: frame_10000.xyz",
    )
    assert progress_events[3] == (
        3,
        7,
        "Validating XYZ atom order [3/3]: frame_10001.xyz",
    )
    assert progress_events[4] == (
        4,
        7,
        "Mapping template from frame_1002.xyz...",
    )
    assert progress_events[-1] == (7, 7, "[3/3] Wrote frame_10001.xyz")


def test_read_xyz_frame_parses_standard_two_line_xyz_header(tmp_path):
    xyz_path = tmp_path / "frame_1002.xyz"
    xyz_path.write_text(
        "\n".join(
            [
                "2",
                " i =     9167, time =     4583.500, E =     -2581.9092625648",
                "C 0.000 0.000 0.000",
                "H 0.000 0.000 1.000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frame = workflow_module.XYZToPDBWorkflow.read_xyz_frame(xyz_path)

    assert frame.comment == (
        " i =     9167, time =     4583.500, E =     -2581.9092625648"
    )
    assert len(frame.atoms) == 2
    assert [atom.element for atom in frame.atoms] == ["C", "H"]


def test_materialized_match_preserves_source_index_order_for_template_reuse(
    tmp_path,
):
    frame_path = tmp_path / "frame_1002.xyz"
    frame = XYZFrame(
        filepath=frame_path,
        comment="frame_1002",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="H",
                coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="C",
                coordinates=np.array([1.0, 0.0, 0.0], dtype=float),
            ),
        ],
    )
    next_frame = XYZFrame(
        filepath=tmp_path / "frame_1003.xyz",
        comment="frame_1003",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="H",
                coordinates=np.array([0.0, 1.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="C",
                coordinates=np.array([1.0, 1.0, 0.0], dtype=float),
            ),
        ],
    )
    reference_atoms = (
        PDBAtom(
            atom_id=1,
            atom_name="C1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[1.0, 0.0, 0.0],
            element="C",
        ),
        PDBAtom(
            atom_id=2,
            atom_name="H1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[0.0, 0.0, 0.0],
            element="H",
        ),
    )
    molecule_definition = MoleculeDefinition(
        name="DMF",
        reference_name="dmf",
        reference_path=default_reference_library_dir() / "dmf.pdb",
        residue_name="DMF",
        reference_atoms=reference_atoms,
        anchors=(
            AnchorPairDefinition(
                atom1_name="C1",
                atom2_name="H1",
                tolerance=0.2,
            ),
        ),
        resolved_anchor_indices=((0, 1, 0.2),),
        preferred_anchor_indices=((0, 1),),
    )
    variant = mapping_workflow_module._ResolvedMoleculeVariant(
        molecule_definition=molecule_definition,
        variant_reference_atoms=reference_atoms,
        variant_bonds=(),
        kept_full_indices=(0, 1),
        missing_full_indices=(),
    )
    match = mapping_workflow_module._VariantMatch(
        variant=variant,
        assignment=(1, 0),
        pass_name="tight",
        fit_rmsd=0.0,
        mean_bond_deviation=0.0,
        transformed_full_coordinates=np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=float,
        ),
    )

    residue, _used_indices, _messages, _warnings, _atom_serial = (
        mapping_workflow_module._materialize_match(
            match=match,
            frame=frame,
            residue_number=1,
            atom_serial=1,
            hydrogen_mode="leave_unassigned",
            used=[False, False],
            match_classification=None,
        )
    )

    assert residue.source_atom_indices == (1, 0)

    applied = mapping_workflow_module._apply_template_mapping(
        next_frame,
        (residue,),
    )

    assert len(applied) == 1
    assert [atom.element for atom in applied[0].atoms] == ["C", "H"]
    assert np.allclose(applied[0].atoms[0].coordinates, [1.0, 1.0, 0.0])
    assert np.allclose(applied[0].atoms[1].coordinates, [0.0, 1.0, 0.0])


def test_export_validates_folder_atom_order_before_template_mapping(
    tmp_path,
    monkeypatch,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_1002.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.0, 0.0, 0.0), ("H", 2.0, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_1003.xyz",
        [("Pb", 0.1, 0.0, 0.0), ("I", 1.1, 0.0, 0.0), ("C", 2.1, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    map_frame_calls = 0
    original_map_frame = workflow._map_frame

    def counting_map_frame(
        frame,
        plan,
        solution,
        mapping_status_callback=None,
        log_callback=None,
        cancel_callback=None,
    ):
        nonlocal map_frame_calls
        map_frame_calls += 1
        return original_map_frame(
            frame,
            plan,
            solution,
            mapping_status_callback=mapping_status_callback,
            log_callback=log_callback,
            cancel_callback=cancel_callback,
        )

    monkeypatch.setattr(workflow, "_map_frame", counting_map_frame)

    with pytest.raises(ValueError) as excinfo:
        workflow.export_with_mapping(
            molecule_inputs=[
                MoleculeMappingInput(reference_name="pbi", residue_name="PBI")
            ],
            free_atom_inputs=[FreeAtomMappingInput(element="H", residue_name="SOL")],
        )

    assert map_frame_calls == 0
    assert "frame_1003.xyz" in str(excinfo.value)
    assert "frame_1002.xyz" in str(excinfo.value)
    assert "expected H, found C" in str(excinfo.value)


def test_preferred_backbone_pairs_do_not_fall_back_to_other_bonds(tmp_path):
    xyz_path = tmp_path / "frame_0000.xyz"
    xyz_path.write_text("0\nframe_0000\n", encoding="utf-8")
    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    molecule = MoleculeDefinition(
        name="DMF",
        reference_name="dmf",
        reference_path=default_reference_library_dir() / "dmf.pdb",
        residue_name="DMF",
        reference_atoms=(
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="N1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="N",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="C1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[2.4, 0.0, 0.0],
                element="C",
            ),
        ),
        anchors=(
            AnchorPairDefinition(atom1_name="O1", atom2_name="N1", tolerance=0.2),
            AnchorPairDefinition(atom1_name="N1", atom2_name="C1", tolerance=0.2),
        ),
        resolved_anchor_indices=((0, 1, 0.2), (1, 2, 0.2)),
        preferred_anchor_indices=((0, 1),),
    )

    assert workflow._anchor_search_stages(molecule) == (((0, 1, 0.2),),)


def test_export_assertion_mode_writes_molecule_files_and_report(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0000.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_0001.xyz",
        [("Pb", 0.1, 0.0, 0.0), ("I", 1.1, 0.0, 0.0), ("O", 2.1, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    output_dir = tmp_path / "mapped_pdb"
    progress_events: list[tuple[int, int, str]] = []
    log_messages: list[str] = []

    export = workflow.export_with_mapping(
        molecule_inputs=[
            MoleculeMappingInput(reference_name="pbi", residue_name="PBI")
        ],
        free_atom_inputs=[FreeAtomMappingInput(element="O", residue_name="SOL")],
        output_dir=output_dir,
        assert_molecule_shapes=True,
        progress_callback=lambda processed, total, message: progress_events.append(
            (processed, total, message)
        ),
        log_callback=log_messages.append,
    )

    assert export.progress_total_steps == 6
    assert export.assertion_result is not None
    assert export.assertion_result.passed is True
    assert export.assertion_result.total_molecules == 2
    assert export.assertion_result.molecule_dir == (
        output_dir / "assertion_molecules"
    )
    assert export.assertion_result.report_file.exists()
    assert len(export.assertion_result.reference_update_candidates) == 1
    residue_summary = export.assertion_result.residue_summaries[0]
    assert residue_summary.residue_name == "PBI"
    assert residue_summary.molecule_count == 2
    assert residue_summary.common_atom_count == 2
    assert residue_summary.distance_pair_count == 1
    candidate = export.assertion_result.reference_update_candidates[0]
    assert candidate.reference_name == "pbi"
    assert candidate.reference_path == refs_dir / "pbi.pdb"
    assert candidate.average_structure_file.exists()
    averaged_structure = PDBStructure.from_file(candidate.average_structure_file)
    averaged_distance = float(
        np.linalg.norm(
            averaged_structure.atoms[1].coordinates
            - averaged_structure.atoms[0].coordinates
        )
    )
    assert averaged_distance == pytest.approx(1.0, abs=1.0e-6)

    residue_dir = export.assertion_result.molecule_dir / "PBI"
    molecule_files = sorted(residue_dir.glob("*.pdb"))
    assert [path.name for path in molecule_files] == [
        "frame_0000__PBI_0001.pdb",
        "frame_0001__PBI_0001.pdb",
    ]
    report_text = export.assertion_result.report_file.read_text(encoding="utf-8")
    assert "xyz2pdb assertion mode report" in report_text
    assert "PASS PBI" in report_text
    assert progress_events[-1] == (
        6,
        6,
        "Analyzing molecule distance distributions...",
    )
    assert any(
        "Assertion mode wrote 1 molecule file(s) for frame_0000.xyz."
        in message
        for message in log_messages
    )
    assert any(
        "Assertion report written to" in message
        for message in log_messages
    )


def test_export_assertion_mode_warns_when_molecules_drift_from_reference(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "pbi.pdb",
        [
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
        ],
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0000.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.0, 0.0, 0.0), ("O", 2.0, 0.0, 0.0)],
    )
    _write_xyz(
        frames_dir / "frame_0001.xyz",
        [("Pb", 0.0, 0.0, 0.0), ("I", 1.8, 0.0, 0.0), ("O", 2.8, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )

    export = workflow.export_with_mapping(
        molecule_inputs=[
            MoleculeMappingInput(reference_name="pbi", residue_name="PBI")
        ],
        free_atom_inputs=[FreeAtomMappingInput(element="O", residue_name="SOL")],
        output_dir=tmp_path / "mapped_pdb",
        assert_molecule_shapes=True,
    )

    assert export.assertion_result is not None
    assert export.assertion_result.passed is False
    assert export.assertion_result.total_molecules == 2
    assert export.assertion_result.reference_update_candidates == ()
    assert any(
        "vary noticeably from the reference" in warning
        for warning in export.assertion_result.warnings
    )
    residue_summary = export.assertion_result.residue_summaries[0]
    assert residue_summary.residue_name == "PBI"
    assert residue_summary.passed is False
    assert residue_summary.max_distribution_rmsd == pytest.approx(0.8)
    assert residue_summary.max_max_distance_delta == pytest.approx(0.8)


def test_create_reference_molecule_writes_backbone_metadata(tmp_path):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    source_path = tmp_path / "pbi_source.pdb"
    _write_reference(
        source_path,
        [
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
        ],
    )

    result = create_reference_molecule(
        source_path,
        reference_name="pbi",
        residue_name="PBI",
        library_dir=refs_dir,
    )

    assert result.backbone_pairs == (("PB1", "I1"),)
    metadata = json.loads((refs_dir / "pbi.json").read_text())
    assert metadata["backbone_pairs"] == [["PB1", "I1"]]
    entry = list_reference_library(refs_dir)[0]
    assert entry.backbone_pairs == (("PB1", "I1"),)


def test_create_reference_molecule_uses_explicit_backbone_pair_metadata(tmp_path):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    source_path = tmp_path / "ocn_source.pdb"
    _write_reference(
        source_path,
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="C1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="N1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[2.4, 0.0, 0.0],
                element="N",
            ),
        ],
    )

    result = create_reference_molecule(
        source_path,
        reference_name="ocn",
        residue_name="OCN",
        library_dir=refs_dir,
        backbone_pairs=(("O1", "N1"),),
    )

    assert result.backbone_pairs == (("O1", "N1"),)
    metadata = json.loads((refs_dir / "ocn.json").read_text(encoding="utf-8"))
    assert metadata["backbone_pairs"] == [["O1", "N1"]]


def test_mapping_uses_preferred_backbone_pairs_from_reference_metadata(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "ocn.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="C1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="N1",
                residue_name="OCN",
                residue_number=1,
                coordinates=[2.4, 0.0, 0.0],
                element="N",
            ),
        ],
    )
    (refs_dir / "ocn.json").write_text(
        json.dumps(
            {
                "residue_name": "OCN",
                "backbone_pairs": [["N1", "C1"], ["O1", "C1"]],
            }
        )
        + "\n"
    )
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_xyz(
        frames_dir / "frame_0000.xyz",
        [("O", 0.0, 0.0, 0.0), ("C", 1.2, 0.0, 0.0), ("N", 2.4, 0.0, 0.0)],
    )

    workflow = XYZToPDBMappingWorkflow(
        frames_dir,
        reference_library_dir=refs_dir,
    )
    test_result = workflow.test_mapping(
        molecule_inputs=[MoleculeMappingInput(reference_name="ocn", residue_name="OCN")],
        free_atom_inputs=[],
    )

    backbone_messages = [
        line for line in test_result.console_messages if "Backbone" in line
    ]
    assert backbone_messages
    assert "Backbone N1-C1: found 1 candidate pair(s)." in backbone_messages[0]


def test_bundled_reference_backbone_pairs_match_single_requested_pairs():
    entries = {
        entry.name: entry
        for entry in list_reference_library(default_reference_library_dir())
    }

    assert entries["dmso"].backbone_pairs == (("O1", "S1"),)
    assert entries["dmf"].backbone_pairs == (("O1", "N1"),)
    assert entries["dmf_md"].backbone_pairs == (("O1", "N1"),)
    assert entries["ma"].backbone_pairs == (("N1", "C1"),)


def test_dmf_single_backbone_pair_scans_fewer_pairs_than_three_pair_setup(
    tmp_path,
):
    default_refs = default_reference_library_dir()
    xyz_path = tmp_path / "frame_0000.xyz"
    _write_xyz_from_pdb(xyz_path, default_refs / "dmf.pdb")

    def run_case(
        reference_dir: Path,
        backbone_pairs: list[list[str]],
    ) -> tuple[int, list[str]]:
        reference_dir.mkdir()
        (reference_dir / "dmf.pdb").write_text(
            (default_refs / "dmf.pdb").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (reference_dir / "dmf.json").write_text(
            json.dumps(
                {
                    "residue_name": "DMF",
                    "backbone_pairs": backbone_pairs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        workflow = XYZToPDBMappingWorkflow(
            xyz_path,
            reference_library_dir=reference_dir,
        )
        result = workflow.test_mapping(
            molecule_inputs=[
                MoleculeMappingInput(reference_name="dmf", residue_name="DMF")
            ],
            free_atom_inputs=[],
        )
        found_messages = [
            line
            for line in result.console_messages
            if "Backbone " in line and "found" in line
        ]
        assert result.molecule_counts == {"DMF": 1}
        assert not any(
            "falling back to" in line.lower()
            for line in result.console_messages
        )
        return len(found_messages), found_messages

    single_count, single_messages = run_case(
        tmp_path / "refs_single",
        [["O1", "N1"]],
    )
    three_count, three_messages = run_case(
        tmp_path / "refs_three",
        [["O1", "N1"], ["N1", "C2"], ["N1", "C3"]],
    )

    assert single_count == 1
    assert three_count == 3
    assert single_count < three_count
    assert "Backbone O1-N1: found 1 candidate pair(s)." in single_messages[0]
    assert any(
        "Backbone N1-C2:" in message
        for message in three_messages
    )
    assert any(
        "Backbone N1-C3:" in message
        for message in three_messages
    )


def test_dmso_box_roundtrip_preserves_molecule_assignments_and_positions(
    tmp_path,
    monkeypatch,
):
    xyz_path = tmp_path / "dmso_box.xyz"
    (
        expected_source_blocks,
        expected_centroids,
        expected_free_atom_counts,
    ) = _build_dmso_box_xyz(xyz_path)

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    find_best_match_calls = 0
    original_find_best_match = workflow._find_best_match

    def counting_find_best_match(
        frame,
        used,
        molecule,
        *,
        frame_search_cache=None,
        search_status_callback=None,
        cancel_callback=None,
    ):
        nonlocal find_best_match_calls
        find_best_match_calls += 1
        return original_find_best_match(
            frame,
            used,
            molecule,
            frame_search_cache=frame_search_cache,
            search_status_callback=search_status_callback,
            cancel_callback=cancel_callback,
        )

    monkeypatch.setattr(workflow, "_find_best_match", counting_find_best_match)

    custom_tolerances = tuple(
        ReferenceBondToleranceInput(
            atom1_name=item.atom1_name,
            atom2_name=item.atom2_name,
            tolerance=8.0,
        )
        for item in reference_bond_tolerances(
            "dmso",
            library_dir=default_reference_library_dir(),
        )
    )
    molecule_inputs = [
        MoleculeMappingInput(
            reference_name="dmso",
            residue_name="DMS",
            bond_tolerances=custom_tolerances,
            tight_pass_scale=0.5,
            relaxed_pass_scale=1.0,
        )
    ]
    free_atom_inputs = [
        FreeAtomMappingInput(element="Na", residue_name="SOD"),
        FreeAtomMappingInput(element="Cl", residue_name="CLA"),
    ]

    test_result = workflow.test_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
    )

    assert find_best_match_calls > 0
    assert test_result.molecule_counts == {"DMS": 4}
    assert test_result.free_atom_counts == dict(sorted(expected_free_atom_counts.items()))
    assert test_result.unassigned_counts == {}
    assert any(
        "relaxed full-hydrogen pass" in warning
        for warning in test_result.warnings
    )

    matched_source_blocks = {
        tuple(sorted(int(index) for index in residue.source_atom_indices))
        for residue in test_result.residues
        if residue.residue_name == "DMS"
    }
    assert matched_source_blocks == set(expected_source_blocks)

    export = workflow.export_with_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=free_atom_inputs,
        output_dir=tmp_path / "mapped_pdb",
    )

    written_structure = PDBStructure.from_file(export.written_files[0])
    residue_atoms: dict[int, list[PDBAtom]] = defaultdict(list)
    free_atom_residue_counts: Counter[str] = Counter()
    for atom in written_structure.atoms:
        if atom.residue_name == "DMS":
            residue_atoms[int(atom.residue_number)].append(atom)
        elif atom.residue_name in {"SOD", "CLA"}:
            free_atom_residue_counts[atom.residue_name] += 1

    assert len(residue_atoms) == 4
    assert all(len(atoms) == 10 for atoms in residue_atoms.values())
    assert free_atom_residue_counts == {"CLA": 1, "SOD": 2}

    written_centroids = np.array(
        [_residue_centroid(atoms) for atoms in residue_atoms.values()],
        dtype=float,
    )
    expected_centroid_array = np.array(expected_centroids, dtype=float)
    distance_matrix = np.linalg.norm(
        written_centroids[:, None, :] - expected_centroid_array[None, :, :],
        axis=2,
    )
    row_indices, column_indices = linear_sum_assignment(distance_matrix)
    matched_distances = distance_matrix[row_indices, column_indices]
    assert float(np.max(matched_distances)) <= 1.0e-3


def test_candidate_backbone_pairs_report_dynamic_discovery_progress(tmp_path):
    xyz_path = tmp_path / "frame_0000.xyz"
    xyz_path.write_text("0\nframe_0000\n", encoding="utf-8")
    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    frame = XYZFrame(
        filepath=xyz_path,
        comment="frame_0000",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="O",
                coordinates=np.array([0.0, 0.00, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="O",
                coordinates=np.array([0.0, 0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=3,
                element="O",
                coordinates=np.array([0.0, -0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=4,
                element="O",
                coordinates=np.array([0.0, 0.16, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=5,
                element="N",
                coordinates=np.array([1.20, 0.00, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=6,
                element="N",
                coordinates=np.array([1.20, 0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=7,
                element="N",
                coordinates=np.array([1.20, -0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=8,
                element="N",
                coordinates=np.array([1.20, 0.16, 0.0], dtype=float),
            ),
        ],
    )
    source_indices_by_element = workflow._unused_source_indices_by_element(
        frame,
        [False] * len(frame.atoms),
    )
    source_coordinates_by_element = {
        element: np.array(
            [frame.atoms[index].coordinates for index in indices],
            dtype=float,
        )
        for element, indices in source_indices_by_element.items()
    }
    discovered_counts: list[int] = []

    candidate_pairs = workflow._candidate_backbone_pairs(
        frame,
        [False] * len(frame.atoms),
        source_indices_by_element=source_indices_by_element,
        source_coordinates_by_element=source_coordinates_by_element,
        anchor_atom1=PDBAtom(
            atom_id=1,
            atom_name="O1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[0.0, 0.0, 0.0],
            element="O",
        ),
        anchor_atom2=PDBAtom(
            atom_id=2,
            atom_name="N1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[1.2, 0.0, 0.0],
            element="N",
        ),
        reference_length=1.2,
        tolerance=0.25,
        discovery_callback=discovered_counts.append,
    )

    assert len(candidate_pairs) == 16
    assert discovered_counts
    assert discovered_counts[0] >= 10
    assert discovered_counts[-1] < len(candidate_pairs)


def test_find_best_match_cycles_backbone_candidates_and_reports_summary(
    tmp_path,
    monkeypatch,
):
    xyz_path = tmp_path / "frame_0000.xyz"
    xyz_path.write_text("0\nframe_0000\n", encoding="utf-8")
    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    frame = XYZFrame(
        filepath=xyz_path,
        comment="frame_0000",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="O",
                coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="N",
                coordinates=np.array([1.2, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=3,
                element="O",
                coordinates=np.array([3.0, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=4,
                element="N",
                coordinates=np.array([4.2, 0.0, 0.0], dtype=float),
            ),
        ],
    )
    molecule = MoleculeDefinition(
        name="DMF",
        reference_name="dmf",
        reference_path=default_reference_library_dir() / "dmf.pdb",
        residue_name="DMF",
        reference_atoms=(
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="N1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="N",
            ),
        ),
        anchors=(
            AnchorPairDefinition(
                atom1_name="O1",
                atom2_name="N1",
                tolerance=0.2,
            ),
        ),
        resolved_anchor_indices=((0, 1, 0.2),),
        preferred_anchor_indices=((0, 1),),
    )

    monkeypatch.setattr(
        workflow_module,
        "_BACKBONE_ROTATION_BATCH_SIZE",
        1,
    )
    monkeypatch.setattr(
        workflow_module,
        "_BACKBONE_ROTATION_SAMPLE_COUNTS",
        (1,),
    )

    def fake_candidate_backbone_pairs(*args, discovery_callback=None, **kwargs):
        if discovery_callback is not None:
            discovery_callback(2)
        return ((0, 1), (2, 3))

    def fake_backbone_candidate_states(*args, **kwargs):
        return (
            workflow_module._BackboneCandidateState(
                candidate_index=1,
                source_index1=0,
                source_index2=1,
                base_transformed_coordinates=np.zeros((2, 3), dtype=float),
                fixed_assignment={0: 0, 1: 1},
                axis_origin=np.zeros(3, dtype=float),
                axis_direction=np.array([1.0, 0.0, 0.0], dtype=float),
                pending_angles=deque((0.0,)),
                rotation_level=1,
            ),
            workflow_module._BackboneCandidateState(
                candidate_index=2,
                source_index1=2,
                source_index2=3,
                base_transformed_coordinates=np.zeros((2, 3), dtype=float),
                fixed_assignment={0: 2, 1: 3},
                axis_origin=np.zeros(3, dtype=float),
                axis_direction=np.array([1.0, 0.0, 0.0], dtype=float),
                pending_angles=deque((0.0,)),
                rotation_level=1,
            ),
        )

    assignment_order: list[int] = []

    def fake_assign_reference_atoms(
        _frame,
        _used,
        *,
        reference_atoms,
        target_coordinates,
        fixed_assignment,
        max_assignment_distance,
        source_indices_by_element=None,
        source_coordinates_by_element=None,
    ):
        del (
            reference_atoms,
            target_coordinates,
            max_assignment_distance,
            source_indices_by_element,
            source_coordinates_by_element,
        )
        assignment_order.append(int(fixed_assignment[0] // 2 + 1))
        if fixed_assignment[0] == 2:
            return (2, 3), 0.01
        return None, 0.0

    monkeypatch.setattr(
        workflow,
        "_candidate_backbone_pairs",
        fake_candidate_backbone_pairs,
    )
    monkeypatch.setattr(
        workflow,
        "_backbone_candidate_states",
        fake_backbone_candidate_states,
    )
    monkeypatch.setattr(
        workflow,
        "_assign_reference_atoms",
        fake_assign_reference_atoms,
    )

    search_messages: list[str] = []
    assignment = workflow._find_best_match(
        frame,
        [False] * len(frame.atoms),
        molecule,
        search_status_callback=search_messages.append,
    )

    assert assignment == (2, 3)
    assert assignment_order[:2] == [1, 2]
    assert any(
        "Backbone O1-N1: found 2 candidate pair(s) so far."
        == message
        for message in search_messages
    )
    assert any(
        "Backbone O1-N1: 2 candidate pair(s), 2 unique fit(s), 1 successful fit(s), 1 rejected fit(s), success rate 50.0%, selected candidate 2/2."
        == message
        for message in search_messages
    )


def test_candidate_backbone_pairs_reuse_cached_frame_search_entries(
    tmp_path,
    monkeypatch,
):
    xyz_path = tmp_path / "frame_0000.xyz"
    xyz_path.write_text("0\nframe_0000\n", encoding="utf-8")
    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    frame = XYZFrame(
        filepath=xyz_path,
        comment="frame_0000",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="O",
                coordinates=np.array([0.0, 0.00, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="O",
                coordinates=np.array([0.0, 0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=3,
                element="O",
                coordinates=np.array([0.0, -0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=4,
                element="O",
                coordinates=np.array([0.0, 0.16, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=5,
                element="N",
                coordinates=np.array([1.20, 0.00, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=6,
                element="N",
                coordinates=np.array([1.20, 0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=7,
                element="N",
                coordinates=np.array([1.20, -0.08, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=8,
                element="N",
                coordinates=np.array([1.20, 0.16, 0.0], dtype=float),
            ),
        ],
    )
    frame_search_cache = workflow._build_frame_search_cache(frame)
    discovered_counts: list[int] = []
    original_norm = workflow_module.np.linalg.norm
    norm_call_count = 0

    def counting_norm(*args, **kwargs):
        nonlocal norm_call_count
        norm_call_count += 1
        return original_norm(*args, **kwargs)

    monkeypatch.setattr(workflow_module.np.linalg, "norm", counting_norm)

    first_candidate_pairs = workflow._candidate_backbone_pairs(
        frame,
        [False] * len(frame.atoms),
        source_indices_by_element=frame_search_cache.source_indices_by_element,
        source_coordinates_by_element=(
            frame_search_cache.source_coordinates_by_element
        ),
        anchor_atom1=PDBAtom(
            atom_id=1,
            atom_name="O1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[0.0, 0.0, 0.0],
            element="O",
        ),
        anchor_atom2=PDBAtom(
            atom_id=2,
            atom_name="N1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[1.2, 0.0, 0.0],
            element="N",
        ),
        reference_length=1.2,
        tolerance=0.25,
        discovery_callback=discovered_counts.append,
        frame_search_cache=frame_search_cache,
    )

    assert first_candidate_pairs
    assert norm_call_count > 0
    assert discovered_counts

    norm_call_count = 0
    used = [False] * len(frame.atoms)
    used[first_candidate_pairs[0][0]] = True
    used[first_candidate_pairs[0][1]] = True
    discovered_counts.clear()

    cached_candidate_pairs = workflow._candidate_backbone_pairs(
        frame,
        used,
        source_indices_by_element=frame_search_cache.source_indices_by_element,
        source_coordinates_by_element=(
            frame_search_cache.source_coordinates_by_element
        ),
        anchor_atom1=PDBAtom(
            atom_id=1,
            atom_name="O1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[0.0, 0.0, 0.0],
            element="O",
        ),
        anchor_atom2=PDBAtom(
            atom_id=2,
            atom_name="N1",
            residue_name="DMF",
            residue_number=1,
            coordinates=[1.2, 0.0, 0.0],
            element="N",
        ),
        reference_length=1.2,
        tolerance=0.25,
        discovery_callback=discovered_counts.append,
        frame_search_cache=frame_search_cache,
    )

    assert norm_call_count == 0
    assert not discovered_counts
    assert len(cached_candidate_pairs) < len(first_candidate_pairs)
    assert all(
        not used[source_index1] and not used[source_index2]
        for source_index1, source_index2 in cached_candidate_pairs
    )


def test_find_best_match_reports_cached_candidate_pair_reuse(
    tmp_path,
    monkeypatch,
):
    xyz_path = tmp_path / "frame_0000.xyz"
    xyz_path.write_text("0\nframe_0000\n", encoding="utf-8")
    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=default_reference_library_dir(),
    )
    frame = XYZFrame(
        filepath=xyz_path,
        comment="frame_0000",
        atoms=[
            XYZAtomRecord(
                atom_id=1,
                element="O",
                coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=2,
                element="N",
                coordinates=np.array([1.2, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=3,
                element="O",
                coordinates=np.array([3.0, 0.0, 0.0], dtype=float),
            ),
            XYZAtomRecord(
                atom_id=4,
                element="N",
                coordinates=np.array([4.2, 0.0, 0.0], dtype=float),
            ),
        ],
    )
    molecule = MoleculeDefinition(
        name="DMF",
        reference_name="dmf",
        reference_path=default_reference_library_dir() / "dmf.pdb",
        residue_name="DMF",
        reference_atoms=(
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="N1",
                residue_name="DMF",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="N",
            ),
        ),
        anchors=(
            AnchorPairDefinition(
                atom1_name="O1",
                atom2_name="N1",
                tolerance=0.2,
            ),
        ),
        resolved_anchor_indices=((0, 1, 0.2),),
        preferred_anchor_indices=((0, 1),),
    )
    frame_search_cache = workflow._build_frame_search_cache(frame)

    monkeypatch.setattr(
        workflow_module,
        "_BACKBONE_ROTATION_BATCH_SIZE",
        1,
    )
    monkeypatch.setattr(
        workflow_module,
        "_BACKBONE_ROTATION_SAMPLE_COUNTS",
        (1,),
    )

    first_search_messages: list[str] = []
    first_assignment = workflow._find_best_match(
        frame,
        [False] * len(frame.atoms),
        molecule,
        search_status_callback=first_search_messages.append,
        frame_search_cache=frame_search_cache,
    )

    assert first_assignment == (0, 1)
    assert any(
        "Backbone O1-N1: found 2 candidate pair(s)." == message
        for message in first_search_messages
    )

    second_search_messages: list[str] = []
    second_assignment = workflow._find_best_match(
        frame,
        [True, True, False, False],
        molecule,
        search_status_callback=second_search_messages.append,
        frame_search_cache=frame_search_cache,
    )

    assert second_assignment == (2, 3)
    assert any(
        "Backbone O1-N1: reusing 2 cached candidate pair(s); 1 remain after filtering matched atoms."
        == message
        for message in second_search_messages
    )


def test_reference_bond_tolerances_use_percentages_and_resolve_per_bond(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "co.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="C1",
                residue_name="COA",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="O1",
                residue_name="COA",
                residue_number=1,
                coordinates=[1.5, 0.0, 0.0],
                element="O",
            ),
        ],
    )
    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.5, 0.0, 0.0),
        ],
    )

    tolerance_inputs = reference_bond_tolerances("co", library_dir=refs_dir)
    assert len(tolerance_inputs) == 1
    assert tolerance_inputs[0].tolerance == pytest.approx(12.0)

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )
    estimate = workflow.estimate_mapping(
        molecule_inputs=[
            MoleculeMappingInput(
                reference_name="co",
                residue_name="COA",
                bond_tolerances=tolerance_inputs,
            )
        ],
        free_atom_inputs=[],
    )

    assert estimate.plan.molecules[0].bonds[0].tolerance == pytest.approx(0.18)


def test_estimate_mapping_returns_multiple_complete_solutions_when_ambiguous(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "co.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="C1",
                residue_name="COA",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="O1",
                residue_name="COA",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="O",
            ),
        ],
    )
    _write_reference(
        refs_dir / "c2o2.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="C1",
                residue_name="DIO",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="O1",
                residue_name="DIO",
                residue_number=1,
                coordinates=[1.2, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="C2",
                residue_name="DIO",
                residue_number=1,
                coordinates=[2.4, 0.0, 0.0],
                element="C",
            ),
            PDBAtom(
                atom_id=4,
                atom_name="O2",
                residue_name="DIO",
                residue_number=1,
                coordinates=[3.6, 0.0, 0.0],
                element="O",
            ),
        ],
    )

    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("C", 0.0, 0.0, 0.0),
            ("O", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
        ],
    )

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )
    estimate = workflow.estimate_mapping(
        molecule_inputs=[
            MoleculeMappingInput(reference_name="co", residue_name="COA"),
            MoleculeMappingInput(reference_name="c2o2", residue_name="DIO"),
        ],
        free_atom_inputs=[],
    )

    assert len(estimate.solutions) == 2
    assert all(solution.is_complete for solution in estimate.solutions)
    summaries = [
        solution.molecule_count_by_residue(estimate.plan)
        for solution in estimate.solutions
    ]
    assert {"COA": 2} in summaries
    assert {"DIO": 1} in summaries
    assert estimate.warnings


def test_missing_hydrogen_modes_can_leave_assign_or_restore_hydrogen(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "oh.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="H1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="H",
            ),
        ],
    )
    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("O", 0.0, 0.0, 0.0),
            ("H", 3.0, 0.0, 0.0),
        ],
    )

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )
    molecule_inputs = [
        MoleculeMappingInput(
            reference_name="oh",
            residue_name="HOH",
            max_missing_hydrogens=1,
        )
    ]

    leave_result = workflow.test_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=[],
        hydrogen_mode="leave_unassigned",
    )
    assert leave_result.molecule_counts == {"HOH": 1}
    assert leave_result.unassigned_counts == {"H": 1}
    assert len(leave_result.residues[0].atoms) == 1

    assign_result = workflow.test_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=[],
        hydrogen_mode="assign_orphaned",
    )
    assert assign_result.unassigned_counts == {}
    assert len(assign_result.residues[0].atoms) == 2
    assert round(float(assign_result.residues[0].atoms[1].coordinates[0]), 3) == 3.0

    restore_result = workflow.test_mapping(
        molecule_inputs=molecule_inputs,
        free_atom_inputs=[],
        hydrogen_mode="restore_missing",
    )
    assert restore_result.unassigned_counts == {}
    assert len(restore_result.residues[0].atoms) == 2
    assert round(float(restore_result.residues[0].atoms[1].coordinates[0]), 3) == 1.0


def test_default_missing_hydrogen_limit_does_not_assume_deprotonation(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "oh.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="H1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="H",
            ),
        ],
    )
    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("O", 0.0, 0.0, 0.0),
            ("H", 3.0, 0.0, 0.0),
        ],
    )

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )

    result = workflow.test_mapping(
        molecule_inputs=[MoleculeMappingInput(reference_name="oh", residue_name="HOH")],
        free_atom_inputs=[],
        hydrogen_mode="leave_unassigned",
    )

    assert result.molecule_counts == {}
    assert result.unassigned_counts == {"H": 1, "O": 1}
    assert any(
        "keeping 0/1 matched molecule(s) and terminating that search"
        in warning
        for warning in result.warnings
    )


def test_relaxed_full_hydrogen_match_is_classified_as_tolerance_issue(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "oh.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="H1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="H",
            ),
        ],
    )
    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("O", 0.0, 0.0, 0.0),
            ("H", 1.22, 0.0, 0.0),
        ],
    )

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )

    result = workflow.test_mapping(
        molecule_inputs=[
            MoleculeMappingInput(
                reference_name="oh",
                residue_name="HOH",
                max_missing_hydrogens=1,
            )
        ],
        free_atom_inputs=[],
        hydrogen_mode="leave_unassigned",
    )

    assert result.molecule_counts == {"HOH": 1}
    assert result.unassigned_counts == {}
    assert len(result.residues[0].atoms) == 2
    assert any(
        "trying relaxed full-hydrogen pass" in line
        for line in result.console_messages
    )
    assert any(
        "relaxed full-hydrogen pass" in warning for warning in result.warnings
    )
    assert not any(
        "consistent with deprotonation" in warning
        for warning in result.warnings
    )


def test_missing_hydrogen_match_is_classified_as_deprotonation(
    tmp_path,
):
    refs_dir = tmp_path / "references"
    refs_dir.mkdir()
    _write_reference(
        refs_dir / "oh.pdb",
        [
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[0.0, 0.0, 0.0],
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="H1",
                residue_name="HOH",
                residue_number=1,
                coordinates=[1.0, 0.0, 0.0],
                element="H",
            ),
        ],
    )
    xyz_path = tmp_path / "input.xyz"
    _write_xyz(
        xyz_path,
        [
            ("O", 0.0, 0.0, 0.0),
            ("H", 3.0, 0.0, 0.0),
        ],
    )

    workflow = XYZToPDBMappingWorkflow(
        xyz_path,
        reference_library_dir=refs_dir,
    )

    result = workflow.test_mapping(
        molecule_inputs=[
            MoleculeMappingInput(
                reference_name="oh",
                residue_name="HOH",
                max_missing_hydrogens=1,
            )
        ],
        free_atom_inputs=[],
        hydrogen_mode="leave_unassigned",
    )

    assert result.molecule_counts == {"HOH": 1}
    assert result.unassigned_counts == {"H": 1}
    assert len(result.residues[0].atoms) == 1
    assert any(
        "consistent with deprotonation" in warning
        for warning in result.warnings
    )
