from __future__ import annotations

import json

import pytest

import saxshell.cluster.clusternetwork as cluster_module
from saxshell.cluster import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_BRUTEFORCE,
    SEARCH_MODE_KDTREE,
    SEARCH_MODE_VECTORIZED,
    ClusterNetwork,
    ExtractedFrameFolderClusterAnalyzer,
    TrajectoryClusterAnalyzer,
    XYZClusterNetwork,
    XYZStructure,
    detect_frame_folder_mode,
)
from saxshell.structure import PDBStructure
from saxshell.structure.pdbhandlerplus import AtomPlus, PDBHandlerPlus

ATOM_TYPE_DEFINITIONS = {
    "node": [("Pb", "SOL")],
    "linker": [("I", "SOL")],
    "shell": [("O", "WAT")],
}

PAIR_CUTOFFS = {
    ("Pb", "I"): {0: 1.7},
    ("Pb", "O"): {1: 1.3},
}
XYZ_ATOM_TYPE_DEFINITIONS = {
    "node": [("Pb", None)],
    "linker": [("I", None)],
    "shell": [("O", None)],
}


def _pdb_atom_line(
    atom_id: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {atom_id:5d} {atom_name:<4} {residue_name:>3} X"
        f"{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {element:>2}\n"
    )


def _connected_frame_lines() -> list[str]:
    return [
        "MODEL        1\n",
        _pdb_atom_line(1, "PB1", "SOL", 1, 0.0, 0.0, 0.0, "Pb"),
        _pdb_atom_line(2, "I1", "SOL", 1, 1.0, 0.0, 0.0, "I"),
        _pdb_atom_line(3, "PB2", "SOL", 2, 2.0, 0.0, 0.0, "Pb"),
        _pdb_atom_line(4, "O1", "WAT", 10, 0.2, 1.0, 0.0, "O"),
        _pdb_atom_line(5, "H1", "WAT", 10, 0.2, 1.7, 0.0, "H"),
        "ENDMDL\n",
    ]


def _disconnected_frame_lines() -> list[str]:
    return [
        "MODEL        2\n",
        _pdb_atom_line(1, "PB1", "SOL", 1, 0.0, 0.0, 0.0, "Pb"),
        _pdb_atom_line(2, "I1", "SOL", 1, 5.0, 0.0, 0.0, "I"),
        _pdb_atom_line(3, "PB2", "SOL", 2, 10.0, 0.0, 0.0, "Pb"),
        _pdb_atom_line(4, "O1", "WAT", 10, 0.2, 1.0, 0.0, "O"),
        _pdb_atom_line(5, "H1", "WAT", 10, 0.2, 1.7, 0.0, "H"),
        "ENDMDL\n",
    ]


def _connected_xyz_lines() -> list[str]:
    return [
        "5\n",
        "frame_0000\n",
        "Pb 0.0 0.0 0.0\n",
        "I 1.0 0.0 0.0\n",
        "Pb 2.0 0.0 0.0\n",
        "O 0.2 1.0 0.0\n",
        "H 0.2 1.7 0.0\n",
    ]


def _disconnected_xyz_lines() -> list[str]:
    return [
        "5\n",
        "frame_0001\n",
        "Pb 0.0 0.0 0.0\n",
        "I 5.0 0.0 0.0\n",
        "Pb 10.0 0.0 0.0\n",
        "O 0.2 1.0 0.0\n",
        "H 0.2 1.7 0.0\n",
    ]


def _relative_names(paths):
    return sorted(str(path.relative_to(paths[0].parents[1])) for path in paths)


def _cluster_signatures(clusters):
    return [
        (
            cluster.node_atom_ids,
            cluster.linker_atom_ids,
            cluster.shell_atom_ids,
            cluster.stoichiometry,
        )
        for cluster in clusters
    ]


def test_pdb_structure_from_lines_assigns_atom_types():
    structure = PDBStructure.from_lines(
        _connected_frame_lines(),
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        source_name="frame_0000",
    )

    assert structure.source_name == "frame_0000"
    assert [atom.atom_type for atom in structure.atoms[:4]] == [
        "node",
        "linker",
        "node",
        "shell",
    ]
    assert AtomPlus.__name__ == "PDBAtom"
    assert PDBHandlerPlus.__name__ == "PDBStructure"


def test_cluster_network_finds_solute_cluster_and_shell():
    structure = PDBStructure.from_lines(
        _connected_frame_lines(),
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
    )
    network = ClusterNetwork(
        pdb_structure=structure,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    clusters = network.find_clusters(shell_levels=(1,))

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.node_atom_ids == (1, 3)
    assert cluster.linker_atom_ids == (2,)
    assert cluster.shell_atom_ids == (4,)
    assert cluster.stoichiometry == {"Pb": 2, "I": 1}


def test_cluster_network_can_include_shell_atoms_in_stoichiometry():
    structure = PDBStructure.from_lines(
        _connected_frame_lines(),
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
    )
    network = ClusterNetwork(
        pdb_structure=structure,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        include_shell_atoms_in_stoichiometry=True,
    )

    clusters = network.find_clusters(shell_levels=(1,))

    assert clusters[0].stoichiometry == {"Pb": 2, "I": 1, "O": 1}


@pytest.mark.parametrize(
    "search_mode",
    (
        SEARCH_MODE_KDTREE,
        SEARCH_MODE_VECTORIZED,
        SEARCH_MODE_BRUTEFORCE,
    ),
)
def test_pdb_cluster_search_modes_match(search_mode):
    structure = PDBStructure.from_lines(
        _connected_frame_lines(),
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
    )
    network = ClusterNetwork(
        pdb_structure=structure,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        search_mode=search_mode,
    )

    clusters = network.find_clusters(shell_levels=(1,))

    expected = [(((1, 3)), ((2,)), ((4,)), {"Pb": 2, "I": 1})]
    assert _cluster_signatures(clusters) == expected


def test_xyz_structure_and_cluster_network_find_solute_cluster_and_shell():
    structure = XYZStructure.from_lines(
        _connected_xyz_lines(),
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        source_name="frame_0000",
    )
    network = XYZClusterNetwork(
        xyz_structure=structure,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    clusters = network.find_clusters(shell_levels=(1,))

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.node_atom_ids == (1, 3)
    assert cluster.linker_atom_ids == (2,)
    assert cluster.shell_atom_ids == (4,)
    assert cluster.stoichiometry == {"Pb": 2, "I": 1}


@pytest.mark.parametrize(
    "search_mode",
    (
        SEARCH_MODE_KDTREE,
        SEARCH_MODE_VECTORIZED,
        SEARCH_MODE_BRUTEFORCE,
    ),
)
def test_xyz_cluster_search_modes_match(search_mode):
    structure = XYZStructure.from_lines(
        _connected_xyz_lines(),
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        source_name="frame_0000",
    )
    network = XYZClusterNetwork(
        xyz_structure=structure,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        search_mode=search_mode,
    )

    clusters = network.find_clusters(shell_levels=(1,))

    expected = [(((1, 3)), ((2,)), ((4,)), {"Pb": 2, "I": 1})]
    assert _cluster_signatures(clusters) == expected


def test_trajectory_cluster_analyzer_runs_across_frames_and_exports(
    tmp_path,
):
    trajectory_file = tmp_path / "cluster_traj.pdb"
    trajectory_file.write_text(
        "".join(_connected_frame_lines() + _disconnected_frame_lines())
    )

    analyzer = TrajectoryClusterAnalyzer(
        trajectory_file=trajectory_file,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    results = analyzer.analyze_frames(shell_levels=(1,))
    export = analyzer.export_cluster_pdbs(
        tmp_path / "clusters",
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    assert len(results) == 2
    assert results[0].n_clusters == 1
    assert results[1].n_clusters == 3
    assert results[0].clusters[0].stoichiometry == {"Pb": 2, "I": 1}
    assert _relative_names(export.written_files) == [
        "I/frame_0001_AAC.pdb",
        "Pb/frame_0001_AAA.pdb",
        "Pb/frame_0001_AAB.pdb",
        "Pb2I/frame_0000_AAA.pdb",
    ]
    assert (tmp_path / "clusters" / "Pb2I").is_dir()
    assert (tmp_path / "clusters" / "Pb").is_dir()
    assert (tmp_path / "clusters" / "I").is_dir()
    assert not (tmp_path / "clusters" / "frame_0000").exists()
    assert not (tmp_path / "clusters" / "frame_0001").exists()


def test_extracted_frame_folder_cluster_analyzer_runs_and_exports(tmp_path):
    frames_dir = tmp_path / "splitpdb0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.pdb").write_text(
        "".join(_connected_frame_lines())
    )
    (frames_dir / "frame_0001.pdb").write_text(
        "".join(_disconnected_frame_lines())
    )

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    summary = analyzer.inspect()
    results = analyzer.analyze_frames(shell_levels=(1,))
    export = analyzer.export_cluster_pdbs(
        tmp_path / "clusters_from_folder",
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    assert summary["n_frames"] == 2
    assert summary["frame_format"] == "pdb"
    assert summary["output_file_extension"] == ".pdb"
    assert summary["estimated_box_dimensions"] == (2.0, 1.7, 0.0)
    assert summary["first_frame"] == "frame_0000.pdb"
    assert summary["last_frame"] == "frame_0001.pdb"
    assert len(results) == 2
    assert results[0].n_clusters == 1
    assert results[1].n_clusters == 3
    assert _relative_names(export.written_files) == [
        "I/frame_0001_AAC.pdb",
        "Pb/frame_0001_AAA.pdb",
        "Pb/frame_0001_AAB.pdb",
        "Pb2I/frame_0000_AAA.pdb",
    ]
    assert (tmp_path / "clusters_from_folder" / "Pb2I").is_dir()
    assert (tmp_path / "clusters_from_folder" / "Pb").is_dir()
    assert (tmp_path / "clusters_from_folder" / "I").is_dir()
    assert not (tmp_path / "clusters_from_folder" / "frame_0000").exists()
    assert not (tmp_path / "clusters_from_folder" / "frame_0001").exists()


def test_shell_atoms_can_control_stoichiometry_bins(tmp_path):
    frames_dir = tmp_path / "splitpdb0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.pdb").write_text(
        "".join(_connected_frame_lines())
    )

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        include_shell_atoms_in_stoichiometry=True,
    )

    export = analyzer.export_cluster_pdbs(
        tmp_path / "clusters_from_folder",
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    assert _relative_names(export.written_files) == [
        "Pb2IO/frame_0000_AAA.pdb"
    ]


def test_extracted_xyz_frame_folder_cluster_analyzer_runs_and_exports(
    tmp_path,
):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))
    (frames_dir / "frame_0001.xyz").write_text(
        "".join(_disconnected_xyz_lines())
    )

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        save_state_frequency=7,
    )

    summary = analyzer.inspect()
    results = analyzer.analyze_frames(shell_levels=(1,))
    export = analyzer.export_cluster_files(
        tmp_path / "clusters_from_xyz_folder",
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    assert summary["frame_format"] == "xyz"
    assert summary["file_type"] == "xyz_frames"
    assert summary["output_file_extension"] == ".xyz"
    assert not summary["supports_full_molecule_shells"]
    assert summary["estimated_box_dimensions"] == (2.0, 1.7, 0.0)
    assert summary["n_frames"] == 2
    assert summary["first_frame"] == "frame_0000.xyz"
    assert summary["last_frame"] == "frame_0001.xyz"
    assert len(results) == 2
    assert results[0].n_clusters == 1
    assert results[1].n_clusters == 3
    assert _relative_names(export.written_files) == [
        "I/frame_0001_AAC.xyz",
        "Pb/frame_0001_AAA.xyz",
        "Pb/frame_0001_AAB.xyz",
        "Pb2I/frame_0000_AAA.xyz",
    ]
    assert (tmp_path / "clusters_from_xyz_folder" / "Pb2I").is_dir()
    assert (tmp_path / "clusters_from_xyz_folder" / "Pb").is_dir()
    assert (tmp_path / "clusters_from_xyz_folder" / "I").is_dir()
    assert not (tmp_path / "clusters_from_xyz_folder" / "frame_0000").exists()
    assert not (tmp_path / "clusters_from_xyz_folder" / "frame_0001").exists()


def test_detect_frame_folder_mode_rejects_mixed_formats(tmp_path):
    frames_dir = tmp_path / "mixed_frames"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.pdb").write_text(
        "".join(_connected_frame_lines())
    )
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))

    with pytest.raises(ValueError, match="mixes \\.pdb and \\.xyz"):
        detect_frame_folder_mode(frames_dir)


def test_extracted_frame_folder_uses_natural_frame_order(tmp_path):
    frames_dir = tmp_path / "splitxyz"
    frames_dir.mkdir()
    (frames_dir / "frame_10.xyz").write_text("".join(_connected_xyz_lines()))
    (frames_dir / "frame_2.xyz").write_text("".join(_connected_xyz_lines()))
    (frames_dir / "frame_1.xyz").write_text("".join(_connected_xyz_lines()))

    frame_format, frame_paths = detect_frame_folder_mode(frames_dir)
    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        save_state_frequency=7,
    )

    summary = analyzer.inspect()

    assert frame_format == "xyz"
    assert [path.name for path in frame_paths] == [
        "frame_1.xyz",
        "frame_2.xyz",
        "frame_10.xyz",
    ]
    assert summary["first_frame"] == "frame_1.xyz"
    assert summary["last_frame"] == "frame_10.xyz"


def test_extracted_xyz_inspect_uses_pbc_box_from_source_filename(tmp_path):
    source_dir = tmp_path / "cluster_run"
    source_dir.mkdir()
    frames_dir = source_dir / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))
    (
        source_dir / "xyz_pbi2_dmso_1M_RT_den1p47_pbc_17p07x_041-pos-1.xyz"
    ).write_text("2\nsource\n")

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    summary = analyzer.inspect()

    assert summary["box_dimensions"] == (17.07, 17.07, 17.07)
    assert summary["estimated_box_dimensions"] == (17.07, 17.07, 17.07)
    assert summary["box_dimensions_source_kind"] == "source_filename"
    assert (
        summary["box_dimensions_source"]
        == "xyz_pbi2_dmso_1M_RT_den1p47_pbc_17p07x_041-pos-1.xyz"
    )


def test_cluster_export_writes_metadata_file(tmp_path):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))
    (frames_dir / "frame_0001.xyz").write_text(
        "".join(_disconnected_xyz_lines())
    )
    output_dir = tmp_path / "clusters_from_xyz_folder"

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    export = analyzer.export_cluster_files(
        output_dir,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    metadata_path = output_dir / "cluster_extraction_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    assert export.metadata_path == metadata_path
    assert metadata["state"] == "completed"
    assert metadata["progress"]["completed_frames"] == 2
    assert metadata["progress"]["remaining_frames"] == 0
    assert metadata["parameters"]["search_mode"] == "kdtree"
    assert (
        metadata["runtime"]["save_state_frequency_frames"]
        == DEFAULT_SAVE_STATE_FREQUENCY
    )
    assert metadata["input"]["first_frame"] == "frame_0000.xyz"
    assert metadata["input"]["last_frame"] == "frame_0001.xyz"
    assert metadata["output"]["written_files"] == [
        "I/frame_0001_AAC.xyz",
        "Pb/frame_0001_AAA.xyz",
        "Pb/frame_0001_AAB.xyz",
        "Pb2I/frame_0000_AAA.xyz",
    ]


def test_cluster_export_resumes_from_metadata_after_interruption(tmp_path):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))
    (frames_dir / "frame_0001.xyz").write_text(
        "".join(_disconnected_xyz_lines())
    )
    output_dir = tmp_path / "clusters_from_xyz_folder"

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        save_state_frequency=7,
    )

    def stop_after_first(processed, total, frame_label):
        if processed >= 1 and frame_label != "resume":
            raise RuntimeError("stop after one frame")

    with pytest.raises(RuntimeError, match="stop after one frame"):
        analyzer.export_cluster_files(
            output_dir,
            shell_levels=(1,),
            include_shell_levels=(0, 1),
            progress_callback=stop_after_first,
        )

    metadata_path = output_dir / "cluster_extraction_metadata.json"
    interrupted = json.loads(metadata_path.read_text())
    assert interrupted["state"] == "failed"
    assert interrupted["output"]["written_files"] == []
    assert interrupted["progress"]["sorted_frames"] == 0
    assert interrupted["progress"]["sorting_remaining_frames"] == 1
    assert interrupted["progress"]["completed_frames"] == 1

    resumed = analyzer.export_cluster_files(
        output_dir,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )
    completed = json.loads(metadata_path.read_text())

    assert resumed.resumed
    assert not resumed.already_complete
    assert resumed.previously_completed_frames == 1
    assert resumed.newly_processed_frames == 1
    assert completed["state"] == "completed"
    assert completed["progress"]["completed_frames"] == 2
    assert completed["progress"]["sorted_frames"] == 2


def test_cluster_export_batches_metadata_checkpoint_writes(
    tmp_path,
    monkeypatch,
):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    for frame_index in range(12):
        (frames_dir / f"frame_{frame_index:04d}.xyz").write_text(
            "".join(_connected_xyz_lines())
        )
    output_dir = tmp_path / "clusters_from_xyz_folder"

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
        save_state_frequency=7,
    )

    write_count = 0
    original_write_json_file = cluster_module._write_json_file

    def counted_write_json_file(path, payload):
        nonlocal write_count
        write_count += 1
        original_write_json_file(path, payload)

    monkeypatch.setattr(
        cluster_module,
        "_write_json_file",
        counted_write_json_file,
    )

    analyzer.export_cluster_files(
        output_dir,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    metadata = json.loads(
        (output_dir / "cluster_extraction_metadata.json").read_text()
    )

    assert metadata["runtime"]["save_state_frequency_frames"] == 7
    assert write_count <= 6


def test_cluster_export_recognizes_completed_metadata(tmp_path):
    frames_dir = tmp_path / "splitxyz0001"
    frames_dir.mkdir()
    (frames_dir / "frame_0000.xyz").write_text("".join(_connected_xyz_lines()))
    output_dir = tmp_path / "clusters_from_xyz_folder"

    analyzer = ExtractedFrameFolderClusterAnalyzer(
        frames_dir=frames_dir,
        atom_type_definitions=XYZ_ATOM_TYPE_DEFINITIONS,
        pair_cutoffs_def=PAIR_CUTOFFS,
    )

    first = analyzer.export_cluster_files(
        output_dir,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )
    second = analyzer.export_cluster_files(
        output_dir,
        shell_levels=(1,),
        include_shell_levels=(0, 1),
    )

    assert not first.already_complete
    assert second.resumed
    assert second.already_complete
    assert second.previously_completed_frames == 1
    assert second.newly_processed_frames == 0
    assert sorted(path.name for path in second.written_files) == [
        "frame_0000_AAA.xyz"
    ]
