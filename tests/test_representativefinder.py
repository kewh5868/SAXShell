from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea, QSplitter

from saxshell.bondanalysis import (
    AngleTripletDefinition,
    BondAnalysisPreset,
    BondPairDefinition,
)
from saxshell.fullrmc.project_model import ensure_rmcsetup_structure
from saxshell.fullrmc.representatives import (
    load_representative_selection_metadata,
)
from saxshell.fullrmc.solvent_handling import (
    SolventHandlingEntry,
    SolventHandlingMetadata,
    SolventHandlingSettings,
    available_representative_structure_modes,
    load_solvent_handling_metadata,
    representative_structure_mode_is_ready,
    representative_structure_path_for_mode,
    save_solvent_handling_metadata,
)
from saxshell.representativefinder import (
    RepresentativeFinderCandidate,
    RepresentativeFinderOperationCancelled,
    RepresentativeFinderResult,
    RepresentativeFinderSettings,
    analyze_representative_structure_folder,
    build_representativefinder_run_config,
    default_representativefinder_run_file_path,
    inspect_representative_structure_input,
    load_representativefinder_result,
    load_representativefinder_run_config,
    persist_representativefinder_result_to_project,
    run_representativefinder_run_config,
    save_representativefinder_run_config,
)
from saxshell.representativefinder.cli import (
    main as representativefinder_cli_main,
)
from saxshell.representativefinder.ui.batch_queue_window import (
    RepresentativeFinderBatchJob,
    RepresentativeFinderBatchQueueWindow,
    RepresentativeFinderBatchWorker,
)
from saxshell.representativefinder.ui.main_window import (
    RepresentativeStructureFinderMainWindow,
)
from saxshell.representativefinder.ui.run_file_window import (
    RepresentativeFinderRunFileWindow,
)
from saxshell.saxs.electron_density_mapping.ui.viewer import (
    ElectronDensityStructureViewer,
)
from saxshell.saxs.project_manager import SAXSProjectManager
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import create_reference_molecule


def _write_xyz_structure(
    path: Path, atoms: list[tuple[str, float, float, float]]
) -> None:
    lines = [str(len(atoms)), path.stem]
    for element, x_coord, y_coord, z_coord in atoms:
        lines.append(f"{element} {x_coord:.3f} {y_coord:.3f} {z_coord:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_test_solvent_reference_library(
    tmp_path: Path,
) -> tuple[Path, Path]:
    reference_source = tmp_path / "water_source.pdb"
    PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="HOH",
                residue_number=1,
                coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="H1",
                residue_name="HOH",
                residue_number=1,
                coordinates=np.array([0.958, 0.0, 0.0], dtype=float),
                element="H",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="H2",
                residue_name="HOH",
                residue_number=1,
                coordinates=np.array([-0.239, 0.927, 0.0], dtype=float),
                element="H",
            ),
        ],
        source_name="water_source",
    ).write_pdb_file(reference_source)
    reference_library_dir = tmp_path / "reference_library"
    reference_library_dir.mkdir(parents=True, exist_ok=True)
    result = create_reference_molecule(
        reference_source,
        reference_name="water_test",
        residue_name="HOH",
        library_dir=reference_library_dir,
    )
    return reference_library_dir, result.path


def _write_complete_solvent_representative_pdb(
    tmp_path: Path,
    *,
    reference_path: Path,
) -> Path:
    reference_structure = PDBStructure.from_file(reference_path)
    atoms = [
        PDBAtom(
            atom_id=1,
            atom_name="PB1",
            residue_name="PBI",
            residue_number=1,
            coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
            element="Pb",
        )
    ]
    atom_id = 2
    for residue_name, residue_number, shift in (
        ("HOH", 2, np.array([3.0, 0.0, 0.0], dtype=float)),
        ("ALT", 3, np.array([6.0, 0.0, 0.0], dtype=float)),
    ):
        for reference_atom in reference_structure.atoms:
            atoms.append(
                PDBAtom(
                    atom_id=atom_id,
                    atom_name=reference_atom.atom_name,
                    residue_name=residue_name,
                    residue_number=residue_number,
                    coordinates=reference_atom.coordinates.copy() + shift,
                    element=reference_atom.element,
                )
            )
            atom_id += 1
    output_path = tmp_path / "fullsolv_candidate.pdb"
    PDBStructure(atoms=atoms, source_name="fullsolv_candidate").write_pdb_file(
        output_path
    )
    return output_path


def _build_manual_representative_result(
    structure_path: Path,
    *,
    output_dir: Path,
    structure_label: str = "Pb",
    atom_count: int | None = None,
    element_counts: dict[str, int] | None = None,
    solvent_atom_count: int = 0,
) -> RepresentativeFinderResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = RepresentativeFinderCandidate(
        file_path=structure_path.resolve(),
        relative_label=structure_path.name,
        motif_label="no_motif",
        atom_count=int(atom_count if atom_count is not None else 1),
        element_counts=dict(element_counts or {"Pb": 1}),
        bond_values={},
        angle_values={},
        solvent_metrics={},
        solvent_atom_count=int(solvent_atom_count),
        direct_solvent_atom_count=int(solvent_atom_count),
        outer_solvent_atom_count=0,
        mean_direct_solvent_coordination=0.0,
        score_total=0.0,
        score_bond=0.0,
        score_angle=0.0,
        score_solvent=0.0,
    )
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text("{}", encoding="utf-8")
    score_table_path = output_dir / "scores.tsv"
    score_table_path.write_text("", encoding="utf-8")
    summary_text_path = output_dir / "summary.txt"
    summary_text_path.write_text("", encoding="utf-8")
    representative_output_path = output_dir / structure_path.name
    representative_output_path.write_text("", encoding="utf-8")
    return RepresentativeFinderResult(
        input_dir=structure_path.parent.resolve(),
        output_dir=output_dir.resolve(),
        structure_label=structure_label,
        expected_core_counts=dict(element_counts or {"Pb": 1}),
        settings=RepresentativeFinderSettings(),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        candidates=(candidate,),
        selected_candidate=candidate,
        representative_output_path=representative_output_path,
        skipped_files=(),
        target_bond_values={},
        target_angle_values={},
        target_solvent_metrics={},
        summary_json_path=summary_json_path,
        score_table_path=score_table_path,
        summary_text_path=summary_text_path,
    )


def _build_representative_test_folder(tmp_path: Path) -> Path:
    stoich_dir = tmp_path / "PbI2"
    motif_dir = stoich_dir / "motif_corner"
    stoich_dir.mkdir(parents=True)
    motif_dir.mkdir(parents=True)

    _write_xyz_structure(
        stoich_dir / "candidate_low.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.0, 0.0, 0.0),
            ("I", 0.0, 2.0, 0.0),
            ("O", 0.0, 0.0, 2.4),
            ("O", 0.0, 0.0, 4.9),
        ],
    )
    _write_xyz_structure(
        motif_dir / "candidate_mid.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.2, 0.0, 0.0),
            ("I", 0.0, 2.2, 0.0),
            ("O", 0.0, 0.0, 2.6),
            ("O", 0.0, 0.0, 5.2),
        ],
    )
    _write_xyz_structure(
        stoich_dir / "candidate_high.xyz",
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("I", 2.8, 0.0, 0.0),
            ("I", 0.0, 2.8, 0.0),
            ("O", 0.0, 0.0, 1.8),
        ],
    )
    return stoich_dir


def _build_single_atom_test_folder(tmp_path: Path) -> Path:
    stoich_dir = tmp_path / "I"
    stoich_dir.mkdir(parents=True)
    _write_xyz_structure(
        stoich_dir / "candidate_01.xyz",
        [("I", 0.0, 0.0, 0.0)],
    )
    _write_xyz_structure(
        stoich_dir / "candidate_02.xyz",
        [("I", 0.0, 0.0, 0.0)],
    )
    return stoich_dir


def _build_multi_stoichiometry_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root_dir = tmp_path / "cluster_root"
    root_dir.mkdir(parents=True)
    pb_dir = _build_representative_test_folder(root_dir)

    sn_dir = root_dir / "SnBr2"
    sn_motif_dir = sn_dir / "motif_edge"
    sn_dir.mkdir(parents=True)
    sn_motif_dir.mkdir(parents=True)
    _write_xyz_structure(
        sn_dir / "candidate_low.xyz",
        [
            ("Sn", 0.0, 0.0, 0.0),
            ("Br", 2.0, 0.0, 0.0),
            ("Br", 0.0, 2.0, 0.0),
            ("O", 0.0, 0.0, 2.4),
            ("O", 0.0, 0.0, 4.9),
        ],
    )
    _write_xyz_structure(
        sn_motif_dir / "candidate_mid.xyz",
        [
            ("Sn", 0.0, 0.0, 0.0),
            ("Br", 2.2, 0.0, 0.0),
            ("Br", 0.0, 2.2, 0.0),
            ("O", 0.0, 0.0, 2.6),
            ("O", 0.0, 0.0, 5.2),
        ],
    )
    _write_xyz_structure(
        sn_dir / "candidate_high.xyz",
        [
            ("Sn", 0.0, 0.0, 0.0),
            ("Br", 2.8, 0.0, 0.0),
            ("Br", 0.0, 2.8, 0.0),
            ("O", 0.0, 0.0, 1.8),
        ],
    )
    return root_dir, pb_dir, sn_dir


def _build_zinc_stoichiometry_folder(root_dir: Path) -> Path:
    stoich_dir = root_dir / "ZnCl2"
    motif_dir = stoich_dir / "motif_edge"
    stoich_dir.mkdir(parents=True)
    motif_dir.mkdir(parents=True)
    _write_xyz_structure(
        stoich_dir / "candidate_low.xyz",
        [
            ("Zn", 0.0, 0.0, 0.0),
            ("Cl", 2.0, 0.0, 0.0),
            ("Cl", 0.0, 2.0, 0.0),
        ],
    )
    _write_xyz_structure(
        motif_dir / "candidate_mid.xyz",
        [
            ("Zn", 0.0, 0.0, 0.0),
            ("Cl", 2.2, 0.0, 0.0),
            ("Cl", 0.0, 2.2, 0.0),
        ],
    )
    return stoich_dir


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_representativefinder_workflow_selects_middle_candidate(tmp_path):
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )

    result = analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "representative_output",
    )

    assert result.selected_candidate.file_name == "candidate_mid.xyz"
    assert result.representative_output_path.is_file()
    assert result.summary_json_path.is_file()
    assert result.score_table_path.is_file()
    assert result.summary_text_path.is_file()
    assert "candidate_mid.xyz" in result.summary_text()


def test_representativefinder_result_json_preserves_analysis_details(tmp_path):
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    result = analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "representative_output",
    )

    loaded = load_representativefinder_result(result.summary_json_path)
    bond_definition = loaded.settings.bond_pairs[0]
    angle_definition = loaded.settings.angle_triplets[0]

    assert (
        loaded.selected_candidate.file_name
        == result.selected_candidate.file_name
    )
    assert loaded.settings.bond_pairs == settings.bond_pairs
    assert loaded.settings.angle_triplets == settings.angle_triplets
    assert loaded.target_bond_values[bond_definition].size > 0
    assert loaded.target_angle_values[angle_definition].size > 0
    assert loaded.selected_candidate.bond_values[bond_definition]
    assert loaded.selected_candidate.angle_values[angle_definition]
    assert loaded.candidates[0].score_total is not None


def test_representativefinder_single_atom_shortcuts_full_analysis(
    tmp_path,
    monkeypatch,
):
    stoich_dir = _build_single_atom_test_folder(tmp_path)
    progress_events: list[tuple[int, int, str]] = []
    log_messages: list[str] = []

    def fail_measurement(*_args, **_kwargs):
        pytest.fail(
            "Single-atom representative selection should not run full "
            "bond/angle measurement."
        )

    monkeypatch.setattr(
        "saxshell.representativefinder.workflow."
        "BondAnalyzer.measure_structure_data",
        fail_measurement,
    )

    result = analyze_representative_structure_folder(
        stoich_dir,
        settings=RepresentativeFinderSettings(),
        output_dir=tmp_path / "single_atom_output",
        progress_callback=lambda processed, total, message: progress_events.append(
            (processed, total, message)
        ),
        log_callback=log_messages.append,
    )

    assert result.selected_candidate.atom_count == 1
    assert result.selected_candidate.element_counts == {"I": 1}
    assert result.selected_candidate.score_total == pytest.approx(0.0)
    assert result.selected_candidate.score_bond == pytest.approx(0.0)
    assert result.selected_candidate.score_angle == pytest.approx(0.0)
    assert result.selected_candidate.score_solvent == pytest.approx(0.0)
    assert any(
        "Single-atom candidate structures were detected" in note
        for note in result.selected_candidate.descriptor_notes
    )
    assert result.representative_output_path.is_file()
    assert result.summary_json_path.is_file()
    assert result.score_table_path.is_file()
    assert result.summary_text_path.is_file()
    assert not any(
        "Aggregating bond and angle distributions" in message
        for _processed, _total, message in progress_events
    )
    assert not any(
        "Measuring " in message for _p, _t, message in progress_events
    )
    assert not any(
        "Scoring " in message for _p, _t, message in progress_events
    )
    assert any(
        "single-atom candidate set" in message.lower()
        for message in log_messages
    )
    assert progress_events[-1][0] == progress_events[-1][1]
    assert (
        progress_events[-1][2]
        == "Representative-structure selection complete."
    )


def test_representativefinder_workflow_reports_post_measurement_progress(
    tmp_path,
):
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    progress_events: list[tuple[int, int, str]] = []

    analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "representative_output",
        progress_callback=lambda processed, total, message: progress_events.append(
            (processed, total, message)
        ),
    )

    assert progress_events
    assert any("Scoring " in message for _p, _t, message in progress_events)
    assert any(
        "Writing representative outputs" in message
        for _p, _t, message in progress_events
    )
    assert progress_events[-1][0] == progress_events[-1][1]
    assert (
        progress_events[-1][2]
        == "Representative-structure selection complete."
    )


def test_representativefinder_workflow_generates_optional_predicted_output(
    tmp_path,
):
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
        generate_predicted_optimized_representative=True,
    )

    result = analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "representative_output",
    )

    assert result.selected_candidate.file_name == "candidate_mid.xyz"
    assert result.predicted_candidate is not None
    assert result.predicted_output_path is not None
    assert result.predicted_output_path.is_file()
    assert result.predicted_candidate.file_path == result.predicted_output_path
    assert result.predicted_candidate.atom_count == 3
    assert result.solvent_completed_predicted_candidate is None
    assert result.solvent_completed_predicted_output_path is None
    assert any(
        "Cluster Dynamics ML geometry scaffold" in note
        for note in result.predicted_generation_notes
    )
    assert "Predicted optimized representative" in result.summary_text()


def test_representativefinder_project_persistence_writes_shared_partialsolv_outputs(
    tmp_path,
):
    stoich_dir = _build_representative_test_folder(tmp_path)
    sn_root, _pb_dir, sn_dir = _build_multi_stoichiometry_root(
        tmp_path / "multi"
    )
    del sn_root
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    pb_result = analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "pb_output",
    )
    sn_result = analyze_representative_structure_folder(
        sn_dir,
        settings=settings,
        output_dir=tmp_path / "sn_output",
    )

    pb_shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        pb_result,
    )
    sn_shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        sn_result,
    )

    rmcsetup_paths = ensure_rmcsetup_structure(tmp_path)
    metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    assert metadata is not None
    assert metadata.selection_mode == "representative_finder"
    assert pb_shared_path.is_file()
    assert sn_shared_path.is_file()
    assert pb_shared_path.parent == (
        rmcsetup_paths.representative_partial_solvent_dir / "PbI2"
    )
    assert sn_shared_path.parent == (
        rmcsetup_paths.representative_partial_solvent_dir / "SnBr2"
    )
    assert [entry.structure for entry in metadata.representative_entries] == [
        "PbI2",
        "SnBr2",
    ]
    assert all(
        entry.motif == "no_motif" for entry in metadata.representative_entries
    )
    assert all(
        entry.source_solvent_mode == "partialsolv"
        for entry in metadata.representative_entries
    )
    assert all(
        Path(entry.source_file).is_file()
        for entry in metadata.representative_entries
    )
    assert all(
        entry.project_cached_results_path
        and Path(entry.project_cached_results_path).is_file()
        for entry in metadata.representative_entries
    )
    cached_pb_result = load_representativefinder_result(
        metadata.representative_entries[0].project_cached_results_path
    )
    assert cached_pb_result.structure_label == "PbI2"
    assert cached_pb_result.target_bond_values
    assert cached_pb_result.selected_candidate.bond_values
    assert (
        pytest.approx(
            sum(
                entry.selected_weight
                for entry in metadata.representative_entries
            ),
            rel=0.0,
            abs=1.0e-9,
        )
        == 1.0
    )

    state = SAXSProjectManager().inspect_representative_structures(tmp_path)
    assert state.representative_count == 2
    assert state.source_files_ready is True
    assert "partialsolv" in state.available_modes
    assert (
        state.partialsolv_dir
        == rmcsetup_paths.representative_partial_solvent_dir
    )


def test_representativefinder_run_file_round_trip_uses_project_relative_paths(
    tmp_path,
):
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    output_dir = tmp_path / "representative_finder" / "batch_run"
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        solvent_weight=0.0,
        parallel_workers=2,
    )
    config = build_representativefinder_run_config(
        project_dir=tmp_path,
        input_dir=root_dir,
        output_dir=output_dir,
        analysis_mode="all",
        settings=settings,
        overwrite_existing=True,
    )
    run_file_path = default_representativefinder_run_file_path(tmp_path)

    save_representativefinder_run_config(run_file_path, config)
    loaded = load_representativefinder_run_config(run_file_path)

    assert loaded.input_dir == "cluster_root"
    assert loaded.output_dir == "representative_finder/batch_run"
    assert loaded.analysis_mode == "all"
    assert loaded.overwrite_existing is True
    assert loaded.settings.parallel_workers == 2
    assert [pair.display_label for pair in loaded.settings.bond_pairs] == [
        "Pb-I",
        "Sn-Br",
    ]


def test_representativefinder_cli_run_file_publishes_project_registry(
    tmp_path,
):
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.0,
        parallel_workers=2,
    )
    config = build_representativefinder_run_config(
        project_dir=tmp_path,
        input_dir=root_dir,
        output_dir=tmp_path / "representative_finder" / "cli_batch",
        analysis_mode="all",
        settings=settings,
        overwrite_existing=True,
    )
    run_file_path = default_representativefinder_run_file_path(tmp_path)
    save_representativefinder_run_config(run_file_path, config)

    summary = run_representativefinder_run_config(
        tmp_path,
        load_representativefinder_run_config(run_file_path),
        run_file_path=run_file_path,
    )

    assert summary.completed_count == 2
    assert summary.failed_count == 0
    assert len(summary.project_representative_paths) == 2
    assert all(path.is_file() for path in summary.project_representative_paths)

    rmcsetup_paths = ensure_rmcsetup_structure(tmp_path)
    metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    assert metadata is not None
    assert metadata.selection_mode == "representative_finder"
    assert [entry.structure for entry in metadata.representative_entries] == [
        "PbI2",
        "SnBr2",
    ]
    assert all(
        Path(entry.source_file).is_file()
        for entry in metadata.representative_entries
    )
    state = SAXSProjectManager().inspect_representative_structures(tmp_path)
    assert state.representative_count == 2
    assert state.source_files_ready is True
    assert "nosolv" in state.available_modes


def test_representativefinder_cli_command_uses_project_default_run_file(
    tmp_path,
    capsys,
):
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        solvent_weight=0.0,
        parallel_workers=2,
    )
    config = build_representativefinder_run_config(
        project_dir=tmp_path,
        input_dir=root_dir,
        output_dir=tmp_path / "representative_finder" / "cli_entrypoint",
        analysis_mode="all",
        settings=settings,
        overwrite_existing=True,
    )
    save_representativefinder_run_config(
        default_representativefinder_run_file_path(tmp_path),
        config,
    )

    exit_code = representativefinder_cli_main(["run", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Representative CLI run complete" in captured.out
    assert "Completed: 2" in captured.out
    assert "Failed: 0" in captured.out


def test_representativefinder_run_file_window_builds_beta_config(
    qapp,
    tmp_path,
):
    del qapp
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    window = RepresentativeFinderRunFileWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )
    window.bond_pairs_edit.setPlainText("Pb:I:3.2\nSn:Br:3.2")
    window.angle_triplets_edit.setPlainText("Pb:I:I:3.2:3.2\nSn:Br:Br:3.2:3.2")
    window.output_dir_edit.setText(
        str(tmp_path / "representative_finder" / "window_config")
    )
    window.analysis_mode_combo.setCurrentIndex(0)
    window.worker_spin.setValue(4)

    config = window._current_config(tmp_path)

    assert config.analysis_mode == "all"
    assert config.input_dir == "cluster_root"
    assert config.output_dir == "representative_finder/window_config"
    assert config.settings.parallel_workers == 4
    assert [pair.display_label for pair in config.settings.bond_pairs] == [
        "Pb-I",
        "Sn-Br",
    ]
    window.close()


def test_representativefinder_project_persistence_classifies_fullsolv_outputs(
    tmp_path,
):
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    representative_path = _write_complete_solvent_representative_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    result = _build_manual_representative_result(
        representative_path,
        output_dir=tmp_path / "representative_output",
        structure_label="Pb",
        atom_count=7,
        element_counts={"Pb": 1, "O": 2, "H": 4},
        solvent_atom_count=6,
    )

    rmcsetup_paths = ensure_rmcsetup_structure(tmp_path)
    solvent_metadata = SolventHandlingMetadata(
        settings=SolventHandlingSettings.from_dict(
            {
                "reference_source": "custom",
                "custom_reference_path": str(reference_path),
                "reference_match_tolerance_a": 0.25,
            }
        ),
        reference_path=str(reference_path),
        reference_name="water_test",
        reference_residue_name="HOH",
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_selection_mode="representative_finder",
        detected_distribution_status="unknown",
        detected_distribution_note="",
        aggregate_solute_element_counts={"Pb": 1},
        entries=[],
    )
    save_solvent_handling_metadata(
        rmcsetup_paths.solvent_handling_path,
        solvent_metadata,
    )

    shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        result,
    )

    metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    assert metadata is not None
    assert shared_path.parent == rmcsetup_paths.pdb_with_solvent_dir / "Pb"
    assert metadata.representative_entries[0].source_solvent_mode == "fullsolv"
    assert (
        Path(metadata.representative_entries[0].source_file).resolve()
        == shared_path.resolve()
    )

    state = SAXSProjectManager().inspect_representative_structures(tmp_path)
    assert state.representative_count == 1
    assert state.source_files_ready is True
    assert "fullsolv" in state.available_modes
    assert "partialsolv" not in state.available_modes

    assert available_representative_structure_modes(metadata, None) == [
        "full_solvent"
    ]
    assert (
        representative_structure_path_for_mode(
            metadata.representative_entries[0],
            None,
            "full_solvent",
        )
        == shared_path.resolve()
    )


def test_representativefinder_project_persistence_mirrors_single_atom_outputs_to_all_solvent_variants(
    tmp_path,
):
    single_atom_path = tmp_path / "I_candidate.xyz"
    _write_xyz_structure(
        single_atom_path,
        [("I", 0.0, 0.0, 0.0)],
    )
    result = _build_manual_representative_result(
        single_atom_path,
        output_dir=tmp_path / "single_atom_output",
        structure_label="I",
        atom_count=1,
        element_counts={"I": 1},
        solvent_atom_count=0,
    )

    shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        result,
    )

    rmcsetup_paths = ensure_rmcsetup_structure(tmp_path)
    metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    assert metadata is not None
    assert len(metadata.representative_entries) == 1

    entry = metadata.representative_entries[0]
    nosolv_path = (
        rmcsetup_paths.pdb_no_solvent_dir / "I" / shared_path.name
    ).resolve()
    partialsolv_path = (
        rmcsetup_paths.representative_partial_solvent_dir
        / "I"
        / shared_path.name
    ).resolve()
    fullsolv_path = (
        rmcsetup_paths.pdb_with_solvent_dir / "I" / shared_path.name
    ).resolve()

    assert shared_path.resolve() == nosolv_path
    assert entry.source_solvent_mode == "nosolv"
    assert nosolv_path.is_file()
    assert partialsolv_path.is_file()
    assert fullsolv_path.is_file()

    state = SAXSProjectManager().inspect_representative_structures(tmp_path)
    assert state.representative_count == 1
    assert state.source_files_ready is True
    assert set(state.available_modes) == {"nosolv", "partialsolv", "fullsolv"}

    assert available_representative_structure_modes(metadata, None) == [
        "no_solvent",
        "partial_solvent",
        "full_solvent",
    ]
    assert representative_structure_mode_is_ready(metadata, None) is True
    assert (
        representative_structure_path_for_mode(
            entry,
            None,
            "no_solvent",
        )
        == nosolv_path
    )
    assert (
        representative_structure_path_for_mode(
            entry,
            None,
            "partial_solvent",
        )
        == partialsolv_path
    )
    assert (
        representative_structure_path_for_mode(
            entry,
            None,
            "full_solvent",
        )
        == fullsolv_path
    )


def test_representativefinder_project_persistence_replaces_stale_mode_artifacts(
    tmp_path,
):
    partial_source = tmp_path / "partial_candidate.xyz"
    _write_xyz_structure(
        partial_source,
        [
            ("Pb", 0.0, 0.0, 0.0),
            ("O", 2.5, 0.0, 0.0),
        ],
    )
    partial_result = _build_manual_representative_result(
        partial_source,
        output_dir=tmp_path / "partial_output",
        structure_label="Pb",
        atom_count=2,
        element_counts={"Pb": 1, "O": 1},
        solvent_atom_count=1,
    )
    partial_shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        partial_result,
    )

    rmcsetup_paths = ensure_rmcsetup_structure(tmp_path)
    stale_no_solvent_path = (
        rmcsetup_paths.pdb_no_solvent_dir / "Pb" / "Pb__stale_nosolv.pdb"
    )
    stale_full_solvent_path = (
        rmcsetup_paths.pdb_with_solvent_dir / "Pb" / "Pb__stale_fullsolv.pdb"
    )
    stale_no_solvent_path.parent.mkdir(parents=True, exist_ok=True)
    stale_full_solvent_path.parent.mkdir(parents=True, exist_ok=True)
    stale_no_solvent_path.write_text("stale\n", encoding="utf-8")
    stale_full_solvent_path.write_text("stale\n", encoding="utf-8")

    _reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    solvent_metadata = SolventHandlingMetadata(
        settings=SolventHandlingSettings.from_dict(
            {
                "reference_source": "custom",
                "custom_reference_path": str(reference_path),
                "reference_match_tolerance_a": 0.25,
            }
        ),
        reference_path=str(reference_path),
        reference_name="water_test",
        reference_residue_name="HOH",
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_selection_mode="representative_finder",
        detected_distribution_status="partial_solvent",
        detected_distribution_note="",
        aggregate_solute_element_counts={"Pb": 1},
        entries=[
            SolventHandlingEntry(
                structure="Pb",
                motif="no_motif",
                param="Pb",
                source_file=str(partial_shared_path),
                no_solvent_pdb=str(stale_no_solvent_path),
                completed_pdb=str(stale_full_solvent_path),
                atom_count_no_solvent=1,
                atom_count_completed=4,
                solvent_atoms_added=3,
                solvent_molecules_added=1,
                solvent_mode="partial_solvent",
                completion_strategy="stale",
                heuristic_note="stale",
            )
        ],
    )
    save_solvent_handling_metadata(
        rmcsetup_paths.solvent_handling_path,
        solvent_metadata,
    )

    fullsolv_source = _write_complete_solvent_representative_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    fullsolv_result = _build_manual_representative_result(
        fullsolv_source,
        output_dir=tmp_path / "fullsolv_output",
        structure_label="Pb",
        atom_count=7,
        element_counts={"Pb": 1, "O": 2, "H": 4},
        solvent_atom_count=6,
    )

    shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        fullsolv_result,
    )

    metadata = load_representative_selection_metadata(
        rmcsetup_paths.representative_selection_path
    )
    reloaded_solvent_metadata = load_solvent_handling_metadata(
        rmcsetup_paths.solvent_handling_path
    )

    assert metadata is not None
    assert shared_path.parent == rmcsetup_paths.pdb_with_solvent_dir / "Pb"
    assert shared_path.is_file()
    assert not partial_shared_path.exists()
    assert not stale_no_solvent_path.exists()
    assert not stale_full_solvent_path.exists()
    assert metadata.representative_entries[0].source_solvent_mode == "fullsolv"
    assert (
        Path(metadata.representative_entries[0].source_file).resolve()
        == shared_path.resolve()
    )
    assert reloaded_solvent_metadata is not None
    assert reloaded_solvent_metadata.entries == []


def test_representativefinder_window_restores_saved_project_representative_attributes(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    root_dir, pb_dir, sn_dir = _build_multi_stoichiometry_root(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    pb_result = analyze_representative_structure_folder(
        pb_dir,
        settings=settings,
        output_dir=tmp_path / "pb_output",
    )
    sn_result = analyze_representative_structure_folder(
        sn_dir,
        settings=settings,
        output_dir=tmp_path / "sn_output",
    )
    pb_shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        pb_result,
    )
    sn_shared_path = persist_representativefinder_result_to_project(
        tmp_path,
        sn_result,
    )

    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )

    assert window.stoichiometry_table.item(0, 3).text() == "Complete"
    assert window.stoichiometry_table.item(0, 4).text() == pb_shared_path.name
    assert window.stoichiometry_table.item(0, 5).text() == (
        f"{float(pb_result.selected_candidate.score_total):.6f}"
    )
    assert window.stoichiometry_table.item(0, 6).text() == "pb_output"
    pb_open_button = window.stoichiometry_table.cellWidget(0, 7)
    assert isinstance(pb_open_button, QPushButton)
    assert pb_open_button.isEnabled() is True

    assert window.stoichiometry_table.item(1, 3).text() == "Complete"
    assert window.stoichiometry_table.item(1, 4).text() == sn_shared_path.name
    assert window.stoichiometry_table.item(1, 5).text() == (
        f"{float(sn_result.selected_candidate.score_total):.6f}"
    )
    assert window.stoichiometry_table.item(1, 6).text() == "sn_output"
    sn_open_button = window.stoichiometry_table.cellWidget(1, 7)
    assert isinstance(sn_open_button, QPushButton)
    assert sn_open_button.isEnabled() is True

    window._select_stoichiometry_row_by_key(str(sn_dir.resolve()))
    summary = window.result_summary_box.toPlainText()
    assert (
        window.run_status_label.text()
        == "Representative selection: restored from saved project analysis"
    )
    assert "Status: Complete" in summary
    assert f"Representative: {sn_shared_path.name}" in summary
    assert (
        f"Project representative file: {sn_shared_path.resolve()}" in summary
    )
    assert "Source solvent mode: partialsolv" in summary
    assert window.candidate_table.rowCount() == len(sn_result.candidates)
    assert window.plot_widget.distribution_selector_combo.count() == 2
    labels = [
        window.plot_widget.distribution_selector_combo.itemText(index)
        for index in range(
            window.plot_widget.distribution_selector_combo.count()
        )
    ]
    assert any("Sn-Br" in label for label in labels)
    assert window.bond_pair_table.rowCount() == 2
    assert window.angle_triplet_table.rowCount() == 2
    window.close()


def test_representativefinder_all_mode_skips_saved_project_representatives_until_overwrite(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    root_dir, pb_dir, sn_dir = _build_multi_stoichiometry_root(tmp_path)
    zn_dir = _build_zinc_stoichiometry_folder(root_dir)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    pb_result = analyze_representative_structure_folder(
        pb_dir,
        settings=settings,
        output_dir=tmp_path / "pb_output",
    )
    sn_result = analyze_representative_structure_folder(
        sn_dir,
        settings=settings,
        output_dir=tmp_path / "sn_output",
    )
    persist_representativefinder_result_to_project(tmp_path, pb_result)
    persist_representativefinder_result_to_project(tmp_path, sn_result)

    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )
    window.analysis_mode_combo.setCurrentIndex(1)
    assert window.overwrite_existing_checkbox.isChecked() is False

    targets = window._analysis_targets_from_inputs(
        output_root=tmp_path / "batch_output",
        settings=RepresentativeFinderSettings(),
    )
    assert [target.inspection.input_dir for target in targets] == [
        zn_dir.resolve()
    ]

    window._reset_stoichiometry_run_state(
        {str(target.inspection.input_dir) for target in targets}
    )
    pb_row = window._stoichiometry_row_by_input_dir[str(pb_dir.resolve())]
    sn_row = window._stoichiometry_row_by_input_dir[str(sn_dir.resolve())]
    zn_row = window._stoichiometry_row_by_input_dir[str(zn_dir.resolve())]
    assert window.stoichiometry_table.item(pb_row, 3).text() == "Complete"
    assert window.stoichiometry_table.item(sn_row, 3).text() == "Complete"
    assert window.stoichiometry_table.item(zn_row, 3).text() == "Queued"

    window.overwrite_existing_checkbox.setChecked(True)
    overwrite_targets = window._analysis_targets_from_inputs(
        output_root=tmp_path / "batch_output",
        settings=RepresentativeFinderSettings(),
    )
    assert [target.inspection.input_dir for target in overwrite_targets] == [
        pb_dir.resolve(),
        sn_dir.resolve(),
        zn_dir.resolve(),
    ]
    window.close()


def test_representativefinder_workflow_supports_cancellation(tmp_path):
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )

    with pytest.raises(RepresentativeFinderOperationCancelled):
        analyze_representative_structure_folder(
            stoich_dir,
            settings=settings,
            output_dir=tmp_path / "representative_output",
            cancel_callback=lambda: True,
        )


def test_representativefinder_input_inspection_discovers_stoichiometry_subfolders(
    tmp_path,
):
    root_dir, pb_dir, sn_dir = _build_multi_stoichiometry_root(tmp_path)

    inspection = inspect_representative_structure_input(root_dir)

    assert inspection.input_dir == root_dir.resolve()
    assert inspection.input_is_stoichiometry_folder is False
    assert inspection.stoichiometry_count == 2
    assert inspection.total_candidate_count == 6
    assert [
        item.structure_label for item in inspection.stoichiometry_folders
    ] == [
        pb_dir.name,
        sn_dir.name,
    ]


def test_representativefinder_batch_queue_prefills_project_clusters_and_all_mode(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    project_dir = tmp_path / "project"
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.create_project(project_dir)
    settings.clusters_dir = str(root_dir.resolve())
    manager.save_project(settings)

    window = RepresentativeFinderBatchQueueWindow(
        initial_project_dir=project_dir,
    )

    assert window.queue_list.count() == 1
    widget = window.queue_list.itemWidget(window.queue_list.item(0))
    assert widget.project_dir_edit.text() == str(project_dir.resolve())
    assert widget.clusters_dir_edit.text() == str(root_dir.resolve())
    assert "representativefinder_batch_cluster_root" in (
        widget.output_dir_edit.text()
    )
    assert (
        widget.analysis_mode_label.text() == "All Discovered Stoichiometries"
    )
    assert window.preset_combo.count() >= 1

    window.close()


def test_representativefinder_batch_worker_publishes_project_results_and_restores_ui(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    project_dir = tmp_path / "project"
    root_dir, _pb_dir, _sn_dir = _build_multi_stoichiometry_root(tmp_path)
    manager = SAXSProjectManager()
    project_settings = manager.create_project(project_dir)
    project_settings.clusters_dir = str(root_dir.resolve())
    manager.save_project(project_settings)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        solvent_weight=0.0,
        parallel_workers=1,
    )
    output_dir = project_dir / "representative_finder" / "batch_run"
    config = build_representativefinder_run_config(
        project_dir=project_dir,
        input_dir=root_dir,
        output_dir=output_dir,
        analysis_mode="all",
        settings=settings,
        overwrite_existing=False,
    )
    job = RepresentativeFinderBatchJob(
        project_dir=project_dir.resolve(),
        clusters_dir=root_dir.resolve(),
        output_dir=output_dir.resolve(),
        config=config,
    )
    worker = RepresentativeFinderBatchWorker([("job-1", job)])
    finished_results: list[object] = []
    failed_items: list[tuple[str, str]] = []
    changed_projects: list[str] = []
    worker.finished.connect(finished_results.append)
    worker.failed.connect(
        lambda item_id, message: failed_items.append((item_id, message))
    )
    worker.project_results_changed.connect(changed_projects.append)

    worker.run()

    assert failed_items == []
    assert changed_projects == [str(project_dir.resolve())]
    assert len(finished_results) == 1
    assert finished_results[0][0].completed_count == 2
    state = manager.inspect_representative_structures(project_dir)
    assert state.representative_count == 2

    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=project_dir,
        initial_input_path=root_dir,
    )
    assert window.stoichiometry_table.rowCount() == 2
    assert window.stoichiometry_table.item(0, 3).text() == "Complete"
    assert window.stoichiometry_table.item(1, 3).text() == "Complete"
    assert (
        window.run_status_label.text()
        == "Representative selection: restored from saved project analysis"
    )
    window.close()


def test_representativefinder_window_builds_split_scrollable_layout(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    stoich_dir = _build_representative_test_folder(tmp_path)
    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=stoich_dir,
    )

    assert window.windowTitle() == "Representative Structures"
    assert isinstance(window._pane_splitter, QSplitter)
    assert isinstance(window._left_scroll, QScrollArea)
    assert isinstance(window._right_scroll, QScrollArea)
    assert isinstance(window._right_splitter, QSplitter)
    assert window._right_splitter.count() == 5
    assert window.stoichiometry_table.columnCount() == 8
    assert isinstance(window.viewer_widget, ElectronDensityStructureViewer)
    open_button = window.stoichiometry_table.cellWidget(0, 7)
    assert isinstance(open_button, QPushButton)
    assert open_button.isEnabled() is False
    assert window.input_dir_edit.text() == str(stoich_dir.resolve())
    assert (
        "Discovered stoichiometries: 1"
        in window.input_preview_box.toPlainText()
    )
    assert window.stoichiometry_table.rowCount() == 1
    assert window.run_button.text() == "Analyze Selected Stoichiometry"
    assert window.overwrite_existing_checkbox.isChecked() is False
    assert "representativefinder_PbI2" in window.output_dir_edit.text()
    assert window.plot_widget.distribution_selector_combo.count() == 0
    assert window.plot_widget.distribution_selector_combo.isEnabled() is False
    assert window.solvent_shell_toggle_button.isChecked() is False
    assert window.solvent_shell_body.isVisible() is False

    window.show()
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    right_sizes = window._right_splitter.sizes()
    assert len(right_sizes) == 5
    assert right_sizes[0] >= right_sizes[1]
    assert right_sizes[0] >= right_sizes[2]

    window.load_preset("DMF")
    assert window.bond_pair_table.rowCount() == 7
    assert window.angle_triplet_table.rowCount() == 5
    window.close()


def test_representativefinder_window_switches_display_by_stoichiometry_row(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    root_dir, pb_dir, sn_dir = _build_multi_stoichiometry_root(tmp_path)
    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )

    assert window.stoichiometry_table.rowCount() == 2
    assert (
        "Discovered stoichiometries: 2"
        in window.input_preview_box.toPlainText()
    )
    assert (
        "representativefinder_batch_cluster_root"
        in window.output_dir_edit.text()
    )

    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    pb_result = analyze_representative_structure_folder(
        pb_dir,
        settings=settings,
        output_dir=tmp_path / "pb_output",
    )
    sn_result = analyze_representative_structure_folder(
        sn_dir,
        settings=settings,
        output_dir=tmp_path / "sn_output",
    )

    window._on_target_result_ready(pb_result)
    window._on_target_result_ready(sn_result)

    opened_paths: list[Path] = []
    monkeypatch.setattr(
        window,
        "_reveal_path_in_file_manager",
        lambda path: opened_paths.append(path),
    )

    window._select_stoichiometry_row_by_key(str(sn_dir.resolve()))
    assert "Stoichiometry: SnBr2" in window.result_summary_box.toPlainText()
    assert window.candidate_table.rowCount() == len(sn_result.candidates)
    assert window.plot_widget.distribution_selector_combo.count() == 2
    assert window.viewer_widget.current_structure is not None
    assert window.viewer_widget.current_structure.display_label == (
        sn_result.selected_candidate.relative_label
    )
    assert window.viewer_widget.current_mesh_geometry is not None
    sn_labels = [
        window.plot_widget.distribution_selector_combo.itemText(index)
        for index in range(
            window.plot_widget.distribution_selector_combo.count()
        )
    ]
    assert any("Sn-Br" in label for label in sn_labels)
    assert all("Pb-I" not in label for label in sn_labels)
    assert window.stoichiometry_table.item(1, 3).text() == "Complete"
    sn_open_button = window.stoichiometry_table.cellWidget(1, 7)
    assert isinstance(sn_open_button, QPushButton)
    assert sn_open_button.isEnabled() is True
    sn_open_button.click()
    assert opened_paths[-1] == window._project_representative_path_for_key(
        str(sn_dir.resolve())
    )

    window._select_stoichiometry_row_by_key(str(pb_dir.resolve()))
    assert "Stoichiometry: PbI2" in window.result_summary_box.toPlainText()
    assert window.candidate_table.rowCount() == len(pb_result.candidates)
    assert window.plot_widget.distribution_selector_combo.count() == 2
    assert window.viewer_widget.current_structure is not None
    assert window.viewer_widget.current_structure.display_label == (
        pb_result.selected_candidate.relative_label
    )
    assert window.viewer_widget.current_mesh_geometry is not None
    pb_labels = [
        window.plot_widget.distribution_selector_combo.itemText(index)
        for index in range(
            window.plot_widget.distribution_selector_combo.count()
        )
    ]
    assert any("Pb-I" in label for label in pb_labels)
    assert all("Sn-Br" not in label for label in pb_labels)
    assert window.stoichiometry_table.item(0, 3).text() == "Complete"
    pb_open_button = window.stoichiometry_table.cellWidget(0, 7)
    assert isinstance(pb_open_button, QPushButton)
    assert pb_open_button.isEnabled() is True
    pb_open_button.click()
    assert opened_paths[-1] == window._project_representative_path_for_key(
        str(pb_dir.resolve())
    )
    window.close()


def test_representativefinder_window_switches_between_observed_and_predicted_outputs(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    stoich_dir = _build_representative_test_folder(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
        generate_predicted_optimized_representative=True,
    )
    result = analyze_representative_structure_folder(
        stoich_dir,
        settings=settings,
        output_dir=tmp_path / "representative_output",
    )

    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=stoich_dir,
    )
    window._on_target_result_ready(result)

    assert window.display_mode_combo.count() == 3
    display_modes = [
        window.display_mode_combo.itemData(index)
        for index in range(window.display_mode_combo.count())
    ]
    assert display_modes == [
        "selected_candidate",
        "observed_representative",
        "predicted_optimized_representative",
    ]

    assert (
        window._active_representative_input_path()
        == result.selected_candidate.file_path
    )

    predicted_index = display_modes.index("predicted_optimized_representative")
    window.display_mode_combo.setCurrentIndex(predicted_index)

    assert "Displayed structure: Predicted Optimized Representative" in (
        window.result_summary_box.toPlainText()
    )
    assert (
        window._active_representative_input_path()
        == result.predicted_output_path
    )
    assert window.viewer_widget.current_structure is not None
    assert (
        window.viewer_widget.current_structure.display_label
        == "Predicted Optimized Representative"
    )

    observed_index = display_modes.index("observed_representative")
    window.display_mode_combo.setCurrentIndex(observed_index)
    assert "Displayed structure: Observed Representative" in (
        window.result_summary_box.toPlainText()
    )
    assert window._active_representative_input_path() == (
        window._project_representative_path_for_key(str(stoich_dir.resolve()))
    )
    window.close()


def test_representativefinder_window_restores_project_session_results(
    qapp,
    tmp_path,
    monkeypatch,
):
    del qapp
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    root_dir, pb_dir, sn_dir = _build_multi_stoichiometry_root(tmp_path)
    settings = RepresentativeFinderSettings(
        bond_pairs=(
            BondPairDefinition("Pb", "I", 3.2),
            BondPairDefinition("Sn", "Br", 3.2),
        ),
        angle_triplets=(
            AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
            AngleTripletDefinition("Sn", "Br", "Br", 3.2, 3.2),
        ),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
    )
    pb_result = analyze_representative_structure_folder(
        pb_dir,
        settings=settings,
        output_dir=tmp_path / "pb_output",
    )
    sn_result = analyze_representative_structure_folder(
        sn_dir,
        settings=settings,
        output_dir=tmp_path / "sn_output",
    )

    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )
    window.analysis_mode_combo.setCurrentIndex(1)
    window.output_dir_edit.setText(str(tmp_path / "restored_batch_output"))
    window._on_target_result_ready(pb_result)
    window._on_target_result_ready(sn_result)
    window._select_stoichiometry_row_by_key(str(sn_dir.resolve()))
    window.close()

    restored_window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=root_dir,
    )

    assert (
        restored_window.run_status_label.text()
        == "Representative selection: restored from project session"
    )
    assert restored_window._current_analysis_mode() == "all"
    assert restored_window.output_dir_edit.text() == str(
        tmp_path / "restored_batch_output"
    )
    assert restored_window.stoichiometry_table.item(0, 3).text() == "Complete"
    assert restored_window.stoichiometry_table.item(1, 3).text() == "Complete"
    assert (
        "Stoichiometry: SnBr2"
        in restored_window.result_summary_box.toPlainText()
    )
    assert restored_window.candidate_table.rowCount() == len(
        sn_result.candidates
    )
    assert restored_window.viewer_widget.current_structure is not None
    assert restored_window.viewer_widget.current_structure.display_label == (
        sn_result.selected_candidate.relative_label
    )
    restored_window.close()


def test_representativefinder_window_close_cancels_active_analysis(
    qapp,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    monkeypatch.setattr(
        "saxshell.representativefinder.ui.main_window.load_presets",
        lambda: {
            "Test": BondAnalysisPreset(
                name="Test",
                bond_pairs=(BondPairDefinition("Pb", "I", 3.2),),
                angle_triplets=(
                    AngleTripletDefinition("Pb", "I", "I", 3.2, 3.2),
                ),
            )
        },
    )

    cancel_seen: list[bool] = []

    def _fake_analyze(
        _input_dir,
        *,
        settings,
        output_dir,
        project_dir=None,
        progress_callback=None,
        log_callback=None,
        cancel_callback=None,
    ):
        del settings, output_dir, project_dir
        if log_callback is not None:
            log_callback("Starting fake analysis.")
        start = time.monotonic()
        while time.monotonic() - start < 1.0:
            if cancel_callback is not None and cancel_callback():
                cancel_seen.append(True)
                raise RepresentativeFinderOperationCancelled(
                    "Representative-structure analysis canceled."
                )
            time.sleep(0.01)
            if progress_callback is not None:
                progress_callback(0, 1, "Waiting for cancellation...")
        raise AssertionError(
            "Representative finder close did not cancel the worker."
        )

    monkeypatch.setattr(
        "saxshell.representativefinder.ui.main_window.analyze_representative_structure_folder",
        _fake_analyze,
    )

    stoich_dir = _build_representative_test_folder(tmp_path)
    output_dir = tmp_path / "representative_output"
    window = RepresentativeStructureFinderMainWindow(
        initial_project_dir=tmp_path,
        initial_input_path=stoich_dir,
    )
    window.output_dir_edit.setText(str(output_dir))
    window.show()
    qapp.processEvents()

    window._run_analysis()

    deadline = time.monotonic() + 2.0
    while (
        window._analysis_thread is None
        or not window._analysis_thread.isRunning()
    ) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert window._analysis_thread is not None
    assert window._analysis_thread.isRunning()

    window.close()

    deadline = time.monotonic() + 2.0
    while window._analysis_thread is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert window._analysis_thread is None
    assert window.isVisible() is False
    assert cancel_seen in ([], [True])
