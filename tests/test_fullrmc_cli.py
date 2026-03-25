from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

import saxshell.fullrmc.cli as fullrmc_cli
from saxshell.fullrmc import (
    ConstraintGenerationSettings,
    PackmolPlanningSettings,
    PackmolSetupSettings,
    RepresentativeSelectionSettings,
    SolutionPropertiesSettings,
    SolventHandlingSettings,
    build_constraint_generation,
    build_distribution_selection,
    build_packmol_plan,
    build_packmol_setup,
    build_representative_preview_clusters,
    build_representative_solvent_outputs,
    calculate_solution_properties,
    load_constraint_generation_metadata,
    load_packmol_planning_metadata,
    load_packmol_setup_metadata,
    load_representative_selection_metadata,
    load_rmc_project_source,
    load_solvent_handling_metadata,
    parse_angle_triplet_text,
    parse_bond_pair_text,
    save_solution_properties_metadata,
    select_distribution_representatives,
    select_first_file_representatives,
)
from saxshell.fullrmc.cli import main as fullrmc_main
from saxshell.fullrmc.ui.main_window import RMCSetupMainWindow
from saxshell.saxs.dream import DreamRunSettings
from saxshell.saxs.project_manager import (
    DreamBestFitSelection,
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.stoichiometry import format_stoich_for_axis
from saxshell.saxshell import main as saxshell_main
from saxshell.structure import PDBStructure


def _build_sample_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "rmcsetup_source"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)
    clusters_dir = _build_sample_clusters_dir(tmp_path)

    run_a = _write_sample_dream_run(
        paths.dream_runtime_dir / "dream_run_001",
        settings=DreamRunSettings(
            bestfit_method="map",
            posterior_filter_mode="all_post_burnin",
            credible_interval_low=10.0,
            credible_interval_high=90.0,
            model_name="template_pd_likelihood_monosq_decoupled",
        ),
        template_name="template_pd_likelihood_monosq_decoupled",
    )
    run_b = _write_sample_dream_run(
        paths.dream_runtime_dir / "dream_run_002",
        settings=DreamRunSettings(
            bestfit_method="median",
            posterior_filter_mode="top_percent_logp",
            posterior_top_percent=7.5,
            posterior_top_n=200,
            credible_interval_low=20.0,
            credible_interval_high=80.0,
            model_name="template_pd_likelihood_monosq_decoupled",
        ),
        template_name="template_pd_likelihood_monosq_decoupled",
    )

    favorite = DreamBestFitSelection(
        run_name=run_a.name,
        run_relative_path=str(run_a.relative_to(project_dir)),
        bestfit_method="chain_mean",
        posterior_filter_mode="top_n_logp",
        posterior_top_percent=10.0,
        posterior_top_n=42,
        credible_interval_low=25.0,
        credible_interval_high=75.0,
        label="2026-03-23T10:00:00 • dream_run_001 • chain_mean",
        template_name="template_pd_likelihood_monosq_decoupled",
        model_name="template_pd_likelihood_monosq_decoupled",
        selection_source="rmcsetup",
        selected_at="2026-03-23T10:00:00",
    )
    history_entry = DreamBestFitSelection(
        run_name=run_b.name,
        run_relative_path=str(run_b.relative_to(project_dir)),
        bestfit_method="median",
        posterior_filter_mode="top_percent_logp",
        posterior_top_percent=7.5,
        posterior_top_n=200,
        credible_interval_low=20.0,
        credible_interval_high=80.0,
        label="2026-03-22T09:00:00 • dream_run_002 • median",
        template_name="template_pd_likelihood_monosq_decoupled",
        model_name="template_pd_likelihood_monosq_decoupled",
        selection_source="rmcsetup",
        selected_at="2026-03-22T09:00:00",
    )
    settings.dream_favorite_selection = favorite
    settings.dream_favorite_history = [history_entry]
    settings.clusters_dir = str(clusters_dir)
    settings.cluster_inventory_rows = [
        {
            "structure": "PbI2",
            "motif": "no_motif",
            "count": 2,
            "source_dir": str(clusters_dir / "PbI2"),
        },
        {
            "structure": "PbI2O",
            "motif": "motif_1",
            "count": 1,
            "source_dir": str(clusters_dir / "PbI2O" / "motif_1"),
        },
    ]
    manager.save_project(settings)
    return project_dir, paths


def _write_sample_dream_run(
    run_dir: Path,
    *,
    settings: DreamRunSettings,
    template_name: str,
    weight_entries: list[dict[str, object]] | None = None,
    sampled_params: np.ndarray | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pd_settings.json").write_text(
        json.dumps(settings.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_weight_entries = [
        dict(entry)
        for entry in (
            weight_entries
            or [
                {
                    "structure": "PbI2",
                    "motif": "no_motif",
                    "param_type": "Both",
                    "param": "w0",
                    "value": 0.6,
                    "vary": True,
                    "distribution": "lognorm",
                    "dist_params": {
                        "loc": 0.0,
                        "scale": 0.6,
                        "s": 0.3,
                    },
                },
                {
                    "structure": "PbI2O",
                    "motif": "motif_1",
                    "param_type": "Both",
                    "param": "w1",
                    "value": 0.4,
                    "vary": True,
                    "distribution": "lognorm",
                    "dist_params": {
                        "loc": 0.0,
                        "scale": 0.4,
                        "s": 0.3,
                    },
                },
            ]
        )
    ]
    resolved_sampled_params = (
        np.asarray(sampled_params, dtype=float)
        if sampled_params is not None
        else np.asarray([[[0.0, 1.0], [0.25, 0.75]]], dtype=float)
    )
    fit_parameter_entries = [
        {
            "structure": "",
            "motif": "",
            "param_type": "SAXS",
            "param": "solv_w",
            "value": 0.0,
            "vary": False,
            "distribution": "norm",
            "dist_params": {
                "loc": 0.0,
                "scale": 0.1,
            },
        },
        {
            "structure": "",
            "motif": "",
            "param_type": "SAXS",
            "param": "offset",
            "value": 0.0,
            "vary": False,
            "distribution": "norm",
            "dist_params": {
                "loc": 0.0,
                "scale": 0.1,
            },
        },
        {
            "structure": "",
            "motif": "",
            "param_type": "SAXS",
            "param": "eff_r",
            "value": 9.0,
            "vary": False,
            "distribution": "norm",
            "dist_params": {
                "loc": 9.0,
                "scale": 0.5,
            },
        },
        {
            "structure": "",
            "motif": "",
            "param_type": "SAXS",
            "param": "vol_frac",
            "value": 0.0,
            "vary": False,
            "distribution": "norm",
            "dist_params": {
                "loc": 0.0,
                "scale": 0.01,
            },
        },
        {
            "structure": "",
            "motif": "",
            "param_type": "SAXS",
            "param": "scale",
            "value": 1.0,
            "vary": False,
            "distribution": "norm",
            "dist_params": {
                "loc": 1.0,
                "scale": 0.1,
            },
        },
    ]
    metadata = {
        "settings": settings.to_dict(),
        "template_name": template_name,
        "full_parameter_names": [
            *[
                str(entry.get("param", "")).strip()
                for entry in resolved_weight_entries
            ],
            "solv_w",
            "offset",
            "eff_r",
            "vol_frac",
            "scale",
        ],
        "active_parameter_indices": list(range(len(resolved_weight_entries))),
        "fixed_parameter_values": [
            *([0.0] * len(resolved_weight_entries)),
            0.0,
            0.0,
            9.0,
            0.0,
            1.0,
        ],
        "active_parameter_entries": resolved_weight_entries,
        "parameter_map": [
            *resolved_weight_entries,
            *fit_parameter_entries,
        ],
        "q_values": [0.05, 0.1],
        "experimental_intensities": [1.0, 0.8],
        "theoretical_intensities": [[0.8, 0.7], [0.6, 0.5]],
        "solvent_intensities": [0.0, 0.0],
    }
    (run_dir / "dream_runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(
        run_dir / "dream_sampled_params.npy",
        resolved_sampled_params,
    )
    np.save(
        run_dir / "dream_log_ps.npy",
        np.asarray([[1.0, 2.0]], dtype=float),
    )
    return run_dir


def _build_sample_saxs_project_with_single_atom_model(tmp_path):
    project_dir, paths = _build_sample_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    single_atom_dir = tmp_path / "single_atom_models"
    single_atom_dir.mkdir(parents=True, exist_ok=True)
    single_atom_path = single_atom_dir / "Zn1_reference.xyz"
    single_atom_path.write_text(
        "\n".join(
            [
                "1",
                "single Zn reference",
                "Zn 0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    settings.cluster_inventory_rows = [
        *settings.cluster_inventory_rows,
        {
            "structure": "Zn1",
            "motif": "no_motif",
            "count": 1,
            "source_kind": "single_structure_file",
            "source_file": str(single_atom_path),
            "source_file_name": single_atom_path.name,
        },
    ]
    manager.save_project(settings)

    _write_sample_dream_run(
        paths.dream_runtime_dir / "dream_run_001",
        settings=DreamRunSettings(
            bestfit_method="map",
            posterior_filter_mode="all_post_burnin",
            credible_interval_low=10.0,
            credible_interval_high=90.0,
            model_name="template_pd_likelihood_monosq_decoupled",
        ),
        template_name="template_pd_likelihood_monosq_decoupled",
        weight_entries=[
            {
                "structure": "PbI2",
                "motif": "no_motif",
                "param_type": "Both",
                "param": "w0",
                "value": 0.6,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.6,
                    "s": 0.3,
                },
            },
            {
                "structure": "PbI2O",
                "motif": "motif_1",
                "param_type": "Both",
                "param": "w1",
                "value": 0.4,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.4,
                    "s": 0.3,
                },
            },
            {
                "structure": "Zn1",
                "motif": "no_motif",
                "param_type": "Both",
                "param": "w2",
                "value": 0.2,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.2,
                    "s": 0.3,
                },
            },
        ],
        sampled_params=np.asarray(
            [[[0.0, 1.0, 0.2], [0.25, 0.75, 0.5]]],
            dtype=float,
        ),
    )
    return project_dir, paths, single_atom_path.resolve()


def _build_sample_clusters_dir(tmp_path: Path) -> Path:
    clusters_dir = tmp_path / "clusters_splitxyz0001"
    pbi2_dir = clusters_dir / "PbI2"
    motif_dir = clusters_dir / "PbI2O" / "motif_1"
    pbi2_dir.mkdir(parents=True)
    motif_dir.mkdir(parents=True)
    (pbi2_dir / "frame_0001.xyz").write_text("corrupt xyz\n", encoding="utf-8")
    (pbi2_dir / "frame_0002.xyz").write_text(
        "\n".join(
            [
                "3",
                "PbI2 cluster",
                "Pb 0.0 0.0 0.0",
                "I 1.0 0.0 0.0",
                "I 0.0 1.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (motif_dir / "frame_0003.xyz").write_text(
        "\n".join(
            [
                "4",
                "PbI2O cluster",
                "Pb 0.0 0.0 0.0",
                "I 1.0 0.0 0.0",
                "I 0.0 1.0 0.0",
                "O 0.0 0.0 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return clusters_dir


def _write_sample_bondanalysis_results(project_dir: Path) -> Path:
    output_dir = project_dir / "bondanalysis_cached"
    cluster_root = output_dir / "cluster_types"
    pbi2_dir = cluster_root / "PbI2"
    pbi2o_dir = cluster_root / "PbI2O"
    pbi2_dir.mkdir(parents=True, exist_ok=True)
    pbi2o_dir.mkdir(parents=True, exist_ok=True)

    bond_dtype = [
        ("cluster_type", "U12"),
        ("structure_file", "U20"),
        ("value", np.float64),
    ]
    np.save(
        pbi2_dir / "Pb_I_distribution.npy",
        np.array(
            [
                ("PbI2", "frame_0002.xyz", 1.0),
                ("PbI2", "frame_0002.xyz", 1.0),
            ],
            dtype=bond_dtype,
        ),
    )
    np.save(
        pbi2_dir / "Pb_I_I_angles.npy",
        np.array(
            [("PbI2", "frame_0002.xyz", 90.0)],
            dtype=bond_dtype,
        ),
    )
    np.save(
        pbi2o_dir / "Pb_I_distribution.npy",
        np.array(
            [
                ("PbI2O", "frame_0003.xyz", 1.0),
                ("PbI2O", "frame_0003.xyz", 1.0),
            ],
            dtype=bond_dtype,
        ),
    )
    np.save(
        pbi2o_dir / "Pb_I_I_angles.npy",
        np.array(
            [("PbI2O", "frame_0003.xyz", 90.0)],
            dtype=bond_dtype,
        ),
    )

    results_index = {
        "clusters_dir": str(project_dir.parent / "clusters_splitxyz0001"),
        "output_dir": str(output_dir),
        "selected_cluster_types": ["PbI2", "PbI2O"],
        "total_structure_files": 2,
        "bond_pairs": [{"atom1": "Pb", "atom2": "I", "cutoff_angstrom": 3.5}],
        "angle_triplets": [
            {
                "vertex": "Pb",
                "arm1": "I",
                "arm2": "I",
                "cutoff1_angstrom": 3.5,
                "cutoff2_angstrom": 3.5,
            }
        ],
        "cluster_results": [
            {
                "cluster_type": "PbI2",
                "structure_count": 1,
                "output_dir": str(pbi2_dir),
                "bond_value_counts": {"Pb-I": 2},
                "angle_value_counts": {"I-Pb-I": 1},
            },
            {
                "cluster_type": "PbI2O",
                "structure_count": 1,
                "output_dir": str(pbi2o_dir),
                "bond_value_counts": {"Pb-I": 2},
                "angle_value_counts": {"I-Pb-I": 1},
            },
        ],
        "aggregate_output_dir": str(output_dir / "all_clusters"),
        "comparison_output_dir": str(output_dir / "comparisons"),
    }
    results_index_path = output_dir / "bondanalysis_results_index.json"
    results_index_path.write_text(
        json.dumps(results_index, indent=2) + "\n",
        encoding="utf-8",
    )
    return results_index_path


def _write_custom_solvent_pdb(tmp_path: Path) -> Path:
    solvent_path = tmp_path / "water_ref.pdb"
    solvent_path.write_text(
        "\n".join(
            [
                "ATOM      1 O1   HOH X   1       0.000   0.000   0.000  1.00  0.00           O",
                "ATOM      2 H1   HOH X   1       0.957   0.000   0.000  1.00  0.00           H",
                "ATOM      3 H2   HOH X   1      -0.239   0.927   0.000  1.00  0.00           H",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return solvent_path


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_for_representative_worker(
    window: RMCSetupMainWindow,
    *,
    timeout_seconds: float = 10.0,
) -> None:
    app = qapp()
    deadline = time.monotonic() + timeout_seconds
    while (
        window._representative_thread is not None
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert window._representative_thread is None


def test_project_settings_roundtrip_preserves_dream_favorite_data(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    settings = SAXSProjectManager().load_project(project_dir)

    assert settings.dream_favorite_selection is not None
    assert settings.dream_favorite_selection.bestfit_method == "chain_mean"
    assert settings.dream_favorite_selection.run_relative_path.endswith(
        "dream_run_001"
    )
    assert len(settings.dream_favorite_history) == 1
    assert settings.dream_favorite_history[0].bestfit_method == "median"
    assert [row["count"] for row in settings.cluster_inventory_rows] == [2, 1]


def test_fullrmc_project_loader_discovers_valid_runs_and_favorites(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    state = load_rmc_project_source(project_dir)

    assert [run.run_name for run in state.valid_runs] == [
        "dream_run_002",
        "dream_run_001",
    ]
    assert state.favorite_selection is not None
    assert state.favorite_selection.bestfit_method == "chain_mean"
    assert len(state.favorite_history) == 1
    assert state.find_run_for_selection(state.favorite_selection) is not None
    assert state.cluster_validation.is_valid is True
    assert state.cluster_validation.current_rows[0]["count"] == 2
    assert state.rmcsetup_paths.rmcsetup_dir.is_dir()
    assert state.rmcsetup_paths.representative_clusters_dir.is_dir()
    assert state.rmcsetup_paths.pdb_no_solvent_dir.is_dir()
    assert state.rmcsetup_paths.pdb_with_solvent_dir.is_dir()
    assert state.rmcsetup_paths.packmol_inputs_dir.is_dir()
    assert state.rmcsetup_paths.constraints_dir.is_dir()
    assert state.rmcsetup_paths.reports_dir.is_dir()
    assert state.rmcsetup_paths.distribution_selection_path.is_file()
    assert state.rmcsetup_paths.solution_properties_path.is_file()
    assert state.rmcsetup_paths.solvent_handling_path.is_file()
    assert state.rmcsetup_paths.representative_selection_path.is_file()


def test_fullrmc_project_loader_detects_cluster_count_drift(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    drift_file = (
        tmp_path
        / "clusters_splitxyz0001"
        / "PbI2O"
        / "motif_1"
        / "frame_0004.xyz"
    )
    drift_file.write_text("new file\n", encoding="utf-8")

    state = load_rmc_project_source(project_dir)

    assert state.cluster_validation.is_valid is False
    assert state.cluster_validation.count_mismatches == [
        {
            "structure": "PbI2O",
            "motif": "motif_1",
            "expected_count": 1,
            "actual_count": 2,
        }
    ]


def test_first_file_distribution_selection_uses_current_dream_choice(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    distribution = build_distribution_selection(state, selection)

    assert distribution.selection_mode == "first_file"
    assert [entry.param for entry in distribution.entries] == ["w1", "w0"]
    assert [entry.structure for entry in distribution.entries] == [
        "PbI2O",
        "PbI2",
    ]
    assert all(entry.is_active for entry in distribution.entries)
    assert distribution.entries[0].cluster_count == 1
    assert distribution.entries[1].cluster_count == 2


def test_distribution_selection_supports_single_atom_model_source_files(
    tmp_path,
):
    project_dir, _paths, single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    distribution = build_distribution_selection(state, selection)
    zn_entry = next(
        entry for entry in distribution.entries if entry.structure == "Zn1"
    )

    assert state.cluster_validation.is_valid is True
    assert zn_entry.param == "w2"
    assert zn_entry.cluster_count == 1
    assert zn_entry.source_kind == "single_structure_file"
    assert zn_entry.source_file == str(single_atom_path)
    assert zn_entry.source_file_name == single_atom_path.name


def test_first_file_representative_selection_writes_metadata(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_first_file_representatives(state, selection)
    reloaded = load_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path
    )
    distribution_payload = json.loads(
        state.rmcsetup_paths.distribution_selection_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata.selection_mode == "first_file"
    assert len(metadata.representative_entries) == 2
    assert metadata.missing_bins == []
    assert metadata.invalid_bins == []
    assert [
        entry.source_file_name for entry in metadata.representative_entries
    ] == [
        "frame_0003.xyz",
        "frame_0002.xyz",
    ]
    assert metadata.representative_entries[0].element_counts["O"] == 1
    assert metadata.representative_entries[1].atom_count == 3
    assert reloaded is not None
    assert len(reloaded.representative_entries) == 2
    assert distribution_payload["entries"][0]["param"] == "w1"


def test_first_file_representative_selection_supports_single_atom_sources(
    tmp_path,
):
    project_dir, _paths, single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_first_file_representatives(state, selection)
    zn_entry = next(
        entry
        for entry in metadata.representative_entries
        if entry.structure == "Zn1"
    )

    assert len(metadata.representative_entries) == 3
    assert metadata.missing_bins == []
    assert metadata.invalid_bins == []
    assert zn_entry.param == "w2"
    assert zn_entry.source_file == str(single_atom_path)
    assert zn_entry.source_file_name == single_atom_path.name
    assert zn_entry.atom_count == 1
    assert zn_entry.element_counts == {"Zn": 1}


def test_distribution_representative_selection_uses_cached_bondanalysis_results(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    # Add a second valid candidate so the score-based selection has a choice.
    alternate_candidate = (
        tmp_path / "clusters_splitxyz0001" / "PbI2" / "frame_0003.xyz"
    )
    alternate_candidate.write_text(
        "\n".join(
            [
                "3",
                "PbI2 alternate cluster",
                "Pb 0.0 0.0 0.0",
                "I 2.0 0.0 0.0",
                "I 0.0 2.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results_index_path = _write_sample_bondanalysis_results(project_dir)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_distribution_representatives(
        state,
        selection,
        settings=RepresentativeSelectionSettings(
            selection_mode="bond_angle_distribution",
        ),
    )

    assert metadata.selection_mode == "bond_angle_distribution"
    assert len(metadata.representative_entries) == 2
    pbi2_entry = next(
        entry
        for entry in metadata.representative_entries
        if entry.structure == "PbI2"
    )
    assert pbi2_entry.analysis_source == "cached_bondanalysis"
    assert pbi2_entry.source_file_name == "frame_0002.xyz"
    assert pbi2_entry.cached_results_path == str(results_index_path)
    assert metadata.settings.bond_pairs == ()


def test_distribution_representative_selection_can_recompute_without_cache(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_distribution_representatives(
        state,
        selection,
        settings=RepresentativeSelectionSettings(
            selection_mode="bond_angle_distribution",
            bond_pairs=parse_bond_pair_text("Pb-I:3.5"),
            angle_triplets=parse_angle_triplet_text("I-Pb-I:3.5,3.5"),
        ),
    )

    assert metadata.selection_mode == "bond_angle_distribution"
    assert len(metadata.representative_entries) == 2
    assert metadata.missing_bins == []
    assert metadata.invalid_bins == []
    assert all(
        entry.analysis_source == "recomputed"
        for entry in metadata.representative_entries
    )
    assert metadata.representative_entries[0].score_total is not None


def test_representative_preview_uses_cached_bondanalysis_data(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    results_index_path = _write_sample_bondanalysis_results(project_dir)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_distribution_representatives(
        state,
        selection,
        settings=RepresentativeSelectionSettings(
            selection_mode="bond_angle_distribution",
        ),
    )
    preview_clusters = build_representative_preview_clusters(state, metadata)

    assert len(preview_clusters) == 2
    pbi2_preview = next(
        preview for preview in preview_clusters if preview.structure == "PbI2"
    )
    assert pbi2_preview.analysis_source == "cached_bondanalysis"
    assert pbi2_preview.tab_label == "PbI2"
    assert [series.display_label for series in pbi2_preview.bond_series] == [
        "Pb-I"
    ]
    assert pbi2_preview.bond_series[0].distribution_values.tolist() == [
        1.0,
        1.0,
    ]
    assert pbi2_preview.angle_series[0].representative_values == (90.0,)
    cached_entry = next(
        entry
        for entry in metadata.representative_entries
        if entry.structure == "PbI2"
    )
    assert cached_entry.cached_results_path == str(results_index_path)


def test_representative_preview_can_recompute_distribution_data(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None

    metadata = select_distribution_representatives(
        state,
        selection,
        settings=RepresentativeSelectionSettings(
            selection_mode="bond_angle_distribution",
            bond_pairs=parse_bond_pair_text("Pb-I:3.5"),
            angle_triplets=parse_angle_triplet_text("I-Pb-I:3.5,3.5"),
        ),
    )
    preview_clusters = build_representative_preview_clusters(state, metadata)

    assert len(preview_clusters) == 2
    pbi2_preview = next(
        preview for preview in preview_clusters if preview.structure == "PbI2"
    )
    assert pbi2_preview.analysis_source == "recomputed"
    assert pbi2_preview.bond_series[0].distribution_values.size >= 2
    assert pbi2_preview.bond_series[0].representative_values == (1.0,)
    assert pbi2_preview.angle_series[0].representative_values == (90.0,)


def test_build_representative_solvent_outputs_with_preset_reference(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )

    metadata = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )
    reloaded = load_solvent_handling_metadata(
        state.rmcsetup_paths.solvent_handling_path
    )

    assert metadata.reference_name == "dmf"
    assert len(metadata.entries) == 2
    assert all(
        entry.atom_count_completed > entry.atom_count_no_solvent
        for entry in metadata.entries
    )
    assert all(entry.solvent_atoms_added > 0 for entry in metadata.entries)
    assert reloaded is not None
    assert reloaded.settings.preset_name == "dmf"
    assert reloaded.settings.minimum_solvent_atom_separation_a == (
        pytest.approx(1.2)
    )


def test_partial_coordinated_solvent_replaces_anchor_atoms_and_points_outward(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )

    metadata = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
            minimum_solvent_atom_separation_a=1.2,
        ),
    )

    anchored_entry = next(
        entry
        for entry in metadata.entries
        if entry.structure == "PbI2O" and entry.motif == "motif_1"
    )
    assert anchored_entry.atom_count_no_solvent == 4
    assert anchored_entry.atom_count_completed == 15
    assert anchored_entry.solvent_atoms_added == 11
    assert anchored_entry.solvent_molecules_added == 1
    assert anchored_entry.completion_strategy.startswith(
        "anchored_solvent_completion"
    )

    completed_structure = PDBStructure.from_file(anchored_entry.completed_pdb)
    solvent_atoms = [
        atom
        for atom in completed_structure.atoms
        if atom.residue_name == "DMF"
    ]
    solute_atoms = [
        atom
        for atom in completed_structure.atoms
        if atom.residue_name != "DMF"
    ]
    assert len(solvent_atoms) == 12
    assert len(solute_atoms) == 3

    anchor_atom = next(atom for atom in solvent_atoms if atom.element == "O")
    assert anchor_atom.coordinates == pytest.approx([0.0, 0.0, 1.0], abs=1e-3)

    solute_center = np.mean(
        [atom.coordinates for atom in solute_atoms],
        axis=0,
    )
    solvent_body_center = np.mean(
        [
            atom.coordinates
            for atom in solvent_atoms
            if atom.atom_id != anchor_atom.atom_id
        ],
        axis=0,
    )
    outward_alignment = np.dot(
        solvent_body_center - anchor_atom.coordinates,
        anchor_atom.coordinates - solute_center,
    )
    assert outward_alignment > 0.0

    minimum_distance = min(
        np.linalg.norm(solvent_atom.coordinates - solute_atom.coordinates)
        for solvent_atom in solvent_atoms
        if solvent_atom.atom_id != anchor_atom.atom_id
        for solute_atom in solute_atoms
    )
    assert minimum_distance >= 1.2 - 1e-6


def test_build_representative_solvent_outputs_with_custom_reference(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    custom_solvent = _write_custom_solvent_pdb(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )

    metadata = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="custom",
            custom_reference_path=str(custom_solvent),
        ),
    )

    assert metadata.reference_name == "water_ref"
    assert metadata.reference_residue_name == "HOH"
    added_counts = {
        (entry.structure, entry.motif): entry.solvent_atoms_added
        for entry in metadata.entries
    }
    molecule_counts = {
        (entry.structure, entry.motif): entry.solvent_molecules_added
        for entry in metadata.entries
    }
    assert added_counts[("PbI2", "no_motif")] == 3
    assert added_counts[("PbI2O", "motif_1")] == 2
    assert molecule_counts[("PbI2", "no_motif")] == 1
    assert molecule_counts[("PbI2O", "motif_1")] == 1


def test_build_representative_solvent_outputs_preserves_single_atom_sources(
    tmp_path,
):
    project_dir, _paths, _single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )

    metadata = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )

    zn_entry = next(
        entry for entry in metadata.entries if entry.structure == "Zn1"
    )
    completed_structure = PDBStructure.from_file(zn_entry.completed_pdb)

    assert len(metadata.entries) == 3
    assert zn_entry.atom_count_no_solvent == 1
    assert zn_entry.atom_count_completed == 1
    assert zn_entry.solvent_atoms_added == 0
    assert zn_entry.solvent_molecules_added == 0
    assert zn_entry.completion_strategy == "preserved_single_structure_file"
    assert [atom.element for atom in completed_structure.atoms] == ["Zn"]


def test_build_packmol_plan_writes_metadata_and_reports(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    solution_settings = SolutionPropertiesSettings(
        mode="mass",
        solution_density=1.05,
        solute_stoich="Pb1I2",
        solvent_stoich="C3H7NO",
        molar_mass_solute=461.0,
        molar_mass_solvent=73.09,
        mass_solute=4.61,
        mass_solvent=95.39,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=solution_settings,
        result=calculate_solution_properties(solution_settings),
    )

    metadata = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )
    reloaded = load_packmol_planning_metadata(
        state.rmcsetup_paths.packmol_plan_path
    )

    assert metadata.entries
    assert sum(entry.planned_count for entry in metadata.entries) > 0
    assert metadata.entries[0].planned_count_weight >= 0
    assert metadata.entries[0].composition_source in {
        "representative_selection",
        "pdb_no_solvent",
    }
    assert reloaded is not None
    assert reloaded.settings.box_side_length_a == pytest.approx(80.0)
    assert state.rmcsetup_paths.packmol_plan_report_path.is_file()
    assert state.rmcsetup_paths.cluster_counts_csv_path.is_file()
    assert state.rmcsetup_paths.planned_count_weights_csv_path.is_file()
    assert state.rmcsetup_paths.planned_atom_weights_csv_path.is_file()
    assert "Counts per cluster bin:" in (
        state.rmcsetup_paths.packmol_plan_report_path.read_text(
            encoding="utf-8"
        )
    )


def test_build_packmol_plan_includes_single_atom_model_sources(tmp_path):
    project_dir, _paths, _single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=SolutionPropertiesSettings(
            mode="mass",
            solution_density=1.05,
            solute_stoich="Pb1I2",
            solvent_stoich="C3H7NO",
            molar_mass_solute=461.0,
            molar_mass_solvent=73.09,
            mass_solute=4.61,
            mass_solvent=95.39,
        ),
        result=calculate_solution_properties(
            SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.05,
                solute_stoich="Pb1I2",
                solvent_stoich="C3H7NO",
                molar_mass_solute=461.0,
                molar_mass_solvent=73.09,
                mass_solute=4.61,
                mass_solvent=95.39,
            )
        ),
    )
    state.solvent_handling = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )

    metadata = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )

    zn_entry = next(
        entry for entry in metadata.entries if entry.structure == "Zn1"
    )

    assert len(metadata.entries) == 3
    assert zn_entry.atom_count == 1
    assert zn_entry.element_counts == {"Zn": 1}
    assert zn_entry.composition_source == "pdb_no_solvent"
    assert zn_entry.planned_count_weight >= 0.0


def test_build_packmol_setup_writes_input_files_and_audit(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=SolutionPropertiesSettings(
            mode="mass",
            solution_density=1.05,
            solute_stoich="Pb1I2",
            solvent_stoich="C3H7NO",
            molar_mass_solute=461.0,
            molar_mass_solvent=73.09,
            mass_solute=4.61,
            mass_solvent=95.39,
        ),
        result=calculate_solution_properties(
            SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.05,
                solute_stoich="Pb1I2",
                solvent_stoich="C3H7NO",
                molar_mass_solute=461.0,
                molar_mass_solvent=73.09,
                mass_solute=4.61,
                mass_solvent=95.39,
            )
        ),
    )
    state.solvent_handling = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )
    state.packmol_planning = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )

    metadata = build_packmol_setup(
        state,
        PackmolSetupSettings(
            tolerance_angstrom=2.2,
        ),
    )
    reloaded = load_packmol_setup_metadata(
        state.rmcsetup_paths.packmol_setup_path
    )

    assert metadata.entries
    assert Path(metadata.packmol_input_path).is_file()
    assert metadata.solvent_pdb_path is not None
    assert Path(metadata.solvent_pdb_path).is_file()
    assert Path(metadata.audit_report_path).is_file()
    assert reloaded is not None
    assert reloaded.free_solvent_molecules >= 0
    assert len({entry.residue_name for entry in metadata.entries}) == len(
        metadata.entries
    )
    packmol_text = Path(metadata.packmol_input_path).read_text(
        encoding="utf-8"
    )
    assert "tolerance 2.200" in packmol_text
    assert "structure dmf_single.pdb" in packmol_text
    assert "structure 001_PbI2O_motif_1_CAA.pdb" in packmol_text
    audit_text = Path(metadata.audit_report_path).read_text(encoding="utf-8")
    assert "# Packmol Build Audit" in audit_text
    assert "Count-normalized weights" in audit_text


def test_build_packmol_setup_includes_single_atom_model_sources(tmp_path):
    project_dir, _paths, _single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=SolutionPropertiesSettings(
            mode="mass",
            solution_density=1.05,
            solute_stoich="Pb1I2",
            solvent_stoich="C3H7NO",
            molar_mass_solute=461.0,
            molar_mass_solvent=73.09,
            mass_solute=4.61,
            mass_solvent=95.39,
        ),
        result=calculate_solution_properties(
            SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.05,
                solute_stoich="Pb1I2",
                solvent_stoich="C3H7NO",
                molar_mass_solute=461.0,
                molar_mass_solvent=73.09,
                mass_solute=4.61,
                mass_solvent=95.39,
            )
        ),
    )
    state.solvent_handling = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )
    state.packmol_planning = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )

    metadata = build_packmol_setup(
        state,
        PackmolSetupSettings(
            tolerance_angstrom=2.2,
        ),
    )

    zn_entry = next(
        entry for entry in metadata.entries if entry.structure == "Zn1"
    )
    packed_structure = PDBStructure.from_file(zn_entry.packmol_pdb)

    assert len(metadata.entries) == 3
    assert zn_entry.atom_count == 1
    assert [atom.element for atom in packed_structure.atoms] == ["Zn"]
    assert Path(zn_entry.source_pdb).is_file()


def test_build_constraint_generation_writes_per_structure_and_merged_files(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    solution_settings = SolutionPropertiesSettings(
        mode="mass",
        solution_density=1.05,
        solute_stoich="Pb1I2",
        solvent_stoich="C3H7NO",
        molar_mass_solute=461.0,
        molar_mass_solvent=73.09,
        mass_solute=4.61,
        mass_solvent=95.39,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=solution_settings,
        result=calculate_solution_properties(solution_settings),
    )
    state.solvent_handling = build_representative_solvent_outputs(
        state,
        SolventHandlingSettings(
            coordinated_solvent_mode="partial_coordinated_solvent",
            reference_source="preset",
            preset_name="dmf",
        ),
    )
    state.packmol_planning = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )
    state.packmol_setup = build_packmol_setup(
        state,
        PackmolSetupSettings(),
    )

    metadata = build_constraint_generation(
        state,
        ConstraintGenerationSettings(
            bond_length_tolerance_angstrom=0.08,
            bond_angle_tolerance_degrees=7.0,
        ),
    )
    reloaded = load_constraint_generation_metadata(
        state.rmcsetup_paths.constraint_generation_path
    )

    assert metadata.entries
    assert reloaded is not None
    assert reloaded.settings.bond_length_tolerance_angstrom == pytest.approx(
        0.08
    )
    assert reloaded.settings.bond_angle_tolerance_degrees == pytest.approx(7.0)
    assert all(
        Path(entry.per_structure_constraints_path).is_file()
        for entry in metadata.entries
    )
    merged_text = Path(metadata.merged_constraints_path).read_text(
        encoding="utf-8"
    )
    assert "BOND_ANGLE_CONSTRAINTS" in merged_text
    assert "BOND_LENGTH_CONSTRAINTS" in merged_text
    assert any(entry.bond_length_count > 0 for entry in metadata.entries)
    assert any(entry.bond_angle_count > 0 for entry in metadata.entries)


def test_solution_properties_support_all_input_modes():
    mass_result = calculate_solution_properties(
        SolutionPropertiesSettings(
            mode="mass",
            solution_density=1.05,
            solute_stoich="Cs1Pb1I3",
            solvent_stoich="H2O",
            molar_mass_solute=620.0,
            molar_mass_solvent=18.015,
            mass_solute=6.2,
            mass_solvent=93.8,
        )
    )
    assert mass_result.mode == "mass"
    assert mass_result.moles_solute == pytest.approx(0.01)
    assert mass_result.element_ratio_string

    mass_percent_result = calculate_solution_properties(
        SolutionPropertiesSettings(
            mode="mass_percent",
            solution_density=0.95,
            solute_stoich="PbI2",
            solvent_stoich="C3H7NO",
            molar_mass_solute=461.0,
            molar_mass_solvent=73.09,
            mass_percent_solute=12.5,
            total_mass_solution=80.0,
        )
    )
    assert mass_percent_result.mode == "mass_percent"
    assert mass_percent_result.total_mass_solution == pytest.approx(80.0)
    assert mass_percent_result.number_density_a3 > 0

    molarity_result = calculate_solution_properties(
        SolutionPropertiesSettings(
            mode="molarity_per_liter",
            solution_density=1.1,
            solute_stoich="Pb1I2",
            solvent_stoich="C3H7NO",
            molar_mass_solute=461.0,
            molar_mass_solvent=73.09,
            molarity=0.5,
            molarity_element="Pb",
        )
    )
    assert molarity_result.mode == "molarity_per_liter"
    assert molarity_result.volume_solution_cm3 == pytest.approx(1000.0)
    assert molarity_result.mass_solvent > 0


def test_rmcsetup_main_window_prefills_project_and_applies_favorite(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window.project_dir_edit.text() == str(project_dir.resolve())
    assert window.dream_group.title() == "Select DREAM Model"
    assert window.favorite_group.title() == "Saved DREAM Model"
    assert window.dream_run_combo.count() == 2
    assert window.dream_run_combo.currentText() == "dream_run_001"
    assert window.bestfit_method_combo.currentData() == "chain_mean"
    assert window.posterior_filter_combo.currentData() == "top_n_logp"
    assert window.posterior_top_n_spin.value() == 42
    assert window.favorite_history_combo.count() == 1
    assert "Valid DREAM runs discovered: 2" in (
        window.project_summary_box.toPlainText()
    )
    assert "Matches saved DREAM model: yes" in (
        window.dream_source_summary_box.toPlainText()
    )
    assert window._project_source_state is not None
    assert window._project_source_state.cluster_validation.is_valid is True
    assert not hasattr(window, "cluster_validation_box")
    assert not hasattr(window, "validation_warning_label")
    assert "/rmcsetup/representative_clusters" in (
        window.output_summary_box.toPlainText()
    )
    assert window.solution_group.isEnabled() is True
    assert "No saved solution-properties calculation yet" in (
        window.solution_output_box.toPlainText()
    )
    assert "No representative selection has been saved yet" in (
        window.representative_summary_box.toPlainText()
    )


def test_rmcsetup_software_details_section_is_collapsible_and_link_ready(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window.software_details_button.isCheckable()
    assert window.software_details_button.isChecked() is False
    assert window.software_details_panel.isHidden() is True
    assert window.software_details_label.openExternalLinks() is True
    details_text = window.software_details_label.text()
    assert "reverse Monte-Carlo workflows for pair distribution function" in (
        details_text
    )
    assert "https://www.fullrmc.com/" in details_text
    assert "10.1002/jcc.24304" in details_text
    assert "https://github.com/m3g/packmol" in details_text
    assert "10.1002/jcc.21224" in details_text

    window.software_details_button.setChecked(True)

    assert window.software_details_panel.isHidden() is False


def test_rmcsetup_main_window_uses_two_scrollable_panes_with_splitter(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window._main_splitter.count() == 2
    assert window._main_splitter.widget(0) is window._left_column
    assert window._main_splitter.widget(1) is window._right_scroll_area
    assert window._left_column.isAncestorOf(window.status_group)
    assert window._left_column.isAncestorOf(window._left_scroll_area)
    assert window._left_scroll_area.widget() is window._left_panel
    assert window._right_scroll_area.widget() is window._right_panel
    assert window._left_scroll_area.widgetResizable() is True
    assert window._right_scroll_area.widgetResizable() is True
    assert window._left_panel.isAncestorOf(window.status_group) is False
    assert window._left_panel.isAncestorOf(window.project_group)
    assert window._left_panel.isAncestorOf(window.solution_group)
    assert window._right_panel.isAncestorOf(window.dream_preview_group)
    assert window._right_panel.isAncestorOf(window.representative_group)
    assert window._right_panel.isAncestorOf(window.packmol_group)
    assert window._right_panel.isAncestorOf(window.run_log_group)


def test_rmcsetup_main_window_renders_selected_dream_preview_and_tooltips(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window.dream_preview_group.title() == "Selected DREAM Model Preview"
    assert window.set_favorite_button.text() == "Save Current Selection"
    assert window.use_project_favorite_button.text() == (
        "Load Saved DREAM Model"
    )
    assert window.dream_run_combo.toolTip()
    assert window.posterior_filter_combo.toolTip()
    assert window.solution_mode_combo.toolTip()
    assert "Project Readiness Tasks" in window.readiness_help_button.toolTip()
    assert "Information needed:" in (
        window._readiness_checkboxes["packmol_plan"].toolTip()
    )
    assert window.show_solvent_trace_checkbox.isChecked() is False
    assert "Selected run: dream_run_001" in (
        window.dream_preview_status_label.text()
    )

    model_axis = window._dream_model_preview_figure.axes[0]
    assert "DREAM refinement:" in model_axis.get_title()
    assert len(model_axis.lines) >= 1
    model_legend = model_axis.get_legend()
    assert model_legend is not None
    assert "Solvent contribution" not in {
        text.get_text() for text in model_legend.get_texts()
    }
    initial_collection_count = len(model_axis.collections)

    window.show_experimental_trace_checkbox.setChecked(False)

    refreshed_model_axis = window._dream_model_preview_figure.axes[0]
    assert len(refreshed_model_axis.collections) < initial_collection_count

    window.show_experimental_trace_checkbox.setChecked(True)
    window.show_solvent_trace_checkbox.setChecked(True)
    assert window.show_solvent_trace_checkbox.isChecked() is True

    violin_axis = window._dream_violin_preview_figure.axes[0]
    assert violin_axis.get_title() == "Posterior weight distributions"
    assert len(violin_axis.collections) >= 1

    window.dream_run_combo.setCurrentIndex(0)

    assert "Selected run: dream_run_002" in (
        window.dream_preview_status_label.text()
    )


def test_rmcsetup_solution_properties_mode_switch_changes_active_page(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert (
        "Mass mode uses the directly entered solute and solvent masses."
        in (window.solution_mode_hint_label.text())
    )

    window.solution_mode_combo.setCurrentIndex(1)
    assert window.solution_mode_combo.currentData() == "mass_percent"
    assert window.solution_mode_stack.currentIndex() == 1
    assert "Mass-percent mode uses the entered solute mass percent" in (
        window.solution_mode_hint_label.text()
    )

    window.solution_mode_combo.setCurrentIndex(2)
    assert window.solution_mode_combo.currentData() == "molarity_per_liter"
    assert window.solution_mode_stack.currentIndex() == 2
    assert "Molarity mode assumes 1 L of solution." in (
        window.solution_mode_hint_label.text()
    )
    assert "solution density is still required" in (
        window.solution_mode_hint_label.text().lower()
    )


def test_rmcsetup_solution_presets_load_bundled_values(
    tmp_path,
    monkeypatch,
):
    qapp()
    monkeypatch.setenv(
        "SAXSHELL_SOLUTION_PROPERTY_PRESETS_PATH",
        str(tmp_path / "solution_property_presets.json"),
    )
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    preset_names = [
        window.solution_preset_combo.itemText(index)
        for index in range(window.solution_preset_combo.count())
    ]
    assert "PbI2 - DMF - 0.49 M (Built-in)" in preset_names
    assert "MAPbI3 - DMSO - 0.05 M (Built-in)" in preset_names

    preset_index = window.solution_preset_combo.findData(
        "PbI2 - DMSO - 0.405 M"
    )
    assert preset_index >= 0
    window.solution_preset_combo.setCurrentIndex(preset_index)
    window._load_selected_solution_preset()

    assert window.solution_preset_combo.currentData() == (
        "PbI2 - DMSO - 0.405 M"
    )
    assert window._solution_presets[
        "PbI2 - DMSO - 0.405 M"
    ].solvent_density_g_per_ml == pytest.approx(1.10)
    assert window._solution_presets[
        "PbI2 - DMF - 0.49 M"
    ].solvent_density_g_per_ml == pytest.approx(0.944)
    assert window.solution_mode_combo.currentData() == "molarity_per_liter"
    assert window.solution_density_spin.value() == pytest.approx(1.273)
    assert window.solute_stoich_edit.text() == "PbI2"
    assert window.solvent_stoich_edit.text() == "C2H6OS"
    assert window.molar_mass_solute_spin.value() == pytest.approx(461.01)
    assert window.molar_mass_solvent_spin.value() == pytest.approx(78.13)
    assert window.molarity_spin.value() == pytest.approx(0.405)
    assert window.molarity_element_edit.text() == "Pb"


def test_rmcsetup_solution_presets_can_save_and_reload_custom_preset(
    tmp_path,
    monkeypatch,
):
    qapp()
    preset_path = tmp_path / "solution_property_presets.json"
    monkeypatch.setenv(
        "SAXSHELL_SOLUTION_PROPERTY_PRESETS_PATH",
        str(preset_path),
    )
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_mode_combo.setCurrentIndex(2)
    window.solution_density_spin.setValue(1.111)
    window.solute_stoich_edit.setText("PbI2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.01)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.molarity_spin.setValue(0.123)
    window.molarity_element_edit.setText("Pb")

    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.QInputDialog.getText",
        lambda *args, **kwargs: ("My PbI2 DMF", True),
    )

    window._save_current_solution_as_preset()

    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    assert payload["presets"]["My PbI2 DMF"]["settings"]["molarity"] == (
        pytest.approx(0.123)
    )
    assert "Saved solution preset: My PbI2 DMF" in (
        window.run_log_box.toPlainText()
    )

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    preset_index = reloaded.solution_preset_combo.findData("My PbI2 DMF")
    assert preset_index >= 0
    reloaded.solution_preset_combo.setCurrentIndex(preset_index)
    reloaded._load_selected_solution_preset()

    assert reloaded.solution_preset_combo.currentData() == "My PbI2 DMF"
    assert reloaded.solution_density_spin.value() == pytest.approx(1.111)
    assert reloaded.molarity_spin.value() == pytest.approx(0.123)
    assert reloaded.molarity_element_edit.text() == "Pb"


def test_rmcsetup_solution_properties_calculation_persists_and_reloads(
    tmp_path,
):
    qapp()
    project_dir, paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_mode_combo.setCurrentIndex(2)
    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.molarity_spin.setValue(0.5)
    window.molarity_element_edit.setText("Pb")
    window._calculate_solution_properties()

    metadata = json.loads(
        paths.project_dir.joinpath(
            "rmcsetup", "solution_properties.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["settings"]["mode"] == "molarity_per_liter"
    assert metadata["settings"]["molarity_element"] == "Pb"
    assert metadata["result"]["number_density_a3"] > 0
    assert "Number density:" in window.solution_output_box.toPlainText()
    assert "Calculated and saved solution properties metadata." in (
        window.run_log_box.toPlainText()
    )

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.solution_mode_combo.currentData() == "molarity_per_liter"
    assert reloaded.solution_mode_stack.currentIndex() == 2
    assert reloaded.molarity_element_edit.text() == "Pb"
    assert reloaded.molarity_spin.value() == pytest.approx(0.5)
    assert "Saved calculation:" in reloaded.solution_output_box.toPlainText()


def test_rmcsetup_ui_can_compute_first_file_representatives(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window._compute_representative_clusters()
    _wait_for_representative_worker(window)

    assert "Representative files resolved: 2" in (
        window.representative_summary_box.toPlainText()
    )
    assert window.representative_status_label.text() == (
        "Representative selection: complete"
    )
    assert "Computed representative clusters in first_file mode." in (
        window.run_log_box.toPlainText()
    )


def test_rmcsetup_distribution_mode_loads_bondanalysis_presets(
    tmp_path,
    monkeypatch,
):
    qapp()
    monkeypatch.setenv(
        "SAXSHELL_BONDANALYSIS_PRESETS_PATH",
        str(tmp_path / "bondanalysis_presets.json"),
    )
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.representative_mode_combo.setCurrentIndex(1)

    preset_names = [
        window.representative_preset_combo.itemText(index)
        for index in range(window.representative_preset_combo.count())
    ]
    assert "DMSO (Built-in)" in preset_names
    assert "DMF (Built-in)" in preset_names

    preset_index = window.representative_preset_combo.findData("DMF")
    assert preset_index >= 0
    window.representative_preset_combo.setCurrentIndex(preset_index)
    window._load_selected_representative_preset()

    assert window.representative_bond_pair_table.rowCount() == 7
    assert window.representative_bond_pair_table.item(0, 0).text() == "Pb"
    assert window.representative_bond_pair_table.item(0, 1).text() == "I"
    assert window.representative_bond_pair_table.item(0, 2).text() == "4"
    assert window.representative_angle_triplet_table.rowCount() == 5
    assert window.representative_angle_triplet_table.item(3, 0).text() == "O"
    assert window.representative_angle_triplet_table.item(3, 1).text() == "Pb"
    assert window.representative_angle_triplet_table.item(3, 2).text() == "N"


def test_rmcsetup_ui_exposes_distribution_mode_and_computes_it(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.representative_mode_combo.setCurrentIndex(1)
    assert window.representative_preset_group.isHidden() is False
    assert window.representative_bond_pairs_row.isHidden() is False
    assert window.representative_angle_triplets_row.isHidden() is False
    window.representative_bond_pair_table.item(0, 0).setText("Pb")
    window.representative_bond_pair_table.item(0, 1).setText("I")
    window.representative_bond_pair_table.item(0, 2).setText("3.5")
    window.representative_angle_triplet_table.item(0, 0).setText("Pb")
    window.representative_angle_triplet_table.item(0, 1).setText("I")
    window.representative_angle_triplet_table.item(0, 2).setText("I")
    window.representative_angle_triplet_table.item(0, 3).setText("3.5")
    window.representative_angle_triplet_table.item(0, 4).setText("3.5")

    window._compute_representative_clusters()
    _wait_for_representative_worker(window)

    assert "Selection mode: bond_angle_distribution" in (
        window.representative_summary_box.toPlainText()
    )
    assert (
        "Computed representative clusters in bond_angle_distribution mode."
        in (window.run_log_box.toPlainText())
    )


def test_rmcsetup_ui_advanced_representative_settings_apply_cutoff(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.representative_mode_combo.setCurrentIndex(1)
    window.representative_advanced_toggle.setChecked(True)
    window.representative_algorithm_combo.setCurrentIndex(1)
    window.representative_count_cutoff_spin.setValue(2)
    window.representative_bond_pair_table.item(0, 0).setText("Pb")
    window.representative_bond_pair_table.item(0, 1).setText("I")
    window.representative_bond_pair_table.item(0, 2).setText("3.5")
    window.representative_angle_triplet_table.item(0, 0).setText("Pb")
    window.representative_angle_triplet_table.item(0, 1).setText("I")
    window.representative_angle_triplet_table.item(0, 2).setText("I")
    window.representative_angle_triplet_table.item(0, 3).setText("3.5")
    window.representative_angle_triplet_table.item(0, 4).setText("3.5")

    window._compute_representative_clusters()
    _wait_for_representative_worker(window)

    text = window.representative_summary_box.toPlainText()
    assert "Algorithm: target_distribution_moment_distance" in text
    assert "Cluster count cutoff: 2" in text
    assert "Skipped by count cutoff: 1" in text
    assert "Representative files resolved: 1" in text


def test_rmcsetup_ui_can_open_representative_preview_window(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    _write_sample_bondanalysis_results(project_dir)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.representative_mode_combo.setCurrentIndex(1)
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window._preview_representative_clusters()

    preview_window = window._representative_preview_window
    assert preview_window is not None
    assert preview_window.tab_widget.count() == 2
    assert preview_window.tab_widget.tabText(0) == "PbI2O/motif_1"
    first_tab = preview_window.tab_widget.widget(0)
    assert first_tab is not None
    assert len(first_tab.figure.axes) == 2
    assert any(
        line.get_linestyle() == "--" for line in first_tab.figure.axes[0].lines
    )
    assert "Opened representative preview window." in (
        window.run_log_box.toPlainText()
    )


def test_rmcsetup_solvent_handling_ui_builds_and_reloads(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.coordinated_solvent_mode_combo.setCurrentIndex(1)
    window.solvent_reference_source_combo.setCurrentIndex(0)
    window.solvent_minimum_separation_spin.setValue(1.4)
    window._build_representative_solvent_outputs()

    assert "reference molecule: dmf" in (
        window.solvent_summary_box.toPlainText().lower()
    )
    assert "Minimum solvent atom separation: 1.4 A" in (
        window.solvent_summary_box.toPlainText()
    )
    assert window.generated_pdb_table.rowCount() == 4
    generated_rows = {
        (
            window.generated_pdb_table.item(row, 0).text(),
            window.generated_pdb_table.item(row, 1).text(),
        ): row
        for row in range(window.generated_pdb_table.rowCount())
    }
    assert set(generated_rows) == {
        ("PbI2", "No solvent"),
        ("PbI2", "With solvent"),
        ("PbI2O/motif_1", "No solvent"),
        ("PbI2O/motif_1", "With solvent"),
    }
    window.generated_pdb_table.selectRow(
        generated_rows[("PbI2", "With solvent")]
    )
    details_text = window.generated_pdb_details_box.toPlainText()
    assert "Atom count: 15" in details_text
    assert "Element counts: C:3, H:7, I:2, N:1, O:1, Pb:1" in details_text
    assert "Solvent molecules matching reference residue DMF: 1" in (
        details_text
    )
    assert "PBI 1: 3 atoms (I:2, Pb:1)" in details_text
    assert "DMF 2: 12 atoms (C:3, H:7, N:1, O:1)" in details_text
    assert "1: Pb1 (Pb) -> PBI 1" in details_text
    assert "4: O1 (O) -> DMF 2" in details_text

    window._open_selected_generated_pdb_preview()

    generated_preview_window = window._generated_pdb_preview_window
    assert generated_preview_window is not None
    assert (
        generated_preview_window.show_solvent_atoms_checkbox.isChecked()
        is False
    )
    assert (
        generated_preview_window.show_solvent_network_checkbox.isChecked()
        is False
    )
    assert len(generated_preview_window.figure.axes) == 1
    assert generated_preview_window.figure.axes[0].name == "3d"
    initial_line_count = len(generated_preview_window.figure.axes[0].lines)
    initial_collection_count = len(
        generated_preview_window.figure.axes[0].collections
    )
    assert initial_line_count > 0
    generated_preview_window.show_solvent_network_checkbox.setChecked(True)
    generated_preview_window.show_solvent_atoms_checkbox.setChecked(True)
    assert (
        len(generated_preview_window.figure.axes[0].lines) > initial_line_count
    )
    assert (
        len(generated_preview_window.figure.axes[0].collections)
        > initial_collection_count
    )
    assert "Built representative solvent-aware PDB outputs." in (
        window.run_log_box.toPlainText()
    )
    assert "Opened generated PDB preview:" in window.run_log_box.toPlainText()

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.coordinated_solvent_mode_combo.currentData() == (
        "partial_coordinated_solvent"
    )
    assert reloaded.solvent_reference_source_combo.currentData() == "preset"
    assert reloaded.solvent_preset_combo.currentData() == "dmf"
    assert reloaded.solvent_minimum_separation_spin.value() == pytest.approx(
        1.4
    )
    assert reloaded.generated_pdb_table.rowCount() == 4
    assert "Representative entries exported: 2" in (
        reloaded.solvent_summary_box.toPlainText()
    )


def test_rmcsetup_ui_can_compute_packmol_plan_and_reload(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.mass_solute_spin.setValue(4.61)
    window.mass_solvent_spin.setValue(95.39)
    window._calculate_solution_properties()
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.packmol_box_side_spin.setValue(80.0)

    window._compute_packmol_plan()

    assert "Planned clusters:" in (
        window.packmol_plan_summary_box.toPlainText()
    )
    axes = window._packmol_plan_figure.axes
    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == [
        "Original cluster distribution",
        "DREAM fit model distribution used",
        "Packmol planned distribution",
    ]
    assert all(len(axis.patches) == 4 for axis in axes)
    assert [tick.get_text() for tick in axes[-1].get_xticklabels()] == [
        format_stoich_for_axis("PbI2"),
        format_stoich_for_axis("PbI2O"),
    ]

    plan_entries = {
        (entry.structure, entry.motif): entry
        for entry in window._project_source_state.packmol_planning.entries
    }
    assert [patch.get_height() for patch in axes[0].patches] == pytest.approx(
        [66.6666666667, 0.0, 0.0, 33.3333333333]
    )
    dream_total = sum(entry.selected_weight for entry in plan_entries.values())
    assert [patch.get_height() for patch in axes[1].patches] == pytest.approx(
        [
            100.0
            * plan_entries[("PbI2", "no_motif")].selected_weight
            / dream_total,
            0.0,
            0.0,
            100.0
            * plan_entries[("PbI2O", "motif_1")].selected_weight
            / dream_total,
        ]
    )
    assert [patch.get_height() for patch in axes[2].patches] == pytest.approx(
        [
            100.0 * plan_entries[("PbI2", "no_motif")].planned_count_weight,
            0.0,
            0.0,
            100.0 * plan_entries[("PbI2O", "motif_1")].planned_count_weight,
        ]
    )
    assert "Computed Packmol planning counts." in (
        window.run_log_box.toPlainText()
    )
    assert "packmol_plan.json" in window.output_summary_box.toPlainText()

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.packmol_planning_mode_combo.currentData() == (
        "per_element"
    )
    assert reloaded.packmol_box_side_spin.value() == pytest.approx(80.0)
    assert "Planned clusters:" in (
        reloaded.packmol_plan_summary_box.toPlainText()
    )


def test_rmcsetup_ui_packmol_preview_includes_single_atom_model_sources(
    tmp_path,
):
    qapp()
    project_dir, _paths, _single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.mass_solute_spin.setValue(4.61)
    window.mass_solvent_spin.setValue(95.39)
    window._calculate_solution_properties()
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.coordinated_solvent_mode_combo.setCurrentIndex(1)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)

    window._compute_packmol_plan()

    axes = window._packmol_plan_figure.axes
    assert len(axes) == 3
    assert all(len(axis.patches) == 6 for axis in axes)
    assert format_stoich_for_axis("Zn1") in [
        tick.get_text() for tick in axes[-1].get_xticklabels()
    ]
    assert any(
        entry.structure == "Zn1"
        for entry in window._project_source_state.packmol_planning.entries
    )


def test_rmcsetup_ui_can_build_packmol_setup_and_reload(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.mass_solute_spin.setValue(4.61)
    window.mass_solvent_spin.setValue(95.39)
    window._calculate_solution_properties()
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.coordinated_solvent_mode_combo.setCurrentIndex(1)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()

    window._build_packmol_setup()

    assert "Representative PDBs copied:" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert "Built Packmol setup inputs and audit report." in (
        window.run_log_box.toPlainText()
    )
    assert "packmol_combined.inp" in window.output_summary_box.toPlainText()

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert "Representative PDBs copied:" in (
        reloaded.packmol_build_summary_box.toPlainText()
    )
    assert "packmol_audit.md" in reloaded.output_summary_box.toPlainText()


def test_rmcsetup_ui_can_generate_constraints_and_reload(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.mass_solute_spin.setValue(4.61)
    window.mass_solvent_spin.setValue(95.39)
    window._calculate_solution_properties()
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.coordinated_solvent_mode_combo.setCurrentIndex(1)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window._build_packmol_setup()
    window.constraint_length_tolerance_spin.setValue(0.08)
    window.constraint_angle_tolerance_spin.setValue(7.0)

    window._generate_constraints()

    assert "Per-structure files:" in (
        window.constraints_summary_box.toPlainText()
    )
    assert (
        "Generated per-structure constraints and merged fullrmc constraints."
        in (window.run_log_box.toPlainText())
    )
    assert "merged_fullrmc_constraints.py" in (
        window.output_summary_box.toPlainText()
    )

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.constraint_length_tolerance_spin.value() == pytest.approx(
        0.08
    )
    assert reloaded.constraint_angle_tolerance_spin.value() == pytest.approx(
        7.0
    )
    assert "Per-structure files:" in (
        reloaded.constraints_summary_box.toPlainText()
    )


def test_rmcsetup_cluster_validation_runs_in_backend_for_cluster_drift(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    drift_file = (
        tmp_path
        / "clusters_splitxyz0001"
        / "PbI2O"
        / "motif_1"
        / "frame_0004.xyz"
    )
    drift_file.write_text("new file\n", encoding="utf-8")

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window._project_source_state is not None
    assert window._project_source_state.cluster_validation.is_valid is False
    assert window.readiness_progress_bar.value() == 2


def test_rmcsetup_representative_selection_runs_in_background(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    original_select = select_first_file_representatives

    def slow_select(*args, **kwargs):
        time.sleep(0.05)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.select_first_file_representatives",
        slow_select,
    )

    window._compute_representative_clusters()

    assert window._representative_thread is not None
    assert window.compute_representatives_button.isEnabled() is False
    assert "running in the background" in (
        window.representative_summary_box.toPlainText().lower()
    )

    window.solution_density_spin.setValue(1.23)
    assert window.solution_density_spin.value() == pytest.approx(1.23)

    _wait_for_representative_worker(window)

    assert window.compute_representatives_button.isEnabled() is True
    assert "Computed representative clusters in first_file mode." in (
        window.run_log_box.toPlainText()
    )


def test_rmcsetup_representative_selection_failure_keeps_window_alive(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.select_first_file_representatives",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )

    window._compute_representative_clusters()
    _wait_for_representative_worker(window)

    assert window.compute_representatives_button.isEnabled() is True
    assert window.representative_status_label.text() == (
        "Representative selection: failed"
    )
    assert "Unable to compute representative clusters: boom" in (
        window.representative_summary_box.toPlainText()
    )
    window.solution_density_spin.setValue(1.11)
    assert window.solution_density_spin.value() == pytest.approx(1.11)


def test_rmcsetup_ui_end_to_end_pipeline_updates_readiness_and_outputs(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.dream_run_combo.setCurrentIndex(0)
    window.bestfit_method_combo.setCurrentIndex(2)
    window.posterior_filter_combo.setCurrentIndex(1)
    window.posterior_top_percent_spin.setValue(7.5)
    window.solution_density_spin.setValue(1.05)
    window.solute_stoich_edit.setText("Pb1I2")
    window.solvent_stoich_edit.setText("C3H7NO")
    window.molar_mass_solute_spin.setValue(461.0)
    window.molar_mass_solvent_spin.setValue(73.09)
    window.mass_solute_spin.setValue(4.61)
    window.mass_solvent_spin.setValue(95.39)
    window._calculate_solution_properties()
    window.representative_mode_combo.setCurrentIndex(1)
    window.representative_advanced_toggle.setChecked(True)
    window.representative_count_cutoff_spin.setValue(1)
    window.representative_bond_pair_table.item(0, 0).setText("Pb")
    window.representative_bond_pair_table.item(0, 1).setText("I")
    window.representative_bond_pair_table.item(0, 2).setText("3.5")
    window.representative_angle_triplet_table.item(0, 0).setText("Pb")
    window.representative_angle_triplet_table.item(0, 1).setText("I")
    window.representative_angle_triplet_table.item(0, 2).setText("I")
    window.representative_angle_triplet_table.item(0, 3).setText("3.5")
    window.representative_angle_triplet_table.item(0, 4).setText("3.5")
    window._compute_representative_clusters()
    _wait_for_representative_worker(window)
    window.coordinated_solvent_mode_combo.setCurrentIndex(1)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window._build_packmol_setup()
    window.constraint_length_tolerance_spin.setValue(0.08)
    window.constraint_angle_tolerance_spin.setValue(7.0)
    window._generate_constraints()

    assert window.readiness_progress_bar.value() == 7
    assert window.readiness_label.text() == "Project readiness: 7 / 7 complete"
    assert list(window._readiness_checkboxes) == [
        "project_source",
        "dream_selection",
        "solution_properties",
        "representative_selection",
        "solvent_outputs",
        "packmol_plan",
        "packmol_setup",
    ]
    assert all(
        checkbox.isChecked()
        for checkbox in window._readiness_checkboxes.values()
    )
    assert window.task_progress_bar.value() == 100
    assert "Constraint generation complete." in window.task_status_label.text()
    assert "Built Packmol setup inputs and audit report." in (
        window.run_log_box.toPlainText()
    )
    assert (
        "Generated per-structure constraints and merged fullrmc constraints."
        in (window.run_log_box.toPlainText())
    )

    state = load_rmc_project_source(project_dir)
    assert state.packmol_setup is not None
    assert state.constraint_generation is not None
    assert Path(state.packmol_setup.packmol_input_path).is_file()
    assert Path(state.constraint_generation.merged_constraints_path).is_file()


def test_rmcsetup_can_save_new_project_favorite_and_extend_history(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.dream_run_combo.setCurrentIndex(0)
    window.bestfit_method_combo.setCurrentIndex(2)
    window.posterior_filter_combo.setCurrentIndex(1)
    window.posterior_top_percent_spin.setValue(12.5)
    window.credible_interval_low_spin.setValue(5.0)
    window.credible_interval_high_spin.setValue(95.0)
    window._save_current_selection_as_favorite()

    reloaded = SAXSProjectManager().load_project(project_dir)
    assert reloaded.dream_favorite_selection is not None
    assert reloaded.dream_favorite_selection.run_name == "dream_run_002"
    assert reloaded.dream_favorite_selection.bestfit_method == "median"
    assert reloaded.dream_favorite_selection.posterior_filter_mode == (
        "top_percent_logp"
    )
    assert reloaded.dream_favorite_selection.posterior_top_percent == 12.5
    assert len(reloaded.dream_favorite_history) == 2


def test_rmcsetup_can_load_saved_history_entry(tmp_path):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    window.favorite_history_combo.setCurrentIndex(0)
    window._load_history_entry()

    assert window.dream_run_combo.currentText() == "dream_run_002"
    assert window.bestfit_method_combo.currentData() == "median"
    assert window.posterior_filter_combo.currentData() == "top_percent_logp"
    assert window.posterior_top_percent_spin.value() == 7.5


def test_fullrmc_cli_defaults_to_ui(monkeypatch):
    captured: dict[str, object] = {}

    def fake_launcher(project_dir=None):
        captured["project_dir"] = project_dir
        return 0

    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.launch_rmcsetup_ui",
        fake_launcher,
    )

    exit_code = fullrmc_main([])

    assert exit_code == 0
    assert captured["project_dir"] is None


def test_fullrmc_cli_ui_subcommand_forwards_project_dir(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    captured: dict[str, object] = {}

    def fake_launcher(project_dir=None):
        captured["project_dir"] = project_dir
        return 0

    monkeypatch.setattr(
        "saxshell.fullrmc.ui.main_window.launch_rmcsetup_ui",
        fake_launcher,
    )

    exit_code = fullrmc_main(["ui", str(project_dir)])

    assert exit_code == 0
    assert captured["project_dir"] == project_dir


def test_saxshell_cli_forwards_to_fullrmc_subcommand(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(fullrmc_cli, "main", fake_main)

    exit_code = saxshell_main(["fullrmc", "ui", str(project_dir)])

    assert exit_code == 0
    assert captured["argv"] == ["ui", str(project_dir)]
