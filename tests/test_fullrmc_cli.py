from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QMessageBox,
    QWidget,
)

import saxshell.fullrmc.cli as fullrmc_cli
import saxshell.fullrmc.packmol_setup as packmol_setup_module
import saxshell.fullrmc.solvent_shell_builder as solvent_shell_builder_module
import saxshell.fullrmc.ui.main_window as fullrmc_ui_module
from saxshell.fullrmc import (
    DEFAULT_REFERENCE_MATCH_TOLERANCE_A,
    ConstraintGenerationSettings,
    PackmolDockerContainerRecord,
    PackmolDockerDirectoryEntry,
    PackmolDockerLink,
    PackmolDockerSyncResult,
    PackmolDockerValidationResult,
    PackmolPlanningSettings,
    PackmolSetupSettings,
    PackmolSupplementalComponentSettings,
    RepresentativeSelectionSettings,
    SolutionPropertiesSettings,
    SolventHandlingSettings,
    analyze_solvent_shell,
    build_constraint_generation,
    build_distribution_selection,
    build_packmol_plan,
    build_packmol_setup,
    build_representative_preview_clusters,
    build_representative_solvent_outputs,
    build_solvent_shell_output,
    calculate_solution_properties,
    container_project_root_is_valid,
    load_constraint_generation_metadata,
    load_packmol_planning_metadata,
    load_packmol_setup_metadata,
    load_representative_selection_metadata,
    load_rmc_project_source,
    load_solvent_handling_metadata,
    parse_angle_triplet_text,
    parse_bond_pair_text,
    save_packmol_docker_link_metadata,
    save_representative_selection_metadata,
    save_solution_properties_metadata,
    select_distribution_representatives,
    select_first_file_representatives,
)
from saxshell.fullrmc.cli import main as fullrmc_main
from saxshell.fullrmc.solvent_handling import (
    analyze_representative_solvent_distribution,
)
from saxshell.fullrmc.ui.main_window import RMCSetupMainWindow
from saxshell.fullrmc.ui.solvent_shell_builder_window import (
    SolventShellBuilderMainWindow,
)
from saxshell.saxs.dream import DreamRunSettings
from saxshell.saxs.project_manager import (
    DreamBestFitSelection,
    SAXSProjectManager,
    build_project_paths,
    project_artifact_paths,
)
from saxshell.saxs.stoichiometry import format_stoich_for_axis
from saxshell.saxshell import main as saxshell_main
from saxshell.structure import PDBAtom, PDBStructure
from saxshell.xyz2pdb import create_reference_molecule


def _integrated_solvent_handling_settings(
    *,
    reference_source: str = "preset",
    preset_name: str = "dmf",
    custom_reference_path: str | None = None,
    director_atom_name: str | None = None,
    pb_target_coordination: float = 1.0,
    pb_cutoff_a: float = 2.6,
    i_cutoff_a: float = 3.0,
) -> SolventHandlingSettings:
    return SolventHandlingSettings.from_dict(
        {
            "coordinated_solvent_mode": "automatic_detection",
            "reference_source": reference_source,
            "preset_name": preset_name,
            "custom_reference_path": custom_reference_path,
            "director_atom_name": director_atom_name,
            "minimum_solvent_atom_separation_a": 1.2,
            "solute_atom_settings": {
                "Pb": {
                    "coordination_center": True,
                    "target_coordination_number": pb_target_coordination,
                    "director_distance_cutoff_a": pb_cutoff_a,
                },
                "I": {
                    "coordination_center": False,
                    "target_coordination_number": 0.0,
                    "director_distance_cutoff_a": i_cutoff_a,
                },
            },
        }
    )


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


def _build_sample_distribution_scoped_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "rmcsetup_distribution_source"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)
    clusters_dir = _build_sample_clusters_dir(tmp_path)
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
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    manager.ensure_project_dirs(paths)
    manager.ensure_artifact_dirs(artifact_paths)

    run_a = _write_sample_dream_run(
        artifact_paths.dream_runtime_dir / "dream_run_001",
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
        artifact_paths.dream_runtime_dir / "dream_run_002",
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
    manager.save_project(settings)
    manager._write_distribution_metadata(
        settings,
        artifact_paths=artifact_paths,
    )
    return project_dir, paths, artifact_paths


def _build_sample_predicted_distribution_scoped_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "rmcsetup_predicted_distribution_source"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)
    clusters_dir = _build_sample_clusters_dir(tmp_path)
    predicted_dir = tmp_path / "predicted_structures"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_source = predicted_dir / "zn1_rank01.xyz"
    predicted_source.write_text(
        "\n".join(
            [
                "1",
                "predicted Zn1 structure",
                "Zn 0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings.use_predicted_structure_weights = True
    settings.clusters_dir = str(clusters_dir)
    # Keep the saved settings stale on purpose so the loader has to prefer
    # the active predicted-structure artifact inventory on window open.
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
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    manager.ensure_project_dirs(paths)
    manager.ensure_artifact_dirs(artifact_paths)

    run = _write_sample_dream_run(
        artifact_paths.dream_runtime_dir / "dream_run_predicted",
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
                "value": 0.45,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.45,
                    "s": 0.3,
                },
            },
            {
                "structure": "PbI2O",
                "motif": "motif_1",
                "param_type": "Both",
                "param": "w1",
                "value": 0.25,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.25,
                    "s": 0.3,
                },
            },
            {
                "structure": "Zn1",
                "motif": "predicted_rank01",
                "param_type": "Both",
                "param": "w2",
                "value": 0.30,
                "vary": True,
                "distribution": "lognorm",
                "dist_params": {
                    "loc": 0.0,
                    "scale": 0.30,
                    "s": 0.3,
                },
            },
        ],
        sampled_params=np.asarray(
            [[[0.45, 0.25, 0.30], [0.40, 0.30, 0.30]]],
            dtype=float,
        ),
    )

    settings.dream_favorite_selection = DreamBestFitSelection(
        run_name=run.name,
        run_relative_path=str(run.relative_to(project_dir)),
        bestfit_method="chain_mean",
        posterior_filter_mode="top_n_logp",
        posterior_top_percent=10.0,
        posterior_top_n=42,
        credible_interval_low=25.0,
        credible_interval_high=75.0,
        label="2026-03-23T11:00:00 • dream_run_predicted • chain_mean",
        template_name="template_pd_likelihood_monosq_decoupled",
        model_name="template_pd_likelihood_monosq_decoupled",
        selection_source="rmcsetup",
        selected_at="2026-03-23T11:00:00",
    )
    manager.save_project(settings)

    artifact_paths.prior_weights_file.write_text(
        json.dumps(
            {
                "origin": "clusters_predicted_structures",
                "total_files": 4,
                "includes_predicted_structures": True,
                "structures": {
                    "PbI2": {
                        "no_motif": {
                            "count": 2,
                            "weight": 0.45,
                            "representative": "frame_0002.xyz",
                            "profile_file": "PbI2_no_motif.txt",
                            "source_kind": "cluster_dir",
                            "source_dir": str(clusters_dir / "PbI2"),
                            "source_file": str(
                                (
                                    clusters_dir / "PbI2" / "frame_0002.xyz"
                                ).resolve()
                            ),
                            "source_file_name": "frame_0002.xyz",
                        }
                    },
                    "PbI2O": {
                        "motif_1": {
                            "count": 1,
                            "weight": 0.25,
                            "representative": "frame_0003.xyz",
                            "profile_file": "PbI2O_motif_1.txt",
                            "source_kind": "cluster_dir",
                            "source_dir": str(
                                clusters_dir / "PbI2O" / "motif_1"
                            ),
                            "source_file": str(
                                (
                                    clusters_dir
                                    / "PbI2O"
                                    / "motif_1"
                                    / "frame_0003.xyz"
                                ).resolve()
                            ),
                            "source_file_name": "frame_0003.xyz",
                        }
                    },
                    "Zn1": {
                        "predicted_rank01": {
                            "count": 1,
                            "weight": 0.30,
                            "representative": predicted_source.name,
                            "profile_file": "Zn1_predicted_rank01.txt",
                            "source_kind": "predicted_structure",
                            "source_dir": str(predicted_source.parent),
                            "source_file": str(predicted_source.resolve()),
                            "source_file_name": predicted_source.name,
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manager._write_distribution_metadata(
        settings,
        artifact_paths=artifact_paths,
    )
    return project_dir, predicted_source.resolve()


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


def _build_test_solvent_reference_library(
    tmp_path: Path,
) -> tuple[Path, Path]:
    reference_source = _write_custom_solvent_pdb(tmp_path)
    reference_library_dir = tmp_path / "reference_library"
    reference_library_dir.mkdir(parents=True, exist_ok=True)
    result = create_reference_molecule(
        reference_source,
        reference_name="water_test",
        residue_name="HOH",
        library_dir=reference_library_dir,
    )
    return reference_library_dir, result.path


def _write_test_solvent_shell_pdb(
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
    structure = PDBStructure(atoms=atoms, source_name="solvent_shell")
    output_path = tmp_path / "solvent_shell_input.pdb"
    structure.write_pdb_file(output_path)
    return output_path


def _write_test_incomplete_solvent_shell_pdb(
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
    complete_shift = np.array([3.0, 0.0, 0.0], dtype=float)
    for reference_atom in reference_structure.atoms:
        atoms.append(
            PDBAtom(
                atom_id=atom_id,
                atom_name=reference_atom.atom_name,
                residue_name="HOH",
                residue_number=2,
                coordinates=reference_atom.coordinates.copy() + complete_shift,
                element=reference_atom.element,
            )
        )
        atom_id += 1
    partial_shift = np.array([6.0, 0.0, 0.0], dtype=float)
    for reference_atom in reference_structure.atoms[:2]:
        atoms.append(
            PDBAtom(
                atom_id=atom_id,
                atom_name=reference_atom.atom_name,
                residue_name="HOH",
                residue_number=3,
                coordinates=reference_atom.coordinates.copy() + partial_shift,
                element=reference_atom.element,
            )
        )
        atom_id += 1
    structure = PDBStructure(
        atoms=atoms,
        source_name="solvent_shell_incomplete",
    )
    output_path = tmp_path / "solvent_shell_incomplete_input.pdb"
    structure.write_pdb_file(output_path)
    return output_path


def _write_test_no_solvent_shell_pdb(tmp_path: Path) -> Path:
    structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="PB1",
                residue_name="PBI",
                residue_number=1,
                coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
                element="Pb",
            )
        ],
        source_name="solvent_shell_none",
    )
    output_path = tmp_path / "solvent_shell_none_input.pdb"
    structure.write_pdb_file(output_path)
    return output_path


def _write_test_no_solvent_mixed_shell_pdb(tmp_path: Path) -> Path:
    structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="PB1",
                residue_name="PBI",
                residue_number=1,
                coordinates=np.array([-1.2, 0.0, 0.0], dtype=float),
                element="Pb",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="PB2",
                residue_name="PBI",
                residue_number=1,
                coordinates=np.array([1.2, 0.0, 0.0], dtype=float),
                element="Pb",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="I1",
                residue_name="PBI",
                residue_number=1,
                coordinates=np.array([-1.2, 2.6, 0.0], dtype=float),
                element="I",
            ),
            PDBAtom(
                atom_id=4,
                atom_name="I2",
                residue_name="PBI",
                residue_number=1,
                coordinates=np.array([1.2, -2.6, 0.0], dtype=float),
                element="I",
            ),
        ],
        source_name="solvent_shell_mixed_none",
    )
    output_path = tmp_path / "solvent_shell_mixed_none_input.pdb"
    structure.write_pdb_file(output_path)
    return output_path


def _write_test_solvent_shell_xyz(
    tmp_path: Path,
    *,
    reference_path: Path,
) -> Path:
    reference_structure = PDBStructure.from_file(reference_path)
    xyz_lines = ["7", "solvent shell xyz"]
    for shift in (
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([4.0, 0.0, 0.0], dtype=float),
    ):
        for atom in reference_structure.atoms:
            coordinates = atom.coordinates + shift
            xyz_lines.append(
                f"{atom.element} "
                f"{coordinates[0]:.6f} "
                f"{coordinates[1]:.6f} "
                f"{coordinates[2]:.6f}"
            )
    xyz_lines.append("Pb 9.000000 0.000000 0.000000")
    output_path = tmp_path / "solvent_shell_input.xyz"
    output_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    return output_path


def _write_test_partial_solvent_shell_xyz(tmp_path: Path) -> Path:
    xyz_lines = [
        "3",
        "partial solvent xyz",
        "Pb 0.000000 0.000000 0.000000",
        "I 2.000000 0.000000 0.000000",
        "O 0.000000 2.000000 0.000000",
    ]
    output_path = tmp_path / "partial_solvent_shell_input.xyz"
    output_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    return output_path


def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeSettings:
    def __init__(self):
        self.values: dict[str, object] = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class _FakePackmolDockerClient:
    def __init__(self):
        self.listed_containers: int = 0
        self.verified_links: list[PackmolDockerLink] = []
        self.listed_directories: list[tuple[str, str]] = []
        self.synced_calls: list[tuple[str, Path, str | None]] = []

    def list_containers(self) -> list[PackmolDockerContainerRecord]:
        self.listed_containers += 1
        return [
            PackmolDockerContainerRecord(
                name="analysis-helper",
                image_name="ubuntu:22.04",
                status="Exited (0) 2 hours ago",
            ),
            PackmolDockerContainerRecord(
                name="packmol-dev",
                image_name="packmol:test-image",
                status="Up 8 minutes",
            ),
        ]

    def verify_link(
        self,
        link: PackmolDockerLink,
    ) -> PackmolDockerValidationResult:
        self.verified_links.append(
            PackmolDockerLink.from_dict(link.to_dict()) or link
        )
        return PackmolDockerValidationResult(
            verified_at="2026-04-17T12:30:00",
            container_id="sha256:fakepackmol",
            image_name="packmol:test-image",
            packmol_command_path="/usr/local/bin/packmol",
            packmol_version="Packmol version 20.14.4",
            container_project_root=link.container_project_root,
        )

    def list_directories(
        self,
        link: PackmolDockerLink,
        directory: str,
    ) -> list[PackmolDockerDirectoryEntry]:
        self.listed_directories.append((link.container_name, directory))
        mapping = {
            "/packmol_input_files": [
                PackmolDockerDirectoryEntry(
                    name="project_alpha",
                    path="/packmol_input_files/project_alpha",
                ),
                PackmolDockerDirectoryEntry(
                    name="project_beta",
                    path="/packmol_input_files/project_beta",
                ),
            ],
            "/packmol_input_files/project_alpha": [
                PackmolDockerDirectoryEntry(
                    name="subrun",
                    path="/packmol_input_files/project_alpha/subrun",
                )
            ],
        }
        return mapping.get(directory, [])

    def sync_packmol_inputs(
        self,
        link: PackmolDockerLink,
        local_packmol_inputs_dir: str | Path,
        *,
        packmol_setup_metadata=None,
    ) -> PackmolDockerSyncResult:
        local_dir = Path(local_packmol_inputs_dir).resolve()
        self.synced_calls.append(
            (
                link.container_name,
                local_dir,
                (
                    None
                    if packmol_setup_metadata is None
                    else packmol_setup_metadata.packmol_input_path
                ),
            )
        )
        input_name = (
            "packmol_combined.inp"
            if packmol_setup_metadata is None
            else Path(packmol_setup_metadata.packmol_input_path).name
        )
        output_name = (
            "packed_combined.pdb"
            if packmol_setup_metadata is None
            else packmol_setup_metadata.packed_output_filename
        )
        return PackmolDockerSyncResult(
            synced_at="2026-04-17T12:45:00",
            remote_packmol_inputs_dir=str(link.remote_packmol_inputs_dir()),
            remote_packmol_input_path=str(
                link.remote_packmol_inputs_dir() / input_name
            ),
            remote_packed_output_path=str(
                link.remote_packmol_inputs_dir() / output_name
            ),
            synced_file_count=4,
        )


class _FakeRepresentativeStructuresWindow(QWidget):
    project_results_changed = Signal(str)

    def __init__(self):
        super().__init__()


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


def _configure_integrated_rmcsetup_solvent_panel(
    window: RMCSetupMainWindow,
    *,
    minimum_separation_a: float = 1.4,
    pb_target_coordination: float = 1.0,
    pb_cutoff_a: float = 2.6,
) -> None:
    window.solvent_reference_source_combo.setCurrentIndex(0)
    preset_index = window.solvent_preset_combo.findData("dmf")
    assert preset_index >= 0
    window.solvent_preset_combo.setCurrentIndex(preset_index)
    window.solvent_minimum_separation_spin.setValue(minimum_separation_a)
    window._analyze_representative_solvent_states()

    pb_row = None
    for row in range(window.solvent_cutoff_table.rowCount()):
        if window.solvent_cutoff_table.item(row, 0).text() == "Pb":
            pb_row = row
            break
    assert pb_row is not None

    center_item = window.solvent_cutoff_table.item(pb_row, 2)
    assert center_item is not None
    center_item.setCheckState(Qt.CheckState.Checked)

    coordination_spin = window.solvent_cutoff_table.cellWidget(pb_row, 3)
    cutoff_spin = window.solvent_cutoff_table.cellWidget(pb_row, 4)
    assert isinstance(coordination_spin, QDoubleSpinBox)
    assert isinstance(cutoff_spin, QDoubleSpinBox)
    coordination_spin.setValue(pb_target_coordination)
    cutoff_spin.setValue(pb_cutoff_a)


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
    assert state.rmcsetup_paths.packmol_docker_link_path.is_file()


def test_fullrmc_project_loader_restores_packmol_docker_link(tmp_path):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    link = PackmolDockerLink(
        display_name="Mounted Packmol Container",
        container_name="packmol-test",
        container_project_root="/packmol_input_files/project_alpha",
        packmol_command="packmol",
        shell_command="bash",
        packmol_version="Packmol version 20.14.4",
        linked_at="2026-04-17T09:00:00",
        last_verified_at="2026-04-17T09:05:00",
        container_id="sha256:restoreme",
        image_name="packmol:latest",
        packmol_command_path="/usr/local/bin/packmol",
    )
    save_packmol_docker_link_metadata(
        project_dir / "rmcsetup" / "packmol_docker_link.json",
        link,
    )

    state = load_rmc_project_source(project_dir)

    assert state.packmol_docker_link is not None
    assert state.packmol_docker_link.container_name == "packmol-test"
    assert state.packmol_docker_link.container_project_root == (
        "/packmol_input_files/project_alpha"
    )
    assert state.packmol_docker_link.packmol_command_path == (
        "/usr/local/bin/packmol"
    )


def test_packmol_docker_project_root_validation_targets_input_mount():
    assert container_project_root_is_valid("/packmol_input_files")
    assert container_project_root_is_valid(
        "/packmol_input_files/project_alpha"
    )
    assert container_project_root_is_valid(
        "/packmol_input_files/project_alpha/.."
    )
    assert container_project_root_is_valid(
        "/packmol_input_files/project_alpha/subrun"
    )
    assert container_project_root_is_valid(
        "/packmol_input_files/project_alpha/../subrun"
    )
    assert (
        container_project_root_is_valid("/packmol_input_files_extra") is False
    )
    assert (
        container_project_root_is_valid("/packmol_input_files/../etc") is False
    )
    assert (
        container_project_root_is_valid(
            "/packmol_input_files/project_alpha/../../etc"
        )
        is False
    )
    assert container_project_root_is_valid("/tmp/project_alpha") is False


def test_fullrmc_project_loader_discovers_distribution_scoped_runs(
    tmp_path,
):
    project_dir, _paths, artifact_paths = (
        _build_sample_distribution_scoped_saxs_project(tmp_path)
    )

    state = load_rmc_project_source(project_dir)

    assert [run.run_name for run in state.valid_runs] == [
        "dream_run_002",
        "dream_run_001",
    ]
    assert all(
        run.relative_path.startswith("saved_distributions/")
        for run in state.valid_runs
    )
    assert state.favorite_selection is not None
    assert state.find_run_for_selection(state.favorite_selection) is not None
    assert (
        state.valid_runs[0].run_dir.parent == artifact_paths.dream_runtime_dir
    )


def test_fullrmc_project_loader_uses_active_predicted_distribution_rows(
    tmp_path,
):
    project_dir, predicted_source = (
        _build_sample_predicted_distribution_scoped_saxs_project(tmp_path)
    )

    state = load_rmc_project_source(project_dir)

    assert state.cluster_validation.is_valid is True
    assert any(
        row.get("source_kind") == "predicted_structure"
        for row in state.cluster_validation.expected_rows
    )
    assert all(
        row.get("structure") != "Zn1"
        for row in state.cluster_validation.current_rows
    )

    selection = state.favorite_selection
    assert selection is not None
    distribution = build_distribution_selection(state, selection)
    predicted_entry = next(
        entry
        for entry in distribution.entries
        if entry.source_kind == "predicted_structure"
    )

    assert predicted_entry.structure == "Zn1"
    assert predicted_entry.motif == "predicted_rank01"
    assert predicted_entry.source_file == str(predicted_source)
    assert predicted_entry.source_file_name == predicted_source.name
    assert predicted_entry.cluster_count == 1


def test_fullrmc_project_loader_falls_back_to_project_root_runs(
    tmp_path,
):
    project_dir, paths = _build_sample_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
        allow_legacy_fallback=False,
    )
    manager.ensure_artifact_dirs(artifact_paths)
    manager._write_distribution_metadata(
        settings,
        artifact_paths=artifact_paths,
    )

    state = load_rmc_project_source(project_dir)

    assert [run.run_name for run in state.valid_runs] == [
        "dream_run_002",
        "dream_run_001",
    ]
    assert all(
        run.run_dir.parent == paths.dream_runtime_dir
        for run in state.valid_runs
    )
    assert all(
        not run.relative_path.startswith("saved_distributions/")
        for run in state.valid_runs
    )


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
        _integrated_solvent_handling_settings(),
    )
    reloaded = load_solvent_handling_metadata(
        state.rmcsetup_paths.solvent_handling_path
    )

    assert metadata.reference_name == "dmf"
    assert metadata.detected_distribution_status == "no_solvent"
    assert len(metadata.entries) == 2
    assert all(
        entry.atom_count_completed > entry.atom_count_no_solvent
        for entry in metadata.entries
    )
    assert all(entry.solvent_atoms_added > 0 for entry in metadata.entries)
    assert all(
        Path(entry.completed_pdb).parent
        == state.rmcsetup_paths.pdb_with_solvent_dir / entry.structure
        for entry in metadata.entries
    )
    assert all(
        Path(entry.no_solvent_pdb).parent
        == state.rmcsetup_paths.pdb_no_solvent_dir / entry.structure
        for entry in metadata.entries
    )
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
        _integrated_solvent_handling_settings(),
    )

    anchored_entry = next(
        entry
        for entry in metadata.entries
        if entry.structure == "PbI2O" and entry.motif == "motif_1"
    )
    assert metadata.detected_distribution_status == "no_solvent"
    assert anchored_entry.atom_count_no_solvent == 3
    assert anchored_entry.atom_count_completed == 15
    assert anchored_entry.solvent_atoms_added == 12
    assert anchored_entry.solvent_molecules_added == 1
    assert (
        anchored_entry.completion_strategy
        == "rebuilt_from_no_solvent_distribution"
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

    minimum_distance = min(
        np.linalg.norm(solvent_atom.coordinates - solute_atom.coordinates)
        for solvent_atom in solvent_atoms
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
        _integrated_solvent_handling_settings(
            reference_source="custom",
            custom_reference_path=str(custom_solvent),
            director_atom_name="O1",
        ),
    )

    assert metadata.reference_name == "water_ref"
    assert metadata.reference_residue_name == "HOH"
    assert metadata.detected_distribution_status == "no_solvent"
    added_counts = {
        (entry.structure, entry.motif): entry.solvent_atoms_added
        for entry in metadata.entries
    }
    molecule_counts = {
        (entry.structure, entry.motif): entry.solvent_molecules_added
        for entry in metadata.entries
    }
    assert added_counts[("PbI2", "no_motif")] == 3
    assert added_counts[("PbI2O", "motif_1")] == 3
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
        _integrated_solvent_handling_settings(),
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
    assert (
        zn_entry.completion_strategy
        == "preserved_without_matching_coordination_settings"
    )
    assert [atom.element for atom in completed_structure.atoms] == ["Zn"]


def test_representative_solvent_distribution_ignores_single_atom_status(
    tmp_path,
):
    project_dir, _paths, _single_atom_path = (
        _build_sample_saxs_project_with_single_atom_model(tmp_path)
    )
    reference_path = _write_custom_solvent_pdb(tmp_path)
    complete_solvent_path = _write_test_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    for entry in state.representative_selection.representative_entries:
        if entry.atom_count > 1:
            entry.source_file = str(complete_solvent_path)
            entry.source_file_name = complete_solvent_path.name

    analysis = analyze_representative_solvent_distribution(
        state,
        _integrated_solvent_handling_settings(
            reference_source="custom",
            custom_reference_path=str(reference_path),
            director_atom_name="O1",
        ),
    )

    assert analysis.distribution_status == "complete_solvent"
    assert analysis.distribution_status_entry_count == 2
    assert analysis.ignored_distribution_status_entry_count == 1
    assert "Ignored 1 single-atom representative" in (
        analysis.distribution_note
    )
    assert "Zn1: No solvent molecules detected" in analysis.summary_text()
    assert "ignored for distribution state" in analysis.summary_text()


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


def test_build_packmol_plan_requires_all_positive_weight_representatives(
    tmp_path,
):
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    metadata = select_first_file_representatives(
        state,
        selection,
    )
    metadata.representative_entries = metadata.representative_entries[:1]
    state.representative_selection = metadata
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

    with pytest.raises(
        ValueError, match="exactly one representative structure"
    ):
        build_packmol_plan(
            state,
            PackmolPlanningSettings(
                planning_mode="per_element",
                box_side_length_a=80.0,
            ),
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
        _integrated_solvent_handling_settings(),
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


def test_build_packmol_plan_tracks_solvent_allocation(tmp_path):
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
        _integrated_solvent_handling_settings(),
    )

    metadata = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )

    allocation = metadata.solvent_allocation

    assert allocation is not None
    assert allocation.reference_name == "dmf"
    assert allocation.target_solvent_molecules == int(
        metadata.target_box_composition["solvent_molecules"]
    )
    assert allocation.solvent_molecules_in_clusters == sum(
        entry.solvent_molecules_total for entry in allocation.entries
    )
    assert allocation.free_solvent_molecules == max(
        0,
        allocation.target_solvent_molecules
        - allocation.solvent_molecules_in_clusters,
    )
    assert any(
        entry.solvent_molecules_total > 0 for entry in allocation.entries
    )
    assert "Cluster solvent molecules:" in metadata.summary_text()


def test_build_packmol_plan_allocates_missing_solute_components(tmp_path):
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
        solute_stoich="C1H6N1Pb1I2",
        solvent_stoich="C3H7NO",
        molar_mass_solute=493.0,
        molar_mass_solvent=73.09,
        mass_solute=4.93,
        mass_solvent=95.07,
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
            supplemental_components=(
                PackmolSupplementalComponentSettings(
                    role="solute",
                    reference="ma",
                    residue_name="MAI",
                ),
            ),
        ),
    )

    allocation = metadata.supplemental_allocation

    assert allocation is not None
    assert allocation.target_solute_formula_units > 0
    assert allocation.unfilled_solute_element_totals == {}
    assert allocation.entries[0].planned_count == (
        allocation.target_solute_formula_units
    )
    assert allocation.entries[0].element_counts == {
        "C": 1,
        "H": 6,
        "N": 1,
    }
    assert metadata.achieved_element_number_density_a3["C"] > 0
    assert "Supplemental solute accounting:" in (
        state.rmcsetup_paths.packmol_plan_report_path.read_text(
            encoding="utf-8"
        )
    )


def test_build_packmol_plan_requires_components_for_absent_solute_species(
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
        solute_stoich="C1H6N1Pb1I2",
        solvent_stoich="C3H7NO",
        molar_mass_solute=493.0,
        molar_mass_solvent=73.09,
        mass_solute=4.93,
        mass_solvent=95.07,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=solution_settings,
        result=calculate_solution_properties(solution_settings),
    )

    with pytest.raises(
        ValueError,
        match="Supplemental solute components are required",
    ):
        build_packmol_plan(
            state,
            PackmolPlanningSettings(
                planning_mode="per_element",
                box_side_length_a=80.0,
            ),
        )


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
        _integrated_solvent_handling_settings(),
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
    assert metadata.free_solvent_reference_name == "dmf"
    assert state.packmol_planning.solvent_allocation is not None
    assert metadata.target_solvent_molecules == (
        state.packmol_planning.solvent_allocation.target_solvent_molecules
    )
    assert metadata.solvent_molecules_in_clusters == (
        state.packmol_planning.solvent_allocation.solvent_molecules_in_clusters
    )
    assert metadata.free_solvent_molecules == (
        state.packmol_planning.solvent_allocation.free_solvent_molecules
    )
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
    assert "Cluster solvent molecules:" in audit_text
    assert "Count-normalized weights" in audit_text
    assert Path(metadata.build_report_path).is_file()
    build_report_text = Path(metadata.build_report_path).read_text(
        encoding="utf-8"
    )
    assert "Source input information" in build_report_text
    assert "Computed solvent molecules:" in build_report_text
    assert "Cluster solvent molecules:" in build_report_text
    assert "Free solvent molecules:" in build_report_text
    assert "Target total number density:" in build_report_text

    solvated_entry = next(
        entry for entry in metadata.entries if entry.solvent_atom_count > 0
    )
    solvated_structure = PDBStructure.from_file(solvated_entry.packmol_pdb)
    solute_atoms = [
        atom
        for atom in solvated_structure.atoms
        if atom.residue_name == solvated_entry.residue_name
    ]
    solvent_atoms = [
        atom
        for atom in solvated_structure.atoms
        if atom.residue_name != solvated_entry.residue_name
    ]

    assert solute_atoms
    assert solvent_atoms
    assert len(solute_atoms) == solvated_entry.solute_atom_count
    assert len(solvent_atoms) == solvated_entry.solvent_atom_count
    assert {atom.residue_number for atom in solute_atoms} == {1}
    assert {atom.residue_name for atom in solvent_atoms} == {"DMF"}
    assert min(atom.residue_number for atom in solvent_atoms) >= 2
    assert len({atom.residue_number for atom in solvent_atoms}) == (
        solvated_entry.solvent_residue_count
    )


def test_packmol_preparation_keeps_solvent_residue_when_solute_is_last():
    source_structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=229,
                atom_name="O1",
                residue_name="DMF",
                residue_number=20,
                coordinates=np.asarray([10.073, 14.645, 2.873]),
                element="O",
            ),
            PDBAtom(
                atom_id=230,
                atom_name="N1",
                residue_name="DMF",
                residue_number=20,
                coordinates=np.asarray([8.539, 14.935, 4.538]),
                element="N",
            ),
            PDBAtom(
                atom_id=639,
                atom_name="PB3",
                residue_name="PBI",
                residue_number=56,
                coordinates=np.asarray([13.198, 14.311, 3.729]),
                element="Pb",
            ),
            PDBAtom(
                atom_id=642,
                atom_name="I3",
                residue_name="PBI",
                residue_number=59,
                coordinates=np.asarray([16.059, 14.562, 4.307]),
                element="I",
            ),
            PDBAtom(
                atom_id=643,
                atom_name="I4",
                residue_name="PBI",
                residue_number=60,
                coordinates=np.asarray([12.481, 15.296, 6.518]),
                element="I",
            ),
        ],
        source_name="solvent_first_cluster",
    )

    prepared = packmol_setup_module._prepare_packmol_structure(
        source_structure,
        residue_name="CAH",
        solvent_residue_names=frozenset({"DMF"}),
        expected_solute_element_counts={"Pb": 1, "I": 2},
        solute_atom_count=2,
    )

    residue_names = [atom.residue_name for atom in prepared.structure.atoms]
    residue_numbers = [
        atom.residue_number for atom in prepared.structure.atoms
    ]

    assert residue_names == ["DMF", "DMF", "CAH", "CAH", "CAH"]
    assert residue_numbers == [2, 2, 1, 1, 1]
    assert prepared.solute_atom_count == 3
    assert prepared.solvent_atom_count == 2
    assert prepared.solvent_residue_names == ("DMF",)


def test_packmol_preparation_can_identify_formula_solute_without_metadata():
    source_structure = PDBStructure(
        atoms=[
            PDBAtom(
                atom_id=1,
                atom_name="O1",
                residue_name="DMF",
                residue_number=20,
                coordinates=np.asarray([0.0, 0.0, 0.0]),
                element="O",
            ),
            PDBAtom(
                atom_id=2,
                atom_name="C1",
                residue_name="DMF",
                residue_number=20,
                coordinates=np.asarray([1.0, 0.0, 0.0]),
                element="C",
            ),
            PDBAtom(
                atom_id=3,
                atom_name="PB1",
                residue_name="PBI",
                residue_number=56,
                coordinates=np.asarray([2.0, 0.0, 0.0]),
                element="Pb",
            ),
            PDBAtom(
                atom_id=4,
                atom_name="I1",
                residue_name="PBI",
                residue_number=59,
                coordinates=np.asarray([3.0, 0.0, 0.0]),
                element="I",
            ),
            PDBAtom(
                atom_id=5,
                atom_name="I2",
                residue_name="PBI",
                residue_number=60,
                coordinates=np.asarray([4.0, 0.0, 0.0]),
                element="I",
            ),
        ],
        source_name="formula_fallback_cluster",
    )

    prepared = packmol_setup_module._prepare_packmol_structure(
        source_structure,
        residue_name="CAH",
        expected_solute_element_counts={"Pb": 1, "I": 2},
    )

    assert [atom.residue_name for atom in prepared.structure.atoms] == [
        "DMF",
        "DMF",
        "CAH",
        "CAH",
        "CAH",
    ]
    assert prepared.solute_atom_count == 3
    assert prepared.solvent_residue_names == ("DMF",)


def test_build_packmol_setup_writes_supplemental_solute_components(tmp_path):
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
        solute_stoich="C1H6N1Pb1I2",
        solvent_stoich="C3H7NO",
        molar_mass_solute=493.0,
        molar_mass_solvent=73.09,
        mass_solute=4.93,
        mass_solvent=95.07,
    )
    state.solution_properties = save_solution_properties_metadata(
        state.rmcsetup_paths.solution_properties_path,
        settings=solution_settings,
        result=calculate_solution_properties(solution_settings),
    )
    state.packmol_planning = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
            free_solvent_reference="dmf",
            supplemental_components=(
                PackmolSupplementalComponentSettings(
                    role="solute",
                    reference="ma",
                    residue_name="MAI",
                ),
            ),
        ),
    )

    metadata = build_packmol_setup(
        state,
        PackmolSetupSettings(
            tolerance_angstrom=2.2,
            free_solvent_reference="dmf",
        ),
    )

    assert metadata.supplemental_entries
    supplemental_entry = metadata.supplemental_entries[0]
    supplemental_structure = PDBStructure.from_file(
        supplemental_entry.packmol_pdb
    )
    packmol_text = Path(metadata.packmol_input_path).read_text(
        encoding="utf-8"
    )
    build_report_text = Path(metadata.build_report_path).read_text(
        encoding="utf-8"
    )

    assert supplemental_entry.planned_count == (
        state.packmol_planning.supplemental_allocation.target_solute_formula_units
    )
    assert {atom.residue_name for atom in supplemental_structure.atoms} == {
        "MAI"
    }
    assert supplemental_entry.atom_count == 8
    assert f"structure {Path(supplemental_entry.packmol_pdb).name}" in (
        packmol_text
    )
    assert f"  number {supplemental_entry.planned_count}" in packmol_text
    assert "Supplemental solute accounting" in build_report_text
    assert "Supplemental Packmol components" in build_report_text


def test_build_packmol_setup_requires_all_positive_weight_representatives(
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
    state.packmol_planning = build_packmol_plan(
        state,
        PackmolPlanningSettings(
            planning_mode="per_element",
            box_side_length_a=80.0,
        ),
    )
    assert state.representative_selection is not None
    state.representative_selection.representative_entries = (
        state.representative_selection.representative_entries[:1]
    )

    with pytest.raises(
        ValueError, match="exactly one representative structure"
    ):
        build_packmol_setup(
            state,
            PackmolSetupSettings(
                include_free_solvent=False,
            ),
        )


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
        _integrated_solvent_handling_settings(),
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
        _integrated_solvent_handling_settings(),
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
    assert "DMF" not in merged_text
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
    assert "/rmcsetup/representative_structures" in (
        window.output_summary_box.toPlainText()
    )
    assert window.solution_group.isEnabled() is True
    assert "No saved solution-properties calculation yet" in (
        window.solution_output_box.toPlainText()
    )
    assert window.representative_group.title() == "Representative Structures"
    assert window.compute_representatives_button.text() == (
        "Open Representative Structures"
    )
    assert window.preview_representatives_button.text() == (
        "Reload Saved Representative Structures"
    )
    assert "No representative structures have been saved yet" in (
        window.representative_summary_box.toPlainText()
    )


def test_rmcsetup_main_window_lists_distribution_scoped_dream_runs(tmp_path):
    qapp()
    project_dir, _paths, _artifact_paths = (
        _build_sample_distribution_scoped_saxs_project(tmp_path)
    )

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert [window.dream_run_combo.itemText(i) for i in range(2)] == [
        "dream_run_002",
        "dream_run_001",
    ]
    assert window.dream_run_combo.currentText() == "dream_run_001"
    assert window._project_source_state is not None
    assert all(
        run.relative_path.startswith("saved_distributions/")
        for run in window._project_source_state.valid_runs
    )


def test_rmcsetup_representative_panel_opens_representative_structures_tool(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    fake_window = _FakeRepresentativeStructuresWindow()
    launched: dict[str, Path | None] = {}

    import saxshell.representativefinder.ui.main_window as representativefinder_ui_module

    def fake_launch(
        *,
        initial_project_dir=None,
        initial_input_path=None,
    ):
        launched["initial_project_dir"] = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        launched["initial_input_path"] = (
            None
            if initial_input_path is None
            else Path(initial_input_path).expanduser().resolve()
        )
        return fake_window

    monkeypatch.setattr(
        representativefinder_ui_module,
        "launch_representativefinder_ui",
        fake_launch,
    )

    window.compute_representatives_button.click()

    assert launched["initial_project_dir"] == project_dir.resolve()
    assert launched["initial_input_path"] == (
        window._project_source_state.settings.resolved_clusters_dir
    )
    assert fake_window in window._child_tool_windows

    fake_window.project_results_changed.emit(str(project_dir.resolve()))
    QApplication.processEvents()

    assert "Representative structures were updated in the dedicated tool" in (
        window.run_log_box.toPlainText()
    )
    fake_window.close()
    window.close()


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
    assert window._right_panel.isAncestorOf(window.solvent_group)
    assert window._right_panel.isAncestorOf(window.packmol_group)
    assert window._right_panel.isAncestorOf(window.run_log_group)


def test_rmcsetup_readiness_sections_can_collapse_without_hiding_status(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    readiness_sections = {
        "project_source": ["project_source"],
        "dream_selection": ["dream_selection"],
        "solution_properties": ["solution_properties"],
        "representative_selection": ["representative_selection"],
        "solvent_outputs": ["solvent_outputs"],
        "packmol": ["packmol_plan", "packmol_setup"],
    }

    for section_key, readiness_keys in readiness_sections.items():
        toggle = window._section_toggle_buttons[section_key]
        content = window._section_content_widgets[section_key]
        assert content.isHidden() is False
        for readiness_key in readiness_keys:
            checkbox = window._readiness_checkboxes[readiness_key]
            assert content.isAncestorOf(checkbox) is False
            assert checkbox.isHidden() is False
        toggle.click()
        assert content.isHidden() is True
        assert toggle.text() == "Expand"
        for readiness_key in readiness_keys:
            assert (
                window._readiness_checkboxes[readiness_key].isHidden() is False
            )
        toggle.click()
        assert content.isHidden() is False
        assert toggle.text() == "Collapse"

    representative_content = window._section_content_widgets[
        "representative_selection"
    ]
    assert representative_content.isAncestorOf(window.solvent_group) is False
    window._section_toggle_buttons["representative_selection"].click()
    assert representative_content.isHidden() is True
    assert window.solvent_group.isHidden() is False
    assert window._readiness_checkboxes["solvent_outputs"].isHidden() is False


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


def test_packmol_docker_link_dialog_validates_container_and_updates_tree(
    tmp_path,
):
    qapp()
    client = _FakePackmolDockerClient()
    dialog = fullrmc_ui_module.PackmolDockerLinkDialog(
        recent_presets=[
            PackmolDockerLink(
                display_name="Mounted Packmol",
                container_name="packmol-dev",
                container_project_root="/packmol_input_files",
            )
        ],
        docker_client=client,
    )

    assert client.listed_containers >= 1
    assert dialog.available_container_combo.count() == 2
    assert "packmol-dev" in dialog.available_container_combo.itemText(1)
    assert dialog._test_connection() is True
    assert "Docker validation succeeded." in dialog.status_box.toPlainText()
    assert "Packmol version 20.14.4" in dialog.status_box.toPlainText()
    assert dialog.directory_tree.topLevelItemCount() == 1

    root_item = dialog.directory_tree.topLevelItem(0)
    assert root_item.text(0) == "/packmol_input_files"
    assert root_item.childCount() == 2

    first_child = root_item.child(0)
    dialog.directory_tree.setCurrentItem(first_child)
    dialog._use_selected_directory()

    assert dialog.container_root_edit.text() == (
        "/packmol_input_files/project_alpha"
    )
    dialog.close()


def test_packmol_docker_link_dialog_can_load_discovered_container_name():
    qapp()
    dialog = fullrmc_ui_module.PackmolDockerLinkDialog(
        docker_client=_FakePackmolDockerClient(),
    )

    assert dialog.available_container_combo.count() == 2
    dialog.available_container_combo.setCurrentIndex(1)
    dialog._use_available_container()

    assert dialog.container_name_edit.text() == "packmol-dev"
    assert "Press Test Container to verify Packmol" in (
        dialog.status_box.toPlainText()
    )
    dialog.close()


def test_packmol_docker_link_dialog_rejects_invalid_container_project_root():
    qapp()

    class _RejectingClient(_FakePackmolDockerClient):
        def verify_link(self, link):
            raise RuntimeError(
                "Container project root must be inside /packmol_input_files "
                "so Packmol input files stay inside the expected bind-mounted folder."
            )

    dialog = fullrmc_ui_module.PackmolDockerLinkDialog(
        docker_client=_RejectingClient(),
    )
    dialog.container_name_edit.setText("packmol-dev")
    dialog.container_root_edit.setText("/tmp/not_allowed")

    assert dialog._test_connection() is False
    assert "must be inside /packmol_input_files" in (
        dialog.status_box.toPlainText()
    )
    dialog.close()


def test_packmol_docker_link_dialog_explains_docker_daemon_failure():
    qapp()

    class _DaemonDownClient(_FakePackmolDockerClient):
        def verify_link(self, link):
            del link
            raise RuntimeError(
                'WARNING: Plugin "/Users/test/.docker/cli-plugins/docker-scan" '
                "is not valid: failed to fetch metadata: fork/exec "
                "/Users/test/.docker/cli-plugins/docker-scan: no such file "
                "or directory\n"
                "ERROR: Cannot connect to the Docker daemon at "
                "unix:///Users/test/.docker/run/docker.sock. Is the docker "
                "daemon running?\n"
                "errors pretty printing info"
            )

    dialog = fullrmc_ui_module.PackmolDockerLinkDialog(
        docker_client=_DaemonDownClient(),
    )
    dialog.container_name_edit.setText("packmol-dev")

    assert dialog._test_connection() is False
    assert (
        "Docker Desktop or the Docker daemon does not appear to be running."
        in (dialog.status_box.toPlainText())
    )
    assert "wait for `docker info` to succeed" in (
        dialog.status_box.toPlainText()
    )
    assert "Cannot connect to the Docker daemon" in (
        dialog.status_box.toPlainText()
    )
    dialog.close()


def test_rmcsetup_tools_menu_can_link_packmol_docker_container(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    settings_store = _FakeSettings()
    linked = PackmolDockerLink(
        display_name="Saved Container",
        container_name="packmol-dev",
        container_project_root="/packmol_input_files/project_alpha",
        packmol_command="packmol",
        shell_command="sh",
        packmol_version="Packmol version 20.14.4",
        last_verified_at="2026-04-17T12:30:00",
        container_id="sha256:dialog",
        image_name="packmol:test-image",
        packmol_command_path="/usr/local/bin/packmol",
    )

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def exec(self):
            return 1

        def selected_link(self):
            return PackmolDockerLink.from_dict(linked.to_dict())

    monkeypatch.setattr(
        RMCSetupMainWindow,
        "_packmol_docker_settings",
        lambda self: settings_store,
    )
    monkeypatch.setattr(
        fullrmc_ui_module,
        "PackmolDockerLinkDialog",
        _FakeDialog,
    )

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window.tools_menu.title() == "Tools"
    assert window.link_packmol_docker_action.text() == (
        "Link Packmol Docker Container"
    )

    window._open_packmol_docker_link_dialog()

    assert window._project_source_state is not None
    assert window._project_source_state.packmol_docker_link is not None
    assert window._project_source_state.packmol_docker_link.container_name == (
        "packmol-dev"
    )
    assert "packmol-dev" in window.packmol_docker_summary_box.toPlainText()
    assert "Packmol version 20.14.4" in (
        window.packmol_docker_summary_box.toPlainText()
    )
    raw_presets = settings_store.value("packmol_docker_presets", "[]")
    preset_payload = json.loads(raw_presets)
    assert preset_payload[0]["container_name"] == "packmol-dev"

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert "packmol-dev" in reloaded.packmol_docker_summary_box.toPlainText()


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
        "PbI2 - DMSO - 0.405 M"
    ].solute_density_g_per_ml == pytest.approx(6.16)
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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._solvent_distribution_analysis = None
    assert window.build_solvent_outputs_button.isEnabled() is True
    window._build_representative_solvent_outputs()

    assert "reference molecule: dmf" in (
        window.solvent_summary_box.toPlainText().lower()
    )
    assert (
        "Detected representative distribution state: No solvent molecules detected"
        in (window.solvent_summary_box.toPlainText())
    )
    assert "Minimum solvent atom separation: 1.4 A" in (
        window.solvent_summary_box.toPlainText()
    )
    assert window.generated_pdb_table.rowCount() == 2
    generated_rows = {
        (
            window.generated_pdb_table.item(row, 0).text(),
            window.generated_pdb_table.item(row, 1).text(),
        ): row
        for row in range(window.generated_pdb_table.rowCount())
    }
    assert set(generated_rows) == {
        ("PbI2", "No solvent molecules detected"),
        ("PbI2O/motif_1", "Partial solvent molecules detected"),
    }
    window.generated_pdb_table.selectRow(
        generated_rows[("PbI2", "No solvent molecules detected")]
    )
    details_text = window.generated_pdb_details_box.toPlainText()
    assert "Atom count: 15" in details_text
    assert "Element counts: C:3, H:7, I:2, N:1, O:1, Pb:1" in details_text
    assert "Solvent molecules matching reference residue DMF: 1" in (
        details_text
    )
    assert "PBI 1: 3 atoms (I:2, Pb:1)" in details_text
    assert "12 atoms (C:3, H:7, N:1, O:1)" in details_text
    assert "1: Pb1 (Pb) -> PBI 1" in details_text
    assert "O1 (O) -> DMF" in details_text
    assert window.generated_pdb_viewer.current_structure is not None

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
    assert "Built solvent-decorated representative PDB outputs." in (
        window.run_log_box.toPlainText()
    )
    assert "Opened generated PDB preview:" in window.run_log_box.toPlainText()

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.solvent_reference_source_combo.currentData() == "preset"
    assert reloaded.solvent_preset_combo.currentData() == "dmf"
    assert reloaded.solvent_minimum_separation_spin.value() == pytest.approx(
        1.4
    )
    assert reloaded.generated_pdb_table.rowCount() == 2
    assert "Representative entries exported: 2" in (
        reloaded.solvent_summary_box.toPlainText()
    )


def test_rmcsetup_reload_maps_representatives_to_current_dream_weights(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    metadata = select_first_file_representatives(state, selection)
    for entry in metadata.representative_entries:
        entry.param = entry.structure
        entry.selected_weight = 0.5
    for entry in metadata.distribution_selection.entries:
        entry.param = entry.structure
        entry.selected_weight = 0.5
    save_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path,
        metadata,
    )

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    headers = [
        window.generated_pdb_table.horizontalHeaderItem(column).text()
        for column in range(window.generated_pdb_table.columnCount())
    ]
    assert "DREAM Weight" in headers
    assert "DREAM Value" in headers
    weight_column = headers.index("DREAM Weight")
    value_column = headers.index("DREAM Value")
    mapped = {
        window.generated_pdb_table.item(row, 0).text(): (
            window.generated_pdb_table.item(row, weight_column).text(),
            window.generated_pdb_table.item(row, value_column).text(),
        )
        for row in range(window.generated_pdb_table.rowCount())
    }

    assert mapped["PbI2"] == ("w0", "0.25")
    assert mapped["PbI2O/motif_1"] == ("w1", "0.75")
    reloaded = load_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path
    )
    assert reloaded is not None
    assert {
        (entry.structure, entry.motif): (entry.param, entry.selected_weight)
        for entry in reloaded.representative_entries
    } == {
        ("PbI2", "no_motif"): ("w0", pytest.approx(0.25)),
        ("PbI2O", "motif_1"): ("w1", pytest.approx(0.75)),
    }


def test_rmcsetup_representative_reset_keeps_saved_representative_sources(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    state.representative_selection = select_first_file_representatives(
        state,
        selection,
    )
    state.solvent_handling = build_representative_solvent_outputs(
        state,
        _integrated_solvent_handling_settings(),
    )
    source_paths = [
        Path(entry.source_file)
        for entry in state.representative_selection.representative_entries
    ]
    tracked_outputs = [
        Path(entry.completed_pdb) for entry in state.solvent_handling.entries
    ]
    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    reset = window._reset_representative_dependent_state(
        confirm=False,
        refresh=False,
        clear_reason="test",
    )

    assert reset is True
    assert window._project_source_state is not None
    assert window._project_source_state.representative_selection is not None
    assert all(path.is_file() for path in source_paths)
    assert not any(path.exists() for path in tracked_outputs)
    assert (
        load_solvent_handling_metadata(
            state.rmcsetup_paths.solvent_handling_path
        )
        is None
    )


def test_rmcsetup_imported_full_solvent_representatives_mark_solvent_ready(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    metadata = select_first_file_representatives(
        state,
        selection,
    )
    state.representative_selection = metadata
    solvent_metadata = build_representative_solvent_outputs(
        state,
        _integrated_solvent_handling_settings(),
    )

    solvent_lookup = {
        (entry.structure, entry.motif, entry.param): entry
        for entry in solvent_metadata.entries
    }
    for entry in metadata.representative_entries:
        solvent_entry = solvent_lookup[
            (entry.structure, entry.motif, entry.param)
        ]
        completed_path = Path(solvent_entry.completed_pdb).resolve()
        entry.source_dir = str(completed_path.parent)
        entry.source_file = str(completed_path)
        entry.source_file_name = completed_path.name
        entry.source_solvent_mode = "unknown"
    save_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path,
        metadata,
    )
    state.rmcsetup_paths.solvent_handling_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    reloaded_selection = load_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path
    )
    assert reloaded_selection is not None
    assert {
        entry.source_solvent_mode
        for entry in reloaded_selection.representative_entries
    } == {"fullsolv"}

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window._readiness_checkboxes["solvent_outputs"].isChecked() is True
    assert window.readiness_progress_bar.value() == 4
    assert window.generated_pdb_mode_combo.currentData() == "full_solvent"
    assert window.generated_pdb_table.rowCount() == 2
    assert {
        window.generated_pdb_table.item(row, 1).text()
        for row in range(window.generated_pdb_table.rowCount())
    } == {"Full solvent analyzed"}
    assert (
        "Imported representative structures already include the Full solvent structure set"
        in (window.solvent_summary_box.toPlainText())
    )
    assert (
        "The active representative source files already provide the Full "
        "solvent structure set."
    ) in window.solvent_status_stats_label.text()
    assert "Solvent Shell Builder readiness: Ready for Packmol" in (
        window.solvent_status_stats_label.text()
    )
    assert window.solvent_group.title() == "Solvent Shell Builder"
    assert window.analyze_solvent_outputs_button.isEnabled() is False
    assert window.build_solvent_outputs_button.isEnabled() is False
    assert window.solvent_cutoff_group.isEnabled() is False
    window.close()


def test_rmcsetup_imported_full_solvent_representatives_can_build_packmol_setup(
    tmp_path,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    state = load_rmc_project_source(project_dir)
    selection = state.favorite_selection
    assert selection is not None
    metadata = select_first_file_representatives(
        state,
        selection,
    )
    state.representative_selection = metadata
    save_solution_properties_metadata(
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
    solvent_metadata = build_representative_solvent_outputs(
        state,
        _integrated_solvent_handling_settings(),
    )

    solvent_lookup = {
        (entry.structure, entry.motif, entry.param): entry
        for entry in solvent_metadata.entries
    }
    for entry in metadata.representative_entries:
        solvent_entry = solvent_lookup[
            (entry.structure, entry.motif, entry.param)
        ]
        completed_path = Path(solvent_entry.completed_pdb).resolve()
        entry.source_dir = str(completed_path.parent)
        entry.source_file = str(completed_path)
        entry.source_file_name = completed_path.name
        entry.source_solvent_mode = "unknown"
    save_representative_selection_metadata(
        state.rmcsetup_paths.representative_selection_path,
        metadata,
    )
    state.rmcsetup_paths.solvent_handling_path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    window = RMCSetupMainWindow(initial_project_dir=project_dir)

    assert window._project_source_state is not None
    assert window._project_source_state.solvent_handling is None

    dmf_index = window.packmol_free_solvent_combo.findText("dmf")
    assert dmf_index >= 0
    window.packmol_free_solvent_combo.setCurrentIndex(dmf_index)
    window.packmol_box_side_spin.setValue(80.0)

    window._compute_packmol_plan()
    window._build_packmol_setup()

    assert "Total solvent molecules:" in (
        window.packmol_plan_summary_box.toPlainText()
    )
    assert "Cluster solvent molecules:" in (
        window.packmol_plan_summary_box.toPlainText()
    )
    assert "Free solvent structure: dmf" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert "Free solvent molecules:" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert window.open_packmol_setup_folder_button.isEnabled() is True
    assert "Built Packmol setup inputs and audit report." in (
        window.run_log_box.toPlainText()
    )
    window.close()


def test_solvent_shell_builder_analysis_detects_pdb_residue_types(tmp_path):
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )

    result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )

    assert result.input_format == "pdb"
    assert result.detected_solvent_molecules == 2
    assert result.has_solvent_molecules is True
    assert result.matched_atom_count == 6
    assert result.unmatched_atom_count == 1
    assert [
        summary.residue_name for summary in result.matched_residue_summaries
    ] == [
        "ALT",
        "HOH",
    ]
    assert {
        summary.residue_name: summary.residue_numbers
        for summary in result.matched_residue_summaries
    } == {
        "ALT": (3,),
        "HOH": (2,),
    }
    assert "Solvent molecules detected: 2" in result.summary_text()
    assert "ALT: 1 molecule(s)" in result.summary_text()


def test_solvent_shell_builder_analysis_detects_xyz_solvent_count(tmp_path):
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_solvent_shell_xyz(
        tmp_path,
        reference_path=reference_path,
    )

    result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )

    assert result.input_format == "xyz"
    assert result.detected_solvent_molecules == 2
    assert result.matched_residue_summaries == ()
    assert result.matched_atom_count == 6
    assert result.unmatched_atom_count == 1
    assert result.no_solvent_status_text == "no"
    assert result.partial_solvent_status_text == "no"
    assert result.complete_solvent_status_text == "yes"
    assert (
        result.cluster_solvent_status_text
        == "Complete solvent molecules detected."
    )
    assert "Matched residue types: n/a for XYZ inputs" in result.summary_text()


def test_solvent_shell_builder_analysis_infers_partial_xyz_solvent_candidates(
    tmp_path,
):
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_partial_solvent_shell_xyz(tmp_path)

    result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )

    assert result.input_format == "xyz"
    assert result.complete_solvent_molecule_count == 0
    assert result.partial_solvent_molecule_count == 1
    assert result.no_solvent_status_text == "no"
    assert result.partial_solvent_status_text == "yes"
    assert result.complete_solvent_status_text == "no"
    assert (
        result.cluster_solvent_status_text
        == "Partial solvent molecules detected."
    )
    assert len(result.residue_mismatch_summaries) == 1
    candidate = result.residue_mismatch_summaries[0]
    assert candidate.residue_name == "HOH"
    assert candidate.common_atom_count == 1
    assert candidate.reference_atom_count == 3
    assert candidate.missing_atom_names == ("H1", "H2")
    assert candidate.source_atom_ids == (3,)
    summary_text = result.summary_text()
    assert "Partial solvent candidate count: 1" in summary_text
    assert "XYZ partial solvent candidates:" in summary_text
    assert "source atom ids 3" in summary_text


def test_solvent_shell_builder_analysis_identifies_no_solvent_status(tmp_path):
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_no_solvent_shell_pdb(tmp_path)

    result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )

    assert result.input_format == "pdb"
    assert result.complete_solvent_molecule_count == 0
    assert result.partial_solvent_molecule_count == 0
    assert result.no_solvent_status_text == "yes"
    assert result.partial_solvent_status_text == "no"
    assert result.complete_solvent_status_text == "no"
    assert (
        result.cluster_solvent_status_text == "No solvent molecules detected."
    )
    assert "Cluster solvent status: No solvent molecules detected." in (
        result.summary_text()
    )


def test_solvent_shell_builder_analysis_preserves_incomplete_pdb_residues(
    tmp_path,
):
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_incomplete_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )

    result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )

    assert result.input_format == "pdb"
    assert result.detected_solvent_molecules == 1
    assert result.matched_atom_count == 3
    assert result.unmatched_atom_count == 3
    assert [
        summary.residue_name for summary in result.matched_residue_summaries
    ] == ["HOH"]
    assert result.matched_residue_summaries[0].residue_numbers == (2,)
    assert len(result.residue_mismatch_summaries) == 1
    mismatch = result.residue_mismatch_summaries[0]
    assert mismatch.residue_name == "HOH"
    assert mismatch.residue_number == 3
    assert mismatch.observed_atom_count == 2
    assert mismatch.common_atom_count == 2
    assert mismatch.reference_atom_count == 3
    assert mismatch.missing_atom_names == ("H2",)
    assert mismatch.extra_atom_names == ()
    assert result.no_solvent_status_text == "no"
    assert result.partial_solvent_status_text == "yes"
    assert result.complete_solvent_status_text == "yes"
    assert (
        result.cluster_solvent_status_text
        == "Complete and partial solvent molecules detected."
    )
    summary_text = result.summary_text()
    assert "Residue mismatches preserved: 1" in summary_text
    assert "HOH 3: missing reference atoms" in summary_text
    assert "missing H2" in summary_text


def test_solvent_shell_builder_builds_no_solvent_output_pdb(tmp_path):
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_no_solvent_shell_pdb(tmp_path)

    analysis_result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )
    output_path = tmp_path / "solvent_shell_no_solvent_output.pdb"
    build_result = build_solvent_shell_output(
        input_path,
        "water_test",
        output_path=output_path,
        director_atom_name="O1",
        minimum_solvent_atom_separation_a=1.2,
        solute_distance_cutoffs_a={"Pb": 2.6},
        coordinating_center_elements=("Pb",),
        target_average_coordination_numbers={"Pb": 1.0},
        reference_library_dir=reference_library_dir,
        analysis_result=analysis_result,
    )

    output_structure = PDBStructure.from_file(output_path)
    solvent_atoms = [
        atom for atom in output_structure.atoms if atom.residue_name == "HOH"
    ]
    oxygen_atoms = [atom for atom in solvent_atoms if atom.element == "O"]

    assert build_result.build_mode == "no_solvent_shell_build"
    assert build_result.solvent_molecules_added == 1
    assert build_result.solvent_atoms_added == 3
    assert build_result.partial_candidates_completed == 0
    assert len(output_structure.atoms) == 4
    assert len(solvent_atoms) == 3
    assert len(oxygen_atoms) == 1
    assert np.allclose(
        oxygen_atoms[0].coordinates,
        np.array([2.6, 0.0, 0.0], dtype=float),
    )
    assert build_result.target_average_coordination_numbers == {"Pb": 1.0}
    assert build_result.achieved_average_coordination_numbers == {"Pb": 1.0}


def test_solvent_shell_builder_builds_no_solvent_output_using_average_coordination(
    tmp_path,
):
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_no_solvent_mixed_shell_pdb(tmp_path)

    analysis_result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )
    output_path = tmp_path / "solvent_shell_coordination_average_output.pdb"
    build_result = build_solvent_shell_output(
        input_path,
        "water_test",
        output_path=output_path,
        director_atom_name="O1",
        minimum_solvent_atom_separation_a=1.2,
        solute_distance_cutoffs_a={"I": 3.0, "Pb": 2.6},
        coordinating_center_elements=("Pb",),
        target_average_coordination_numbers={"Pb": 2.0},
        reference_library_dir=reference_library_dir,
        analysis_result=analysis_result,
    )

    assert build_result.build_mode == "no_solvent_shell_build"
    assert 2 <= build_result.solvent_molecules_added <= 4
    assert build_result.target_average_coordination_numbers == {"Pb": 2.0}
    assert build_result.achieved_average_coordination_numbers is not None
    assert build_result.achieved_average_coordination_numbers["Pb"] >= 2.0


def test_solvent_shell_builder_prefers_octahedral_vacancies_for_center_candidates():
    center_atom = PDBAtom(
        atom_id=1,
        atom_name="PB1",
        residue_name="PBI",
        residue_number=1,
        coordinates=np.array([0.0, 0.0, 0.0], dtype=float),
        element="Pb",
    )
    solute_atoms = [
        center_atom,
        PDBAtom(
            atom_id=2,
            atom_name="I1",
            residue_name="PBI",
            residue_number=1,
            coordinates=np.array([2.6, 0.0, 0.0], dtype=float),
            element="I",
        ),
        PDBAtom(
            atom_id=3,
            atom_name="I2",
            residue_name="PBI",
            residue_number=1,
            coordinates=np.array([-2.6, 0.0, 0.0], dtype=float),
            element="I",
        ),
    ]

    candidate_positions = (
        solvent_shell_builder_module._coordination_candidate_positions(
            center_atoms=[center_atom],
            solute_atoms=solute_atoms,
            existing_anchor_positions=[],
            solute_distance_cutoffs_a={"Pb": 2.6},
        )
    )

    assert len(candidate_positions) >= 4
    leading_positions = candidate_positions[:4]
    assert all(abs(float(position[0])) < 0.2 for position in leading_positions)
    assert any(abs(float(position[1])) > 2.0 for position in leading_positions)
    assert any(abs(float(position[2])) > 2.0 for position in leading_positions)


def test_solvent_shell_builder_builds_partial_xyz_output_pdb(tmp_path):
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_partial_solvent_shell_xyz(tmp_path)

    analysis_result = analyze_solvent_shell(
        input_path,
        "water_test",
        reference_library_dir=reference_library_dir,
    )
    output_path = tmp_path / "solvent_shell_partial_xyz_output.pdb"
    build_result = build_solvent_shell_output(
        input_path,
        "water_test",
        output_path=output_path,
        director_atom_name="O1",
        minimum_solvent_atom_separation_a=1.2,
        solute_distance_cutoffs_a={"I": 3.0, "Pb": 2.6},
        reference_library_dir=reference_library_dir,
        analysis_result=analysis_result,
    )

    output_structure = PDBStructure.from_file(output_path)
    solvent_atoms = [
        atom for atom in output_structure.atoms if atom.residue_name == "HOH"
    ]
    oxygen_atoms = [atom for atom in solvent_atoms if atom.element == "O"]

    assert build_result.build_mode == "partial_solvent_completion"
    assert build_result.solvent_molecules_added == 1
    assert build_result.solvent_atoms_added == 2
    assert build_result.partial_candidates_completed == 1
    assert build_result.replaced_source_atom_count == 1
    assert len(output_structure.atoms) == 5
    assert len(solvent_atoms) == 3
    assert len(oxygen_atoms) == 1
    assert np.allclose(
        oxygen_atoms[0].coordinates,
        np.array([0.0, 2.0, 0.0], dtype=float),
    )


def test_solvent_shell_builder_window_reports_residue_breakdown(tmp_path):
    qapp()
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    central_layout = window.centralWidget().layout()
    assert central_layout.itemAt(1).widget() is window._pane_splitter
    assert window._pane_splitter.count() == 2
    assert window._pane_splitter.widget(0) is window._left_scroll_area
    assert window._pane_splitter.widget(1) is window._right_scroll_area
    assert window._left_scroll_area.widget() is window._left_panel
    assert window._right_scroll_area.widget() is window._right_panel
    assert window.cluster_status_group.parentWidget() is window._left_panel
    assert "Residue HOH" in window.reference_details_box.toPlainText()
    assert window.structure_viewer.current_structure is not None
    assert window.structure_viewer.current_structure.file_path == input_path
    assert "complete solvent molecules detected" in (
        window.cluster_status_headline_label.text().lower()
    )
    assert (
        "No solvent molecules: no" in window.cluster_status_stats_label.text()
    )
    assert "Partial solvent molecules: no" in (
        window.cluster_status_stats_label.text()
    )
    assert "Complete solvent molecules: yes" in (
        window.cluster_status_stats_label.text()
    )
    assert (
        "Complete solvent count: 2" in window.cluster_status_stats_label.text()
    )
    assert "Solvent molecules detected: 2" in window.summary_box.toPlainText()
    assert "PDB residue matches:" in window.summary_box.toPlainText()
    assert window.residue_table.rowCount() == 2
    table_rows = {
        window.residue_table.item(row, 0).text(): {
            "molecules": window.residue_table.item(row, 1).text(),
            "numbers": window.residue_table.item(row, 2).text(),
        }
        for row in range(window.residue_table.rowCount())
    }
    assert table_rows == {
        "ALT": {"molecules": "1", "numbers": "3"},
        "HOH": {"molecules": "1", "numbers": "2"},
    }
    assert "matched the selected solvent geometry" in (
        window.residue_status_label.text().lower()
    )
    window.close()


def test_solvent_shell_builder_window_reports_incomplete_residue_mismatch(
    tmp_path,
):
    qapp()
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_incomplete_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    assert "complete and partial solvent molecules detected" in (
        window.cluster_status_headline_label.text().lower()
    )
    assert (
        "No solvent molecules: no" in window.cluster_status_stats_label.text()
    )
    assert "Partial solvent molecules: yes" in (
        window.cluster_status_stats_label.text()
    )
    assert "Complete solvent molecules: yes" in (
        window.cluster_status_stats_label.text()
    )
    assert (
        "Complete solvent count: 1" in window.cluster_status_stats_label.text()
    )
    assert "Partial solvent residue count: 1" in (
        window.cluster_status_stats_label.text()
    )
    assert (
        "Residue mismatches preserved: 1" in window.summary_box.toPlainText()
    )
    assert window.mismatch_table.rowCount() == 1
    assert window.mismatch_table.item(0, 0).text() == "HOH"
    assert window.mismatch_table.item(0, 1).text() == "3"
    assert window.mismatch_table.item(0, 2).text() == "2"
    assert window.mismatch_table.item(0, 3).text() == "2/3"
    assert window.mismatch_table.item(0, 4).text() == "H2"
    assert window.mismatch_table.item(0, 5).text() == "none"
    assert (
        "missing-atom details" in window.mismatch_status_label.text().lower()
    )
    window.close()


def test_solvent_shell_builder_window_reports_partial_xyz_candidates(tmp_path):
    qapp()
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_partial_solvent_shell_xyz(tmp_path)
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    assert "partial solvent molecules detected" in (
        window.cluster_status_headline_label.text().lower()
    )
    assert "Partial solvent molecules: yes" in (
        window.cluster_status_stats_label.text()
    )
    assert "Partial solvent candidate count: 1" in (
        window.cluster_status_stats_label.text()
    )
    assert window.mismatch_table.rowCount() == 1
    assert window.mismatch_table.item(0, 0).text() == "HOH"
    assert window.mismatch_table.item(0, 3).text() == "1/3"
    assert window.mismatch_table.item(0, 4).text() == "H1, H2"
    assert "xyz atom sets" in window.mismatch_status_label.text().lower()
    window.close()


def test_solvent_shell_builder_window_populates_build_controls_and_writes_output(
    tmp_path,
):
    qapp()
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_partial_solvent_shell_xyz(tmp_path)
    output_path = tmp_path / "beta_builder_output.pdb"
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    assert [
        window.director_atom_combo.itemText(index) for index in range(3)
    ] == [
        "O1",
        "H1",
        "H2",
    ]
    assert window.director_atom_combo.currentData() == "O1"
    assert window.build_output_button.isEnabled() is True
    assert window.solute_cutoff_table.rowCount() == 2
    assert window.solute_cutoff_table.item(0, 0).text() == "I"
    assert window.solute_cutoff_table.item(0, 1).text() == "1"
    assert window.solute_cutoff_table.item(1, 0).text() == "Pb"
    assert window.solute_cutoff_table.item(1, 1).text() == "1"

    window.output_path_edit.setText(str(output_path))
    window._build_solvated_output()

    assert output_path.is_file()
    assert (
        "Generated solvent shell output:" in window.summary_box.toPlainText()
    )
    assert "Build mode: partial_solvent_completion" in (
        window.summary_box.toPlainText()
    )
    assert (
        "Previewing generated output" in window.visualizer_status_label.text()
    )
    window.close()


def test_solvent_shell_builder_window_requires_coordination_targets_for_no_solvent_build(
    tmp_path,
):
    qapp()
    reference_library_dir, _reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_no_solvent_mixed_shell_pdb(tmp_path)
    output_path = tmp_path / "beta_coordination_target_output.pdb"
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    assert window.solute_cutoff_table.rowCount() == 2
    assert window.build_output_button.isEnabled() is False
    assert (
        "needs coordination targets"
        in window.build_status_label.text().lower()
    )
    assert window.solute_cutoff_table.item(0, 0).text() == "I"
    assert window.solute_cutoff_table.item(1, 0).text() == "Pb"

    pb_center_item = window.solute_cutoff_table.item(1, 2)
    assert pb_center_item is not None
    pb_center_item.setCheckState(Qt.CheckState.Checked)
    pb_coordination_spin = window.solute_cutoff_table.cellWidget(1, 3)
    assert isinstance(pb_coordination_spin, QDoubleSpinBox)
    pb_coordination_spin.setValue(2.0)

    assert window.build_output_button.isEnabled() is True
    window.output_path_edit.setText(str(output_path))
    window._build_solvated_output()

    assert output_path.is_file()
    assert (
        "Target average coordination: Pb:2" in window.summary_box.toPlainText()
    )
    assert (
        "Achieved average coordination: Pb:"
        in window.summary_box.toPlainText()
    )
    window.close()


def test_solvent_shell_builder_window_uses_selected_match_tolerance(tmp_path):
    qapp()
    reference_library_dir, reference_path = (
        _build_test_solvent_reference_library(tmp_path)
    )
    input_path = _write_test_solvent_shell_pdb(
        tmp_path,
        reference_path=reference_path,
    )
    window = SolventShellBuilderMainWindow(
        initial_input_path=input_path,
        reference_library_dir=reference_library_dir,
    )

    assert window.reference_match_tolerance_spin.value() == pytest.approx(
        DEFAULT_REFERENCE_MATCH_TOLERANCE_A
    )
    window.reference_match_tolerance_spin.setValue(0.5)
    preset_index = window.reference_preset_combo.findData("water_test")
    assert preset_index >= 0
    window.reference_preset_combo.setCurrentIndex(preset_index)
    window._analyze_input_structure()

    assert (
        "Reference match tolerance: 0.5 A" in window.summary_box.toPlainText()
    )
    window.close()


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


def test_rmcsetup_packmol_single_atom_uses_periodic_table_picker(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    monkeypatch.setattr(
        fullrmc_ui_module.PeriodicTableElementDialog,
        "get_element_symbol",
        lambda **_kwargs: "Cs",
    )
    monkeypatch.setattr(
        fullrmc_ui_module.QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: ("solute", True),
    )
    monkeypatch.setattr(
        fullrmc_ui_module.QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("CES", True),
    )

    window._add_packmol_supplemental_atom_component()

    components = window._current_packmol_supplemental_components()
    assert len(components) == 1
    assert components[0].element == "Cs"
    assert components[0].residue_name == "CES"
    assert window.packmol_supplemental_table.item(0, 2).text() == "Cs"
    window.close()


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
    _configure_integrated_rmcsetup_solvent_panel(window)
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

    assert window.packmol_tolerance_spin.value() == pytest.approx(2.0)

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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window.packmol_tolerance_spin.setValue(2.2)

    window._build_packmol_setup()

    assert window._project_source_state is not None
    assert window._project_source_state.packmol_setup is not None
    assert "Packmol tolerance: 2.200 A" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert "tolerance 2.200" in Path(
        window._project_source_state.packmol_setup.packmol_input_path
    ).read_text(encoding="utf-8")
    assert "Representative PDBs copied:" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert window.open_packmol_setup_folder_button.isEnabled() is True
    assert "Built Packmol setup inputs and audit report." in (
        window.run_log_box.toPlainText()
    )
    assert "packmol_combined.inp" in window.output_summary_box.toPlainText()

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert reloaded.packmol_tolerance_spin.value() == pytest.approx(2.2)
    assert "Packmol tolerance: 2.200 A" in (
        reloaded.packmol_build_summary_box.toPlainText()
    )
    assert "Representative PDBs copied:" in (
        reloaded.packmol_build_summary_box.toPlainText()
    )
    assert reloaded.open_packmol_setup_folder_button.isEnabled() is True
    assert "packmol_audit.md" in reloaded.output_summary_box.toPlainText()


def test_rmcsetup_ui_can_open_packmol_setup_folder(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    opened_paths: list[Path] = []
    monkeypatch.setattr(
        window,
        "_open_path_in_file_manager",
        lambda path: opened_paths.append(path),
    )

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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window._build_packmol_setup()

    assert window.open_packmol_setup_folder_button.isEnabled() is True

    window.open_packmol_setup_folder_button.click()

    assert opened_paths == [
        window._project_source_state.rmcsetup_paths.packmol_inputs_dir.resolve()
    ]
    assert "Opened Packmol setup folder in Finder/file manager:" in (
        window.run_log_box.toPlainText()
    )
    assert window.statusBar().currentMessage() == (
        "Opened Packmol setup folder: packmol_inputs"
    )


def test_rmcsetup_ui_syncs_packmol_setup_to_linked_docker_container(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    fake_client = _FakePackmolDockerClient()
    monkeypatch.setattr(
        RMCSetupMainWindow,
        "_create_packmol_docker_client",
        lambda self: fake_client,
    )
    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert window._project_source_state is not None

    linked = PackmolDockerLink(
        display_name="Mounted Packmol",
        container_name="packmol-dev",
        container_project_root="/packmol_input_files/project_alpha",
        packmol_command="packmol",
        shell_command="sh",
        packmol_version="Packmol version 20.14.4",
        last_verified_at="2026-04-17T12:30:00",
        container_id="sha256:sync",
        image_name="packmol:test-image",
        packmol_command_path="/usr/local/bin/packmol",
    )
    window._save_packmol_docker_link(linked)
    window.packmol_docker_summary_box.setPlainText(
        window._packmol_docker_summary_text()
    )

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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()

    window._build_packmol_setup()

    assert fake_client.synced_calls
    assert fake_client.synced_calls[0][0] == "packmol-dev"
    assert window._project_source_state.packmol_docker_link is not None
    assert (
        window._project_source_state.packmol_docker_link.last_sync_status
        == ("success")
    )
    assert "/packmol_input_files/project_alpha/rmcsetup/packmol_inputs" in (
        window.packmol_build_summary_box.toPlainText()
    )
    assert "Last sync status: success" in (
        window.packmol_docker_summary_box.toPlainText()
    )

    reloaded = RMCSetupMainWindow(initial_project_dir=project_dir)
    assert "Last sync status: success" in (
        reloaded.packmol_docker_summary_box.toPlainText()
    )


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
    _configure_integrated_rmcsetup_solvent_panel(window)
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
    assert window.open_constraints_folder_button.isEnabled() is True
    assert window.preview_constraints_button.isEnabled() is True
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
    assert reloaded.open_constraints_folder_button.isEnabled() is True
    assert reloaded.preview_constraints_button.isEnabled() is True


def test_rmcsetup_ui_can_open_constraints_folder(
    tmp_path,
    monkeypatch,
):
    qapp()
    project_dir, _paths = _build_sample_saxs_project(tmp_path)
    window = RMCSetupMainWindow(initial_project_dir=project_dir)
    opened_paths: list[Path] = []
    monkeypatch.setattr(
        window,
        "_open_path_in_file_manager",
        lambda path: opened_paths.append(path),
    )

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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window._build_packmol_setup()
    window._generate_constraints()

    assert window.open_constraints_folder_button.isEnabled() is True

    window.open_constraints_folder_button.click()

    assert window._project_source_state is not None
    assert window._project_source_state.constraint_generation is not None
    assert opened_paths == [
        Path(
            window._project_source_state.constraint_generation.merged_constraints_path
        ).resolve()
    ]
    assert "Opened constraints file location in Finder/file manager:" in (
        window.run_log_box.toPlainText()
    )
    assert window.statusBar().currentMessage() == (
        "Opened constraints file location: merged_fullrmc_constraints.py"
    )


def test_rmcsetup_ui_can_preview_merged_constraints(
    tmp_path,
):
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
    _configure_integrated_rmcsetup_solvent_panel(window)
    window._build_representative_solvent_outputs()
    window.packmol_box_side_spin.setValue(80.0)
    window._compute_packmol_plan()
    window._build_packmol_setup()
    window._generate_constraints()

    assert window.preview_constraints_button.isEnabled() is True

    window.preview_constraints_button.click()

    assert window._constraints_preview_window is not None
    assert "BOND_ANGLE_CONSTRAINTS = {" in (
        window._constraints_preview_window.text_box.toPlainText()
    )
    assert "BOND_LENGTH_CONSTRAINTS = {" in (
        window._constraints_preview_window.text_box.toPlainText()
    )
    assert "Opened merged constraints preview:" in (
        window.run_log_box.toPlainText()
    )
    window._constraints_preview_window.close()


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
    _configure_integrated_rmcsetup_solvent_panel(window)
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
