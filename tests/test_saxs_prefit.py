from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import saxshell.saxs.prefit.cluster_geometry as cluster_geometry_module
import saxshell.saxs.prefit.workflow as prefit_workflow_module
import saxshell.saxs.project_manager.project as project_module
from saxshell.clusterdynamicsml.workflow import (
    ClusterDynamicsMLTrainingObservation,
    PredictedClusterCandidate,
)
from saxshell.fullrmc.solution_properties import SolutionPropertiesSettings
from saxshell.saxs._model_templates import load_template_module
from saxshell.saxs.debye.profiles import AveragedComponent, ClusterBin
from saxshell.saxs.prefit import (
    SAXSPrefitWorkflow,
    compute_cluster_geometry_metadata,
    load_cluster_geometry_metadata,
    resolve_prefit_parameter_entries,
)
from saxshell.saxs.prefit.workflow import constrained_prefit_residuals
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
    project_artifact_paths,
)
from saxshell.saxs.solute_volume_fraction import (
    SoluteVolumeFractionSettings,
    calculate_solute_volume_fraction_estimate,
)
from saxshell.saxs.solution_scattering_estimator import (
    BeamGeometrySettings,
    SolutionScatteringEstimatorSettings,
    calculate_solution_scattering_estimate,
)

POLY_LMA_HS_TEMPLATE = "template_pydream_poly_lma_hs"
POLY_LMA_HS_MIX_TEMPLATE = "template_pydream_poly_lma_hs_mix_approx"


def _write_component_file(path, q_values, intensities):
    data = np.column_stack(
        [
            q_values,
            intensities,
            np.zeros_like(q_values),
            np.zeros_like(q_values),
        ]
    )
    np.savetxt(
        path,
        data,
        header="# Number of files: 1\n# Columns: q, S(q)_avg, S(q)_std, S(q)_se",
        comments="",
    )


def _build_minimal_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "saxs_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    template_name = "template_pd_likelihood_monosq_decoupled"
    template_module = load_template_module(template_name)
    experimental = template_module.lmfit_model_profile(
        q_values,
        np.zeros_like(q_values),
        [component],
        w0=0.6,
        solv_w=0.0,
        offset=0.05,
        eff_r=9.0,
        vol_frac=0.0,
        scale=5e-4,
    )
    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(
        experimental_path,
        np.column_stack([q_values, experimental]),
    )
    _write_component_file(
        paths.scattering_components_dir / "A_no_motif.txt",
        q_values,
        component,
    )

    md_prior_payload = {
        "origin": "clusters",
        "total_files": 1,
        "structures": {
            "A": {
                "no_motif": {
                    "count": 1,
                    "weight": 0.6,
                    "representative": "frame_0001.xyz",
                    "profile_file": "A_no_motif.txt",
                }
            }
        },
    }
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(md_prior_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map.json").write_text(
        json.dumps(
            {"saxs_map": {"A": {"no_motif": "A_no_motif.txt"}}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    settings.experimental_data_path = str(experimental_path)
    settings.copied_experimental_data_file = str(experimental_path)
    settings.selected_model_template = template_name
    manager.save_project(settings)
    return project_dir, paths


def _write_predicted_structure_artifacts(
    paths,
    *,
    observed_weight: float = 0.75,
    predicted_weight: float = 0.25,
):
    q_values = np.linspace(0.05, 0.3, 8)
    observed = np.linspace(10.0, 17.0, 8)
    predicted = np.linspace(4.0, 11.0, 8)
    _write_component_file(
        paths.predicted_scattering_components_dir / "A_no_motif.txt",
        q_values,
        observed,
    )
    _write_component_file(
        paths.predicted_scattering_components_dir / "A2_predicted_rank01.txt",
        q_values,
        predicted,
    )
    dataset_dir = (
        paths.exported_data_dir
        / "clusterdynamicsml"
        / "saved_results"
        / "20260330_120000_demo"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / "20260330_120000_demo_clusterdynamicsml.json"
    dataset_file.write_text(
        json.dumps({"predictions": [{"label": "A2", "rank": 1}]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (
        paths.project_dir / "md_prior_weights_predicted_structures.json"
    ).write_text(
        json.dumps(
            {
                "origin": "clusters_predicted_structures",
                "total_files": 1,
                "available_elements": ["A"],
                "value_kind": "normalized_weight",
                "includes_predicted_structures": True,
                "prediction_dataset_file": str(dataset_file),
                "structures": {
                    "A": {
                        "no_motif": {
                            "count": 1,
                            "weight": observed_weight,
                            "normalized_weight": observed_weight,
                            "observed_only_weight": 1.0,
                            "representative": "frame_0001.xyz",
                            "profile_file": "A_no_motif.txt",
                            "source_kind": "cluster_dir",
                        }
                    },
                    "A2": {
                        "predicted_rank01": {
                            "count": 1,
                            "weight": predicted_weight,
                            "normalized_weight": predicted_weight,
                            "observed_only_weight": 0.0,
                            "representative": "02_rank01_A2.xyz",
                            "profile_file": "A2_predicted_rank01.txt",
                            "source_kind": "predicted_structure",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map_predicted_structures.json").write_text(
        json.dumps(
            {
                "saxs_map": {
                    "A": {"no_motif": "A_no_motif.txt"},
                    "A2": {"predicted_rank01": "A2_predicted_rank01.txt"},
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_file


def _build_poly_lma_geometry_project(
    tmp_path,
    *,
    template_name: str = POLY_LMA_HS_MIX_TEMPLATE,
    single_atom: bool = False,
):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "poly_lma_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)

    clusters_dir = paths.project_dir / "clusters"
    structure_dir = clusters_dir / "A"
    structure_dir.mkdir(parents=True, exist_ok=True)
    if single_atom:
        (structure_dir / "frame_0001.xyz").write_text(
            "1\nframe 1\nI 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        cluster_count = 1
    else:
        (structure_dir / "frame_0001.xyz").write_text(
            "4\nframe 1\nPb 0.0 0.0 0.0\nI 2.0 0.0 0.0\n"
            "I 0.0 2.0 0.0\nI 0.0 0.0 2.0\n",
            encoding="utf-8",
        )
        (structure_dir / "frame_0002.xyz").write_text(
            "4\nframe 2\nPb 0.0 0.0 0.0\nI 2.2 0.0 0.0\n"
            "I 0.0 1.8 0.0\nI 0.0 0.0 2.1\n",
            encoding="utf-8",
        )
        cluster_count = 2
    geometry_table = compute_cluster_geometry_metadata(
        clusters_dir,
        template_name=template_name,
    )
    effective_radius = geometry_table.rows[0].effective_radius

    template_module = load_template_module(template_name)
    experimental = template_module.lmfit_model_profile(
        q_values,
        np.zeros_like(q_values),
        [component],
        np.asarray([effective_radius], dtype=float),
        w0=1.0,
        phi_solute=0.02,
        phi_int=0.02,
        solvent_scale=1.0,
        scale=1.0,
        offset=0.0,
        log_sigma=-9.21,
    )
    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(
        experimental_path,
        np.column_stack([q_values, experimental]),
    )
    _write_component_file(
        paths.scattering_components_dir / "A_no_motif.txt",
        q_values,
        component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": cluster_count,
                "structures": {
                    "A": {
                        "no_motif": {
                            "count": cluster_count,
                            "weight": 1.0,
                            "representative": "frame_0001.xyz",
                            "profile_file": "A_no_motif.txt",
                        }
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map.json").write_text(
        json.dumps(
            {"saxs_map": {"A": {"no_motif": "A_no_motif.txt"}}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    settings.experimental_data_path = str(experimental_path)
    settings.copied_experimental_data_file = str(experimental_path)
    settings.selected_model_template = template_name
    settings.clusters_dir = str(clusters_dir)
    manager.save_project(settings)
    return project_dir, paths, effective_radius


def test_saxs_prefit_workflow_evaluates_and_autosaves_when_enabled(tmp_path):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.set_autosave(True)

    evaluation = workflow.evaluate()

    assert evaluation.q_values.shape == evaluation.model_intensities.shape
    assert len(workflow.parameter_entries) == 6

    result = workflow.run_fit(method="leastsq", max_nfev=500)

    assert result.nfev >= 0
    assert (paths.prefit_dir / "pd_prefit_params.json").is_file()
    assert (paths.prefit_dir / "prefit_state.json").is_file()
    assert (paths.prefit_dir / "latest_prefit_curve.txt").is_file()
    saved_states = workflow.list_saved_states()
    assert saved_states
    snapshot_dir = paths.prefit_dir / saved_states[0]
    assert (snapshot_dir / "prefit_state.json").is_file()
    assert (snapshot_dir / "pd_prefit_params.json").is_file()
    assert (snapshot_dir / "prefit_curve.txt").is_file()
    assert (snapshot_dir / "prefit_report.txt").is_file()


def test_saxs_prefit_workflow_uses_grid_seed_for_small_varying_sets(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    for entry in entries:
        entry.vary = False

    scale_entry = next(entry for entry in entries if entry.name == "scale")
    scale_entry.vary = True
    scale_entry.value = 1e-5
    scale_entry.minimum = 1e-5
    scale_entry.maximum = 1e-3

    seeded_values: dict[str, float] = {}

    class FakeResult:
        def __init__(self, params):
            self.params = params
            self.nfev = 0

    def fake_minimize(objective, params, method, max_nfev):
        del objective, method, max_nfev
        seeded_values["scale"] = float(params["scale"].value)
        return FakeResult(params)

    monkeypatch.setattr(prefit_workflow_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        prefit_workflow_module,
        "fit_report",
        lambda _result: "fake lmfit report",
    )

    result = workflow.run_fit(entries, method="leastsq", max_nfev=50)

    assert seeded_values["scale"] > 1e-5
    assert result.optimization_strategy.startswith("coarse-to-fine grid")
    assert result.grid_evaluations > 0
    assert "Coarse-to-fine grid sweep" in result.fit_report


def test_saxs_prefit_workflow_skips_grid_seed_for_large_varying_sets(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    varying_names = [entry.name for entry in entries[:4]]

    for entry in entries:
        entry.vary = entry.name in varying_names

    seeded_values: dict[str, float] = {}

    class FakeResult:
        def __init__(self, params):
            self.params = params
            self.nfev = 0

    def fake_minimize(objective, params, method, max_nfev):
        del objective, method, max_nfev
        seeded_values.update(
            {name: float(params[name].value) for name in varying_names}
        )
        return FakeResult(params)

    monkeypatch.setattr(prefit_workflow_module, "minimize", fake_minimize)
    monkeypatch.setattr(
        prefit_workflow_module,
        "fit_report",
        lambda _result: "fake lmfit report",
    )

    result = workflow.run_fit(entries, method="leastsq", max_nfev=50)

    assert seeded_values == {
        name: pytest.approx(
            next(entry.value for entry in entries if entry.name == name)
        )
        for name in varying_names
    }
    assert result.optimization_strategy == "lmfit leastsq"
    assert result.grid_evaluations == 0
    assert "Coarse-to-fine grid sweep" not in result.fit_report


def test_saxs_prefit_workflow_uses_predicted_structure_artifacts_when_enabled(
    tmp_path,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    _write_predicted_structure_artifacts(paths)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.use_predicted_structure_weights = True
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    component_lookup = {
        (component.structure, component.motif): component
        for component in workflow.components
    }

    assert workflow.component_dir == paths.predicted_scattering_components_dir
    assert (
        workflow.component_map_path.name
        == "md_saxs_map_predicted_structures.json"
    )
    assert (
        workflow.prior_weights_path.name
        == "md_prior_weights_predicted_structures.json"
    )
    assert set(component_lookup) == {
        ("A", "no_motif"),
        ("A2", "predicted_rank01"),
    }
    assert component_lookup[("A", "no_motif")].weight_value == pytest.approx(
        0.75
    )
    assert component_lookup[
        ("A2", "predicted_rank01")
    ].weight_value == pytest.approx(0.25)


def test_project_manager_builds_predicted_structure_components_and_prior_weights(
    tmp_path,
    monkeypatch,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    clusters_dir = paths.project_dir / "clusters"
    structure_dir = clusters_dir / "A"
    structure_dir.mkdir(parents=True, exist_ok=True)
    observed_structure = structure_dir / "frame_0001.xyz"
    observed_structure.write_text(
        "1\nframe 1\nA 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    settings.clusters_dir = str(clusters_dir)
    settings.use_predicted_structure_weights = True
    manager.save_project(settings)

    observed_component = np.linspace(10.0, 17.0, 8)
    predicted_component = np.linspace(4.0, 11.0, 8)

    def fake_build_profiles(
        builder,
        *,
        cluster_bins,
        progress_callback=None,
        progress_total=None,
        **kwargs,
    ):
        del progress_callback, progress_total, kwargs
        output_path = builder.output_dir / "A_no_motif.txt"
        _write_component_file(
            output_path, builder.q_values, observed_component
        )
        cluster_bin = cluster_bins[0]
        return [
            AveragedComponent(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                file_count=len(cluster_bin.files),
                representative=cluster_bin.representative,
                source_dir=cluster_bin.source_dir,
                q_values=np.asarray(builder.q_values, dtype=float),
                mean_intensity=np.asarray(observed_component, dtype=float),
                std_intensity=np.zeros_like(builder.q_values, dtype=float),
                se_intensity=np.zeros_like(builder.q_values, dtype=float),
                output_path=output_path,
            )
        ]

    monkeypatch.setattr(
        project_module.DebyeProfileBuilder,
        "build_profiles",
        fake_build_profiles,
    )
    monkeypatch.setattr(
        project_module,
        "compute_debye_intensity",
        lambda coordinates, elements, q_values: np.asarray(
            predicted_component,
            dtype=float,
        ),
    )

    dataset_dir = (
        paths.exported_data_dir
        / "clusterdynamicsml"
        / "saved_results"
        / "20260330_120000_demo"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / "20260330_120000_demo_clusterdynamicsml.json"
    dataset_file.write_text("{}\n", encoding="utf-8")
    predicted_dir = dataset_dir / (f"{dataset_file.stem}_predicted_structures")
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure = predicted_dir / "02_rank01_A2.xyz"
    predicted_structure.write_text(
        "2\npredicted\nA 0.0 0.0 0.0\nA 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    loaded_dataset = SimpleNamespace(
        dataset_file=dataset_file,
        result=SimpleNamespace(
            training_observations=(
                ClusterDynamicsMLTrainingObservation(
                    label="A",
                    node_count=1,
                    cluster_size=1,
                    element_counts={"A": 1},
                    file_count=1,
                    representative_path=observed_structure,
                    structure_dir=structure_dir,
                    motifs=("no_motif",),
                    mean_atom_count=1.0,
                    mean_radius_of_gyration=1.0,
                    mean_max_radius=1.0,
                    mean_semiaxis_a=1.0,
                    mean_semiaxis_b=1.0,
                    mean_semiaxis_c=1.0,
                    total_observations=1,
                    occupied_frames=1,
                    mean_count_per_frame=1.0,
                    occupancy_fraction=1.0,
                    association_events=0,
                    dissociation_events=0,
                    association_rate_per_ps=0.0,
                    dissociation_rate_per_ps=0.0,
                    completed_lifetime_count=1,
                    window_truncated_lifetime_count=0,
                    mean_lifetime_fs=10.0,
                    std_lifetime_fs=0.0,
                ),
            ),
            predictions=(
                PredictedClusterCandidate(
                    target_node_count=2,
                    rank=1,
                    label="A2",
                    element_counts={"A": 2},
                    predicted_mean_count_per_frame=0.2,
                    predicted_occupancy_fraction=1.0,
                    predicted_mean_lifetime_fs=20.0,
                    predicted_association_rate_per_ps=0.0,
                    predicted_dissociation_rate_per_ps=0.0,
                    predicted_mean_radius_of_gyration=1.0,
                    predicted_mean_max_radius=1.0,
                    predicted_mean_semiaxis_a=1.0,
                    predicted_mean_semiaxis_b=1.0,
                    predicted_mean_semiaxis_c=1.0,
                    predicted_population_share=100.0,
                    predicted_stability_score=1.0,
                    source_label="A",
                    notes="",
                    generated_elements=("A", "A"),
                    generated_coordinates=np.asarray(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        dtype=float,
                    ),
                ),
            ),
            dynamics_result=SimpleNamespace(
                preview=SimpleNamespace(frame_timestep_fs=1.0)
            ),
            saxs_comparison=None,
        ),
    )
    monkeypatch.setattr(
        manager,
        "_load_latest_predicted_structures_dataset",
        lambda project_dir: loaded_dataset,
    )

    component_result = manager.build_scattering_components(settings)
    prior_result = manager.generate_prior_weights(settings)
    artifact_paths = project_artifact_paths(
        settings,
        storage_mode="distribution",
    )
    predicted_map = json.loads(
        artifact_paths.component_map_file.read_text(encoding="utf-8")
    )
    prior_payload = json.loads(
        artifact_paths.prior_weights_file.read_text(encoding="utf-8")
    )
    predicted_entry = prior_payload["structures"]["A2"]["predicted_rank01"]

    assert component_result.used_predicted_structure_weights
    assert component_result.predicted_component_count == 1
    assert prior_result.used_predicted_structure_weights
    assert prior_result.predicted_component_count == 1
    assert (
        predicted_map["saxs_map"]["A2"]["predicted_rank01"]
        == "A2_predicted_rank01.txt"
    )
    assert prior_payload["value_kind"] == "normalized_weight"
    assert prior_payload["includes_predicted_structures"] is True
    assert predicted_entry["source_kind"] == "predicted_structure"
    assert predicted_entry["profile_file"] == "A2_predicted_rank01.txt"
    assert predicted_entry["normalized_weight"] > 0.0
    assert predicted_entry["observed_only_weight"] == pytest.approx(0.0)
    assert (artifact_paths.component_dir / "A2_predicted_rank01.txt").is_file()


def test_project_manager_reuses_observed_components_for_predicted_distribution(
    tmp_path,
    monkeypatch,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    clusters_dir = paths.project_dir / "clusters"
    structure_dir = clusters_dir / "A"
    structure_dir.mkdir(parents=True, exist_ok=True)
    observed_structure = structure_dir / "frame_0001.xyz"
    observed_structure.write_text(
        "1\nframe 1\nA 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    settings.clusters_dir = str(clusters_dir)
    manager.save_project(settings)

    observed_component = np.linspace(10.0, 17.0, 8)
    predicted_component = np.linspace(4.0, 11.0, 8)
    build_calls: list[str] = []

    def fake_build_profiles(
        builder,
        *,
        cluster_bins,
        progress_callback=None,
        progress_total=None,
        **kwargs,
    ):
        del progress_callback, progress_total, kwargs
        build_calls.append(str(builder.output_dir))
        output_path = builder.output_dir / "A_no_motif.txt"
        _write_component_file(
            output_path,
            builder.q_values,
            observed_component,
        )
        cluster_bin = cluster_bins[0]
        return [
            AveragedComponent(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                file_count=len(cluster_bin.files),
                representative=cluster_bin.representative,
                source_dir=cluster_bin.source_dir,
                q_values=np.asarray(builder.q_values, dtype=float),
                mean_intensity=np.asarray(observed_component, dtype=float),
                std_intensity=np.zeros_like(builder.q_values, dtype=float),
                se_intensity=np.zeros_like(builder.q_values, dtype=float),
                output_path=output_path,
            )
        ]

    monkeypatch.setattr(
        project_module.DebyeProfileBuilder,
        "build_profiles",
        fake_build_profiles,
    )
    monkeypatch.setattr(
        project_module,
        "compute_debye_intensity",
        lambda coordinates, elements, q_values: np.asarray(
            predicted_component,
            dtype=float,
        ),
    )

    dataset_dir = (
        paths.exported_data_dir
        / "clusterdynamicsml"
        / "saved_results"
        / "20260330_120000_demo"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = dataset_dir / "20260330_120000_demo_clusterdynamicsml.json"
    dataset_file.write_text("{}\n", encoding="utf-8")
    predicted_dir = dataset_dir / (f"{dataset_file.stem}_predicted_structures")
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure = predicted_dir / "02_rank01_A2.xyz"
    predicted_structure.write_text(
        "2\npredicted\nA 0.0 0.0 0.0\nA 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    loaded_dataset = SimpleNamespace(
        dataset_file=dataset_file,
        result=SimpleNamespace(
            training_observations=(
                ClusterDynamicsMLTrainingObservation(
                    label="A",
                    node_count=1,
                    cluster_size=1,
                    element_counts={"A": 1},
                    file_count=1,
                    representative_path=observed_structure,
                    structure_dir=structure_dir,
                    motifs=("no_motif",),
                    mean_atom_count=1.0,
                    mean_radius_of_gyration=1.0,
                    mean_max_radius=1.0,
                    mean_semiaxis_a=1.0,
                    mean_semiaxis_b=1.0,
                    mean_semiaxis_c=1.0,
                    total_observations=1,
                    occupied_frames=1,
                    mean_count_per_frame=1.0,
                    occupancy_fraction=1.0,
                    association_events=0,
                    dissociation_events=0,
                    association_rate_per_ps=0.0,
                    dissociation_rate_per_ps=0.0,
                    completed_lifetime_count=1,
                    window_truncated_lifetime_count=0,
                    mean_lifetime_fs=10.0,
                    std_lifetime_fs=0.0,
                ),
            ),
            predictions=(
                PredictedClusterCandidate(
                    target_node_count=2,
                    rank=1,
                    label="A2",
                    element_counts={"A": 2},
                    predicted_mean_count_per_frame=0.2,
                    predicted_occupancy_fraction=1.0,
                    predicted_mean_lifetime_fs=20.0,
                    predicted_association_rate_per_ps=0.0,
                    predicted_dissociation_rate_per_ps=0.0,
                    predicted_mean_radius_of_gyration=1.0,
                    predicted_mean_max_radius=1.0,
                    predicted_mean_semiaxis_a=1.0,
                    predicted_mean_semiaxis_b=1.0,
                    predicted_mean_semiaxis_c=1.0,
                    predicted_population_share=100.0,
                    predicted_stability_score=1.0,
                    source_label="A",
                    notes="",
                    generated_elements=("A", "A"),
                    generated_coordinates=np.asarray(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        dtype=float,
                    ),
                ),
            ),
            dynamics_result=SimpleNamespace(
                preview=SimpleNamespace(frame_timestep_fs=1.0)
            ),
            saxs_comparison=None,
        ),
    )
    monkeypatch.setattr(
        manager,
        "_load_latest_predicted_structures_dataset",
        lambda project_dir: loaded_dataset,
    )

    manager.build_scattering_components(settings)
    observed_artifacts = project_artifact_paths(
        settings,
        storage_mode="distribution",
    )

    predicted_settings = manager.load_project(project_dir)
    predicted_settings.clusters_dir = str(clusters_dir)
    predicted_settings.use_predicted_structure_weights = True
    manager.save_project(predicted_settings)
    component_result = manager.build_scattering_components(predicted_settings)
    predicted_artifacts = project_artifact_paths(
        predicted_settings,
        storage_mode="distribution",
    )

    assert build_calls == [str(observed_artifacts.component_dir)]
    assert (
        component_result.model_map_path
        == predicted_artifacts.component_map_file
    )
    assert (predicted_artifacts.component_dir / "A_no_motif.txt").is_file()
    assert (
        predicted_artifacts.component_dir / "A2_predicted_rank01.txt"
    ).is_file()
    np.testing.assert_allclose(
        np.loadtxt(
            predicted_artifacts.component_dir / "A_no_motif.txt",
            comments="#",
        )[:, 1],
        observed_component,
    )


def test_project_manager_saves_distribution_artifacts_and_metadata(
    tmp_path,
    monkeypatch,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    clusters_dir = paths.project_dir / "clusters"
    structure_dir = clusters_dir / "A"
    structure_dir.mkdir(parents=True, exist_ok=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "2\nframe 1\nA 0.0 0.0 0.0\nH 1.0 0.0 0.0\n",
        encoding="utf-8",
    )
    settings.clusters_dir = str(clusters_dir)
    settings.exclude_elements = ["H"]
    manager.save_project(settings)

    observed_component = np.linspace(10.0, 17.0, 8)
    captured_exclusions: list[set[str]] = []

    def fake_build_profiles(
        builder,
        *,
        cluster_bins,
        progress_callback=None,
        progress_total=None,
        **kwargs,
    ):
        del progress_callback, progress_total, kwargs
        captured_exclusions.append(set(builder.exclude_elements))
        output_path = builder.output_dir / "A_no_motif.txt"
        _write_component_file(
            output_path,
            builder.q_values,
            observed_component,
        )
        cluster_bin = cluster_bins[0]
        return [
            AveragedComponent(
                structure=cluster_bin.structure,
                motif=cluster_bin.motif,
                file_count=len(cluster_bin.files),
                representative=cluster_bin.representative,
                source_dir=cluster_bin.source_dir,
                q_values=np.asarray(builder.q_values, dtype=float),
                mean_intensity=np.asarray(observed_component, dtype=float),
                std_intensity=np.zeros_like(builder.q_values, dtype=float),
                se_intensity=np.zeros_like(builder.q_values, dtype=float),
                output_path=output_path,
            )
        ]

    monkeypatch.setattr(
        project_module.DebyeProfileBuilder,
        "build_profiles",
        fake_build_profiles,
    )

    component_result = manager.build_scattering_components(settings)
    prior_result = manager.generate_prior_weights(settings)
    artifact_paths = project_artifact_paths(settings)
    records = manager.list_saved_distributions(project_dir)
    metadata_payload = json.loads(
        artifact_paths.distribution_metadata_file.read_text(encoding="utf-8")
    )

    assert captured_exclusions == [{"H"}]
    assert artifact_paths.uses_distribution_storage
    assert component_result.model_map_path == artifact_paths.component_map_file
    assert (
        prior_result.md_prior_weights_path == artifact_paths.prior_weights_file
    )
    assert artifact_paths.component_map_file.is_file()
    assert artifact_paths.prior_weights_file.is_file()
    assert artifact_paths.distribution_metadata_file.is_file()
    assert metadata_payload["exclude_elements"] == ["H"]
    assert metadata_payload["component_artifacts_ready"] is True
    assert metadata_payload["prior_artifacts_ready"] is True
    assert len(records) == 1
    assert records[0].distribution_id == artifact_paths.distribution_id
    assert records[0].exclude_elements == ("H",)
    assert records[0].component_artifacts_ready
    assert records[0].prior_artifacts_ready


def test_prefit_cluster_geometry_includes_predicted_structures_when_enabled(
    tmp_path,
    monkeypatch,
):
    project_dir, paths, _ = _build_poly_lma_geometry_project(tmp_path)
    _write_predicted_structure_artifacts(paths)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.use_predicted_structure_weights = True
    manager.save_project(settings)

    predicted_dir = tmp_path / "predicted_geometry"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure = predicted_dir / "02_rank01_A2.xyz"
    predicted_structure.write_text(
        "2\npredicted\nA 0.0 0.0 0.0\nA 1.4 0.0 0.0\n",
        encoding="utf-8",
    )

    workflow = SAXSPrefitWorkflow(project_dir)
    monkeypatch.setattr(
        workflow.project_manager,
        "predicted_structure_cluster_bins",
        lambda project_dir, included_components=None: [
            ClusterBin(
                structure="A2",
                motif="predicted_rank01",
                source_dir=predicted_structure.parent,
                files=(predicted_structure,),
                representative=predicted_structure.name,
            )
        ],
    )

    table = workflow.compute_cluster_geometry_table()
    row_lookup = {(row.structure, row.motif): row for row in table.rows}
    geometry_entries = [
        entry
        for entry in workflow.parameter_entries
        if entry.category == "geometry"
    ]

    assert workflow.cluster_geometry_metadata_path == (
        paths.predicted_cluster_geometry_metadata_file
    )
    assert paths.predicted_cluster_geometry_metadata_file.is_file()
    assert set(row_lookup) == {
        ("A", "no_motif"),
        ("A2", "predicted_rank01"),
    }
    assert row_lookup[("A2", "predicted_rank01")].effective_radius > 0.0
    assert row_lookup[("A2", "predicted_rank01")].mapped_parameter == "w1"
    assert row_lookup[("A2", "predicted_rank01")].cluster_path == str(
        predicted_structure.parent.resolve()
    )
    assert any(
        entry.structure == "A2" and entry.motif == "predicted_rank01"
        for entry in geometry_entries
    )


def test_prefit_workflow_apply_project_settings_updates_geometry_mode(
    tmp_path,
    monkeypatch,
):
    project_dir, paths, _ = _build_poly_lma_geometry_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    observed_table = workflow.compute_cluster_geometry_table()

    assert workflow.cluster_geometry_metadata_path == (
        paths.cluster_geometry_metadata_file
    )
    assert {(row.structure, row.motif) for row in observed_table.rows} == {
        ("A", "no_motif")
    }

    _write_predicted_structure_artifacts(paths)
    predicted_dir = tmp_path / "predicted_geometry_toggle"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    predicted_structure = predicted_dir / "02_rank01_A2.xyz"
    predicted_structure.write_text(
        "2\npredicted\nA 0.0 0.0 0.0\nA 1.6 0.0 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        workflow.project_manager,
        "predicted_structure_cluster_bins",
        lambda project_dir, included_components=None: [
            ClusterBin(
                structure="A2",
                motif="predicted_rank01",
                source_dir=predicted_structure.parent,
                files=(predicted_structure,),
                representative=predicted_structure.name,
            )
        ],
    )
    manager = SAXSProjectManager()
    predicted_settings = manager.load_project(project_dir)
    predicted_settings.use_predicted_structure_weights = True
    manager.save_project(predicted_settings)

    workflow.apply_project_settings(predicted_settings)
    predicted_table = workflow.compute_cluster_geometry_table()

    assert workflow.cluster_geometry_metadata_path == (
        paths.predicted_cluster_geometry_metadata_file
    )
    assert {(row.structure, row.motif) for row in predicted_table.rows} == {
        ("A", "no_motif"),
        ("A2", "predicted_rank01"),
    }
    assert {
        (component.structure, component.motif)
        for component in workflow.components
    } == {("A", "no_motif"), ("A2", "predicted_rank01")}
    assert any(
        entry.structure == "A2"
        for entry in workflow.parameter_entries
        if entry.category in {"weight", "geometry"}
    )

    observed_settings = manager.load_project(project_dir)
    observed_settings.use_predicted_structure_weights = False
    manager.save_project(observed_settings)
    workflow.apply_project_settings(observed_settings)

    assert workflow.cluster_geometry_metadata_path == (
        paths.cluster_geometry_metadata_file
    )
    assert workflow.cluster_geometry_table is not None
    assert {
        (row.structure, row.motif)
        for row in workflow.cluster_geometry_table.rows
    } == {("A", "no_motif")}
    assert {
        (component.structure, component.motif)
        for component in workflow.components
    } == {("A", "no_motif")}
    assert all(entry.structure != "A2" for entry in workflow.parameter_entries)


def test_saxs_prefit_workflow_recommends_scale_from_model_difference(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    scale_entry = next(entry for entry in entries if entry.name == "scale")
    scale_entry.value = 1e-6
    scale_entry.minimum = 1e-7
    scale_entry.maximum = 1e-5

    recommendation = workflow.recommend_scale_settings(entries)

    assert recommendation.current_scale == pytest.approx(1e-6)
    assert recommendation.recommended_scale == pytest.approx(5e-4)
    assert recommendation.recommended_minimum == pytest.approx(5e-5)
    assert recommendation.recommended_maximum == pytest.approx(5e-3)
    assert recommendation.current_offset == pytest.approx(0.0)
    assert recommendation.recommended_offset == pytest.approx(0.05)
    assert recommendation.points_used == 8


def test_constrained_prefit_residuals_penalize_non_positive_model_values():
    experimental = np.asarray([1.0, 0.8, 0.6], dtype=float)
    model = np.asarray([1.1, -0.2, np.nan], dtype=float)

    residuals = constrained_prefit_residuals(experimental, model)

    assert residuals.shape == (6,)
    assert residuals[0] == pytest.approx(0.1)
    assert residuals[3] == pytest.approx(0.0)
    assert residuals[4] > 25.0
    assert residuals[5] > residuals[4]


def test_saxs_prefit_workflow_resolves_and_persists_linked_parameters(
    tmp_path,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    scale_entry = next(entry for entry in entries if entry.name == "scale")
    offset_entry = next(entry for entry in entries if entry.name == "offset")
    scale_entry.value = 2.5e-4
    offset_entry.initial_value_expression = "*scale"
    offset_entry.vary = True

    resolved_entries = resolve_prefit_parameter_entries(entries)
    resolved_offset = next(
        entry for entry in resolved_entries if entry.name == "offset"
    )
    assert resolved_offset.initial_value_expression == "*scale"
    assert resolved_offset.value_expression is None
    assert resolved_offset.value == pytest.approx(2.5e-4)
    assert resolved_offset.vary is True

    fit_result = workflow.run_fit(
        resolved_entries, method="leastsq", max_nfev=50
    )
    fitted_offset = next(
        entry
        for entry in fit_result.parameter_entries
        if entry.name == "offset"
    )
    assert fitted_offset.initial_value_expression == "*scale"
    assert fitted_offset.value_expression is None
    assert fitted_offset.vary is True

    workflow.save_fit(fit_result.parameter_entries)
    state_payload = json.loads(
        (paths.prefit_dir / "prefit_state.json").read_text(encoding="utf-8")
    )
    state_offset = next(
        entry
        for entry in state_payload["parameter_entries"]
        if entry["name"] == "offset"
    )
    assert state_offset["initial_value_expression"] == "*scale"

    prefit_payload = json.loads(
        (paths.prefit_dir / "pd_prefit_params.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        prefit_payload["fit_parameter_meta"]["offset"]["initial_expression"]
        == "*scale"
    )


def test_saxs_prefit_workflow_preserves_dependent_parameter_expressions(
    tmp_path,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    scale_entry = next(entry for entry in entries if entry.name == "scale")
    offset_entry = next(entry for entry in entries if entry.name == "offset")
    scale_entry.value = 2.5e-4
    offset_entry.value_expression = "*scale"
    offset_entry.vary = False

    resolved_entries = resolve_prefit_parameter_entries(entries)
    resolved_offset = next(
        entry for entry in resolved_entries if entry.name == "offset"
    )
    assert resolved_offset.value_expression == "*scale"
    assert resolved_offset.initial_value_expression is None
    assert resolved_offset.value == pytest.approx(2.5e-4)
    assert resolved_offset.vary is False

    fit_result = workflow.run_fit(
        resolved_entries,
        method="leastsq",
        max_nfev=50,
    )
    fitted_offset = next(
        entry
        for entry in fit_result.parameter_entries
        if entry.name == "offset"
    )
    assert fitted_offset.value_expression == "*scale"
    assert fitted_offset.initial_value_expression is None
    assert fitted_offset.vary is False

    workflow.save_fit(fit_result.parameter_entries)
    state_payload = json.loads(
        (paths.prefit_dir / "prefit_state.json").read_text(encoding="utf-8")
    )
    state_offset = next(
        entry
        for entry in state_payload["parameter_entries"]
        if entry["name"] == "offset"
    )
    assert state_offset["value_expression"] == "*scale"


def test_saxs_prefit_workflow_rejects_autoscale_for_linked_scale_or_offset(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    offset_entry = next(entry for entry in entries if entry.name == "offset")
    offset_entry.value_expression = "*scale"

    with pytest.raises(ValueError, match="offset is linked"):
        workflow.recommend_scale_settings(entries)


def test_saxs_prefit_workflow_allows_autoscale_for_expression_seed_parameters(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    offset_entry = next(entry for entry in entries if entry.name == "offset")
    offset_entry.initial_value_expression = "*scale"
    offset_entry.vary = True

    recommendation = workflow.recommend_scale_settings(entries)

    assert recommendation.recommended_scale > 0.0


def test_saxs_prefit_workflow_supports_model_only_evaluation(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.model_only_mode = True
    settings.use_experimental_grid = False
    settings.q_points = 8
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    evaluation = workflow.evaluate()

    assert evaluation.experimental_intensities is None
    assert evaluation.residuals is None
    assert evaluation.is_model_only is True
    assert workflow.can_run_prefit() is False
    with pytest.raises(ValueError, match="Model Only Mode"):
        workflow.run_fit(method="leastsq", max_nfev=100)


def test_saxs_prefit_workflow_recommends_scale_with_weighted_solvent_trace(
    tmp_path,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    solvent_q = np.linspace(0.05, 0.3, 8)
    solvent_intensity = np.linspace(1.5, 2.2, 8)
    solvent_path = tmp_path / "autoscale_solvent_trace.dat"
    np.savetxt(solvent_path, np.column_stack([solvent_q, solvent_intensity]))

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    template_module = load_template_module(
        "template_pd_likelihood_monosq_decoupled"
    )
    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    experimental = template_module.lmfit_model_profile(
        q_values,
        solvent_intensity,
        [component],
        w0=0.6,
        solv_w=0.5,
        offset=0.05,
        eff_r=9.0,
        vol_frac=0.0,
        scale=5e-4,
    )
    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    settings = manager.load_project(project_dir)
    settings.experimental_data_path = str(experimental_path)
    settings.copied_experimental_data_file = str(experimental_path)
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    for entry in entries:
        if entry.name == "solv_w":
            entry.value = 0.5
        if entry.name == "scale":
            entry.value = 1e-6
            entry.minimum = 1e-7
            entry.maximum = 1e-5

    recommendation = workflow.recommend_scale_settings(entries)

    assert recommendation.current_scale == pytest.approx(1e-6)
    assert recommendation.recommended_scale == pytest.approx(5e-4)
    assert recommendation.recommended_minimum == pytest.approx(5e-5)
    assert recommendation.recommended_maximum == pytest.approx(5e-3)
    assert recommendation.current_offset == pytest.approx(0.0)
    assert recommendation.recommended_offset == pytest.approx(0.05)
    assert recommendation.points_used == 8


def test_solute_volume_fraction_estimate_uses_component_densities():
    estimate = calculate_solute_volume_fraction_estimate(
        SoluteVolumeFractionSettings(
            solution=SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.2,
                solute_stoich="Cs1Pb1I3",
                solvent_stoich="H2O",
                molar_mass_solute=620.0,
                molar_mass_solvent=18.015,
                mass_solute=2.0,
                mass_solvent=8.0,
            ),
            solute_density_g_per_ml=2.0,
            solvent_density_g_per_ml=1.0,
        )
    )

    assert estimate.solute_volume_cm3 == pytest.approx(1.0)
    assert estimate.solvent_volume_cm3 == pytest.approx(8.0)
    assert estimate.solute_mass_concentration_g_per_cm3 == pytest.approx(
        2.0 / (10.0 / 1.2)
    )
    assert (
        estimate.approximate_solute_specific_volume_cm3_per_g
        == pytest.approx(0.5)
    )
    assert estimate.solute_volume_fraction == pytest.approx(1.0 / (10.0 / 1.2))
    assert estimate.solvent_volume_fraction == pytest.approx(
        1.0 - (1.0 / (10.0 / 1.2))
    )
    assert estimate.additive_to_solution_volume_ratio == pytest.approx(
        9.0 / (10.0 / 1.2)
    )
    summary = estimate.summary_text()
    assert "Solute mass / density: 2.000000 g / 2.0000 g/mL" in summary
    assert "Solvent mass / density: 8.000000 g / 1.0000 g/mL" in summary
    assert "Estimated solute volume: 1.000 cm^3" in summary
    assert "Estimated solvent volume: 8.000 cm^3" in summary


def test_solute_volume_fraction_molarity_mode_does_not_require_solute_density():
    estimate = calculate_solute_volume_fraction_estimate(
        SoluteVolumeFractionSettings(
            solution=SolutionPropertiesSettings(
                mode="molarity_per_liter",
                solution_density=1.1,
                solute_stoich="Pb1I2",
                solvent_stoich="C3H7NO",
                molar_mass_solute=461.0,
                molar_mass_solvent=73.09,
                molarity=0.5,
                molarity_element="Pb",
            ),
            solute_density_g_per_ml=None,
            solvent_density_g_per_ml=0.94,
        )
    )

    assert estimate.solution_result.mode == "molarity_per_liter"
    assert estimate.solvent_volume_cm3 is not None
    assert estimate.additive_volume_cm3 is None
    assert estimate.additive_to_solution_volume_ratio is None
    assert estimate.solute_volume_fraction == pytest.approx(0.075, rel=1e-6)
    assert estimate.approximate_solute_specific_volume_cm3_per_g > 0.0
    summary = estimate.summary_text()
    assert "Solute volume from solvent-density closure" in summary
    assert "Solvent mass / density:" in summary
    assert "solute density was not required" in summary.lower()


def test_poly_lma_prefit_workflow_exposes_solute_volume_fraction_target(
    tmp_path,
):
    project_dir, _paths, _radius = _build_poly_lma_geometry_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)

    assert workflow.supports_volume_fraction_estimator()
    assert workflow.volume_fraction_estimator_target() == (
        "phi_solute",
        "solute",
    )


def test_monosq_prefit_workflow_does_not_expose_solute_volume_fraction_target(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)

    assert not workflow.supports_volume_fraction_estimator()
    assert workflow.volume_fraction_estimator_target() is None


def test_monosq_prefit_workflow_exposes_solvent_weight_target(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)

    assert workflow.solvent_weight_estimator_target() == "solv_w"


def test_poly_lma_prefit_workflow_exposes_solvent_weight_target(tmp_path):
    project_dir, _paths, _radius = _build_poly_lma_geometry_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)

    assert workflow.solvent_weight_estimator_target() == "solvent_scale"


def test_poly_lma_prefit_defaults_fix_solvent_subtraction_controls(tmp_path):
    project_dir, _paths, _radius = _build_poly_lma_geometry_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries_by_name = {
        entry.name: entry for entry in workflow.load_template_reset_entries()
    }

    assert entries_by_name["phi_solute"].vary is False
    assert entries_by_name["solvent_scale"].vary is False


def test_poly_lma_prefit_rejects_redundant_solvent_subtraction_fit(tmp_path):
    project_dir, _paths, _radius = _build_poly_lma_geometry_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    for entry in entries:
        if entry.name in {"phi_solute", "solvent_scale"}:
            entry.vary = True

    with pytest.raises(
        ValueError,
        match="cannot both vary during fitting",
    ):
        workflow.run_fit(entries, method="leastsq", max_nfev=50)


def test_solution_scattering_estimate_reports_attenuation_and_scale():
    estimate = calculate_solution_scattering_estimate(
        SolutionScatteringEstimatorSettings(
            solution=SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.0,
                solute_stoich="Cs1Pb1I3",
                solvent_stoich="H2O",
                molar_mass_solute=620.0,
                molar_mass_solvent=18.015,
                mass_solute=1.0,
                mass_solvent=9.0,
            ),
            solute_density_g_per_ml=2.0,
            solvent_density_g_per_ml=1.0,
            beam=BeamGeometrySettings(
                incident_energy_kev=17.0,
                capillary_size_mm=1.0,
                capillary_geometry="cylindrical",
                beam_profile="uniform",
                beam_footprint_width_mm=0.4,
                beam_footprint_height_mm=0.4,
            ),
        )
    )

    assert estimate.number_density_estimate is not None
    assert estimate.number_density_estimate.number_density_a3 > 0.0
    assert estimate.volume_fraction_estimate is not None
    assert estimate.interaction_contrast_estimate is not None
    assert estimate.attenuation_estimate is not None
    attenuation = estimate.attenuation_estimate
    assert attenuation.solvent_scattering_scale_factor > 0.0
    assert attenuation.solvent_scattering_scale_factor < 1.0
    assert attenuation.sample_linear_attenuation_inv_cm > 0.0
    assert attenuation.sample_linear_attenuation_inv_cm > (
        attenuation.sample_solvent_linear_attenuation_inv_cm
    )
    assert 0.0 < attenuation.sample_transmission < 1.0
    assert 0.0 < attenuation.neat_solvent_transmission < 1.0
    assert (
        estimate.interaction_contrast_estimate.saxs_effective_solute_interaction_ratio
        < estimate.volume_fraction_estimate.solute_volume_fraction
    )


def test_solution_scattering_estimate_reports_saxs_effective_interaction_ratio():
    estimate = calculate_solution_scattering_estimate(
        SolutionScatteringEstimatorSettings(
            solution=SolutionPropertiesSettings(
                mode="molarity_per_liter",
                solution_density=1.1,
                solute_stoich="Pb1I2",
                solvent_stoich="C3H7NO",
                molar_mass_solute=461.0,
                molar_mass_solvent=73.09,
                molarity=0.5,
                molarity_element="Pb",
            ),
            solute_density_g_per_ml=None,
            solvent_density_g_per_ml=0.94,
        )
    )

    assert estimate.volume_fraction_estimate is not None
    assert estimate.interaction_contrast_estimate is not None
    interaction = estimate.interaction_contrast_estimate
    assert (
        interaction.physical_solute_associated_volume_fraction
        == pytest.approx(
            estimate.volume_fraction_estimate.solute_volume_fraction
        )
    )
    assert 0.0 < interaction.saxs_effective_solute_interaction_ratio < 1.0
    assert 0.0 < interaction.saxs_effective_solvent_background_ratio < 1.0
    assert (
        interaction.saxs_effective_solute_interaction_ratio
        == pytest.approx(
            1.0 - interaction.saxs_effective_solvent_background_ratio
        )
    )
    assert (
        interaction.saxs_effective_solute_interaction_ratio
        > interaction.physical_solute_associated_volume_fraction
    )
    summary = estimate.summary_text()
    assert "Physical solute-associated volume fraction estimate" in summary
    assert "SAXS-effective interaction contrast estimate" in summary
    assert "Model-facing solvent defaults" in summary


def test_solution_scattering_estimate_reports_fluorescence_lines():
    estimate = calculate_solution_scattering_estimate(
        SolutionScatteringEstimatorSettings(
            solution=SolutionPropertiesSettings(
                mode="mass",
                solution_density=1.0,
                solute_stoich="Cs1Pb1I3",
                solvent_stoich="H2O",
                molar_mass_solute=620.0,
                molar_mass_solvent=18.015,
                mass_solute=1.0,
                mass_solvent=9.0,
            ),
            solute_density_g_per_ml=2.0,
            solvent_density_g_per_ml=1.0,
            calculate_solute_volume_fraction=False,
            calculate_solvent_scattering_contribution=False,
            calculate_sample_fluorescence_yield=True,
        )
    )

    assert estimate.fluorescence_estimate is not None
    fluorescence = estimate.fluorescence_estimate
    assert fluorescence.total_primary_detected_yield > 0.0
    assert fluorescence.total_secondary_detected_yield >= 0.0
    assert fluorescence.line_estimates


def test_run_prefit_preserves_manual_parameter_values_outside_old_bounds(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    weight_entry = next(entry for entry in entries if entry.name == "w0")

    assert weight_entry.maximum < 0.9
    weight_entry.value = 0.9
    weight_entry.vary = False

    result = workflow.run_fit(entries, method="leastsq", max_nfev=200)

    fitted_entry = next(
        entry for entry in result.parameter_entries if entry.name == "w0"
    )
    assert fitted_entry.value == pytest.approx(0.9)
    assert fitted_entry.maximum >= 0.9
    assert fitted_entry.minimum <= 0.9


def test_prefit_workflow_uses_reduced_saved_q_range_without_rebuild(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.q_min = 0.12
    settings.q_max = 0.19
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    evaluation = workflow.evaluate()

    assert np.allclose(
        evaluation.q_values,
        [0.12142857142857144, 0.15714285714285714, 0.19285714285714284],
    )
    assert len(evaluation.experimental_intensities) == 3
    assert len(evaluation.model_intensities) == 3


def test_prefit_workflow_snaps_tiny_q_range_edge_mismatch_without_rebuild(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.q_min = 0.04995
    settings.q_max = 0.30005
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    evaluation = workflow.evaluate()

    assert np.allclose(
        evaluation.q_values,
        np.linspace(0.05, 0.3, 8),
    )


def test_prefit_workflow_rejects_expanded_q_range_until_components_rebuilt(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.q_min = 0.04
    settings.q_max = 0.31
    manager.save_project(settings)

    with pytest.raises(
        ValueError,
        match="Recompute the SAXS model components",
    ):
        SAXSPrefitWorkflow(project_dir)


def test_saxs_prefit_workflow_loads_selected_solvent_trace(tmp_path):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    solvent_q = np.linspace(0.05, 0.3, 8)
    solvent_intensity = np.linspace(1.5, 2.2, 8)
    solvent_path = tmp_path / "custom_solvent_trace.dat"
    np.savetxt(solvent_path, np.column_stack([solvent_q, solvent_intensity]))

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)

    assert workflow.solvent_data is not None
    assert np.allclose(workflow.solvent_data, solvent_intensity)

    bundle_like_copy = manager.stage_solvent_data(settings)
    assert bundle_like_copy == (
        paths.experimental_data_dir / solvent_path.name
    )
    assert bundle_like_copy.is_file()


def test_saxs_prefit_workflow_evaluates_solvent_contribution(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    solvent_q = np.linspace(0.05, 0.3, 8)
    solvent_intensity = np.linspace(1.5, 2.2, 8)
    solvent_path = tmp_path / "weighted_solvent_trace.dat"
    np.savetxt(solvent_path, np.column_stack([solvent_q, solvent_intensity]))

    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.solvent_data_path = str(solvent_path)
    settings.copied_solvent_data_file = None
    manager.save_project(settings)

    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    for entry in entries:
        if entry.name == "solv_w":
            entry.value = 0.5
        if entry.name == "scale":
            entry.value = 2e-3

    evaluation = workflow.evaluate(entries)

    assert evaluation.solvent_intensities is not None
    assert evaluation.solvent_contribution is not None
    assert np.allclose(evaluation.solvent_intensities, solvent_intensity)
    assert np.allclose(
        evaluation.solvent_contribution,
        solvent_intensity * 0.5,
    )


def test_saxs_prefit_workflow_evaluates_monosq_structure_factor_trace(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    eff_r = 11.0
    vol_frac = 0.03
    for entry in entries:
        if entry.name == "eff_r":
            entry.value = eff_r
        if entry.name == "vol_frac":
            entry.value = vol_frac

    evaluation = workflow.evaluate(entries)
    template_module = load_template_module(
        "template_pd_likelihood_monosq_decoupled"
    )

    assert evaluation.structure_factor_trace is not None
    assert np.allclose(
        evaluation.structure_factor_trace,
        template_module.calc_monodisperse_sq(
            eff_r,
            vol_frac,
            evaluation.q_values,
        ),
    )


def test_poly_lma_prefit_evaluates_structure_factor_trace(tmp_path):
    project_dir, _paths, effective_radius = _build_poly_lma_geometry_project(
        tmp_path,
        template_name=POLY_LMA_HS_TEMPLATE,
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()
    entries = workflow.load_parameter_entries()
    phi_int = 0.02
    for entry in entries:
        if entry.name == "phi_int":
            entry.value = phi_int

    evaluation = workflow.evaluate(entries)
    template_module = load_template_module(POLY_LMA_HS_TEMPLATE)

    assert evaluation.structure_factor_trace is not None
    assert np.allclose(
        evaluation.structure_factor_trace,
        template_module.calc_hardsphere_sq(
            effective_radius,
            phi_int,
            evaluation.q_values,
        ),
    )


def test_poly_lma_prefit_clamps_saved_solvent_weight_bounds(tmp_path):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_template_reset_entries()
    for entry in entries:
        if entry.name == "solvent_scale":
            entry.value = 1.6
            entry.minimum = 0.0
            entry.maximum = 5.0
    workflow.settings.template_reset_parameter_entries = [
        entry.to_dict() for entry in entries
    ]
    workflow.project_manager.save_project(workflow.settings)

    reloaded = SAXSPrefitWorkflow(project_dir)
    solvent_entry = next(
        entry
        for entry in reloaded.load_template_reset_entries()
        if entry.name == "solvent_scale"
    )

    assert solvent_entry.value == pytest.approx(1.0)
    assert solvent_entry.minimum == pytest.approx(0.0)
    assert solvent_entry.maximum == pytest.approx(1.0)


def test_saxs_prefit_workflow_persists_template_reset_and_best_prefit(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)

    template_entries = workflow.load_template_reset_entries()
    assert next(
        entry.value for entry in template_entries if entry.name == "scale"
    ) == pytest.approx(5e-4)

    best_entries = workflow.load_parameter_entries()
    for entry in best_entries:
        if entry.name == "scale":
            entry.value = 9e-4
        if entry.name == "offset":
            entry.value = 0.22
    workflow.save_best_prefit_entries(best_entries)

    reloaded = SAXSPrefitWorkflow(project_dir)
    reloaded_entries = {
        entry.name: entry for entry in reloaded.parameter_entries
    }
    reloaded_template_entries = {
        entry.name: entry for entry in reloaded.load_template_reset_entries()
    }

    assert reloaded_entries["scale"].value == pytest.approx(9e-4)
    assert reloaded_entries["offset"].value == pytest.approx(0.22)
    assert reloaded_template_entries["scale"].value == pytest.approx(5e-4)
    assert reloaded_template_entries["offset"].value == pytest.approx(0.0)


def test_saxs_prefit_workflow_loads_saved_snapshot_state(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSPrefitWorkflow(project_dir)
    entries = workflow.load_parameter_entries()
    for entry in entries:
        if entry.name == "scale":
            entry.value = 7e-4
            entry.minimum = 1e-5
            entry.maximum = 8e-3
            entry.vary = True
        if entry.name == "offset":
            entry.value = 0.125

    report_path = workflow.save_fit(
        entries,
        method="powell",
        max_nfev=4321,
        autosave_prefits=True,
    )

    saved_state = workflow.load_saved_state(report_path.parent.name)

    loaded_entries = {
        entry.name: entry for entry in saved_state.parameter_entries
    }
    assert saved_state.method == "powell"
    assert saved_state.max_nfev == 4321
    assert saved_state.autosave_prefits is True
    assert loaded_entries["scale"].value == pytest.approx(7e-4)
    assert loaded_entries["offset"].value == pytest.approx(0.125)


def test_poly_lma_saved_snapshot_preserves_cluster_geometry_state(tmp_path):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()
    workflow.set_cluster_geometry_active_radii_type("bond_length")
    updated_rows = workflow.cluster_geometry_rows()
    updated_rows[0].sf_approximation = "ellipsoid"
    workflow.set_cluster_geometry_rows(updated_rows)

    report_path = workflow.save_fit(
        workflow.parameter_entries,
        method="powell",
        max_nfev=1234,
    )
    saved_state = workflow.load_saved_state(report_path.parent.name)

    assert saved_state.cluster_geometry_table is not None
    assert saved_state.cluster_geometry_table.active_radii_type == (
        "bond_length"
    )
    restored_row = saved_state.cluster_geometry_table.rows[0]
    assert restored_row.sf_approximation == "ellipsoid"
    assert restored_row.radii_type_used == "bond_length"
    assert restored_row.effective_radius == pytest.approx(
        workflow.cluster_geometry_rows()[0].effective_radius
    )


def test_poly_lma_prefit_workflow_computes_and_uses_cluster_geometry(
    tmp_path,
):
    project_dir, paths, effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)

    assert workflow.supports_cluster_geometry_metadata()
    assert workflow.cluster_geometry_rows() == []

    with pytest.raises(ValueError, match="cluster geometry metadata"):
        workflow.evaluate()

    table = workflow.compute_cluster_geometry_table()

    assert paths.cluster_geometry_metadata_file.is_file()
    assert len(table.rows) == 1
    assert table.rows[0].mapped_parameter == "w0"
    assert table.active_radii_type == "ionic"
    assert table.rows[0].radii_type_used == "ionic"
    assert table.rows[0].effective_radius == pytest.approx(effective_radius)

    evaluation = workflow.evaluate()

    assert np.allclose(
        evaluation.model_intensities,
        evaluation.experimental_intensities,
    )
    assert workflow.template_runtime_inputs_payload()["effective_radii"] == (
        pytest.approx([effective_radius])
    )


def test_poly_lma_prefit_generates_geometry_parameters_from_sf_shape(
    tmp_path,
):
    project_dir, paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()

    sphere_rows = workflow.cluster_geometry_rows()
    sphere_rows[0].sf_approximation = "sphere"
    workflow.set_cluster_geometry_rows(sphere_rows)
    sphere_entries = {
        entry.name: entry for entry in workflow.parameter_entries
    }

    assert "r_eff_w0" in sphere_entries
    assert "a_eff_w0" not in sphere_entries
    assert "b_eff_w0" not in sphere_entries
    assert "c_eff_w0" not in sphere_entries
    assert sphere_entries["r_eff_w0"].vary is False

    current_entries = workflow.parameter_entries
    for entry in current_entries:
        if entry.name == "scale":
            entry.value = 2.5
    workflow.parameter_entries = current_entries

    ellipsoid_rows = workflow.cluster_geometry_rows()
    ellipsoid_rows[0].sf_approximation = "ellipsoid"
    workflow.set_cluster_geometry_rows(ellipsoid_rows)
    ellipsoid_entries = {
        entry.name: entry for entry in workflow.parameter_entries
    }

    assert "r_eff_w0" not in ellipsoid_entries
    assert "a_eff_w0" in ellipsoid_entries
    assert "b_eff_w0" in ellipsoid_entries
    assert "c_eff_w0" in ellipsoid_entries
    assert ellipsoid_entries["a_eff_w0"].vary is False
    assert ellipsoid_entries["scale"].value == pytest.approx(2.5)

    workflow.save_fit(workflow.parameter_entries)
    payload = json.loads(
        (paths.prefit_dir / "pd_prefit_params.json").read_text(
            encoding="utf-8"
        )
    )
    assert "a_eff_w0" in payload["fit_parameters"]
    assert "b_eff_w0" in payload["fit_parameters"]
    assert "c_eff_w0" in payload["fit_parameters"]
    assert "r_eff_w0" not in payload["fit_parameters"]


def test_strict_poly_lma_hs_restricts_cluster_geometry_to_spheres(tmp_path):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path,
        template_name=POLY_LMA_HS_TEMPLATE,
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    table = workflow.compute_cluster_geometry_table()

    assert workflow.allowed_cluster_geometry_approximations() == ("sphere",)
    assert len(table.rows) == 1
    assert table.rows[0].sf_approximation == "sphere"
    assert table.rows[0].structure_factor_recommendation == "sphere"

    geometry_names = {
        entry.name
        for entry in workflow.parameter_entries
        if entry.category == "geometry"
    }
    assert geometry_names == {"r_eff_w0"}


def test_cluster_geometry_metadata_uses_ionic_radius_for_single_atom_clusters(
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "I"
    structure_dir.mkdir(parents=True, exist_ok=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "1\nsingle atom\nI 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    table = compute_cluster_geometry_metadata(clusters_dir)

    assert table.active_radii_type == "ionic"
    assert len(table.rows) == 1

    row = table.rows[0]
    assert row.radii_type_used == "ionic"
    assert row.structure_factor_recommendation == "sphere"
    assert row.sf_approximation == "sphere"
    assert row.ionic_sphere_effective_radius == pytest.approx(2.2)
    assert row.effective_radius == pytest.approx(2.2)
    assert row.bond_length_sphere_effective_radius == pytest.approx(1.39)
    assert row.ionic_ellipsoid_semiaxis_a == pytest.approx(2.2)
    assert row.ionic_ellipsoid_semiaxis_b == pytest.approx(2.2)
    assert row.ionic_ellipsoid_semiaxis_c == pytest.approx(2.2)
    assert row.bond_length_ellipsoid_semiaxis_a == pytest.approx(1.39)
    assert row.bond_length_ellipsoid_semiaxis_b == pytest.approx(1.39)
    assert row.bond_length_ellipsoid_semiaxis_c == pytest.approx(1.39)
    assert (
        row.bond_length_sphere_effective_radius
        < row.ionic_sphere_effective_radius
    )
    assert (
        "Single-atom bond-length mode uses a covalent-radius proxy"
        in row.notes
    )


def test_cluster_geometry_metadata_loads_only_one_single_atom_structure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "I"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        (structure_dir / f"frame_{index:04d}.xyz").write_text(
            "1\nsingle atom\nI 0.0 0.0 0.0\n",
            encoding="utf-8",
        )

    load_counter = {"count": 0}

    def fake_load_structure_file(_file_path):
        load_counter["count"] += 1
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float), ["I"]

    monkeypatch.setattr(
        cluster_geometry_module,
        "load_structure_file",
        fake_load_structure_file,
    )

    table = compute_cluster_geometry_metadata(clusters_dir)

    assert load_counter["count"] == 1
    assert len(table.rows) == 1
    assert table.rows[0].file_count == 4


def test_cluster_geometry_parallel_path_matches_serial_results(tmp_path):
    clusters_dir = tmp_path / "clusters"

    a_dir = clusters_dir / "A"
    a_dir.mkdir(parents=True, exist_ok=True)
    (a_dir / "frame_0001.xyz").write_text(
        "4\nframe 1\nPb 0.0 0.0 0.0\nI 2.0 0.0 0.0\n"
        "I 0.0 2.0 0.0\nI 0.0 0.0 2.0\n",
        encoding="utf-8",
    )
    (a_dir / "frame_0002.xyz").write_text(
        "4\nframe 2\nPb 0.0 0.0 0.0\nI 2.1 0.0 0.0\n"
        "I 0.0 1.9 0.0\nI 0.0 0.0 2.2\n",
        encoding="utf-8",
    )

    b_dir = clusters_dir / "B"
    b_dir.mkdir(parents=True, exist_ok=True)
    (b_dir / "frame_0001.xyz").write_text(
        "3\nframe 1\nNa 0.0 0.0 0.0\nCl 2.4 0.0 0.0\nCl -2.4 0.0 0.0\n",
        encoding="utf-8",
    )
    (b_dir / "frame_0002.xyz").write_text(
        "3\nframe 2\nNa 0.0 0.0 0.0\nCl 2.2 0.0 0.0\nCl -2.3 0.0 0.0\n",
        encoding="utf-8",
    )

    serial_table = compute_cluster_geometry_metadata(
        clusters_dir,
        max_workers=1,
    )
    parallel_table = compute_cluster_geometry_metadata(
        clusters_dir,
        max_workers=2,
    )

    assert serial_table.active_radii_type == parallel_table.active_radii_type
    assert (
        serial_table.active_ionic_radius_type
        == parallel_table.active_ionic_radius_type
    )
    assert [row.to_dict() for row in serial_table.rows] == [
        row.to_dict() for row in parallel_table.rows
    ]


def test_cluster_geometry_metadata_supports_crystal_ionic_radii(
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    structure_dir = clusters_dir / "I"
    structure_dir.mkdir(parents=True, exist_ok=True)
    (structure_dir / "frame_0001.xyz").write_text(
        "1\nsingle atom\nI 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    table = compute_cluster_geometry_metadata(
        clusters_dir,
        active_ionic_radius_type="crystal",
    )

    assert table.active_radii_type == "ionic"
    assert table.active_ionic_radius_type == "crystal"
    assert len(table.rows) == 1

    row = table.rows[0]
    assert row.radii_type_used == "ionic"
    assert row.ionic_radius_type_used == "crystal"
    assert row.crystal_ionic_sphere_effective_radius == pytest.approx(2.34)
    assert row.effective_radius == pytest.approx(2.34)
    assert row.crystal_ionic_ellipsoid_semiaxis_a == pytest.approx(2.34)
    assert row.crystal_ionic_ellipsoid_semiaxis_b == pytest.approx(2.34)
    assert row.crystal_ionic_ellipsoid_semiaxis_c == pytest.approx(2.34)


def test_single_atom_bond_length_radius_is_smaller_than_pb_i_pair_cluster(
    tmp_path,
):
    clusters_dir = tmp_path / "clusters"
    iodide_dir = clusters_dir / "I"
    iodide_dir.mkdir(parents=True, exist_ok=True)
    (iodide_dir / "frame_0001.xyz").write_text(
        "1\nsingle atom\nI 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    pair_dir = clusters_dir / "PbI_pair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "frame_0001.xyz").write_text(
        "2\npair\nPb 0.0 0.0 0.0\nI 2.8 0.0 0.0\n",
        encoding="utf-8",
    )

    table = compute_cluster_geometry_metadata(clusters_dir)
    rows = {row.structure: row for row in table.rows}

    assert rows["I"].bond_length_sphere_effective_radius == pytest.approx(1.39)
    assert (
        rows["I"].bond_length_sphere_effective_radius
        < rows["PbI_pair"].bond_length_sphere_effective_radius
    )


def test_cluster_geometry_metadata_upgrades_legacy_payload(tmp_path):
    metadata_path = tmp_path / "cluster_geometry_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "cluster_id": "A",
                        "structure": "A",
                        "motif": "no_motif",
                        "cluster_path": "/tmp/A",
                        "avg_size_metric": 4.0,
                        "effective_radius": 2.0,
                        "structure_factor_recommendation": "ellipsoid",
                        "anisotropy_metric": 1.8,
                        "notes": "legacy payload",
                        "mapped_parameter": "w0",
                        "mean_semiaxis_a": 3.0,
                        "mean_semiaxis_b": 2.0,
                        "mean_semiaxis_c": 1.0,
                        "mean_radius_of_gyration": 1.4,
                        "mean_max_radius": 3.0,
                        "mean_atom_count": 4.0,
                        "file_count": 2,
                    }
                ],
                "source_clusters_dir": "/tmp/clusters",
                "computed_at": "2026-03-24T00:00:00",
                "anisotropy_threshold": 1.25,
                "template_name": "template_pydream_poly_lma_hs",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table = load_cluster_geometry_metadata(metadata_path)

    assert table.schema_version == 3
    assert table.active_radii_type == "ionic"
    assert table.active_ionic_radius_type == "effective"
    assert len(table.rows) == 1

    row = table.rows[0]
    assert row.structure_factor_recommendation == "ellipsoid"
    assert row.sf_approximation == "sphere"
    assert row.radii_type_used == "ionic"
    assert row.ionic_radius_type_used == "effective"
    assert row.ionic_sphere_effective_radius == pytest.approx(2.0)
    assert row.crystal_ionic_sphere_effective_radius == pytest.approx(2.14)
    assert row.bond_length_sphere_effective_radius == pytest.approx(2.0)
    assert row.ionic_ellipsoid_semiaxis_a == pytest.approx(3.0)
    assert row.ionic_ellipsoid_semiaxis_b == pytest.approx(2.0)
    assert row.ionic_ellipsoid_semiaxis_c == pytest.approx(1.0)
    assert row.crystal_ionic_ellipsoid_semiaxis_a == pytest.approx(3.14)
    assert row.crystal_ionic_ellipsoid_semiaxis_b == pytest.approx(2.14)
    assert row.crystal_ionic_ellipsoid_semiaxis_c == pytest.approx(1.14)
    assert row.bond_length_ellipsoid_semiaxis_a == pytest.approx(3.0)
    assert row.bond_length_ellipsoid_semiaxis_b == pytest.approx(2.0)
    assert row.bond_length_ellipsoid_semiaxis_c == pytest.approx(1.0)
    assert row.effective_radius == pytest.approx(2.0)


def test_single_atom_cluster_geometry_upgrades_old_ionic_bond_length_fallback(
    tmp_path,
):
    metadata_path = tmp_path / "cluster_geometry_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "cluster_id": "I",
                        "structure": "I",
                        "motif": "no_motif",
                        "cluster_path": "/tmp/I",
                        "avg_size_metric": 4.4,
                        "effective_radius": 2.2,
                        "structure_factor_recommendation": "sphere",
                        "anisotropy_metric": 1.0,
                        "notes": "older single-atom payload",
                        "mapped_parameter": "w0",
                        "sf_approximation": "sphere",
                        "radii_type_used": "bond_length",
                        "ionic_radius_type_used": "effective",
                        "ionic_sphere_effective_radius": 2.2,
                        "bond_length_sphere_effective_radius": 2.2,
                        "ionic_ellipsoid_semiaxis_a": 2.2,
                        "ionic_ellipsoid_semiaxis_b": 2.2,
                        "ionic_ellipsoid_semiaxis_c": 2.2,
                        "bond_length_ellipsoid_semiaxis_a": 2.2,
                        "bond_length_ellipsoid_semiaxis_b": 2.2,
                        "bond_length_ellipsoid_semiaxis_c": 2.2,
                        "mean_semiaxis_a": 2.2,
                        "mean_semiaxis_b": 2.2,
                        "mean_semiaxis_c": 2.2,
                        "mean_radius_of_gyration": 0.0,
                        "mean_max_radius": 2.2,
                        "mean_atom_count": 1.0,
                        "file_count": 1,
                    }
                ],
                "source_clusters_dir": "/tmp/clusters",
                "computed_at": "2026-03-25T00:00:00",
                "anisotropy_threshold": 1.25,
                "template_name": "template_pydream_poly_lma_hs",
                "schema_version": 3,
                "active_radii_type": "bond_length",
                "active_ionic_radius_type": "effective",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table = load_cluster_geometry_metadata(metadata_path)
    row = table.rows[0]

    assert row.bond_length_sphere_effective_radius == pytest.approx(1.39)
    assert row.bond_length_ellipsoid_semiaxis_a == pytest.approx(1.39)
    assert row.bond_length_ellipsoid_semiaxis_b == pytest.approx(1.39)
    assert row.bond_length_ellipsoid_semiaxis_c == pytest.approx(1.39)
    assert row.effective_radius == pytest.approx(1.39)


def test_poly_lma_prefit_workflow_persists_active_cluster_geometry_mode(
    tmp_path,
):
    project_dir, paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()

    default_row = workflow.cluster_geometry_rows()[0]
    workflow.set_cluster_geometry_active_radii_type("bond_length")
    bond_length_row = workflow.cluster_geometry_rows()[0]

    assert workflow.cluster_geometry_active_radii_type() == "bond_length"
    assert bond_length_row.radii_type_used == "bond_length"
    assert bond_length_row.effective_radius == pytest.approx(
        bond_length_row.bond_length_sphere_effective_radius
    )
    assert bond_length_row.effective_radius != pytest.approx(
        default_row.ionic_sphere_effective_radius
    )

    updated_rows = workflow.cluster_geometry_rows()
    updated_rows[0].sf_approximation = "ellipsoid"
    workflow.set_cluster_geometry_rows(updated_rows)

    reloaded = SAXSPrefitWorkflow(project_dir)
    reloaded_row = reloaded.cluster_geometry_rows()[0]
    expected_effective_radius = float(
        np.cbrt(
            reloaded_row.bond_length_ellipsoid_semiaxis_a
            * reloaded_row.bond_length_ellipsoid_semiaxis_b
            * reloaded_row.bond_length_ellipsoid_semiaxis_c
        )
    )

    assert paths.cluster_geometry_metadata_file.is_file()
    assert reloaded.cluster_geometry_active_radii_type() == "bond_length"
    assert reloaded_row.radii_type_used == "bond_length"
    assert reloaded_row.sf_approximation == "ellipsoid"
    assert reloaded_row.active_semiaxis_a == pytest.approx(
        reloaded_row.bond_length_ellipsoid_semiaxis_a
    )
    assert reloaded_row.active_semiaxis_b == pytest.approx(
        reloaded_row.bond_length_ellipsoid_semiaxis_b
    )
    assert reloaded_row.active_semiaxis_c == pytest.approx(
        reloaded_row.bond_length_ellipsoid_semiaxis_c
    )
    assert reloaded_row.effective_radius == pytest.approx(
        expected_effective_radius
    )
    assert reloaded.template_runtime_inputs_payload()["effective_radii"] == (
        pytest.approx([expected_effective_radius])
    )


def test_poly_lma_prefit_workflow_persists_active_ionic_radius_type(
    tmp_path,
):
    project_dir, paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()
    default_row = workflow.cluster_geometry_rows()[0]

    workflow.set_cluster_geometry_active_ionic_radius_type("crystal")
    crystal_row = workflow.cluster_geometry_rows()[0]

    assert workflow.cluster_geometry_active_radii_type() == "ionic"
    assert workflow.cluster_geometry_active_ionic_radius_type() == "crystal"
    assert crystal_row.ionic_radius_type_used == "crystal"
    assert crystal_row.effective_radius == pytest.approx(
        crystal_row.crystal_ionic_sphere_effective_radius
    )
    assert crystal_row.effective_radius > default_row.effective_radius

    reloaded = SAXSPrefitWorkflow(project_dir)
    reloaded_row = reloaded.cluster_geometry_rows()[0]

    assert paths.cluster_geometry_metadata_file.is_file()
    assert reloaded.cluster_geometry_active_radii_type() == "ionic"
    assert reloaded.cluster_geometry_active_ionic_radius_type() == "crystal"
    assert reloaded_row.ionic_radius_type_used == "crystal"
    assert reloaded_row.effective_radius == pytest.approx(
        reloaded_row.crystal_ionic_sphere_effective_radius
    )
    assert reloaded.template_runtime_inputs_payload()["effective_radii"] == (
        pytest.approx([reloaded_row.crystal_ionic_sphere_effective_radius])
    )


def test_poly_lma_prefit_workflow_rejects_nonpositive_active_radii(
    tmp_path,
):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()

    updated_rows = workflow.cluster_geometry_rows()
    updated_rows[0].bond_length_sphere_effective_radius = 0.0
    updated_rows[0].bond_length_ellipsoid_semiaxis_a = 0.0
    updated_rows[0].bond_length_ellipsoid_semiaxis_b = 0.0
    updated_rows[0].bond_length_ellipsoid_semiaxis_c = 0.0
    workflow.set_cluster_geometry_rows(updated_rows)

    with pytest.raises(ValueError, match="positive active radii"):
        workflow.set_cluster_geometry_active_radii_type("bond_length")

    assert workflow.cluster_geometry_active_radii_type() == "ionic"


def test_poly_lma_prefit_workflow_allows_bond_length_mode_for_single_atom_clusters(
    tmp_path,
):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path,
        template_name=POLY_LMA_HS_TEMPLATE,
        single_atom=True,
    )
    workflow = SAXSPrefitWorkflow(project_dir)
    workflow.compute_cluster_geometry_table()

    workflow.set_cluster_geometry_active_radii_type("bond_length")

    row = workflow.cluster_geometry_rows()[0]
    assert workflow.cluster_geometry_active_radii_type() == "bond_length"
    assert row.radii_type_used == "bond_length"
    assert row.effective_radius == pytest.approx(
        row.bond_length_sphere_effective_radius
    )
    assert row.effective_radius == pytest.approx(1.39)
    assert row.effective_radius < row.ionic_sphere_effective_radius
