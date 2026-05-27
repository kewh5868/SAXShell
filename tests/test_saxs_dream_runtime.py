from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from saxshell.saxs._model_templates import (
    list_template_specs,
    load_template_module,
)
from saxshell.saxs.dream import (
    DreamRunSettings,
    PosteriorFilterSettings,
    SAXSDreamResultsLoader,
    SAXSDreamWorkflow,
)
from saxshell.saxs.dream import batch as dream_batch_module
from saxshell.saxs.dream import load_dream_settings
from saxshell.saxs.dream.batch import (
    DreamBatchRunSetManager,
    load_dream_batch_manifest,
    run_dream_batch_manifest,
)
from saxshell.saxs.dream.distributions import (
    DreamParameterEntry,
    save_parameter_map,
)
from saxshell.saxs.prefit import (
    PrefitParameterEntry,
    SAXSPrefitWorkflow,
    compute_cluster_geometry_metadata,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.stoichiometry_compensator import (
    STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT,
    STOICH_COMPENSATOR_MASK_INPUT,
    STOICH_COMPONENT_COUNTS_INPUT,
    STOICH_TARGET_RATIO_INPUT,
    STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME,
)

POLY_LMA_HS_MIX_TEMPLATE = "template_pydream_poly_lma_hs_mix_approx"


def _runtime_contract_extra_inputs(template_spec):
    values = {
        binding.runtime_name: np.asarray([9.0], dtype=float)
        for binding in template_spec.cluster_geometry_support.runtime_bindings
    }
    for input_name in template_spec.extra_lmfit_inputs:
        if input_name in values:
            continue
        if input_name == STOICH_TARGET_RATIO_INPUT:
            values[input_name] = np.asarray([1.0, 2.0], dtype=float)
        elif input_name == STOICH_COMPONENT_COUNTS_INPUT:
            values[input_name] = np.asarray([[1.0, 2.0]], dtype=float)
        elif input_name == STOICH_COMPENSATOR_MASK_INPUT:
            values[input_name] = np.asarray([0.0], dtype=float)
        elif input_name == STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT:
            values[input_name] = np.asarray([0.0], dtype=float)
    return values


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
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    _write_component_file(
        paths.scattering_components_dir / "A_no_motif.txt",
        q_values,
        component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
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
    manager.save_project(settings)
    return project_dir, paths


def _build_two_component_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "two_component_saxs_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    pbi3_component = np.linspace(10.0, 17.0, 8)
    pbi4_component = np.linspace(3.0, 10.0, 8)
    template_name = "template_pd_likelihood_monosq_decoupled"
    template_module = load_template_module(template_name)
    experimental = template_module.lmfit_model_profile(
        q_values,
        np.zeros_like(q_values),
        [pbi3_component, pbi4_component],
        w0=0.4,
        w1=0.6,
        solv_w=0.0,
        offset=0.05,
        eff_r=9.0,
        vol_frac=0.0,
        scale=5e-4,
    )
    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    _write_component_file(
        paths.scattering_components_dir / "PbI3_no_motif.txt",
        q_values,
        pbi3_component,
    )
    _write_component_file(
        paths.scattering_components_dir / "PbI4_no_motif.txt",
        q_values,
        pbi4_component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 2,
                "structures": {
                    "PbI3": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.4,
                            "representative": "frame_0001.xyz",
                            "profile_file": "PbI3_no_motif.txt",
                        }
                    },
                    "PbI4": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.6,
                            "representative": "frame_0002.xyz",
                            "profile_file": "PbI4_no_motif.txt",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map.json").write_text(
        json.dumps(
            {
                "saxs_map": {
                    "PbI3": {"no_motif": "PbI3_no_motif.txt"},
                    "PbI4": {"no_motif": "PbI4_no_motif.txt"},
                }
            },
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


def _build_stoichiometry_compensator_saxs_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "stoich_compensator_saxs_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    pbi2_component = np.linspace(10.0, 17.0, 8)
    iodine_component = np.linspace(0.5, 0.8, 8)
    template_module = load_template_module(
        STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME
    )
    experimental = template_module.lmfit_model_profile(
        q_values,
        np.zeros_like(q_values),
        [iodine_component, pbi2_component],
        np.asarray([1.0, 3.0], dtype=float),
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float),
        np.asarray([1.0, 0.0], dtype=float),
        np.asarray([0.2, 0.0], dtype=float),
        w0=0.2,
        w1=0.5,
        solv_w=0.0,
        offset=0.05,
        eff_r=9.0,
        vol_frac=0.0,
        scale=5e-4,
    )
    experimental_path = paths.experimental_data_dir / "exp_demo.txt"
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    _write_component_file(
        paths.scattering_components_dir / "PbI2_no_motif.txt",
        q_values,
        pbi2_component,
    )
    _write_component_file(
        paths.scattering_components_dir / "I_no_motif.txt",
        q_values,
        iodine_component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 2,
                "structures": {
                    "PbI2": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.5,
                            "representative": "frame_0001.xyz",
                            "profile_file": "PbI2_no_motif.txt",
                        }
                    },
                    "I": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.2,
                            "representative": "frame_0002.xyz",
                            "profile_file": "I_no_motif.txt",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.project_dir / "md_saxs_map.json").write_text(
        json.dumps(
            {
                "saxs_map": {
                    "PbI2": {"no_motif": "PbI2_no_motif.txt"},
                    "I": {"no_motif": "I_no_motif.txt"},
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    settings.experimental_data_path = str(experimental_path)
    settings.copied_experimental_data_file = str(experimental_path)
    settings.selected_model_template = STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME
    settings.stoichiometry_compensator_target_elements_text = "Pb, I"
    settings.stoichiometry_compensator_target_ratio_text = "1:3"
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


def _build_poly_lma_geometry_project(
    tmp_path,
    *,
    template_name: str = POLY_LMA_HS_MIX_TEMPLATE,
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
    np.savetxt(experimental_path, np.column_stack([q_values, experimental]))
    _write_component_file(
        paths.scattering_components_dir / "A_no_motif.txt",
        q_values,
        component,
    )
    (paths.project_dir / "md_prior_weights.json").write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 2,
                "structures": {
                    "A": {
                        "no_motif": {
                            "count": 2,
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


def test_saxs_templates_support_runtime_contract():
    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    solvent = np.linspace(1.0, 2.4, 8)

    for template_spec in list_template_specs():
        module = load_template_module(template_spec.name)
        assert template_spec.lmfit_inputs[0:3] == (
            "q",
            "solvent_data",
            "model_data",
        )
        assert template_spec.lmfit_inputs[-1] == "params"

        fit_params = {
            parameter.name: parameter.initial_value
            for parameter in template_spec.parameters
        }
        template_runtime_inputs = _runtime_contract_extra_inputs(template_spec)
        extra_inputs = [
            template_runtime_inputs[input_name]
            for input_name in template_spec.extra_lmfit_inputs
        ]
        profile = getattr(module, template_spec.lmfit_model_name)(
            q_values,
            solvent,
            [component],
            *extra_inputs,
            w0=0.6,
            **fit_params,
        )
        assert profile.shape == q_values.shape
        assert np.all(np.isfinite(profile))

        module.q_values = q_values
        module.experimental_intensities = np.asarray(profile, dtype=float)
        module.theoretical_intensities = [component]
        module.solvent_intensities = np.asarray(solvent, dtype=float)
        for input_name, values in template_runtime_inputs.items():
            setattr(module, input_name, np.asarray(values, dtype=float))

        dream_params = np.asarray(
            [0.6]
            + [
                fit_params[parameter.name]
                for parameter in template_spec.parameters
            ],
            dtype=float,
        )
        log_likelihood = getattr(module, template_spec.dream_model_name)(
            dream_params
        )
        assert np.isfinite(float(log_likelihood))


def test_poly_lma_runtime_bundle_includes_cluster_geometry_inputs(tmp_path):
    project_dir, _paths, _effective_radius = _build_poly_lma_geometry_project(
        tmp_path
    )
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.compute_cluster_geometry_table()
    prefit.set_cluster_geometry_active_radii_type("bond_length")
    updated_rows = prefit.cluster_geometry_rows()
    updated_rows[0].sf_approximation = "ellipsoid"
    prefit.set_cluster_geometry_rows(updated_rows)
    active_row = prefit.cluster_geometry_rows()[0]
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))

    assert metadata["template_name"] == POLY_LMA_HS_MIX_TEMPLATE
    assert metadata["lmfit_extra_inputs"] == ["effective_radii"]
    assert metadata["template_runtime_inputs"]["effective_radii"] == (
        pytest.approx([active_row.effective_radius])
    )
    assert metadata["cluster_geometry_metadata"]["active_radii_type"] == (
        "bond_length"
    )
    assert (
        metadata["cluster_geometry_metadata"]["rows"][0]["sf_approximation"]
        == "ellipsoid"
    )

    active_count = sum(1 for entry in entries if entry.vary)
    np.save(
        bundle.run_dir / "dream_sampled_params.npy",
        np.zeros((2, 3, active_count), dtype=float),
    )
    np.save(
        bundle.run_dir / "dream_log_ps.npy",
        np.zeros((2, 3), dtype=float),
    )
    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)
    assert loader.cluster_geometry_table is not None
    assert loader.cluster_geometry_table.active_radii_type == "bond_length"
    assert loader.cluster_geometry_table.rows[0].sf_approximation == (
        "ellipsoid"
    )


def test_stoichiometry_compensator_template_runtime_and_results(tmp_path):
    project_dir, _paths = _build_stoichiometry_compensator_saxs_project(
        tmp_path
    )
    prefit = SAXSPrefitWorkflow(project_dir)
    runtime_inputs = prefit.template_runtime_inputs_payload()

    assert runtime_inputs[STOICH_TARGET_RATIO_INPUT] == pytest.approx(
        [1.0, 3.0]
    )
    np.testing.assert_allclose(
        runtime_inputs[STOICH_COMPONENT_COUNTS_INPUT],
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float),
    )
    assert runtime_inputs[STOICH_COMPENSATOR_MASK_INPUT] == pytest.approx(
        [1.0, 0.0]
    )
    assert runtime_inputs[STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT] == (
        pytest.approx([0.2, 0.0])
    )
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    entries_by_param = {entry.param: entry for entry in entries}

    assert entries_by_param["w0"].vary is False
    assert entries_by_param["w1"].vary is True

    bundle = workflow.create_runtime_bundle(entries=entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    assert "w0" not in [
        entry["param"] for entry in metadata["active_parameter_entries"]
    ]

    active_values = np.asarray(
        [entry["value"] for entry in metadata["active_parameter_entries"]],
        dtype=float,
    )
    active_values[
        [
            entry["param"] for entry in metadata["active_parameter_entries"]
        ].index("w1")
    ] = 0.5
    np.save(
        bundle.run_dir / "dream_sampled_params.npy",
        active_values.reshape(1, 1, -1),
    )
    np.save(
        bundle.run_dir / "dream_log_ps.npy",
        np.zeros((1, 1), dtype=float),
    )

    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)
    expanded = loader.expand_params(active_values)
    full_names = metadata["full_parameter_names"]
    assert expanded[full_names.index("w0")] == pytest.approx(0.5)

    summary = loader.get_summary(
        stoichiometry_target_elements_text="Pb, I",
        stoichiometry_target_ratio_text="1:3",
        stoichiometry_filter_enabled=True,
        stoichiometry_tolerance_percent=0.01,
    )
    assert summary.stoichiometry_evaluation is not None
    assert summary.stoichiometry_evaluation.is_valid is True
    assert summary.stoichiometry_evaluation.observed_ratio == pytest.approx(
        (1.0, 3.0)
    )
    assert summary.posterior_candidate_sample_count == 1


def test_runtime_bundles_created_back_to_back_use_unique_run_directories(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)
    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map()

    first_bundle = workflow.create_runtime_bundle(entries=entries)
    second_bundle = workflow.create_runtime_bundle(entries=entries)

    assert first_bundle.run_dir != second_bundle.run_dir
    assert first_bundle.run_dir.is_dir()
    assert second_bundle.run_dir.is_dir()
    assert first_bundle.settings_path.is_file()
    assert second_bundle.settings_path.is_file()


def test_dream_batch_run_set_creates_runtime_bundles_and_commands(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="preset sweep")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()
    settings.run_label = "strict_priors"
    settings.niterations = 3
    settings.nseedchains = 20

    item = manager.add_queue_item(
        label="strict priors",
        settings=settings,
        entries=entries,
    )
    filter_settings = PosteriorFilterSettings.from_run_settings(settings)
    filter_settings.posterior_filter_mode = "top_n_logp"
    filter_settings.posterior_top_n = 2
    manager.add_filter_set(label="top n", settings=filter_settings)
    script_path, commands_path = manager.generate_shell_script()

    loaded = load_dream_batch_manifest(manager.run_set.manifest_path)
    manifest_payload = json.loads(
        manager.run_set.manifest_path.read_text(encoding="utf-8")
    )
    queue_settings_payload = json.loads(
        Path(item.settings_path).read_text(encoding="utf-8")
    )
    queue_metadata_payload = json.loads(
        Path(item.metadata_path).read_text(encoding="utf-8")
    )
    commands_text = commands_path.read_text(encoding="utf-8")

    assert commands_text.startswith("# DREAM backend run set commands")
    assert loaded.queue_items[0].label == "strict priors"
    assert loaded.filter_sets[0].label == "top n"
    assert loaded.filter_sets[0].settings.posterior_top_n == 2
    stored_filter = manifest_payload["filter_sets"][0]
    assert "posterior_filter_settings" in stored_filter
    assert "settings" not in stored_filter
    assert "niterations" not in stored_filter["posterior_filter_settings"]
    assert "niterations" in queue_settings_payload
    assert "posterior_filter_mode" not in queue_settings_payload
    assert "posterior_top_percent" not in queue_settings_payload
    assert "posterior_filter_mode" not in queue_metadata_payload["settings"]
    assert Path(item.run_dir).is_dir()
    assert Path(item.runtime_script_path).is_file()
    assert script_path.is_file()
    assert "conda activate 'saxshell-py312'" in script_path.read_text(
        encoding="utf-8"
    )
    assert "# Start batch process:" in commands_text
    assert "tail -f" in commands_text
    assert "kill -STOP" in commands_text
    assert "kill -CONT" in commands_text
    assert "# Check current status:" in commands_text
    assert "No live DREAM batch process" in commands_text
    assert "Comparison report:" in commands_text
    assert "Fit report PDF:" in commands_text


def test_dream_batch_queue_item_fit_range_update_rewrites_runtime_metadata(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="fit range edit")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    item = manager.add_queue_item(
        label="editable range",
        settings=manager.workflow.load_settings(),
        entries=entries,
    )

    manager.update_queue_item_fit_range(
        item.item_id,
        fit_q_min=0.11,
        fit_q_max=0.21,
    )

    saved_settings = load_dream_settings(item.settings_path)
    metadata = json.loads(Path(item.metadata_path).read_text(encoding="utf-8"))
    fit_mask = np.asarray(metadata["output_fit_mask"], dtype=bool)

    assert saved_settings.fit_q_min == pytest.approx(0.11)
    assert saved_settings.fit_q_max == pytest.approx(0.21)
    assert metadata["settings"]["fit_q_min"] == pytest.approx(0.11)
    assert metadata["settings"]["fit_q_max"] == pytest.approx(0.21)
    assert metadata["prefit_fit_q_range"]["q_min"] == pytest.approx(0.11)
    assert metadata["prefit_fit_q_range"]["q_max"] == pytest.approx(0.21)
    assert np.any(fit_mask)
    assert not np.all(fit_mask)


def test_dream_batch_queue_items_keep_prefit_weight_toggles_per_item(tmp_path):
    project_dir, _paths = _build_two_component_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="weight toggles")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()
    disabled_prefit_entries = [
        PrefitParameterEntry.from_dict(entry.to_dict())
        for entry in manager.workflow.prefit_workflow.parameter_entries
    ]
    for entry in disabled_prefit_entries:
        if entry.name == "w0":
            entry.active = False

    disabled_item = manager.add_queue_item(
        label="w0 off",
        settings=settings,
        entries=entries,
        prefit_parameter_entries=disabled_prefit_entries,
    )
    enabled_item = manager.add_queue_item(
        label="all weights on",
        settings=settings,
        entries=entries,
        prefit_parameter_entries=(
            manager.workflow.prefit_workflow.parameter_entries
        ),
    )

    disabled_metadata = json.loads(
        Path(disabled_item.metadata_path).read_text(encoding="utf-8")
    )
    enabled_metadata = json.loads(
        Path(enabled_item.metadata_path).read_text(encoding="utf-8")
    )
    disabled_params = [
        entry["param"] for entry in disabled_metadata["parameter_map"]
    ]
    enabled_params = [
        entry["param"] for entry in enabled_metadata["parameter_map"]
    ]

    assert "w0" not in disabled_metadata["full_parameter_names"]
    assert "w1" in disabled_metadata["full_parameter_names"]
    assert "w0" not in disabled_params
    assert "w1" in disabled_params
    assert "w0" in enabled_metadata["full_parameter_names"]
    assert "w1" in enabled_metadata["full_parameter_names"]
    assert [name for name in enabled_params if name.startswith("w")] == [
        "w0",
        "w1",
    ]

    manager.update_queue_item_active_weights(
        disabled_item.item_id,
        active_weight_names={"w0"},
    )
    edited_metadata = json.loads(
        Path(disabled_item.metadata_path).read_text(encoding="utf-8")
    )
    edited_params = [
        entry["param"] for entry in edited_metadata["parameter_map"]
    ]
    edited_state = json.loads(
        (Path(disabled_item.run_dir) / "prefit_state.json").read_text(
            encoding="utf-8"
        )
    )
    edited_active = {
        entry["name"]: bool(entry.get("active", True))
        for entry in edited_state["parameter_entries"]
        if str(entry.get("name", "")).startswith("w")
    }
    assert "w0" in edited_metadata["full_parameter_names"]
    assert "w1" not in edited_metadata["full_parameter_names"]
    assert [name for name in edited_params if name.startswith("w")] == ["w0"]
    assert edited_active["w0"] is True
    assert edited_active["w1"] is False


def test_dream_batch_queue_items_generate_labels_when_empty(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="auto labels")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()
    settings.run_label = "dream"

    first = manager.add_queue_item(
        label="",
        settings=settings,
        entries=entries,
    )
    second = manager.add_queue_item(
        label="",
        settings=settings,
        entries=entries,
    )

    assert first.label == "DREAM queue item 1"
    assert second.label == "DREAM queue item 2"
    assert json.loads(Path(first.settings_path).read_text())["run_label"] == (
        "DREAM queue item 1"
    )
    assert (
        load_dream_batch_manifest(manager.run_set.manifest_path)
        .queue_items[1]
        .label
        == "DREAM queue item 2"
    )


def test_dream_batch_remove_queue_item_deletes_queued_runtime_bundle(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="remove queued")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()

    item = manager.add_queue_item(
        label="remove me",
        settings=settings,
        entries=entries,
    )
    run_dir = Path(item.run_dir)

    assert run_dir.is_dir()

    removed = manager.remove_queue_item(item.item_id)

    assert removed.label == "remove me"
    assert not run_dir.exists()
    assert manager.run_set.queue_items == []
    assert (
        load_dream_batch_manifest(manager.run_set.manifest_path).queue_items
        == []
    )


def test_dream_batch_manifest_runner_executes_queue_and_reports(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="runner")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()
    settings.niterations = 3
    settings.nseedchains = 20
    settings.search_filter_preset = "legacy_gui_default"
    for entry in entries:
        entry.smart_preset_status = "proportional"
    entries[0].smart_preset_status = "legacy_md_weights"
    item = manager.add_queue_item(
        label="runtime item",
        settings=settings,
        entries=entries,
    )
    second_item = manager.add_queue_item(
        label="runtime item 2",
        settings=settings,
        entries=entries,
    )
    filter_settings = PosteriorFilterSettings.from_run_settings(settings)
    filter_settings.posterior_filter_mode = "top_percent_logp"
    filter_settings.posterior_top_percent = 50.0
    failing_filter_settings = PosteriorFilterSettings.from_run_settings(
        settings
    )
    failing_filter_settings.stoichiometry_filter_enabled = True
    failing_filter_settings.stoichiometry_target_elements_text = "Pb, I"
    failing_filter_settings.stoichiometry_target_ratio_text = "999:1"
    failing_filter_settings.stoichiometry_tolerance_percent = 0.0
    manager.add_filter_set(
        label="impossible stoich",
        settings=failing_filter_settings,
    )
    manager.add_filter_set(label="top half", settings=filter_settings)
    manager.generate_shell_script()

    def fake_run_bundle(self, bundle, **kwargs):
        del self
        output_callback = kwargs.get("output_callback")
        metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
        active_values = np.asarray(
            [entry["value"] for entry in metadata["active_parameter_entries"]],
            dtype=float,
        )
        np.save(
            bundle.run_dir / "dream_sampled_params.npy",
            np.tile(active_values, (2, 3, 1)),
        )
        np.save(bundle.run_dir / "dream_log_ps.npy", np.zeros((2, 3)))
        with manager.run_set.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"fake DREAM fit log line for {bundle.run_dir.name}\n"
            )
        if output_callback is not None:
            output_callback(
                f"fake DREAM runtime finished {bundle.run_dir.name}"
            )
        return {
            "sampled_params_path": str(
                bundle.run_dir / "dream_sampled_params.npy"
            ),
            "log_ps_path": str(bundle.run_dir / "dream_log_ps.npy"),
        }

    monkeypatch.setattr(
        "saxshell.saxs.dream.batch.SAXSDreamWorkflow.run_bundle",
        fake_run_bundle,
    )

    messages: list[str] = []
    completed = run_dream_batch_manifest(
        manager.run_set.manifest_path,
        output_callback=messages.append,
    )
    reloaded = load_dream_batch_manifest(manager.run_set.manifest_path)

    assert completed.queue_items[0].status == "completed"
    assert completed.queue_items[1].status == "completed"
    assert reloaded.queue_items[0].status == "completed"
    assert reloaded.queue_items[1].status == "completed"
    assert any(
        message.startswith("fake DREAM runtime finished")
        for message in messages
    )
    completed_fit_indexes = [
        index
        for index, message in enumerate(messages)
        if message.startswith("Completed DREAM fit")
    ]
    filter_indexes = [
        index
        for index, message in enumerate(messages)
        if (
            message.startswith("Skipped filter set")
            or message.startswith("Applied filter set")
        )
    ]
    assert len(completed_fit_indexes) == 2
    assert filter_indexes
    assert max(completed_fit_indexes) < min(filter_indexes)
    assert any(
        message.startswith("Skipped filter set impossible stoich")
        for message in messages
    )
    assert (Path(item.run_dir) / "dream_statistics_top_half.txt").is_file()
    assert (
        Path(second_item.run_dir) / "dream_statistics_top_half.txt"
    ).is_file()
    assert not (
        Path(item.run_dir) / "dream_statistics_impossible_stoich.txt"
    ).exists()
    comparison_report = manager.run_set.comparison_report_path
    assert comparison_report.is_file()
    comparison_text = comparison_report.read_text(encoding="utf-8")
    assert "Run/filter matrix:" in comparison_text
    assert "impossible stoich" in comparison_text
    assert "report_failed" in comparison_text
    assert "top half" in comparison_text
    assert (
        "DREAM search preset: Legacy GUI Default (legacy_gui_default)"
        in comparison_text
    )
    assert "Legacy MD Weights: 1 parameter" in comparison_text
    assert "Proportional:" in comparison_text
    assert "fit_q_range" in comparison_text
    assert "output_q_range" in comparison_text
    assert "report_index" in comparison_text
    assert "fit_index" in comparison_text
    assert "filter_subindex" in comparison_text
    assert "1.2" in comparison_text
    assert "Best-fit parameter guide bounds:" in comparison_text
    assert "guide_low" in comparison_text
    assert "guide_high" in comparison_text
    statistics_text = (
        Path(item.run_dir) / "dream_statistics_top_half.txt"
    ).read_text(encoding="utf-8")
    assert (
        "DREAM search preset: Legacy GUI Default (legacy_gui_default)"
        in statistics_text
    )
    assert "Legacy MD Weights: 1 parameter" in statistics_text
    assert "Fit q-range:" in statistics_text
    assert "Output q-range:" in statistics_text
    assert "guide_low=" in statistics_text
    assert "guide_high=" in statistics_text
    assert any(
        "DREAM batch comparison report:" in message for message in messages
    )
    fit_report_pdf = manager.run_set.fit_report_pdf_path
    assert fit_report_pdf.is_file()
    assert fit_report_pdf.read_bytes().startswith(b"%PDF")
    assert fit_report_pdf.stat().st_size > 1000
    assert any(
        "DREAM batch fit PDF report:" in message for message in messages
    )
    assert json.loads(manager.run_set.status_path.read_text())["status"] == (
        "completed"
    )


def test_dream_batch_pdf_model_fit_omits_solvent_curve():
    captured_figures = []

    class CapturePdf:
        def savefig(self, figure):
            captured_figures.append(figure)

    item = SimpleNamespace(label="runtime item", run_dir="/tmp/dream_run")
    filter_set = SimpleNamespace(label="top half")
    summary = SimpleNamespace(
        posterior_filter_mode="top_percent_logp",
        posterior_sample_count=3,
        posterior_candidate_sample_count=6,
        map_chain=0,
        map_step=2,
        stoichiometry_evaluation=None,
    )
    model_data = SimpleNamespace(
        q_values=np.asarray([0.05, 0.1, 0.2], dtype=float),
        experimental_intensities=np.asarray([1.0, 0.8, 0.6], dtype=float),
        model_intensities=np.asarray([0.95, 0.82, 0.61], dtype=float),
        solvent_contribution=np.asarray([100.0, 120.0, 150.0], dtype=float),
        bestfit_method="map",
        rmse=0.04,
        mean_abs_residual=0.03,
        r_squared=0.98,
        active_fit_mask=np.asarray([False, True, True], dtype=bool),
    )

    dream_batch_module._add_pdf_model_fit_page(
        CapturePdf(),
        item,
        filter_set,
        summary,
        model_data,
        report_index="1.2",
    )

    assert captured_figures
    assert captured_figures[0]._suptitle.get_text().startswith("[1.2]")
    top_axis = captured_figures[0].axes[0]
    labels = top_axis.get_legend_handles_labels()[1]
    assert "Experimental" in labels
    assert "Best fit" in labels
    assert "Fit q-range" in labels
    assert "Solvent contribution" not in labels
    assert top_axis.patches
    assert top_axis.patches[0].get_facecolor() == pytest.approx(
        to_rgba("#fef3c7", alpha=0.35)
    )


def test_dream_batch_parameter_summary_marks_guide_clipping(tmp_path):
    captured_figures = []

    class CapturePdf:
        def savefig(self, figure):
            captured_figures.append(figure)

    map_path = tmp_path / "parameter_map.json"
    save_parameter_map(
        map_path,
        [
            DreamParameterEntry(
                structure="A",
                motif="no_motif",
                param_type="Both",
                param="w0",
                value=0.5,
                vary=True,
                distribution="uniform",
                dist_params={"loc": 0.0, "scale": 1.0},
                smart_preset_status="proportional",
            )
        ],
    )
    item = SimpleNamespace(
        label="runtime item",
        parameter_map_path=str(map_path),
    )
    filter_set = SimpleNamespace(label="top half")
    summary = SimpleNamespace(
        full_parameter_names=["w0"],
        bestfit_params=np.asarray([1.0], dtype=float),
        map_params=np.asarray([1.0], dtype=float),
        chain_mean_params=np.asarray([0.9], dtype=float),
        median_params=np.asarray([0.95], dtype=float),
        interval_low_values=np.asarray([0.8], dtype=float),
        interval_high_values=np.asarray([1.0], dtype=float),
        credible_interval_low=16.0,
        credible_interval_high=84.0,
    )
    model_data = SimpleNamespace(
        fit_parameters=[
            SimpleNamespace(
                name="w0",
                varied=True,
                structure="A",
                motif="no_motif",
                param_type="Both",
            )
        ]
    )

    dream_batch_module._add_pdf_parameter_summary_pages(
        CapturePdf(),
        item,
        filter_set,
        summary,
        model_data,
    )

    assert captured_figures
    table = captured_figures[0].axes[0].tables[0]
    headers = [
        table[(0, column)].get_text().get_text() for column in range(12)
    ]
    assert headers[5:9] == [
        "Selected",
        "Guide Low",
        "Guide High",
        "Guide Clip",
    ]
    assert table[(1, 6)].get_text().get_text() == "0"
    assert table[(1, 7)].get_text().get_text() == "1"
    assert table[(1, 8)].get_text().get_text() == "Guide High"
    assert table[(1, 5)].get_facecolor() == pytest.approx(to_rgba("#ffedd5"))
    assert table[(1, 7)].get_facecolor() == pytest.approx(to_rgba("#ffedd5"))


def test_dream_batch_pdf_violin_marks_selected_model_values():
    captured_figures = []

    class CapturePdf:
        def savefig(self, figure):
            captured_figures.append(figure)

    item = SimpleNamespace(label="runtime item", run_dir="/tmp/dream_run")
    filter_set = SimpleNamespace(label="top half")
    summary = SimpleNamespace(
        full_parameter_names=["w0", "w1", "radius"],
        bestfit_params=np.asarray([0.25, 0.75, 42.0], dtype=float),
    )
    violin_data = SimpleNamespace(
        parameter_names=["radius", "w0"],
        display_names=["radius", "w0"],
        samples=np.asarray(
            [
                [40.0, 0.2],
                [42.0, 0.25],
                [44.0, 0.3],
            ],
            dtype=float,
        ),
        mode="fit_parameters",
        sample_source="filtered_posterior",
        sample_count=3,
    )

    dream_batch_module._add_pdf_violin_page(
        CapturePdf(),
        item,
        filter_set,
        summary,
        violin_data,
    )

    assert captured_figures
    axis = captured_figures[0].axes[0]
    red_offsets = []
    for collection in axis.collections:
        facecolors = collection.get_facecolors()
        if facecolors.size == 0:
            continue
        if np.allclose(facecolors[0][:3], [214 / 255, 39 / 255, 40 / 255]):
            red_offsets.extend(collection.get_offsets().tolist())

    assert np.asarray(red_offsets, dtype=float).shape == (2, 2)
    np.testing.assert_allclose(
        np.asarray(red_offsets, dtype=float),
        np.asarray([[1.0, 42.0], [2.0, 0.25]], dtype=float),
    )
    assert "Selected model value" in axis.get_legend_handles_labels()[1]


def test_dream_batch_pdf_table_of_contents_links_fit_types(tmp_path):
    captured_figures = []

    class CapturePdf:
        def savefig(self, figure):
            captured_figures.append(figure)

    report_path = tmp_path / "dream_batch_fit_report.pdf"
    entries = [
        dream_batch_module._PdfTocEntry(
            section="Fit",
            label="runtime item / top half",
            detail="best fit: map | posterior: top_percent_logp",
            page_number=7,
        )
    ]

    dream_batch_module._add_pdf_table_of_contents_pages(
        CapturePdf(),
        report_path,
        entries,
    )

    text = "\n".join(
        artist.get_text()
        for figure in captured_figures
        for artist in figure.texts
    )
    urls = [
        artist.get_url()
        for figure in captured_figures
        for artist in figure.texts
        if artist.get_url()
    ]

    assert "Table of Contents" in text
    assert "runtime item / top half" in text
    assert "best fit: map | posterior: top_percent_logp" in text
    assert "Appendix A: DREAM Fit Log" not in text
    assert f"{report_path.resolve().as_uri()}#page=7" in urls


def test_dream_batch_pdf_prior_distribution_params_are_compact():
    text = dream_batch_module._format_distribution_params(
        {
            "scale": 0.12345678912345678,
            "loc": 1.2345678912345678e-6,
            "s": 0.3333333333333333,
        }
    )

    assert text == "loc=1.2346e-06, scale=0.12346, s=0.33333"
    assert "0.333333333333" not in text
    assert "{" not in text
    assert len(text) < 50


def test_dream_batch_manifest_runner_logs_failure_status(
    tmp_path,
    monkeypatch,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    manager = DreamBatchRunSetManager(project_dir, label="runner failure")
    entries = manager.workflow.create_default_parameter_map(persist=False)
    settings = manager.workflow.load_settings()
    manager.add_queue_item(
        label="runtime item",
        settings=settings,
        entries=entries,
    )

    def fake_run_bundle(self, bundle, **kwargs):
        del self, bundle, kwargs
        raise ValueError("synthetic DREAM failure")

    monkeypatch.setattr(
        "saxshell.saxs.dream.batch.SAXSDreamWorkflow.run_bundle",
        fake_run_bundle,
    )

    messages: list[str] = []
    completed = run_dream_batch_manifest(
        manager.run_set.manifest_path,
        output_callback=messages.append,
    )
    reloaded = load_dream_batch_manifest(manager.run_set.manifest_path)
    status_payload = json.loads(
        manager.run_set.status_path.read_text(encoding="utf-8")
    )

    assert completed.queue_items[0].status == "failed"
    assert reloaded.queue_items[0].status == "failed"
    assert reloaded.queue_items[0].error == "synthetic DREAM failure"
    assert status_payload["status"] == "completed_with_errors"
    assert status_payload["error"] == "1 DREAM queue item(s) failed."
    assert any(
        "DREAM fit failed for runtime item: synthetic DREAM failure" in message
        for message in messages
    )
    assert any(
        "DREAM backend run set complete with 1 failed queue item(s)."
        in message
        for message in messages
    )


def test_dream_default_prior_distribution_policy(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    entries_by_param = {entry.param: entry for entry in entries}
    weight_entry = next(entry for entry in entries if entry.param == "w0")
    prefit_weight = next(
        entry
        for entry in workflow.prefit_workflow.parameter_entries
        if entry.name == "w0"
    )
    prefit_entries = {
        entry.name: entry
        for entry in workflow.prefit_workflow.parameter_entries
    }

    assert weight_entry.distribution == "lognorm"
    assert weight_entry.dist_params["loc"] == pytest.approx(0.0)
    assert weight_entry.dist_params["scale"] == pytest.approx(
        max(prefit_weight.value, 1e-6)
    )
    assert weight_entry.dist_params["s"] > 0.0
    assert entries_by_param["scale"].distribution == "uniform"
    assert entries_by_param["scale"].dist_params["loc"] == pytest.approx(
        prefit_entries["scale"].minimum
    )
    assert entries_by_param["scale"].dist_params["scale"] == pytest.approx(
        prefit_entries["scale"].maximum - prefit_entries["scale"].minimum
    )
    assert entries_by_param["offset"].distribution == "uniform"
    assert entries_by_param["offset"].dist_params["loc"] == pytest.approx(
        prefit_entries["offset"].minimum
    )
    assert entries_by_param["offset"].dist_params["scale"] == pytest.approx(
        prefit_entries["offset"].maximum - prefit_entries["offset"].minimum
    )
    assert entries_by_param["eff_r"].distribution == "lognorm"
    assert entries_by_param["vol_frac"].distribution == "norm"


def test_dream_default_weight_priors_vary_when_prefit_weight_is_fixed(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    for entry in prefit.parameter_entries:
        if entry.name == "w0":
            entry.vary = False
            break
    prefit.save_parameter_state(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    weight_entry = next(entry for entry in entries if entry.param == "w0")

    assert weight_entry.vary is True


def test_inactive_prefit_weight_is_excluded_from_dream_without_renumbering(
    tmp_path,
):
    project_dir, _paths = _build_two_component_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    entries = prefit.parameter_entries
    for entry in entries:
        if entry.name == "w0":
            entry.active = False
    prefit.save_parameter_state(entries)

    workflow = SAXSDreamWorkflow(project_dir)
    default_entries = workflow.create_default_parameter_map(persist=False)
    default_params = [entry.param for entry in default_entries]
    assert "w0" not in default_params
    assert "w1" in default_params

    bundle = workflow.create_runtime_bundle(entries=default_entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    weight_components = [
        entry["param"]
        for entry in metadata["parameter_components"]
        if entry["category"] == "weight"
    ]

    assert weight_components == ["w1"]
    assert "w0" not in metadata["full_parameter_names"]
    assert "w1" in metadata["full_parameter_names"]
    assert len(metadata["theoretical_intensities"]) == 1

    for entry in entries:
        if entry.name == "w0":
            entry.active = True
    prefit.save_parameter_state(entries)

    re_enabled = SAXSDreamWorkflow(project_dir).create_default_parameter_map(
        persist=False
    )
    assert [
        entry.param for entry in re_enabled if entry.param.startswith("w")
    ] == [
        "w0",
        "w1",
    ]


def test_dream_parameter_map_tracks_weight_priors_by_component(tmp_path):
    project_dir, _paths = _build_two_component_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    defaults = workflow.create_default_parameter_map(persist=False)
    default_weights = {
        entry.param: entry for entry in defaults if entry.param.startswith("w")
    }
    assert default_weights["w0"].structure == "PbI3"
    assert default_weights["w1"].structure == "PbI4"

    shifted_weight_priors = [
        DreamParameterEntry(
            structure="PbI4",
            motif="no_motif",
            param_type="Both",
            param="w0",
            value=0.6,
            vary=True,
            distribution="uniform",
            dist_params={"loc": 0.5, "scale": 0.2},
        ),
        DreamParameterEntry(
            structure="PbI3",
            motif="no_motif",
            param_type="Both",
            param="w1",
            value=0.4,
            vary=True,
            distribution="norm",
            dist_params={"loc": 0.4, "scale": 0.03},
        ),
    ]
    save_parameter_map(
        workflow.parameter_map_path,
        [
            *shifted_weight_priors,
            *[entry for entry in defaults if not entry.param.startswith("w")],
        ],
    )

    normalized = workflow.load_parameter_map()
    normalized_weights = {
        entry.param: entry
        for entry in normalized
        if entry.param.startswith("w")
    }

    assert normalized_weights["w0"].structure == "PbI3"
    assert normalized_weights["w0"].distribution == "norm"
    assert normalized_weights["w0"].value == pytest.approx(
        default_weights["w0"].value
    )
    assert normalized_weights["w0"].dist_params["loc"] == pytest.approx(
        default_weights["w0"].value
    )
    assert normalized_weights["w0"].dist_params["scale"] == pytest.approx(0.03)
    assert normalized_weights["w1"].structure == "PbI4"
    assert normalized_weights["w1"].distribution == "uniform"
    assert normalized_weights["w1"].value == pytest.approx(
        default_weights["w1"].value
    )
    assert normalized_weights["w1"].dist_params["loc"] == pytest.approx(
        default_weights["w1"].value - 0.1
    )
    assert normalized_weights["w1"].dist_params["scale"] == pytest.approx(0.2)


def test_prefit_locks_custom_project_setup_weight_order_for_dream(tmp_path):
    project_dir, paths = _build_two_component_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.prior_histogram_x_axis_order = [
        ["PbI4", "PbI4"],
        ["PbI3", "PbI3"],
    ]
    manager.save_project(settings)

    prefit = SAXSPrefitWorkflow(project_dir)
    weight_components = [
        (component.param_name, component.structure, component.weight_value)
        for component in prefit.components
    ]

    assert weight_components == [
        ("w0", "PbI4", pytest.approx(0.6)),
        ("w1", "PbI3", pytest.approx(0.4)),
    ]

    map_payload = json.loads(
        (paths.project_dir / "md_saxs_map.json").read_text(encoding="utf-8")
    )
    prior_payload = json.loads(
        (paths.project_dir / "md_prior_weights.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        entry["structure"] for entry in map_payload["component_order"]
    ] == ["PbI4", "PbI3"]
    assert [
        entry["structure"] for entry in prior_payload["component_order"]
    ] == ["PbI4", "PbI3"]
    assert list(map_payload["saxs_map"]) == ["PbI4", "PbI3"]

    settings = manager.load_project(project_dir)
    settings.prior_histogram_x_axis_order = [
        ["PbI3", "PbI3"],
        ["PbI4", "PbI4"],
    ]
    manager.save_project(settings)

    reloaded_prefit = SAXSPrefitWorkflow(project_dir)
    assert [
        (component.param_name, component.structure)
        for component in reloaded_prefit.components
    ] == [("w0", "PbI4"), ("w1", "PbI3")]

    workflow = SAXSDreamWorkflow(project_dir)
    dream_entries = workflow.create_default_parameter_map(persist=False)
    dream_weights = [
        (entry.param, entry.structure, entry.value)
        for entry in dream_entries
        if entry.param.startswith("w")
    ]
    assert dream_weights == [
        ("w0", "PbI4", pytest.approx(0.6)),
        ("w1", "PbI3", pytest.approx(0.4)),
    ]

    bundle = workflow.create_runtime_bundle(entries=dream_entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    weight_components = [
        (entry["param"], entry["structure"])
        for entry in metadata["parameter_components"]
        if entry["category"] == "weight"
    ]
    assert weight_components == [("w0", "PbI4"), ("w1", "PbI3")]


def test_dream_workflow_uses_predicted_structure_weights_when_enabled(
    tmp_path,
):
    project_dir, paths = _build_minimal_saxs_project(tmp_path)
    _write_predicted_structure_artifacts(paths)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.use_predicted_structure_weights = True
    manager.save_project(settings)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    predicted_entries = [
        entry
        for entry in entries
        if entry.structure == "A2"
        and entry.motif == "predicted_rank01"
        and entry.param_type == "Both"
    ]

    assert workflow.prefit_workflow.component_dir == (
        paths.predicted_scattering_components_dir
    )
    assert predicted_entries
    assert predicted_entries[0].value == pytest.approx(0.25)


def test_runtime_bundle_uses_reduced_saved_q_range_without_rebuild(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.q_min = 0.12
    settings.q_max = 0.19
    manager.save_project(settings)

    workflow = SAXSDreamWorkflow(project_dir)
    bundle = workflow.create_runtime_bundle()
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))

    assert np.allclose(
        np.asarray(metadata["q_values"], dtype=float),
        [0.12142857142857144, 0.15714285714285714, 0.19285714285714284],
    )
    assert len(metadata["experimental_intensities"]) == 3
    assert len(metadata["solvent_intensities"]) == 3
    assert len(metadata["theoretical_intensities"]) == 1
    assert len(metadata["theoretical_intensities"][0]) == 3


def test_runtime_bundle_uses_active_prefit_fit_q_range_without_rebuild(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.prefit_fit_q_min = 0.11
    settings.prefit_fit_q_max = 0.21
    manager.save_project(settings)

    workflow = SAXSDreamWorkflow(project_dir)
    bundle = workflow.create_runtime_bundle()
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))

    assert np.allclose(
        np.asarray(metadata["q_values"], dtype=float),
        [0.12142857142857144, 0.15714285714285714, 0.19285714285714284],
    )
    assert metadata["prefit_fit_q_range"]["q_min"] == pytest.approx(0.11)
    assert metadata["prefit_fit_q_range"]["q_max"] == pytest.approx(0.21)
    assert metadata["prefit_fit_q_range"]["model_q_min"] == pytest.approx(0.05)
    assert metadata["prefit_fit_q_range"]["model_q_max"] == pytest.approx(0.3)
    assert len(metadata["experimental_intensities"]) == 3
    assert len(metadata["solvent_intensities"]) == 3
    assert len(metadata["theoretical_intensities"]) == 1
    assert len(metadata["theoretical_intensities"][0]) == 3
    assert np.allclose(
        np.asarray(metadata["output_q_values"], dtype=float),
        np.linspace(0.05, 0.3, 8),
    )
    assert np.asarray(metadata["output_fit_mask"], dtype=bool).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert len(metadata["output_experimental_intensities"]) == 8
    assert len(metadata["output_solvent_intensities"]) == 8
    assert len(metadata["output_theoretical_intensities"][0]) == 8


def test_runtime_bundle_uses_dream_fit_q_range_override(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    manager = SAXSProjectManager()
    settings = manager.load_project(project_dir)
    settings.prefit_fit_q_min = 0.11
    settings.prefit_fit_q_max = 0.21
    manager.save_project(settings)

    workflow = SAXSDreamWorkflow(project_dir)
    bundle = workflow.create_runtime_bundle(
        settings=DreamRunSettings(fit_q_min=0.15, fit_q_max=0.26)
    )
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    saved_settings = manager.load_project(project_dir)

    assert np.allclose(
        np.asarray(metadata["q_values"], dtype=float),
        [0.15714285714285714, 0.19285714285714284, 0.2285714285714286],
    )
    assert metadata["prefit_fit_q_range"]["q_min"] == pytest.approx(0.15)
    assert metadata["prefit_fit_q_range"]["q_max"] == pytest.approx(0.26)
    assert metadata["settings"]["fit_q_min"] == pytest.approx(0.15)
    assert metadata["settings"]["fit_q_max"] == pytest.approx(0.26)
    assert saved_settings.prefit_fit_q_min == pytest.approx(0.11)
    assert saved_settings.prefit_fit_q_max == pytest.approx(0.21)


def test_results_loader_displays_full_range_for_fit_subset(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    workflow = SAXSDreamWorkflow(project_dir)
    bundle = workflow.create_runtime_bundle(
        settings=DreamRunSettings(fit_q_min=0.11, fit_q_max=0.21)
    )
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    output_experimental = np.asarray(
        metadata["output_experimental_intensities"],
        dtype=float,
    )
    output_experimental[0] += 1.0e6
    output_experimental[-1] += 1.0e6
    metadata["output_experimental_intensities"] = output_experimental.tolist()
    bundle.metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    active_values = np.asarray(
        [entry["value"] for entry in metadata["active_parameter_entries"]],
        dtype=float,
    )
    np.save(
        bundle.run_dir / "dream_sampled_params.npy",
        np.tile(active_values, (2, 4, 1)),
    )
    np.save(
        bundle.run_dir / "dream_log_ps.npy",
        np.zeros((2, 4), dtype=float),
    )

    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)
    plot_data = loader.build_model_fit_data(bestfit_method="map")
    fit_mask = np.asarray(plot_data.active_fit_mask, dtype=bool)
    residuals = np.asarray(
        plot_data.model_intensities - plot_data.experimental_intensities,
        dtype=float,
    )
    expected_fit_rmse = float(np.sqrt(np.mean(residuals[fit_mask] ** 2)))

    assert np.allclose(plot_data.q_values, np.linspace(0.05, 0.3, 8))
    assert int(np.count_nonzero(fit_mask)) == 3
    assert len(plot_data.model_intensities) == 8
    assert plot_data.rmse == pytest.approx(expected_fit_rmse)
    assert abs(residuals[0]) > 1.0e5


def test_runtime_bundle_rejects_active_values_outside_prior_support(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    for entry in entries:
        entry.vary = entry.param == "w0"
    weight_entry = next(entry for entry in entries if entry.param == "w0")
    unsupported_weight_value = weight_entry.dist_params["loc"] - 0.1
    old_central_guide_low = float(
        weight_entry.dist_params["loc"]
        + weight_entry.dist_params["scale"]
        * np.exp(-3.0 * weight_entry.dist_params["s"])
    )
    guide_low = 0.0
    guide_high = float(
        weight_entry.dist_params["loc"]
        + weight_entry.dist_params["scale"]
        * np.exp(3.0 * weight_entry.dist_params["s"])
    )
    outside_guide_value = guide_high * 1.01
    below_old_central_guide_value = old_central_guide_low / 2.0
    bundle = workflow.create_runtime_bundle(entries=entries)

    module_name, module, added_sys_path = workflow._load_runtime_module(bundle)
    try:
        assert module.active_params_inside_prior_support([weight_entry.value])
        assert np.isfinite(
            float(module.active_log_likelihood([weight_entry.value]))
        )
        assert module.active_params_inside_prior_support(
            [below_old_central_guide_value]
        )
        assert not module.active_params_inside_prior_support(
            [unsupported_weight_value]
        )
        assert not module.active_params_inside_prior_support(
            [outside_guide_value]
        )
        assert (
            module.active_log_likelihood([unsupported_weight_value]) == -np.inf
        )
        assert module.active_log_likelihood([outside_guide_value]) == -np.inf
        sampled_parameter = module.build_sampled_parameters()[0]
        draws = np.asarray(
            [sampled_parameter.random() for _ in range(512)],
            dtype=float,
        ).reshape(-1)
        assert np.all(draws >= guide_low)
        assert np.all(draws <= guide_high)
        assert sampled_parameter.prior([outside_guide_value]) == -np.inf
    finally:
        workflow._unload_runtime_module(
            module_name,
            added_sys_path=added_sys_path,
        )


def test_runtime_bundle_rejects_negative_weight_values_for_unbounded_priors(
    tmp_path,
):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=False)
    for entry in entries:
        entry.vary = entry.param == "w0"
    weight_entry = next(entry for entry in entries if entry.param == "w0")
    weight_entry.distribution = "norm"
    weight_entry.dist_params = {"loc": 0.2, "scale": 0.5}
    bundle = workflow.create_runtime_bundle(entries=entries)

    module_name, module, added_sys_path = workflow._load_runtime_module(bundle)
    try:
        assert module.active_params_inside_prior_support([0.1])
        assert not module.active_params_inside_prior_support([-0.1])
        assert module.active_log_likelihood([-0.1]) == -np.inf
        sampled_parameter = module.build_sampled_parameters()[0]
        draws = np.asarray(
            [sampled_parameter.random() for _ in range(256)],
            dtype=float,
        ).reshape(-1)
        assert np.all(draws >= 0.0)
    finally:
        workflow._unload_runtime_module(
            module_name,
            added_sys_path=added_sys_path,
        )


def test_dream_runtime_bundle_smoke_executes_and_writes_outputs(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    settings = workflow.load_settings()
    settings.nchains = 5
    settings.niterations = 3
    settings.nseedchains = 20
    settings.parallel = False
    settings.verbose = False
    settings.adapt_crossover = False
    settings.crossover_burnin = 1000

    bundle = workflow.create_runtime_bundle(settings=settings, entries=entries)
    result = workflow.run_bundle(bundle)

    sampled_params_path = bundle.run_dir / "dream_sampled_params.npy"
    log_ps_path = bundle.run_dir / "dream_log_ps.npy"

    assert result["sampled_params_path"] == str(sampled_params_path)
    assert result["log_ps_path"] == str(log_ps_path)
    assert sampled_params_path.is_file()
    assert log_ps_path.is_file()

    sampled_params = np.load(sampled_params_path)
    log_ps = np.load(log_ps_path)

    assert sampled_params.shape[0:2] == (5, 3)
    assert log_ps.shape == (5, 3)
    assert np.all(np.isfinite(sampled_params))
    assert np.all(np.isfinite(log_ps))


def test_results_loader_uses_runtime_components_for_weight_labels(tmp_path):
    project_dir, _paths = _build_two_component_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    for entry in metadata["parameter_map"]:
        if entry["param"] == "w0":
            entry["structure"] = "PbI4"
        elif entry["param"] == "w1":
            entry["structure"] = "PbI3"
    bundle.metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    active_values = np.asarray(
        [entry["value"] for entry in metadata["active_parameter_entries"]],
        dtype=float,
    )
    np.save(
        bundle.run_dir / "dream_sampled_params.npy",
        np.tile(active_values, (2, 4, 1)),
    )
    np.save(
        bundle.run_dir / "dream_log_ps.npy",
        np.zeros((2, 4), dtype=float),
    )

    loader = SAXSDreamResultsLoader(bundle.run_dir, burnin_percent=0)
    violin_data = loader.build_violin_data(mode="weights_only")

    assert violin_data.parameter_names[:2] == ["w0", "w1"]
    assert violin_data.display_names[:2] == ["w0 (PbI3)", "w1 (PbI4)"]


def test_run_bundle_streams_runtime_output(tmp_path, monkeypatch):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    sampled_params_path = bundle.run_dir / "dream_sampled_params.npy"
    log_ps_path = bundle.run_dir / "dream_log_ps.npy"
    np.save(sampled_params_path, np.asarray([[[1.0]]], dtype=float))
    np.save(log_ps_path, np.asarray([[1.0]], dtype=float))

    streamed_lines: list[str] = []

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            del args
            assert kwargs["stdout"] is not None
            self.stdout = io.StringIO(
                "Starting DREAM sampler\nFinished DREAM sampler\n"
            )
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "saxshell.saxs.dream.runtime.subprocess.Popen",
        _FakePopen,
    )

    result = workflow.run_bundle(
        bundle,
        output_callback=streamed_lines.append,
    )

    assert streamed_lines == [
        "Starting DREAM sampler",
        "Finished DREAM sampler",
    ]
    assert result["sampled_params_path"] == str(sampled_params_path)
    assert result["log_ps_path"] == str(log_ps_path)


def test_run_bundle_batches_runtime_output_by_interval(tmp_path, monkeypatch):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    sampled_params_path = bundle.run_dir / "dream_sampled_params.npy"
    log_ps_path = bundle.run_dir / "dream_log_ps.npy"
    np.save(sampled_params_path, np.asarray([[[1.0]]], dtype=float))
    np.save(log_ps_path, np.asarray([[1.0]], dtype=float))

    streamed_lines: list[str] = []
    time_points = iter([0.0, 0.1, 0.2, 0.3, 0.4])

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            del args
            assert kwargs["stdout"] is not None
            self.stdout = io.StringIO("line 1\nline 2\nline 3\n")
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "saxshell.saxs.dream.runtime.subprocess.Popen",
        _FakePopen,
    )
    monkeypatch.setattr(
        "saxshell.saxs.dream.runtime.time.monotonic",
        lambda: next(time_points),
    )

    workflow.run_bundle(
        bundle,
        output_callback=streamed_lines.append,
        output_interval_seconds=1.0,
    )

    assert streamed_lines == ["line 1\nline 2\nline 3"]


def test_generated_runtime_bundle_passes_burnin_to_pydream(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    settings = workflow.load_settings()
    settings.niterations = 2500
    settings.burnin_percent = 24
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(settings=settings, entries=entries)

    runtime_source = bundle.runtime_script_path.read_text(encoding="utf-8")

    assert "burnin=int(" in runtime_source  # codespell:ignore burnin
    assert 'SETTINGS["burnin_percent"]' in runtime_source
    assert 'SETTINGS["niterations"]' in runtime_source


def test_generated_runtime_bundle_suppresses_pydream_warning_flood(tmp_path):
    project_dir, _paths = _build_minimal_saxs_project(tmp_path)
    prefit = SAXSPrefitWorkflow(project_dir)
    prefit.save_fit(prefit.parameter_entries)

    workflow = SAXSDreamWorkflow(project_dir)
    entries = workflow.create_default_parameter_map(persist=True)
    bundle = workflow.create_runtime_bundle(entries=entries)

    runtime_source = bundle.runtime_script_path.read_text(encoding="utf-8")

    assert "import warnings" in runtime_source
    assert "warnings.filterwarnings(" in runtime_source
    assert "'where' used without 'out'" in runtime_source
    assert "memory in output" in runtime_source
    assert 'module=r"pydream\\.Dream"' in runtime_source
