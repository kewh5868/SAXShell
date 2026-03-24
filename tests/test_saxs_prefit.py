from __future__ import annotations

import json

import numpy as np
import pytest

from saxshell.saxs._model_templates import load_template_module
from saxshell.saxs.prefit import (
    SAXSPrefitWorkflow,
    compute_cluster_geometry_metadata,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)


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


def _build_poly_lma_geometry_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "poly_lma_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)

    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    template_name = "template_pydream_poly_lma_hs"

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
    assert recommendation.points_used == 8


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
        solvent_intensity * 0.5 * 2e-3,
    )


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
    assert table.rows[0].effective_radius == pytest.approx(effective_radius)

    evaluation = workflow.evaluate()

    assert np.allclose(
        evaluation.model_intensities,
        evaluation.experimental_intensities,
    )
    assert workflow.template_runtime_inputs_payload()["effective_radii"] == (
        pytest.approx([effective_radius])
    )
