from __future__ import annotations

import json

import numpy as np
import pytest

import saxshell.saxs.prefit.cluster_geometry as cluster_geometry_module
from saxshell.fullrmc.solution_properties import SolutionPropertiesSettings
from saxshell.saxs._model_templates import load_template_module
from saxshell.saxs.prefit import (
    SAXSPrefitWorkflow,
    compute_cluster_geometry_metadata,
    load_cluster_geometry_metadata,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)
from saxshell.saxs.solute_volume_fraction import (
    SoluteVolumeFractionSettings,
    calculate_solute_volume_fraction_estimate,
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
