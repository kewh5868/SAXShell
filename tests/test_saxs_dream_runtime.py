from __future__ import annotations

import io
import json

import numpy as np
import pytest

from saxshell.saxs._model_templates import (
    list_template_specs,
    load_template_module,
)
from saxshell.saxs.dream import SAXSDreamResultsLoader, SAXSDreamWorkflow
from saxshell.saxs.prefit import (
    SAXSPrefitWorkflow,
    compute_cluster_geometry_metadata,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)

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
        template_runtime_inputs = {
            binding.runtime_name: np.asarray([9.0], dtype=float)
            for binding in template_spec.cluster_geometry_support.runtime_bindings
        }
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
