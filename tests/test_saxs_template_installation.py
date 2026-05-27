from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from saxshell.saxs._model_templates import (
    list_template_specs,
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.template_installation import (
    install_template_candidate,
    validate_template_candidate,
)

TEMPLATE_CANDIDATE_DIR = Path("tests/template_candidates")
CHARGED_MONOSQ_TEMPLATE = (
    "template_pydream_charged_monosq_normalized_scaled_solvent"
)
LEGACY_MONOSQ_TEMPLATE = "template_pydream_monosq_legacy"


def _write_template(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def _write_metadata(
    path: Path, display_name: str = "Candidate Template"
) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            {{
              "display_name": "{display_name}",
              "description": "Validation candidate template used for tests."
            }}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_validate_template_candidate_passes_for_bundled_template(tmp_path):
    source_spec = load_template_spec("template_pd_likelihood_monosq_decoupled")
    source_template = source_spec.module_path
    source_metadata = source_spec.metadata_path
    assert source_metadata is not None
    candidate_template = tmp_path / source_template.name
    candidate_metadata = tmp_path / source_metadata.name
    candidate_template.write_text(
        source_template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_metadata.write_text(
        source_metadata.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert result.passed
    assert result.display_name.startswith("MonoSQ Decoupled")
    check_names = {check.name for check in result.checks if check.passed}
    assert "prefit-workflow" in check_names
    assert "dream-runtime" in check_names


def test_validate_template_candidate_passes_for_bundled_poly_lma_template(
    tmp_path,
):
    source_spec = load_template_spec("template_pydream_poly_lma_hs")
    source_template = source_spec.module_path
    source_metadata = source_spec.metadata_path
    assert source_metadata is not None
    candidate_template = tmp_path / source_template.name
    candidate_metadata = tmp_path / source_metadata.name
    candidate_template.write_text(
        source_template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_metadata.write_text(
        source_metadata.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert result.passed
    assert result.display_name == "pyDREAM Poly LMA Hard-Sphere"
    check_names = {check.name for check in result.checks if check.passed}
    assert "prefit-workflow" in check_names
    assert "dream-runtime" in check_names


def test_validate_template_candidate_passes_for_bundled_poly_lma_mixed_template(
    tmp_path,
):
    source_spec = load_template_spec("template_pydream_poly_lma_hs_mix_approx")
    source_template = source_spec.module_path
    source_metadata = source_spec.metadata_path
    assert source_metadata is not None
    candidate_template = tmp_path / source_template.name
    candidate_metadata = tmp_path / source_metadata.name
    candidate_template.write_text(
        source_template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_metadata.write_text(
        source_metadata.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert result.passed
    assert result.display_name == (
        "pyDREAM Poly LMA Hard-Sphere/Ellipsoid Mix (Approx.)"
    )
    check_names = {check.name for check in result.checks if check.passed}
    assert "prefit-workflow" in check_names
    assert "dream-runtime" in check_names


def test_poly_lma_template_spec_exposes_sphere_only_geometry_parameterization():
    spec = load_template_spec("template_pydream_poly_lma_hs")

    assert spec.cluster_geometry_support.supported
    assert spec.cluster_geometry_support.allowed_sf_approximations == (
        "sphere",
    )
    assert spec.cluster_geometry_support.dynamic_parameters is True
    assert spec.cluster_geometry_support.sphere_parameter_prefix == "r_eff"
    assert spec.cluster_geometry_support.ellipsoid_parameter_prefixes == (
        "a_eff",
        "b_eff",
        "c_eff",
    )


def test_poly_lma_mixed_template_spec_exposes_ellipsoid_geometry_parameterization():
    spec = load_template_spec("template_pydream_poly_lma_hs_mix_approx")

    assert spec.display_name == (
        "pyDREAM Poly LMA Hard-Sphere/Ellipsoid Mix (Approx.)"
    )
    assert spec.cluster_geometry_support.supported
    assert spec.cluster_geometry_support.allowed_sf_approximations == (
        "sphere",
        "ellipsoid",
    )
    assert spec.cluster_geometry_support.dynamic_parameters is True


def test_poly_lma_legacy_template_is_marked_deprecated():
    spec = load_template_spec("template_pydream_poly_lma_hs_legacy")

    assert spec.display_name == "pyDREAM Poly LMA Hard-Sphere (deprecated)"
    assert spec.deprecated is True
    assert spec.cluster_geometry_support.allowed_sf_approximations == (
        "sphere",
        "ellipsoid",
    )


def test_template_listing_hides_deprecated_by_default():
    visible_names = {spec.name for spec in list_template_specs()}
    all_names = {
        spec.name for spec in list_template_specs(include_deprecated=True)
    }

    assert CHARGED_MONOSQ_TEMPLATE in visible_names
    assert LEGACY_MONOSQ_TEMPLATE in visible_names
    assert "template_pd_likelihood_monosq_decoupled" not in visible_names
    assert "template_pd_likelihood_monosq_decoupled" in all_names
    assert "template_pydream_poly_lma_hs_legacy" not in visible_names
    assert "template_pydream_poly_lma_hs_legacy" in all_names


def test_legacy_monosq_template_preserves_mdscatter_likelihood_convention():
    spec = load_template_spec(LEGACY_MONOSQ_TEMPLATE)
    module = load_template_module(LEGACY_MONOSQ_TEMPLATE)

    assert spec.display_name == "pyDREAM MonoSQ Legacy (MDScatter)"
    assert spec.deprecated is False
    assert [parameter.name for parameter in spec.parameters] == [
        "solv_w",
        "offset",
        "eff_r",
        "vol_frac",
        "scale",
    ]
    assert (
        spec.solution_scattering_support.solvent_contribution_scale_mode
        == "global_scale"
    )

    q_values = np.linspace(0.05, 0.30, 6)
    component = np.linspace(10.0, 20.0, 6)
    solvent = np.linspace(1.0, 2.0, 6)
    params = {
        "w0": 0.75,
        "solv_w": 2.0,
        "offset": 0.05,
        "eff_r": 9.0,
        "vol_frac": 0.02,
        "scale": 1e-10,
    }
    profile = module.lmfit_model_profile(
        q_values,
        solvent,
        [component],
        **params,
    )

    module.q_values = q_values
    module.experimental_intensities = profile
    module.theoretical_intensities = [component]
    module.solvent_intensities = solvent
    dream_params = np.asarray(
        [
            params["w0"],
            params["solv_w"],
            params["offset"],
            params["eff_r"],
            params["vol_frac"],
            params["scale"],
        ],
        dtype=float,
    )

    log_likelihood = module.log_likelihood_monosq_legacy(dream_params)
    expected_per_point = -np.log(
        np.sqrt(2.0 * np.pi) * module.LEGACY_LIKELIHOOD_SIGMA
    )
    assert log_likelihood == pytest.approx(len(q_values) * expected_per_point)


def test_charged_monosq_template_spec_exposes_scaled_solvent_metadata():
    spec = load_template_spec(CHARGED_MONOSQ_TEMPLATE)

    assert spec.display_name == (
        "pyDREAM Charged MonoSQ Normalized (Scaled Solvent Weight)"
    )
    assert [parameter.name for parameter in spec.parameters] == [
        "solv_w",
        "offset",
        "eff_r",
        "vol_frac",
        "charge",
        "temperature",
        "concentration_salt",
        "dielectconst",
        "scale",
    ]
    assert (
        spec.solution_scattering_support.volume_fraction_parameter
        == "vol_frac"
    )
    assert spec.solution_scattering_support.volume_fraction_kind == "solute"
    assert (
        spec.solution_scattering_support.volume_fraction_source == "physical"
    )
    assert (
        spec.solution_scattering_support.solvent_contribution_scale_mode
        == "global_scale"
    )
    assert (
        spec.solution_scattering_support.molar_concentration_parameter
        == "concentration_salt"
    )
    assert spec.prefit_support.auto_apply_autoscale_on_load
    assert spec.prefit_support.autoscale_bounds_mode == "adaptive"

    charge_entry = next(
        parameter
        for parameter in spec.parameters
        if parameter.name == "charge"
    )
    assert charge_entry.minimum == pytest.approx(1e-6)
    assert charge_entry.maximum == pytest.approx(200.0)


def test_charged_monosq_hayter_msa_matches_sasview_reference_values():
    module = load_template_module(CHARGED_MONOSQ_TEMPLATE)

    q_values = np.asarray([0.00001, 0.0010, 0.01, 0.075])
    expected = np.asarray([0.0711646, 0.0712928, 0.0847006, 1.07150])

    actual = module.calc_hayter_msa_sq(
        20.75,
        0.0192,
        19.0,
        298.0,
        0.0,
        78.0,
        q_values,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)


def test_charged_monosq_lmfit_model_uses_scaled_solvent_convention():
    module = load_template_module(CHARGED_MONOSQ_TEMPLATE)
    q_values = np.linspace(0.01, 0.08, 5)
    solvent = np.linspace(1.0, 2.0, 5)
    components = [
        np.linspace(10.0, 14.0, 5),
        np.linspace(4.0, 8.0, 5),
    ]
    params = {
        "w0": 0.25,
        "w1": 0.75,
        "solv_w": 0.5,
        "offset": 0.2,
        "eff_r": 20.75,
        "vol_frac": 0.0192,
        "charge": 19.0,
        "temperature": 298.0,
        "concentration_salt": 0.0,
        "dielectconst": 78.0,
        "scale": 2.0e-4,
    }

    structure_factor = module.calc_hayter_msa_sq(
        params["eff_r"],
        params["vol_frac"],
        params["charge"],
        params["temperature"],
        params["concentration_salt"],
        params["dielectconst"],
        q_values,
    )
    mixture = params["w0"] * components[0] + params["w1"] * components[1]
    expected = (
        params["scale"]
        * (mixture * structure_factor + params["solv_w"] * solvent)
        + params["offset"]
    )

    actual = module.lmfit_model_profile(
        q_values,
        solvent,
        components,
        **params,
    )

    np.testing.assert_allclose(actual, expected)


def test_validate_template_candidate_passes_for_sphere_only_geometry_constraints(
    tmp_path,
):
    candidate_template = _write_template(
        tmp_path / "candidate_geometry_sphere_only.py",
        """
        import numpy as np

        # model_lmfit: lmfit_model_profile
        # model_pydream: log_likelihood_candidate
        # inputs_lmfit: q, solvent_data, model_data, effective_radii, params
        # inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
        # param_columns: Structure, Motif, Param, Value, Vary, Min, Max
        # cluster_geometry_metadata: true
        # param: phi_int,0.02,True,0.0,0.4
        # param: scale,1.0,True,1e-8,1e8
        # param: offset,0.0,True,-1.0,1.0
        # param: log_sigma,-9.21,True,-20.0,5.0

        def lmfit_model_profile(q, solvent_data, model_data, effective_radii, **params):
            del solvent_data
            component = np.asarray(model_data[0], dtype=float)
            sq = 1.0 / (1.0 + params["phi_int"] * np.asarray(q, dtype=float) * float(effective_radii[0]))
            return params["scale"] * component * sq + params["offset"]

        def log_likelihood_candidate(params):
            weight = float(params[0])
            phi_int = float(params[1])
            scale = float(params[2])
            offset = float(params[3])
            _log_sigma = float(params[4])
            model = scale * weight * np.asarray(theoretical_intensities[0], dtype=float)
            model = model / (1.0 + phi_int * np.asarray(q_values, dtype=float) * float(effective_radii[0]))
            model = model + offset
            residuals = np.asarray(experimental_intensities, dtype=float) - model
            return float(-0.5 * np.mean(residuals**2))
        """,
    )
    candidate_metadata = _write_template(
        tmp_path / "candidate_geometry_sphere_only.json",
        """
        {
          "display_name": "Candidate Sphere-Only Geometry Template",
          "description": "Validation candidate template used for sphere-only geometry tests.",
          "capabilities": {
            "cluster_geometry_metadata": {
              "supported": true,
              "mapping_target": "component_weights",
              "allowed_sf_approximations": ["sphere"],
              "metadata_fields": [
                "effective_radius",
                "structure_factor_recommendation"
              ],
              "runtime_bindings": {
                "effective_radii": "effective_radius"
              }
            }
          }
        }
        """,
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert result.passed
    spec = load_template_spec(candidate_template.stem, tmp_path)
    assert spec.cluster_geometry_support.allowed_sf_approximations == (
        "sphere",
    )
    check_names = {check.name for check in result.checks if check.passed}
    assert "cluster-geometry-constraints" in check_names


def test_validate_template_candidate_rejects_invalid_geometry_constraints(
    tmp_path,
):
    candidate_template = _write_template(
        tmp_path / "candidate_invalid_geometry_constraint.py",
        """
        import numpy as np

        # model_lmfit: lmfit_model_profile
        # model_pydream: log_likelihood_candidate
        # inputs_lmfit: q, solvent_data, model_data, effective_radii, params
        # inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
        # param_columns: Structure, Motif, Param, Value, Vary, Min, Max
        # cluster_geometry_metadata: true
        # param: scale,1.0,True,0.0,10.0
        # param: offset,0.0,True,-1.0,1.0
        # param: log_sigma,-9.21,True,-20.0,5.0

        def lmfit_model_profile(q, solvent_data, model_data, effective_radii, **params):
            del q, solvent_data, effective_radii
            return params["scale"] * np.asarray(model_data[0], dtype=float) + params["offset"]

        def log_likelihood_candidate(params):
            del params
            return -1.0
        """,
    )
    candidate_metadata = _write_template(
        tmp_path / "candidate_invalid_geometry_constraint.json",
        """
        {
          "display_name": "Candidate Invalid Geometry Constraint",
          "description": "Validation candidate template used for invalid geometry constraint tests.",
          "capabilities": {
            "cluster_geometry_metadata": {
              "supported": true,
              "mapping_target": "component_weights",
              "allowed_sf_approximations": ["cube"],
              "metadata_fields": [
                "effective_radius"
              ],
              "runtime_bindings": {
                "effective_radii": "effective_radius"
              }
            }
          }
        }
        """,
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert not result.passed
    assert any(
        "unsupported allowed_sf_approximations values: cube" in error
        for error in result.errors
    )


def test_validate_template_candidate_fails_for_missing_dream_callable(
    tmp_path,
):
    candidate_template = _write_template(
        tmp_path / "candidate_template.py",
        """
        import numpy as np

        # model_lmfit: lmfit_model_profile
        # model_pydream: log_likelihood_candidate
        # inputs_lmfit: q, solvent_data, model_data, params
        # inputs_pydream: q, solvent_data, model_data, params
        # param_columns: Structure, Motif, Param, Value, Vary, Min, Max
        # param: solv_w,0.0,True,0.0,1.0
        # param: offset,0.0,True,-1.0,1.0
        # param: scale,1.0,True,0.1,10.0

        def lmfit_model_profile(q, solvent_data, model_data, **params):
            return np.asarray(model_data[0], dtype=float) * params["scale"] + params["offset"]
        """,
    )
    candidate_metadata = _write_metadata(tmp_path / "candidate_template.json")

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert not result.passed
    assert any(
        "Missing callable log_likelihood_candidate" in error
        for error in result.errors
    )


def test_validate_template_candidate_rejects_cluster_geometry_binding_missing_from_declared_inputs(
    tmp_path,
):
    candidate_template = _write_template(
        tmp_path / "candidate_template.py",
        """
        import numpy as np

        # model_lmfit: lmfit_model_profile
        # model_pydream: log_likelihood_candidate
        # inputs_lmfit: q, solvent_data, model_data, params
        # inputs_pydream: q_values, experimental_intensities, solvent_intensities, theoretical_intensities, effective_radii, params
        # param_columns: Structure, Motif, Param, Value, Vary, Min, Max
        # cluster_geometry_metadata: true
        # param: phi_solute,0.02,True,0.0,0.5
        # param: phi_int,0.02,True,0.0,0.4
        # param: solvent_scale,1.0,True,0.0,1.0
        # param: scale,1.0,True,1e-8,1e8
        # param: offset,0.0,True,-1.0,1.0
        # param: log_sigma,-9.21,True,-20.0,5.0

        def lmfit_model_profile(q, solvent_data, model_data, **params):
            return np.asarray(model_data[0], dtype=float)

        def log_likelihood_candidate(params):
            return -1.0
        """,
    )
    candidate_metadata = _write_template(
        tmp_path / "candidate_template.json",
        """
        {
          "display_name": "Candidate Cluster Geometry Template",
          "description": "Validation candidate template used for tests.",
          "capabilities": {
            "cluster_geometry_metadata": {
              "supported": true,
              "mapping_target": "component_weights",
              "metadata_fields": [
                "effective_radius"
              ],
              "runtime_bindings": {
                "effective_radii": "effective_radius"
              }
            }
          }
        }
        """,
    )

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert not result.passed
    assert any(
        "missing from inputs_lmfit: effective_radii" in error
        for error in result.errors
    )


def test_install_template_candidate_copies_files_and_loads_spec(tmp_path):
    source_template = Path(
        "src/saxshell/saxs/_model_templates/_deprecated/"
        "template_pd_likelihood_monosq.py"
    )
    source_metadata = source_template.with_suffix(".json")
    candidate_template = tmp_path / source_template.name
    candidate_metadata = tmp_path / source_metadata.name
    candidate_template.write_text(
        source_template.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    candidate_metadata.write_text(
        source_metadata.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    install_dir = tmp_path / "installed_templates"

    installed = install_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
        destination_dir=install_dir,
    )

    assert installed.validation_result.passed
    assert installed.installed_template_path.is_file()
    assert installed.installed_metadata_path is not None
    assert installed.installed_metadata_path.is_file()

    spec = load_template_spec(installed.template_name, install_dir)
    assert spec.display_name.startswith("MonoSQ PD")


def test_install_template_candidate_generates_metadata_from_model_inputs(
    tmp_path,
):
    candidate_template = TEMPLATE_CANDIDATE_DIR / "valid_installable_model.py"
    install_dir = tmp_path / "installed_templates"

    installed = install_template_candidate(
        candidate_template,
        model_name="Installed Candidate Model",
        model_description="Installed from the generated metadata workflow.",
        destination_dir=install_dir,
    )

    assert installed.validation_result.passed
    assert installed.template_name == "template_installed_candidate_model"
    assert installed.installed_template_path.is_file()
    assert installed.installed_metadata_path is not None
    assert installed.installed_metadata_path.is_file()

    spec = load_template_spec(installed.template_name, install_dir)
    assert spec.display_name == "Installed Candidate Model"
    assert "generated metadata workflow" in spec.description


def test_install_template_candidate_rejects_missing_header_candidate(
    tmp_path,
):
    candidate_template = (
        TEMPLATE_CANDIDATE_DIR / "fail_missing_header_model.py"
    )

    with pytest.raises(ValueError, match="missing directives"):
        install_template_candidate(
            candidate_template,
            model_name="Broken Header Model",
            model_description="Expected to fail because the header is incomplete.",
            destination_dir=tmp_path / "installed_templates",
        )


def test_install_template_candidate_rejects_missing_dream_callable_candidate(
    tmp_path,
):
    candidate_template = (
        TEMPLATE_CANDIDATE_DIR / "fail_missing_dream_callable_model.py"
    )

    with pytest.raises(
        ValueError,
        match="Missing callable log_likelihood_candidate",
    ):
        install_template_candidate(
            candidate_template,
            model_name="Broken Dream Callable Model",
            model_description="Expected to fail because the DREAM callable is missing.",
            destination_dir=tmp_path / "installed_templates",
        )


def test_install_template_candidate_rejects_nonfinite_likelihood_candidate(
    tmp_path,
):
    candidate_template = (
        TEMPLATE_CANDIDATE_DIR / "fail_nonfinite_likelihood_model.py"
    )

    with pytest.raises(
        ValueError,
        match="DREAM likelihood returned a non-finite scalar",
    ):
        install_template_candidate(
            candidate_template,
            model_name="Broken Likelihood Model",
            model_description="Expected to fail because the DREAM likelihood is non-finite.",
            destination_dir=tmp_path / "installed_templates",
        )
