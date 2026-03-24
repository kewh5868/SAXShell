from __future__ import annotations

import textwrap
from pathlib import Path

from saxshell.saxs._model_templates import load_template_spec
from saxshell.saxs.template_installation import (
    install_template_candidate,
    validate_template_candidate,
)


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
    source_template = Path(
        "src/saxshell/saxs/_model_templates/template_pd_likelihood_monosq_decoupled.py"
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
    source_template = Path(
        "src/saxshell/saxs/_model_templates/template_pydream_poly_lma_hs.py"
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

    result = validate_template_candidate(
        candidate_template,
        metadata_path=candidate_metadata,
    )

    assert result.passed
    assert result.display_name == "pyDREAM Poly LMA Hard-Sphere"
    check_names = {check.name for check in result.checks if check.passed}
    assert "prefit-workflow" in check_names
    assert "dream-runtime" in check_names


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
        # param: solvent_scale,1.0,True,0.0,5.0
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
        "src/saxshell/saxs/_model_templates/template_pd_likelihood_monosq.py"
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
