from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from saxshell.saxs._model_templates import (
    TemplateSpec,
    clear_template_caches,
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.dream import SAXSDreamWorkflow
from saxshell.saxs.prefit import (
    ClusterGeometryMetadataRow,
    ClusterGeometryMetadataTable,
    SAXSPrefitWorkflow,
    save_cluster_geometry_metadata,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
)


@dataclass(slots=True)
class TemplateValidationCheck:
    name: str
    passed: bool
    message: str


@dataclass(slots=True)
class TemplateValidationResult:
    template_name: str
    template_path: Path
    metadata_path: Path | None
    display_name: str | None = None
    checks: list[TemplateValidationCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def errors(self) -> list[str]:
        return [check.message for check in self.checks if not check.passed]


def validate_template_candidate(
    template_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> TemplateValidationResult:
    source_path = Path(template_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Template file not found: {source_path}")
    metadata_source = _resolve_metadata_path(source_path, metadata_path)
    result = TemplateValidationResult(
        template_name=source_path.stem,
        template_path=source_path,
        metadata_path=metadata_source,
    )

    if metadata_source is None:
        result.warnings.append(
            "No template metadata JSON was provided. The template can still "
            "run, but the UI will use fallback display text until a paired "
            "metadata file is added."
        )

    with tempfile.TemporaryDirectory(
        prefix="saxs_template_validation_"
    ) as tmp:
        candidate_dir = Path(tmp)
        staged_template = candidate_dir / f"{source_path.stem}.py"
        shutil.copy2(source_path, staged_template)
        if metadata_source is not None:
            shutil.copy2(
                metadata_source, candidate_dir / f"{source_path.stem}.json"
            )

        clear_template_caches()
        try:
            spec = _run_spec_validation(
                result, source_path.stem, candidate_dir
            )
            if spec is None:
                return result
            result.display_name = spec.display_name

            module = _run_module_validation(result, spec, candidate_dir)
            if module is None:
                return result

            _run_function_contract_validation(result, spec, module)
            if not result.passed:
                return result

            project_dir = _build_validation_project(
                candidate_dir, spec, module
            )
            _run_prefit_validation(result, project_dir, spec, candidate_dir)
            _run_cluster_geometry_constraint_validation(
                result,
                project_dir,
                spec,
                candidate_dir,
            )
            _run_dream_validation(result, project_dir, candidate_dir)
        finally:
            clear_template_caches()

    return result


def format_validation_report(result: TemplateValidationResult) -> str:
    lines = [
        f"Template: {result.template_name}",
        f"Source: {result.template_path}",
    ]
    if result.display_name:
        lines.append(f"Display name: {result.display_name}")
    if result.metadata_path is not None:
        lines.append(f"Metadata: {result.metadata_path}")
    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    lines.append("Checks:")
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"  - [{status}] {check.name}: {check.message}")
    lines.append(
        "Validation result: " + ("passed" if result.passed else "failed")
    )
    return "\n".join(lines)


def _resolve_metadata_path(
    template_path: Path,
    metadata_path: str | Path | None,
) -> Path | None:
    if metadata_path is not None:
        resolved = Path(metadata_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Metadata file not found: {resolved}")
        return resolved
    sibling = template_path.with_suffix(".json")
    return sibling if sibling.is_file() else None


def _append_check(
    result: TemplateValidationResult,
    name: str,
    passed: bool,
    message: str,
) -> None:
    result.checks.append(
        TemplateValidationCheck(
            name=name,
            passed=passed,
            message=message,
        )
    )


def _run_spec_validation(
    result: TemplateValidationResult,
    template_name: str,
    template_dir: Path,
) -> TemplateSpec | None:
    try:
        spec = load_template_spec(template_name, template_dir)
    except Exception as exc:
        _append_check(
            result,
            "template-spec",
            False,
            f"Unable to load template directives/metadata: {exc}",
        )
        return None

    duplicate_names = _find_duplicate_parameter_names(spec)
    reserved_names = [
        name
        for name in (parameter.name for parameter in spec.parameters)
        if name.startswith("w") and name[1:].isdigit()
    ]
    if duplicate_names:
        _append_check(
            result,
            "template-parameters",
            False,
            "Duplicate parameter names are not allowed: "
            + ", ".join(sorted(duplicate_names)),
        )
        return None
    if reserved_names:
        _append_check(
            result,
            "template-parameters",
            False,
            "Template-defined parameters cannot use generated weight names: "
            + ", ".join(sorted(reserved_names)),
        )
        return None

    _append_check(
        result,
        "template-spec",
        True,
        "Loaded template directives, metadata, and parameter definitions.",
    )
    return spec


def _run_module_validation(
    result: TemplateValidationResult,
    spec: TemplateSpec,
    template_dir: Path,
):
    try:
        module = load_template_module(spec.name, template_dir)
    except Exception as exc:
        _append_check(
            result,
            "template-import",
            False,
            f"Unable to import template module: {exc}",
        )
        return None

    lmfit_callable = getattr(module, spec.lmfit_model_name, None)
    dream_callable = getattr(module, spec.dream_model_name, None)
    if not callable(lmfit_callable):
        _append_check(
            result,
            "template-callables",
            False,
            f"Missing callable {spec.lmfit_model_name} for lmfit.",
        )
        return None
    if not callable(dream_callable):
        _append_check(
            result,
            "template-callables",
            False,
            f"Missing callable {spec.dream_model_name} for DREAM.",
        )
        return None

    _append_check(
        result,
        "template-import",
        True,
        "Imported template module and resolved declared lmfit/DREAM callables.",
    )
    return module


def _run_function_contract_validation(
    result: TemplateValidationResult,
    spec: TemplateSpec,
    module: Any,
) -> None:
    q_values = np.linspace(0.05, 0.3, 8)
    component = np.linspace(10.0, 17.0, 8)
    solvent = np.linspace(1.0, 2.4, 8)
    fit_params = {
        parameter.name: parameter.initial_value
        for parameter in spec.parameters
    }
    template_runtime_inputs = _build_synthetic_template_runtime_inputs(spec)
    extra_lmfit_inputs = [
        template_runtime_inputs[input_name]
        for input_name in spec.extra_lmfit_inputs
    ]
    try:
        profile = getattr(module, spec.lmfit_model_name)(
            q_values,
            solvent,
            [component],
            *extra_lmfit_inputs,
            w0=0.6,
            **fit_params,
        )
        profile_array = np.asarray(profile, dtype=float)
        if profile_array.shape != q_values.shape:
            raise ValueError(
                "lmfit model returned shape "
                f"{profile_array.shape}; expected {q_values.shape}."
            )
        if not np.all(np.isfinite(profile_array)):
            raise ValueError(
                "lmfit model returned non-finite values for the validation grid."
            )
    except Exception as exc:
        _append_check(
            result,
            "lmfit-function",
            False,
            f"lmfit function contract failed: {exc}",
        )
        return

    try:
        module.q_values = q_values
        module.experimental_intensities = profile_array
        module.theoretical_intensities = [component]
        module.solvent_intensities = solvent
        for input_name, values in template_runtime_inputs.items():
            setattr(module, input_name, np.asarray(values, dtype=float))
        dream_params = np.asarray(
            [0.6]
            + [fit_params[parameter.name] for parameter in spec.parameters],
            dtype=float,
        )
        log_likelihood = getattr(module, spec.dream_model_name)(dream_params)
        if not np.isfinite(float(log_likelihood)):
            raise ValueError("DREAM likelihood returned a non-finite scalar.")
    except Exception as exc:
        _append_check(
            result,
            "dream-function",
            False,
            f"DREAM function contract failed: {exc}",
        )
        return

    _append_check(
        result,
        "template-functions",
        True,
        "Validated lmfit and DREAM callables with synthetic data.",
    )


def _build_validation_project(
    template_dir: Path,
    spec: TemplateSpec,
    module: Any,
) -> Path:
    with tempfile.TemporaryDirectory(
        prefix="saxs_template_project_seed_"
    ) as tmp:
        seed_dir = Path(tmp)
        manager = SAXSProjectManager()
        project_dir = seed_dir / "validation_project"
        settings = manager.create_project(project_dir)
        paths = build_project_paths(project_dir)

        q_values = np.linspace(0.05, 0.3, 8)
        component = np.linspace(10.0, 17.0, 8)
        solvent = np.linspace(1.0, 2.4, 8)
        fit_params = {
            parameter.name: parameter.initial_value
            for parameter in spec.parameters
        }
        template_runtime_inputs = _build_synthetic_template_runtime_inputs(
            spec
        )
        extra_lmfit_inputs = [
            template_runtime_inputs[input_name]
            for input_name in spec.extra_lmfit_inputs
        ]
        experimental = getattr(module, spec.lmfit_model_name)(
            q_values,
            solvent,
            [component],
            *extra_lmfit_inputs,
            w0=0.6,
            **fit_params,
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

        solvent_path = paths.experimental_data_dir / "solvent_demo.txt"
        np.savetxt(
            solvent_path,
            np.column_stack([q_values, solvent]),
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
        settings.solvent_data_path = str(solvent_path)
        settings.copied_solvent_data_file = str(solvent_path)
        clusters_dir = paths.project_dir / "clusters"
        cluster_dir = clusters_dir / "A"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        (cluster_dir / "frame_0001.xyz").write_text(
            "1\nvalidation cluster\nPb 0.0 0.0 0.0\n",
            encoding="utf-8",
        )
        settings.clusters_dir = str(clusters_dir)
        settings.selected_model_template = spec.name
        manager.save_project(settings)
        if spec.cluster_geometry_support.supported:
            save_cluster_geometry_metadata(
                paths.cluster_geometry_metadata_file,
                ClusterGeometryMetadataTable(
                    rows=[
                        ClusterGeometryMetadataRow(
                            cluster_id="A",
                            structure="A",
                            motif="no_motif",
                            cluster_path=str(cluster_dir),
                            avg_size_metric=18.0,
                            effective_radius=9.0,
                            structure_factor_recommendation="sphere",
                            anisotropy_metric=1.0,
                            notes="Validation geometry seed.",
                            mapped_parameter="w0",
                            sf_approximation="sphere",
                            radii_type_used="ionic",
                            ionic_sphere_effective_radius=9.0,
                            bond_length_sphere_effective_radius=9.0,
                            ionic_ellipsoid_semiaxis_a=9.0,
                            ionic_ellipsoid_semiaxis_b=9.0,
                            ionic_ellipsoid_semiaxis_c=9.0,
                            bond_length_ellipsoid_semiaxis_a=9.0,
                            bond_length_ellipsoid_semiaxis_b=9.0,
                            bond_length_ellipsoid_semiaxis_c=9.0,
                            active_semiaxis_a=9.0,
                            active_semiaxis_b=9.0,
                            active_semiaxis_c=9.0,
                            mean_semiaxis_a=9.0,
                            mean_semiaxis_b=9.0,
                            mean_semiaxis_c=9.0,
                            mean_radius_of_gyration=6.0,
                            mean_max_radius=9.0,
                            mean_atom_count=1.0,
                            file_count=1,
                        )
                    ],
                    source_clusters_dir=str(clusters_dir),
                    template_name=spec.name,
                ),
            )

        persistent_project_dir = template_dir / "validation_project"
        if persistent_project_dir.exists():
            shutil.rmtree(persistent_project_dir)
        shutil.copytree(project_dir, persistent_project_dir)
        persistent_paths = build_project_paths(persistent_project_dir)
        persistent_settings = manager.load_project(persistent_project_dir)
        persistent_settings.project_dir = str(persistent_project_dir)
        persistent_settings.experimental_data_path = str(
            persistent_paths.experimental_data_dir / experimental_path.name
        )
        persistent_settings.copied_experimental_data_file = str(
            persistent_paths.experimental_data_dir / experimental_path.name
        )
        persistent_settings.solvent_data_path = str(
            persistent_paths.experimental_data_dir / solvent_path.name
        )
        persistent_settings.copied_solvent_data_file = str(
            persistent_paths.experimental_data_dir / solvent_path.name
        )
        persistent_settings.clusters_dir = str(
            persistent_paths.project_dir / "clusters"
        )
        manager.save_project(persistent_settings)
        return persistent_project_dir


def _build_synthetic_template_runtime_inputs(
    spec: TemplateSpec,
) -> dict[str, np.ndarray]:
    runtime_inputs: dict[str, np.ndarray] = {}
    for binding in spec.cluster_geometry_support.runtime_bindings:
        runtime_inputs[binding.runtime_name] = np.asarray([9.0], dtype=float)
    return runtime_inputs


def _run_prefit_validation(
    result: TemplateValidationResult,
    project_dir: Path,
    spec: TemplateSpec,
    template_dir: Path,
) -> None:
    try:
        workflow = SAXSPrefitWorkflow(
            project_dir,
            template_name=spec.name,
            template_dir=template_dir,
        )
        evaluation = workflow.evaluate()
        if not np.all(np.isfinite(evaluation.model_intensities)):
            raise ValueError(
                "Prefit evaluation returned non-finite model intensities."
            )
        workflow.run_fit(method="leastsq", max_nfev=20)
    except Exception as exc:
        _append_check(
            result,
            "prefit-workflow",
            False,
            f"Prefit workflow compatibility failed: {exc}",
        )
        return

    _append_check(
        result,
        "prefit-workflow",
        True,
        "Validated template through SAXSPrefitWorkflow evaluate/run_fit.",
    )


def _run_cluster_geometry_constraint_validation(
    result: TemplateValidationResult,
    project_dir: Path,
    spec: TemplateSpec,
    template_dir: Path,
) -> None:
    if not spec.cluster_geometry_support.supported:
        return

    allowed = spec.cluster_geometry_support.allowed_sf_approximations
    disallowed = tuple(
        option for option in ("sphere", "ellipsoid") if option not in allowed
    )

    try:
        workflow = SAXSPrefitWorkflow(
            project_dir,
            template_name=spec.name,
            template_dir=template_dir,
        )
        rows = workflow.cluster_geometry_rows()
        if not rows:
            raise ValueError(
                "Cluster geometry metadata was not loaded for validation."
            )
        if any(row.sf_approximation not in allowed for row in rows):
            raise ValueError(
                "Loaded cluster geometry rows contain disallowed "
                "sf_approximation values."
            )
        if any(
            row.structure_factor_recommendation not in allowed for row in rows
        ):
            raise ValueError(
                "Loaded cluster geometry rows contain disallowed "
                "structure_factor_recommendation values."
            )

        weight_name = (
            workflow.components[0].param_name if workflow.components else "w0"
        )
        for approximation in allowed:
            updated_rows = workflow.cluster_geometry_rows()
            updated_rows[0].sf_approximation = approximation
            workflow.set_cluster_geometry_rows(updated_rows)
            refreshed_row = workflow.cluster_geometry_rows()[0]
            if refreshed_row.sf_approximation != approximation:
                raise ValueError(
                    "Cluster geometry constraint sync failed to preserve "
                    f"allowed approximation {approximation!r}."
                )
            evaluation = workflow.evaluate()
            if not np.all(np.isfinite(evaluation.model_intensities)):
                raise ValueError(
                    "Prefit evaluation returned non-finite values after "
                    f"switching to {approximation!r} geometry."
                )
            if spec.cluster_geometry_support.dynamic_parameters:
                geometry_entries = [
                    entry
                    for entry in workflow.parameter_entries
                    if entry.category == "geometry"
                ]
                geometry_names = {entry.name for entry in geometry_entries}
                if approximation == "sphere":
                    expected_names = {
                        f"{spec.cluster_geometry_support.sphere_parameter_prefix}_{weight_name}"
                    }
                else:
                    expected_names = {
                        f"{prefix}_{weight_name}"
                        for prefix in (
                            spec.cluster_geometry_support.ellipsoid_parameter_prefixes
                        )
                    }
                if geometry_names != expected_names:
                    raise ValueError(
                        "Dynamic geometry parameters did not match the "
                        f"allowed {approximation!r} approximation."
                    )
                if any(entry.vary for entry in geometry_entries):
                    raise ValueError(
                        "Generated geometry parameters must default to vary off."
                    )

        if disallowed:
            updated_rows = workflow.cluster_geometry_rows()
            updated_rows[0].sf_approximation = disallowed[0]
            workflow.set_cluster_geometry_rows(updated_rows)
            normalized_row = workflow.cluster_geometry_rows()[0]
            if normalized_row.sf_approximation not in allowed:
                raise ValueError(
                    "Disallowed cluster geometry approximation was not "
                    "normalized to an allowed value."
                )
    except Exception as exc:
        _append_check(
            result,
            "cluster-geometry-constraints",
            False,
            "Cluster geometry constraint validation failed: " f"{exc}",
        )
        return

    _append_check(
        result,
        "cluster-geometry-constraints",
        True,
        "Validated allowed cluster geometry approximations and dynamic "
        "parameter triggering for the installed template.",
    )


def _run_dream_validation(
    result: TemplateValidationResult,
    project_dir: Path,
    template_dir: Path,
) -> None:
    try:
        prefit = SAXSPrefitWorkflow(project_dir, template_dir=template_dir)
        prefit.save_fit(prefit.parameter_entries)

        workflow = SAXSDreamWorkflow(project_dir, template_dir=template_dir)
        entries = workflow.create_default_parameter_map(persist=True)
        settings = workflow.load_settings()
        settings.nchains = 5
        settings.niterations = 3
        settings.nseedchains = 20
        settings.parallel = False
        settings.verbose = False
        settings.adapt_crossover = False
        settings.crossover_burnin = 1000

        expected_min_chains = settings.nchains
        expected_iterations = settings.niterations
        expected_active_parameters = sum(1 for entry in entries if entry.vary)
        bundle = workflow.create_runtime_bundle(
            settings=settings, entries=entries
        )
        run_result = workflow.run_bundle(bundle)
        sampled_params = np.load(run_result["sampled_params_path"])
        log_ps = np.load(run_result["log_ps_path"])
        if sampled_params.ndim != 3:
            raise ValueError(
                "DREAM runtime wrote sampled parameters with unexpected "
                f"rank {sampled_params.ndim}; expected 3."
            )
        if sampled_params.shape[0] < expected_min_chains:
            raise ValueError(
                "DREAM runtime wrote fewer chains than requested: "
                f"{sampled_params.shape[0]} < {expected_min_chains}."
            )
        if sampled_params.shape[1] != expected_iterations:
            raise ValueError(
                "DREAM runtime wrote unexpected iteration count in "
                "sampled parameters: "
                f"{sampled_params.shape[1]} != {expected_iterations}."
            )
        if sampled_params.shape[2] != expected_active_parameters:
            raise ValueError(
                "DREAM runtime wrote unexpected active-parameter width in "
                "sampled parameters: "
                f"{sampled_params.shape[2]} != {expected_active_parameters}."
            )
        if log_ps.ndim != 2:
            raise ValueError(
                "DREAM runtime wrote log probabilities with unexpected "
                f"rank {log_ps.ndim}; expected 2."
            )
        if log_ps.shape[0] != sampled_params.shape[0]:
            raise ValueError(
                "DREAM runtime wrote mismatched chain counts for sampled "
                f"parameters and log probabilities: "
                f"{sampled_params.shape[0]} != {log_ps.shape[0]}."
            )
        if log_ps.shape[1] != expected_iterations:
            raise ValueError(
                "DREAM runtime wrote unexpected iteration count in log "
                f"probabilities: {log_ps.shape[1]} != {expected_iterations}."
            )
        if not np.all(np.isfinite(sampled_params)):
            raise ValueError(
                "DREAM runtime produced non-finite sampled parameters."
            )
        if np.any(np.isnan(log_ps)) or np.any(np.isposinf(log_ps)):
            raise ValueError(
                "DREAM runtime produced invalid log probabilities "
                "(NaN or +inf)."
            )
    except Exception as exc:
        _append_check(
            result,
            "dream-runtime",
            False,
            f"DREAM runtime compatibility failed: {exc}",
        )
        return

    _append_check(
        result,
        "dream-runtime",
        True,
        "Validated template through SAXSDreamWorkflow bundle generation and execution.",
    )


def _find_duplicate_parameter_names(spec: TemplateSpec) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for parameter in spec.parameters:
        if parameter.name in seen:
            duplicates.add(parameter.name)
        seen.add(parameter.name)
    return duplicates


def _write_component_file(
    path: Path, q_values: np.ndarray, intensities: np.ndarray
) -> None:
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
