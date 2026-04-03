from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from inspect import Parameter, signature
from itertools import product
from pathlib import Path

import numpy as np
from lmfit import Parameters, fit_report, minimize

from saxshell.saxs._model_templates import (
    TemplateSpec,
    list_template_specs,
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.prefit.cluster_geometry import (
    DEFAULT_IONIC_RADIUS_TYPE,
    DEFAULT_RADIUS_TYPE,
    ClusterGeometryMetadataRow,
    ClusterGeometryMetadataTable,
    apply_default_component_mapping,
    compute_cluster_geometry_metadata,
    copy_cluster_geometry_rows,
    load_cluster_geometry_metadata,
    save_cluster_geometry_metadata,
    synchronize_cluster_geometry_table,
    validate_positive_cluster_geometry_table,
)
from saxshell.saxs.project_manager import (
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
    distribution_id_for_settings,
    effective_q_range_for_settings,
    load_built_component_q_range,
    project_artifact_paths,
)


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


MINIMUM_POSITIVE_RADIUS = float(np.nextafter(0.0, 1.0))
Q_RANGE_EDGE_TOLERANCE_ABS = 1e-4
SOLUTE_VOLUME_FRACTION_PARAMETER_NAMES = (
    "phi_solute",
    "solute_volume_fraction",
)
SOLVENT_VOLUME_FRACTION_PARAMETER_NAMES = (
    "phi_solvent",
    "solvent_fraction",
    "solvent_volume_fraction",
)
SOLVENT_WEIGHT_PARAMETER_NAMES = (
    "solv_w",
    "solvent_scale",
)
PREFIT_MODEL_POSITIVE_FLOOR_RELATIVE = 1e-9
PREFIT_MODEL_NEGATIVE_PENALTY_MULTIPLIER = 25.0
PREFIT_MODEL_NONFINITE_PENALTY_MULTIPLIER = 100.0
PREFIT_GRID_SWEEP_PARAMETER_LIMIT = 3
PREFIT_GRID_SWEEP_LEVEL_POINTS: dict[int, tuple[int, ...]] = {
    1: (9, 7, 7),
    2: (5, 5, 5),
    3: (4, 4, 4),
}
PREFIT_SEQUENCE_HISTORY_FILENAME = "prefit_sequence_history.json"


def q_range_boundary_tolerance(
    lower: float,
    upper: float,
) -> float:
    return max(
        1e-12,
        1e-9 * max(abs(lower), abs(upper), 1.0),
        Q_RANGE_EDGE_TOLERANCE_ABS,
    )


def normalize_requested_q_range_to_supported(
    requested_min: float,
    requested_max: float,
    supported_min: float,
    supported_max: float,
) -> tuple[float, float]:
    tolerance = q_range_boundary_tolerance(supported_min, supported_max)
    normalized_min = float(requested_min)
    normalized_max = float(requested_max)
    if abs(normalized_min - supported_min) <= tolerance:
        normalized_min = float(supported_min)
    if abs(normalized_max - supported_max) <= tolerance:
        normalized_max = float(supported_max)
    return normalized_min, normalized_max


@dataclass(slots=True)
class PrefitComponent:
    structure: str
    motif: str
    param_name: str
    weight_value: float
    profile_file: str
    q_values: np.ndarray
    intensities: np.ndarray


@dataclass(slots=True)
class PrefitParameterEntry:
    structure: str
    motif: str
    name: str
    value: float
    vary: bool
    minimum: float
    maximum: float
    category: str
    value_expression: str | None = None
    initial_value_expression: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PrefitParameterEntry":
        return cls(
            structure=str(payload.get("structure", "")),
            motif=str(payload.get("motif", "")),
            name=str(payload.get("name", "")),
            value=float(payload.get("value", 0.0)),
            value_expression=_optional_str(
                payload.get("value_expression", payload.get("expression"))
            ),
            initial_value_expression=_optional_str(
                payload.get(
                    "initial_value_expression",
                    payload.get("initial_expression"),
                )
            ),
            vary=bool(payload.get("vary", True)),
            minimum=float(payload.get("minimum", payload.get("min", 0.0))),
            maximum=float(payload.get("maximum", payload.get("max", 0.0))),
            category=str(payload.get("category", "fit")),
        )


def _parameter_value_expression(
    entry: PrefitParameterEntry,
) -> str | None:
    return _optional_str(entry.value_expression)


def _parameter_initial_value_expression(
    entry: PrefitParameterEntry,
) -> str | None:
    return _optional_str(entry.initial_value_expression)


def normalize_prefit_parameter_expression(expression: str) -> str:
    normalized = str(expression).strip()
    if not normalized:
        raise ValueError("Linked parameter expressions cannot be empty.")
    if normalized[0] in {"*", "/"}:
        return f"1{normalized}"
    if normalized[0] == "+":
        return f"0{normalized}"
    return normalized


def build_prefit_lmfit_parameters(
    entries: list[PrefitParameterEntry],
) -> tuple[Parameters, list[PrefitParameterEntry]]:
    working_entries = [
        PrefitParameterEntry.from_dict(entry.to_dict()) for entry in entries
    ]
    seed_params = Parameters()
    runtime_expression_entries: list[PrefitParameterEntry] = []
    seed_expression_entries: list[PrefitParameterEntry] = []

    for entry in working_entries:
        value = float(entry.value)
        seed_params.add(
            entry.name,
            value=value,
            vary=False,
            min=-np.inf,
            max=np.inf,
        )
        if _parameter_value_expression(entry) is not None:
            runtime_expression_entries.append(entry)
        if _parameter_initial_value_expression(entry) is not None:
            seed_expression_entries.append(entry)

    for entry in runtime_expression_entries + seed_expression_entries:
        raw_expression = _parameter_value_expression(
            entry
        ) or _parameter_initial_value_expression(entry)
        if raw_expression is None:
            continue
        normalized_expression = normalize_prefit_parameter_expression(
            raw_expression
        )
        try:
            seed_params[entry.name].set(
                expr=normalized_expression,
                vary=False,
                min=-np.inf,
                max=np.inf,
            )
        except Exception as exc:
            raise ValueError(
                "Invalid linked parameter expression for "
                f"{entry.name}: {raw_expression}"
            ) from exc

    for entry in runtime_expression_entries + seed_expression_entries:
        raw_expression = _parameter_value_expression(
            entry
        ) or _parameter_initial_value_expression(entry)
        if raw_expression is None:
            continue
        try:
            entry.value = float(seed_params[entry.name].value)
        except Exception as exc:
            raise ValueError(
                "Invalid linked parameter expression for "
                f"{entry.name}: {raw_expression}"
            ) from exc

    lmfit_params = Parameters()
    for entry in working_entries:
        lower = float(entry.minimum)
        upper = float(entry.maximum)
        value = float(entry.value)
        if lower > upper:
            lower, upper = upper, lower
        if value < lower:
            lower = value
        if value > upper:
            upper = value
        entry.minimum = lower
        entry.maximum = upper
        lmfit_params.add(
            entry.name,
            value=value,
            vary=bool(entry.vary),
            min=lower,
            max=upper,
        )

    for entry in runtime_expression_entries:
        raw_expression = _parameter_value_expression(entry)
        if raw_expression is None:
            continue
        normalized_expression = normalize_prefit_parameter_expression(
            raw_expression
        )
        try:
            lmfit_params[entry.name].set(
                expr=normalized_expression,
                vary=False,
                min=-np.inf,
                max=np.inf,
            )
        except Exception as exc:
            raise ValueError(
                "Invalid linked parameter expression for "
                f"{entry.name}: {raw_expression}"
            ) from exc

    for entry in runtime_expression_entries:
        raw_expression = _parameter_value_expression(entry)
        if raw_expression is None:
            continue
        try:
            entry.value = float(lmfit_params[entry.name].value)
        except Exception as exc:
            raise ValueError(
                "Invalid linked parameter expression for "
                f"{entry.name}: {raw_expression}"
            ) from exc
        entry.vary = False

    values = lmfit_params.valuesdict()
    for entry in working_entries:
        if _parameter_value_expression(entry) is not None:
            entry.value = float(values[entry.name])
            entry.vary = False
            continue
        parameter = lmfit_params[entry.name]
        entry.value = float(parameter.value)
        entry.vary = bool(parameter.vary)
        entry.minimum = float(parameter.min)
        entry.maximum = float(parameter.max)
    return lmfit_params, working_entries


def resolve_prefit_parameter_entries(
    entries: list[PrefitParameterEntry],
) -> list[PrefitParameterEntry]:
    _params, resolved_entries = build_prefit_lmfit_parameters(entries)
    return resolved_entries


def constrained_prefit_residuals(
    experimental: np.ndarray,
    model: np.ndarray,
) -> np.ndarray:
    experimental_values = np.asarray(experimental, dtype=float)
    model_values = np.asarray(model, dtype=float)
    residuals = model_values - experimental_values
    penalty = np.zeros_like(residuals, dtype=float)

    finite_experimental = experimental_values[np.isfinite(experimental_values)]
    finite_model = model_values[np.isfinite(model_values)]
    penalty_scale_candidates = [1e-12]
    if finite_experimental.size:
        penalty_scale_candidates.append(
            float(np.max(np.abs(finite_experimental)))
        )
    if finite_model.size:
        penalty_scale_candidates.append(float(np.max(np.abs(finite_model))))
    penalty_scale = max(penalty_scale_candidates)
    positive_floor = max(
        penalty_scale * PREFIT_MODEL_POSITIVE_FLOOR_RELATIVE,
        1e-15,
    )

    invalid_mask = ~np.isfinite(model_values)
    if np.any(invalid_mask):
        penalty[invalid_mask] = (
            PREFIT_MODEL_NONFINITE_PENALTY_MULTIPLIER * penalty_scale
        )

    non_positive_mask = np.isfinite(model_values) & (
        model_values <= positive_floor
    )
    if np.any(non_positive_mask):
        reference = np.maximum(
            np.abs(experimental_values[non_positive_mask]),
            positive_floor,
        )
        deficit = (
            positive_floor - model_values[non_positive_mask]
        ) / reference
        penalty[non_positive_mask] = (
            PREFIT_MODEL_NEGATIVE_PENALTY_MULTIPLIER
            * penalty_scale
            * (1.0 + deficit)
        )

    return np.concatenate([residuals, penalty])


def validate_prefit_parameter_identifiability(
    entries: list[PrefitParameterEntry],
) -> None:
    phi_solute_entry = next(
        (entry for entry in entries if entry.name == "phi_solute"),
        None,
    )
    solvent_entry = next(
        (
            entry
            for entry in entries
            if entry.name in SOLVENT_WEIGHT_PARAMETER_NAMES
        ),
        None,
    )
    if phi_solute_entry is None or solvent_entry is None:
        return
    if bool(phi_solute_entry.vary) and bool(solvent_entry.vary):
        raise ValueError(
            "phi_solute and "
            f"{solvent_entry.name} cannot both vary during fitting. "
            "Their product controls the solvent subtraction term, so they "
            "must be fixed by prior estimates or only one may vary at a time."
        )


def _prefit_grid_searchable_entries(
    entries: list[PrefitParameterEntry],
) -> list[PrefitParameterEntry]:
    return [
        entry
        for entry in entries
        if bool(entry.vary)
        and _parameter_value_expression(entry) is None
        and np.isfinite(float(entry.minimum))
        and np.isfinite(float(entry.maximum))
        and float(entry.maximum) > float(entry.minimum)
    ]


@dataclass(slots=True)
class PrefitEvaluation:
    q_values: np.ndarray
    experimental_intensities: np.ndarray | None
    model_intensities: np.ndarray
    residuals: np.ndarray | None
    solvent_intensities: np.ndarray | None = None
    solvent_contribution: np.ndarray | None = None
    structure_factor_trace: np.ndarray | None = None

    @property
    def is_model_only(self) -> bool:
        return self.experimental_intensities is None


@dataclass(slots=True)
class PrefitFitResult:
    parameter_entries: list[PrefitParameterEntry]
    evaluation: PrefitEvaluation
    fit_report: str
    method: str
    nfev: int
    chi_square: float
    reduced_chi_square: float
    r_squared: float
    optimization_strategy: str
    grid_evaluations: int = 0
    report_path: Path | None = None


@dataclass(slots=True)
class PrefitScaleRecommendation:
    current_scale: float
    recommended_scale: float
    recommended_minimum: float
    recommended_maximum: float
    adjustment_factor: float
    points_used: int
    current_offset: float | None = None
    recommended_offset: float | None = None
    recommended_offset_minimum: float | None = None
    recommended_offset_maximum: float | None = None


@dataclass(slots=True)
class PrefitGridSweepResult:
    parameter_names: tuple[str, ...]
    best_values: dict[str, float]
    evaluations: int
    best_score: float
    levels: int

    @property
    def strategy_label(self) -> str:
        return f"coarse-to-fine grid ({len(self.parameter_names)}D)"


@dataclass(slots=True)
class PrefitSavedState:
    name: str
    path: Path
    saved_at: str
    template_name: str
    parameter_entries: list[PrefitParameterEntry]
    cluster_geometry_table: ClusterGeometryMetadataTable | None = None
    method: str | None = None
    max_nfev: int | None = None
    autosave_prefits: bool | None = None


class SAXSPrefitWorkflow:
    """Load SAXS project components and run the lmfit prefit
    workflow."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        template_name: str | None = None,
        template_dir: str | Path | None = None,
    ) -> None:
        self.project_manager = SAXSProjectManager()
        self.settings = self.project_manager.load_project(project_dir)
        self.paths = build_project_paths(self.settings.project_dir)
        self.experimental_data = self._load_experimental_trace()
        self.template_dir = (
            Path(template_dir).expanduser().resolve()
            if template_dir is not None
            else None
        )
        self.template_spec = self._resolve_template_spec(template_name)
        self.template_module = load_template_module(
            self.template_spec.name,
            self.template_dir,
        )
        self.artifact_paths = project_artifact_paths(self.settings)
        self.component_map_path = self.artifact_paths.component_map_file
        self.prior_weights_path = self.artifact_paths.prior_weights_file
        self.component_dir = self.artifact_paths.component_dir
        self.prefit_dir = self.artifact_paths.prefit_dir
        self.solvent_data = self._load_solvent_trace()
        self.components = self._load_components()
        self.cluster_geometry_metadata_path = (
            self._cluster_geometry_metadata_path_for_settings(self.settings)
        )
        self.cluster_geometry_table = self._load_cluster_geometry_table()
        self._template_default_entries = (
            self._build_default_parameter_entries()
        )
        self._ensure_project_parameter_presets()
        self.parameter_entries = self.load_parameter_entries()

    @staticmethod
    def available_templates(
        template_dir: str | Path | None = None,
    ) -> list[TemplateSpec]:
        return list_template_specs(template_dir)

    def has_experimental_data(self) -> bool:
        return self.experimental_data is not None

    def can_run_prefit(self) -> bool:
        return (
            self.has_experimental_data() and not self.settings.model_only_mode
        )

    def _save_project_settings(self) -> Path:
        # Prefit interactions update project metadata frequently; avoid
        # rescanning registered input folders/files on each internal save.
        save_project = self.project_manager.save_project
        try:
            save_signature = signature(save_project)
        except (TypeError, ValueError):
            save_signature = None
        if save_signature is not None:
            supports_refresh_flag = (
                "refresh_registered_paths" in save_signature.parameters
                or any(
                    parameter.kind == Parameter.VAR_KEYWORD
                    for parameter in save_signature.parameters.values()
                )
            )
            if not supports_refresh_flag:
                return save_project(self.settings)
        return save_project(
            self.settings,
            refresh_registered_paths=False,
        )

    @staticmethod
    def _evaluation_statistics(
        entries: list[PrefitParameterEntry],
        evaluation: PrefitEvaluation,
    ) -> dict[str, float]:
        if (
            evaluation.residuals is None
            or evaluation.experimental_intensities is None
        ):
            return {}
        chi_square = float(np.sum(evaluation.residuals**2))
        dof = max(
            len(evaluation.q_values)
            - sum(1 for entry in entries if entry.vary),
            1,
        )
        reduced_chi_square = chi_square / dof
        ss_total = float(
            np.sum(
                (
                    evaluation.experimental_intensities
                    - np.mean(evaluation.experimental_intensities)
                )
                ** 2
            )
        )
        r_squared = (
            1.0 - chi_square / ss_total if ss_total > 0.0 else float("nan")
        )
        return {
            "chi_square": chi_square,
            "reduced_chi_square": reduced_chi_square,
            "r_squared": r_squared,
        }

    @staticmethod
    def _saved_state_display_label(
        state_name: str,
        payload: dict[str, object],
    ) -> str:
        statistics = payload.get("statistics", {})
        if not isinstance(statistics, dict):
            return state_name
        try:
            r_squared = float(statistics.get("r_squared"))
        except (TypeError, ValueError):
            return state_name
        if not np.isfinite(r_squared):
            return state_name
        return f"{state_name} (R^2={r_squared:.6g})"

    def set_model_only_mode(self, enabled: bool) -> None:
        self.settings.model_only_mode = bool(enabled)
        if self.settings.model_only_mode:
            self.settings.use_experimental_grid = False
            if self.settings.q_points is None or self.settings.q_points <= 1:
                self.settings.q_points = 500
        self._save_project_settings()
        self.experimental_data = self._load_experimental_trace()
        self.solvent_data = self._load_solvent_trace()

    def apply_project_settings(
        self,
        settings: ProjectSettings,
    ) -> None:
        incoming_settings = ProjectSettings.from_dict(settings.to_dict())
        if incoming_settings.resolved_project_dir != self.paths.project_dir:
            raise ValueError(
                "Cannot apply project settings from a different SAXS project."
            )
        selected_template = (
            str(incoming_settings.selected_model_template or "").strip()
            or self.template_spec.name
        )
        if selected_template != self.template_spec.name:
            raise ValueError(
                "The active Prefit template changed. Reload the project "
                "workflows instead of applying settings in place."
            )
        self.settings = incoming_settings
        self.paths = build_project_paths(self.settings.project_dir)
        self.artifact_paths = project_artifact_paths(self.settings)
        self.component_map_path = self.artifact_paths.component_map_file
        self.prior_weights_path = self.artifact_paths.prior_weights_file
        self.component_dir = self.artifact_paths.component_dir
        self.prefit_dir = self.artifact_paths.prefit_dir
        self.experimental_data = self._load_experimental_trace()
        self.solvent_data = self._load_solvent_trace()
        self.components = self._load_components()
        self.cluster_geometry_metadata_path = (
            self._cluster_geometry_metadata_path_for_settings(self.settings)
        )
        self.cluster_geometry_table = self._load_cluster_geometry_table()
        self._template_default_entries = (
            self._build_default_parameter_entries()
        )
        self._ensure_project_parameter_presets()
        self.parameter_entries = self.load_parameter_entries()

    def load_parameter_entries(self) -> list[PrefitParameterEntry]:
        best_entries = self.load_best_prefit_entries()
        if best_entries is not None:
            return self._apply_parameter_constraints(best_entries)
        state_path = self.prefit_dir / "prefit_state.json"
        if state_path.is_file():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            entries = payload.get("parameter_entries", [])
            parsed_entries = [
                PrefitParameterEntry.from_dict(entry) for entry in entries
            ]
            if self._has_matching_entry_signature(parsed_entries):
                return self._apply_parameter_constraints(parsed_entries)
            if parsed_entries:
                return self._merge_parameter_entries(
                    parsed_entries,
                    self._copy_entries(self._template_default_entries),
                )
        return self.load_template_reset_entries()

    def evaluate(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> PrefitEvaluation:
        entries = parameter_entries or self.parameter_entries
        q_values = self._component_q_values()
        model_data = self._model_data_for_q_values(q_values)
        experimental = (
            np.interp(
                q_values,
                self.experimental_data.q_values,
                self.experimental_data.intensities,
            )
            if self.experimental_data is not None
            else None
        )
        _lmfit_params, resolved_entries = build_prefit_lmfit_parameters(
            entries
        )
        params = {entry.name: float(entry.value) for entry in resolved_entries}
        solvent_data = (
            self._solvent_trace_for_q_values(q_values)
            if self.solvent_data is not None
            else np.zeros_like(q_values)
        )
        solvent_intensities = (
            np.asarray(solvent_data, dtype=float)
            if self.solvent_data is not None
            else None
        )
        extra_inputs = self._lmfit_extra_inputs()
        model_intensities = self._lmfit_model_function()(
            q_values,
            solvent_data,
            model_data,
            *extra_inputs,
            **params,
        )
        solvent_contribution = self._evaluate_solvent_contribution(
            q_values,
            solvent_data=solvent_data,
            model_data=model_data,
            params=params,
            extra_inputs=extra_inputs,
        )
        structure_factor_trace = self._evaluate_structure_factor_trace(
            q_values,
            solvent_data=solvent_data,
            model_data=model_data,
            params=params,
            extra_inputs=extra_inputs,
        )
        residuals = (
            np.asarray(model_intensities, dtype=float) - experimental
            if experimental is not None
            else None
        )
        return PrefitEvaluation(
            q_values=q_values,
            experimental_intensities=experimental,
            model_intensities=np.asarray(model_intensities, dtype=float),
            residuals=(
                np.asarray(residuals, dtype=float)
                if residuals is not None
                else None
            ),
            solvent_intensities=solvent_intensities,
            solvent_contribution=solvent_contribution,
            structure_factor_trace=structure_factor_trace,
        )

    def grid_searchable_parameter_names(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> list[str]:
        entries = resolve_prefit_parameter_entries(
            parameter_entries or self.parameter_entries
        )
        return [
            str(entry.name)
            for entry in _prefit_grid_searchable_entries(entries)
        ]

    def optimization_strategy_preview(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        *,
        method: str = "leastsq",
    ) -> str:
        varying_names = self.grid_searchable_parameter_names(parameter_entries)
        varying_count = len(varying_names)
        if 1 <= varying_count <= PREFIT_GRID_SWEEP_PARAMETER_LIMIT:
            return f"coarse-to-fine grid ({varying_count}D) + lmfit {method}"
        return f"lmfit {method}"

    def _coarse_to_fine_grid_sweep(
        self,
        lmfit_params: Parameters,
        objective,
        entries: list[PrefitParameterEntry],
    ) -> PrefitGridSweepResult | None:
        varying_entries = _prefit_grid_searchable_entries(entries)
        varying_count = len(varying_entries)
        if not (1 <= varying_count <= PREFIT_GRID_SWEEP_PARAMETER_LIMIT):
            return None

        parameter_names = tuple(entry.name for entry in varying_entries)
        base_ranges = {
            entry.name: (float(entry.minimum), float(entry.maximum))
            for entry in varying_entries
        }
        active_ranges = dict(base_ranges)
        best_values = {
            name: float(lmfit_params[name].value) for name in parameter_names
        }
        best_score = float("inf")
        total_evaluations = 0
        point_counts = PREFIT_GRID_SWEEP_LEVEL_POINTS[varying_count]

        for point_count in point_counts:
            axes: dict[str, np.ndarray] = {}
            for name in parameter_names:
                lower, upper = active_ranges[name]
                if np.isclose(lower, upper):
                    axes[name] = np.asarray([lower], dtype=float)
                else:
                    axes[name] = np.linspace(
                        lower,
                        upper,
                        point_count,
                        dtype=float,
                    )

            for coordinate in product(
                *(axes[name] for name in parameter_names)
            ):
                trial_params = lmfit_params.copy()
                for name, value in zip(
                    parameter_names, coordinate, strict=False
                ):
                    trial_params[name].set(value=float(value))
                residuals = np.asarray(objective(trial_params), dtype=float)
                score = float(np.sum(residuals**2))
                total_evaluations += 1
                if score < best_score:
                    best_score = score
                    best_values = {
                        name: float(value)
                        for name, value in zip(
                            parameter_names,
                            coordinate,
                            strict=False,
                        )
                    }

            next_ranges: dict[str, tuple[float, float]] = {}
            for name in parameter_names:
                axis_values = axes[name]
                base_lower, base_upper = base_ranges[name]
                if len(axis_values) <= 1:
                    next_ranges[name] = (base_lower, base_upper)
                    continue
                best_index = int(
                    np.argmin(np.abs(axis_values - float(best_values[name])))
                )
                if best_index <= 0:
                    lower = float(axis_values[0])
                    upper = float(axis_values[min(2, len(axis_values) - 1)])
                elif best_index >= len(axis_values) - 1:
                    lower = float(axis_values[max(len(axis_values) - 3, 0)])
                    upper = float(axis_values[-1])
                else:
                    lower = float(axis_values[best_index - 1])
                    upper = float(axis_values[best_index + 1])
                next_ranges[name] = (
                    max(base_lower, lower),
                    min(base_upper, upper),
                )
            active_ranges = next_ranges

        return PrefitGridSweepResult(
            parameter_names=parameter_names,
            best_values=best_values,
            evaluations=total_evaluations,
            best_score=best_score,
            levels=len(point_counts),
        )

    def _cluster_geometry_metadata_path_for_settings(
        self,
        settings: ProjectSettings,
    ) -> Path:
        artifact_paths = project_artifact_paths(settings)
        if settings.use_predicted_structure_weights:
            return artifact_paths.predicted_cluster_geometry_metadata_file
        return artifact_paths.cluster_geometry_metadata_file

    def _predicted_structure_cluster_bins_for_active_components(
        self,
    ) -> list:
        if not self.settings.use_predicted_structure_weights:
            return []
        predicted_components = {
            (
                str(component.structure).strip(),
                str(component.motif).strip() or "no_motif",
            )
            for component in self.components
            if str(component.motif).strip().startswith("predicted_rank")
        }
        if not predicted_components:
            return []
        return self.project_manager.predicted_structure_cluster_bins(
            self.paths.project_dir,
            included_components=predicted_components,
        )

    def run_fit(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        *,
        method: str = "leastsq",
        max_nfev: int = 10000,
    ) -> PrefitFitResult:
        if self.settings.model_only_mode:
            raise ValueError(
                "Prefit is disabled in Model Only Mode. Disable Model Only "
                "Mode and load experimental SAXS data to run a fit."
            )
        if self.experimental_data is None:
            raise ValueError(
                "Prefit requires experimental SAXS data before a fit can be run."
            )
        validate_prefit_parameter_identifiability(
            parameter_entries or self.parameter_entries
        )
        lmfit_params, entries = build_prefit_lmfit_parameters(
            parameter_entries or self.parameter_entries
        )
        q_values = self._component_q_values()
        model_data = self._model_data_for_q_values(q_values)
        experimental = np.interp(
            q_values,
            self.experimental_data.q_values,
            self.experimental_data.intensities,
        )
        solvent_data = (
            self._solvent_trace_for_q_values(q_values)
            if self.solvent_data is not None
            else np.zeros_like(q_values)
        )
        lmfit_model = self._lmfit_model_function()
        extra_inputs = self._lmfit_extra_inputs()
        grid_sweep_result: PrefitGridSweepResult | None = None
        optimization_strategy = f"lmfit {method}"

        def objective(active_params: Parameters) -> np.ndarray:
            params = active_params.valuesdict()
            model = lmfit_model(
                q_values,
                solvent_data,
                model_data,
                *extra_inputs,
                **params,
            )
            return constrained_prefit_residuals(
                experimental,
                np.asarray(model, dtype=float),
            )

        grid_sweep_result = self._coarse_to_fine_grid_sweep(
            lmfit_params,
            objective,
            entries,
        )
        if grid_sweep_result is not None:
            for name, value in grid_sweep_result.best_values.items():
                lmfit_params[name].set(value=float(value))
            optimization_strategy = (
                f"{grid_sweep_result.strategy_label} + lmfit {method}"
            )

        result = minimize(
            objective,
            lmfit_params,
            method=method,
            max_nfev=max_nfev,
        )

        for entry in entries:
            fitted = result.params[entry.name]
            entry.value = float(fitted.value)
            if _parameter_value_expression(entry) is not None:
                entry.vary = False
                continue
            entry.vary = bool(fitted.vary)
            entry.minimum = float(fitted.min)
            entry.maximum = float(fitted.max)

        evaluation = self.evaluate(entries)
        if (
            evaluation.residuals is None
            or evaluation.experimental_intensities is None
        ):
            raise ValueError(
                "Prefit fit statistics are unavailable without experimental SAXS data."
            )
        statistics = self._evaluation_statistics(entries, evaluation)
        report_text = fit_report(result)
        if grid_sweep_result is not None:
            report_text = (
                "Coarse-to-fine grid sweep:\n"
                f"  varying parameters: {', '.join(grid_sweep_result.parameter_names)}\n"
                f"  levels: {grid_sweep_result.levels}\n"
                f"  evaluations: {grid_sweep_result.evaluations}\n"
                f"  best grid chi^2 proxy: {grid_sweep_result.best_score:.6g}\n"
                "\n"
                "LMFit report:\n"
                f"{report_text}"
            )
        fit_result = PrefitFitResult(
            parameter_entries=entries,
            evaluation=evaluation,
            fit_report=report_text,
            method=method,
            nfev=int(getattr(result, "nfev", 0) or 0),
            chi_square=statistics["chi_square"],
            reduced_chi_square=statistics["reduced_chi_square"],
            r_squared=statistics["r_squared"],
            optimization_strategy=optimization_strategy,
            grid_evaluations=(
                0
                if grid_sweep_result is None
                else int(grid_sweep_result.evaluations)
            ),
        )
        self.parameter_entries = entries
        if self.settings.autosave_prefits:
            report_path = self.save_fit(
                entries,
                evaluation=evaluation,
                fit_result=fit_result,
                method=method,
                max_nfev=max_nfev,
                autosave_prefits=self.settings.autosave_prefits,
            )
            fit_result.report_path = report_path
        return fit_result

    def save_fit(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        *,
        evaluation: PrefitEvaluation | None = None,
        fit_result: PrefitFitResult | None = None,
        method: str | None = None,
        max_nfev: int | None = None,
        autosave_prefits: bool | None = None,
    ) -> Path:
        entries = parameter_entries or self.parameter_entries
        evaluation = evaluation or self.evaluate(entries)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prefit_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = self.prefit_dir / f"prefit_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        template_runtime_inputs = self.template_runtime_inputs_payload()
        cluster_geometry_payload = (
            None
            if self.cluster_geometry_table is None
            else self.cluster_geometry_table.to_dict()
        )
        evaluation_statistics = (
            {}
            if fit_result is not None
            else self._evaluation_statistics(entries, evaluation)
        )

        state_payload = {
            "saved_at": timestamp,
            "template_name": self.template_spec.name,
            "parameter_entries": [entry.to_dict() for entry in entries],
            "run_settings": {
                "method": method,
                "max_nfev": max_nfev,
                "optimization_strategy": (
                    None
                    if fit_result is None
                    else fit_result.optimization_strategy
                ),
                "grid_evaluations": (
                    None
                    if fit_result is None
                    else int(fit_result.grid_evaluations)
                ),
                "autosave_prefits": (
                    self.settings.autosave_prefits
                    if autosave_prefits is None
                    else bool(autosave_prefits)
                ),
            },
            "statistics": (
                {
                    "method": fit_result.method,
                    "nfev": fit_result.nfev,
                    "chi_square": fit_result.chi_square,
                    "reduced_chi_square": fit_result.reduced_chi_square,
                    "r_squared": fit_result.r_squared,
                }
                if fit_result is not None
                else evaluation_statistics
            ),
            "template_runtime_inputs": template_runtime_inputs,
            "cluster_geometry_metadata": cluster_geometry_payload,
        }
        state_path = self.prefit_dir / "prefit_state.json"
        state_path.write_text(
            json.dumps(state_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_state_path = snapshot_dir / "prefit_state.json"
        snapshot_state_path.write_text(
            json.dumps(state_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        weights_payload = []
        fit_parameters_payload: dict[str, float] = {}
        fit_parameter_meta: dict[str, dict[str, object]] = {}
        for entry in entries:
            meta = {
                "vary": entry.vary,
                "min": entry.minimum,
                "max": entry.maximum,
            }
            expression = _parameter_value_expression(entry)
            if expression is not None:
                meta["expression"] = expression
            initial_expression = _parameter_initial_value_expression(entry)
            if initial_expression is not None:
                meta["initial_expression"] = initial_expression
            if entry.category == "weight":
                weights_payload.append(
                    {
                        "structure": entry.structure,
                        "motif": entry.motif,
                        "name": entry.name,
                        "value": entry.value,
                        **meta,
                    }
                )
            else:
                fit_parameters_payload[entry.name] = entry.value
                fit_parameter_meta[entry.name] = meta

        pd_prefit_payload = {
            "saved_at": timestamp,
            "project_dir": str(self.paths.project_dir),
            "template_name": self.template_spec.name,
            "weights": weights_payload,
            "fit_parameters": fit_parameters_payload,
            "fit_parameter_meta": fit_parameter_meta,
            "component_order": [
                {
                    "structure": component.structure,
                    "motif": component.motif,
                    "name": component.param_name,
                    "profile_file": component.profile_file,
                }
                for component in self.components
            ],
            "template_runtime_inputs": template_runtime_inputs,
            "cluster_geometry_metadata": cluster_geometry_payload,
        }
        prefit_json_path = self.prefit_dir / "pd_prefit_params.json"
        prefit_json_path.write_text(
            json.dumps(pd_prefit_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_prefit_json_path = snapshot_dir / "pd_prefit_params.json"
        snapshot_prefit_json_path.write_text(
            json.dumps(pd_prefit_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.cluster_geometry_table is not None:
            save_cluster_geometry_metadata(
                snapshot_dir / self.cluster_geometry_metadata_path.name,
                self.cluster_geometry_table,
            )

        curve_path = self.prefit_dir / "latest_prefit_curve.txt"
        if (
            evaluation.experimental_intensities is not None
            and evaluation.residuals is not None
        ):
            curve_data = np.column_stack(
                [
                    evaluation.q_values,
                    evaluation.experimental_intensities,
                    evaluation.model_intensities,
                    evaluation.residuals,
                ]
            )
            curve_header = "q experimental_intensity model_intensity residual"
        else:
            curve_data = np.column_stack(
                [
                    evaluation.q_values,
                    evaluation.model_intensities,
                ]
            )
            curve_header = "q model_intensity"
        np.savetxt(
            curve_path,
            curve_data,
            header=curve_header,
            comments="",
        )
        snapshot_curve_path = snapshot_dir / "prefit_curve.txt"
        np.savetxt(
            snapshot_curve_path,
            curve_data,
            header=curve_header,
            comments="",
        )

        report_path = snapshot_dir / "prefit_report.txt"
        report_path.write_text(
            self._build_report_text(entries, fit_result, evaluation),
            encoding="utf-8",
        )
        history_path = self.sequence_history_path()
        if history_path.is_file():
            snapshot_history_path = snapshot_dir / history_path.name
            snapshot_history_path.write_text(
                history_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        self._save_project_settings()
        return report_path

    def set_template(self, template_name: str) -> None:
        self.template_spec = self._resolve_template_spec(template_name)
        self.template_module = load_template_module(
            self.template_spec.name,
            self.template_dir,
        )
        self.settings.selected_model_template = self.template_spec.name
        self.cluster_geometry_table = self._load_cluster_geometry_table()
        self._template_default_entries = (
            self._build_default_parameter_entries()
        )
        self.settings.template_reset_template = self.template_spec.name
        self.settings.template_reset_parameter_entries = [
            entry.to_dict() for entry in self._template_default_entries
        ]
        self.settings.best_prefit_template = None
        self.settings.best_prefit_parameter_entries = []
        self._save_project_settings()
        self.parameter_entries = self.load_template_reset_entries()

    def set_autosave(self, enabled: bool) -> None:
        self.settings.autosave_prefits = bool(enabled)
        self._save_project_settings()

    def set_sequence_history_enabled(self, enabled: bool) -> None:
        self.settings.prefit_sequence_history_enabled = bool(enabled)
        self._save_project_settings()

    def sequence_history_path(self) -> Path:
        return self.prefit_dir / PREFIT_SEQUENCE_HISTORY_FILENAME

    def append_sequence_history_event(
        self,
        event_type: str,
        summary: str,
        *,
        details: dict[str, object] | None = None,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        force: bool = False,
    ) -> Path | None:
        if not force and not self.settings.prefit_sequence_history_enabled:
            return None
        history_path = self.sequence_history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_prefit_sequence_history_payload(history_path)
        events = payload.setdefault("events", [])
        if not isinstance(events, list):
            events = []
            payload["events"] = events
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        event_payload: dict[str, object] = {
            "index": len(events) + 1,
            "timestamp": timestamp,
            "event_type": str(event_type).strip() or "prefit_event",
            "summary": str(summary).strip() or "Prefit history event",
        }
        if details:
            event_payload["details"] = dict(details)
        if parameter_entries is not None:
            event_payload["parameter_entries"] = [
                entry.to_dict() for entry in parameter_entries
            ]
        events.append(event_payload)
        payload["updated_at"] = timestamp
        payload["template_name"] = self.template_spec.name
        payload["distribution_id"] = distribution_id_for_settings(
            self.settings
        )
        payload["sequence_logger_enabled"] = bool(
            self.settings.prefit_sequence_history_enabled
        )
        history_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return history_path

    def _load_prefit_sequence_history_payload(
        self,
        path: Path,
    ) -> dict[str, object]:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = self._new_prefit_sequence_history_payload()
            else:
                if not isinstance(payload, dict):
                    payload = self._new_prefit_sequence_history_payload()
        else:
            payload = self._new_prefit_sequence_history_payload()
        payload.setdefault("events", [])
        payload["project_dir"] = str(self.paths.project_dir)
        payload["template_name"] = self.template_spec.name
        payload["distribution_id"] = distribution_id_for_settings(
            self.settings
        )
        payload["sequence_logger_enabled"] = bool(
            self.settings.prefit_sequence_history_enabled
        )
        return payload

    def _new_prefit_sequence_history_payload(self) -> dict[str, object]:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        return {
            "format_version": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "project_dir": str(self.paths.project_dir),
            "template_name": self.template_spec.name,
            "distribution_id": distribution_id_for_settings(self.settings),
            "sequence_logger_enabled": bool(
                self.settings.prefit_sequence_history_enabled
            ),
            "events": [],
        }

    def load_template_reset_entries(self) -> list[PrefitParameterEntry]:
        stored_entries = self._entries_from_project_payload(
            self.settings.template_reset_template,
            self.settings.template_reset_parameter_entries,
        )
        if stored_entries is not None:
            return self._apply_parameter_constraints(stored_entries)
        return self._apply_parameter_constraints(
            self._copy_entries(self._template_default_entries)
        )

    def load_best_prefit_entries(
        self,
    ) -> list[PrefitParameterEntry] | None:
        entries = self._entries_from_project_payload(
            self.settings.best_prefit_template,
            self.settings.best_prefit_parameter_entries,
        )
        if entries is None:
            return None
        return self._apply_parameter_constraints(entries)

    def save_best_prefit_entries(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> None:
        entries = parameter_entries or self.parameter_entries
        copied_entries = self._copy_entries(entries)
        self.settings.best_prefit_template = self.template_spec.name
        self.settings.best_prefit_parameter_entries = [
            entry.to_dict() for entry in copied_entries
        ]
        self._save_project_settings()

    def has_best_prefit_entries(self) -> bool:
        return self.load_best_prefit_entries() is not None

    def list_saved_states(self) -> list[str]:
        if not self.prefit_dir.is_dir():
            return []
        state_names = [
            path.name
            for path in self.prefit_dir.iterdir()
            if path.is_dir() and (path / "prefit_state.json").is_file()
        ]
        return sorted(state_names, reverse=True)

    def list_saved_state_options(self) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        for state_name in self.list_saved_states():
            payload: dict[str, object] = {}
            state_path = self.prefit_dir / state_name / "prefit_state.json"
            try:
                loaded_payload = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded_payload, dict):
                    payload = loaded_payload
            except (OSError, json.JSONDecodeError):
                payload = {}
            options.append(
                (
                    self._saved_state_display_label(state_name, payload),
                    state_name,
                )
            )
        return options

    def load_saved_state(self, state_name: str) -> PrefitSavedState:
        state_dir = self.prefit_dir / state_name
        state_path = state_dir / "prefit_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"No prefit_state.json file was found in {state_dir}."
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        run_settings = payload.get("run_settings", {})
        cluster_geometry_table = self._load_saved_state_cluster_geometry_table(
            state_dir,
            payload,
        )
        parameter_entries = [
            PrefitParameterEntry.from_dict(entry)
            for entry in payload.get("parameter_entries", [])
        ]
        expected_signature = self._entry_signature(
            self._build_default_parameter_entries(
                cluster_geometry_table=cluster_geometry_table,
            )
        )
        if self._entry_signature(parameter_entries) != expected_signature:
            raise ValueError(
                f"The saved prefit snapshot {state_name} is not compatible "
                "with the current project component layout."
            )
        return PrefitSavedState(
            name=state_dir.name,
            path=state_dir,
            saved_at=str(payload.get("saved_at", state_dir.name)),
            template_name=str(payload.get("template_name", "")).strip(),
            parameter_entries=self._apply_parameter_constraints(
                parameter_entries,
                default_entries=self._build_default_parameter_entries(
                    cluster_geometry_table=cluster_geometry_table,
                ),
            ),
            cluster_geometry_table=cluster_geometry_table,
            method=_optional_str(run_settings.get("method")),
            max_nfev=_optional_int(run_settings.get("max_nfev")),
            autosave_prefits=(
                bool(run_settings.get("autosave_prefits"))
                if "autosave_prefits" in run_settings
                else None
            ),
        )

    def recommend_scale_settings(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        *,
        span_factor: float = 10.0,
    ) -> PrefitScaleRecommendation:
        if self.settings.model_only_mode:
            raise ValueError(
                "Scale recommendations are unavailable in Model Only Mode."
            )
        if self.experimental_data is None:
            raise ValueError(
                "Scale recommendations require experimental SAXS data."
            )
        entries = parameter_entries or self.parameter_entries
        scale_entry = next(
            (entry for entry in entries if entry.name == "scale"),
            None,
        )
        if scale_entry is None:
            raise ValueError(
                "The current SAXS template does not define a scale parameter."
            )
        if _parameter_value_expression(scale_entry) is not None:
            raise ValueError(
                "Scale recommendations are unavailable when scale is linked "
                "to another parameter expression."
            )
        offset_entry = next(
            (entry for entry in entries if entry.name == "offset"),
            None,
        )
        if (
            offset_entry is not None
            and _parameter_value_expression(offset_entry) is not None
        ):
            raise ValueError(
                "Scale recommendations are unavailable when offset is linked "
                "to another parameter expression."
            )
        current_offset = (
            float(offset_entry.value) if offset_entry is not None else None
        )
        recommended_entries = self._copy_entries(entries)
        for entry in recommended_entries:
            if entry.name == "scale":
                entry.value = 1.0
            elif entry.name == "offset":
                entry.value = 0.0
        evaluation = self.evaluate(recommended_entries)
        offset_value = next(
            (
                float(entry.value)
                for entry in recommended_entries
                if entry.name == "offset"
            ),
            0.0,
        )
        solvent_contribution = (
            np.asarray(evaluation.solvent_contribution, dtype=float)
            if evaluation.solvent_contribution is not None
            else np.zeros_like(evaluation.model_intensities, dtype=float)
        )
        target = np.asarray(
            evaluation.experimental_intensities
            - offset_value
            - solvent_contribution,
            dtype=float,
        )
        model = np.asarray(
            evaluation.model_intensities - offset_value - solvent_contribution,
            dtype=float,
        )
        mask = np.isfinite(target) & np.isfinite(model)
        if not np.any(mask):
            raise ValueError(
                "A scale recommendation is not available because the current "
                "model and experimental traces do not overlap on a usable "
                "positive intensity range."
            )
        masked_target = np.asarray(target[mask], dtype=float)
        masked_model = np.asarray(model[mask], dtype=float)
        if np.count_nonzero(np.isfinite(masked_model)) < 2:
            raise ValueError(
                "A scale recommendation requires at least two finite SAXS "
                "model points."
            )
        centered_model = masked_model - float(np.nanmean(masked_model))
        centered_target = masked_target - float(np.nanmean(masked_target))
        recommended_offset: float | None = None
        if offset_entry is not None:
            design_matrix = np.column_stack(
                [
                    masked_model,
                    np.ones_like(masked_model, dtype=float),
                ]
            )
            coefficients, *_ = np.linalg.lstsq(
                design_matrix,
                masked_target,
                rcond=None,
            )
            recommended_scale = float(coefficients[0])
            recommended_offset = float(coefficients[1])
        else:
            numerator = float(np.dot(masked_model, masked_target))
            denominator = float(np.dot(masked_model, masked_model))
            recommended_scale = (
                numerator / denominator if denominator > 0.0 else float("nan")
            )
        if not np.isfinite(recommended_scale) or recommended_scale <= 0.0:
            numerator = float(np.dot(centered_model, centered_target))
            denominator = float(np.dot(centered_model, centered_model))
            adjustment_factor = (
                numerator / denominator if denominator > 0.0 else float("nan")
            )
            if not np.isfinite(adjustment_factor) or adjustment_factor <= 0.0:
                adjustment_factor = float(
                    np.median(
                        np.abs(masked_target)
                        / np.maximum(np.abs(masked_model), 1e-30)
                    )
                )
            if not np.isfinite(adjustment_factor) or adjustment_factor <= 0.0:
                target_span = float(
                    np.nanmax(masked_target) - np.nanmin(masked_target)
                )
                model_span = float(
                    np.nanmax(masked_model) - np.nanmin(masked_model)
                )
                if model_span > 0.0 and target_span > 0.0:
                    adjustment_factor = target_span / model_span
            recommended_scale = float(adjustment_factor)
            if offset_entry is not None:
                recommended_offset = float(
                    np.nanmean(
                        masked_target - recommended_scale * masked_model
                    )
                )
        if not np.isfinite(recommended_scale) or recommended_scale <= 0.0:
            raise ValueError(
                "A positive scale recommendation could not be estimated from "
                "the current model and experimental traces."
            )
        raw_current_scale = float(scale_entry.value)
        current_scale = raw_current_scale
        if current_scale <= 0.0:
            current_scale = max(float(scale_entry.maximum), 1.0) / span_factor
        recommended_scale = max(float(recommended_scale), 1e-12)
        adjustment_factor = (
            recommended_scale / current_scale
            if current_scale > 0.0
            else float("nan")
        )
        recommended_minimum = max(recommended_scale / span_factor, 1e-12)
        recommended_maximum = max(
            recommended_scale * span_factor,
            recommended_scale * 1.5,
            float(scale_entry.maximum),
        )
        recommended_offset_minimum: float | None = None
        recommended_offset_maximum: float | None = None
        if offset_entry is not None and recommended_offset is not None:
            target_span = float(
                np.nanmax(masked_target) - np.nanmin(masked_target)
            )
            offset_padding = max(
                target_span / span_factor,
                abs(float(recommended_offset)) / span_factor,
                1e-12,
            )
            recommended_offset_minimum = min(
                float(offset_entry.minimum),
                float(recommended_offset) - offset_padding,
            )
            recommended_offset_maximum = max(
                float(offset_entry.maximum),
                float(recommended_offset) + offset_padding,
            )
        return PrefitScaleRecommendation(
            current_scale=raw_current_scale,
            recommended_scale=recommended_scale,
            recommended_minimum=min(recommended_minimum, recommended_scale),
            recommended_maximum=max(recommended_maximum, recommended_scale),
            current_offset=current_offset,
            recommended_offset=recommended_offset,
            recommended_offset_minimum=recommended_offset_minimum,
            recommended_offset_maximum=recommended_offset_maximum,
            adjustment_factor=adjustment_factor,
            points_used=int(np.count_nonzero(mask)),
        )

    def volume_fraction_estimator_target(self) -> tuple[str, str] | None:
        parameter_names = {
            str(parameter.name).strip()
            for parameter in self.template_spec.parameters
            if str(parameter.name).strip()
        }
        for candidate in SOLUTE_VOLUME_FRACTION_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate, "solute"
        for candidate in SOLVENT_VOLUME_FRACTION_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate, "solvent"
        return None

    def supports_volume_fraction_estimator(self) -> bool:
        return self.volume_fraction_estimator_target() is not None

    def solvent_weight_estimator_target(self) -> str | None:
        parameter_names = {
            str(parameter.name).strip()
            for parameter in self.template_spec.parameters
            if str(parameter.name).strip()
        }
        for candidate in SOLVENT_WEIGHT_PARAMETER_NAMES:
            if candidate in parameter_names:
                return candidate
        return None

    def supports_cluster_geometry_metadata(self) -> bool:
        return bool(self.template_spec.cluster_geometry_support.supported)

    def _allowed_cluster_geometry_approximations(self) -> tuple[str, ...]:
        return (
            self.template_spec.cluster_geometry_support.allowed_sf_approximations
        )

    def allowed_cluster_geometry_approximations(self) -> tuple[str, ...]:
        return self._allowed_cluster_geometry_approximations()

    def _synchronize_cluster_geometry_table(
        self,
        table: ClusterGeometryMetadataTable,
    ) -> bool:
        return synchronize_cluster_geometry_table(
            table,
            allowed_sf_approximations=(
                self._allowed_cluster_geometry_approximations()
            ),
        )

    def cluster_geometry_rows(self) -> list[ClusterGeometryMetadataRow]:
        if self.cluster_geometry_table is None:
            return []
        return copy_cluster_geometry_rows(self.cluster_geometry_table.rows)

    def cluster_geometry_active_radii_type(self) -> str:
        if self.cluster_geometry_table is None:
            return DEFAULT_RADIUS_TYPE
        return str(self.cluster_geometry_table.active_radii_type).strip() or (
            DEFAULT_RADIUS_TYPE
        )

    def cluster_geometry_active_ionic_radius_type(self) -> str:
        if self.cluster_geometry_table is None:
            return DEFAULT_IONIC_RADIUS_TYPE
        return (
            str(self.cluster_geometry_table.active_ionic_radius_type).strip()
            or DEFAULT_IONIC_RADIUS_TYPE
        )

    def set_cluster_geometry_active_radii_type(self, radii_type: str) -> None:
        working_table = self._working_cluster_geometry_table()
        working_table.active_radii_type = radii_type
        self._apply_cluster_geometry_table(working_table)

    def set_cluster_geometry_active_ionic_radius_type(
        self,
        ionic_radius_type: str,
    ) -> None:
        working_table = self._working_cluster_geometry_table()
        working_table.active_ionic_radius_type = ionic_radius_type
        self._apply_cluster_geometry_table(working_table)

    def cluster_geometry_mapping_options(self) -> list[tuple[str, str]]:
        return [
            (
                component.param_name,
                (
                    f"{component.param_name} "
                    f"({component.structure}/{component.motif})"
                ),
            )
            for component in self.components
        ]

    def cluster_geometry_status_text(self) -> str:
        if not self.supports_cluster_geometry_metadata():
            return (
                "The active template does not use per-cluster geometry "
                "metadata."
            )
        if self.settings.resolved_clusters_dir is None:
            return (
                "Select a clusters directory in Project Setup before "
                "computing cluster geometry metadata."
            )
        if (
            self.cluster_geometry_table is None
            or not self.cluster_geometry_table.rows
        ):
            return (
                "Compute cluster geometry metadata to provide per-cluster "
                "effective radii and structure-factor recommendations for "
                f"{self.template_spec.display_name}."
            )
        try:
            runtime_inputs = sorted(self.template_runtime_inputs().keys())
        except Exception as exc:
            return str(exc)
        return (
            f"Loaded cluster geometry metadata for "
            f"{len(self.cluster_geometry_table.rows)} clusters. "
            f"Active radii mode: {self.cluster_geometry_active_radii_type()} "
            f"({self.cluster_geometry_active_ionic_radius_type()} ionic). "
            "Runtime inputs: " + ", ".join(runtime_inputs)
        )

    def compute_cluster_geometry_table(self) -> ClusterGeometryMetadataTable:
        return self.compute_cluster_geometry_table_with_progress()

    def compute_cluster_geometry_table_with_progress(
        self,
        *,
        progress_callback=None,
    ) -> ClusterGeometryMetadataTable:
        clusters_dir = self.settings.resolved_clusters_dir
        if clusters_dir is None:
            raise ValueError(
                "Select a clusters directory in Project Setup before "
                "computing cluster geometry metadata."
            )
        table = compute_cluster_geometry_metadata(
            clusters_dir,
            extra_cluster_bins=(
                self._predicted_structure_cluster_bins_for_active_components()
            ),
            template_name=self.template_spec.name,
            active_radii_type=self.cluster_geometry_active_radii_type(),
            active_ionic_radius_type=(
                self.cluster_geometry_active_ionic_radius_type()
            ),
            allowed_sf_approximations=(
                self._allowed_cluster_geometry_approximations()
            ),
            progress_callback=progress_callback,
        )
        apply_default_component_mapping(table.rows, self.components)
        validate_positive_cluster_geometry_table(table)
        self.cluster_geometry_table = table
        self._save_cluster_geometry_table()
        self._refresh_dynamic_cluster_geometry_parameter_entries()
        return table

    def set_cluster_geometry_rows(
        self,
        rows: list[ClusterGeometryMetadataRow],
        *,
        preserve_geometry_entries: bool = False,
    ) -> None:
        working_table = self._working_cluster_geometry_table()
        working_table.rows = copy_cluster_geometry_rows(rows)
        self._apply_cluster_geometry_table(
            working_table,
            preserve_geometry_entries=preserve_geometry_entries,
        )

    def set_cluster_geometry_state(
        self,
        *,
        rows: list[ClusterGeometryMetadataRow],
        active_radii_type: str,
        active_ionic_radius_type: str,
        preserve_geometry_entries: bool = False,
    ) -> None:
        working_table = self._working_cluster_geometry_table()
        working_table.active_radii_type = active_radii_type
        working_table.active_ionic_radius_type = active_ionic_radius_type
        working_table.rows = copy_cluster_geometry_rows(rows)
        self._apply_cluster_geometry_table(
            working_table,
            preserve_geometry_entries=preserve_geometry_entries,
        )

    def restore_cluster_geometry_table(
        self,
        table: ClusterGeometryMetadataTable | None,
    ) -> None:
        if table is None:
            self.cluster_geometry_table = None
            if self.cluster_geometry_metadata_path.is_file():
                self.cluster_geometry_metadata_path.unlink()
            self._refresh_dynamic_cluster_geometry_parameter_entries()
            return
        self._apply_cluster_geometry_table(table)

    def template_runtime_inputs(self) -> dict[str, np.ndarray]:
        required_names = {
            *self.template_spec.extra_lmfit_inputs,
            *self.template_spec.cluster_geometry_support.runtime_input_names,
        }
        if not required_names:
            return {}
        available_inputs = self._available_template_runtime_inputs()
        missing = [
            name
            for name in sorted(required_names)
            if name not in available_inputs
        ]
        if missing:
            raise ValueError(
                "The selected template requires runtime metadata inputs that "
                "are not available: " + ", ".join(missing)
            )
        return {
            name: np.asarray(available_inputs[name], dtype=float)
            for name in sorted(required_names)
        }

    def template_runtime_inputs_payload(self) -> dict[str, list[float]]:
        return {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in self.template_runtime_inputs().items()
        }

    def _build_default_parameter_entries(
        self,
        cluster_geometry_table: ClusterGeometryMetadataTable | None = None,
    ) -> list[PrefitParameterEntry]:
        entries: list[PrefitParameterEntry] = []
        for component in self.components:
            value = float(component.weight_value)
            minimum = 0.0
            maximum = 1.0 if value == 0.0 else value * 1.1
            entries.append(
                PrefitParameterEntry(
                    structure=component.structure,
                    motif=component.motif,
                    name=component.param_name,
                    value=value,
                    vary=False,
                    minimum=minimum,
                    maximum=maximum,
                    category="weight",
                )
            )
        entries.extend(
            self._build_cluster_geometry_parameter_entries(
                cluster_geometry_table=cluster_geometry_table,
            )
        )
        for parameter in self.template_spec.parameters:
            entries.append(
                PrefitParameterEntry(
                    structure="",
                    motif="",
                    name=parameter.name,
                    value=parameter.initial_value,
                    vary=parameter.vary,
                    minimum=parameter.minimum,
                    maximum=parameter.maximum,
                    category="fit",
                )
            )
        return entries

    def _build_cluster_geometry_parameter_entries(
        self,
        *,
        cluster_geometry_table: ClusterGeometryMetadataTable | None = None,
    ) -> list[PrefitParameterEntry]:
        capability = self.template_spec.cluster_geometry_support
        if not capability.dynamic_parameters:
            return []

        table = cluster_geometry_table or self.cluster_geometry_table
        if table is None or not table.rows:
            return []

        working_table = ClusterGeometryMetadataTable.from_dict(table.to_dict())
        self._synchronize_cluster_geometry_table(working_table)
        apply_default_component_mapping(working_table.rows, self.components)
        row_by_parameter = self._cluster_geometry_row_by_parameter(
            working_table.rows
        )
        if row_by_parameter is None:
            return []

        entries: list[PrefitParameterEntry] = []
        for component in self.components:
            row = row_by_parameter.get(component.param_name)
            if row is None:
                return []
            if row.sf_approximation == "ellipsoid":
                parameter_specs = [
                    (
                        capability.ellipsoid_parameter_prefixes[0],
                        row.active_semiaxis_a,
                    ),
                    (
                        capability.ellipsoid_parameter_prefixes[1],
                        row.active_semiaxis_b,
                    ),
                    (
                        capability.ellipsoid_parameter_prefixes[2],
                        row.active_semiaxis_c,
                    ),
                ]
            else:
                parameter_specs = [
                    (
                        capability.sphere_parameter_prefix,
                        row.effective_radius,
                    )
                ]
            for prefix, raw_value in parameter_specs:
                value = float(raw_value)
                minimum = MINIMUM_POSITIVE_RADIUS
                maximum = self._default_geometry_parameter_maximum(value)
                entries.append(
                    PrefitParameterEntry(
                        structure=component.structure,
                        motif=component.motif,
                        name=f"{prefix}_{component.param_name}",
                        value=value,
                        vary=False,
                        minimum=minimum,
                        maximum=maximum,
                        category="geometry",
                    )
                )
        return entries

    def _cluster_geometry_row_by_parameter(
        self,
        rows: list[ClusterGeometryMetadataRow],
    ) -> dict[str, ClusterGeometryMetadataRow] | None:
        row_by_parameter: dict[str, ClusterGeometryMetadataRow] = {}
        for row in rows:
            mapped_parameter = (
                str(row.mapped_parameter).strip()
                if row.mapped_parameter is not None
                else ""
            )
            if not mapped_parameter:
                return None
            if mapped_parameter in row_by_parameter:
                return None
            row_by_parameter[mapped_parameter] = row
        if any(
            component.param_name not in row_by_parameter
            for component in self.components
        ):
            return None
        return row_by_parameter

    @staticmethod
    def _default_geometry_parameter_maximum(value: float) -> float:
        magnitude = abs(float(value))
        if magnitude <= 0.0:
            return 1.0
        return max(magnitude * 2.0, magnitude + 1.0)

    @staticmethod
    def _ensure_entry_bounds_include_current_values(
        entries: list[PrefitParameterEntry],
    ) -> None:
        for entry in entries:
            lower = float(entry.minimum)
            upper = float(entry.maximum)
            value = float(entry.value)
            if lower > upper:
                lower, upper = upper, lower
            if value < lower:
                lower = value
            if value > upper:
                upper = value
            entry.minimum = lower
            entry.maximum = upper

    def _refresh_dynamic_cluster_geometry_parameter_entries(
        self,
        *,
        preserve_geometry_entries: bool = False,
    ) -> None:
        refreshed_defaults = self._build_default_parameter_entries()
        self.parameter_entries = self._merge_parameter_entries(
            self.parameter_entries,
            refreshed_defaults,
            preserve_geometry_entries=preserve_geometry_entries,
        )
        self._template_default_entries = refreshed_defaults
        self.settings.template_reset_template = self.template_spec.name
        self.settings.template_reset_parameter_entries = [
            entry.to_dict() for entry in refreshed_defaults
        ]
        if (
            self.settings.best_prefit_template == self.template_spec.name
            and self.settings.best_prefit_parameter_entries
        ):
            best_entries = [
                PrefitParameterEntry.from_dict(entry)
                for entry in self.settings.best_prefit_parameter_entries
            ]
            if not self._has_matching_entry_signature(best_entries):
                self.settings.best_prefit_template = None
                self.settings.best_prefit_parameter_entries = []
        self._save_project_settings()

    @staticmethod
    def _merge_parameter_entries(
        existing_entries: list[PrefitParameterEntry],
        default_entries: list[PrefitParameterEntry],
        *,
        preserve_geometry_entries: bool = False,
    ) -> list[PrefitParameterEntry]:
        existing_by_name = {
            entry.name: entry for entry in existing_entries if entry.name
        }
        merged_entries: list[PrefitParameterEntry] = []
        for default_entry in default_entries:
            existing_entry = existing_by_name.get(default_entry.name)
            preserve_existing_entry = existing_entry is not None and (
                default_entry.category != "geometry"
                or preserve_geometry_entries
            )
            if not preserve_existing_entry:
                merged_entries.append(
                    PrefitParameterEntry.from_dict(default_entry.to_dict())
                )
                continue
            merged_entries.append(
                PrefitParameterEntry(
                    structure=default_entry.structure,
                    motif=default_entry.motif,
                    name=default_entry.name,
                    value=float(existing_entry.value),
                    vary=bool(existing_entry.vary),
                    minimum=float(existing_entry.minimum),
                    maximum=float(existing_entry.maximum),
                    category=default_entry.category,
                    value_expression=_parameter_value_expression(
                        existing_entry
                    ),
                    initial_value_expression=(
                        _parameter_initial_value_expression(existing_entry)
                    ),
                )
            )
        defaults_by_name = {
            entry.name: entry for entry in default_entries if entry.name
        }
        return [
            SAXSPrefitWorkflow._apply_parameter_entry_constraints(
                entry,
                defaults_by_name.get(entry.name),
            )
            for entry in merged_entries
        ]

    def _ensure_project_parameter_presets(self) -> None:
        dirty = False
        if (
            self._entries_from_project_payload(
                self.settings.template_reset_template,
                self.settings.template_reset_parameter_entries,
            )
            is None
        ):
            self.settings.template_reset_template = self.template_spec.name
            self.settings.template_reset_parameter_entries = [
                entry.to_dict() for entry in self._template_default_entries
            ]
            dirty = True
        if (
            self.settings.best_prefit_parameter_entries
            and self._entries_from_project_payload(
                self.settings.best_prefit_template,
                self.settings.best_prefit_parameter_entries,
            )
            is None
        ):
            self.settings.best_prefit_template = None
            self.settings.best_prefit_parameter_entries = []
            dirty = True
        if dirty:
            self._save_project_settings()

    def _entries_from_project_payload(
        self,
        template_name: str | None,
        payload: list[dict[str, object]],
    ) -> list[PrefitParameterEntry] | None:
        if template_name != self.template_spec.name or not payload:
            return None
        entries = [PrefitParameterEntry.from_dict(entry) for entry in payload]
        if not self._has_matching_entry_signature(entries):
            return None
        return self._apply_parameter_constraints(entries)

    def _apply_parameter_constraints(
        self,
        entries: list[PrefitParameterEntry],
        *,
        default_entries: list[PrefitParameterEntry] | None = None,
    ) -> list[PrefitParameterEntry]:
        defaults = default_entries or self._template_default_entries
        defaults_by_name = {
            entry.name: entry for entry in defaults if entry.name
        }
        return [
            self._apply_parameter_entry_constraints(
                entry,
                defaults_by_name.get(entry.name),
            )
            for entry in entries
        ]

    @staticmethod
    def _apply_parameter_entry_constraints(
        entry: PrefitParameterEntry,
        default_entry: PrefitParameterEntry | None = None,
    ) -> PrefitParameterEntry:
        constrained = PrefitParameterEntry.from_dict(entry.to_dict())
        if constrained.name not in SOLVENT_WEIGHT_PARAMETER_NAMES:
            return constrained
        minimum = 0.0
        maximum = 1.0
        if default_entry is not None:
            minimum = max(minimum, float(default_entry.minimum))
            maximum = min(maximum, float(default_entry.maximum))
        bounded_minimum = max(float(constrained.minimum), minimum)
        bounded_maximum = min(float(constrained.maximum), maximum)
        if bounded_maximum < bounded_minimum:
            bounded_minimum = minimum
            bounded_maximum = maximum
        constrained.minimum = bounded_minimum
        constrained.maximum = bounded_maximum
        constrained.value = min(
            max(float(constrained.value), constrained.minimum),
            constrained.maximum,
        )
        return constrained

    def _has_matching_entry_signature(
        self,
        entries: list[PrefitParameterEntry],
    ) -> bool:
        return self._entry_signature(entries) == self._entry_signature(
            self._template_default_entries
        )

    @staticmethod
    def _copy_entries(
        entries: list[PrefitParameterEntry],
    ) -> list[PrefitParameterEntry]:
        return [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in entries
        ]

    @staticmethod
    def _entry_signature(
        entries: list[PrefitParameterEntry],
    ) -> list[tuple[str, str, str]]:
        return [
            (entry.structure, entry.motif, entry.name) for entry in entries
        ]

    def _load_components(self) -> list[PrefitComponent]:
        if not self.component_map_path.is_file():
            if self.settings.use_predicted_structure_weights:
                predicted_state = (
                    self.project_manager.inspect_predicted_structures(
                        self.paths.project_dir
                    )
                )
                if predicted_state.dataset_file is None:
                    raise FileNotFoundError(
                        "Predicted Structures mode is enabled, but no "
                        "Cluster Dynamics ML prediction bundle was found in "
                        "this project. Open Tools > Cluster Dynamics > Open "
                        "Cluster Dynamics (ML), run a prediction, then "
                        "rebuild the SAXS components."
                    )
                raise FileNotFoundError(
                    "Predicted Structures mode is enabled, but the "
                    "predicted-structure SAXS components have not been built "
                    "for this project. Rebuild SAXS components in Project "
                    "Setup with Use Predicted Structure Weights enabled."
                )
            raise FileNotFoundError(
                "No md_saxs_map.json file was found. Build the project "
                "components from the Project Setup tab first."
            )
        if not self.prior_weights_path.is_file():
            if self.settings.use_predicted_structure_weights:
                predicted_state = (
                    self.project_manager.inspect_predicted_structures(
                        self.paths.project_dir
                    )
                )
                if predicted_state.dataset_file is None:
                    raise FileNotFoundError(
                        "Predicted Structures mode is enabled, but no "
                        "Cluster Dynamics ML prediction bundle was found in "
                        "this project. Open Tools > Cluster Dynamics > Open "
                        "Cluster Dynamics (ML), run a prediction, then "
                        "generate the prior weights."
                    )
                raise FileNotFoundError(
                    "Predicted Structures mode is enabled, but the "
                    "predicted-structure prior weights have not been "
                    "generated for this project. Generate Prior Weights in "
                    "Project Setup with Use Predicted Structure Weights enabled."
                )
            raise FileNotFoundError(
                "No md_prior_weights.json file was found. Build the project "
                "components from the Project Setup tab first."
            )
        map_payload = json.loads(
            self.component_map_path.read_text(encoding="utf-8")
        )
        prior_payload = json.loads(
            self.prior_weights_path.read_text(encoding="utf-8")
        )
        saxs_map = map_payload.get("saxs_map", {})
        structures = prior_payload.get("structures", {})
        components: list[PrefitComponent] = []
        index = 0
        for structure in sorted(saxs_map, key=_natural_sort_key):
            motif_map = saxs_map[structure]
            for motif in sorted(motif_map, key=_natural_sort_key):
                profile_file = str(motif_map[motif])
                profile_path = self.component_dir / profile_file
                raw_data = np.loadtxt(profile_path, comments="#")
                q_values = np.asarray(raw_data[:, 0], dtype=float)
                intensities = np.asarray(raw_data[:, 1], dtype=float)
                components.append(
                    PrefitComponent(
                        structure=structure,
                        motif=motif,
                        param_name=f"w{index}",
                        weight_value=float(
                            structures.get(structure, {})
                            .get(motif, {})
                            .get(
                                "normalized_weight",
                                structures.get(structure, {})
                                .get(motif, {})
                                .get("weight", 0.0),
                            )
                        ),
                        profile_file=profile_file,
                        q_values=q_values,
                        intensities=intensities,
                    )
                )
                index += 1
        if not components:
            raise ValueError(
                "No SAXS component profiles were found for the selected "
                "project."
            )
        return components

    def _load_experimental_trace(self):
        if self.settings.model_only_mode:
            return None
        try:
            return self.project_manager.load_experimental_data(self.settings)
        except Exception:
            return None

    def _load_solvent_trace(self) -> np.ndarray | None:
        if self.settings.model_only_mode:
            return None
        q_values = self._component_q_values_from_candidates()
        solvent_summary = self.project_manager.load_solvent_data(self.settings)
        if solvent_summary is not None:
            return np.interp(
                q_values,
                np.asarray(solvent_summary.q_values, dtype=float),
                np.asarray(solvent_summary.intensities, dtype=float),
            )
        for candidate in sorted(
            self.paths.experimental_data_dir.glob("solv_*")
        ):
            if candidate.is_file():
                raw_data = np.loadtxt(candidate, comments="#")
                return np.interp(
                    q_values,
                    np.asarray(raw_data[:, 0], dtype=float),
                    np.asarray(raw_data[:, 1], dtype=float),
                )
        return None

    def _component_q_values(self) -> np.ndarray:
        return self._component_q_values_from_candidates(self.components)

    def _supported_component_q_range(self) -> tuple[float, float]:
        supported = load_built_component_q_range(
            self.paths.project_dir,
            include_predicted_structures=(
                self.settings.use_predicted_structure_weights
            ),
            component_dir=self.component_dir,
        )
        if supported is None:
            components = getattr(self, "components", [])
            if self.experimental_data is not None:
                q_values = np.asarray(
                    self.experimental_data.q_values, dtype=float
                )
            elif components:
                q_values = np.asarray(components[0].q_values, dtype=float)
            else:
                source = np.asarray(
                    [
                        value
                        for value in (self.settings.q_min, self.settings.q_max)
                        if value is not None
                    ],
                    dtype=float,
                )
                if source.size == 0:
                    raise ValueError(
                        "No q-range is available for the current SAXS project."
                    )
                q_values = source
            return (float(np.min(q_values)), float(np.max(q_values)))
        return supported

    def _requested_q_bounds(
        self,
        source_q_values: np.ndarray,
    ) -> tuple[float, float]:
        q_values = (
            np.asarray(self.experimental_data.q_values, dtype=float)
            if self.experimental_data is not None
            else np.asarray(source_q_values, dtype=float)
        )
        if q_values.size == 0:
            raise ValueError("No SAXS component q-values are available.")
        return effective_q_range_for_settings(self.settings, q_values)

    def _ensure_requested_q_range_supported(
        self,
        source_q_values: np.ndarray,
    ) -> tuple[float, float]:
        q_values = np.asarray(source_q_values, dtype=float)
        requested_min, requested_max = self._requested_q_bounds(q_values)
        supported_min, supported_max = self._supported_component_q_range()
        requested_min, requested_max = (
            normalize_requested_q_range_to_supported(
                requested_min,
                requested_max,
                supported_min,
                supported_max,
            )
        )
        tolerance = q_range_boundary_tolerance(
            supported_min,
            supported_max,
        )
        if requested_min < (supported_min - tolerance) or requested_max > (
            supported_max + tolerance
        ):
            raise ValueError(
                "The requested q-range "
                f"{requested_min:.6g} to {requested_max:.6g} extends beyond "
                "the q-range covered by the built SAXS model components "
                f"({supported_min:.6g} to {supported_max:.6g}). Recompute "
                "the SAXS model components in Project Setup for the updated "
                "q-range to be applied."
            )
        return requested_min, requested_max

    def _component_q_values_from_candidates(
        self,
        candidates: list[PrefitComponent] | None = None,
    ) -> np.ndarray:
        if candidates:
            source_q_values = np.asarray(candidates[0].q_values, dtype=float)
        else:
            component_files = sorted(self.component_dir.glob("*.txt"))
            if not component_files:
                if self.experimental_data is not None:
                    source_q_values = np.asarray(
                        self.experimental_data.q_values,
                        dtype=float,
                    )
                else:
                    raise ValueError(
                        "No SAXS component q-grid is available yet. Build the "
                        "SAXS components in Project Setup before previewing the model."
                    )
            else:
                raw_data = np.loadtxt(component_files[0], comments="#")
                source_q_values = np.asarray(raw_data[:, 0], dtype=float)

        requested_min, requested_max = (
            self._ensure_requested_q_range_supported(source_q_values)
        )
        if self.settings.use_experimental_grid:
            return self._nearest_supported_q_values(
                source_q_values,
                requested_min,
                requested_max,
            )
        mask = (source_q_values >= requested_min) & (
            source_q_values <= requested_max
        )
        filtered_q = np.asarray(source_q_values[mask], dtype=float)
        if filtered_q.size == 0:
            raise ValueError(
                "The requested q-range does not overlap the built SAXS "
                "component q-grid."
            )
        if self.settings.q_points is not None and self.settings.q_points > 1:
            return np.linspace(
                float(requested_min),
                float(requested_max),
                int(self.settings.q_points),
            )
        return filtered_q

    @staticmethod
    def _nearest_supported_q_values(
        q_values: np.ndarray,
        q_min: float,
        q_max: float,
    ) -> np.ndarray:
        q_values = np.asarray(q_values, dtype=float)
        start_index = int(np.argmin(np.abs(q_values - q_min)))
        end_index = int(np.argmin(np.abs(q_values - q_max)))
        lo_index, hi_index = sorted((start_index, end_index))
        cropped_q = np.asarray(q_values[lo_index : hi_index + 1], dtype=float)
        if cropped_q.size == 0:
            raise ValueError(
                "The requested q-range does not overlap the built SAXS "
                "component q-grid."
            )
        return cropped_q

    def _component_intensities_on_grid(
        self,
        component: PrefitComponent,
        q_values: np.ndarray,
    ) -> np.ndarray:
        source_q = np.asarray(component.q_values, dtype=float)
        source_i = np.asarray(component.intensities, dtype=float)
        target_q = np.asarray(q_values, dtype=float)
        if source_q.shape == target_q.shape and np.allclose(
            source_q, target_q
        ):
            return source_i
        return np.interp(target_q, source_q, source_i)

    def _model_data_for_q_values(
        self,
        q_values: np.ndarray,
    ) -> list[np.ndarray]:
        return [
            self._component_intensities_on_grid(component, q_values)
            for component in self.components
        ]

    def _solvent_trace_for_q_values(
        self,
        q_values: np.ndarray,
    ) -> np.ndarray | None:
        if self.solvent_data is None:
            return None
        if self.solvent_data.shape == np.asarray(
            q_values, dtype=float
        ).shape and (
            np.allclose(
                self._component_q_values(), np.asarray(q_values, dtype=float)
            )
        ):
            return np.asarray(self.solvent_data, dtype=float)
        solvent_summary = self.project_manager.load_solvent_data(self.settings)
        if solvent_summary is not None:
            return np.interp(
                np.asarray(q_values, dtype=float),
                np.asarray(solvent_summary.q_values, dtype=float),
                np.asarray(solvent_summary.intensities, dtype=float),
            )
        component_files = sorted(
            self.paths.experimental_data_dir.glob("solv_*")
        )
        for candidate in component_files:
            if candidate.is_file():
                raw_data = np.loadtxt(candidate, comments="#")
                return np.interp(
                    np.asarray(q_values, dtype=float),
                    np.asarray(raw_data[:, 0], dtype=float),
                    np.asarray(raw_data[:, 1], dtype=float),
                )
        return np.asarray(self.solvent_data, dtype=float)

    def _resolve_template_spec(
        self,
        template_name: str | None,
    ) -> TemplateSpec:
        selected = (
            template_name
            or self.settings.selected_model_template
            or self._default_template_name()
        )
        self.settings.selected_model_template = selected
        self._save_project_settings()
        return load_template_spec(selected, self.template_dir)

    def _working_cluster_geometry_table(
        self,
    ) -> ClusterGeometryMetadataTable:
        if self.cluster_geometry_table is None:
            return ClusterGeometryMetadataTable(
                source_clusters_dir=(
                    str(self.settings.resolved_clusters_dir)
                    if self.settings.resolved_clusters_dir is not None
                    else None
                ),
                template_name=self.template_spec.name,
            )
        return ClusterGeometryMetadataTable.from_dict(
            self.cluster_geometry_table.to_dict()
        )

    def _apply_cluster_geometry_table(
        self,
        table: ClusterGeometryMetadataTable,
        *,
        preserve_geometry_entries: bool = False,
    ) -> None:
        working_table = ClusterGeometryMetadataTable.from_dict(table.to_dict())
        working_table.template_name = self.template_spec.name
        self._synchronize_cluster_geometry_table(working_table)
        apply_default_component_mapping(
            working_table.rows,
            self.components,
        )
        validate_positive_cluster_geometry_table(working_table)
        self.cluster_geometry_table = working_table
        self._save_cluster_geometry_table()
        self._refresh_dynamic_cluster_geometry_parameter_entries(
            preserve_geometry_entries=preserve_geometry_entries,
        )

    def _default_template_name(self) -> str:
        templates = self.available_templates()
        if not templates:
            raise ValueError("No SAXS model templates are available.")
        for preferred in (
            "template_pd_likelihood_monosq_decoupled",
            "template_pd_likelihood_monosq",
        ):
            for template in templates:
                if template.name == preferred:
                    return template.name
        return templates[0].name

    def _lmfit_model_function(self):
        try:
            return getattr(
                self.template_module,
                self.template_spec.lmfit_model_name,
            )
        except AttributeError as exc:
            raise AttributeError(
                f"Template {self.template_spec.name} does not define "
                f"{self.template_spec.lmfit_model_name}."
            ) from exc

    def _evaluate_solvent_contribution(
        self,
        q_values: np.ndarray,
        *,
        solvent_data: np.ndarray,
        model_data: list[np.ndarray],
        params: dict[str, float],
        extra_inputs: list[np.ndarray],
    ) -> np.ndarray | None:
        if self.solvent_data is None:
            return None
        lmfit_model = self._lmfit_model_function()
        isolated_params = dict(params)
        if "offset" in isolated_params:
            isolated_params["offset"] = 0.0
        zero_model_data = [
            np.zeros_like(np.asarray(component, dtype=float))
            for component in model_data
        ]
        contribution = lmfit_model(
            q_values,
            np.asarray(solvent_data, dtype=float),
            zero_model_data,
            *extra_inputs,
            **isolated_params,
        )
        return np.asarray(contribution, dtype=float)

    def _evaluate_structure_factor_trace(
        self,
        q_values: np.ndarray,
        *,
        solvent_data: np.ndarray,
        model_data: list[np.ndarray],
        params: dict[str, float],
        extra_inputs: list[np.ndarray],
    ) -> np.ndarray | None:
        structure_factor_function = getattr(
            self.template_module,
            "structure_factor_profile",
            None,
        )
        if structure_factor_function is None:
            return None
        try:
            structure_factor = structure_factor_function(
                q_values,
                np.asarray(solvent_data, dtype=float),
                model_data,
                *extra_inputs,
                **params,
            )
        except Exception:
            return None
        structure_factor_array = np.asarray(structure_factor, dtype=float)
        if structure_factor_array.shape != np.asarray(q_values).shape:
            return None
        if not np.all(np.isfinite(structure_factor_array)):
            return None
        return structure_factor_array

    def _lmfit_extra_inputs(self) -> list[np.ndarray]:
        runtime_inputs = self.template_runtime_inputs()
        return [
            np.asarray(runtime_inputs[input_name], dtype=float)
            for input_name in self.template_spec.extra_lmfit_inputs
        ]

    def _load_cluster_geometry_table(
        self,
    ) -> ClusterGeometryMetadataTable | None:
        if not self.cluster_geometry_metadata_path.is_file():
            return None
        table = load_cluster_geometry_metadata(
            self.cluster_geometry_metadata_path
        )
        table.template_name = self.template_spec.name
        dirty = self._synchronize_cluster_geometry_table(table)
        if apply_default_component_mapping(table.rows, self.components):
            dirty = True
        if dirty:
            save_cluster_geometry_metadata(
                self.cluster_geometry_metadata_path,
                table,
            )
        return table

    def _load_saved_state_cluster_geometry_table(
        self,
        state_dir: Path,
        payload: dict[str, object],
    ) -> ClusterGeometryMetadataTable | None:
        inline_payload = payload.get("cluster_geometry_metadata")
        if isinstance(inline_payload, dict):
            table = ClusterGeometryMetadataTable.from_dict(inline_payload)
            table.template_name = self.template_spec.name
            self._synchronize_cluster_geometry_table(table)
            apply_default_component_mapping(table.rows, self.components)
            return table

        snapshot_table_path = (
            state_dir / self.cluster_geometry_metadata_path.name
        )
        if snapshot_table_path.is_file():
            table = load_cluster_geometry_metadata(snapshot_table_path)
            table.template_name = self.template_spec.name
            self._synchronize_cluster_geometry_table(table)
            apply_default_component_mapping(table.rows, self.components)
            return table

        snapshot_prefit_payload_path = state_dir / "pd_prefit_params.json"
        if snapshot_prefit_payload_path.is_file():
            snapshot_payload = json.loads(
                snapshot_prefit_payload_path.read_text(encoding="utf-8")
            )
            snapshot_inline_payload = snapshot_payload.get(
                "cluster_geometry_metadata"
            )
            if isinstance(snapshot_inline_payload, dict):
                table = ClusterGeometryMetadataTable.from_dict(
                    snapshot_inline_payload
                )
                table.template_name = self.template_spec.name
                self._synchronize_cluster_geometry_table(table)
                apply_default_component_mapping(table.rows, self.components)
                return table
        return None

    def _save_cluster_geometry_table(self) -> None:
        if self.cluster_geometry_table is None:
            return
        self._synchronize_cluster_geometry_table(self.cluster_geometry_table)
        self.cluster_geometry_table.template_name = self.template_spec.name
        if (
            self.cluster_geometry_table.source_clusters_dir is None
            and self.settings.resolved_clusters_dir is not None
        ):
            self.cluster_geometry_table.source_clusters_dir = str(
                self.settings.resolved_clusters_dir
            )
        save_cluster_geometry_metadata(
            self.cluster_geometry_metadata_path,
            self.cluster_geometry_table,
        )

    def _available_template_runtime_inputs(self) -> dict[str, np.ndarray]:
        runtime_inputs: dict[str, np.ndarray] = {}
        cluster_geometry_inputs = self._cluster_geometry_runtime_inputs()
        runtime_inputs.update(cluster_geometry_inputs)
        return runtime_inputs

    def _cluster_geometry_runtime_inputs(self) -> dict[str, np.ndarray]:
        capability = self.template_spec.cluster_geometry_support
        if not capability.supported:
            return {}
        if (
            self.cluster_geometry_table is None
            or not self.cluster_geometry_table.rows
        ):
            raise ValueError(
                "This template requires computed cluster geometry metadata. "
                "Use the Cluster Geometry Metadata section in SAXS Prefit "
                "to compute and map effective radii before updating the "
                "model or running a fit."
            )
        if self._synchronize_cluster_geometry_table(
            self.cluster_geometry_table
        ):
            self._save_cluster_geometry_table()
        validate_positive_cluster_geometry_table(self.cluster_geometry_table)
        rows = copy_cluster_geometry_rows(self.cluster_geometry_table.rows)
        if apply_default_component_mapping(rows, self.components):
            self.cluster_geometry_table.rows = rows
            self._save_cluster_geometry_table()
        row_by_parameter: dict[str, ClusterGeometryMetadataRow] = {}
        duplicate_parameters: list[str] = []
        unmapped_clusters: list[str] = []
        for row in rows:
            mapped_parameter = (
                str(row.mapped_parameter).strip()
                if row.mapped_parameter is not None
                else ""
            )
            if not mapped_parameter:
                unmapped_clusters.append(row.cluster_id)
                continue
            if mapped_parameter in row_by_parameter:
                duplicate_parameters.append(mapped_parameter)
                continue
            row_by_parameter[mapped_parameter] = row
        if unmapped_clusters:
            raise ValueError(
                "Cluster geometry metadata rows must be mapped to the "
                "component weight parameters before this template can run. "
                "Unmapped clusters: " + ", ".join(unmapped_clusters)
            )
        if duplicate_parameters:
            raise ValueError(
                "Cluster geometry metadata maps multiple rows to the same "
                "component weight parameter: "
                + ", ".join(sorted(set(duplicate_parameters)))
            )
        missing_components = [
            component.param_name
            for component in self.components
            if component.param_name not in row_by_parameter
        ]
        if missing_components:
            raise ValueError(
                "Cluster geometry metadata is missing mappings for the "
                "component weight parameters: " + ", ".join(missing_components)
            )
        runtime_inputs = {
            binding.runtime_name: [] for binding in capability.runtime_bindings
        }
        for component in self.components:
            row = row_by_parameter[component.param_name]
            for binding in capability.runtime_bindings:
                runtime_inputs[binding.runtime_name].append(
                    _coerce_runtime_metadata_value(
                        getattr(row, binding.metadata_field),
                        runtime_name=binding.runtime_name,
                        cluster_id=row.cluster_id,
                    )
                )
        return {
            name: np.asarray(values, dtype=float)
            for name, values in runtime_inputs.items()
        }

    def _build_report_text(
        self,
        entries: list[PrefitParameterEntry],
        fit_result: PrefitFitResult | None,
        evaluation: PrefitEvaluation,
    ) -> str:
        lines = [
            f"Project: {self.paths.project_dir}",
            f"Template: {self.template_spec.name}",
            f"Saved: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Parameters:",
        ]
        for entry in entries:
            lines.append(
                f"  {entry.name}: value={entry.value:.6g}, "
                f"vary={entry.vary}, min={entry.minimum:.6g}, "
                f"max={entry.maximum:.6g}"
            )
        lines.extend(
            [
                "",
                "Fit statistics:",
                f"  q points: {len(evaluation.q_values)}",
            ]
        )
        if fit_result is None and (
            evaluation.experimental_intensities is None
            or evaluation.residuals is None
        ):
            lines.extend(
                [
                    "  mode: model_only",
                    "  experimental_data: unavailable",
                    "  fit_metrics: unavailable",
                ]
            )
        if fit_result is not None:
            lines.extend(
                [
                    f"  method: {fit_result.method}",
                    f"  optimization_strategy: {fit_result.optimization_strategy}",
                    f"  grid_evaluations: {fit_result.grid_evaluations}",
                    f"  nfev: {fit_result.nfev}",
                    f"  chi_square: {fit_result.chi_square:.6g}",
                    "  reduced_chi_square: "
                    f"{fit_result.reduced_chi_square:.6g}",
                    f"  r_squared: {fit_result.r_squared:.6g}",
                    "",
                    "LMFit report:",
                    fit_result.fit_report,
                ]
            )
        return "\n".join(lines) + "\n"


def _coerce_runtime_metadata_value(
    value: object,
    *,
    runtime_name: str,
    cluster_id: str,
) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(
            f"Cluster geometry metadata for {cluster_id} provides a "
            f"non-numeric value {value!r} for runtime input {runtime_name!r}."
        ) from exc


__all__ = [
    "PrefitComponent",
    "PrefitEvaluation",
    "PrefitFitResult",
    "PrefitParameterEntry",
    "PrefitScaleRecommendation",
    "PrefitSavedState",
    "SAXSPrefitWorkflow",
]
