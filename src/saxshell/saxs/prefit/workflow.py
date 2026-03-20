from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from lmfit import Parameters, fit_report, minimize

from saxshell.saxs._model_templates import (
    TemplateSpec,
    list_template_specs,
    load_template_module,
    load_template_spec,
)
from saxshell.saxs.project_manager import (
    SAXSProjectManager,
    build_project_paths,
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PrefitParameterEntry":
        return cls(
            structure=str(payload.get("structure", "")),
            motif=str(payload.get("motif", "")),
            name=str(payload.get("name", "")),
            value=float(payload.get("value", 0.0)),
            vary=bool(payload.get("vary", True)),
            minimum=float(payload.get("minimum", payload.get("min", 0.0))),
            maximum=float(payload.get("maximum", payload.get("max", 0.0))),
            category=str(payload.get("category", "fit")),
        )


@dataclass(slots=True)
class PrefitEvaluation:
    q_values: np.ndarray
    experimental_intensities: np.ndarray
    model_intensities: np.ndarray
    residuals: np.ndarray
    solvent_intensities: np.ndarray | None = None
    solvent_contribution: np.ndarray | None = None


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
    report_path: Path | None = None


@dataclass(slots=True)
class PrefitScaleRecommendation:
    current_scale: float
    recommended_scale: float
    recommended_minimum: float
    recommended_maximum: float
    adjustment_factor: float
    points_used: int


@dataclass(slots=True)
class PrefitSavedState:
    name: str
    path: Path
    saved_at: str
    template_name: str
    parameter_entries: list[PrefitParameterEntry]
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
        self.experimental_data = self.project_manager.load_experimental_data(
            self.settings
        )
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
        self.component_map_path = self.paths.project_dir / "md_saxs_map.json"
        self.prior_weights_path = (
            self.paths.project_dir / "md_prior_weights.json"
        )
        self.solvent_data = self._load_solvent_trace()
        self.components = self._load_components()
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

    def load_parameter_entries(self) -> list[PrefitParameterEntry]:
        best_entries = self.load_best_prefit_entries()
        if best_entries is not None:
            return best_entries
        state_path = self.paths.prefit_dir / "prefit_state.json"
        if state_path.is_file():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            entries = payload.get("parameter_entries", [])
            parsed_entries = [
                PrefitParameterEntry.from_dict(entry) for entry in entries
            ]
            if self._has_matching_entry_signature(parsed_entries):
                return parsed_entries
        return self.load_template_reset_entries()

    def evaluate(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> PrefitEvaluation:
        entries = parameter_entries or self.parameter_entries
        q_values = self._component_q_values()
        experimental = np.interp(
            q_values,
            self.experimental_data.q_values,
            self.experimental_data.intensities,
        )
        params = {entry.name: float(entry.value) for entry in entries}
        model_data = [component.intensities for component in self.components]
        solvent_data = (
            self.solvent_data
            if self.solvent_data is not None
            else np.zeros_like(q_values)
        )
        solvent_intensities = (
            np.asarray(solvent_data, dtype=float)
            if self.solvent_data is not None
            else None
        )
        model_intensities = self._lmfit_model_function()(
            q_values,
            solvent_data,
            model_data,
            **params,
        )
        solvent_contribution = self._evaluate_solvent_contribution(
            q_values,
            solvent_data=solvent_data,
            model_data=model_data,
            params=params,
        )
        residuals = model_intensities - experimental
        return PrefitEvaluation(
            q_values=q_values,
            experimental_intensities=experimental,
            model_intensities=np.asarray(model_intensities, dtype=float),
            residuals=np.asarray(residuals, dtype=float),
            solvent_intensities=solvent_intensities,
            solvent_contribution=solvent_contribution,
        )

    def run_fit(
        self,
        parameter_entries: list[PrefitParameterEntry] | None = None,
        *,
        method: str = "leastsq",
        max_nfev: int = 10000,
    ) -> PrefitFitResult:
        entries = [
            PrefitParameterEntry.from_dict(entry.to_dict())
            for entry in (parameter_entries or self.parameter_entries)
        ]
        q_values = self._component_q_values()
        experimental = np.interp(
            q_values,
            self.experimental_data.q_values,
            self.experimental_data.intensities,
        )
        solvent_data = (
            self.solvent_data
            if self.solvent_data is not None
            else np.zeros_like(q_values)
        )
        model_data = [component.intensities for component in self.components]
        lmfit_model = self._lmfit_model_function()
        lmfit_params = Parameters()
        for entry in entries:
            lmfit_params.add(
                entry.name,
                value=float(entry.value),
                vary=bool(entry.vary),
                min=float(entry.minimum),
                max=float(entry.maximum),
            )

        def objective(active_params: Parameters) -> np.ndarray:
            params = active_params.valuesdict()
            model = lmfit_model(
                q_values,
                solvent_data,
                model_data,
                **params,
            )
            return np.asarray(model, dtype=float) - experimental

        result = minimize(
            objective,
            lmfit_params,
            method=method,
            max_nfev=max_nfev,
        )

        for entry in entries:
            fitted = result.params[entry.name]
            entry.value = float(fitted.value)
            entry.vary = bool(fitted.vary)
            entry.minimum = float(fitted.min)
            entry.maximum = float(fitted.max)

        evaluation = self.evaluate(entries)
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
        fit_result = PrefitFitResult(
            parameter_entries=entries,
            evaluation=evaluation,
            fit_report=fit_report(result),
            method=method,
            nfev=int(getattr(result, "nfev", 0) or 0),
            chi_square=chi_square,
            reduced_chi_square=reduced_chi_square,
            r_squared=r_squared,
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
        self.paths.prefit_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = self.paths.prefit_dir / f"prefit_{timestamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        state_payload = {
            "saved_at": timestamp,
            "template_name": self.template_spec.name,
            "parameter_entries": [entry.to_dict() for entry in entries],
            "run_settings": {
                "method": method,
                "max_nfev": max_nfev,
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
                else {}
            ),
        }
        state_path = self.paths.prefit_dir / "prefit_state.json"
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
        }
        prefit_json_path = self.paths.prefit_dir / "pd_prefit_params.json"
        prefit_json_path.write_text(
            json.dumps(pd_prefit_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_prefit_json_path = snapshot_dir / "pd_prefit_params.json"
        snapshot_prefit_json_path.write_text(
            json.dumps(pd_prefit_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        curve_path = self.paths.prefit_dir / "latest_prefit_curve.txt"
        curve_data = np.column_stack(
            [
                evaluation.q_values,
                evaluation.experimental_intensities,
                evaluation.model_intensities,
                evaluation.residuals,
            ]
        )
        np.savetxt(
            curve_path,
            curve_data,
            header="q experimental_intensity model_intensity residual",
            comments="",
        )
        snapshot_curve_path = snapshot_dir / "prefit_curve.txt"
        np.savetxt(
            snapshot_curve_path,
            curve_data,
            header="q experimental_intensity model_intensity residual",
            comments="",
        )

        report_path = snapshot_dir / "prefit_report.txt"
        report_path.write_text(
            self._build_report_text(entries, fit_result, evaluation),
            encoding="utf-8",
        )
        self.project_manager.save_project(self.settings)
        return report_path

    def set_template(self, template_name: str) -> None:
        self.template_spec = self._resolve_template_spec(template_name)
        self.template_module = load_template_module(
            self.template_spec.name,
            self.template_dir,
        )
        self.settings.selected_model_template = self.template_spec.name
        self._template_default_entries = (
            self._build_default_parameter_entries()
        )
        self.settings.template_reset_template = self.template_spec.name
        self.settings.template_reset_parameter_entries = [
            entry.to_dict() for entry in self._template_default_entries
        ]
        self.settings.best_prefit_template = None
        self.settings.best_prefit_parameter_entries = []
        self.project_manager.save_project(self.settings)
        self.parameter_entries = self.load_template_reset_entries()

    def set_autosave(self, enabled: bool) -> None:
        self.settings.autosave_prefits = bool(enabled)
        self.project_manager.save_project(self.settings)

    def load_template_reset_entries(self) -> list[PrefitParameterEntry]:
        stored_entries = self._entries_from_project_payload(
            self.settings.template_reset_template,
            self.settings.template_reset_parameter_entries,
        )
        if stored_entries is not None:
            return stored_entries
        return self._copy_entries(self._template_default_entries)

    def load_best_prefit_entries(
        self,
    ) -> list[PrefitParameterEntry] | None:
        return self._entries_from_project_payload(
            self.settings.best_prefit_template,
            self.settings.best_prefit_parameter_entries,
        )

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
        self.project_manager.save_project(self.settings)

    def has_best_prefit_entries(self) -> bool:
        return self.load_best_prefit_entries() is not None

    def list_saved_states(self) -> list[str]:
        if not self.paths.prefit_dir.is_dir():
            return []
        state_names = [
            path.name
            for path in self.paths.prefit_dir.iterdir()
            if path.is_dir() and (path / "prefit_state.json").is_file()
        ]
        return sorted(state_names, reverse=True)

    def load_saved_state(self, state_name: str) -> PrefitSavedState:
        state_dir = self.paths.prefit_dir / state_name
        state_path = state_dir / "prefit_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"No prefit_state.json file was found in {state_dir}."
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        run_settings = payload.get("run_settings", {})
        parameter_entries = [
            PrefitParameterEntry.from_dict(entry)
            for entry in payload.get("parameter_entries", [])
        ]
        if not self._has_matching_entry_signature(parameter_entries):
            raise ValueError(
                f"The saved prefit snapshot {state_name} is not compatible "
                "with the current project component layout."
            )
        return PrefitSavedState(
            name=state_dir.name,
            path=state_dir,
            saved_at=str(payload.get("saved_at", state_dir.name)),
            template_name=str(payload.get("template_name", "")).strip(),
            parameter_entries=parameter_entries,
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
        entries = parameter_entries or self.parameter_entries
        scale_entry = next(
            (entry for entry in entries if entry.name == "scale"),
            None,
        )
        if scale_entry is None:
            raise ValueError(
                "The current SAXS template does not define a scale parameter."
            )
        evaluation = self.evaluate(entries)
        offset_value = next(
            (
                float(entry.value)
                for entry in entries
                if entry.name == "offset"
            ),
            0.0,
        )
        target = np.asarray(
            evaluation.experimental_intensities - offset_value,
            dtype=float,
        )
        model = np.asarray(
            evaluation.model_intensities - offset_value,
            dtype=float,
        )
        mask = np.isfinite(target) & np.isfinite(model) & (np.abs(model) > 0.0)
        positive_mask = mask & (target > 0.0) & (model > 0.0)
        if np.count_nonzero(positive_mask) >= 3:
            mask = positive_mask
        if not np.any(mask):
            raise ValueError(
                "A scale recommendation is not available because the current "
                "model and experimental traces do not overlap on a usable "
                "positive intensity range."
            )
        masked_target = np.asarray(target[mask], dtype=float)
        masked_model = np.asarray(model[mask], dtype=float)
        centered_target = masked_target - float(np.nanmin(masked_target))
        centered_model = masked_model - float(np.nanmin(masked_model))
        numerator = float(np.dot(centered_model, centered_target))
        denominator = float(np.dot(centered_model, centered_model))
        adjustment_factor = (
            numerator / denominator if denominator > 0.0 else float("nan")
        )
        if not np.isfinite(adjustment_factor) or adjustment_factor <= 0.0:
            adjustment_factor = float(
                np.median(
                    np.abs(centered_target)
                    / np.maximum(np.abs(centered_model), 1e-30)
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
        if not np.isfinite(adjustment_factor) or adjustment_factor <= 0.0:
            raise ValueError(
                "A positive scale recommendation could not be estimated from "
                "the current model and experimental traces."
            )
        current_scale = float(scale_entry.value)
        if current_scale <= 0.0:
            current_scale = max(float(scale_entry.maximum), 1.0) / span_factor
        recommended_scale = max(current_scale * adjustment_factor, 1e-12)
        recommended_minimum = max(recommended_scale / span_factor, 1e-12)
        recommended_maximum = max(
            recommended_scale * span_factor,
            recommended_scale * 1.5,
            float(scale_entry.maximum),
        )
        return PrefitScaleRecommendation(
            current_scale=float(scale_entry.value),
            recommended_scale=recommended_scale,
            recommended_minimum=min(recommended_minimum, recommended_scale),
            recommended_maximum=max(recommended_maximum, recommended_scale),
            adjustment_factor=adjustment_factor,
            points_used=int(np.count_nonzero(mask)),
        )

    def _build_default_parameter_entries(self) -> list[PrefitParameterEntry]:
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
            self.project_manager.save_project(self.settings)

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
        return entries

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
            raise FileNotFoundError(
                "No md_saxs_map.json file was found. Build the project "
                "components from the Project Setup tab first."
            )
        if not self.prior_weights_path.is_file():
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
                profile_path = (
                    self.paths.scattering_components_dir / profile_file
                )
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
                            .get("weight", 0.0)
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

    def _load_solvent_trace(self) -> np.ndarray | None:
        solvent_summary = self.project_manager.load_solvent_data(self.settings)
        if solvent_summary is not None:
            q_values = self._component_q_values_from_candidates()
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
                q_values = self._component_q_values_from_candidates()
                return np.interp(
                    q_values,
                    np.asarray(raw_data[:, 0], dtype=float),
                    np.asarray(raw_data[:, 1], dtype=float),
                )
        return None

    def _component_q_values(self) -> np.ndarray:
        return self._component_q_values_from_candidates(self.components)

    def _component_q_values_from_candidates(
        self,
        candidates: list[PrefitComponent] | None = None,
    ) -> np.ndarray:
        if candidates:
            return np.asarray(candidates[0].q_values, dtype=float)
        component_files = sorted(
            self.paths.scattering_components_dir.glob("*.txt")
        )
        if not component_files:
            return np.asarray(self.experimental_data.q_values, dtype=float)
        raw_data = np.loadtxt(component_files[0], comments="#")
        return np.asarray(raw_data[:, 0], dtype=float)

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
        self.project_manager.save_project(self.settings)
        return load_template_spec(selected, self.template_dir)

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
            **isolated_params,
        )
        return np.asarray(contribution, dtype=float)

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
        if fit_result is not None:
            lines.extend(
                [
                    f"  method: {fit_result.method}",
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


__all__ = [
    "PrefitComponent",
    "PrefitEvaluation",
    "PrefitFitResult",
    "PrefitParameterEntry",
    "PrefitScaleRecommendation",
    "PrefitSavedState",
    "SAXSPrefitWorkflow",
]
