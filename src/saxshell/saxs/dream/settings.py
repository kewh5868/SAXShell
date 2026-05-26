from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DREAM_SEARCH_FILTER_PRESETS: dict[str, dict[str, object]] = {
    "less_aggressive": {
        "nchains": 4,
        "niterations": 5000,
        "burnin_percent": 15,
        "nseedchains": 24,
        "crossover_burnin": 500,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 20.0,
        "posterior_top_n": 1000,
        "violin_sample_source": "filtered_posterior",
    },
    "medium": {
        "nchains": 4,
        "niterations": 10000,
        "burnin_percent": 20,
        "nseedchains": 40,
        "crossover_burnin": 1000,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
    "more_aggressive": {
        "nchains": 8,
        "niterations": 20000,
        "burnin_percent": 25,
        "nseedchains": 80,
        "crossover_burnin": 2000,
        "posterior_filter_mode": "top_percent_logp",
        "posterior_top_percent": 5.0,
        "posterior_top_n": 250,
        "violin_sample_source": "filtered_posterior",
    },
    "legacy_gui_default": {
        "nchains": 4,
        "niterations": 10000,
        "burnin_percent": 20,
        "nseedchains": 40,
        "crossover_burnin": 1000,
        "history_thin": 10,
        "adapt_crossover": True,
        "lamb": 0.05,
        "zeta": 1e-12,
        "snooker": 0.1,
        "p_gamma_unity": 0.2,
        "verbose": True,
        "parallel": True,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
    "legacy_saxs_notebook": {
        "nchains": 50,
        "niterations": 15000,
        "burnin_percent": 40,
        "nseedchains": 500,
        "crossover_burnin": 1000,
        "history_thin": 10,
        "adapt_crossover": True,
        "lamb": 0.05,
        "zeta": 1e-12,
        "snooker": 0.1,
        "p_gamma_unity": 0.2,
        "verbose": True,
        "parallel": True,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
    "legacy_kwhite_long": {
        "nchains": 50,
        "niterations": 20000,
        "burnin_percent": 40,
        "nseedchains": 500,
        "crossover_burnin": 1000,
        "history_thin": 10,
        "adapt_crossover": True,
        "lamb": 0.05,
        "zeta": 1e-12,
        "snooker": 0.1,
        "p_gamma_unity": 0.2,
        "verbose": True,
        "parallel": True,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
    "legacy_tchaney_production": {
        "nchains": 25,
        "niterations": 50000,
        "burnin_percent": 10,
        "nseedchains": 250,
        "crossover_burnin": 1000,
        "history_thin": 10,
        "adapt_crossover": True,
        "lamb": 0.05,
        "zeta": 1e-12,
        "snooker": 0.1,
        "p_gamma_unity": 0.2,
        "verbose": True,
        "parallel": True,
        "posterior_filter_mode": "all_post_burnin",
        "posterior_top_percent": 10.0,
        "posterior_top_n": 500,
        "violin_sample_source": "filtered_posterior",
    },
}

DREAM_SEARCH_FILTER_PRESET_LABELS: dict[str, str] = {
    "less_aggressive": "Less Aggressive",
    "medium": "Medium",
    "more_aggressive": "More Aggressive",
    "legacy_gui_default": "Legacy GUI Default",
    "legacy_saxs_notebook": "Legacy SAXS Notebook",
    "legacy_kwhite_long": "Legacy KWhite Long",
    "legacy_tchaney_production": "Legacy TChaney Production",
}


def normalize_dream_search_filter_preset(value: object) -> str:
    text = str(value or "").strip()
    return text or "custom"


def dream_search_filter_preset_label(value: object) -> str:
    normalized = normalize_dream_search_filter_preset(value)
    if normalized == "custom":
        return "Custom"
    return DREAM_SEARCH_FILTER_PRESET_LABELS.get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def format_dream_search_filter_preset(value: object) -> str:
    normalized = normalize_dream_search_filter_preset(value)
    label = dream_search_filter_preset_label(normalized)
    if normalized == "custom":
        return label
    return f"{label} ({normalized})"


@dataclass(slots=True)
class DreamRunSettings:
    nchains: int = 4
    niterations: int = 10000
    burnin_percent: int = 20
    restart: bool = False
    verbose: bool = True
    verbose_output_interval_seconds: float = 5.0
    parallel: bool = True
    nseedchains: int = 40
    adapt_crossover: bool = True
    crossover_burnin: int = 1000
    lamb: float = 0.05
    zeta: float = 1e-12
    snooker: float = 0.1
    p_gamma_unity: float = 0.2
    history_thin: int = 10
    history_file: str | None = None
    model_name: str | None = None
    run_label: str = "dream"
    search_filter_preset: str = "medium"
    fit_q_min: float | None = None
    fit_q_max: float | None = None
    bestfit_method: str = "map"
    posterior_filter_mode: str = "all_post_burnin"
    posterior_top_percent: float = 10.0
    posterior_top_n: int = 500
    auto_select_best_posterior_filter: bool = True
    credible_interval_low: float = 16.0
    credible_interval_high: float = 84.0
    violin_parameter_mode: str = "varying_parameters"
    violin_sample_source: str = "filtered_posterior"
    violin_weight_order: str = "weight_index"
    violin_value_scale_mode: str = "parameter_value"
    violin_selected_parameter: str = ""
    stoichiometry_target_elements_text: str = ""
    stoichiometry_target_ratio_text: str = ""
    stoichiometry_filter_enabled: bool = False
    stoichiometry_tolerance_percent: float = 5.0
    violin_palette: str = "Blues"
    violin_custom_color: str = "#4c72b0"
    violin_point_color: str = "tab:red"
    violin_interval_color: str = "#8c8c8c"
    violin_median_color: str = "#4d4d4d"
    violin_outline_color: str = "#000000"
    violin_outline_width: float = 0.8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DreamRunSettings":
        nchains = int(_get_first(payload, "nchains", "nChains", default=4))
        return cls(
            nchains=nchains,
            niterations=int(
                _get_first(
                    payload,
                    "niterations",
                    "nIterations",
                    default=10000,
                )
            ),
            burnin_percent=int(
                _get_first(
                    payload,
                    "burnin_percent",
                    "Burn-in (%)",
                    default=20,
                )
            ),
            restart=_coerce_bool(
                _get_first(payload, "restart", default=False)
            ),
            verbose=_coerce_bool(
                _get_first(payload, "verbose", "Verbose", default=True)
            ),
            verbose_output_interval_seconds=max(
                float(
                    _get_first(
                        payload,
                        "verbose_output_interval_seconds",
                        "verbose_output_interval_s",
                        "Verbose Output Interval (s)",
                        default=5.0,
                    )
                ),
                0.1,
            ),
            parallel=_coerce_bool(
                _get_first(payload, "parallel", "Parallel", default=True)
            ),
            nseedchains=int(
                _get_first(
                    payload,
                    "nseedchains",
                    default=max(10 * nchains, 1),
                )
            ),
            adapt_crossover=_coerce_bool(
                _get_first(
                    payload,
                    "adapt_crossover",
                    default=True,
                )
            ),
            crossover_burnin=int(
                _get_first(
                    payload,
                    "crossover_burnin",
                    "crossover_burn_in",
                    default=1000,
                )
            ),
            lamb=float(_get_first(payload, "lamb", "lambda", default=0.05)),
            zeta=float(_get_first(payload, "zeta", default=1e-12)),
            snooker=float(_get_first(payload, "snooker", default=0.1)),
            p_gamma_unity=float(
                _get_first(
                    payload,
                    "p_gamma_unity",
                    "p_gamma_unit",
                    default=0.2,
                )
            ),
            history_thin=int(_get_first(payload, "history_thin", default=10)),
            history_file=_optional_str(
                _get_first(payload, "history_file", default=None)
            ),
            model_name=(
                str(_get_first(payload, "model_name", "Model Name"))
                if _get_first(payload, "model_name", "Model Name") is not None
                else None
            ),
            run_label=str(payload.get("run_label", "dream")),
            search_filter_preset=str(
                _get_first(
                    payload,
                    "search_filter_preset",
                    default="medium",
                )
            ),
            fit_q_min=_optional_float(
                _get_first(payload, "fit_q_min", default=None)
            ),
            fit_q_max=_optional_float(
                _get_first(payload, "fit_q_max", default=None)
            ),
            bestfit_method=str(
                _get_first(
                    payload,
                    "bestfit_method",
                    default="map",
                )
            ),
            posterior_filter_mode=str(
                _get_first(
                    payload,
                    "posterior_filter_mode",
                    default="all_post_burnin",
                )
            ),
            posterior_top_percent=float(
                _get_first(
                    payload,
                    "posterior_top_percent",
                    default=10.0,
                )
            ),
            posterior_top_n=int(
                _get_first(
                    payload,
                    "posterior_top_n",
                    default=500,
                )
            ),
            auto_select_best_posterior_filter=_coerce_bool(
                _get_first(
                    payload,
                    "auto_select_best_posterior_filter",
                    default=True,
                )
            ),
            credible_interval_low=float(
                _get_first(
                    payload,
                    "credible_interval_low",
                    default=16.0,
                )
            ),
            credible_interval_high=float(
                _get_first(
                    payload,
                    "credible_interval_high",
                    default=84.0,
                )
            ),
            violin_parameter_mode=str(
                _get_first(
                    payload,
                    "violin_parameter_mode",
                    default="varying_parameters",
                )
            ),
            violin_sample_source=str(
                _get_first(
                    payload,
                    "violin_sample_source",
                    default="filtered_posterior",
                )
            ),
            violin_weight_order=str(
                _get_first(
                    payload,
                    "violin_weight_order",
                    default="weight_index",
                )
            ),
            violin_value_scale_mode=str(
                _get_first(
                    payload,
                    "violin_value_scale_mode",
                    default="parameter_value",
                )
            ),
            violin_selected_parameter=(
                _optional_str(
                    _get_first(
                        payload,
                        "violin_selected_parameter",
                        default="",
                    )
                )
                or ""
            ),
            stoichiometry_target_elements_text=(
                _optional_str(
                    _get_first(
                        payload,
                        "stoichiometry_target_elements_text",
                        default="",
                    )
                )
                or ""
            ),
            stoichiometry_target_ratio_text=(
                _optional_str(
                    _get_first(
                        payload,
                        "stoichiometry_target_ratio_text",
                        default="",
                    )
                )
                or ""
            ),
            stoichiometry_filter_enabled=_coerce_bool(
                _get_first(
                    payload,
                    "stoichiometry_filter_enabled",
                    default=False,
                )
            ),
            stoichiometry_tolerance_percent=float(
                _get_first(
                    payload,
                    "stoichiometry_tolerance_percent",
                    default=5.0,
                )
            ),
            violin_palette=str(
                _get_first(
                    payload,
                    "violin_palette",
                    default="Blues",
                )
            ),
            violin_custom_color=str(
                _get_first(
                    payload,
                    "violin_custom_color",
                    default="#4c72b0",
                )
            ),
            violin_point_color=str(
                _get_first(
                    payload,
                    "violin_point_color",
                    default="tab:red",
                )
            ),
            violin_interval_color=str(
                _get_first(
                    payload,
                    "violin_interval_color",
                    default="#8c8c8c",
                )
            ),
            violin_median_color=str(
                _get_first(
                    payload,
                    "violin_median_color",
                    default="#4d4d4d",
                )
            ),
            violin_outline_color=str(
                _get_first(
                    payload,
                    "violin_outline_color",
                    default="#000000",
                )
            ),
            violin_outline_width=float(
                _get_first(
                    payload,
                    "violin_outline_width",
                    default=0.8,
                )
            ),
        )


DREAM_RUN_SETTING_NAMES = tuple(
    field.name for field in fields(DreamRunSettings)
)

POSTERIOR_FILTER_SETTING_NAMES = (
    "bestfit_method",
    "posterior_filter_mode",
    "posterior_top_percent",
    "posterior_top_n",
    "auto_select_best_posterior_filter",
    "credible_interval_low",
    "credible_interval_high",
    "violin_parameter_mode",
    "violin_sample_source",
    "violin_weight_order",
    "violin_value_scale_mode",
    "violin_selected_parameter",
    "stoichiometry_target_elements_text",
    "stoichiometry_target_ratio_text",
    "stoichiometry_filter_enabled",
    "stoichiometry_tolerance_percent",
    "violin_palette",
    "violin_custom_color",
    "violin_point_color",
    "violin_interval_color",
    "violin_median_color",
    "violin_outline_color",
    "violin_outline_width",
)

DREAM_SAMPLER_SETTING_NAMES = tuple(
    name
    for name in DREAM_RUN_SETTING_NAMES
    if name not in POSTERIOR_FILTER_SETTING_NAMES
)


def dream_run_settings_to_dict(
    settings: DreamRunSettings,
    *,
    include_posterior_filter_settings: bool = True,
) -> dict[str, object]:
    payload = settings.to_dict()
    if include_posterior_filter_settings:
        return payload
    return {name: payload[name] for name in DREAM_SAMPLER_SETTING_NAMES}


@dataclass(slots=True)
class PosteriorFilterSettings:
    bestfit_method: str = "map"
    posterior_filter_mode: str = "all_post_burnin"
    posterior_top_percent: float = 10.0
    posterior_top_n: int = 500
    auto_select_best_posterior_filter: bool = True
    credible_interval_low: float = 16.0
    credible_interval_high: float = 84.0
    violin_parameter_mode: str = "varying_parameters"
    violin_sample_source: str = "filtered_posterior"
    violin_weight_order: str = "weight_index"
    violin_value_scale_mode: str = "parameter_value"
    violin_selected_parameter: str = ""
    stoichiometry_target_elements_text: str = ""
    stoichiometry_target_ratio_text: str = ""
    stoichiometry_filter_enabled: bool = False
    stoichiometry_tolerance_percent: float = 5.0
    violin_palette: str = "Blues"
    violin_custom_color: str = "#4c72b0"
    violin_point_color: str = "tab:red"
    violin_interval_color: str = "#8c8c8c"
    violin_median_color: str = "#4d4d4d"
    violin_outline_color: str = "#000000"
    violin_outline_width: float = 0.8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: dict[str, object]
    ) -> "PosteriorFilterSettings":
        settings = DreamRunSettings.from_dict(payload)
        return cls.from_run_settings(settings)

    @classmethod
    def from_run_settings(
        cls,
        settings: DreamRunSettings,
    ) -> "PosteriorFilterSettings":
        return cls(
            **{
                name: getattr(settings, name)
                for name in POSTERIOR_FILTER_SETTING_NAMES
            }
        )

    def apply_to_run_settings(
        self,
        settings: DreamRunSettings | None = None,
    ) -> DreamRunSettings:
        target = (
            DreamRunSettings.from_dict(settings.to_dict())
            if settings is not None
            else DreamRunSettings()
        )
        for name in POSTERIOR_FILTER_SETTING_NAMES:
            setattr(target, name, getattr(self, name))
        return target


def _get_first(
    payload: dict[str, object],
    *keys: str,
    default: object = None,
) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return bool(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return float(text)


def load_dream_settings(path: str | Path) -> DreamRunSettings:
    settings_path = Path(path)
    if not settings_path.is_file():
        return DreamRunSettings()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    return DreamRunSettings.from_dict(payload)


def save_dream_settings(
    path: str | Path,
    settings: DreamRunSettings,
    *,
    include_posterior_filter_settings: bool = True,
) -> Path:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            dream_run_settings_to_dict(
                settings,
                include_posterior_filter_settings=(
                    include_posterior_filter_settings
                ),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return settings_path


__all__ = [
    "DREAM_SAMPLER_SETTING_NAMES",
    "DREAM_SEARCH_FILTER_PRESET_LABELS",
    "DREAM_SEARCH_FILTER_PRESETS",
    "DreamRunSettings",
    "PosteriorFilterSettings",
    "dream_search_filter_preset_label",
    "dream_run_settings_to_dict",
    "format_dream_search_filter_preset",
    "load_dream_settings",
    "normalize_dream_search_filter_preset",
    "save_dream_settings",
]
