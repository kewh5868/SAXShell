from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class DreamRunSettings:
    nchains: int = 4
    niterations: int = 10000
    burnin_percent: int = 20
    restart: bool = False
    verbose: bool = True
    verbose_output_interval_seconds: float = 1.0
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
                        default=1.0,
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


def load_dream_settings(path: str | Path) -> DreamRunSettings:
    settings_path = Path(path)
    if not settings_path.is_file():
        return DreamRunSettings()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    return DreamRunSettings.from_dict(payload)


def save_dream_settings(
    path: str | Path,
    settings: DreamRunSettings,
) -> Path:
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return settings_path


__all__ = [
    "DreamRunSettings",
    "load_dream_settings",
    "save_dream_settings",
]
