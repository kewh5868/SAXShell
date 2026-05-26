from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "lognorm": {"loc": 0.0, "scale": 1.0, "s": 1.0},
    "norm": {"loc": 0.0, "scale": 1.0},
    "uniform": {"loc": 0.0, "scale": 1.0},
}

DREAM_PRIOR_PRESET_STATUS_LABELS: dict[str, str] = {
    "custom": "Custom / Manual",
    "very_strict": "Very Strict",
    "strict": "Strict",
    "proportional": "Proportional",
    "rounded_guides": "Rounded Guides",
    "legacy_md_weights": "Legacy MD Weights",
    "lenient": "Lenient",
    "very_lenient": "Very Lenient",
    "strict_small_lenient_large": "Strict Small / Lenient Large",
    "lenient_small_strict_large": "Lenient Small / Strict Large",
}
_DREAM_PRIOR_PRESET_STATUS_ORDER = tuple(DREAM_PRIOR_PRESET_STATUS_LABELS)
GUIDE_INTERVAL_LOWER_Q = float(stats.norm.cdf(-3.0))
GUIDE_INTERVAL_UPPER_Q = float(stats.norm.cdf(3.0))
GUIDE_CLIP_RELATIVE_TOLERANCE = 1e-9


@dataclass(slots=True)
class DreamParameterEntry:
    structure: str
    motif: str
    param_type: str
    param: str
    value: float
    vary: bool
    distribution: str
    dist_params: dict[str, float]
    smart_preset_status: str = "custom"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dist_params"] = normalize_distribution_params(
            self.distribution,
            self.dist_params,
            self.value,
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DreamParameterEntry":
        distribution = str(payload.get("distribution", "lognorm"))
        value = float(payload.get("value", 0.0))
        return cls(
            structure=str(payload.get("structure", "")),
            motif=str(payload.get("motif", "")),
            param_type=str(payload.get("param_type", "SAXS")),
            param=str(payload.get("param", "")),
            value=value,
            vary=bool(payload.get("vary", True)),
            distribution=distribution,
            dist_params=normalize_distribution_params(
                distribution,
                {
                    str(key): float(dist_value)
                    for key, dist_value in dict(
                        payload.get("dist_params", {})
                    ).items()
                },
                value,
            ),
            smart_preset_status=str(
                payload.get("smart_preset_status", "custom")
            ).strip()
            or "custom",
        )


def build_default_parameter_map(
    prefit_payload: dict[str, object],
) -> list[DreamParameterEntry]:
    cv_default = 1 / math.e
    eps = 1e-6
    entries: list[DreamParameterEntry] = []

    for weight in prefit_payload.get("weights", []):
        value = float(weight.get("value", 0.0))
        distribution, dist_params = _default_distribution_for_weight(
            value=value,
            minimum=weight.get("min"),
            maximum=weight.get("max"),
            cv_default=cv_default,
            eps=eps,
        )
        entries.append(
            DreamParameterEntry(
                structure=str(weight.get("structure", "")),
                motif=str(weight.get("motif", "")),
                param_type="Both",
                param=str(weight.get("name", "")),
                value=value,
                vary=True,
                distribution=distribution,
                dist_params=dist_params,
            )
        )

    fit_parameter_meta = prefit_payload.get("fit_parameter_meta", {})
    for name, value in dict(prefit_payload.get("fit_parameters", {})).items():
        meta = dict(fit_parameter_meta.get(name, {}))
        float_value = float(value)
        distribution = _default_distribution_for_fit_parameter(str(name))
        entries.append(
            DreamParameterEntry(
                structure="",
                motif="",
                param_type="SAXS",
                param=str(name),
                value=float_value,
                vary=bool(meta.get("vary", True)),
                distribution=distribution,
                dist_params=_default_distribution_params_for_fit_parameter(
                    float_value,
                    distribution=distribution,
                    minimum=meta.get("min"),
                    maximum=meta.get("max"),
                    cv_default=cv_default,
                    eps=eps,
                ),
            )
        )
    return entries


def _is_prefit_weight_entry(entry: object) -> bool:
    name = str(getattr(entry, "name", "")).strip()
    return (
        str(getattr(entry, "category", "")).strip() == "weight"
        or re.fullmatch(r"w\d+", name) is not None
    )


def _component_scoped_weight_name(parameter_name: str) -> str | None:
    match = re.search(r"(?:^|_)w\d+$", str(parameter_name or "").strip())
    if match is None:
        return None
    return match.group(0).lstrip("_")


def build_default_parameter_map_from_prefit_entries(
    entries,
) -> list[DreamParameterEntry]:
    inactive_weight_names = {
        str(getattr(entry, "name", "")).strip()
        for entry in entries
        if _is_prefit_weight_entry(entry)
        and not bool(getattr(entry, "active", True))
    }
    payload: dict[str, object] = {
        "weights": [],
        "fit_parameters": {},
        "fit_parameter_meta": {},
    }
    for entry in entries:
        name = str(getattr(entry, "name", "")).strip()
        if not name:
            continue
        meta = {
            "vary": bool(getattr(entry, "vary", True)),
            "min": float(getattr(entry, "minimum", 0.0)),
            "max": float(getattr(entry, "maximum", 0.0)),
        }
        if _is_prefit_weight_entry(entry):
            if name in inactive_weight_names:
                continue
            payload["weights"].append(
                {
                    "structure": str(getattr(entry, "structure", "")),
                    "motif": str(getattr(entry, "motif", "")),
                    "name": name,
                    "value": float(getattr(entry, "value", 0.0)),
                    **meta,
                }
            )
            continue
        if _component_scoped_weight_name(name) in inactive_weight_names:
            continue
        payload["fit_parameters"][name] = float(getattr(entry, "value", 0.0))
        payload["fit_parameter_meta"][name] = meta
    return build_default_parameter_map(payload)


def recentered_parameter_entry(
    entry: DreamParameterEntry,
    value: float,
) -> DreamParameterEntry:
    updated = DreamParameterEntry.from_dict(entry.to_dict())
    target_value = float(value)
    updated.value = target_value
    updated.dist_params = recentered_distribution_params(
        updated.distribution,
        updated.dist_params,
        target_value,
    )
    return updated


def normalize_prior_preset_status(value: object) -> str:
    text = str(value or "").strip()
    return text or "custom"


def prior_preset_status_label(value: object) -> str:
    normalized = normalize_prior_preset_status(value)
    return DREAM_PRIOR_PRESET_STATUS_LABELS.get(
        normalized,
        normalized.replace("_", " ").title(),
    )


def prior_preset_status_counts(
    entries,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in entries:
        if isinstance(entry, dict):
            status = entry.get("smart_preset_status", "custom")
        else:
            status = getattr(entry, "smart_preset_status", "custom")
        counts[normalize_prior_preset_status(status)] += 1
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                _prior_preset_status_sort_index(item[0]),
                prior_preset_status_label(item[0]).lower(),
            ),
        )
    )


def prior_preset_status_summary(entries) -> list[dict[str, object]]:
    return [
        {
            "status": status,
            "label": prior_preset_status_label(status),
            "parameter_count": count,
        }
        for status, count in prior_preset_status_counts(entries).items()
    ]


def format_prior_preset_summary(entries) -> str:
    summary = prior_preset_status_summary(entries)
    if not summary:
        return "Unavailable"
    parts = []
    for item in summary:
        count = int(item["parameter_count"])
        unit = "parameter" if count == 1 else "parameters"
        parts.append(f"{item['label']}: {count} {unit}")
    return "; ".join(parts)


def distribution_guide_bounds(
    entry: DreamParameterEntry | dict[str, object],
) -> tuple[float | None, float | None, str]:
    distribution_name = _entry_distribution_name(entry)
    dist_params = _entry_distribution_params(entry)
    try:
        distribution = getattr(stats, distribution_name)
    except AttributeError:
        return None, None, "Unavailable"
    try:
        support_low, support_high = distribution.support(**dist_params)
        support_low = float(support_low)
        support_high = float(support_high)
    except Exception:
        support_low = float("nan")
        support_high = float("nan")

    if np.isfinite(support_low) and np.isfinite(support_high):
        if support_low <= support_high:
            return support_low, support_high, "Exact support"

    try:
        guide_low = float(
            distribution.ppf(GUIDE_INTERVAL_LOWER_Q, **dist_params)
        )
        guide_high = float(
            distribution.ppf(GUIDE_INTERVAL_UPPER_Q, **dist_params)
        )
        if np.isfinite(support_low):
            guide_low = (
                max(guide_low, support_low)
                if np.isfinite(guide_low)
                else support_low
            )
        if np.isfinite(support_high):
            guide_high = (
                min(guide_high, support_high)
                if np.isfinite(guide_high)
                else support_high
            )
        if np.isfinite(guide_low) and np.isfinite(guide_high):
            if guide_low <= guide_high:
                guide_kind = "Central 99.73% interval (3sigma equivalent)"
                if _lognormal_weight_uses_zero_lower_bound(
                    entry,
                    support_low,
                ):
                    guide_low = 0.0
                    guide_kind = (
                        "Zero lower support with central 99.73% upper "
                        "interval"
                    )
                return guide_low, guide_high, guide_kind
    except Exception:
        pass

    return None, None, "Unavailable"


def format_distribution_guide_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(float_value):
        return "n/a"
    return f"{float_value:.6g}"


def guide_clip_status(
    value: object,
    guide_low: float | None,
    guide_high: float | None,
) -> str:
    if guide_low is None or guide_high is None:
        return ""
    try:
        value_float = float(value)
        low = float(guide_low)
        high = float(guide_high)
    except (TypeError, ValueError):
        return ""
    if not (
        np.isfinite(value_float) and np.isfinite(low) and np.isfinite(high)
    ):
        return ""
    tolerance = GUIDE_CLIP_RELATIVE_TOLERANCE * max(
        abs(value_float),
        abs(low),
        abs(high),
        1.0,
    )
    if value_float <= low + tolerance:
        return "low"
    if value_float >= high - tolerance:
        return "high"
    return ""


def guide_clip_status_label(status: object) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "low":
        return "Guide Low"
    if normalized == "high":
        return "Guide High"
    return ""


def _entry_distribution_name(
    entry: DreamParameterEntry | dict[str, object],
) -> str:
    if isinstance(entry, dict):
        return str(entry.get("distribution", "lognorm")).strip() or "lognorm"
    return str(getattr(entry, "distribution", "lognorm")).strip() or "lognorm"


def _entry_distribution_params(
    entry: DreamParameterEntry | dict[str, object],
) -> dict[str, float]:
    raw_params = (
        entry.get("dist_params", {})
        if isinstance(entry, dict)
        else getattr(entry, "dist_params", {})
    )
    return {
        str(key): float(value) for key, value in dict(raw_params or {}).items()
    }


def _entry_param_name(entry: DreamParameterEntry | dict[str, object]) -> str:
    if isinstance(entry, dict):
        return str(entry.get("param", "")).strip()
    return str(getattr(entry, "param", "")).strip()


def _lognormal_weight_uses_zero_lower_bound(
    entry: DreamParameterEntry | dict[str, object],
    support_low: float,
) -> bool:
    param_name = _entry_param_name(entry)
    return bool(
        _entry_distribution_name(entry) == "lognorm"
        and re.fullmatch(r"w\d+", param_name)
        and np.isfinite(support_low)
        and float(support_low) <= 0.0
    )


def _prior_preset_status_sort_index(status: str) -> int:
    try:
        return _DREAM_PRIOR_PRESET_STATUS_ORDER.index(status)
    except ValueError:
        return len(_DREAM_PRIOR_PRESET_STATUS_ORDER)


def load_parameter_map(path: str | Path) -> list[DreamParameterEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        DreamParameterEntry.from_dict(entry)
        for entry in payload.get("entries", [])
    ]


def save_parameter_map(
    path: str | Path,
    entries: list[DreamParameterEntry],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [entry.to_dict() for entry in entries],
    }
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def normalize_distribution_params(
    distribution: str,
    dist_params: dict[str, float] | None,
    value: float,
) -> dict[str, float]:
    normalized_distribution = str(distribution).strip() or "lognorm"
    raw_params = {
        str(key): float(param_value)
        for key, param_value in dict(dist_params or {}).items()
    }
    params = _default_distribution_params(value, 1 / math.e, 1e-6).get(
        normalized_distribution,
        {},
    )
    if not params:
        return raw_params
    for key in list(params):
        if key in raw_params:
            params[key] = float(raw_params[key])
    return params


def recentered_distribution_params(
    distribution: str,
    dist_params: dict[str, float] | None,
    value: float,
) -> dict[str, float]:
    normalized_distribution = str(distribution).strip() or "lognorm"
    target_value = float(value)
    epsilon = 1e-9
    params = normalize_distribution_params(
        normalized_distribution,
        dist_params,
        target_value,
    )
    if normalized_distribution == "norm":
        params["loc"] = target_value
        params["scale"] = max(
            float(params.get("scale", epsilon)),
            epsilon,
        )
        return params
    if normalized_distribution == "uniform":
        width = max(float(params.get("scale", epsilon)), epsilon)
        params["scale"] = width
        params["loc"] = target_value - width / 2.0
        return params
    if normalized_distribution == "lognorm":
        scale_value = max(
            float(params.get("scale", epsilon)),
            epsilon,
        )
        params["scale"] = scale_value
        params["s"] = max(
            float(params.get("s", epsilon)),
            epsilon,
        )
        params["loc"] = target_value - scale_value
        return params
    return params


def _default_distribution_params(
    value: float,
    cv_default: float,
    eps: float,
) -> dict[str, dict[str, float]]:
    scale_val = value if value > 0 else eps
    lognorm_s = math.sqrt(math.log(1 + cv_default**2))
    norm_scale = value / math.e if value > 0 else eps
    if value > 0:
        half_range = value * cv_default
        uniform_loc = max(value - half_range, 0.0)
        uniform_scale = half_range * 2
    else:
        uniform_loc = 0.0
        uniform_scale = eps
    return {
        "lognorm": {
            "loc": 0.0,
            "scale": scale_val,
            "s": lognorm_s,
        },
        "norm": {
            "loc": value,
            "scale": norm_scale,
        },
        "uniform": {
            "loc": uniform_loc,
            "scale": uniform_scale,
        },
    }


def _default_distribution_for_weight(
    *,
    value: float,
    minimum: object,
    maximum: object,
    cv_default: float,
    eps: float,
) -> tuple[str, dict[str, float]]:
    return (
        "lognorm",
        _default_distribution_params(
            value,
            cv_default,
            eps,
        )["lognorm"],
    )


def _default_distribution_for_fit_parameter(name: str) -> str:
    normalized_name = str(name).strip().lower()
    if normalized_name in {"scale", "offset"}:
        return "uniform"
    if _is_radius_parameter_name(normalized_name):
        return "lognorm"
    return "norm"


def _default_distribution_params_for_fit_parameter(
    value: float,
    *,
    distribution: str,
    minimum: object,
    maximum: object,
    cv_default: float,
    eps: float,
) -> dict[str, float]:
    normalized_distribution = str(distribution).strip()
    if normalized_distribution == "uniform":
        bounds = _finite_bounds(minimum, maximum)
        if bounds is not None:
            lower, upper = bounds
            return {"loc": lower, "scale": upper - lower}
        half_range = abs(float(value)) * cv_default
        if half_range <= 0.0 or not math.isfinite(half_range):
            half_range = eps
        return {"loc": float(value) - half_range, "scale": half_range * 2.0}
    return _default_distribution_params(
        value,
        cv_default,
        eps,
    )[normalized_distribution]


def _is_radius_parameter_name(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in {"eff_r", "r_eff", "radius", "effective_radius"}:
        return True
    if re.fullmatch(r"[rabc]_eff_w\d+", normalized):
        return True
    return "radius" in normalized or "radii" in normalized


def _finite_bounds(
    minimum: object,
    maximum: object,
) -> tuple[float, float] | None:
    try:
        lower = float(minimum)
        upper = float(maximum)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if upper <= lower:
        return None
    return lower, upper


__all__ = [
    "BASE_DISTRIBUTIONS",
    "DREAM_PRIOR_PRESET_STATUS_LABELS",
    "DreamParameterEntry",
    "build_default_parameter_map",
    "build_default_parameter_map_from_prefit_entries",
    "distribution_guide_bounds",
    "format_prior_preset_summary",
    "format_distribution_guide_value",
    "guide_clip_status",
    "guide_clip_status_label",
    "load_parameter_map",
    "normalize_prior_preset_status",
    "normalize_distribution_params",
    "prior_preset_status_counts",
    "prior_preset_status_label",
    "prior_preset_status_summary",
    "recentered_distribution_params",
    "recentered_parameter_entry",
    "save_parameter_map",
]
