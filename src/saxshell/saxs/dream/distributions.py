from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "lognorm": {"loc": 0.0, "scale": 1.0, "s": 1.0},
    "norm": {"loc": 0.0, "scale": 1.0},
    "uniform": {"loc": 0.0, "scale": 1.0},
}


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
        entries.append(
            DreamParameterEntry(
                structure=str(weight.get("structure", "")),
                motif=str(weight.get("motif", "")),
                param_type="Both",
                param=str(weight.get("name", "")),
                value=value,
                vary=bool(weight.get("vary", True)),
                distribution="lognorm",
                dist_params=_default_distribution_params(
                    value, cv_default, eps
                )["lognorm"],
            )
        )

    fit_parameter_meta = prefit_payload.get("fit_parameter_meta", {})
    for name, value in dict(prefit_payload.get("fit_parameters", {})).items():
        meta = dict(fit_parameter_meta.get(name, {}))
        float_value = float(value)
        entries.append(
            DreamParameterEntry(
                structure="",
                motif="",
                param_type="SAXS",
                param=str(name),
                value=float_value,
                vary=bool(meta.get("vary", True)),
                distribution="norm",
                dist_params=_default_distribution_params(
                    float_value,
                    cv_default,
                    eps,
                )["norm"],
            )
        )
    return entries


def build_default_parameter_map_from_prefit_entries(
    entries,
) -> list[DreamParameterEntry]:
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
        if str(getattr(entry, "category", "")).strip() == "weight":
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
        payload["fit_parameters"][name] = float(getattr(entry, "value", 0.0))
        payload["fit_parameter_meta"][name] = meta
    return build_default_parameter_map(payload)


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


__all__ = [
    "BASE_DISTRIBUTIONS",
    "DreamParameterEntry",
    "build_default_parameter_map",
    "build_default_parameter_map_from_prefit_entries",
    "load_parameter_map",
    "normalize_distribution_params",
    "save_parameter_map",
]
