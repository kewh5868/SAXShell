from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ContrastSolventPreset:
    name: str
    formula: str
    density_g_per_ml: float
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "formula": self.formula,
            "density_g_per_ml": float(self.density_g_per_ml),
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, object],
        *,
        builtin: bool = False,
    ) -> "ContrastSolventPreset":
        formula = str(payload.get("formula") or "").strip()
        if not formula:
            raise ValueError("Solvent presets require a formula.")
        density = float(payload.get("density_g_per_ml") or 0.0)
        if density < 0.0:
            raise ValueError("Solvent presets require a non-negative density.")
        if density == 0.0 and formula.lower() not in {"vacuum", "none"}:
            raise ValueError(
                "Zero-density solvent presets are reserved for vacuum."
            )
        return cls(
            name=name,
            formula=formula,
            density_g_per_ml=density,
            builtin=builtin,
        )


def contrast_solvent_presets_path() -> Path:
    configured = os.environ.get("SAXSHELL_CONTRAST_SOLVENTS_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".saxshell" / "contrast_solvent_presets.json"


def default_solvent_presets() -> dict[str, ContrastSolventPreset]:
    return {
        "Water": ContrastSolventPreset(
            name="Water",
            formula="H2O",
            density_g_per_ml=1.0,
            builtin=True,
        ),
        "Vacuum": ContrastSolventPreset(
            name="Vacuum",
            formula="Vacuum",
            density_g_per_ml=0.0,
            builtin=True,
        ),
        "DMF": ContrastSolventPreset(
            name="DMF",
            formula="C3H7NO",
            density_g_per_ml=0.944,
            builtin=True,
        ),
        "DMSO": ContrastSolventPreset(
            name="DMSO",
            formula="C2H6OS",
            density_g_per_ml=1.10,
            builtin=True,
        ),
    }


def load_solvent_presets() -> dict[str, ContrastSolventPreset]:
    presets = default_solvent_presets()
    for name, payload in _load_custom_preset_payloads().items():
        if not isinstance(payload, dict):
            continue
        try:
            presets[name] = ContrastSolventPreset.from_dict(name, payload)
        except (TypeError, ValueError):
            continue
    return presets


def save_custom_solvent_preset(preset: ContrastSolventPreset) -> Path:
    file_path = contrast_solvent_presets_path()
    payload = _load_custom_preset_payloads()
    payload[preset.name] = preset.to_dict()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps({"presets": payload}, indent=2) + "\n")
    return file_path


def delete_custom_solvent_preset(name: str) -> Path | None:
    normalized_name = str(name).strip()
    if not normalized_name:
        return None
    file_path = contrast_solvent_presets_path()
    payload = _load_custom_preset_payloads()
    if normalized_name not in payload:
        return None
    del payload[normalized_name]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps({"presets": payload}, indent=2) + "\n")
    return file_path


def ordered_solvent_preset_names(
    presets: dict[str, ContrastSolventPreset],
) -> list[str]:
    defaults = default_solvent_presets()
    ordered_names = [name for name in defaults if name in presets]
    custom_names = sorted(name for name in presets if name not in defaults)
    override_names = sorted(
        name
        for name, preset in presets.items()
        if name in defaults and not preset.builtin
    )
    if override_names:
        ordered_names = [
            name for name in ordered_names if name not in override_names
        ] + override_names
    return ordered_names + custom_names


def _load_custom_preset_payloads() -> dict[str, object]:
    file_path = contrast_solvent_presets_path()
    if not file_path.is_file():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    presets = payload.get("presets", {})
    return presets if isinstance(presets, dict) else {}


__all__ = [
    "ContrastSolventPreset",
    "contrast_solvent_presets_path",
    "default_solvent_presets",
    "delete_custom_solvent_preset",
    "load_solvent_presets",
    "ordered_solvent_preset_names",
    "save_custom_solvent_preset",
]
