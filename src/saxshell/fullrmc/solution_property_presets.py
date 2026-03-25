from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .solution_properties import SolutionPropertiesSettings

_PRESET_DIR_NAME = "_solution_property_presets"
_DEFAULT_PRESETS_FILENAME = "default_solution_property_presets.json"
_USER_PRESETS_FILENAME = "user_solution_property_presets.json"


@dataclass(frozen=True, slots=True)
class SolutionPropertiesPreset:
    name: str
    settings: SolutionPropertiesSettings
    solute_molecule_count: int = 1
    solute_density_g_per_ml: float | None = None
    solvent_density_g_per_ml: float | None = None
    notes: str = ""
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "settings": self.settings.to_dict(),
            "solute_molecule_count": self.solute_molecule_count,
        }
        if self.solute_density_g_per_ml is not None:
            payload["solute_density_g_per_ml"] = float(
                self.solute_density_g_per_ml
            )
        if self.solvent_density_g_per_ml is not None:
            payload["solvent_density_g_per_ml"] = float(
                self.solvent_density_g_per_ml
            )
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, object],
        *,
        builtin: bool = False,
    ) -> "SolutionPropertiesPreset":
        settings_payload = payload.get("settings")
        if isinstance(settings_payload, dict):
            settings = SolutionPropertiesSettings.from_dict(settings_payload)
        else:
            settings = SolutionPropertiesSettings.from_dict(payload)
        try:
            solute_molecule_count = max(
                1,
                int(payload.get("solute_molecule_count", 1)),
            )
        except (TypeError, ValueError):
            solute_molecule_count = 1
        return cls(
            name=name,
            settings=settings,
            solute_molecule_count=solute_molecule_count,
            solute_density_g_per_ml=_optional_positive_float(
                payload.get("solute_density_g_per_ml")
            ),
            solvent_density_g_per_ml=_optional_positive_float(
                payload.get("solvent_density_g_per_ml")
            ),
            notes=str(payload.get("notes", "") or "").strip(),
            builtin=builtin,
        )


def _optional_positive_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def solution_property_presets_dir() -> Path:
    return Path(__file__).resolve().parent / _PRESET_DIR_NAME


def default_solution_property_presets_path() -> Path:
    return solution_property_presets_dir() / _DEFAULT_PRESETS_FILENAME


def solution_property_presets_path() -> Path:
    configured = os.environ.get("SAXSHELL_SOLUTION_PROPERTY_PRESETS_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return solution_property_presets_dir() / _USER_PRESETS_FILENAME


def default_solution_property_presets() -> dict[str, SolutionPropertiesPreset]:
    payloads = _load_preset_payloads(default_solution_property_presets_path())
    return {
        name: SolutionPropertiesPreset.from_dict(
            name,
            payload,
            builtin=True,
        )
        for name, payload in payloads.items()
        if isinstance(payload, dict)
    }


def load_solution_property_presets() -> dict[str, SolutionPropertiesPreset]:
    presets = default_solution_property_presets()
    for name, payload in _load_custom_preset_payloads().items():
        if not isinstance(payload, dict):
            continue
        try:
            presets[name] = SolutionPropertiesPreset.from_dict(name, payload)
        except (KeyError, TypeError, ValueError):
            continue
    return presets


def save_custom_solution_property_preset(
    preset: SolutionPropertiesPreset,
) -> Path:
    file_path = solution_property_presets_path()
    payload = _load_custom_preset_payloads()
    payload[preset.name] = preset.to_dict()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps({"presets": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def ordered_solution_property_preset_names(
    presets: dict[str, SolutionPropertiesPreset],
) -> list[str]:
    defaults = default_solution_property_presets()
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


def _load_preset_payloads(file_path: Path) -> dict[str, object]:
    if not file_path.is_file():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    presets = payload.get("presets", {})
    return presets if isinstance(presets, dict) else {}


def _load_custom_preset_payloads() -> dict[str, object]:
    return _load_preset_payloads(solution_property_presets_path())


__all__ = [
    "SolutionPropertiesPreset",
    "default_solution_property_presets",
    "default_solution_property_presets_path",
    "load_solution_property_presets",
    "ordered_solution_property_preset_names",
    "save_custom_solution_property_preset",
    "solution_property_presets_dir",
    "solution_property_presets_path",
]
