from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from saxshell.saxs.solution_scattering_estimator import (
    DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM,
    DEFAULT_BEAM_FOOTPRINT_WIDTH_MM,
    DEFAULT_BEAM_PROFILE,
    DEFAULT_CAPILLARY_GEOMETRY,
    DEFAULT_CAPILLARY_SIZE_MM,
    DEFAULT_INCIDENT_ENERGY_KEV,
    BeamGeometrySettings,
)

_PRESET_DIR_NAME = "_beam_geometry_presets"
_DEFAULT_PRESETS_FILENAME = "default_beam_geometry_presets.json"
_USER_PRESETS_FILENAME = "user_beam_geometry_presets.json"
DEFAULT_BEAM_GEOMETRY_PRESET_NAME = "NSLS-II 28-ID-1 (default)"


@dataclass(frozen=True, slots=True)
class BeamGeometryPreset:
    name: str
    beam: BeamGeometrySettings
    notes: str = ""
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "incident_energy_kev": float(self.beam.incident_energy_kev),
            "capillary_size_mm": float(self.beam.capillary_size_mm),
            "capillary_geometry": str(self.beam.capillary_geometry),
            "beam_profile": str(self.beam.beam_profile),
            "beam_footprint_width_mm": float(
                self.beam.beam_footprint_width_mm
            ),
            "beam_footprint_height_mm": float(
                self.beam.beam_footprint_height_mm
            ),
        }
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
    ) -> "BeamGeometryPreset":
        return cls(
            name=name,
            beam=BeamGeometrySettings(
                incident_energy_kev=_positive_float(
                    payload.get("incident_energy_kev"),
                    default=DEFAULT_INCIDENT_ENERGY_KEV,
                ),
                capillary_size_mm=_positive_float(
                    payload.get("capillary_size_mm"),
                    default=DEFAULT_CAPILLARY_SIZE_MM,
                ),
                capillary_geometry=str(
                    payload.get(
                        "capillary_geometry",
                        DEFAULT_CAPILLARY_GEOMETRY,
                    )
                    or DEFAULT_CAPILLARY_GEOMETRY
                ).strip()
                or DEFAULT_CAPILLARY_GEOMETRY,
                beam_profile=str(
                    payload.get("beam_profile", DEFAULT_BEAM_PROFILE)
                    or DEFAULT_BEAM_PROFILE
                ).strip()
                or DEFAULT_BEAM_PROFILE,
                beam_footprint_width_mm=_positive_float(
                    payload.get("beam_footprint_width_mm"),
                    default=DEFAULT_BEAM_FOOTPRINT_WIDTH_MM,
                ),
                beam_footprint_height_mm=_positive_float(
                    payload.get("beam_footprint_height_mm"),
                    default=DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM,
                ),
            ),
            notes=str(payload.get("notes", "") or "").strip(),
            builtin=builtin,
        )


def _positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed <= 0.0:
        return float(default)
    return parsed


def beam_geometry_presets_dir() -> Path:
    return Path(__file__).resolve().parent / _PRESET_DIR_NAME


def default_beam_geometry_presets_path() -> Path:
    return beam_geometry_presets_dir() / _DEFAULT_PRESETS_FILENAME


def beam_geometry_presets_path() -> Path:
    configured = os.environ.get("SAXSHELL_BEAM_GEOMETRY_PRESETS_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return beam_geometry_presets_dir() / _USER_PRESETS_FILENAME


def default_beam_geometry_presets() -> dict[str, BeamGeometryPreset]:
    payloads = _load_preset_payloads(default_beam_geometry_presets_path())
    return {
        name: BeamGeometryPreset.from_dict(name, payload, builtin=True)
        for name, payload in payloads.items()
        if isinstance(payload, dict)
    }


def load_beam_geometry_presets() -> dict[str, BeamGeometryPreset]:
    presets = default_beam_geometry_presets()
    for name, payload in _load_custom_preset_payloads().items():
        if not isinstance(payload, dict):
            continue
        try:
            presets[name] = BeamGeometryPreset.from_dict(name, payload)
        except (TypeError, ValueError):
            continue
    return presets


def save_custom_beam_geometry_preset(preset: BeamGeometryPreset) -> Path:
    file_path = beam_geometry_presets_path()
    payload = _load_custom_preset_payloads()
    payload[preset.name] = preset.to_dict()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps({"presets": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def delete_custom_beam_geometry_preset(name: str) -> bool:
    file_path = beam_geometry_presets_path()
    payload = _load_custom_preset_payloads()
    if name not in payload:
        return False
    del payload[name]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps({"presets": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def ordered_beam_geometry_preset_names(
    presets: dict[str, BeamGeometryPreset],
) -> list[str]:
    defaults = default_beam_geometry_presets()
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
    return _load_preset_payloads(beam_geometry_presets_path())


__all__ = [
    "DEFAULT_BEAM_GEOMETRY_PRESET_NAME",
    "BeamGeometryPreset",
    "beam_geometry_presets_dir",
    "beam_geometry_presets_path",
    "default_beam_geometry_presets",
    "default_beam_geometry_presets_path",
    "delete_custom_beam_geometry_preset",
    "load_beam_geometry_presets",
    "ordered_beam_geometry_preset_names",
    "save_custom_beam_geometry_preset",
]
