from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .bondanalyzer import AngleTripletDefinition, BondPairDefinition


@dataclass(frozen=True, slots=True)
class BondAnalysisPreset:
    """Serializable preset for bondanalysis UI definitions."""

    name: str
    bond_pairs: tuple[BondPairDefinition, ...]
    angle_triplets: tuple[AngleTripletDefinition, ...]
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "bond_pairs": [
                definition.to_dict() for definition in self.bond_pairs
            ],
            "angle_triplets": [
                definition.to_dict() for definition in self.angle_triplets
            ],
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, object],
        *,
        builtin: bool = False,
    ) -> "BondAnalysisPreset":
        bond_pairs = tuple(
            BondPairDefinition(
                str(entry["atom1"]),
                str(entry["atom2"]),
                float(entry["cutoff_angstrom"]),
            )
            for entry in list(payload.get("bond_pairs", []))
        )
        angle_triplets = tuple(
            AngleTripletDefinition(
                str(entry["vertex"]),
                str(entry["arm1"]),
                str(entry["arm2"]),
                float(entry["cutoff1_angstrom"]),
                float(entry["cutoff2_angstrom"]),
            )
            for entry in list(payload.get("angle_triplets", []))
        )
        return cls(
            name=name,
            bond_pairs=bond_pairs,
            angle_triplets=angle_triplets,
            builtin=builtin,
        )


def bondanalysis_presets_path() -> Path:
    """Return the user presets file path."""
    configured = os.environ.get("SAXSHELL_BONDANALYSIS_PRESETS_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".saxshell" / "bondanalysis_presets.json"


def default_presets() -> dict[str, BondAnalysisPreset]:
    """Return the built-in bondanalysis presets."""
    return {
        "DMSO": BondAnalysisPreset(
            name="DMSO",
            builtin=True,
            bond_pairs=(
                BondPairDefinition("Pb", "I", 4.0),
                BondPairDefinition("Pb", "O", 4.0),
                BondPairDefinition("Pb", "Pb", 6.0),
                BondPairDefinition("Pb", "C", 8.0),
                BondPairDefinition("Pb", "S", 4.8),
                BondPairDefinition("O", "S", 2.2),
                BondPairDefinition("S", "C", 2.2),
            ),
            angle_triplets=(
                AngleTripletDefinition("Pb", "I", "I", 4.0, 4.0),
                AngleTripletDefinition("I", "Pb", "Pb", 4.0, 4.0),
                AngleTripletDefinition("O", "Pb", "S", 4.0, 2.2),
                AngleTripletDefinition("S", "O", "C", 2.2, 2.2),
                AngleTripletDefinition("S", "C", "C", 2.2, 2.2),
            ),
        ),
        "DMF": BondAnalysisPreset(
            name="DMF",
            builtin=True,
            bond_pairs=(
                BondPairDefinition("Pb", "I", 4.0),
                BondPairDefinition("Pb", "O", 4.0),
                BondPairDefinition("Pb", "Pb", 6.0),
                BondPairDefinition("Pb", "C", 8.0),
                BondPairDefinition("Pb", "N", 6.0),
                BondPairDefinition("O", "N", 3.0),
                BondPairDefinition("N", "C", 2.2),
            ),
            angle_triplets=(
                AngleTripletDefinition("Pb", "I", "I", 4.0, 4.0),
                AngleTripletDefinition("I", "Pb", "Pb", 4.0, 4.0),
                AngleTripletDefinition("O", "Pb", "C", 4.0, 2.2),
                AngleTripletDefinition("O", "Pb", "N", 4.0, 3.0),
                AngleTripletDefinition("N", "C", "C", 2.2, 2.2),
            ),
        ),
    }


def load_presets() -> dict[str, BondAnalysisPreset]:
    """Load built-in presets plus any saved custom presets."""
    presets = default_presets()
    for name, payload in _load_custom_preset_payloads().items():
        if not isinstance(payload, dict):
            continue
        try:
            presets[name] = BondAnalysisPreset.from_dict(name, payload)
        except (KeyError, TypeError, ValueError):
            continue
    return presets


def save_custom_preset(preset: BondAnalysisPreset) -> Path:
    """Persist one custom preset to the user presets file."""
    file_path = bondanalysis_presets_path()
    payload = _load_custom_preset_payloads()
    payload[preset.name] = preset.to_dict()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps({"presets": payload}, indent=2) + "\n")
    return file_path


def ordered_preset_names(
    presets: dict[str, BondAnalysisPreset],
) -> list[str]:
    """Return preset names with built-ins first and customs after."""
    defaults = default_presets()
    ordered_names = [name for name in defaults if name in presets]
    custom_names = sorted(
        name for name in presets if name not in defaults
    )
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
    file_path = bondanalysis_presets_path()
    if not file_path.is_file():
        return {}
    try:
        payload = json.loads(file_path.read_text())
    except json.JSONDecodeError:
        return {}
    presets = payload.get("presets", {})
    return presets if isinstance(presets, dict) else {}


__all__ = [
    "BondAnalysisPreset",
    "bondanalysis_presets_path",
    "default_presets",
    "load_presets",
    "ordered_preset_names",
    "save_custom_preset",
]
