from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from saxshell.structure import (
    AtomTypeDefinitions,
    normalize_atom_type_definitions,
)

from .clusternetwork import (
    DEFAULT_SAVE_STATE_FREQUENCY,
    SEARCH_MODE_KDTREE,
    PairCutoffDefinitions,
    normalize_pair_cutoffs,
    normalize_save_state_frequency,
    normalize_search_mode,
)

_PRESET_DIR_NAME = "_cluster_extraction_presets"
_DEFAULT_PRESETS_FILENAME = "default_cluster_extraction_presets.json"
_USER_PRESETS_FILENAME = "user_cluster_extraction_presets.json"
DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME = "Perovskite (DMF) - Partial Solvent"


@dataclass(frozen=True, slots=True)
class ClusterExtractionPreset:
    name: str
    atom_type_definitions: AtomTypeDefinitions
    pair_cutoff_definitions: PairCutoffDefinitions
    use_pbc: bool = False
    search_mode: str = SEARCH_MODE_KDTREE
    save_state_frequency: int = DEFAULT_SAVE_STATE_FREQUENCY
    default_cutoff: float | None = None
    shell_growth_levels: tuple[int, ...] = ()
    shared_shells: bool = False
    smart_solvation_shells: bool = True
    include_shell_atoms_in_stoichiometry: bool = False
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        options: dict[str, object] = {
            "use_pbc": bool(self.use_pbc),
            "search_mode": str(self.search_mode),
            "save_state_frequency": int(self.save_state_frequency),
            "shell_growth_levels": [
                int(level) for level in self.shell_growth_levels
            ],
            "shared_shells": bool(self.shared_shells),
            "smart_solvation_shells": bool(self.smart_solvation_shells),
            "include_shell_atoms_in_stoichiometry": bool(
                self.include_shell_atoms_in_stoichiometry
            ),
        }
        if self.default_cutoff is not None:
            options["default_cutoff"] = float(self.default_cutoff)
        return {
            "atom_type_definitions": _serialize_atom_type_definitions(
                self.atom_type_definitions
            ),
            "pair_cutoff_definitions": _serialize_pair_cutoff_definitions(
                self.pair_cutoff_definitions
            ),
            "options": options,
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, object],
        *,
        builtin: bool = False,
    ) -> "ClusterExtractionPreset":
        options_payload = payload.get("options")
        if not isinstance(options_payload, dict):
            options_payload = payload
        return cls(
            name=name,
            atom_type_definitions=_coerce_atom_type_definitions(
                payload.get("atom_type_definitions")
            ),
            pair_cutoff_definitions=_coerce_pair_cutoff_definitions(
                payload.get("pair_cutoff_definitions")
            ),
            use_pbc=bool(options_payload.get("use_pbc", False)),
            search_mode=normalize_search_mode(
                str(
                    options_payload.get("search_mode", SEARCH_MODE_KDTREE)
                    or SEARCH_MODE_KDTREE
                )
            ),
            save_state_frequency=normalize_save_state_frequency(
                _coerce_positive_int(
                    options_payload.get("save_state_frequency"),
                    default=DEFAULT_SAVE_STATE_FREQUENCY,
                )
            ),
            default_cutoff=_optional_positive_float(
                options_payload.get("default_cutoff")
            ),
            shell_growth_levels=_normalize_shell_growth_levels(
                options_payload.get("shell_growth_levels")
            ),
            shared_shells=bool(options_payload.get("shared_shells", False)),
            smart_solvation_shells=bool(
                options_payload.get("smart_solvation_shells", True)
            ),
            include_shell_atoms_in_stoichiometry=bool(
                options_payload.get(
                    "include_shell_atoms_in_stoichiometry",
                    False,
                )
            ),
            builtin=builtin,
        )


def cluster_extraction_presets_dir() -> Path:
    return Path(__file__).resolve().parent / _PRESET_DIR_NAME


def default_cluster_extraction_presets_path() -> Path:
    return cluster_extraction_presets_dir() / _DEFAULT_PRESETS_FILENAME


def cluster_extraction_presets_path() -> Path:
    configured = os.environ.get("SAXSHELL_CLUSTER_EXTRACTION_PRESETS_PATH", "")
    if configured.strip():
        return Path(configured).expanduser()
    return cluster_extraction_presets_dir() / _USER_PRESETS_FILENAME


def default_cluster_extraction_presets() -> dict[str, ClusterExtractionPreset]:
    payloads = _load_preset_payloads(default_cluster_extraction_presets_path())
    return {
        name: ClusterExtractionPreset.from_dict(name, payload, builtin=True)
        for name, payload in payloads.items()
        if isinstance(payload, dict)
    }


def load_cluster_extraction_presets() -> dict[str, ClusterExtractionPreset]:
    presets = default_cluster_extraction_presets()
    for name, payload in _load_custom_preset_payloads().items():
        if not isinstance(payload, dict):
            continue
        try:
            presets[name] = ClusterExtractionPreset.from_dict(name, payload)
        except (TypeError, ValueError):
            continue
    return presets


def save_custom_cluster_extraction_preset(
    preset: ClusterExtractionPreset,
) -> Path:
    file_path = cluster_extraction_presets_path()
    payload = _load_custom_preset_payloads()
    payload[preset.name] = preset.to_dict()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps({"presets": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def ordered_cluster_extraction_preset_names(
    presets: dict[str, ClusterExtractionPreset],
) -> list[str]:
    defaults = default_cluster_extraction_presets()
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


def _serialize_atom_type_definitions(
    definitions: AtomTypeDefinitions,
) -> dict[str, list[dict[str, str | None]]]:
    normalized = normalize_atom_type_definitions(definitions)
    return {
        atom_type: [
            {
                "element": element,
                "residue": residue,
            }
            for element, residue in criteria
        ]
        for atom_type, criteria in normalized.items()
    }


def _serialize_pair_cutoff_definitions(
    definitions: PairCutoffDefinitions,
) -> list[dict[str, object]]:
    normalized = normalize_pair_cutoffs(definitions)
    payload: list[dict[str, object]] = []
    for atom1, atom2 in sorted(normalized):
        shell_cutoffs = normalized[(atom1, atom2)]
        payload.append(
            {
                "atom1": atom1,
                "atom2": atom2,
                "shell_cutoffs": {
                    str(level): float(cutoff)
                    for level, cutoff in sorted(shell_cutoffs.items())
                },
            }
        )
    return payload


def _coerce_atom_type_definitions(value: object) -> AtomTypeDefinitions:
    if not isinstance(value, dict):
        return {}
    definitions: AtomTypeDefinitions = {}
    for atom_type, entries in value.items():
        atom_type_name = str(atom_type).strip()
        if not atom_type_name or not isinstance(entries, list):
            continue
        criteria: list[tuple[str, str | None]] = []
        for entry in entries:
            element, residue = _coerce_atom_type_entry(entry)
            if element:
                criteria.append((element, residue))
        if criteria:
            definitions[atom_type_name] = criteria
    return normalize_atom_type_definitions(definitions)


def _coerce_atom_type_entry(entry: object) -> tuple[str, str | None]:
    if isinstance(entry, dict):
        element_value = entry.get("element")
        residue_value = entry.get("residue")
    elif isinstance(entry, (list, tuple)):
        element_value = entry[0] if len(entry) >= 1 else None
        residue_value = entry[1] if len(entry) >= 2 else None
    else:
        element_value = entry
        residue_value = None

    element = str(element_value or "").strip().title()
    residue_text = str(residue_value or "").strip()
    residue = residue_text or None
    return element, residue


def _coerce_pair_cutoff_definitions(value: object) -> PairCutoffDefinitions:
    if not isinstance(value, list):
        return {}
    definitions: PairCutoffDefinitions = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        atom1 = str(entry.get("atom1", "") or "").strip().title()
        atom2 = str(entry.get("atom2", "") or "").strip().title()
        if not atom1 or not atom2:
            continue
        shell_payload = entry.get("shell_cutoffs", {})
        if not isinstance(shell_payload, dict):
            continue
        shell_cutoffs: dict[int, float] = {}
        for level, cutoff in shell_payload.items():
            try:
                parsed_level = int(level)
                parsed_cutoff = float(cutoff)
            except (TypeError, ValueError):
                continue
            if parsed_cutoff > 0.0:
                shell_cutoffs[parsed_level] = parsed_cutoff
        if shell_cutoffs:
            definitions[(atom1, atom2)] = shell_cutoffs
    return normalize_pair_cutoffs(definitions)


def _normalize_shell_growth_levels(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    normalized: set[int] = set()
    for level in value:
        try:
            parsed = int(level)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized.add(parsed)
    return tuple(sorted(normalized))


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


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


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
    return _load_preset_payloads(cluster_extraction_presets_path())


__all__ = [
    "DEFAULT_CLUSTER_EXTRACTION_PRESET_NAME",
    "ClusterExtractionPreset",
    "cluster_extraction_presets_dir",
    "cluster_extraction_presets_path",
    "default_cluster_extraction_presets",
    "default_cluster_extraction_presets_path",
    "load_cluster_extraction_presets",
    "ordered_cluster_extraction_preset_names",
    "save_custom_cluster_extraction_preset",
]
