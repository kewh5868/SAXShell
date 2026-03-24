from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from saxshell.fullrmc._deprecated.constraintpdb import ConstraintPDB
from saxshell.fullrmc.packmol_setup import PackmolSetupMetadata

if False:  # pragma: no cover
    from .project_loader import RMCDreamProjectSource


@dataclass(slots=True)
class ConstraintGenerationSettings:
    bond_length_tolerance_angstrom: float = 0.05
    bond_angle_tolerance_degrees: float = 5.0
    parse_hetatm: bool = True
    merged_output_filename: str = "merged_fullrmc_constraints.py"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "ConstraintGenerationSettings":
        source = dict(payload or {})
        return cls(
            bond_length_tolerance_angstrom=max(
                0.0,
                float(source.get("bond_length_tolerance_angstrom", 0.05)),
            ),
            bond_angle_tolerance_degrees=max(
                0.0,
                float(source.get("bond_angle_tolerance_degrees", 5.0)),
            ),
            parse_hetatm=bool(source.get("parse_hetatm", True)),
            merged_output_filename=_safe_filename(
                str(
                    source.get(
                        "merged_output_filename",
                        "merged_fullrmc_constraints.py",
                    )
                )
            ),
        )


@dataclass(slots=True)
class ConstraintGenerationEntry:
    structure: str
    motif: str
    param: str
    residue_name: str
    packmol_pdb: str
    per_structure_constraints_path: str
    bond_length_count: int
    bond_angle_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ConstraintGenerationEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            residue_name=str(payload.get("residue_name", "")).strip(),
            packmol_pdb=str(payload.get("packmol_pdb", "")).strip(),
            per_structure_constraints_path=str(
                payload.get("per_structure_constraints_path", "")
            ).strip(),
            bond_length_count=int(payload.get("bond_length_count", 0)),
            bond_angle_count=int(payload.get("bond_angle_count", 0)),
        )


@dataclass(slots=True)
class ConstraintGenerationMetadata:
    settings: ConstraintGenerationSettings
    updated_at: str
    merged_constraints_path: str
    entries: list[ConstraintGenerationEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.settings.to_dict(),
            "updated_at": self.updated_at,
            "merged_constraints_path": self.merged_constraints_path,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "ConstraintGenerationMetadata | None":
        if not payload:
            return None
        return cls(
            settings=ConstraintGenerationSettings.from_dict(
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else None
            ),
            updated_at=str(payload.get("updated_at", "")).strip(),
            merged_constraints_path=str(
                payload.get("merged_constraints_path", "")
            ).strip(),
            entries=[
                ConstraintGenerationEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
        )

    def summary_text(self) -> str:
        lines = [
            (
                "Bond-length tolerance: "
                f"{self.settings.bond_length_tolerance_angstrom:.3f} A"
            ),
            (
                "Bond-angle tolerance: "
                f"{self.settings.bond_angle_tolerance_degrees:.3f} deg"
            ),
            f"Saved at: {self.updated_at}",
            f"Per-structure files: {len(self.entries)}",
            f"Merged constraints: {Path(self.merged_constraints_path).name}",
        ]
        if self.entries:
            first = self.entries[0]
            lines.extend(
                [
                    "",
                    "Example constraint file:",
                    f"  {first.structure}/{first.motif}",
                    f"  residue: {first.residue_name}",
                    f"  bond lengths: {first.bond_length_count}",
                    f"  bond angles: {first.bond_angle_count}",
                    f"  file: {Path(first.per_structure_constraints_path).name}",
                ]
            )
        return "\n".join(lines)


def build_constraint_generation(
    project_source: "RMCDreamProjectSource",
    settings: ConstraintGenerationSettings | None = None,
    *,
    packmol_setup_metadata: PackmolSetupMetadata | None = None,
) -> ConstraintGenerationMetadata:
    active_settings = settings or ConstraintGenerationSettings()
    active_setup = packmol_setup_metadata or project_source.packmol_setup
    if active_setup is None or not active_setup.entries:
        raise ValueError(
            "Build Packmol setup inputs before generating constraints."
        )

    merged_bond_lengths: dict[str, list[list[object]]] = {}
    merged_bond_angles: dict[str, list[list[object]]] = {}
    entries: list[ConstraintGenerationEntry] = []

    for setup_entry in active_setup.entries:
        pdb_path = Path(setup_entry.packmol_pdb).expanduser().resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(
                f"Packmol input PDB was not found: {pdb_path}"
            )
        residue_name = setup_entry.residue_name.strip().upper()
        constraint_pdb = ConstraintPDB(
            pdb_file=str(pdb_path),
            parse_hetatm=active_settings.parse_hetatm,
            residue_filter={residue_name},
        )
        bond_lengths, bond_angles, _debug = (
            constraint_pdb.generate_all_constraints(
                length_variance=active_settings.bond_length_tolerance_angstrom,
                angle_variance=active_settings.bond_angle_tolerance_degrees,
            )
        )
        safe_stem = (
            f"constraints_{_safe_name(setup_entry.structure)}_"
            f"{_safe_name(setup_entry.motif)}_{residue_name}.txt"
        )
        per_structure_path = (
            project_source.rmcsetup_paths.constraints_dir / safe_stem
        )
        constraint_pdb.write_constraints_txt(
            per_structure_path,
            bond_lengths,
            bond_angles,
        )
        bond_length_rows = list(bond_lengths.get(residue_name, []))
        bond_angle_rows = list(bond_angles.get(residue_name, []))
        merged_bond_lengths.setdefault(residue_name, []).extend(
            bond_length_rows
        )
        merged_bond_angles.setdefault(residue_name, []).extend(bond_angle_rows)
        entries.append(
            ConstraintGenerationEntry(
                structure=setup_entry.structure,
                motif=setup_entry.motif,
                param=setup_entry.param,
                residue_name=residue_name,
                packmol_pdb=setup_entry.packmol_pdb,
                per_structure_constraints_path=str(per_structure_path),
                bond_length_count=len(bond_length_rows),
                bond_angle_count=len(bond_angle_rows),
            )
        )

    merged_path = (
        project_source.rmcsetup_paths.constraints_dir
        / active_settings.merged_output_filename
    )
    _write_merged_constraints_file(
        merged_path,
        merged_bond_angles=merged_bond_angles,
        merged_bond_lengths=merged_bond_lengths,
    )
    metadata = ConstraintGenerationMetadata(
        settings=active_settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        merged_constraints_path=str(merged_path),
        entries=entries,
    )
    save_constraint_generation_metadata(
        project_source.rmcsetup_paths.constraint_generation_path,
        metadata,
    )
    return metadata


def save_constraint_generation_metadata(
    output_path: str | Path,
    metadata: ConstraintGenerationMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_constraint_generation_metadata(
    metadata_path: str | Path,
) -> ConstraintGenerationMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return ConstraintGenerationMetadata.from_dict(payload)


def _write_merged_constraints_file(
    output_path: Path,
    *,
    merged_bond_angles: dict[str, list[list[object]]],
    merged_bond_lengths: dict[str, list[list[object]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by SAXShell rmcsetup",
        "",
        "BOND_ANGLE_CONSTRAINTS = {",
    ]
    for residue_name in sorted(merged_bond_angles):
        lines.append(f"    {residue_name!r}: [")
        for row in merged_bond_angles[residue_name]:
            lines.append(f"        {_format_row(row)},")
        lines.append("    ],")
    lines.extend(
        [
            "}",
            "",
            "BOND_LENGTH_CONSTRAINTS = {",
        ]
    )
    for residue_name in sorted(merged_bond_lengths):
        lines.append(f"    {residue_name!r}: [")
        for row in merged_bond_lengths[residue_name]:
            lines.append(f"        {_format_row(row)},")
        lines.append("    ],")
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_row(row: list[object]) -> str:
    return "[" + ", ".join(repr(value) for value in row) + "]"


def _safe_name(text: str) -> str:
    collapsed = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip())
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return collapsed or "item"


def _safe_filename(text: str) -> str:
    name = Path(text.strip() or "item.py").name
    return name or "item.py"


__all__ = [
    "ConstraintGenerationEntry",
    "ConstraintGenerationMetadata",
    "ConstraintGenerationSettings",
    "build_constraint_generation",
    "load_constraint_generation_metadata",
    "save_constraint_generation_metadata",
]
