from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

AtomTypeDefinitions = dict[str, list[tuple[str, str | None]]]


def normalize_atom_type_definitions(
    atom_type_definitions: AtomTypeDefinitions | None,
) -> AtomTypeDefinitions:
    """Normalize atom-type definitions for case-insensitive matching."""
    if atom_type_definitions is None:
        return {}

    normalized: AtomTypeDefinitions = {}
    for atom_type, criteria_list in atom_type_definitions.items():
        normalized[atom_type] = []
        for element, residue_name in criteria_list:
            residue_value = residue_name
            if residue_value is not None:
                residue_value = residue_value.strip() or None
            normalized[atom_type].append((element.title(), residue_value))
    return normalized


@dataclass(slots=True)
class PDBAtom:
    """Structured representation of one PDB atom record."""

    atom_id: int
    atom_name: str
    residue_name: str
    residue_number: int
    coordinates: np.ndarray
    element: str
    cluster_id: str | None = None
    atom_type: str = "unassigned"
    shell_level: int | str | None = None
    shell_history: list[tuple[str, int | str | None]] = field(
        default_factory=list
    )
    coordination_data: dict[tuple[str, float], int] = field(
        default_factory=dict
    )

    def update_cluster_assignment(
        self,
        cluster_id: str,
        shell_level: int | str | None = None,
    ) -> None:
        """Store the current cluster assignment and the previous one."""
        if self.cluster_id is not None:
            self.shell_history.append((self.cluster_id, self.shell_level))
        self.cluster_id = cluster_id
        self.shell_level = shell_level

    def copy(self) -> "PDBAtom":
        """Return a deep-ish copy suitable for edited output
        coordinates."""
        return PDBAtom(
            atom_id=self.atom_id,
            atom_name=self.atom_name,
            residue_name=self.residue_name,
            residue_number=self.residue_number,
            coordinates=self.coordinates.copy(),
            element=self.element,
            cluster_id=self.cluster_id,
            atom_type=self.atom_type,
            shell_level=self.shell_level,
            shell_history=list(self.shell_history),
            coordination_data=dict(self.coordination_data),
        )


class PDBStructure:
    """PDB atom container with atom-type assignment helpers."""

    def __init__(
        self,
        filepath: str | Path | None = None,
        atom_type_definitions: AtomTypeDefinitions | None = None,
        atoms: list[PDBAtom] | None = None,
        *,
        source_name: str | None = None,
    ) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        self.source_name = source_name or (
            self.filepath.stem if self.filepath is not None else None
        )
        self.atom_type_definitions = normalize_atom_type_definitions(
            atom_type_definitions
        )
        self.atoms: list[PDBAtom] = list(atoms) if atoms is not None else []
        self.detected_motifs: dict[str, int] = {}

        if self.filepath is not None and not self.atoms:
            self.read_pdb_file()
        elif self.atom_type_definitions:
            self.assign_atom_types()

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        atom_type_definitions: AtomTypeDefinitions | None = None,
    ) -> "PDBStructure":
        """Build a structure from a PDB file."""
        return cls(
            filepath=filepath,
            atom_type_definitions=atom_type_definitions,
        )

    @classmethod
    def from_lines(
        cls,
        lines: list[str],
        atom_type_definitions: AtomTypeDefinitions | None = None,
        *,
        source_name: str | None = None,
    ) -> "PDBStructure":
        """Build a structure from already-loaded PDB lines."""
        atoms: list[PDBAtom] = []
        structure = cls(
            atom_type_definitions=atom_type_definitions,
            atoms=atoms,
            source_name=source_name,
        )
        structure.read_pdb_lines(lines)
        return structure

    def read_pdb_file(self) -> None:
        """Load atom records from ``self.filepath``."""
        if self.filepath is None:
            raise ValueError("No PDB filepath was provided.")
        try:
            with self.filepath.open("r") as handle:
                self.read_pdb_lines(handle.readlines())
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"PDB file not found at path: {self.filepath}"
            ) from exc

    def read_pdb_lines(self, lines: list[str]) -> None:
        """Parse atom records from a list of PDB lines."""
        self.atoms = []
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom = self.parse_atom_line(line)
                if self.atom_type_definitions:
                    self.assign_atom_type(atom)
                self.atoms.append(atom)

    @staticmethod
    def parse_atom_line(line: str) -> PDBAtom:
        """Parse a PDB atom line into a :class:`PDBAtom`."""
        atom_id = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        residue_number = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        element = line[76:78].strip()
        if not element:
            inferred = re.sub(r"[^A-Za-z]+", "", atom_name)
            if not inferred:
                raise ValueError(
                    f"Could not infer element from line: {line!r}"
                )
            if len(inferred) >= 2 and inferred[1].islower():
                element = inferred[:2]
            else:
                element = inferred[:1]

        return PDBAtom(
            atom_id=atom_id,
            atom_name=atom_name,
            residue_name=residue_name,
            residue_number=residue_number,
            coordinates=np.array([x, y, z], dtype=float),
            element=element.title(),
        )

    def assign_atom_type(self, atom: PDBAtom) -> None:
        """Assign the atom type based on element and residue
        definitions."""
        atom.atom_type = "unassigned"
        atom.shell_level = None

        atom_element = atom.element.title()
        atom_residue = atom.residue_name.upper()
        for atom_type, criteria_list in self.atom_type_definitions.items():
            for element, residue_name in criteria_list:
                residue_matches = (
                    residue_name is None
                    or atom_residue == residue_name.upper()
                )
                if atom_element == element.title() and residue_matches:
                    atom.atom_type = atom_type
                    atom.shell_level = "free" if atom_type == "shell" else None
                    return

    def assign_atom_types(self) -> None:
        """Apply atom-type definitions to all atoms in the structure."""
        for atom in self.atoms:
            self.assign_atom_type(atom)

    def get_atoms_data(
        self,
        include_atom_types: set[str] | None = None,
    ) -> tuple[
        list[int],
        list[np.ndarray],
        dict[int, str],
        dict[int, str],
        dict[int, str],
        dict[int, PDBAtom],
        dict[int, PDBAtom],
    ]:
        """Return atom metadata maps keyed by atom id."""
        if include_atom_types is None:
            include_atom_types = {"node", "linker", "shell"}

        atom_ids_list: list[int] = []
        coordinates_list: list[np.ndarray] = []
        elements: dict[int, str] = {}
        atom_types: dict[int, str] = {}
        residue_names: dict[int, str] = {}
        atom_id_map: dict[int, PDBAtom] = {}
        all_atoms_map: dict[int, PDBAtom] = {}

        for atom in self.atoms:
            atom_id = atom.atom_id
            all_atoms_map[atom_id] = atom
            if atom.atom_type not in include_atom_types:
                continue
            atom_ids_list.append(atom_id)
            coordinates_list.append(atom.coordinates.copy())
            elements[atom_id] = atom.element
            atom_types[atom_id] = atom.atom_type
            residue_names[atom_id] = atom.residue_name
            atom_id_map[atom_id] = atom

        return (
            atom_ids_list,
            coordinates_list,
            elements,
            atom_types,
            residue_names,
            atom_id_map,
            all_atoms_map,
        )

    def residue_atom_ids(
        self,
        residue_number: int,
        residue_name: str | None = None,
    ) -> list[int]:
        """Return atom ids for one residue."""
        return [
            atom.atom_id
            for atom in self.atoms
            if atom.residue_number == residue_number
            and (residue_name is None or atom.residue_name == residue_name)
        ]

    def write_pdb_file(
        self,
        output_path: str | Path,
        atoms: list[PDBAtom] | None = None,
        *,
        header_lines: Sequence[str] | None = None,
    ) -> Path:
        """Write atoms to a PDB file and return the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atoms_to_write = sorted(
            atoms if atoms is not None else self.atoms,
            key=lambda atom: atom.atom_id,
        )

        with output_path.open("w") as handle:
            if header_lines:
                for line in header_lines:
                    handle.write(line.rstrip() + "\n")
            for atom in atoms_to_write:
                handle.write(
                    f"ATOM  {atom.atom_id:5d} {atom.atom_name:<4} "
                    f"{atom.residue_name:>3} X{atom.residue_number:4d}    "
                    f"{atom.coordinates[0]:8.3f}"
                    f"{atom.coordinates[1]:8.3f}"
                    f"{atom.coordinates[2]:8.3f}"
                    f"  1.00  0.00          {atom.element:>2}\n"
                )
            handle.write("END\n")
        return output_path

    def write_xyz_file(
        self,
        output_path: str | Path,
        atoms: list[PDBAtom] | None = None,
    ) -> Path:
        """Write atoms to an XYZ file and return the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atoms_to_write = atoms if atoms is not None else self.atoms

        with output_path.open("w") as handle:
            handle.write(f"{len(atoms_to_write)}\n")
            handle.write("XYZ file generated from PDB data\n")
            for atom in atoms_to_write:
                handle.write(
                    f"{atom.element} "
                    f"{atom.coordinates[0]:.6f} "
                    f"{atom.coordinates[1]:.6f} "
                    f"{atom.coordinates[2]:.6f}\n"
                )
        return output_path

    def add_atoms(self, atoms: list[PDBAtom]) -> None:
        """Append atoms to the structure."""
        self.atoms.extend(atoms)

    def rename_atom_names_by_element(
        self,
        reindex_serial: bool = True,
    ) -> dict[str, int]:
        """Rename atom names so each element is numbered in file
        order."""
        counters: dict[str, int] = {}
        for atom in self.atoms:
            index = counters.get(atom.element, 0) + 1
            counters[atom.element] = index
            atom.atom_name = f"{atom.element}{index}"

        if reindex_serial:
            for index, atom in enumerate(self.atoms, start=1):
                atom.atom_id = index

        return counters
