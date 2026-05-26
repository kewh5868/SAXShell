from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


def _normalized_element_symbol(value: str) -> str:
    text = re.sub(r"[^A-Za-z]", "", str(value or "")).strip()
    if not text:
        raise ValueError("Element symbols must contain at least one letter.")
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Simple atom container used for bond and angle measurements."""

    atom_id: int
    atom_name: str
    residue_name: str
    residue_number: int
    x: float
    y: float
    z: float
    element: str


@dataclass(frozen=True, slots=True)
class BondPairDefinition:
    """One requested bond-pair distribution with its cutoff."""

    atom1: str
    atom2: str
    cutoff_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "atom1",
            _normalized_element_symbol(self.atom1),
        )
        object.__setattr__(
            self,
            "atom2",
            _normalized_element_symbol(self.atom2),
        )
        cutoff = float(self.cutoff_angstrom)
        if cutoff <= 0.0:
            raise ValueError("Bond-pair cutoffs must be greater than zero.")
        object.__setattr__(self, "cutoff_angstrom", cutoff)

    @property
    def normalized_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.atom1, self.atom2)))

    @property
    def display_label(self) -> str:
        return f"{self.atom1}-{self.atom2}"

    @property
    def filename_stem(self) -> str:
        return f"{self.atom1}_{self.atom2}"

    def to_dict(self) -> dict[str, object]:
        return {
            "atom1": self.atom1,
            "atom2": self.atom2,
            "cutoff_angstrom": self.cutoff_angstrom,
        }


@dataclass(frozen=True, slots=True)
class AngleTripletDefinition:
    """One requested angle-triplet distribution with cutoffs."""

    vertex: str
    arm1: str
    arm2: str
    cutoff1_angstrom: float
    cutoff2_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vertex",
            _normalized_element_symbol(self.vertex),
        )
        object.__setattr__(
            self,
            "arm1",
            _normalized_element_symbol(self.arm1),
        )
        object.__setattr__(
            self,
            "arm2",
            _normalized_element_symbol(self.arm2),
        )
        cutoff1 = float(self.cutoff1_angstrom)
        cutoff2 = float(self.cutoff2_angstrom)
        if cutoff1 <= 0.0 or cutoff2 <= 0.0:
            raise ValueError(
                "Angle-triplet cutoffs must be greater than zero."
            )
        object.__setattr__(self, "cutoff1_angstrom", cutoff1)
        object.__setattr__(self, "cutoff2_angstrom", cutoff2)

    @property
    def display_label(self) -> str:
        return f"{self.arm1}-{self.vertex}-{self.arm2}"

    @property
    def filename_stem(self) -> str:
        return f"{self.vertex}_{self.arm1}_{self.arm2}"

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex": self.vertex,
            "arm1": self.arm1,
            "arm2": self.arm2,
            "cutoff1_angstrom": self.cutoff1_angstrom,
            "cutoff2_angstrom": self.cutoff2_angstrom,
        }


@dataclass(frozen=True, slots=True)
class CoordinationNumberDefinition:
    """One requested first-shell coordination-number distribution."""

    center_atom: str
    neighbor_atom: str
    cutoff_angstrom: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_atom",
            _normalized_element_symbol(self.center_atom),
        )
        object.__setattr__(
            self,
            "neighbor_atom",
            _normalized_element_symbol(self.neighbor_atom),
        )
        cutoff = float(self.cutoff_angstrom)
        if cutoff <= 0.0:
            raise ValueError(
                "Coordination-number cutoffs must be greater than zero."
            )
        object.__setattr__(self, "cutoff_angstrom", cutoff)

    @property
    def display_label(self) -> str:
        return f"CN {self.center_atom}-{self.neighbor_atom}"

    @property
    def filename_stem(self) -> str:
        return f"CN_{self.center_atom}_{self.neighbor_atom}"

    def to_dict(self) -> dict[str, object]:
        return {
            "center_atom": self.center_atom,
            "neighbor_atom": self.neighbor_atom,
            "cutoff_angstrom": self.cutoff_angstrom,
        }


class BondAnalyzer:
    """Measure bond-pair, angle-triplet, and coordination distributions
    from flat cluster folders.

    The analyzer expects one cluster-type directory to contain single-frame
    ``.pdb`` or ``.xyz`` files directly inside the directory. The higher-level
    workflow is responsible for discovering multiple stoichiometry folders and
    collecting output files.
    """

    structure_suffixes = (".pdb", ".xyz")

    def __init__(
        self,
        bond_pairs: Iterable[BondPairDefinition] | None = None,
        angle_triplets: Iterable[AngleTripletDefinition] | None = None,
        coordination_numbers: (
            Iterable[CoordinationNumberDefinition] | None
        ) = None,
    ) -> None:
        self.bond_pairs = tuple(self._dedupe_bond_pairs(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))
        self.coordination_numbers = tuple(
            dict.fromkeys(coordination_numbers or ())
        )

    def structure_files(self, cluster_dir: str | Path) -> list[Path]:
        """Return all structure files directly inside one cluster
        folder."""
        path = Path(cluster_dir)
        return sorted(
            file_path
            for file_path in path.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in self.structure_suffixes
        )

    def read_structure(self, structure_file: str | Path) -> list[AtomRecord]:
        path = Path(structure_file)
        if path.suffix.lower() == ".pdb":
            return self._read_pdb(path)
        if path.suffix.lower() == ".xyz":
            return self._read_xyz(path)
        raise ValueError(f"Unsupported structure format: {path.suffix}")

    def measure_structure(
        self,
        structure_file: str | Path,
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        return self.measure_atoms(self.read_structure(structure_file))

    def measure_structure_with_coordination(
        self,
        structure_file: str | Path,
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        return self.measure_atoms_with_coordination(
            self.read_structure(structure_file)
        )

    def measure_atoms(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        if not atoms:
            return (
                {definition: [] for definition in self.bond_pairs},
                {definition: [] for definition in self.angle_triplets},
            )
        coords = np.asarray(
            [[atom.x, atom.y, atom.z] for atom in atoms], dtype=float
        )
        elements = [atom.element for atom in atoms]
        return self.measure_structure_data(coords, elements)

    def measure_atoms_with_coordination(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        if not atoms:
            return (
                {definition: [] for definition in self.bond_pairs},
                {definition: [] for definition in self.angle_triplets},
                {definition: [] for definition in self.coordination_numbers},
            )
        coords = np.asarray(
            [[atom.x, atom.y, atom.z] for atom in atoms], dtype=float
        )
        elements = [atom.element for atom in atoms]
        return self.measure_structure_data_with_coordination(coords, elements)

    def measure_structure_data(
        self,
        coordinates: np.ndarray,
        elements: Iterable[str],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        bond_values, angle_values, _coordination_values = (
            self.measure_structure_data_with_coordination(
                coordinates,
                elements,
            )
        )
        return bond_values, angle_values

    def measure_structure_data_with_coordination(
        self,
        coordinates: np.ndarray,
        elements: Iterable[str],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
        dict[CoordinationNumberDefinition, list[float]],
    ]:
        bond_values = {definition: [] for definition in self.bond_pairs}
        angle_values = {definition: [] for definition in self.angle_triplets}
        coordination_values = {
            definition: [] for definition in self.coordination_numbers
        }

        coords = np.asarray(coordinates, dtype=float)
        normalized_elements = tuple(
            _normalized_element_symbol(element) for element in elements
        )
        if coords.size == 0 or not normalized_elements:
            return bond_values, angle_values, coordination_values
        if coords.ndim != 2 or coords.shape[0] != len(normalized_elements):
            raise ValueError(
                "Coordinates and element symbols must describe the same atoms."
            )

        tree = cKDTree(coords)
        element_array = np.asarray(normalized_elements, dtype=object)

        bond_groups: defaultdict[float, list[BondPairDefinition]] = (
            defaultdict(list)
        )
        for definition in self.bond_pairs:
            bond_groups[float(definition.cutoff_angstrom)].append(definition)
        for cutoff, definitions in bond_groups.items():
            raw_pairs = tree.query_pairs(cutoff)
            if not raw_pairs:
                continue
            pair_indices = np.asarray(list(raw_pairs), dtype=int)
            if pair_indices.size == 0:
                continue
            pair_indices = pair_indices.reshape(-1, 2)
            left_elements = element_array[pair_indices[:, 0]]
            right_elements = element_array[pair_indices[:, 1]]
            distances = np.linalg.norm(
                coords[pair_indices[:, 0]] - coords[pair_indices[:, 1]],
                axis=1,
            )
            for definition in definitions:
                pair_a, pair_b = definition.normalized_pair
                if pair_a == pair_b:
                    mask = (left_elements == pair_a) & (
                        right_elements == pair_b
                    )
                else:
                    mask = (
                        (left_elements == pair_a) & (right_elements == pair_b)
                    ) | (
                        (left_elements == pair_b) & (right_elements == pair_a)
                    )
                if np.any(mask):
                    bond_values[definition].extend(
                        distances[mask].astype(float).tolist()
                    )

        angle_groups: defaultdict[
            tuple[str, float], list[AngleTripletDefinition]
        ] = defaultdict(list)
        for definition in self.angle_triplets:
            angle_groups[
                (
                    definition.vertex,
                    max(
                        float(definition.cutoff1_angstrom),
                        float(definition.cutoff2_angstrom),
                    ),
                )
            ].append(definition)
        for (vertex, max_cutoff), definitions in angle_groups.items():
            center_indices = np.flatnonzero(element_array == vertex)
            if center_indices.size == 0:
                continue
            for center_index in center_indices.tolist():
                neighbor_indices = np.asarray(
                    tree.query_ball_point(coords[center_index], r=max_cutoff),
                    dtype=int,
                )
                if neighbor_indices.size == 0:
                    continue
                neighbor_indices = neighbor_indices[
                    neighbor_indices != center_index
                ]
                if neighbor_indices.size == 0:
                    continue
                neighbor_vectors = (
                    coords[neighbor_indices] - coords[center_index]
                )
                neighbor_distances = np.linalg.norm(neighbor_vectors, axis=1)
                valid_mask = neighbor_distances > 0.0
                if not np.any(valid_mask):
                    continue
                neighbor_elements = element_array[neighbor_indices]
                unit_vectors = np.zeros_like(neighbor_vectors)
                unit_vectors[valid_mask] = (
                    neighbor_vectors[valid_mask]
                    / neighbor_distances[valid_mask, np.newaxis]
                )
                for definition in definitions:
                    arm1_positions = np.flatnonzero(
                        (neighbor_elements == definition.arm1)
                        & (neighbor_distances <= definition.cutoff1_angstrom)
                        & valid_mask
                    )
                    arm2_positions = np.flatnonzero(
                        (neighbor_elements == definition.arm2)
                        & (neighbor_distances <= definition.cutoff2_angstrom)
                        & valid_mask
                    )
                    if arm1_positions.size == 0 or arm2_positions.size == 0:
                        continue
                    if definition.arm1 == definition.arm2:
                        for offset, arm1_position in enumerate(
                            arm1_positions[:-1]
                        ):
                            other_positions = arm1_positions[offset + 1 :]
                            if other_positions.size == 0:
                                continue
                            angles = self._angles_from_unit_vectors(
                                unit_vectors[arm1_position],
                                unit_vectors[other_positions],
                            )
                            angle_values[definition].extend(angles)
                        continue
                    for arm1_position in arm1_positions.tolist():
                        angles = self._angles_from_unit_vectors(
                            unit_vectors[arm1_position],
                            unit_vectors[arm2_positions],
                        )
                        angle_values[definition].extend(angles)

        coordination_groups: defaultdict[
            tuple[str, float], list[CoordinationNumberDefinition]
        ] = defaultdict(list)
        for definition in self.coordination_numbers:
            coordination_groups[
                (definition.center_atom, float(definition.cutoff_angstrom))
            ].append(definition)
        for (center_atom, cutoff), definitions in coordination_groups.items():
            center_indices = np.flatnonzero(element_array == center_atom)
            if center_indices.size == 0:
                continue
            for center_index in center_indices.tolist():
                neighbor_indices = np.asarray(
                    tree.query_ball_point(coords[center_index], r=cutoff),
                    dtype=int,
                )
                if neighbor_indices.size == 0:
                    neighbor_elements = np.asarray((), dtype=object)
                else:
                    neighbor_indices = neighbor_indices[
                        neighbor_indices != center_index
                    ]
                    neighbor_elements = element_array[neighbor_indices]
                for definition in definitions:
                    count = int(
                        np.count_nonzero(
                            neighbor_elements == definition.neighbor_atom
                        )
                    )
                    coordination_values[definition].append(float(count))

        return bond_values, angle_values, coordination_values

    def _read_pdb(self, filepath: Path) -> list[AtomRecord]:
        atoms: list[AtomRecord] = []
        with filepath.open() as stream:
            for line in stream:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                element = (
                    line[76:78].strip()
                    or re.sub(r"[^A-Za-z]", "", atom_name)[:2]
                )
                atoms.append(
                    AtomRecord(
                        atom_id=int(line[6:11]),
                        atom_name=atom_name,
                        residue_name=line[17:20].strip(),
                        residue_number=int(line[22:26] or 0),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        element=_normalized_element_symbol(element),
                    )
                )
        return atoms

    def _read_xyz(self, filepath: Path) -> list[AtomRecord]:
        lines = filepath.read_text().splitlines()
        if not lines:
            return []
        atom_count = int(lines[0].strip())
        atoms: list[AtomRecord] = []
        for index, line in enumerate(lines[2 : 2 + atom_count], start=1):
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(
                AtomRecord(
                    atom_id=index,
                    atom_name=parts[0],
                    residue_name="",
                    residue_number=0,
                    x=float(parts[1]),
                    y=float(parts[2]),
                    z=float(parts[3]),
                    element=_normalized_element_symbol(parts[0]),
                )
            )
        return atoms

    @staticmethod
    def _distance(
        coords: np.ndarray,
        index1: int,
        index2: int,
    ) -> float:
        return float(np.linalg.norm(coords[index1] - coords[index2]))

    @staticmethod
    def _angle_between(
        vector1: np.ndarray, vector2: np.ndarray
    ) -> float | None:
        norm1 = float(np.linalg.norm(vector1))
        norm2 = float(np.linalg.norm(vector2))
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        cosine = float(np.dot(vector1, vector2) / (norm1 * norm2))
        return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))

    @staticmethod
    def _angles_from_unit_vectors(
        vector: np.ndarray,
        other_vectors: np.ndarray,
    ) -> list[float]:
        vectors = np.asarray(other_vectors, dtype=float)
        if vectors.size == 0:
            return []
        dots = np.clip(vectors @ np.asarray(vector, dtype=float), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        return np.asarray(angles, dtype=float).tolist()

    @staticmethod
    def _dedupe_bond_pairs(
        bond_pairs: Iterable[BondPairDefinition],
    ) -> list[BondPairDefinition]:
        deduped: list[BondPairDefinition] = []
        seen: set[tuple[tuple[str, str], float]] = set()
        for definition in bond_pairs:
            key = (definition.normalized_pair, definition.cutoff_angstrom)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(definition)
        return deduped


__all__ = [
    "AngleTripletDefinition",
    "AtomRecord",
    "BondAnalyzer",
    "BondPairDefinition",
    "CoordinationNumberDefinition",
]
