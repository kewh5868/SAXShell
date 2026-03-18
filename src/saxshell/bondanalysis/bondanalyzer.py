from __future__ import annotations

import math
import re
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


class BondAnalyzer:
    """Measure bond-pair and angle-triplet distributions from flat cluster
    folders.

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
    ) -> None:
        self.bond_pairs = tuple(self._dedupe_bond_pairs(bond_pairs or ()))
        self.angle_triplets = tuple(dict.fromkeys(angle_triplets or ()))

    def structure_files(self, cluster_dir: str | Path) -> list[Path]:
        """Return all structure files directly inside one cluster folder."""
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

    def measure_atoms(
        self,
        atoms: list[AtomRecord],
    ) -> tuple[
        dict[BondPairDefinition, list[float]],
        dict[AngleTripletDefinition, list[float]],
    ]:
        bond_values = {definition: [] for definition in self.bond_pairs}
        angle_values = {
            definition: [] for definition in self.angle_triplets
        }
        if not atoms:
            return bond_values, angle_values

        coords = np.array([[atom.x, atom.y, atom.z] for atom in atoms])
        elements = [atom.element for atom in atoms]
        tree = cKDTree(coords)

        for definition in self.bond_pairs:
            expected = definition.normalized_pair
            for index1, index2 in tree.query_pairs(definition.cutoff_angstrom):
                actual = tuple(sorted((elements[index1], elements[index2])))
                if actual != expected:
                    continue
                distance = float(
                    np.linalg.norm(coords[index1] - coords[index2])
                )
                bond_values[definition].append(distance)

        for definition in self.angle_triplets:
            max_cutoff = max(
                definition.cutoff1_angstrom,
                definition.cutoff2_angstrom,
            )
            for center_index, element in enumerate(elements):
                if element != definition.vertex:
                    continue
                neighbor_indices = [
                    index
                    for index in tree.query_ball_point(
                        coords[center_index],
                        r=max_cutoff,
                    )
                    if index != center_index
                ]
                arm1_candidates = [
                    index
                    for index in neighbor_indices
                    if elements[index] == definition.arm1
                    and self._distance(coords, center_index, index)
                    <= definition.cutoff1_angstrom
                ]
                arm2_candidates = [
                    index
                    for index in neighbor_indices
                    if elements[index] == definition.arm2
                    and self._distance(coords, center_index, index)
                    <= definition.cutoff2_angstrom
                ]
                if not arm1_candidates or not arm2_candidates:
                    continue

                if definition.arm1 == definition.arm2:
                    seen_pairs: set[tuple[int, int]] = set()
                    for arm1_index in arm1_candidates:
                        for arm2_index in arm2_candidates:
                            if arm1_index == arm2_index:
                                continue
                            pair = tuple(sorted((arm1_index, arm2_index)))
                            if pair in seen_pairs:
                                continue
                            seen_pairs.add(pair)
                            angle = self._angle_between(
                                coords[arm1_index] - coords[center_index],
                                coords[arm2_index] - coords[center_index],
                            )
                            if angle is not None:
                                angle_values[definition].append(angle)
                    continue

                for arm1_index in arm1_candidates:
                    for arm2_index in arm2_candidates:
                        if arm1_index == arm2_index:
                            continue
                        angle = self._angle_between(
                            coords[arm1_index] - coords[center_index],
                            coords[arm2_index] - coords[center_index],
                        )
                        if angle is not None:
                            angle_values[definition].append(angle)

        return bond_values, angle_values

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
    def _angle_between(vector1: np.ndarray, vector2: np.ndarray) -> float | None:
        norm1 = float(np.linalg.norm(vector1))
        norm2 = float(np.linalg.norm(vector2))
        if norm1 == 0.0 or norm2 == 0.0:
            return None
        cosine = float(np.dot(vector1, vector2) / (norm1 * norm2))
        return float(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))

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
]
