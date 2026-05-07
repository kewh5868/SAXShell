from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from saxshell.saxs.debye import load_structure_file

_CONTACT_DISTANCE_SCALE = 1.15
_OUTER_SHELL_DISTANCE_SCALE = 1.8


def _pair_key(element_a: str, element_b: str) -> str:
    ordered = sorted((str(element_a).strip(), str(element_b).strip()))
    return f"{ordered[0]}-{ordered[1]}"


def _triplet_key(
    neighbor_a: str,
    center: str,
    neighbor_b: str,
) -> str:
    ordered_neighbors = sorted(
        (str(neighbor_a).strip(), str(neighbor_b).strip())
    )
    return (
        f"{ordered_neighbors[0]}-{str(center).strip()}-{ordered_neighbors[1]}"
    )


def _coordination_key(center: str, neighbor: str) -> str:
    return f"{str(center).strip()}->{str(neighbor).strip()}"


def _normalized_count_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(element): int(counter[element])
        for element in sorted(counter)
        if int(counter[element]) > 0
    }


def _median(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def _mean(values: list[float] | tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _element_index_lookup(
    elements: tuple[str, ...],
) -> dict[str, np.ndarray]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for atom_index, element in enumerate(elements):
        grouped[str(element)].append(int(atom_index))
    return {
        element: np.asarray(indices, dtype=int)
        for element, indices in sorted(grouped.items())
        if indices
    }


def _angle_between_vectors(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float | None:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = min(1.0, max(-1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


@dataclass(slots=True, frozen=True)
class ParsedContrastStructure:
    file_path: Path
    coordinates: np.ndarray
    elements: tuple[str, ...]
    element_counts: dict[str, int]

    @property
    def atom_count(self) -> int:
        return int(len(self.elements))


@dataclass(slots=True, frozen=True)
class ContrastStructureDescriptor:
    file_path: Path
    atom_count: int
    element_counts: dict[str, int]
    core_atom_count: int
    core_element_counts: dict[str, int]
    solvent_atom_count: int
    solvent_element_counts: dict[str, int]
    direct_solvent_atom_count: int
    outer_solvent_atom_count: int
    direct_solvent_element_counts: dict[str, int]
    outer_solvent_element_counts: dict[str, int]
    mean_direct_solvent_coordination: float
    direct_solvent_coordination_by_core_element: dict[str, float]
    bond_length_medians: dict[str, float]
    angle_medians: dict[str, float]
    coordination_medians: dict[str, float]
    notes: tuple[str, ...]

    def solvent_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "total_solvent_atoms": float(self.solvent_atom_count),
            "direct_solvent_atoms": float(self.direct_solvent_atom_count),
            "outer_solvent_atoms": float(self.outer_solvent_atom_count),
            "mean_direct_solvent_coordination": float(
                self.mean_direct_solvent_coordination
            ),
        }
        for element, count in sorted(
            self.direct_solvent_element_counts.items()
        ):
            metrics[f"direct:{element}"] = float(count)
        for element, count in sorted(
            self.outer_solvent_element_counts.items()
        ):
            metrics[f"outer:{element}"] = float(count)
        for (
            element,
            coordination,
        ) in sorted(self.direct_solvent_coordination_by_core_element.items()):
            metrics[f"core_direct_coordination:{element}"] = float(
                coordination
            )
        return metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "file_path": str(self.file_path),
            "atom_count": self.atom_count,
            "element_counts": dict(sorted(self.element_counts.items())),
            "core_atom_count": self.core_atom_count,
            "core_element_counts": dict(
                sorted(self.core_element_counts.items())
            ),
            "solvent_atom_count": self.solvent_atom_count,
            "solvent_element_counts": dict(
                sorted(self.solvent_element_counts.items())
            ),
            "direct_solvent_atom_count": self.direct_solvent_atom_count,
            "outer_solvent_atom_count": self.outer_solvent_atom_count,
            "direct_solvent_element_counts": dict(
                sorted(self.direct_solvent_element_counts.items())
            ),
            "outer_solvent_element_counts": dict(
                sorted(self.outer_solvent_element_counts.items())
            ),
            "mean_direct_solvent_coordination": float(
                self.mean_direct_solvent_coordination
            ),
            "direct_solvent_coordination_by_core_element": dict(
                sorted(
                    self.direct_solvent_coordination_by_core_element.items()
                )
            ),
            "bond_length_medians": dict(
                sorted(self.bond_length_medians.items())
            ),
            "angle_medians": dict(sorted(self.angle_medians.items())),
            "coordination_medians": dict(
                sorted(self.coordination_medians.items())
            ),
            "notes": list(self.notes),
        }


def load_parsed_contrast_structure(
    file_path: str | Path,
    *,
    exclude_elements: list[str] | tuple[str, ...] | set[str] | None = None,
) -> ParsedContrastStructure:
    path = Path(file_path).expanduser().resolve()
    coordinates, elements = load_structure_file(path)
    normalized_exclude = {
        str(element).strip().upper()
        for element in (exclude_elements or [])
        if str(element).strip()
    }
    normalized_elements = tuple(str(element).strip() for element in elements)
    coordinates_array = np.asarray(coordinates, dtype=float)
    if normalized_exclude:
        mask = np.asarray(
            [
                str(element).strip().upper() not in normalized_exclude
                for element in normalized_elements
            ],
            dtype=bool,
        )
        coordinates_array = coordinates_array[mask]
        normalized_elements = tuple(
            element
            for element, keep in zip(normalized_elements, mask, strict=False)
            if keep
        )
    element_counts = Counter(normalized_elements)
    return ParsedContrastStructure(
        file_path=path,
        coordinates=coordinates_array,
        elements=normalized_elements,
        element_counts=_normalized_count_dict(element_counts),
    )


def estimate_pair_contact_distance_medians(
    parsed_structures: (
        list[ParsedContrastStructure] | tuple[ParsedContrastStructure, ...]
    ),
) -> dict[str, float]:
    pair_distances: defaultdict[str, list[float]] = defaultdict(list)
    for parsed in parsed_structures:
        coordinates = np.asarray(parsed.coordinates, dtype=float)
        elements = parsed.elements
        if len(elements) < 2:
            continue
        element_indices = _element_index_lookup(elements)
        ordered_elements = tuple(element_indices.keys())
        for index, element_a in enumerate(ordered_elements):
            indices_a = element_indices[element_a]
            coords_a = coordinates[indices_a]
            for element_b in ordered_elements[index:]:
                indices_b = element_indices[element_b]
                if element_a == element_b and len(indices_a) < 2:
                    continue
                coords_b = coordinates[indices_b]
                pair_key = _pair_key(element_a, element_b)
                distances = np.linalg.norm(
                    coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :],
                    axis=2,
                )
                if element_a == element_b:
                    np.fill_diagonal(distances, np.inf)
                    nearest = distances.min(axis=1)
                    finite = nearest[np.isfinite(nearest)]
                    if finite.size > 0:
                        pair_distances[pair_key].extend(
                            finite.astype(float).tolist()
                        )
                    continue
                nearest_a = distances.min(axis=1)
                nearest_b = distances.min(axis=0)
                pair_distances[pair_key].extend(
                    nearest_a.astype(float).tolist()
                )
                pair_distances[pair_key].extend(
                    nearest_b.astype(float).tolist()
                )
    return {
        pair_key: _median(values)
        for pair_key, values in sorted(pair_distances.items())
        if values
    }


def _build_contact_neighbors(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    pair_contact_distance_medians: dict[str, float],
) -> tuple[list[set[int]], set[tuple[int, int]]]:
    neighbors = [set() for _ in range(len(elements))]
    contact_pairs: set[tuple[int, int]] = set()
    if len(elements) < 2:
        return neighbors, contact_pairs

    coordinate_array = np.asarray(coordinates, dtype=float)
    distance_matrix = np.linalg.norm(
        coordinate_array[:, np.newaxis, :]
        - coordinate_array[np.newaxis, :, :],
        axis=2,
    )
    thresholds = {
        key: float(value) * _CONTACT_DISTANCE_SCALE
        for key, value in pair_contact_distance_medians.items()
        if float(value) > 0.0
    }
    if not thresholds:
        return neighbors, contact_pairs

    row_indices, column_indices = np.triu_indices(len(elements), k=1)
    for atom_index, neighbor_index, distance in zip(
        row_indices.tolist(),
        column_indices.tolist(),
        distance_matrix[row_indices, column_indices].tolist(),
        strict=False,
    ):
        threshold = thresholds.get(
            _pair_key(elements[atom_index], elements[neighbor_index]),
            0.0,
        )
        if threshold <= 0.0 or float(distance) > threshold:
            continue
        neighbors[atom_index].add(neighbor_index)
        neighbors[neighbor_index].add(atom_index)
        contact_pairs.add((atom_index, neighbor_index))
    return neighbors, contact_pairs


def _infer_core_and_solvent_indices(
    coordinates: np.ndarray,
    elements: tuple[str, ...],
    expected_core_counts: dict[str, int],
    neighbors: list[set[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    total_expected = sum(
        int(count) for count in expected_core_counts.values() if int(count) > 0
    )
    if total_expected <= 0 or total_expected >= len(elements):
        return (
            tuple(range(len(elements))),
            tuple(),
            tuple(),
        )

    notes: list[str] = []
    centroid = np.mean(np.asarray(coordinates, dtype=float), axis=0)
    indices_by_element: defaultdict[str, list[int]] = defaultdict(list)
    for index, element in enumerate(elements):
        indices_by_element[element].append(index)

    core_indices: set[int] = set()
    for element, desired_count in sorted(expected_core_counts.items()):
        desired = max(int(desired_count), 0)
        if desired <= 0:
            continue
        candidate_indices = list(indices_by_element.get(element, ()))
        if not candidate_indices:
            notes.append(
                f"Expected {desired} {element} atom(s) from the stoichiometry label, "
                "but none were present in the candidate structure."
            )
            continue
        if desired >= len(candidate_indices):
            core_indices.update(candidate_indices)
            continue
        ranked_indices = sorted(
            candidate_indices,
            key=lambda atom_index: (
                -len(neighbors[atom_index]),
                float(np.linalg.norm(coordinates[atom_index] - centroid)),
                atom_index,
            ),
        )
        chosen = ranked_indices[:desired]
        shell_count = len(candidate_indices) - len(chosen)
        core_indices.update(chosen)
        if shell_count > 0:
            notes.append(
                f"Classified {shell_count} extra {element} atom(s) beyond the "
                "stoichiometry core as solvent-shell atoms."
            )

    if not core_indices:
        notes.append(
            "Unable to isolate a stoichiometry-matched core, so all atoms were "
            "treated as part of the core for representative screening."
        )
        return tuple(range(len(elements))), tuple(), tuple(notes)

    solvent_indices = tuple(
        index for index in range(len(elements)) if index not in core_indices
    )
    return tuple(sorted(core_indices)), solvent_indices, tuple(notes)


def describe_parsed_contrast_structure(
    parsed_structure: ParsedContrastStructure,
    *,
    expected_core_counts: dict[str, int],
    pair_contact_distance_medians: dict[str, float],
    include_geometry_metrics: bool = True,
) -> ContrastStructureDescriptor:
    coordinates = np.asarray(parsed_structure.coordinates, dtype=float)
    elements = parsed_structure.elements
    neighbors, _contact_pairs = _build_contact_neighbors(
        coordinates,
        elements,
        pair_contact_distance_medians,
    )
    core_indices, solvent_indices, core_notes = (
        _infer_core_and_solvent_indices(
            coordinates,
            elements,
            expected_core_counts,
            neighbors,
        )
    )
    core_set = set(core_indices)
    solvent_set = set(solvent_indices)

    bond_lengths: defaultdict[str, list[float]] = defaultdict(list)
    angle_values: defaultdict[str, list[float]] = defaultdict(list)
    coordination_values: defaultdict[str, list[int]] = defaultdict(list)
    if include_geometry_metrics:
        for atom_index in range(len(elements)):
            for neighbor_index in neighbors[atom_index]:
                if neighbor_index <= atom_index:
                    continue
                bond_lengths[
                    _pair_key(elements[atom_index], elements[neighbor_index])
                ].append(
                    float(
                        np.linalg.norm(
                            coordinates[atom_index]
                            - coordinates[neighbor_index]
                        )
                    )
                )

        for center_index, center_neighbors in enumerate(neighbors):
            if len(center_neighbors) < 2:
                continue
            for neighbor_a, neighbor_b in combinations(
                sorted(center_neighbors), 2
            ):
                angle = _angle_between_vectors(
                    coordinates[neighbor_a] - coordinates[center_index],
                    coordinates[neighbor_b] - coordinates[center_index],
                )
                if angle is None:
                    continue
                angle_values[
                    _triplet_key(
                        elements[neighbor_a],
                        elements[center_index],
                        elements[neighbor_b],
                    )
                ].append(float(angle))

        present_elements = tuple(sorted(set(elements)))
        for center_index, center_element in enumerate(elements):
            neighbor_counts = Counter(
                elements[neighbor_index]
                for neighbor_index in neighbors[center_index]
            )
            for neighbor_element in present_elements:
                coordination_values[
                    _coordination_key(center_element, neighbor_element)
                ].append(int(neighbor_counts.get(neighbor_element, 0)))

    direct_solvent_indices: set[int] = set()
    for atom_index in solvent_set:
        if any(
            neighbor_index in core_set
            for neighbor_index in neighbors[atom_index]
        ):
            direct_solvent_indices.add(atom_index)

    outer_solvent_indices: set[int] = set()
    for atom_index in solvent_set - direct_solvent_indices:
        if any(
            neighbor_index in direct_solvent_indices
            for neighbor_index in neighbors[atom_index]
        ):
            outer_solvent_indices.add(atom_index)
            continue
        for core_index in core_indices:
            pair_key = _pair_key(elements[atom_index], elements[core_index])
            cutoff = float(pair_contact_distance_medians.get(pair_key, 0.0))
            if cutoff <= 0.0:
                continue
            distance = float(
                np.linalg.norm(
                    coordinates[atom_index] - coordinates[core_index]
                )
            )
            if distance <= cutoff * _OUTER_SHELL_DISTANCE_SCALE:
                outer_solvent_indices.add(atom_index)
                break

    direct_coordination_by_core_element: defaultdict[str, list[int]] = (
        defaultdict(list)
    )
    direct_solvent_counts_per_core: list[int] = []
    for core_index in core_indices:
        direct_count = sum(
            1
            for neighbor_index in neighbors[core_index]
            if neighbor_index in direct_solvent_indices
        )
        direct_solvent_counts_per_core.append(direct_count)
        direct_coordination_by_core_element[elements[core_index]].append(
            int(direct_count)
        )

    direct_counts = Counter(
        elements[index] for index in direct_solvent_indices
    )
    outer_counts = Counter(elements[index] for index in outer_solvent_indices)
    core_counts = Counter(elements[index] for index in core_indices)
    solvent_counts = Counter(elements[index] for index in solvent_indices)

    return ContrastStructureDescriptor(
        file_path=parsed_structure.file_path,
        atom_count=parsed_structure.atom_count,
        element_counts=dict(sorted(parsed_structure.element_counts.items())),
        core_atom_count=len(core_indices),
        core_element_counts=_normalized_count_dict(core_counts),
        solvent_atom_count=len(solvent_indices),
        solvent_element_counts=_normalized_count_dict(solvent_counts),
        direct_solvent_atom_count=len(direct_solvent_indices),
        outer_solvent_atom_count=len(outer_solvent_indices),
        direct_solvent_element_counts=_normalized_count_dict(direct_counts),
        outer_solvent_element_counts=_normalized_count_dict(outer_counts),
        mean_direct_solvent_coordination=_mean(direct_solvent_counts_per_core),
        direct_solvent_coordination_by_core_element={
            element: _mean(values)
            for element, values in sorted(
                direct_coordination_by_core_element.items()
            )
        },
        bond_length_medians={
            key: _median(values)
            for key, values in sorted(bond_lengths.items())
            if values
        },
        angle_medians={
            key: _median(values)
            for key, values in sorted(angle_values.items())
            if values
        },
        coordination_medians={
            key: _median([float(value) for value in values])
            for key, values in sorted(coordination_values.items())
            if values
        },
        notes=core_notes,
    )
