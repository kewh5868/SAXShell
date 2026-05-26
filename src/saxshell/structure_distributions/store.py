from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from saxshell.bondanalysis.bondanalyzer import (
    AngleTripletDefinition,
    BondAnalyzer,
    BondPairDefinition,
    CoordinationNumberDefinition,
    _normalized_element_symbol,
)
from saxshell.saxs.debye import load_structure_file

STORE_SCHEMA_VERSION = 1
ANALYZER_VERSION = "structure-distributions-v1"
STRUCTURE_SUFFIXES = {".pdb", ".xyz"}


@dataclass(frozen=True, slots=True)
class CachedStructureMeasurement:
    structure_path: Path
    bond_values: dict[BondPairDefinition, list[float]]
    angle_values: dict[AngleTripletDefinition, list[float]]
    coordination_values: dict[CoordinationNumberDefinition, list[float]]
    from_cache: bool
    cache_key: str


@dataclass(frozen=True, slots=True)
class CachedCutoffPairMeasurement:
    structure_path: Path
    pair_distances: dict[tuple[str, str], np.ndarray]
    from_cache: bool
    cache_key: str


def project_structure_distribution_store_dir(project_dir: str | Path) -> Path:
    return (
        Path(project_dir).expanduser().resolve()
        / "analysis"
        / "structure_distributions"
    )


def application_structure_distribution_store_dir(
    *,
    project_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    application: str = "shared",
) -> Path:
    app_name = _safe_path_label(application) or "shared"
    if project_dir is not None:
        return project_structure_distribution_store_dir(project_dir) / app_name
    if output_dir is None:
        return Path.cwd() / "structure_distribution_store" / app_name
    return (
        Path(output_dir).expanduser().resolve()
        / "structure_distribution_store"
        / app_name
    )


class StructureDistributionStore:
    """Project-local cache for reusable structure distribution
    measurements."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.manifest_path = self.root_dir / "manifest.json"
        self.measurements_path = self.root_dir / "measurements.npz"
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()
        self._arrays: dict[str, np.ndarray] | None = None
        self._dirty = False

    def measure_structure_file(
        self,
        structure_path: str | Path,
        *,
        bond_pairs: Sequence[BondPairDefinition],
        angle_triplets: Sequence[AngleTripletDefinition],
        coordination_numbers: Sequence[CoordinationNumberDefinition] = (),
        cluster_label: str | None = None,
        relative_label: str | None = None,
        motif_label: str | None = None,
        autosave: bool = True,
    ) -> CachedStructureMeasurement:
        path = Path(structure_path).expanduser().resolve()
        analyzer = BondAnalyzer(
            bond_pairs=tuple(bond_pairs),
            angle_triplets=tuple(angle_triplets),
            coordination_numbers=tuple(coordination_numbers),
        )
        with self._lock:
            cache_key = self._cache_key(
                path,
                self._bond_angle_definition_signature(
                    bond_pairs,
                    angle_triplets,
                    coordination_numbers,
                ),
            )
            cached = self._cached_bond_angle_measurement(
                cache_key,
                path,
                tuple(bond_pairs),
                tuple(angle_triplets),
                tuple(coordination_numbers),
            )
            if cached is not None:
                return cached

            bond_values, angle_values, coordination_values = (
                analyzer.measure_structure_with_coordination(path)
            )
            measurement = self._store_bond_angle_measurement(
                cache_key,
                path,
                tuple(bond_pairs),
                tuple(angle_triplets),
                tuple(coordination_numbers),
                bond_values,
                angle_values,
                coordination_values,
                cluster_label=cluster_label,
                relative_label=relative_label,
                motif_label=motif_label,
                from_cache=False,
            )
            if autosave:
                self.flush()
            return measurement

    def measure_structure_data(
        self,
        structure_path: str | Path,
        coordinates: np.ndarray,
        elements: Sequence[str],
        *,
        bond_pairs: Sequence[BondPairDefinition],
        angle_triplets: Sequence[AngleTripletDefinition],
        coordination_numbers: Sequence[CoordinationNumberDefinition] = (),
        cluster_label: str | None = None,
        relative_label: str | None = None,
        motif_label: str | None = None,
        autosave: bool = True,
    ) -> CachedStructureMeasurement:
        path = Path(structure_path).expanduser().resolve()
        analyzer = BondAnalyzer(
            bond_pairs=tuple(bond_pairs),
            angle_triplets=tuple(angle_triplets),
            coordination_numbers=tuple(coordination_numbers),
        )
        normalized_elements = tuple(
            str(element).strip() for element in elements
        )
        with self._lock:
            cache_key = self._cache_key(
                path,
                self._bond_angle_definition_signature(
                    bond_pairs,
                    angle_triplets,
                    coordination_numbers,
                ),
            )
            cached = self._cached_bond_angle_measurement(
                cache_key,
                path,
                tuple(bond_pairs),
                tuple(angle_triplets),
                tuple(coordination_numbers),
            )
            if cached is not None:
                return cached

            bond_values, angle_values, coordination_values = (
                analyzer.measure_structure_data_with_coordination(
                    np.asarray(coordinates, dtype=float),
                    normalized_elements,
                )
            )
            measurement = self._store_bond_angle_measurement(
                cache_key,
                path,
                tuple(bond_pairs),
                tuple(angle_triplets),
                tuple(coordination_numbers),
                bond_values,
                angle_values,
                coordination_values,
                cluster_label=cluster_label,
                relative_label=relative_label,
                motif_label=motif_label,
                from_cache=False,
            )
            if autosave:
                self.flush()
            return measurement

    def measure_cutoff_pair_distances(
        self,
        structure_path: str | Path,
        *,
        pair_cutoff_definitions: Mapping[tuple[str, str], Mapping[int, float]],
        node_elements: Sequence[str] = (),
        allowed_elements: Sequence[str] | None = None,
        include_node_scaffold: bool = False,
        cluster_label: str | None = None,
        relative_label: str | None = None,
        motif_label: str | None = None,
        autosave: bool = True,
    ) -> CachedCutoffPairMeasurement:
        path = Path(structure_path).expanduser().resolve()
        signature = self._cutoff_pair_definition_signature(
            pair_cutoff_definitions,
            node_elements=node_elements,
            allowed_elements=allowed_elements,
            include_node_scaffold=include_node_scaffold,
        )
        with self._lock:
            cache_key = self._cache_key(path, signature)
            cached = self._cached_cutoff_pair_measurement(cache_key, path)
            if cached is not None:
                return cached

            coordinates, elements = load_structure_file(path)
            if allowed_elements is not None:
                allowed = {
                    _normalized_element_symbol(element)
                    for element in allowed_elements
                    if _normalized_element_symbol(element)
                }
                mask = [
                    _normalized_element_symbol(element) in allowed
                    for element in elements
                ]
                coordinates = np.asarray(coordinates, dtype=float)[mask]
                elements = [
                    element
                    for element, keep in zip(elements, mask, strict=False)
                    if keep
                ]
            pair_distances = _measure_cutoff_pair_distances(
                np.asarray(coordinates, dtype=float),
                elements,
                pair_cutoff_definitions=pair_cutoff_definitions,
                node_elements=node_elements,
                include_node_scaffold=include_node_scaffold,
            )
            measurement = self._store_cutoff_pair_measurement(
                cache_key,
                path,
                pair_distances,
                signature=signature,
                cluster_label=cluster_label,
                relative_label=relative_label,
                motif_label=motif_label,
                from_cache=False,
            )
            if autosave:
                self.flush()
            return measurement

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.root_dir.mkdir(parents=True, exist_ok=True)
            now = datetime.now().isoformat(timespec="seconds")
            self._manifest.setdefault("created_at", now)
            self._manifest["updated_at"] = now
            arrays = self._load_arrays()
            tmp_manifest = self.manifest_path.with_suffix(".json.tmp")
            tmp_arrays = self.measurements_path.with_suffix(".npz.tmp")
            tmp_manifest.write_text(
                json.dumps(self._manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with tmp_arrays.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            tmp_manifest.replace(self.manifest_path)
            tmp_arrays.replace(self.measurements_path)
            self._dirty = False

    def _cached_bond_angle_measurement(
        self,
        cache_key: str,
        path: Path,
        bond_pairs: tuple[BondPairDefinition, ...],
        angle_triplets: tuple[AngleTripletDefinition, ...],
        coordination_numbers: tuple[CoordinationNumberDefinition, ...],
    ) -> CachedStructureMeasurement | None:
        entry = self._manifest.get("entries", {}).get(cache_key)
        if not isinstance(entry, dict) or entry.get("kind") != "bond_angle":
            return None
        if not self._entry_fingerprint_matches(entry, path):
            return None
        arrays = self._load_arrays()
        bond_values: dict[BondPairDefinition, list[float]] = {}
        angle_values: dict[AngleTripletDefinition, list[float]] = {}
        coordination_values: dict[
            CoordinationNumberDefinition, list[float]
        ] = {}
        bond_arrays = dict(entry.get("bond_arrays", {}))
        angle_arrays = dict(entry.get("angle_arrays", {}))
        coordination_arrays = dict(entry.get("coordination_arrays", {}))
        for definition in bond_pairs:
            array_key = bond_arrays.get(_definition_key(definition))
            if array_key is None or array_key not in arrays:
                return None
            bond_values[definition] = arrays[array_key].astype(float).tolist()
        for definition in angle_triplets:
            array_key = angle_arrays.get(_definition_key(definition))
            if array_key is None or array_key not in arrays:
                return None
            angle_values[definition] = arrays[array_key].astype(float).tolist()
        for definition in coordination_numbers:
            array_key = coordination_arrays.get(_definition_key(definition))
            if array_key is None or array_key not in arrays:
                return None
            coordination_values[definition] = (
                arrays[array_key].astype(float).tolist()
            )
        return CachedStructureMeasurement(
            structure_path=path,
            bond_values=bond_values,
            angle_values=angle_values,
            coordination_values=coordination_values,
            from_cache=True,
            cache_key=cache_key,
        )

    def _cached_cutoff_pair_measurement(
        self,
        cache_key: str,
        path: Path,
    ) -> CachedCutoffPairMeasurement | None:
        entry = self._manifest.get("entries", {}).get(cache_key)
        if not isinstance(entry, dict) or entry.get("kind") != "cutoff_pairs":
            return None
        if not self._entry_fingerprint_matches(entry, path):
            return None
        arrays = self._load_arrays()
        pair_distances: dict[tuple[str, str], np.ndarray] = {}
        for pair_key, array_key in dict(entry.get("pair_arrays", {})).items():
            if array_key not in arrays:
                return None
            pair_distances[_pair_tuple_from_key(pair_key)] = np.asarray(
                arrays[array_key],
                dtype=float,
            )
        return CachedCutoffPairMeasurement(
            structure_path=path,
            pair_distances=pair_distances,
            from_cache=True,
            cache_key=cache_key,
        )

    def _store_bond_angle_measurement(
        self,
        cache_key: str,
        path: Path,
        bond_pairs: tuple[BondPairDefinition, ...],
        angle_triplets: tuple[AngleTripletDefinition, ...],
        coordination_numbers: tuple[CoordinationNumberDefinition, ...],
        bond_values: dict[BondPairDefinition, list[float]],
        angle_values: dict[AngleTripletDefinition, list[float]],
        coordination_values: dict[CoordinationNumberDefinition, list[float]],
        *,
        cluster_label: str | None,
        relative_label: str | None,
        motif_label: str | None,
        from_cache: bool,
    ) -> CachedStructureMeasurement:
        arrays = self._load_arrays()
        bond_arrays: dict[str, str] = {}
        angle_arrays: dict[str, str] = {}
        coordination_arrays: dict[str, str] = {}
        for definition in bond_pairs:
            definition_key = _definition_key(definition)
            array_key = f"{cache_key}__bond__{_short_hash(definition_key)}"
            arrays[array_key] = np.asarray(
                bond_values.get(definition, []),
                dtype=float,
            )
            bond_arrays[definition_key] = array_key
        for definition in angle_triplets:
            definition_key = _definition_key(definition)
            array_key = f"{cache_key}__angle__{_short_hash(definition_key)}"
            arrays[array_key] = np.asarray(
                angle_values.get(definition, []),
                dtype=float,
            )
            angle_arrays[definition_key] = array_key
        for definition in coordination_numbers:
            definition_key = _definition_key(definition)
            array_key = (
                f"{cache_key}__coordination__{_short_hash(definition_key)}"
            )
            arrays[array_key] = np.asarray(
                coordination_values.get(definition, []),
                dtype=float,
            )
            coordination_arrays[definition_key] = array_key

        self._manifest.setdefault("entries", {})[cache_key] = {
            "kind": "bond_angle",
            "schema_version": STORE_SCHEMA_VERSION,
            "analyzer_version": ANALYZER_VERSION,
            "structure": self._structure_payload(path),
            "definition_signature": self._bond_angle_definition_signature(
                bond_pairs,
                angle_triplets,
                coordination_numbers,
            ),
            "cluster_label": cluster_label,
            "relative_label": relative_label,
            "motif_label": motif_label,
            "bond_arrays": bond_arrays,
            "angle_arrays": angle_arrays,
            "coordination_arrays": coordination_arrays,
        }
        self._dirty = True
        return CachedStructureMeasurement(
            structure_path=path,
            bond_values={
                definition: list(bond_values.get(definition, []))
                for definition in bond_pairs
            },
            angle_values={
                definition: list(angle_values.get(definition, []))
                for definition in angle_triplets
            },
            coordination_values={
                definition: list(coordination_values.get(definition, []))
                for definition in coordination_numbers
            },
            from_cache=from_cache,
            cache_key=cache_key,
        )

    def _store_cutoff_pair_measurement(
        self,
        cache_key: str,
        path: Path,
        pair_distances: dict[tuple[str, str], np.ndarray],
        *,
        signature: str,
        cluster_label: str | None,
        relative_label: str | None,
        motif_label: str | None,
        from_cache: bool,
    ) -> CachedCutoffPairMeasurement:
        arrays = self._load_arrays()
        pair_arrays: dict[str, str] = {}
        for pair_key, values in sorted(pair_distances.items()):
            pair_text = _pair_key(pair_key[0], pair_key[1])
            array_key = f"{cache_key}__cutoff_pair__{_short_hash(pair_text)}"
            arrays[array_key] = np.asarray(values, dtype=float)
            pair_arrays[pair_text] = array_key
        self._manifest.setdefault("entries", {})[cache_key] = {
            "kind": "cutoff_pairs",
            "schema_version": STORE_SCHEMA_VERSION,
            "analyzer_version": ANALYZER_VERSION,
            "structure": self._structure_payload(path),
            "definition_signature": signature,
            "cluster_label": cluster_label,
            "relative_label": relative_label,
            "motif_label": motif_label,
            "pair_arrays": pair_arrays,
        }
        self._dirty = True
        return CachedCutoffPairMeasurement(
            structure_path=path,
            pair_distances={
                pair: np.asarray(values, dtype=float)
                for pair, values in pair_distances.items()
            },
            from_cache=from_cache,
            cache_key=cache_key,
        )

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.is_file():
            return {
                "schema_version": STORE_SCHEMA_VERSION,
                "analyzer_version": ANALYZER_VERSION,
                "entries": {},
            }
        try:
            payload = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {
                "schema_version": STORE_SCHEMA_VERSION,
                "analyzer_version": ANALYZER_VERSION,
                "entries": {},
            }
        if not isinstance(payload, dict):
            return {
                "schema_version": STORE_SCHEMA_VERSION,
                "analyzer_version": ANALYZER_VERSION,
                "entries": {},
            }
        if not isinstance(payload.get("entries"), dict):
            payload["entries"] = {}
        return payload

    def _load_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is not None:
            return self._arrays
        arrays: dict[str, np.ndarray] = {}
        if self.measurements_path.is_file():
            try:
                with np.load(
                    self.measurements_path, allow_pickle=False
                ) as data:
                    arrays = {
                        str(name): np.asarray(data[name], dtype=float)
                        for name in data.files
                    }
            except (OSError, ValueError):
                arrays = {}
        self._arrays = arrays
        return self._arrays

    def _entry_fingerprint_matches(
        self, entry: dict[str, object], path: Path
    ) -> bool:
        structure = entry.get("structure")
        if not isinstance(structure, dict):
            return False
        current = self._structure_payload(path)
        return (
            structure.get("resolved_path") == current["resolved_path"]
            and int(structure.get("size_bytes", -1)) == current["size_bytes"]
            and int(structure.get("mtime_ns", -1)) == current["mtime_ns"]
        )

    def _cache_key(self, path: Path, definition_signature: str) -> str:
        structure = self._structure_payload(path)
        payload = {
            "analyzer_version": ANALYZER_VERSION,
            "definition_signature": definition_signature,
            "structure": structure,
        }
        return _short_hash(_stable_json(payload), length=24)

    @staticmethod
    def _bond_angle_definition_signature(
        bond_pairs: Sequence[BondPairDefinition],
        angle_triplets: Sequence[AngleTripletDefinition],
        coordination_numbers: Sequence[CoordinationNumberDefinition] = (),
    ) -> str:
        return _stable_json(
            {
                "kind": "bond_angle",
                "analyzer_version": ANALYZER_VERSION,
                "bond_pairs": [
                    definition.to_dict() for definition in bond_pairs
                ],
                "angle_triplets": [
                    definition.to_dict() for definition in angle_triplets
                ],
                "coordination_numbers": [
                    definition.to_dict() for definition in coordination_numbers
                ],
            }
        )

    @staticmethod
    def _cutoff_pair_definition_signature(
        pair_cutoff_definitions: Mapping[tuple[str, str], Mapping[int, float]],
        *,
        node_elements: Sequence[str],
        allowed_elements: Sequence[str] | None,
        include_node_scaffold: bool,
    ) -> str:
        normalized_cutoffs: list[dict[str, object]] = []
        for (element_a, element_b), level_map in sorted(
            pair_cutoff_definitions.items(),
            key=lambda item: (
                _normalized_element_symbol(item[0][0]),
                _normalized_element_symbol(item[0][1]),
            ),
        ):
            pair = sorted(
                (
                    _normalized_element_symbol(element_a),
                    _normalized_element_symbol(element_b),
                )
            )
            normalized_cutoffs.append(
                {
                    "pair": pair,
                    "levels": {
                        str(int(level)): float(cutoff)
                        for level, cutoff in sorted(level_map.items())
                    },
                }
            )
        return _stable_json(
            {
                "kind": "cutoff_pairs",
                "analyzer_version": ANALYZER_VERSION,
                "pair_cutoffs": normalized_cutoffs,
                "node_elements": sorted(
                    _normalized_element_symbol(element)
                    for element in node_elements
                    if _normalized_element_symbol(element)
                ),
                "allowed_elements": (
                    None
                    if allowed_elements is None
                    else sorted(
                        _normalized_element_symbol(element)
                        for element in allowed_elements
                        if _normalized_element_symbol(element)
                    )
                ),
                "include_node_scaffold": bool(include_node_scaffold),
            }
        )

    @staticmethod
    def _structure_payload(path: Path) -> dict[str, object]:
        stat = path.stat()
        return {
            "resolved_path": str(path),
            "file_name": path.name,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(
                getattr(
                    stat,
                    "st_mtime_ns",
                    int(float(stat.st_mtime) * 1_000_000_000),
                )
            ),
        }


def _measure_cutoff_pair_distances(
    coordinates: np.ndarray,
    elements: Sequence[str],
    *,
    pair_cutoff_definitions: Mapping[tuple[str, str], Mapping[int, float]],
    node_elements: Sequence[str],
    include_node_scaffold: bool,
) -> dict[tuple[str, str], np.ndarray]:
    coords = np.asarray(coordinates, dtype=float)
    normalized_elements = [
        _normalized_element_symbol(element) for element in elements
    ]
    pair_indices: set[tuple[int, int]] = set()

    for index_a, index_b in combinations(range(len(normalized_elements)), 2):
        element_a = normalized_elements[index_a]
        element_b = normalized_elements[index_b]
        if not element_a or not element_b:
            continue
        cutoff = _pair_cutoff_distance(
            element_a,
            element_b,
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        if cutoff is None:
            continue
        distance = float(np.linalg.norm(coords[index_a] - coords[index_b]))
        if distance <= float(cutoff) * 1.15:
            pair_indices.add((min(index_a, index_b), max(index_a, index_b)))

    normalized_node_elements = {
        _normalized_element_symbol(element)
        for element in node_elements
        if _normalized_element_symbol(element)
    }
    if include_node_scaffold and normalized_node_elements:
        node_indices = [
            index
            for index, element in enumerate(normalized_elements)
            if element in normalized_node_elements
        ]
        if len(node_indices) >= 2:
            node_edges = _node_scaffold_edges(
                coords[node_indices],
                [normalized_elements[index] for index in node_indices],
                pair_cutoff_definitions=pair_cutoff_definitions,
            )
            for local_index_a, local_index_b in node_edges:
                global_index_a = node_indices[local_index_a]
                global_index_b = node_indices[local_index_b]
                pair_indices.add(
                    (
                        min(global_index_a, global_index_b),
                        max(global_index_a, global_index_b),
                    )
                )

    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index_a, index_b in sorted(pair_indices):
        pair_key = tuple(
            sorted(
                (normalized_elements[index_a], normalized_elements[index_b])
            )
        )
        if not pair_key[0] or not pair_key[1]:
            continue
        pair_values[pair_key].append(
            float(np.linalg.norm(coords[index_a] - coords[index_b]))
        )
    return {
        pair_key: np.asarray(sorted(values), dtype=float)
        for pair_key, values in pair_values.items()
        if values
    }


def _node_scaffold_edges(
    coordinates: np.ndarray,
    node_elements: Sequence[str],
    *,
    pair_cutoff_definitions: Mapping[tuple[str, str], Mapping[int, float]],
) -> list[tuple[int, int]]:
    coords = np.asarray(coordinates, dtype=float)
    point_count = len(coords)
    if point_count < 2:
        return []
    explicit_edges: set[tuple[int, int]] = set()
    all_edges = sorted(
        (
            float(np.linalg.norm(coords[index_a] - coords[index_b])),
            index_a,
            index_b,
        )
        for index_a, index_b in combinations(range(point_count), 2)
    )
    for distance, index_a, index_b in all_edges:
        cutoff = _pair_cutoff_distance(
            node_elements[index_a],
            node_elements[index_b],
            pair_cutoff_definitions=pair_cutoff_definitions,
        )
        if cutoff is not None and distance <= float(cutoff) * 1.15:
            explicit_edges.add((min(index_a, index_b), max(index_a, index_b)))
    if not explicit_edges:
        return _minimum_spanning_tree_edges(coords)

    parent = list(range(point_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> bool:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    for index_a, index_b in explicit_edges:
        union(index_a, index_b)
    for _distance, index_a, index_b in all_edges:
        if len({find(index) for index in range(point_count)}) == 1:
            break
        if union(index_a, index_b):
            explicit_edges.add((min(index_a, index_b), max(index_a, index_b)))
    return sorted(explicit_edges)


def _minimum_spanning_tree_edges(
    coordinates: np.ndarray,
) -> list[tuple[int, int]]:
    coords = np.asarray(coordinates, dtype=float)
    point_count = len(coords)
    if point_count < 2:
        return []
    parent = list(range(point_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> bool:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    selected: list[tuple[int, int]] = []
    for _distance, index_a, index_b in sorted(
        (
            float(np.linalg.norm(coords[index_a] - coords[index_b])),
            index_a,
            index_b,
        )
        for index_a, index_b in combinations(range(point_count), 2)
    ):
        if not union(index_a, index_b):
            continue
        selected.append((index_a, index_b))
        if len(selected) == point_count - 1:
            break
    return selected


def _pair_cutoff_distance(
    element_a: str,
    element_b: str,
    *,
    pair_cutoff_definitions: Mapping[tuple[str, str], Mapping[int, float]],
) -> float | None:
    normalized_a = _normalized_element_symbol(element_a)
    normalized_b = _normalized_element_symbol(element_b)
    for (atom1, atom2), level_map in pair_cutoff_definitions.items():
        if {
            _normalized_element_symbol(atom1),
            _normalized_element_symbol(atom2),
        } != {normalized_a, normalized_b}:
            continue
        if 0 in level_map:
            return float(level_map[0])
        if level_map:
            return float(min(level_map.values()))
    return None


def _definition_key(
    definition: (
        BondPairDefinition
        | AngleTripletDefinition
        | CoordinationNumberDefinition
    ),
) -> str:
    return _stable_json(definition.to_dict())


def _pair_key(element_a: str, element_b: str) -> str:
    return "-".join(sorted((str(element_a), str(element_b))))


def _pair_tuple_from_key(value: str) -> tuple[str, str]:
    left, sep, right = str(value).partition("-")
    if not sep:
        return (str(value), str(value))
    return (left, right)


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _short_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _safe_path_label(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value).strip().lower()
    ).strip("_")
