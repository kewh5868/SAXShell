from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

StructureDistributionCategory = Literal[
    "bond",
    "angle",
    "coordination",
    "cutoff_pair",
]


@dataclass(frozen=True, slots=True)
class StructureDistributionLeaf:
    """One browsable distribution scope loaded from the shared store."""

    source_name: str
    category: StructureDistributionCategory
    definition_id: str
    display_label: str
    xlabel: str
    scope_name: str
    values: np.ndarray
    structure_count: int
    is_all: bool = False

    @property
    def point_count(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True, slots=True)
class StructureDistributionGroup:
    """All cluster-level leaves for one cached distribution."""

    source_name: str
    category: StructureDistributionCategory
    definition_id: str
    display_label: str
    xlabel: str
    cluster_leaves: tuple[StructureDistributionLeaf, ...]
    all_leaf: StructureDistributionLeaf


@dataclass(frozen=True, slots=True)
class StructureDistributionIndex:
    """Project-level summary of all readable structure-distribution
    stores."""

    root_dir: Path
    manifest_paths: tuple[Path, ...]
    entry_count: int
    stale_entry_count: int
    missing_array_count: int
    groups: tuple[StructureDistributionGroup, ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(group.source_name for group in self.groups))

    @property
    def has_values(self) -> bool:
        return any(group.all_leaf.point_count > 0 for group in self.groups)


def load_structure_distribution_index(
    root_dir: str | Path,
) -> StructureDistributionIndex:
    """Load a browsable index from one shared store root.

    The project layout can contain one manifest directly in ``root_dir`` or
    one manifest per application subfolder. Entries whose source structure has
    changed are intentionally skipped here; the store will refresh them on the
    next measurement pass, and the browser should not silently plot stale
    values.
    """

    root_path = Path(root_dir).expanduser().resolve()
    manifest_paths = _discover_manifest_paths(root_path)
    aggregates: dict[
        tuple[str, StructureDistributionCategory, str, str, str],
        dict[str, object],
    ] = {}
    entry_count = 0
    stale_entry_count = 0
    missing_array_count = 0

    for manifest_path in manifest_paths:
        manifest = _read_manifest(manifest_path)
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            continue
        arrays = _read_arrays(manifest_path.parent / "measurements.npz")
        source_name = _source_name(root_path, manifest_path)
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            entry_count += 1
            if not _entry_is_current(entry):
                stale_entry_count += 1
                continue
            cluster_label = _cluster_label(entry)
            structure_id = _structure_id(entry)
            kind = str(entry.get("kind", ""))
            if kind == "bond_angle":
                missing_array_count += _collect_bond_angle_entry(
                    aggregates,
                    arrays,
                    entry,
                    source_name=source_name,
                    cluster_label=cluster_label,
                    structure_id=structure_id,
                )
            elif kind == "cutoff_pairs":
                missing_array_count += _collect_cutoff_pair_entry(
                    aggregates,
                    arrays,
                    entry,
                    source_name=source_name,
                    cluster_label=cluster_label,
                    structure_id=structure_id,
                )

    groups = _build_groups(aggregates)
    return StructureDistributionIndex(
        root_dir=root_path,
        manifest_paths=tuple(manifest_paths),
        entry_count=entry_count,
        stale_entry_count=stale_entry_count,
        missing_array_count=missing_array_count,
        groups=groups,
    )


def validate_structure_distribution_leaves(
    leaves: tuple[StructureDistributionLeaf, ...],
) -> None:
    if not leaves:
        raise ValueError("Select at least one cached distribution to plot.")
    first = leaves[0]
    if any(
        leaf.category != first.category
        or leaf.definition_id != first.definition_id
        for leaf in leaves[1:]
    ):
        raise ValueError(
            "Select distributions of the same type before plotting them "
            "together."
        )
    if len(leaves) > 1 and any(leaf.is_all for leaf in leaves):
        raise ValueError(
            "Select either the all-clusters entry or one or more individual "
            "cluster entries, but not both."
        )


def _discover_manifest_paths(root_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    direct_manifest = root_dir / "manifest.json"
    if direct_manifest.is_file():
        candidates.append(direct_manifest)
    if root_dir.is_dir():
        for child in sorted(root_dir.iterdir(), key=lambda path: path.name):
            manifest_path = child / "manifest.json"
            if child.is_dir() and manifest_path.is_file():
                candidates.append(manifest_path)
    return candidates


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                str(name): np.asarray(data[name], dtype=float)
                for name in data.files
            }
    except (OSError, ValueError):
        return {}


def _source_name(root_dir: Path, manifest_path: Path) -> str:
    store_dir = manifest_path.parent.resolve()
    if store_dir == root_dir:
        return root_dir.name or "shared"
    try:
        relative = store_dir.relative_to(root_dir)
    except ValueError:
        return store_dir.name or "shared"
    return "/".join(relative.parts) or store_dir.name or "shared"


def _entry_is_current(entry: dict[str, object]) -> bool:
    structure = entry.get("structure")
    if not isinstance(structure, dict):
        return False
    resolved_path = str(structure.get("resolved_path", "")).strip()
    if not resolved_path:
        return False
    path = Path(resolved_path)
    if not path.is_file():
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    mtime_ns = int(
        getattr(
            stat,
            "st_mtime_ns",
            int(float(stat.st_mtime) * 1_000_000_000),
        )
    )
    try:
        stored_size = int(structure.get("size_bytes", -1))
        stored_mtime = int(structure.get("mtime_ns", -1))
    except (TypeError, ValueError):
        return False
    return stored_size == int(stat.st_size) and stored_mtime == mtime_ns


def _cluster_label(entry: dict[str, object]) -> str:
    raw_label = str(entry.get("cluster_label") or "").strip()
    return raw_label or "Unassigned cluster"


def _structure_id(entry: dict[str, object]) -> str:
    structure = entry.get("structure")
    if isinstance(structure, dict):
        resolved_path = str(structure.get("resolved_path", "")).strip()
        if resolved_path:
            return resolved_path
        file_name = str(structure.get("file_name", "")).strip()
        if file_name:
            return file_name
    return str(id(entry))


def _collect_bond_angle_entry(
    aggregates: dict[
        tuple[str, StructureDistributionCategory, str, str, str],
        dict[str, object],
    ],
    arrays: dict[str, np.ndarray],
    entry: dict[str, object],
    *,
    source_name: str,
    cluster_label: str,
    structure_id: str,
) -> int:
    missing = 0
    bond_arrays = entry.get("bond_arrays", {})
    if isinstance(bond_arrays, dict):
        for definition_id, array_key in bond_arrays.items():
            values = arrays.get(str(array_key))
            if values is None:
                missing += 1
                continue
            _add_values(
                aggregates,
                source_name=source_name,
                category="bond",
                definition_id=str(definition_id),
                display_label=_bond_definition_label(str(definition_id)),
                xlabel="Distance (A)",
                cluster_label=cluster_label,
                structure_id=structure_id,
                values=values,
            )
    angle_arrays = entry.get("angle_arrays", {})
    if isinstance(angle_arrays, dict):
        for definition_id, array_key in angle_arrays.items():
            values = arrays.get(str(array_key))
            if values is None:
                missing += 1
                continue
            _add_values(
                aggregates,
                source_name=source_name,
                category="angle",
                definition_id=str(definition_id),
                display_label=_angle_definition_label(str(definition_id)),
                xlabel="Angle (deg)",
                cluster_label=cluster_label,
                structure_id=structure_id,
                values=values,
            )
    coordination_arrays = entry.get("coordination_arrays", {})
    if isinstance(coordination_arrays, dict):
        for definition_id, array_key in coordination_arrays.items():
            values = arrays.get(str(array_key))
            if values is None:
                missing += 1
                continue
            _add_values(
                aggregates,
                source_name=source_name,
                category="coordination",
                definition_id=str(definition_id),
                display_label=_coordination_definition_label(
                    str(definition_id)
                ),
                xlabel="Coordination Number",
                cluster_label=cluster_label,
                structure_id=structure_id,
                values=values,
            )
    return missing


def _collect_cutoff_pair_entry(
    aggregates: dict[
        tuple[str, StructureDistributionCategory, str, str, str],
        dict[str, object],
    ],
    arrays: dict[str, np.ndarray],
    entry: dict[str, object],
    *,
    source_name: str,
    cluster_label: str,
    structure_id: str,
) -> int:
    missing = 0
    pair_arrays = entry.get("pair_arrays", {})
    if not isinstance(pair_arrays, dict):
        return missing
    for pair_key, array_key in pair_arrays.items():
        values = arrays.get(str(array_key))
        if values is None:
            missing += 1
            continue
        pair_label = _pair_definition_label(str(pair_key))
        _add_values(
            aggregates,
            source_name=source_name,
            category="cutoff_pair",
            definition_id=f"cutoff_pair:{pair_label}",
            display_label=pair_label,
            xlabel="Distance (A)",
            cluster_label=cluster_label,
            structure_id=structure_id,
            values=values,
        )
    return missing


def _add_values(
    aggregates: dict[
        tuple[str, StructureDistributionCategory, str, str, str],
        dict[str, object],
    ],
    *,
    source_name: str,
    category: StructureDistributionCategory,
    definition_id: str,
    display_label: str,
    xlabel: str,
    cluster_label: str,
    structure_id: str,
    values: np.ndarray,
) -> None:
    key = (source_name, category, definition_id, display_label, xlabel)
    aggregate = aggregates.setdefault(
        key,
        {
            "clusters": defaultdict(list),
            "structures": defaultdict(set),
        },
    )
    clusters = aggregate["clusters"]
    structures = aggregate["structures"]
    clusters[cluster_label].append(np.asarray(values, dtype=float))
    structures[cluster_label].add(structure_id)


def _build_groups(
    aggregates: dict[
        tuple[str, StructureDistributionCategory, str, str, str],
        dict[str, object],
    ],
) -> tuple[StructureDistributionGroup, ...]:
    groups: list[StructureDistributionGroup] = []
    for (
        source_name,
        category,
        definition_id,
        display_label,
        xlabel,
    ), aggregate in sorted(
        aggregates.items(),
        key=lambda item: (
            item[0][0].lower(),
            _category_sort_key(item[0][1]),
            item[0][3].lower(),
        ),
    ):
        clusters = aggregate["clusters"]
        structures = aggregate["structures"]
        cluster_leaves: list[StructureDistributionLeaf] = []
        for cluster_label in sorted(clusters, key=str.lower):
            values = _concatenate_values(clusters[cluster_label])
            cluster_leaves.append(
                StructureDistributionLeaf(
                    source_name=source_name,
                    category=category,
                    definition_id=definition_id,
                    display_label=display_label,
                    xlabel=xlabel,
                    scope_name=cluster_label,
                    values=values,
                    structure_count=len(structures[cluster_label]),
                )
            )
        all_values = _concatenate_values(
            [leaf.values for leaf in cluster_leaves]
        )
        all_leaf = StructureDistributionLeaf(
            source_name=source_name,
            category=category,
            definition_id=definition_id,
            display_label=display_label,
            xlabel=xlabel,
            scope_name="All clusters",
            values=all_values,
            structure_count=sum(
                leaf.structure_count for leaf in cluster_leaves
            ),
            is_all=True,
        )
        groups.append(
            StructureDistributionGroup(
                source_name=source_name,
                category=category,
                definition_id=definition_id,
                display_label=display_label,
                xlabel=xlabel,
                cluster_leaves=tuple(cluster_leaves),
                all_leaf=all_leaf,
            )
        )
    return tuple(groups)


def _concatenate_values(values: list[np.ndarray]) -> np.ndarray:
    non_empty = [
        np.asarray(value, dtype=float) for value in values if value.size
    ]
    if not non_empty:
        return np.array([], dtype=float)
    return np.concatenate(non_empty)


def _category_sort_key(category: StructureDistributionCategory) -> int:
    return {
        "bond": 0,
        "angle": 1,
        "coordination": 2,
        "cutoff_pair": 3,
    }.get(category, 99)


def _bond_definition_label(definition_id: str) -> str:
    payload = _definition_payload(definition_id)
    atom1 = str(payload.get("atom1", "")).strip()
    atom2 = str(payload.get("atom2", "")).strip()
    cutoff = _optional_float(payload.get("cutoff_angstrom"))
    label = "-".join(part for part in (atom1, atom2) if part) or definition_id
    if cutoff is None:
        return label
    return f"{label} <= {cutoff:g} A"


def _angle_definition_label(definition_id: str) -> str:
    payload = _definition_payload(definition_id)
    vertex = str(payload.get("vertex", "")).strip()
    arm1 = str(payload.get("arm1", "")).strip()
    arm2 = str(payload.get("arm2", "")).strip()
    cutoff1 = _optional_float(payload.get("cutoff1_angstrom"))
    cutoff2 = _optional_float(payload.get("cutoff2_angstrom"))
    label = (
        "-".join(part for part in (arm1, vertex, arm2) if part)
        or definition_id
    )
    if cutoff1 is None or cutoff2 is None:
        return label
    return f"{label} <= {cutoff1:g}/{cutoff2:g} A"


def _coordination_definition_label(definition_id: str) -> str:
    payload = _definition_payload(definition_id)
    center_atom = str(payload.get("center_atom", "")).strip()
    neighbor_atom = str(payload.get("neighbor_atom", "")).strip()
    cutoff = _optional_float(payload.get("cutoff_angstrom"))
    label = (
        f"CN {center_atom}-{neighbor_atom}"
        if center_atom and neighbor_atom
        else definition_id
    )
    if cutoff is None:
        return label
    return f"{label} <= {cutoff:g} A"


def _pair_definition_label(pair_key: str) -> str:
    parts = [
        part
        for part in re.split(r"[^0-9A-Za-z]+", str(pair_key).strip())
        if part
    ]
    if len(parts) >= 2:
        return "-".join(sorted(parts[:2]))
    return str(pair_key).strip() or "pair"


def _definition_payload(definition_id: str) -> dict[str, object]:
    try:
        payload = json.loads(definition_id)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
